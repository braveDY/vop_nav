import math
import random
import torch
from isaaclab.utils.math import quat_from_euler_xyz
from assets.mdp.geometry import footprint_clearances
from assets.mdp.scene import ObstacleSceneState
from .layouts import SCENARIOS, layout, obstacle_specs

specs = obstacle_specs()


def episode_seed(seed, env_id, episode):
    return seed * 1000000 + env_id * 10000 + episode


class SceneState(ObstacleSceneState):
    def __init__(self, env):
        super().__init__(env)
        self.scenario = env.cfg.evaluation_scenario
        self.seed = env.cfg.evaluation_seed
        self.episodes = [0] * env.num_envs
        self.sims = [None] * env.num_envs
        self.target = [None] * env.num_envs
        self.reactive = [None] * env.num_envs
        self.speeds = [None] * env.num_envs
        self.ticks = 0

    def build_sim(self, points, active, robot, with_robot):
        import rvo2
        c = SCENARIOS[self.scenario]
        sim = rvo2.PyRVOSimulator(.1, 5., 32, 2., 2., .25, c['max_speed'])
        for i in range(c['dynamic']):
            sim.addAgent(tuple(points[i]))
        for i in range(20, len(specs)):
            if not active[i]: continue
            x,y = points[i]; s = specs[i]
            if s['radius']:
                polygon = [(x+s['radius']*math.cos(t*math.pi/8),
                            y+s['radius']*math.sin(t*math.pi/8)) for t in range(16)]
            else:
                hx,hy=s['half_size']
                polygon=[(x-hx,y-hy),(x+hx,y-hy),(x+hx,y+hy),(x-hx,y+hy)]
            sim.addObstacle(polygon)
        if c['walls']:
            for sign in (-1,1):
                y=sign*(c['width']/2+.1)
                sim.addObstacle([(-5.2,y-.1),(5.2,y-.1),(5.2,y+.1),(-5.2,y+.1)])
        sim.processObstacles()
        sim.addAgent(tuple(robot) if with_robot else (1000.,1000.),
                     5.,32,2.,2.,.35,0.,(0.,0.))
        return sim

    def reset(self, ids):
        c=SCENARIOS[self.scenario]
        for e in ids.tolist():
            reset_seed=episode_seed(self.seed, e, self.episodes[e])
            points,active=layout(self.scenario,reset_seed,specs)
            self.positions[e]=torch.tensor(points,device=self.env.device)
            self.active[e]=torch.tensor(active,device=self.env.device)
            self.velocities[e]=0
            rng=random.Random(reset_seed+1000000000)
            n=c['dynamic']; lower=.3 if self.scenario=='Slow' else .5
            speeds=[rng.uniform(lower,c['max_speed']) for _ in range(n)]
            targets=[]
            for i in range(n):
                x,y=points[i]
                targets.append([(-4.3 if x>=0 else 4.3),rng.uniform(-c['width']/2+.6,c['width']/2-.6)])
            self.target[e]=targets; self.speeds[e]=speeds
            reactive=set(rng.sample(range(n),round(n*c['reactivity'])))
            self.reactive[e]=reactive
            robot=(self.env.scene['robot'].data.root_pos_w[e,:2]-self.env.scene.env_origins[e,:2]).tolist()
            self.sims[e]=(self.build_sim(points,active,robot,False),self.build_sim(points,active,robot,True))
            self.episodes[e]+=1
        for i in range(len(self.assets)): self._write(i,ids)
        for index,name in enumerate(('wall_left','wall_right')):
            wall=self.env.scene[name]; state=wall.data.default_root_state[ids].clone()
            state[:,:3]=self.env.scene.env_origins[ids]
            state[:,1]+=(c['width']/2+.1)*(1 if index==0 else -1)
            state[:,2]+=1. if c['walls'] else -30.
            state[:,7:]=0
            wall.write_root_pose_to_sim(state[:,:7],env_ids=ids)
            wall.write_root_velocity_to_sim(state[:,7:],env_ids=ids)

    def advance(self,dt):
        n=SCENARIOS[self.scenario]['dynamic']
        if self.ticks%20==0:
            positions=self.positions[:,:n].cpu().tolist()
            velocities=self.velocities[:,:n].cpu().tolist()
            robot=self.env.scene['robot'].data
            robot_xy=(robot.root_pos_w[:,:2]-self.env.scene.env_origins[:,:2]).cpu().tolist()
            robot_v=robot.root_lin_vel_w[:,:2].cpu().tolist()
            for e,pair in enumerate(self.sims):
                if pair is None: continue
                preferred=[]
                for i in range(n):
                    p=positions[e][i]; target=self.target[e][i]
                    if math.dist(p,target)<.4:
                        target[0]=-target[0]
                    dx,dy=target[0]-p[0],target[1]-p[1]
                    length=math.hypot(dx,dy)
                    factor=min(self.speeds[e][i],length/.1)/max(length,1e-8)
                    preferred.append((dx*factor,dy*factor))
                for with_robot,sim in enumerate(pair):
                    for i in range(n):
                        sim.setAgentPosition(i,tuple(positions[e][i]))
                        sim.setAgentVelocity(i,tuple(velocities[e][i]))
                        sim.setAgentPrefVelocity(i,preferred[i])
                    sim.setAgentPosition(n,tuple(robot_xy[e]) if with_robot else (1000.,1000.))
                    sim.setAgentVelocity(n,tuple(robot_v[e]) if with_robot else (0.,0.))
                    sim.setAgentPrefVelocity(n,(0.,0.))
                    sim.doStep()
                chosen=[pair[int(i in self.reactive[e])].getAgentVelocity(i) for i in range(n)]
                self.velocities[e,:n]=torch.tensor(chosen,device=self.env.device)
        self.positions[:,:n]+=self.velocities[:,:n]*dt
        ids=torch.arange(self.env.num_envs,device=self.env.device)
        for i in range(n): self._write(i,ids)
        self.ticks+=1

def clearance(env):
    if env.cfg.evaluation_scenario == 'Training':
        from assets.mdp.terminations import obstacle_clearance
        return obstacle_clearance(env)
    state=env.obstacle_scene_state
    xy=env.scene['robot'].data.root_pos_w[:,:2]-env.scene.env_origins[:,:2]
    result=footprint_clearances(xy,state.positions,state.radii,state.half_sizes,state.active).amin(-1)
    c=SCENARIOS[state.scenario]
    if c['walls']: result=torch.minimum(result,c['width']/2-xy[:,1].abs()-.35)
    return result

def outside(env):
    if env.cfg.evaluation_scenario == 'Training':
        from assets.mdp.terminations import outside_navigation_area
        return outside_navigation_area(env)
    xy=env.scene['robot'].data.root_pos_w[:,:2]-env.scene.env_origins[:,:2]
    return (xy[:,0].abs()>5.5)|(xy[:,1].abs()>SCENARIOS[env.obstacle_scene_state.scenario]['width']/2+.5)

def reset_scene(env,env_ids):
    ids=torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
    if not hasattr(env,'obstacle_scene_state'): env.obstacle_scene_state=SceneState(env)
    state=env.obstacle_scene_state
    width=SCENARIOS[state.scenario]['width']
    robot=env.scene['robot']; pose=robot.data.default_root_state[ids].clone()
    pose[:,:3]+=env.scene.env_origins[ids]
    pose[:,0]+=torch.empty(len(ids),device=env.device).uniform_(-4.6,-4.5)
    pose[:,1]+=torch.empty(len(ids),device=env.device).uniform_(-width/2+.8,width/2-.8)
    zero=torch.zeros(len(ids),device=env.device)
    yaw=torch.empty_like(zero).uniform_(-math.pi/4,math.pi/4)
    pose[:,3:7]=quat_from_euler_xyz(zero,zero,yaw); pose[:,7:]=0
    robot.write_root_pose_to_sim(pose[:,:7],env_ids=ids)
    robot.write_root_velocity_to_sim(pose[:,7:],env_ids=ids)
    state.reset(ids)

