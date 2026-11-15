import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from assets import mdp
from assets.mdp.terrains import NavigationTerrainCfg, NavigationTerrainGenerator
from assets.robot.unitree_go2 import UNITREE_GO2_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


BOX_SIZES = ((0.5, 0.5, 1.0), (0.5, 0.5, 0.5), (1.2, 0.6, 0.6))
FOOT_NAMES = [f'{leg}_foot' for leg in ('FL', 'FR', 'RL', 'RR')]
OBSTACLE_SPECS = [
    {'name': f'moving_{i}', 'radius': 0.25, 'half_size': (0., 0.), 'height': 1.7} for i in range(8)
] + [
    {'name': f'static_{i}', 'radius': 0.1, 'half_size': (0., 0.), 'height': 2.0} for i in range(2)
] + [
    {'name': f'box_{i}_{j}', 'radius': 0., 'half_size': (size[0] / 2, size[1] / 2), 'height': size[2]}
    for i in range(4) for j, size in enumerate(BOX_SIZES)
]


class NavTerrainImporter(TerrainImporter):
    def __init__(self, cfg):
        # A unique column per environment prevents shared origins even as levels change.
        cfg.terrain_generator.num_cols = cfg.num_envs
        super().__init__(cfg)

    def import_mesh(self, name, mesh):
        """Split large collision meshes; ray sensors merge all children under /World/ground."""
        # The single 27.5M-face mesh at 4096 environments fails PhysX cooking.
        max_faces = 2**22
        if len(mesh.faces) <= max_faces:
            return super().import_mesh(name, mesh)
        num_chunks = (len(mesh.faces) + max_faces - 1) // max_faces
        print(f'[INFO] Splitting terrain: {len(mesh.faces)} faces into {num_chunks} collision meshes.', flush=True)
        for index, start in enumerate(range(0, len(mesh.faces), max_faces)):
            stop = min(start + max_faces, len(mesh.faces))
            chunk = mesh.submesh([range(start, stop)], append=True, repair=False)
            super().import_mesh(f'{name}_{index}', chunk)


@configclass
class NavTerrainImporterCfg(TerrainImporterCfg):
    obstacle_specs: list = OBSTACLE_SPECS
    dynamic_min_speed: float = 0.5
    dynamic_max_speed: float = 2.0


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = NavTerrainImporterCfg(
        class_type=NavTerrainImporter, prim_path='/World/ground', terrain_type='generator',
        max_init_terrain_level=0, collision_group=-1,
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            class_type=NavigationTerrainGenerator,
            size=(12., 7.), border_width=2., num_rows=10, num_cols=1, curriculum=True,
            difficulty_range=(0., 1.), horizontal_scale=0.5, vertical_scale=0.005,
            slope_threshold=0.75, use_cache=True,
            sub_terrains={
                'flat': NavigationTerrainCfg(proportion=0.34, mode='flat'),
                'rough': NavigationTerrainCfg(proportion=0.33, mode='rough'),
                'blocks': NavigationTerrainCfg(proportion=0.33, mode='blocks'),
            },
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1., dynamic_friction=1., friction_combine_mode='average',
            restitution_combine_mode='average'),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
    )
    robot = UNITREE_GO2_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    contact_forces = ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/Robot/.*', update_period=0., history_length=3, track_air_time=True)
    lidar = MultiMeshRayCasterCfg(
        prim_path='{ENV_REGEX_NS}/Robot/base', ray_alignment='yaw', max_distance=6.,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0., 0., 0.15)),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0., 0.),
            horizontal_fov_range=(-180., 180.),
            horizontal_res=9.,
        ),
        mesh_prim_paths=['/World/ground'],
    )
    ray_caster = MultiMeshRayCasterCameraCfg(
        prim_path='{ENV_REGEX_NS}/Robot/base', ray_alignment='base', max_distance=6.,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.26, 0., 0.055),
            rot=(0.9961946980917455, 0., 0.08715574274765817, 0.),
            convention='world',
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            height=12, width=14,
            focal_length=1.,
            horizontal_aperture=1.8979291334297592,
            vertical_aperture=1.108618102905538,
            horizontal_aperture_offset=0., vertical_aperture_offset=0.,
        ),
        data_types=['distance_to_image_plane'],
        depth_clipping_behavior='max',
        mesh_prim_paths=['/World/ground'],
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    def __post_init__(self):
        targets = [MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr='{ENV_REGEX_NS}/' + name, is_shared=True, track_mesh_transforms=True)
            for name in [spec['name'] for spec in self.terrain.obstacle_specs] + ['wall_left', 'wall_right']]
        self.lidar.mesh_prim_paths = ['/World/ground', *targets]
        self.ray_caster.mesh_prim_paths = ['/World/ground', *targets]
        rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        collision = sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.)
        for spec in self.terrain.obstacle_specs:
            dynamic = spec['name'].startswith('moving_')
            material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.65, 0.05) if dynamic else (0.4, 0.4, 0.45))
            common = dict(rigid_props=rigid, collision_props=collision,
                          mass_props=sim_utils.MassPropertiesCfg(mass=100.), visual_material=material)
            if spec['radius']:
                spawn = sim_utils.CylinderCfg(radius=spec['radius'], height=spec['height'], **common)
            else:
                spawn = sim_utils.CuboidCfg(size=(*[2 * h for h in spec['half_size']], spec['height']), **common)
            setattr(self, spec['name'], RigidObjectCfg(
                prim_path='{ENV_REGEX_NS}/' + spec['name'], spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., -30.))))
        for name in ('wall_left', 'wall_right'):
            setattr(self, name, RigidObjectCfg(
                prim_path='{ENV_REGEX_NS}/' + name,
                spawn=sim_utils.CuboidCfg(size=(10.4, 0.2, 2.), rigid_props=rigid, collision_props=collision),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., -30.))))


@configclass
class ActionsCfg:
    joint_position = mdp.JointPositionActionCfg(asset_name='robot', joint_names=['.*'],
                                                preserve_order=True,scale=0.25, use_default_offset=True)


@configclass
class CommandsCfg:
    goal = mdp.ArenaGoalCommandCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        angular_velocity = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        goal = ObsTerm(func=mdp.generated_commands, params={'command_name': 'goal'})
        timer = ObsTerm(func=mdp.episode_time_left, params={'command_name': 'goal'})
        joint_positions = ObsTerm(
            func=mdp.joint_pos_rel,
            params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'], preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_velocities = ObsTerm(
            func=mdp.joint_vel_rel,
            params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'], preserve_order=True)},
            scale=0.2, noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        lidar = ObsTerm(func=mdp.lidar_ranges, params={'sensor_cfg': SceneEntityCfg('lidar')})
        depth = ObsTerm(func=mdp.optical_depth_image, params={'sensor_cfg': SceneEntityCfg('ray_caster')})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy = PolicyCfg()

    @configclass
    class CriticCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    critic = CriticCfg()


@configclass
class TerminationsCfg:
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='base'), 'threshold': 1.0},
    )
    navigation_deadline = DoneTerm(func=mdp.navigation_time_out, time_out=True, params={'command_name': 'goal'})
    tilted = DoneTerm(func=mdp.bad_orientation, params={'limit_angle': 1.0})
    obstacle_collision = DoneTerm(func=mdp.obstacle_collision)
    outside = DoneTerm(func=mdp.outside_navigation_area)


@configclass
class EventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode='startup',
        params={'asset_cfg': SceneEntityCfg('robot', body_names='.*'), 'static_friction_range': (0.4, 1.1),
                'dynamic_friction_range': (0.4, 1.1), 'restitution_range': (0.0, 0.0), 'num_buckets': 64,
                'make_consistent': True},
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode='startup',
        params={'asset_cfg': SceneEntityCfg('robot', body_names='base'),
                'mass_distribution_params': (-1.5, 1.5), 'operation': 'add', 'recompute_inertia': True},
    )
    reset_robot = EventTerm(func=mdp.reset_navigation_scene, mode='reset')
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale, mode='reset',
        params={'position_range': (0.9, 1.1), 'velocity_range': (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity, mode='interval', interval_range_s=(5.0, 8.0),
        params={'velocity_range': {'x': (-0.25, 0.25), 'y': (-0.25, 0.25)}},
    )


@configclass
class RewardsCfg:
    forward_progress = RewTerm(func=mdp.basic_agility, weight=10.)
    heading = RewTerm(
        func=mdp.heading_target, weight=30.0,
        params={'command_name': 'goal', 'sigma': 1.0, 'distance_threshold': 2.0, 'duration': 2.0},
    )
    position_soft = RewTerm(
        func=mdp.position_target, weight=60.0, params={'command_name': 'goal', 'sigma': 2.0, 'duration': 2.0},
    )
    position_tight = RewTerm(
        func=mdp.position_target, weight=60.0, params={'command_name': 'goal', 'sigma': 0.5, 'duration': 1.0},
    )
    stalled = RewTerm(func=mdp.stalled, weight=-20.0, params={'command_name': 'goal', 'distance_threshold': 2.0})
    stand_at_goal = RewTerm(
        func=mdp.stand_at_goal, weight=-10.0,
        params={'command_name': 'goal', 'asset_cfg': SceneEntityCfg('robot'), 'distance_threshold': 0.5, 'duration': 1.0},
    )
    termination = RewTerm(func=mdp.basic_failure, weight=-100.)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    collision = RewTerm(
        func=mdp.undesired_contacts, weight=-100.0,
        params={'threshold': 0.1, 'sensor_cfg': SceneEntityCfg('contact_forces', body_names=['.*thigh', '.*calf'])},
    )
    joint_acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    hip_posture_front = RewTerm(
        func=mdp.joint_deviation_l1, weight=-3.0,
        params={'asset_cfg': SceneEntityCfg('robot', joint_names=['FL_hip_joint', 'FR_hip_joint'])},
    )
    hip_posture_rear = RewTerm(
        func=mdp.joint_deviation_l1, weight=-7.0,
        params={'asset_cfg': SceneEntityCfg('robot', joint_names=['RL_hip_joint', 'RR_hip_joint'])},
    )
    joint_position_limits = RewTerm(func=mdp.joint_pos_limits, weight=-20.0)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005)
    joint_velocity_limits = RewTerm(func=mdp.joint_vel_limits, weight=-20.0, params={'soft_ratio': 0.9})
    feet_collision = RewTerm(
        func=mdp.feet_collision, weight=-100.0,
        params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_foot')},
    )
    airborne = RewTerm(
        func=mdp.airborne, weight=-20.0,
        params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_foot')},
    )
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-20.0)
    torque_limits = RewTerm(
        func=mdp.torque_soft_limits, weight=-20.0, params={'asset_cfg': SceneEntityCfg('robot'), 'soft_limit': 0.85},
    )
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-0.0005)


@configclass
class CurriculumCfg:
    navigation = CurrTerm(func=mdp.navigation_curriculum)


@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=4096, env_spacing=12., replicate_physics=False)
    actions = ActionsCfg()
    commands = CommandsCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    curriculum = CurriculumCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 9.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.ray_caster.update_period = self.decimation * self.sim.dt
        self.viewer.eye = (4.0, 4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.scene.lidar.update_period = self.decimation * self.sim.dt


@configclass
class Go2PlayEnvCfg(Go2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
