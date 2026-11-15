import math
import random

SCENARIOS = {
    'Forest': dict(width=8., dynamic=6, cylinders=20, boxes=0, walls=False, reactivity=.2, max_speed=2.),
    'Office': dict(width=10., dynamic=8, cylinders=0, boxes=30, walls=True, reactivity=.4, max_speed=2.),
    'Square': dict(width=8., dynamic=20, cylinders=0, boxes=0, walls=True, reactivity=.6, max_speed=2.),
    'Slow': dict(width=10., dynamic=8, cylinders=0, boxes=12, walls=True, reactivity=.2, max_speed=.6),
}


def obstacle_specs():
    rng = random.Random(20261113)
    return ([dict(name=f'moving_{i}', radius=.25, half_size=(0., 0.), height=1.7) for i in range(20)]
            + [dict(name=f'tree_{i}', radius=rng.uniform(.12, .30), half_size=(0., 0.), height=2.) for i in range(20)]
            + [dict(name=f'furniture_{i}', radius=0., half_size=(.30, .40), height=1.) for i in range(30)])


def layout(name, seed, specs):
    cfg = SCENARIOS[name]
    rng = random.Random(seed)
    positions = [(0., 0.) for _ in specs]
    active = [False for _ in specs]
    placed = []

    def add(i, candidate=None):
        r = specs[i]['radius'] or math.hypot(*specs[i]['half_size'])
        for _ in range(10000):
            p = candidate if candidate is not None else (
                rng.uniform(-3.5, 3.5), rng.uniform(-cfg['width']/2+r+.15, cfg['width']/2-r-.15))
            if all(math.dist(p, q) >= r + qr + .10 for q, qr in placed):
                positions[i], active[i] = p, True
                placed.append((p, r))
                return
            if candidate is not None:
                break
        raise RuntimeError(f'{name}: cannot place obstacle {i}')

    if name == 'Forest':
        for i in range(20):
            p = (-3.2 + (i//4)*1.6 + rng.uniform(-.15,.15),
                 -2.7 + (i%4)*1.8 + rng.uniform(-.15,.15))
            add(20+i, p)
    elif name == 'Office':
        for i in range(30):
            col, row = divmod(i,5)
            add(40+i, (-3.+1.2*col, -3.6+1.8*row + (.2 if col%2 else 0.)))
    else:
        for i in range(cfg['boxes']):
            add(40+i)
    for i in range(cfg['dynamic']):
        add(i)
    return positions, active


ASSUMPTIONS = {
    'status': 'paper-informed scenarios, not an exact reproduction',
    'paper_parameters': 'Table I: counts, sizes, 15s, wall presence, reactivity; Slow speed 0.6m/s',
    'floor': 'flat; paper does not specify evaluation ground roughness',
    'office_layout': 'fixed 6 columns x 5 rows, 0.6x0.8m boxes, staggered by 0.2m in alternating columns',
    'forest_layout': '5x4 jittered grid, fixed seeded radii 0.12-0.30m',
    'slow_layout': '12 randomly placed 0.6x0.8m boxes; dimensions not specified in paper',
    'walls': 'two longitudinal side walls, open start/goal ends; Forest has no walls',
    'speed': 'uniform 0.5-2.0m/s except Slow uniform 0.3-0.6m/s',
    'rvo': 'Python-RVO2 ORCA solver, dt=0.1s, neighborDist=5, maxNeighbors=32, timeHorizon=2, timeHorizonObst=2',
    'reactivity': 'exact rounded fraction of agents chosen each episode; nonreactive agents solve without robot',
    'agent_goals': 'opposite side of arena; reverse destination on arrival',
    'robot_start': 'x in [-4.6,-4.5], y in side-wall-safe interval, yaw in [-pi/4,pi/4]',
    'robot_goal': 'x in [4.5,4.6], y in side-wall-safe interval',
    'success': 'report both first-outcome arrival within 0.5m and project deadline success; additionally track whole-episode collisions',
    'robot_collision_radius': .35,
}
