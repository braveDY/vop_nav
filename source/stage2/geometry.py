import math
import numpy as np

import torch


def _roots(a, b, c):
    """Stable real quadratic roots; invalid roots become zero cuts."""
    a, b, c = torch.broadcast_tensors(a, b, c)
    linear = a.abs() < 1e-12
    disc = b.square() - 4 * a * c
    real = disc >= -1e-12 * torch.maximum(torch.ones_like(disc), b.square() + (4 * a * c).abs())
    q = -.5 * (b + torch.where(b >= 0, 1., -1.) * disc.clamp_min(0).sqrt())
    safe_a = torch.where(linear, 1., a)
    safe_q = torch.where(q.abs() < 1e-14, 1., q)
    first = torch.where(q.abs() < 1e-14, -b / (2 * safe_a), q / safe_a)
    second = torch.where(q.abs() < 1e-14, first, c / safe_q)
    linear_root = -c / torch.where(b.abs() < 1e-14, 1., b)
    first = torch.where(linear, linear_root, first)
    second = torch.where(linear, linear_root, second)
    valid = torch.where(linear, b.abs() >= 1e-14, real)
    return torch.stack((torch.where(valid, first, 0.), torch.where(valid, second, 0.)), -1)


def _rectangle_intervals(rectangles, directions, horizon, maximum):
    position, velocity, half_size, angle, active = rectangles
    position, velocity, half_size, angle = (x.double() for x in (position, velocity, half_size, angle))
    if (not all(torch.isfinite(x).all() for x in (position, velocity, half_size, angle))
            or (half_size < 0).any()):
        raise ValueError('invalid rectangle geometry')
    c, s = angle.cos(), angle.sin()
    def rotate(x):
        return torch.stack((c * x[..., 0] + s * x[..., 1], -s * x[..., 0] + c * x[..., 1]), -1)
    center, v = rotate(position)[:, None], rotate(velocity)[:, None]
    lo, hi = center - half_size[:, None], center + half_size[:, None]
    d = directions
    dx = c[:, None] * d[..., 0] + s[:, None] * d[..., 1]
    dy = -s[:, None] * d[..., 0] + c[:, None] * d[..., 1]
    d = torch.stack((dx, dy), -1)
    # Boundary events: contact at horizon and rays through rectangle corners.
    safe_d = torch.where(d.abs() > 1e-14, d, torch.ones_like(d))
    cuts = [torch.where(d.abs() > 1e-14, (v + b / horizon) / safe_d, 0.) for b in (lo, hi)]
    for x in (lo[..., 0], hi[..., 0]):
        for y in (lo[..., 1], hi[..., 1]):
            denominator = d[..., 0] * y - d[..., 1] * x
            numerator = v[..., 0] * y - v[..., 1] * x
            cuts.append(torch.where(denominator.abs() > 1e-14,
                numerator / torch.where(denominator.abs() > 1e-14, denominator, 1.), 0.)[..., None])
    shape = dx.shape + (1,)
    zero = torch.zeros(shape, device=position.device, dtype=position.dtype)
    cuts = torch.cat([zero, zero + maximum, *cuts], -1).clamp(0, maximum).sort(-1).values
    def collides(speed):
        w = speed[..., None] * d[..., None, :] - v[..., None, :]
        parallel = w.abs() < 1e-14
        divisor = torch.where(parallel, 1., w)
        a, b = lo[..., None, :] / divisor, hi[..., None, :] / divisor
        enter = torch.where(parallel, -torch.inf, torch.minimum(a, b)).amax(-1).clamp_min(0.)
        leave = torch.where(parallel, torch.inf, torch.maximum(a, b)).amin(-1).clamp_max(horizon)
        outside = (parallel & ((lo[..., None, :] > 0) | (hi[..., None, :] < 0))).any(-1)
        return ~outside & (enter <= leave + 1e-12)
    point = collides(cuts)
    segment = collides((cuts[..., :-1] + cuts[..., 1:]) / 2)
    low = torch.minimum(torch.where(point, cuts, torch.inf).amin(-1),
                        torch.where(segment, cuts[..., :-1], torch.inf).amin(-1))
    high = torch.maximum(torch.where(point, cuts, -torch.inf).amax(-1),
                         torch.where(segment, cuts[..., 1:], -torch.inf).amax(-1))
    return torch.where(active[:, None], low, torch.inf), torch.where(active[:, None], high, -torch.inf)


@torch.no_grad()
def safe_intervals(positions, velocities, radii, active, horizon=1., maximum=4.5, directions=360, rectangles=None):
    """Return [B,D,2] lowest-speed safe intervals in heading axes.

    Positions/absolute obstacle velocities are [B,S,2], inflated radii and
    activity are [B,S]. Optional rectangles are (center, velocity, half_size,
    yaw, active), with shapes [B,N,2], [B,N,2], [B,N,2], [B,N], [B,N].
    Angles start at -pi and increase counterclockwise.
    [0,0] represents an empty safe set. Calculations use float64 to resolve
    tangent intersections before casting labels to float32.
    """
    if not (math.isfinite(horizon) and math.isfinite(maximum) and horizon > 0 and maximum > 0):
        raise ValueError('horizon and maximum must be finite and positive')
    if directions <= 0 or positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError('expected positive directions and [B,S,2] positions')
    if velocities.shape != positions.shape or radii.shape != positions.shape[:2] or active.shape != radii.shape:
        raise ValueError('inconsistent obstacle shapes')
    p, v, r = positions.double(), velocities.double(), radii.double()
    if not all(torch.isfinite(x).all() for x in (p, v, r)) or (r < 0).any():
        raise ValueError('nonfinite geometry or negative radius')
    batch, shapes = radii.shape
    if shapes == 0 and (rectangles is None or rectangles[0].shape[1] == 0):
        result = torch.zeros(batch, directions, 2, device=p.device)
        result[..., 1] = maximum
        return result
    angles = torch.arange(directions, device=p.device, dtype=p.dtype) * (2 * math.pi / directions) - math.pi
    d = torch.stack((angles.cos(), angles.sin()), -1)[None, :, None, :]
    p, v, r = p[:, None], v[:, None], r[:, None]
    p2, v2 = p.square().sum(-1), v.square().sum(-1)
    pd, pv, dv = (p * d).sum(-1), (p * v).sum(-1), (v * d).sum(-1)
    c = p2 - r.square()
    center = v + p / horizon
    # End-cap circle and cone tangent intersections with each speed ray.
    cap = _roots(torch.ones_like(pd), -2 * (center * d).sum(-1),
                 center.square().sum(-1) - (r / horizon).square())
    tangent = _roots(pd.square() - c, -2 * pd * pv + 2 * c * dv, pv.square() - c * v2)
    zeros = torch.zeros_like(pd)[..., None]
    cuts = torch.cat((zeros, zeros + maximum, cap, tangent), -1).clamp(0, maximum).sort(-1).values

    def collides(speed):
        # Closest point in time gives an independent predicate for each cut.
        relative = speed[..., None] * d[..., None, :] - v[..., None, :]
        denom = relative.square().sum(-1)
        time = ((p[..., None, :] * relative).sum(-1) / denom.clamp_min(1e-30)).clamp(0, horizon)
        return (p[..., None, :] - time[..., None] * relative).square().sum(-1) <= r[..., None].square() + 1e-11

    point_hit = collides(cuts)
    segment_hit = collides((cuts[..., :-1] + cuts[..., 1:]) / 2)
    low = torch.minimum(torch.where(point_hit, cuts, torch.inf).amin(-1),
                        torch.where(segment_hit, cuts[..., :-1], torch.inf).amin(-1))
    high = torch.maximum(torch.where(point_hit, cuts, -torch.inf).amax(-1),
                         torch.where(segment_hit, cuts[..., 1:], -torch.inf).amax(-1))
    low = torch.where(active[:, None], low, torch.inf)
    high = torch.where(active[:, None], high, -torch.inf)
    if rectangles is not None and rectangles[0].shape[1]:
        rect_low, rect_high = _rectangle_intervals(rectangles, d, horizon, maximum)
        low, high = torch.cat((low, rect_low), -1), torch.cat((high, rect_high), -1)
    low, order = low.sort(-1)
    high = high.gather(-1, order)
    covered = high.cummax(-1).values.clamp_min(0)
    previous = torch.cat((torch.zeros_like(covered[..., :1]), covered[..., :-1]), -1)
    gaps = (low > previous + 1e-9) & (previous < maximum - 1e-9)
    # Include the trailing gap when all blocked intervals end before vmax.
    gap_low = torch.cat((previous, covered[..., -1:]), -1)
    gap_high = torch.cat((low.clamp_max(maximum), torch.full_like(low[..., :1], maximum)), -1)
    gaps = torch.cat((gaps, covered[..., -1:] < maximum - 1e-9), -1)
    index = gaps.to(torch.int64).argmax(-1, keepdim=True)
    result = torch.stack((gap_low.gather(-1, index).squeeze(-1),
                          gap_high.gather(-1, index).squeeze(-1)), -1)
    return torch.where(gaps.any(-1)[..., None], result, 0.).float()


def yaw(quaternion):
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def heading_lidar(block):
    """Resample actual world rays onto 40 heading-relative bins [-pi,pi).

    Each historical scan retains its own heading frame, matching raw scan
    history rather than performing an unreported ego-motion compensation.
    """
    angles = np.arctan2(block['lidar_directions_w'][..., 1], block['lidar_directions_w'][..., 0])
    heading = yaw(block['robot_root_state_w'][..., 3:7])
    offset = ((-np.pi - (angles[..., 0] - heading)) % (2 * np.pi)) / (2 * np.pi / 40)
    coordinate = offset[..., None] + np.arange(40)
    low = np.floor(coordinate).astype(np.int64)
    weight = coordinate - low
    distance = block['lidar_ranges']
    result = ((1 - weight) * np.take_along_axis(distance, low % 40, -1)
              + weight * np.take_along_axis(distance, (low + 1) % 40, -1))
    return result.astype(np.float32)


def _compact_geometry(position, velocity, radius, active, horizon, maximum):
    """Pack only discs that could be reached by some permitted robot velocity.

    For any collision time t <= horizon, the triangle inequality requires
    |p| <= (maximum + |v|) * horizon + radius. Keep a numerical margin
    including the squared-distance tolerance used by safe_intervals.
    """
    if not all(np.isfinite(x).all() for x in (position, velocity, radius)) or (radius < 0).any():
        raise ValueError('nonfinite geometry or negative radius')
    reach = (maximum + np.linalg.norm(velocity, axis=-1)) * horizon + np.sqrt(radius ** 2 + 1e-11) + 1e-7
    relevant = active & (np.linalg.norm(position, axis=-1) <= reach)
    counts = relevant.sum(axis=1)
    width = int(counts.max(initial=0))
    batch = len(position)
    p = np.zeros((batch, width, 2), dtype=np.float64)
    v = np.zeros_like(p)
    r = np.zeros((batch, width), dtype=np.float64)
    enabled = np.zeros((batch, width), dtype=bool)
    rows, columns = np.nonzero(relevant)
    slots = np.cumsum(relevant, axis=1)[rows, columns] - 1
    p[rows, slots], v[rows, slots] = position[rows, columns], velocity[rows, columns]
    r[rows, slots], enabled[rows, slots] = radius[rows, columns], True
    return p, v, r, enabled


def make_labels(block, manifest, times, environments, *, horizon=1., maximum=4.5, device='cpu'):
    robot = block['robot_root_state_w'][times, environments].astype(np.float64)
    origin = block['env_origins_w'][environments, :2].astype(np.float64)
    robot[:, :2] -= origin
    position, velocity, radius, activity = [], [], [], []
    count = len(times)
    rectangles = []
    robot_radius = manifest['robot_collision_proxy_radius_m']
    def box(center, speed, hx, hy, angle, enabled):
        c, s = np.cos(angle), np.sin(angle)
        for x in (-hx, hx):
            for y in (-hy, hy):
                delta = np.stack((c * x - s * y, s * x + c * y), -1)
                position.append(center + delta)
                velocity.append(speed)
                radius.append(robot_radius)
                activity.append(enabled)
        # Two strips plus the four corner discs form the exact rounded rectangle.
        for half in ((hx + robot_radius, hy), (hx, hy + robot_radius)):
            rectangles.append((center, speed, np.broadcast_to(half, (count, 2)), angle, enabled))
    for i, spec in enumerate(manifest['obstacle_specs']):
        # Work in local coordinates before rotation; adding a large float32
        # environment origin would lose precision in small relative distances.
        center = block['obstacle_positions_env'][times, environments, i].astype(np.float64)
        speed = block['obstacle_velocities_w'][times, environments, i].astype(np.float64)
        enabled = block['obstacle_active'][times, environments, i]
        if spec['radius'] > 0:
            position.append(center)
            velocity.append(speed)
            radius.append(spec['radius'] + manifest['robot_collision_proxy_radius_m'])
            activity.append(enabled)
        else:
            box(center, speed, *spec['half_size'], np.zeros(count), enabled)
    walls = block['wall_root_state_w'][environments].astype(np.float64)
    walls[..., :2] -= origin[:, None]
    hx, hy = np.asarray(manifest['wall_geometry']['size_m'][:2]) / 2
    for i in range(walls.shape[1]):
        box(walls[:, i, :2], walls[:, i, 7:9], hx, hy,
            yaw(walls[:, i, 3:7]), np.ones(count, dtype=bool))
    angle = yaw(robot[:, 3:7])
    def rotate(values):
        x, y = values[..., 0], values[..., 1]
        c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
        return np.stack((c * x + s * y, -s * x + c * y), -1)
    if position:
        r = np.broadcast_to(radius, (count, len(radius))).copy()
        p, v, r, enabled = _compact_geometry(np.stack(position, 1) - robot[:, None, :2],
            np.stack(velocity, 1), r, np.stack(activity, 1), horizon, maximum)
        p, v = rotate(p), rotate(v)
    else:
        p = v = np.empty((count, 0, 2))
        r = enabled = np.empty((count, 0))
    tensors = [torch.as_tensor(x, device=device) for x in (p, v, r, enabled.astype(bool))]
    rect_tensors = None
    if rectangles:
        centers, speeds, halves, angles, flags = (np.stack(items, 1) for items in zip(*rectangles))
        values = (rotate(centers - robot[:, None, :2]), rotate(speeds), halves,
                  angles - angle[:, None], flags)
        rect_tensors = tuple(torch.as_tensor(x, device=device) for x in values)
    return safe_intervals(*tensors, horizon=horizon, maximum=maximum, rectangles=rect_tensors).cpu().numpy()
