import torch


def reflect_motion(position, velocity, lower, upper, dt):
    """Integrate a point with specular reflection, including multiple crossings."""
    span = upper - lower
    unfolded = position - lower + velocity * dt
    phase = torch.remainder(unfolded, 2.0 * span)
    reflected = lower + torch.where(phase <= span, phase, 2.0 * span - phase)
    sign = torch.where(phase < span, 1.0, -1.0)
    return reflected, velocity * sign


def footprint_clearances(robot_xy, positions, radii, half_sizes, active, robot_radius=0.35):
    """Signed circle/circle and circle/axis-aligned-box clearances in metres."""
    delta = (robot_xy[:, None] - positions).abs()
    circle = delta.norm(dim=-1) - radii - robot_radius
    q = delta - half_sizes
    box = q.clamp_min(0.0).norm(dim=-1) + q.amax(dim=-1).clamp_max(0.0) - robot_radius
    clearance = torch.where(radii > 0, circle, box)
    return torch.where(active, clearance, torch.inf)
