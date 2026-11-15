import math

import torch


def agility_reward(vx, goal_xy, max_speed=4.5, heading_degrees=105.0, goal_radius=0.5):
    """Equation (8), with speed and near-goal branches combined by maximum."""
    distance = goal_xy.norm(dim=-1)
    aligned = goal_xy[:, 0] / distance.clamp_min(1e-8) > math.cos(math.radians(heading_degrees))
    speed = vx.clamp_min(0.0) / max_speed * aligned
    return torch.maximum(speed, (distance < goal_radius).float())


def velocity_interval_reward(velocity, intervals, epsilon=1.e-6):
    """Paper equations 12-15: five nearest circular bins, inverse-distance weights."""
    angle = torch.atan2(velocity[:, 1], velocity[:, 0])
    bins = torch.arange(360, device=velocity.device, dtype=velocity.dtype) * (torch.pi/180) - torch.pi
    delta = torch.atan2(torch.sin(angle[:, None]-bins), torch.cos(angle[:, None]-bins)).abs()
    distances, indices = delta.topk(5, largest=False)
    weights = 1./(distances + epsilon)
    weights = weights / weights.sum(-1, keepdim=True)
    selected = intervals.gather(1, indices[..., None].expand(-1, -1, 2))
    bounds = (selected * weights[..., None]).sum(1)
    speed = velocity[:, :2].norm(dim=-1)
    return -(bounds[:, 0]-speed).clamp_min(0.) - (speed-bounds[:, 1]).clamp_min(0.)
