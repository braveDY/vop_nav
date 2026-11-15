import hashlib
import math
from pathlib import Path

import torch

from .config import validate_contract
from .model import load_model


def heading_scan(ranges, directions, quaternion):
    w, x, y, z = quaternion.unbind(-1)
    heading = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z))
    first = torch.atan2(directions[:, 0, 1], directions[:, 0, 0])
    offset = torch.remainder(-math.pi - (first - heading), 2*math.pi) / (2*math.pi/40)
    coordinate = offset[:, None] + torch.arange(40, device=ranges.device)
    low = coordinate.floor().long()
    weight = coordinate - low
    return (1-weight)*ranges.gather(1, low % 40) + weight*ranges.gather(1, (low+1) % 40)


class OnlineVOP:
    def __init__(self, checkpoint, num_envs, device, control_dt, radius=.35):
        checkpoint = Path(checkpoint).expanduser().resolve()
        self.model, saved = load_model(checkpoint, device)
        self.model.eval().requires_grad_(False)
        inputs, labels = saved['input_config'], saved['label_config']
        validate_contract(saved, control_dt=control_dt, ranges=[.12, 6.],
                          radius=radius, maximum=self.model.config.maximum)
        self.maximum = self.model.config.maximum
        self.history = torch.zeros(num_envs, 5, 40, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stamp = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.intervals = torch.zeros(num_envs, 360, 2, device=device)
        self.intervals[..., 1] = self.maximum
        self.metadata = dict(sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             input_config=inputs, label_config=labels)

    @property
    def valid(self):
        return self.count >= 5

    def reset(self, ids):
        self.history[ids] = 0
        self.count[ids] = 0
        self.stamp[ids] = -1
        self.intervals[ids] = 0
        self.intervals[ids, :, 1] = self.maximum

    @torch.no_grad()
    def update(self, scan, stamp):
        ids = (self.stamp != stamp).nonzero(as_tuple=True)[0]
        if not len(ids):
            return self.intervals
        self.history[ids, :-1] = self.history[ids, 1:].clone()
        self.history[ids, -1] = scan[ids]
        self.count[ids] += 1
        self.stamp[ids] = stamp
        ready = ids[self.valid[ids]]
        if len(ready):
            prediction = self.model(self.history[ready])
            if not torch.isfinite(prediction).all():
                raise FloatingPointError('nonfinite online VOP prediction')
            self.intervals[ready] = prediction
        return self.intervals
