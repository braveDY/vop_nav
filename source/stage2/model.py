import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig


class LidarPreprocessing(nn.Module):
    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (5, 40):
            raise ValueError('VOP-Net expects [B,5,40] oldest-to-current LiDAR')
        # Distances remain in metres. The angular axis is periodic; time is not.
        spatial = torch.roll(history, -1, dims=-1) - history
        temporal = torch.cat((torch.zeros_like(history[:, :1]), history[:, 1:] - history[:, :-1]), 1)
        return torch.stack((history, spatial, temporal), 1)


class AngularConv(nn.Conv2d):
    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, 1, 1))
        return super().forward(x)


class MobileBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        expanded = incoming * 2
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, expanded, 1, bias=False), nn.GroupNorm(1, expanded), nn.Hardswish(),
            AngularConv(expanded, expanded, 3, groups=expanded, bias=False),
            nn.GroupNorm(1, expanded), nn.Hardswish(),
            nn.Conv2d(expanded, outgoing, 1, bias=False), nn.GroupNorm(1, outgoing))
        self.residual = incoming == outgoing

    def forward(self, x):
        result = self.layers(x)
        return result + x if self.residual else result


class OrthogonalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 4)
        self.shared = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                    nn.GroupNorm(1, hidden), nn.Hardswish())
        self.time = nn.Conv2d(hidden, channels, 1)
        self.angle = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        temporal = x.mean(-1, keepdim=True)
        angular = x.mean(-2, keepdim=True).transpose(-1, -2)
        features = self.shared(torch.cat((temporal, angular), dim=-2))
        time, angle = features.split((x.shape[-2], x.shape[-1]), dim=-2)
        return x * self.time(time).sigmoid() * self.angle(angle).transpose(-1, -2).sigmoid()


def load_model(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved['schema_version'] != 1:
        raise ValueError('unsupported checkpoint schema')
    model = VOPNet(ModelConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['model'])
    return model, saved


class VOPNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        a, b, c = self.config.channels
        self.preprocessing = LidarPreprocessing()
        self.backbone = nn.Sequential(AngularConv(3, a, 3, bias=False), nn.GroupNorm(1, a), nn.Hardswish(),
                                      MobileBlock(a, b), MobileBlock(b, c))
        self.attention = OrthogonalAttention(c)
        self.decoder = nn.Sequential(nn.Linear(c * 40, self.config.hidden), nn.Hardswish(),
                                     nn.Linear(self.config.hidden, 720))

    def forward(self, history):
        x = self.attention(self.backbone(self.preprocessing(history)))
        raw = self.decoder(x.mean(-2).flatten(1)).reshape(-1, 360, 2)
        minimum = raw[..., 0].sigmoid() * self.config.maximum
        maximum = minimum + (self.config.maximum - minimum) * raw[..., 1].sigmoid()
        return torch.stack((minimum, maximum), -1)
