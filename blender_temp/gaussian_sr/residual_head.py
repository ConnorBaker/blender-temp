import math

import torch
import torch.nn as nn
from torch import Tensor

from .image_utils import pixel_grid


class ScaleAwareResidualHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, residual_scale: float = 0.05):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim + 4, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
        )

    def forward(
        self,
        latent_map: Tensor,
        scale_x: float,
        scale_y: float,
        chunk: int = 262144,
    ) -> Tensor:
        del chunk
        f_dim, h, w = latent_map.shape
        coords = pixel_grid(h, w, latent_map.device, latent_map.dtype, normalized=True).permute(2, 0, 1).contiguous()
        scale_token = latent_map.new_tensor([math.log(scale_x), math.log(scale_y)]).view(2, 1, 1).expand(2, h, w)
        inp = torch.cat((latent_map.view(f_dim, h, w), coords, scale_token), dim=0).unsqueeze(0)
        residual = self.net(inp).squeeze(0)
        return self.residual_scale * torch.tanh(residual)


__all__ = ["ScaleAwareResidualHead"]
