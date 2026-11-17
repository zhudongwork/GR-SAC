"""Small visual residual actor and an ensemble of state-action critics.

Images are raw uint8 camera stacks. Both actor and critic learn their own image
encoder; no gradient or feature dependency reaches the frozen VLA.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def actor_kwargs(config: dict) -> dict:
    """Legacy artifacts did not store exploration bounds; retain their defaults."""
    keys = ("state_dim", "action_dim", "image_channels", "hidden_dim", "feature_dim",
            "log_std_min", "log_std_max", "initial_log_std")
    return {key: config[key] for key in keys if key in config}


class ObservationEncoder(nn.Module):
    def __init__(self, state_dim: int, image_channels: int = 6, feature_dim: int = 128):
        super().__init__()
        self.image_channels = image_channels
        self.state_dim = state_dim
        self.visual = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(32 * 4 * 4, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh(),
        )
        self.output_dim = feature_dim + state_dim

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        images, state = obs["images"], obs["state"]
        if images.dtype != torch.uint8:
            raise TypeError("Residual images must be raw uint8, not pre-normalized floats")
        if images.ndim != 4 or images.shape[1] != self.image_channels:
            raise ValueError(f"Expected BCHW images with {self.image_channels} channels")
        if state.ndim != 2 or state.shape != (images.shape[0], self.state_dim):
            raise ValueError(f"Expected state shape (batch, {self.state_dim})")
        return torch.cat((self.visual(images.float() / 255.0 - 0.5), state.float()), dim=-1)


class SquashedGaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 7, image_channels: int = 6,
                 hidden_dim: int = 256, feature_dim: int = 128,
                 log_std_min: float = -5.0, log_std_max: float = 2.0, initial_log_std: float = -2.0):
        super().__init__()
        if not all(math.isfinite(x) for x in (log_std_min, log_std_max, initial_log_std)) or not (
                log_std_min <= initial_log_std <= log_std_max):
            raise ValueError("Invalid actor log-standard-deviation bounds")
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.encoder = ObservationEncoder(state_dim, image_channels, feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        # Deterministic deployment initially reproduces the base policy exactly.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, initial_log_std)

    def distribution(self, obs: dict[str, torch.Tensor]):
        hidden = self.trunk(self.encoder(obs))
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(self.log_std_min, self.log_std_max)
        return torch.distributions.Normal(mean, log_std.exp())

    def mean_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.distribution(obs).loc.tanh()

    def sample(self, obs: dict[str, torch.Tensor]):
        distribution = self.distribution(obs)
        raw_action = distribution.rsample()
        action = raw_action.tanh()
        # Stable tanh Jacobian, including for saturated actions.
        log_prob = distribution.log_prob(raw_action)
        log_prob -= 2.0 * (math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))
        return action, log_prob.sum(dim=-1, keepdim=True), distribution.loc.tanh()


class CriticEnsemble(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 7, image_channels: int = 6,
                 hidden_dim: int = 256, feature_dim: int = 128, num_critics: int = 4):
        super().__init__()
        self.encoder = ObservationEncoder(state_dim, image_channels, feature_dim)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.encoder.output_dim + action_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
            ) for _ in range(num_critics)
        ])

    def forward(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat((self.encoder(obs), action), dim=-1)
        return torch.stack([head(hidden) for head in self.heads], dim=0)
