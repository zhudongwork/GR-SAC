"""Residual SAC: conservative Bellman targets and mean-ensemble actor objective."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .networks import CriticEnsemble, SquashedGaussianActor, actor_kwargs


@dataclass(frozen=True)
class SACConfig:
    state_dim: int
    action_dim: int = 7
    image_channels: int = 6
    hidden_dim: int = 256
    feature_dim: int = 128
    num_critics: int = 4
    discount: float = 0.999
    tau: float = 0.005
    learning_rate: float = 3e-4
    initial_alpha: float = 0.1
    target_entropy: float | None = None
    learn_alpha: bool = True
    lambda_q: float = 1.0
    lambda_zero: float = 0.01
    lambda_bc: float = 1.0
    max_grad_norm: float = 10.0
    active_state_index: int | None = None
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    initial_log_std: float = -2.0
    lambda_offset: float = 0.0
    offset_state_index: int | None = None
    offset_decay: float = 1.0
    offset_scale: float = 0.005
    offset_bound: float = 0.1

    def __post_init__(self):
        for name in ("state_dim", "action_dim", "image_channels", "hidden_dim", "feature_dim", "num_critics"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("discount", "tau", "learning_rate", "initial_alpha", "lambda_q",
                     "lambda_zero", "lambda_bc", "max_grad_norm", "target_entropy",
                     "log_std_min", "log_std_max", "initial_log_std", "lambda_offset",
                     "offset_decay", "offset_scale", "offset_bound"):
            value = getattr(self, name)
            if value is None and name == "target_entropy":
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not isinstance(self.learn_alpha, bool):
            raise ValueError("learn_alpha must be a boolean")
        if self.num_critics < 2:
            raise ValueError("SAC requires at least two critics")
        if not 0 <= self.discount <= 1 or not 0 < self.tau <= 1:
            raise ValueError("discount must be in [0, 1] and tau in (0, 1]")
        if min(self.learning_rate, self.initial_alpha, self.max_grad_norm) <= 0:
            raise ValueError("learning_rate, initial_alpha, max_grad_norm must be positive")
        if min(self.lambda_q, self.lambda_zero, self.lambda_bc, self.lambda_offset) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if self.active_state_index is not None:
            if (not isinstance(self.active_state_index, int) or isinstance(self.active_state_index, bool)
                    or not 0 <= self.active_state_index < self.state_dim):
                raise ValueError("active_state_index must index the observation state")
        if not self.log_std_min <= self.initial_log_std <= self.log_std_max:
            raise ValueError("initial_log_std must be within its bounds")
        if not 0 <= self.offset_decay <= 1 or self.offset_scale < 0 or self.offset_bound <= 0:
            raise ValueError("Invalid offset dynamics for regularization")
        if self.lambda_offset and (self.offset_state_index is None
                or not isinstance(self.offset_state_index, int)
                or isinstance(self.offset_state_index, bool)
                or not 0 <= self.offset_state_index <= self.state_dim - self.action_dim):
            raise ValueError("Offset regularization requires a valid offset_state_index")


def bellman_target(reward: torch.Tensor, terminated: torch.Tensor,
                   next_value: torch.Tensor, discount: float) -> torch.Tensor:
    """Time limits bootstrap from the actual final observation; terminations do not."""
    return reward + discount * (1.0 - terminated.float()) * next_value


class ResidualSAC:
    def __init__(self, config: SACConfig, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)
        kwargs = {key: getattr(config, key) for key in (
            "state_dim", "action_dim", "image_channels", "hidden_dim", "feature_dim")}
        self.actor = SquashedGaussianActor(**actor_kwargs(asdict(config))).to(self.device)
        self.critic = CriticEnsemble(**kwargs, num_critics=config.num_critics).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.log_alpha = torch.tensor(math.log(config.initial_alpha), device=self.device,
                                      requires_grad=config.learn_alpha)
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=config.learning_rate) if config.learn_alpha else None
        )
        self.target_entropy = (-float(config.action_dim) if config.target_entropy is None
                               else config.target_entropy)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _obs(self, obs):
        return {key: torch.as_tensor(value, device=self.device) for key, value in obs.items()}

    def _active(self, obs: dict) -> torch.Tensor:
        index = self.config.active_state_index
        if index is None:
            return torch.ones((obs["state"].shape[0], 1), device=self.device)
        active = obs["state"][:, index:index + 1].float()
        if not torch.all((active == 0) | (active == 1)):
            raise ValueError("Residual activation flag must be exactly zero or one")
        return active

    def _sample(self, obs: dict):
        action, log_prob, mean = self.actor.sample(obs)
        active = self._active(obs)
        return action * active, log_prob * active, mean * active

    def _mean(self, obs: dict):
        return self.actor.mean_action(obs) * self._active(obs)

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        tensors = self._obs(obs)
        unbatched = tensors["images"].ndim == 3
        if unbatched:
            tensors = {key: value.unsqueeze(0) for key, value in tensors.items()}
        action = self._mean(tensors) if deterministic else self._sample(tensors)[0]
        result = action.cpu().numpy()
        return result[0] if unbatched else result

    def update(self, batch: dict, success_batch: dict | None = None) -> dict[str, float]:
        cfg = self.config
        obs, next_obs = self._obs(batch["obs"]), self._obs(batch["next_obs"])
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device).reshape(-1, 1)
        terminated = torch.as_tensor(batch["terminated"], device=self.device).reshape(-1, 1)
        if actions.shape != (rewards.shape[0], cfg.action_dim):
            raise ValueError("Replay actions must have shape (batch, action_dim)")
        active = self._active(obs)
        if torch.any(actions[active[:, 0] == 0].abs() > 1e-6):
            raise ValueError("Inactive phases must store the executed zero residual action")
        success_obs = None
        bc_active = False
        if success_batch is not None and cfg.lambda_bc:
            successes = np.asarray(success_batch["success"], dtype=bool)
            if successes.size == 0 or not successes.all():
                raise ValueError("Behavior cloning accepts only successful residual transitions")
            success_obs = self._obs(success_batch["obs"])
            success_action = torch.as_tensor(success_batch["actions"], dtype=torch.float32,
                                             device=self.device)
            success_active = self._active(success_obs)
            if torch.any(success_action[success_active[:, 0] == 0].abs() > 1e-6):
                raise ValueError("Inactive success phases must store zero residual actions")
            bc_active = bool(success_active.any())
        with torch.no_grad():
            next_action, next_log_prob, _ = self._sample(next_obs)
            next_q = self.target_critic(next_obs, next_action).min(dim=0).values
            target = bellman_target(rewards, terminated, next_q - self.alpha * next_log_prob,
                                    cfg.discount)
        qs = self.critic(obs, actions)
        # Each ensemble member regresses the bootstrap target, with independent heads.
        critic_loss = F.mse_loss(qs, target.unsqueeze(0).expand_as(qs))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm,
                                                     error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        # Retain dQ/da for the actor while keeping all critic weights fixed.
        self.critic.requires_grad_(False)
        try:
            sampled_action, log_prob, mean_action = self._sample(obs)
            mean_q = self.critic(obs, sampled_action).mean(dim=0)
            improvement = (self.alpha.detach() * log_prob - cfg.lambda_q * mean_q).mean()
            zero_prior = mean_action.square().sum(dim=-1).mean()
            offset_prior = torch.zeros((), device=self.device)
            if cfg.lambda_offset:
                offset = obs["state"][:, cfg.offset_state_index:cfg.offset_state_index + cfg.action_dim]
                # Penalize the proposed offset BEFORE clipping: gradients must
                # still push back when a sampled action would exceed the bound.
                proposed = (cfg.offset_decay * offset + cfg.offset_scale * sampled_action) * active
                offset_prior = (proposed / cfg.offset_bound).square().sum(-1).mean()
            bc_loss = torch.zeros((), device=self.device)
            if success_obs is not None:
                bc_loss = (self._mean(success_obs) - success_action).square().sum(-1).mean()
            actor_loss = (improvement + cfg.lambda_zero * zero_prior + cfg.lambda_bc * bc_loss
                          + cfg.lambda_offset * offset_prior)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_grad = 0.0
            if bool(active.any()) or bc_active:
                actor_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm,
                                                            error_if_nonfinite=True)
                self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)

        alpha_loss = torch.zeros((), device=self.device)
        if self.alpha_optimizer is not None and bool(active.any()):
            active_log_prob = log_prob.detach()[active.bool()]
            alpha_loss = -(self.log_alpha * (active_log_prob + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.lerp_(param, cfg.tau)
        self.updates += 1
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                "actor_improvement": improvement.item(), "zero_prior": zero_prior.item(),
                "bc_loss": bc_loss.item(), "offset_prior": offset_prior.item(),
                "alpha": self.alpha.item(), "alpha_loss": alpha_loss.item(),
                "q_mean": qs.detach().mean().item(), "target_mean": target.mean().item(),
                "entropy": -log_prob.detach().mean().item(), "critic_grad_norm": float(critic_grad),
                "actor_grad_norm": float(actor_grad), "updates": self.updates}

    def state_dict(self) -> dict:
        return {"config": asdict(self.config), "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(), "target_critic": self.target_critic.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(), "updates": self.updates,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None}

    def load_state_dict(self, state: dict) -> None:
        from .compat import normalize_legacy_sac_config
        if asdict(SACConfig(**normalize_legacy_sac_config(state["config"]))) != asdict(self.config):
            raise ValueError("Checkpoint SAC configuration differs from current configuration")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if self.alpha_optimizer:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.updates = state["updates"]
