"""Serializable experiment configuration; imports no model or simulator."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

TASKS = ["PickXtimes", "StopCube", "SwingXtimes", "BinFill", "VideoUnmaskSwap",
         "VideoUnmask", "ButtonUnmaskSwap", "ButtonUnmask", "VideoRepick",
         "VideoPlaceButton", "VideoPlaceOrder", "PickHighlight", "InsertPeg",
         "MoveCube", "PatternLock", "RouteStick"]
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    checkpoint: str = "checkpoints/simplememvla_robomme"
    output_dir: str = "logs/residual_rl/robomme"
    checkpoint_dir: str | None = None
    metadata_root: str = "robomme_sim/robomme/env_metadata"
    tasks: list[str] = field(default_factory=lambda: list(TASKS))
    gpus: list[str] = field(default_factory=lambda: [str(i) for i in range(8)])
    envs_per_collector: int = 4
    execute_horizon: int = 16
    max_steps: int = 1300
    num_denoising_steps: int = 10
    eval_temperature: float = 1.0
    max_subtask_tokens: int = 64
    compute_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    image_size: int = 84
    delta: float = 0.005
    gain: float = 1.0
    offset_bound: float = 0.1
    enabled_tasks: list[str] | None = None
    start_step: int = 0
    control_version: int = 1
    offset_decay: float = 1.0
    phase_gate: str = "always"
    max_target_span: float = 0.08
    max_tracking_error: float = 0.1
    max_gripper_span: float = 0.1
    seed: int = 42
    bootstrap_episodes_per_task: int = 2
    bootstrap_sampling: str = "first"
    require_demo_task_coverage: bool = False
    demo_archive: str | None = None
    replay_capacity: int = 50000
    demo_capacity: int = 50000
    batch_size: int = 256
    learning_starts: int = 1024
    max_updates: int = 100000
    updates_per_transition: float = 0.25
    sync_every: int = 100
    checkpoint_every: int = 5000
    log_every: int = 100
    queue_size: int = 14
    idle_timeout: float = 14400.0
    step_timeout: float = 300.0
    reset_timeout: float = 3600.0
    train_scene_failure_policy: str = "skip_sampling_exhausted"
    max_consecutive_empty_resets: int = 8
    cpu_threads: int = 4
    validation_every: int = 0
    validation_episodes_per_task: int = 3
    validation_group_size: int = 4
    validation_seed: int = 2026
    validation_min_success_rate: float = 0.5
    validation_max_drop: float = 0.15
    validation_patience: int = 2
    validation_scene_failure_policy: str = "raise"
    sac: dict = field(default_factory=lambda: {"discount": 0.999})

    @property
    def state_dim(self):
        return 42 + 8 * self.execute_horizon

    def validate(self):
        if not self.tasks or len(set(self.tasks)) != len(self.tasks) or set(self.tasks) - set(TASKS):
            raise ValueError("tasks must be a nonempty unique subset of the 16 RoboMME tasks")
        if (len(self.gpus) < 2 or len(set(self.gpus)) != len(self.gpus)
                or any(not isinstance(gpu, str) or not gpu.strip() for gpu in self.gpus)):
            raise ValueError("Use distinct GPUs: first learner, remaining collectors (at least two)")
        for key in ("envs_per_collector", "execute_horizon", "max_steps", "image_size",
                    "replay_capacity", "demo_capacity", "batch_size", "learning_starts",
                    "max_updates", "sync_every", "checkpoint_every", "log_every", "queue_size",
                    "cpu_threads", "num_denoising_steps", "max_subtask_tokens",
                    "max_consecutive_empty_resets", "validation_episodes_per_task",
                    "validation_group_size", "validation_patience"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("bootstrap_episodes_per_task", "start_step", "seed", "validation_every", "validation_seed"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.batch_size % 2:
            raise ValueError("batch_size must be even for 50/50 replay")
        if self.control_version not in (1, 2) or isinstance(self.control_version, bool):
            raise ValueError("control_version must be 1 or 2")
        if self.phase_gate not in ("always", "low_motion"):
            raise ValueError("phase_gate must be always or low_motion")
        if not math.isfinite(self.offset_decay) or not 0 <= self.offset_decay <= 1:
            raise ValueError("offset_decay must be in [0, 1]")
        if self.control_version == 1 and (self.offset_decay != 1 or self.phase_gate != "always"):
            raise ValueError("New controller semantics require control_version=2 and fresh replay")
        for key in ("max_target_span", "max_tracking_error", "max_gripper_span"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.bootstrap_sampling not in ("first", "difficulty_balanced"):
            raise ValueError("Unknown bootstrap_sampling")
        if self.bootstrap_sampling == "difficulty_balanced" and self.bootstrap_episodes_per_task < 3:
            raise ValueError("Difficulty-balanced bootstrap requires at least 3 episodes per task")
        if not isinstance(self.require_demo_task_coverage, bool):
            raise ValueError("require_demo_task_coverage must be boolean")
        for key in ("validation_min_success_rate", "validation_max_drop"):
            if not math.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if self.validation_every and (self.validation_every % self.checkpoint_every
                                      or self.validation_episodes_per_task < 3):
            raise ValueError("Validation must align with checkpoints and cover at least 3 episodes per task")
        if self.train_scene_failure_policy not in ("raise", "skip_sampling_exhausted"):
            raise ValueError("train_scene_failure_policy must be raise or skip_sampling_exhausted")
        if self.validation_scene_failure_policy not in ("raise", "continue_sampling_exhausted"):
            raise ValueError("validation_scene_failure_policy must be raise or continue_sampling_exhausted")
        if self.learning_starts > self.replay_capacity:
            raise ValueError("learning_starts exceeds replay capacity")
        if self.bootstrap_episodes_per_task < 0 or (not self.demo_archive and self.bootstrap_episodes_per_task == 0):
            raise ValueError("Supply real residual demo_archive or positive bootstrap_episodes_per_task")
        for key in ("delta", "gain", "offset_bound", "updates_per_transition", "eval_temperature",
                    "idle_timeout", "step_timeout", "reset_timeout"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if min(self.delta, self.gain, self.offset_bound, self.updates_per_transition,
               self.idle_timeout, self.step_timeout, self.reset_timeout) <= 0:
            raise ValueError("scales, bounds, update ratio and timeouts must be positive")
        if self.start_step < 0 or self.start_step >= self.max_steps:
            raise ValueError("start_step must fall within the control episode")
        if self.enabled_tasks is not None and (not self.enabled_tasks or set(self.enabled_tasks) - set(self.tasks)):
            raise ValueError("enabled_tasks must be a nonempty subset of tasks")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")
        if set(self.sac) & {"state_dim", "action_dim", "image_channels", "active_state_index",
                            "offset_state_index", "offset_decay", "offset_scale", "offset_bound"}:
            raise ValueError("SAC observation/action dimensions are fixed by the RoboMME adapter")
        for name in ("checkpoint", "output_dir", "metadata_root"):
            path = Path(getattr(self, name)).expanduser()
            setattr(self, name, str((ROOT / path).resolve() if not path.is_absolute() else path.resolve()))
        if self.checkpoint_dir:
            path = Path(self.checkpoint_dir).expanduser()
            self.checkpoint_dir = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
        if self.demo_archive:
            self.demo_archive = str(Path(self.demo_archive).expanduser().resolve())
        return self

    def policy_args(self, seed=None):
        return argparse.Namespace(pretrained_checkpoint=self.checkpoint,
            execute_horizon=self.execute_horizon, max_steps=self.max_steps,
            num_denoising_steps=self.num_denoising_steps, eval_temperature=self.eval_temperature,
            max_subtask_tokens=self.max_subtask_tokens, compute_dtype=self.compute_dtype,
            attn_implementation=self.attn_implementation, pipelined=False,
            step_timeout=self.step_timeout, reset_timeout=self.reset_timeout,
            seed=self.seed if seed is None else seed)

    def controller_kwargs(self):
        return {key: getattr(self, key) for key in (
            "offset_decay", "phase_gate", "max_target_span", "max_tracking_error", "max_gripper_span")}

    def sac_kwargs(self):
        values = {"state_dim": self.state_dim, "active_state_index": 24, **self.sac}
        if self.control_version == 2:
            values.update(offset_state_index=16, offset_decay=self.offset_decay,
                          offset_scale=self.delta * self.gain, offset_bound=self.offset_bound)
        return values

    def validation_protocol(self):
        return {"split": "val", "mode": "residual", "selection": "difficulty_balanced",
                **{key: getattr(self, key) for key in ("tasks", "validation_episodes_per_task",
                   "validation_group_size", "validation_seed", "validation_min_success_rate",
                   "validation_max_drop", "validation_patience")}}

    def protocol(self):
        """Fields determining transition meaning, required to match across demos/resumes."""
        keys = ("tasks", "execute_horizon", "max_steps", "num_denoising_steps", "eval_temperature",
                "max_subtask_tokens", "compute_dtype", "attn_implementation", "image_size",
                "delta", "gain", "offset_bound", "enabled_tasks", "start_step")
        protocol = {"schema": "simplememvla-joint-residual-v1", "split": "train",
                "state_dim": self.state_dim, "action_dim": 7,
                "offset_timing": "command_old_offset_then_integrate",
                **{key: getattr(self, key) for key in keys}}
        if self.control_version == 2:
            protocol.update(schema="simplememvla-joint-residual-v2", **self.controller_kwargs())
        return protocol


def load_config(path, **overrides):
    values = json.loads(Path(path).read_text())
    values.update({key: value for key, value in overrides.items() if value is not None})
    return TrainConfig(**values).validate()


def metadata_specs(cfg, split):
    """Read actual frozen episode IDs and reject overlap with either holdout split."""
    if split not in ("train", "val", "test"):
        raise ValueError("Invalid split")
    specs, seen = [], set()
    for task in cfg.tasks:
        path = Path(cfg.metadata_root) / split / f"record_dataset_{task}_metadata.json"
        rows = json.loads(path.read_text())["records"]
        for row in rows:
            if row["task"] != task or (task, row["episode"]) in seen:
                raise ValueError(f"Duplicate or wrong-task metadata in {path}")
            seen.add((task, row["episode"]))
            specs.append({"task": task, "episode": int(row["episode"]), "seed": int(row["seed"])})
        if not rows:
            raise ValueError(f"No episodes in {path}")
    if split == "train":
        train_seeds = {(x["task"], x["seed"]) for x in specs}
        for heldout in ("val", "test"):
            other = {(x["task"], x["seed"]) for x in metadata_specs(cfg, heldout)}
            if train_seeds & other:
                raise ValueError(f"Train seeds overlap {heldout}; refusing leakage")
    return specs


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path, *, hash_weights=True):
    """Hash the actual frozen checkpoint, not merely same-sized weight filenames."""
    path = Path(path)
    required = ("config.json", "stats.json")
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(f"Original SimpleMemVLA checkpoint requires {path / name}")
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No safetensors weights in {path}")
    return {"config_hashes": {p.name: file_sha256(p) for p in sorted(path.glob("*.json"))},
            "weight_files": {p.name: p.stat().st_size for p in weights},
            "weight_hashes": {p.name: file_sha256(p) for p in weights} if hash_weights else None,
            "weight_content_hashed": hash_weights}


def preflight(cfg):
    specs = metadata_specs(cfg, "train")
    missing = [name for name in ("torch", "transformers", "sapien", "mani_skill", "safetensors")
               if importlib.util.find_spec(name) is None]
    if "torch" not in missing:
        from .sac import SACConfig
        SACConfig(**cfg.sac_kwargs())
    if "flash" in cfg.attn_implementation and importlib.util.find_spec("flash_attn") is None:
        missing.append("flash_attn")
    identity = None
    try:
        identity = checkpoint_identity(cfg.checkpoint, hash_weights=False)
    except (FileNotFoundError, ValueError) as exc:
        missing.append(str(exc))
    return {"train_episodes": len(specs), "tasks": cfg.tasks, "state_dim": cfg.state_dim,
            "learner_gpu": cfg.gpus[0], "collector_gpus": cfg.gpus[1:],
            "envs_per_collector": cfg.envs_per_collector,
            "control_protocol": cfg.protocol(), "bootstrap_sampling": cfg.bootstrap_sampling,
            "bootstrap_episodes_per_task": cfg.bootstrap_episodes_per_task,
            "validation_every": cfg.validation_every,
            "train_scene_failure_policy": cfg.train_scene_failure_policy,
            "replay_image_gb_upper_bound": round((cfg.replay_capacity + cfg.demo_capacity)
                  * 2 * 6 * cfg.image_size ** 2 / 1e9, 3),
            "missing": missing, "checkpoint_identity": identity}
