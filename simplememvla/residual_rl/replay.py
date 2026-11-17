"""Replay of executed, bounded residual actions (never absolute VLA targets).

A transition holds obs, action, reward, next_obs, terminated and truncated.
Collated batches use the plural keys actions and rewards.
Successful episodes can be added with add_episode(..., success=True), making
their actual residual actions available for the actor's success imitation loss.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


def _observation(obs: dict) -> dict:
    images, state = np.asarray(obs["images"]), np.asarray(obs["state"], dtype=np.float32)
    if images.dtype != np.uint8 or images.ndim != 3:
        raise ValueError("Replay observation images must be CHW uint8")
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError("Replay observation state must be a finite vector")
    return {"images": images.copy(), "state": state.copy()}


def stack_transitions(transitions: list[dict]) -> dict:
    if not transitions:
        raise ValueError("Cannot stack an empty transition list")
    return {
        "obs": {key: np.stack([t["obs"][key] for t in transitions]) for key in ("images", "state")},
        "next_obs": {key: np.stack([t["next_obs"][key] for t in transitions])
                     for key in ("images", "state")},
        **{key: np.asarray([t[key] for t in transitions])
           for key in ("actions", "rewards", "terminated", "truncated", "success")},
    }


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.storage: list[dict] = []
        self.position = 0
        self._successful: set[int] = set()
        self._signature = None

    def __len__(self) -> int:
        return len(self.storage)

    @property
    def num_successful(self) -> int:
        return len(self._successful)

    def add(self, transition: dict) -> None:
        obs, next_obs = _observation(transition["obs"]), _observation(transition["next_obs"])
        action = np.asarray(transition.get("action", transition.get("actions")), dtype=np.float32)
        if action.ndim != 1 or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
            raise ValueError("Replay actions must be bounded residual vectors in [-1, 1]")
        reward = float(transition.get("reward", transition.get("rewards")))
        if not np.isfinite(reward):
            raise ValueError("Replay reward must be finite")
        signature = (obs["images"].shape, obs["state"].shape, action.shape)
        if any(obs[key].shape != next_obs[key].shape for key in obs):
            raise ValueError("Current and next observation shapes must match")
        if self._signature is not None and self._signature != signature:
            raise ValueError("Replay transition shape changed")
        flags = {key: transition.get(key, False) if key == "success" else transition[key]
                 for key in ("terminated", "truncated", "success")}
        if any(np.ndim(value) != 0 or value not in (False, True) for value in flags.values()):
            raise ValueError("Replay flags must be scalar booleans")
        item = {"obs": obs, "next_obs": next_obs, "actions": action.copy(), "rewards": reward,
                **{key: bool(value) for key, value in flags.items()}}
        self._signature = signature
        self._successful.discard(self.position)
        if item["success"]:
            self._successful.add(self.position)
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def add_episode(self, transitions: list[dict], success: bool) -> None:
        for transition in transitions:
            self.add({**transition, "success": success})

    def _sample_items(self, batch_size: int, successful_only: bool = False) -> list[dict]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        indices = sorted(self._successful) if successful_only else range(len(self.storage))
        if len(indices) == 0:
            raise ValueError("Requested replay partition is empty")
        choices = self.rng.choice(indices, size=batch_size, replace=True)
        return [self.storage[index] for index in choices]

    def sample(self, batch_size: int, successful_only: bool = False) -> dict:
        return stack_transitions(self._sample_items(batch_size, successful_only))

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "storage": self.storage, "position": self.position,
                "rng": copy.deepcopy(self.rng.bit_generator.state), "signature": self._signature}

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity differs from checkpoint")
        if len(state["storage"]) > self.capacity:
            raise ValueError("Checkpoint replay exceeds declared capacity")
        candidate = ReplayBuffer(self.capacity)
        for item in state["storage"]:
            candidate.add(item)
        position = state["position"]
        if not isinstance(position, int) or not 0 <= position < self.capacity:
            raise ValueError("Invalid checkpoint replay position")
        if len(candidate) < self.capacity and position != len(candidate):
            raise ValueError("Invalid checkpoint position for a partially filled replay")
        candidate.rng.bit_generator.state = state["rng"]
        self.storage, self.position = candidate.storage, position
        self._signature, self._successful = candidate._signature, candidate._successful
        self.rng = candidate.rng

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        """Atomically write an archive of genuine executed residual transitions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = {"schema_version": 1, "action_type": "bounded_joint_residual",
                   "replay": self.state_dict(), "metadata": metadata or {}}
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(archive, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path, expected_metadata: dict | None = None):
        """Load a *trusted local* archive and validate transitions and provenance.

        torch serialization can execute code; never use this on untrusted files.
        The runner also validates train split, checkpoint identity and controller
        configuration in metadata before making a demonstration source eligible.
        """
        archive = torch.load(path, map_location="cpu", weights_only=False)
        if archive.get("schema_version") != 1 or archive.get("action_type") != "bounded_joint_residual":
            raise ValueError("Not a supported residual replay archive")
        metadata = archive["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Replay metadata must be a dictionary")
        for key, value in (expected_metadata or {}).items():
            if metadata.get(key) != value:
                raise ValueError(f"Replay provenance mismatch for {key}")
        buffer = cls(archive["replay"]["capacity"])
        buffer.load_state_dict(archive["replay"])
        return buffer, metadata


class MixedReplay:
    """An explicit demo buffer enforces exactly 50/50 online/demo sampling.

    If demos=None, training uses online experience only and must be reported as
    such. Passing an empty demo buffer is an error, never a silent fallback.
    """
    def __init__(self, online: ReplayBuffer, demos: ReplayBuffer | None = None):
        self.online, self.demos = online, demos

    def sample(self, batch_size: int) -> dict:
        if self.demos is None:
            return self.online.sample(batch_size)
        if batch_size % 2:
            raise ValueError("50/50 demo/online replay requires an even batch size")
        items = (self.online._sample_items(batch_size // 2)
                 + self.demos._sample_items(batch_size // 2))
        batch = stack_transitions(items)
        batch["is_demo"] = np.array([False] * (batch_size // 2) + [True] * (batch_size // 2))
        return batch

    def sample_success(self, batch_size: int) -> dict | None:
        buffers = [buffer for buffer in (self.online, self.demos)
                   if buffer is not None and buffer.num_successful]
        if not buffers:
            return None
        # Uniform over successful transitions, regardless of source buffer.
        total = sum(buffer.num_successful for buffer in buffers)
        counts = self.online.rng.multinomial(batch_size,
            [buffer.num_successful / total for buffer in buffers])
        items = []
        for buffer, count in zip(buffers, counts):
            if count:
                items.extend(buffer._sample_items(int(count), successful_only=True))
        return stack_transitions(items)
