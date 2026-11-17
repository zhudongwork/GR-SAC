"""Frozen, batched SimpleMemVLA rollouts with joint-space residual control.

The adapter consumes only camera images, proprioception, the frozen policy's
actions and an observable clock. RoboMME's internal subgoals and dense rewards
are deliberately not part of the residual observation or reward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


TASKS = (
    "PickXtimes", "StopCube", "SwingXtimes", "BinFill", "VideoUnmaskSwap",
    "VideoUnmask", "ButtonUnmaskSwap", "ButtonUnmask", "VideoRepick",
    "VideoPlaceButton", "VideoPlaceOrder", "PickHighlight", "InsertPeg",
    "MoveCube", "PatternLock", "RouteStick",
)
ENABLED_STATE_INDEX = 24


class RoboMMEResetError(RuntimeError):
    """Keep the simulator's reset classification across the evaluator process."""

    def __init__(self, spec, split, result):
        data = result if isinstance(result, dict) else {"reason": str(result)}
        self.failure = {"spec": dict(spec), "split": split,
                        **{key: data.get(key) for key in ("failure_kind", "attempts", "reason", "traceback")}}
        super().__init__(f"RoboMME reset failed for {spec}:\n{data.get('traceback') or data.get('reason')}")


def load_episode_specs(metadata_dir: str | Path, tasks: list[str] | tuple[str, ...]) -> list[dict]:
    """Read actual episode IDs, never substitute test seeds or sequential IDs."""
    directory = Path(metadata_dir)
    specs = []
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"Unknown RoboMME task: {task}")
        path = directory / f"record_dataset_{task}_metadata.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing episode metadata: {path}")
        payload = json.loads(path.read_text())
        seen = set()
        for record in payload.get("records", []):
            if record.get("task", payload.get("env_id")) != task:
                raise ValueError(f"Metadata task mismatch in {path}")
            episode, seed = record.get("episode"), record.get("seed")
            if not isinstance(episode, int) or isinstance(episode, bool) or seed is None:
                raise ValueError(f"Each metadata record needs integer episode and seed: {path}")
            if episode in seen:
                raise ValueError(f"Duplicate episode {episode} in {path}")
            int(seed)  # The builder must be able to reproduce this seed.
            seen.add(episode)
            specs.append({"task": task, "episode": episode})
        if not seen:
            raise ValueError(f"No episodes for {task} in {path}")
    return specs


def _vector7(value: float | list[float], name: str, *, positive: bool) -> np.ndarray:
    vector = np.broadcast_to(np.asarray(value, dtype=np.float32), (7,)).copy()
    if not np.isfinite(vector).all() or (vector < 0).any() or (positive and (vector <= 0).any()):
        raise ValueError(f"{name} must contain seven finite {'positive' if positive else 'nonnegative'} values")
    return vector


class JointResidualController:
    """Equation (17) ordering: command uses o_j; action updates o_(j+1).

    All seven offset coordinates and delta are in radians. No clipping of the
    underlying base command is added, so zero offsets recover it exactly.
    """

    def __init__(self, delta=0.005, gain=1.0, offset_bound=0.1, offset_decay=1.0):
        self.delta = _vector7(delta, "delta", positive=True)
        self.bound = _vector7(offset_bound, "offset_bound", positive=True)
        if not np.isfinite(gain) or gain < 0:
            raise ValueError("gain must be finite and nonnegative")
        self.gain = float(gain)
        if not np.isfinite(offset_decay) or not 0 <= offset_decay <= 1:
            raise ValueError("offset_decay must be in [0, 1]")
        self.decay = float(offset_decay)
        self.offset = np.zeros(7, dtype=np.float32)

    def reset(self) -> None:
        self.offset.fill(0)

    def command(self, base_action: np.ndarray, residual: np.ndarray, enabled=True) -> tuple[np.ndarray, np.ndarray]:
        base = np.asarray(base_action, dtype=np.float32)
        action = np.asarray(residual, dtype=np.float32)
        if base.shape != (8,) or not np.isfinite(base).all():
            raise ValueError("Expected a finite 8-D base action")
        if action.shape != (7,) or not np.isfinite(action).all() or (np.abs(action) > 1.000001).any():
            raise ValueError("Expected a finite 7-D residual action in [-1, 1]")
        action = np.clip(action, -1, 1)
        if not enabled:
            self.reset()
            action = np.zeros(7, dtype=np.float32)
        command = base.copy()
        command[:7] += self.offset
        self.offset = np.clip(self.decay * self.offset + self.gain * self.delta * action,
                              -self.bound, self.bound)
        return command, action.copy()


def _make_pool(num_envs: int, split: str, metadata_dir: str, max_steps: int,
               step_timeout: float, reset_timeout: float):
    # Import simulator dependencies only in a real collector, not in the learner.
    from robomme_sim.inproc_pool import InProcSimPool
    from robomme_sim.robomme_env import RoboMMESimEnv, SimEnvService, _setup_robomme_path
    from robomme_sim.robomme_env import encode_frames, encode_states

    class MetadataEnv(RoboMMESimEnv):
        def __init__(self, task_name):
            _setup_robomme_path()
            import robomme.robomme_env  # noqa: F401 - environment registration
            from robomme.env_record_wrapper import BenchmarkEnvBuilder

            self.task_name, self.dataset_split = task_name, split
            self.max_steps = max_steps
            self.builder = BenchmarkEnvBuilder(
                env_id=task_name, dataset=split, action_space="joint_angle",
                max_steps=max_steps, override_metadata_path=metadata_dir,
            )
            self.num_episodes = self.builder.get_episode_num()
            if self.num_episodes <= 0:
                raise RuntimeError(f"No {split} metadata for {task_name} in {metadata_dir}")
            self.env, self._status, self._done = None, "ongoing", False

    class ResidualService(SimEnvService):
        def _ensure_env(self, task_name):
            if self.env is None or self.task_name != task_name:
                if self.env is not None:
                    self.env.close()
                self.env, self.task_name = MetadataEnv(task_name), task_name

        def step(self, payload):
            chunk = np.asarray(payload["action_chunk"], dtype=np.float32)
            if chunk.shape != (1, 8):
                raise ValueError("Residual collection requires exactly one control step")
            obs, terminated, truncated, info = self.env.step_one(chunk[0])
            if obs is None or self.env.status == "error":
                return {"error": info.get("error_message", "RoboMME step returned no observation")}
            return {
                "frames": encode_frames(obs), "states": encode_states(obs), "consumed": 1,
                "done": self.env.done, "success": self.env.success, "status": self.env.status,
                "terminated": terminated, "truncated": truncated,
            }

    class ResidualPool(InProcSimPool):
        def start(self):
            self.services = [ResidualService(dataset_split=split, max_steps=max_steps)
                             for _ in range(self.num_envs)]
            self.alive = [True] * self.num_envs

    pool = ResidualPool(num_envs, dataset_split=split, max_steps=max_steps,
                        step_timeout=step_timeout, reset_timeout=reset_timeout)
    pool.start()
    return pool


class RoboMMEResidualGroup:
    """One GPU's base-policy replica and the existing parallel SAPIEN pool.

    ``reset(specs)`` returns ``{slot: observation}``; ``step({slot: action})``
    returns transitions. All active slots must be stepped together. Base
    decisions are batched and retain the original execute_horizon; the small
    residual policy takes one decision per control step.

    State layout: proprio(8), current base action(8), offset(7), elapsed fraction
    (1), enabled(1), queue phase(1), task one-hot(16), padded remaining base queue(H*8).
    Official success, failure and benchmark timeouts are terminal outcomes.
    Invalid simulator output raises and is never inserted into replay.
    """

    def __init__(self, args: Any, *, metadata_dir: str | Path, split="train", num_envs=4,
                 delta=0.005, gain=1.0, offset_bound=0.1, image_size=84,
                 enabled_tasks=None, start_step=0, scene_failure_policy="raise", pool=None, batched=None,
                 offset_decay=1.0, phase_gate="always", max_target_span=0.08,
                 max_tracking_error=0.1, max_gripper_span=0.1,
                 buffer_factory=None, normalize_state=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")
        if scene_failure_policy not in ("raise", "skip_sampling_exhausted"):
            raise ValueError("Unknown scene_failure_policy")
        if split != "train" and scene_failure_policy != "raise":
            raise ValueError("Scene skipping is supported only for training; val/test must stay strict")
        self.scene_failure_policy = scene_failure_policy
        self.reset_failures: list[dict] = []
        if getattr(args, "pipelined", False):
            raise ValueError("Residual collection uses the original batched policy, not pipelined prefill")
        self.execute_horizon = int(args.execute_horizon)
        self.max_steps, self.num_envs = int(args.max_steps), int(num_envs)
        self.image_size, self.start_step = int(image_size), int(start_step)
        if min(self.execute_horizon, self.max_steps, self.num_envs, self.image_size) < 1 or self.start_step < 0:
            raise ValueError("Horizons, environment count and image size must be positive; start_step >= 0")
        # Match eval_success.run_group's ceil(max_steps / H) + 2 base decisions.
        # DemonstrationWrapper counts only non-demonstration steps; its counter
        # is not the external control-step clock used by this adapter.
        max_decisions = (self.max_steps + self.execute_horizon - 1) // self.execute_horizon + 2
        self.max_control_steps = max_decisions * self.execute_horizon
        self.enabled_tasks = set(TASKS if enabled_tasks is None else enabled_tasks)
        if phase_gate not in ("always", "low_motion"):
            raise ValueError("Unknown phase_gate")
        self.phase_gate = phase_gate
        self.max_target_span, self.max_tracking_error, self.max_gripper_span = (
            max_target_span, max_tracking_error, max_gripper_span)
        if any(not np.isfinite(x) or x <= 0 for x in (
                max_target_span, max_tracking_error, max_gripper_span)):
            raise ValueError("Gate thresholds must be finite and positive")
        if not self.enabled_tasks.issubset(TASKS):
            raise ValueError("Unknown task in enabled_tasks")
        self.metadata_dir, self.split = str(Path(metadata_dir).resolve()), split
        if batched is None:
            from robomme_sim.eval_success import build_policy
            batched, buffer_factory, normalize_state = build_policy(args)
        if buffer_factory is None:
            raise ValueError("buffer_factory is required")
        self.batched, self.buffer_factory, self.normalize_state = batched, buffer_factory, normalize_state
        self.batched.model.eval().requires_grad_(False)
        # Residual exploration must not advance the base flow sampler's RNG.
        # fork_rng restores the caller's stream after each base decision.
        self._base_device = torch.device(self.batched.device)
        seed = int(getattr(args, "seed", 0))
        self._base_cpu_rng = torch.Generator(device="cpu").manual_seed(seed).get_state()
        self._base_cuda_rng = None
        if self._base_device.type == "cuda":
            if self._base_device.index is None:
                self._base_device = torch.device("cuda", torch.cuda.current_device())
            self._base_cuda_rng = torch.Generator(device=self._base_device).manual_seed(seed).get_state()
        if self.execute_horizon > int(self.batched.action_horizon):
            raise ValueError("execute_horizon exceeds the base policy's action_horizon")
        if int(self.batched.action_dim) != 8:
            raise ValueError("RoboMME requires an 8-D joint-position base policy")
        self.pool = pool if pool is not None else _make_pool(
            self.num_envs, split, self.metadata_dir, self.max_steps,
            float(getattr(args, "step_timeout", 300)), float(getattr(args, "reset_timeout", 3600)),
        )
        self.controllers = [JointResidualController(delta, gain, offset_bound, offset_decay)
                            for _ in range(self.num_envs)]
        self.buffers = [self.buffer_factory() for _ in range(self.num_envs)]
        self.active: list[int] = []
        self._specs: list[dict] = []
        self._frames: dict[int, dict] = {}
        self._states: dict[int, np.ndarray] = {}
        self._instructions: dict[int, str] = {}
        self._queues: dict[int, np.ndarray] = {}
        self._steps: dict[int, int] = {}
        self._observations: dict[int, dict] = {}

    @property
    def state_dim(self):
        return 42 + 8 * self.execute_horizon

    def _enabled(self, slot: int) -> bool:
        if self._specs[slot]["task"] not in self.enabled_tasks or self._steps[slot] < self.start_step:
            return False
        if self.phase_gate == "always":
            return True
        targets = self._queues[slot]
        if not len(targets):
            return False
        # A deployable low-motion eligibility rule, NOT a semantic stage label.
        # Every input is already present in the residual observation. Clear the
        # offset on exit in _observation so observation and executed action agree.
        return bool(np.max(np.ptp(targets[:, :7], axis=0)) <= self.max_target_span
                    and np.max(np.abs(targets[0, :7] - self._states[slot][:7])) <= self.max_tracking_error
                    and np.ptp(targets[:, 7]) <= self.max_gripper_span)

    def _observe(self, slot: int, result: dict) -> None:
        frames, states = result.get("frames"), result.get("states")
        if not frames or not states:
            raise RuntimeError(f"RoboMME slot {slot} returned no valid frames or state")
        cam_map = {key.split(".")[-1]: key for key in self.buffers[slot].image_keys}
        for frame in frames:
            self.buffers[slot].observe({cam_map.get(key, key): np.asarray(value, dtype=np.uint8)
                                        for key, value in frame.items()})
        self._frames[slot] = frames[-1]
        state = np.asarray(states[-1], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise RuntimeError(f"Invalid RoboMME proprioception in slot {slot}")
        self._states[slot] = state

    @torch.no_grad()
    def _replan(self, slots: list[int]) -> None:
        if not slots:
            return
        processed = [self.buffers[slot]._prepare_inputs(self._instructions[slot]) for slot in slots]
        states = []
        for slot in slots:
            state = None
            if self.normalize_state is not None:
                state = self.normalize_state.normalize(torch.from_numpy(self._states[slot]))
                state = state.unsqueeze(0).to(self.batched.device)
            states.append(state)
        devices = [self._base_device.index] if self._base_device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self._base_cpu_rng)
            if self._base_cuda_rng is not None:
                torch.cuda.set_rng_state(self._base_cuda_rng, self._base_device)
            predictions = self.batched.generate_batch(processed, states)
            self._base_cpu_rng = torch.get_rng_state()
            if self._base_cuda_rng is not None:
                self._base_cuda_rng = torch.cuda.get_rng_state(self._base_device)
        if len(predictions) != len(slots):
            raise RuntimeError("Batched base policy returned the wrong number of predictions")
        for slot, (actions, _subtask) in zip(slots, predictions):
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 8 or len(actions) < self.execute_horizon or not np.isfinite(actions).all():
                raise RuntimeError(f"Invalid frozen base action chunk in slot {slot}")
            self._queues[slot] = actions[:self.execute_horizon].copy()

    def _observation(self, slot: int) -> dict:
        enabled = self._enabled(slot)
        if not enabled:
            self.controllers[slot].reset()
        images = []
        for camera in ("front", "wrist"):
            frame = np.asarray(self._frames[slot][camera])
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"Expected HWC RGB {camera} image, got {frame.shape}")
            resized = Image.fromarray(frame.astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR)
            images.append(np.asarray(resized).transpose(2, 0, 1))
        queue = np.zeros((self.execute_horizon, 8), dtype=np.float32)
        remaining = self._queues[slot]
        queue[:len(remaining)] = remaining
        onehot = np.zeros(len(TASKS), dtype=np.float32)
        onehot[TASKS.index(self._specs[slot]["task"])] = 1
        state = np.concatenate([
            self._states[slot], queue[0], self.controllers[slot].offset,
            np.asarray([min(self._steps[slot] / self.max_steps, 1.0), float(enabled),
                        1.0 - len(remaining) / self.execute_horizon], dtype=np.float32),
            onehot, queue.reshape(-1),
        ])
        return {"images": np.concatenate(images, axis=0).copy(), "state": state}

    def reset(self, specs: list[dict]) -> dict[int, dict]:
        if not 1 <= len(specs) <= self.num_envs:
            raise ValueError("reset requires 1..num_envs episode specs")
        allowed = {(spec["task"], spec["episode"]) for spec in load_episode_specs(
            self.metadata_dir, list(dict.fromkeys(spec["task"] for spec in specs)))}
        for spec in specs:
            if (spec["task"], spec["episode"]) not in allowed:
                raise ValueError(f"Episode absent from explicit {self.split} metadata: {spec}")
        results = self.pool.reset(specs)
        self._specs = [dict(spec) for spec in specs]
        self.active = []
        self._observations = {}
        self.reset_failures = []
        for slot in range(len(specs)):
            result = results[slot]
            if not isinstance(result, dict) or not result.get("ok"):
                if (self.scene_failure_policy == "skip_sampling_exhausted"
                        and isinstance(result, dict)
                        and result.get("failure_kind") == "scene_sampling_exhausted"):
                    self.reset_failures.append({"slot": slot, "spec": dict(specs[slot]),
                        "failure_kind": result["failure_kind"], "reason": result.get("reason"),
                        "traceback": result.get("traceback"), "attempts": result.get("attempts")})
                    continue
                raise RoboMMEResetError(specs[slot], self.split, result)
            self.buffers[slot].reset()
            self.controllers[slot].reset()
            self._steps[slot] = 0
            self._instructions[slot] = str(result["instruction"])
            self._observe(slot, result)
            self.active.append(slot)
        self._replan(self.active)
        self._observations = {slot: self._observation(slot) for slot in self.active}
        return dict(self._observations)

    def step(self, actions: dict[int, np.ndarray]) -> list[dict]:
        if set(actions) != set(self.active):
            raise ValueError("Supply exactly one residual action for every active slot")
        if not self.active:
            return []
        slots = list(self.active)
        chunks: list[np.ndarray | None] = [None] * self.num_envs
        effective_actions = {}
        for slot in slots:
            command, effective = self.controllers[slot].command(
                self._queues[slot][0], actions[slot], enabled=self._enabled(slot))
            chunks[slot] = command[None]
            effective_actions[slot] = effective
        results = self.pool.step(chunks, slots)
        transitions = []
        for slot in slots:
            result = results[slot]
            if not isinstance(result, dict) or result.get("error") or result.get("error_message") or result.get("status") == "error":
                raise RuntimeError(f"Invalid RoboMME transition for {self._specs[slot]}: {result}")
            if int(result.get("consumed", 0)) != 1:
                raise RuntimeError("Residual rollout must consume exactly one control step")
            self._steps[slot] += 1
            self._queues[slot] = self._queues[slot][1:]
            self._observe(slot, result)
            status = str(result.get("status", "ongoing"))
            success = status == "success"
            # The benchmark defines a finite-horizon success objective. Its time
            # limit is an outcome, not an administrative replay truncation.
            terminal = bool(result.get("done") or result.get("terminated") or result.get("truncated")
                            or status in ("success", "fail", "timeout"))
            timeout_source = "environment" if status == "timeout" else None
            if not terminal and self._steps[slot] >= self.max_control_steps:
                # The reference evaluator ends an unfinished episode at this
                # bound as an unsuccessful outcome. Keep the final real frame
                # and do not replan or execute an additional base action block.
                terminal, status, timeout_source = True, "timeout", "evaluator"
            if terminal:
                self.active.remove(slot)
            transitions.append({
                "slot": slot, "obs": self._observations[slot], "action": effective_actions[slot],
                "reward": float(success), "terminated": terminal, "truncated": False,
                "success": success, "status": status, "discount_steps": 1,
                "timeout_source": timeout_source,
            })
        self._replan([slot for slot in self.active if len(self._queues[slot]) == 0])
        for transition in transitions:
            slot = transition["slot"]
            next_obs = self._observation(slot)
            transition["next_obs"] = next_obs
            self._observations[slot] = next_obs
        return transitions

    def close(self):
        self.pool.close()
        self.active = []
