"""One SAC learner with asynchronous, batched frozen-VLA collectors.

Run from the repo: python -m simplememvla.residual_rl.train --config ...
No simulator/model imports occur before GPU visibility is set in each process.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from .config import ROOT, TASKS, TrainConfig, checkpoint_identity, file_sha256, load_config, metadata_specs, preflight
from .sampling import difficulty_balanced_specs, demo_task_counts
from .validation import ValidationEvaluationError, ValidationMonitor, validate_actor


class CollectorFailure(RuntimeError):
    """Collector failure observed between complete learner updates."""


class ValidationFailure(CollectorFailure):
    """A completed validation rejected the policy or evaluation itself failed."""


def atomic_save(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def actor_weights(actor):
    return {key: value.detach().cpu().numpy().copy() for key, value in actor.state_dict().items()}


def broadcast_weights(queues, actor, version):
    message = {"version": version, "weights": actor_weights(actor)}
    for channel in queues:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(message)
        except queue.Full:
            pass  # A pending newer snapshot is consumed at the next control step.


def put_message(channel, message, stop):
    while not stop.is_set():
        try:
            channel.put(message, timeout=1)
            return True
        except queue.Full:
            continue
    return False


def bootstrap_specs(specs, per_task):
    counts, selected = {}, []
    for spec in specs:
        task = spec["task"]
        if counts.get(task, 0) < per_task:
            selected.append(spec)
            counts[task] = counts.get(task, 0) + 1
    if any(count < per_task for count in counts.values()):
        raise ValueError("bootstrap_episodes_per_task exceeds available train episodes")
    return selected


def collector_main(rank, cfg_dict, sac_dict, initial_weights, specs, boot_specs,
                   output, weights_queue, start_online, stop, initial_version=0):
    cfg = TrainConfig(**cfg_dict)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus[rank + 1]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(cfg.cpu_threads)
    group = None
    try:
        import numpy as np
        import torch
        from .networks import SquashedGaussianActor, actor_kwargs
        from .robomme import RoboMMEResidualGroup
        torch.set_num_threads(cfg.cpu_threads)
        torch.cuda.set_device(0)
        worker_seed = cfg.seed + 1009 * (rank + 1)
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        rng = np.random.default_rng(worker_seed)
        actor = SquashedGaussianActor(**actor_kwargs(sac_dict)).to("cuda:0")
        actor.load_state_dict({key: torch.from_numpy(value) for key, value in initial_weights.items()})
        actor.eval().requires_grad_(False)
        version = initial_version
        group = RoboMMEResidualGroup(cfg.policy_args(worker_seed),
            metadata_dir=Path(cfg.metadata_root) / "train", split="train",
            num_envs=cfg.envs_per_collector, delta=cfg.delta, gain=cfg.gain,
            offset_bound=cfg.offset_bound, image_size=cfg.image_size,
            enabled_tasks=cfg.enabled_tasks, start_step=cfg.start_step,
            scene_failure_policy=cfg.train_scene_failure_policy, **cfg.controller_kwargs())
        empty_resets = 0

        def run_episode_group(selected, bootstrap):
            nonlocal version, empty_resets
            obs = group.reset(selected)
            for failure in group.reset_failures:
                put_message(output, {"kind": "reset_failed", "rank": rank, "split": "train",
                    "bootstrap": bootstrap, "policy": cfg.train_scene_failure_policy,
                    "policy_version": version, "steps": 0, "skipped": True, **failure}, stop)
            empty_resets = 0 if obs else empty_resets + 1
            if empty_resets >= cfg.max_consecutive_empty_resets:
                raise RuntimeError(f"Collector {rank}: {empty_resets} consecutive reset groups "
                                   "produced no usable environments; inspect reset_failed events")
            trajectories = {slot: [] for slot in obs}
            start_version = version
            started = time.monotonic()
            while obs and not stop.is_set():
                try:
                    message = weights_queue.get_nowait()
                    actor.load_state_dict({key: torch.from_numpy(value)
                                           for key, value in message["weights"].items()})
                    version = message["version"]
                except queue.Empty:
                    pass
                slots = sorted(obs)
                if bootstrap:
                    actions = np.zeros((len(slots), 7), dtype=np.float32)
                else:
                    batch = {key: torch.as_tensor(np.stack([obs[i][key] for i in slots]), device="cuda:0")
                             for key in ("images", "state")}
                    with torch.no_grad():
                        actions = actor.sample(batch)[0]
                        actions = actions * batch["state"][:, 24:25]
                        actions = actions.cpu().numpy()
                transitions = group.step(dict(zip(slots, actions)))
                for transition in transitions:
                    slot = transition["slot"]
                    trajectories[slot].append(transition)
                    if transition["terminated"] or transition["truncated"]:
                        success = bool(transition["success"])
                        offsets = np.stack([row["obs"]["state"][16:23] for row in trajectories[slot]])
                        residuals = np.stack([row["action"] for row in trajectories[slot]])
                        active_fraction = float(np.mean([row["obs"]["state"][24] for row in trajectories[slot]]))
                        # Failed bootstrap episodes are logged, but are never called demos.
                        payload = trajectories.pop(slot) if (not bootstrap or success) else []
                        put_message(output, {"kind": "episode", "rank": rank,
                            "bootstrap": bootstrap, "spec": selected[slot], "split": "train",
                            "success": success, "status": transition["status"],
                            "timeout_source": transition.get("timeout_source"),
                            "mean_abs_residual": float(np.abs(residuals).mean()),
                            "mean_abs_offset_rad": float(np.abs(offsets).mean()),
                            "max_abs_offset_rad": float(np.abs(offsets).max()),
                            "offset_bound_fraction": float((np.abs(offsets) >= .95 * cfg.offset_bound).mean()),
                            "active_fraction": active_fraction,
                            "transitions": payload, "steps": group_steps.get(slot, 0) + 1,
                            "policy_version_start": start_version, "policy_version_end": version,
                            "seconds": time.monotonic() - started}, stop)
                        obs.pop(slot)
                    else:
                        obs[slot] = transition["next_obs"]
                    group_steps[slot] = group_steps.get(slot, 0) + 1

        for offset in range(0, len(boot_specs), cfg.envs_per_collector):
            group_steps = {}
            run_episode_group(boot_specs[offset:offset + cfg.envs_per_collector], True)
            if stop.is_set():
                return
        put_message(output, {"kind": "bootstrap_done", "rank": rank}, stop)
        while not start_online.wait(timeout=1):
            if stop.is_set():
                return
        while not stop.is_set():
            chosen = rng.choice(len(specs), size=cfg.envs_per_collector,
                                replace=len(specs) < cfg.envs_per_collector)
            selected = [specs[int(index)] for index in chosen]
            group_steps = {}
            run_episode_group(selected, False)
    except BaseException:
        put_message(output, {"kind": "error", "rank": rank, "traceback": traceback.format_exc()}, stop)
        raise
    finally:
        if group is not None:
            group.close()
        # Parent shutdown may deliberately discard pending episode messages.
        output.cancel_join_thread()


def write_event(stream, event):
    event = {"time_unix": time.time(), **event}
    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    stream.flush()
    print(json.dumps(event, ensure_ascii=False), flush=True)


def run(cfg, resume=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus[0]
    os.environ["OMP_NUM_THREADS"] = str(cfg.cpu_threads)
    import numpy as np
    import torch
    from .replay import MixedReplay, ReplayBuffer
    from .sac import ResidualSAC, SACConfig
    torch.set_num_threads(cfg.cpu_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; training needs a learner GPU and collector GPUs")
    torch.cuda.set_device(0)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    sac_cfg = SACConfig(**cfg.sac_kwargs())
    learner = ResidualSAC(sac_cfg, "cuda:0")
    online = ReplayBuffer(cfg.replay_capacity, cfg.seed)
    demos = ReplayBuffer(cfg.demo_capacity, cfg.seed + 1)
    print("[residual_rl] Hashing frozen checkpoint weights for provenance...", flush=True)
    identity = checkpoint_identity(cfg.checkpoint)
    specs = metadata_specs(cfg, "train")
    provenance = {"protocol": cfg.protocol(), "base_checkpoint": identity,
                  "train_metadata": {task: file_sha256(Path(cfg.metadata_root) / "train" /
                    f"record_dataset_{task}_metadata.json") for task in cfg.tasks}}
    output_dir = Path(cfg.output_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"Existing run {output_dir}; use --resume or a new output_dir")
    if cfg.demo_archive and not resume:
        loaded, _ = ReplayBuffer.load(cfg.demo_archive, expected_metadata=provenance)
        ordered = (loaded.storage[loaded.position:] + loaded.storage[:loaded.position]
                   if len(loaded) == loaded.capacity else loaded.storage)
        for transition in ordered:
            demos.add(transition)
        del ordered, loaded
        if not len(demos):
            raise ValueError("Empty demo archive")
    online_received, completed_episodes = 0, 0
    validation_monitor = ValidationMonitor()
    validation_invalid_count = 0
    validation_protocol = None
    if cfg.validation_every:
        validation_protocol = {**cfg.validation_protocol(), "metadata_hashes": {
            task: file_sha256(Path(cfg.metadata_root) / "val" / f"record_dataset_{task}_metadata.json")
            for task in cfg.tasks}}
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError("Resume protocol, base identity or metadata mismatch")
        learner.load_state_dict(saved["learner"])
        online.load_state_dict(saved["online"])
        demos.load_state_dict(saved["demos"])
        online_received = saved["online_received"]
        completed_episodes = saved["completed_episodes"]
        validation_monitor = ValidationMonitor(**saved.get("validation_monitor", {}))
        validation_invalid_count = saved.get("validation_invalid_count", 0)
        if validation_monitor.best_score is not None and saved.get("validation_protocol") != validation_protocol:
            raise ValueError("Resume validation protocol changed; cannot reuse best score/guard history")
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device=0)
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        del saved
    if learner.updates >= cfg.max_updates:
        print(f"Checkpoint already reached {learner.updates} updates", flush=True)
        return
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_paths = (list((ROOT / "simplememvla/residual_rl").glob("*.py"))
                    + list((ROOT / "robomme_sim").glob("*.py"))
                    + list((ROOT / "robomme_sim/robomme").rglob("*.py")))
    source_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(source_paths)}
    manifest = {"config": asdict(cfg), "provenance": provenance, "git_revision": revision,
                "source_hashes": source_hashes, "resume_from": str(resume) if resume else None,
                "demo_source": "archive" if cfg.demo_archive else "successful_zero_residual_train_rollouts",
                "resume_boundary": "learner/replay/RNG restored; collectors restart episodes"}
    if manifest_path.exists():
        previous_path = output_dir / f"manifest_before_resume_{time.time_ns()}.json"
        previous_path.write_bytes(manifest_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    mix = MixedReplay(online, demos)
    context = mp.get_context("spawn")
    stop, start_online = context.Event(), context.Event()
    output = context.Queue(maxsize=cfg.queue_size)
    n_collectors = len(cfg.gpus) - 1
    weight_queues = [context.Queue(maxsize=1) for _ in range(n_collectors)]
    bootstrap = []
    if not len(demos):
        bootstrap = (difficulty_balanced_specs(specs, cfg.bootstrap_episodes_per_task,
                     Path(cfg.metadata_root) / "train", cfg.seed)
                     if cfg.bootstrap_sampling == "difficulty_balanced"
                     else bootstrap_specs(specs, cfg.bootstrap_episodes_per_task))
    def check_demo_coverage():
        counts = demo_task_counts(demos, TASKS) if cfg.require_demo_task_coverage else {}
        missing = [task for task in cfg.tasks if not counts.get(task)] if counts else []
        if missing:
            raise CollectorFailure(f"No retained successful demo transitions for {missing}; "
                                   "increase bootstrap budget/demo capacity or supply valid residual demos")
        return counts

    if len(demos):
        check_demo_coverage()
    if len(specs) < n_collectors:
        raise ValueError("Need at least one train episode per collector")
    workers = []
    initial = actor_weights(learner.actor)
    boot_done = set()
    latest_message = time.monotonic()
    last_saved_update = -1

    def save_actor():
        path = checkpoint_dir / f"actor_{learner.updates:08d}.pt"
        atomic_save({"schema": "simplememvla-residual-policy-v1", "config": asdict(cfg),
            "provenance": provenance, "sac_config": asdict(sac_cfg),
            "actor": actor_weights(learner.actor), "updates": learner.updates}, path)
        return path

    def save(filename="latest.pt", *, failure=None):
        nonlocal last_saved_update
        snapshot = {"schema": "simplememvla-residual-training-v1", "config": asdict(cfg),
            "provenance": provenance, "learner": learner.state_dict(),
            "online": online.state_dict(), "demos": demos.state_dict(),
            "online_received": online_received, "completed_episodes": completed_episodes,
            "validation_monitor": validation_monitor.state_dict(),
            "validation_invalid_count": validation_invalid_count,
            "validation_protocol": validation_protocol,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
        if failure is not None:
            snapshot["failure"] = failure
        checkpoint_path = checkpoint_dir / filename
        atomic_save(snapshot, checkpoint_path)
        if failure is not None:
            return checkpoint_path
        # Small policy-only snapshots can be selected on val without duplicating all replay.
        save_actor()
        last_saved_update = learner.updates
        return checkpoint_path

    try:
        for rank in range(n_collectors):
            proc = context.Process(target=collector_main, name=f"residual-collector-{rank}",
                args=(rank, asdict(cfg), asdict(sac_cfg), initial, specs[rank::n_collectors],
                      bootstrap[rank::n_collectors], output, weight_queues[rank], start_online, stop, learner.updates))
            proc.start()
            workers.append(proc)
        if len(demos):
            start_online.set()
        with (output_dir / "metrics.jsonl").open("a", buffering=1) as events:
            write_event(events, {"kind": "start", "updates": learner.updates,
                "online_received": online_received, "collector_count": n_collectors,
                "bootstrap_episodes": len(bootstrap),
                "validation_scene_failure_policy": cfg.validation_scene_failure_policy,
                "validation_invalid_count": validation_invalid_count,
                "train_scene_failure_policy": cfg.train_scene_failure_policy})
            while learner.updates < cfg.max_updates:
                budget = min(cfg.max_updates, int(online_received * cfg.updates_per_transition))
                can_update = len(online) >= cfg.learning_starts and len(demos) > 0 and learner.updates < budget
                try:
                    message = output.get(timeout=0.01 if can_update else 1)
                    latest_message = time.monotonic()
                    kind = message["kind"]
                    if kind == "error":
                        raise CollectorFailure(f"Collector {message['rank']} failed:\n{message['traceback']}")
                    if kind == "bootstrap_done":
                        boot_done.add(message["rank"])
                        if len(boot_done) == n_collectors:
                            if not len(demos):
                                raise RuntimeError("No successful train bootstrap episodes; increase bootstrap budget or supply real residual demos")
                            if not cfg.demo_archive and not resume:
                                demos.save(output_dir / "bootstrap_demos.pt", metadata=provenance)
                            counts = check_demo_coverage()
                            start_online.set()
                            write_event(events, {"kind": "bootstrap_ready", "demo_transitions": len(demos),
                                                 "demo_task_transitions": counts})
                    elif kind == "episode":
                        transitions = message.pop("transitions")
                        if message["split"] != "train":
                            raise ValueError("Refusing non-train transitions")
                        if message["bootstrap"]:
                            if message["success"]:
                                demos.add_episode(transitions, success=True)
                        else:
                            online.add_episode(transitions, success=message["success"])
                            online_received += len(transitions)
                        completed_episodes += 1
                        write_event(events, {**message, "online_received": online_received,
                            "online_size": len(online), "demo_size": len(demos), "updates": learner.updates})
                    elif kind == "reset_failed":
                        if (cfg.train_scene_failure_policy != "skip_sampling_exhausted"
                                or message.get("split") != "train"
                                or message.get("failure_kind") != "scene_sampling_exhausted"):
                            raise CollectorFailure("Unexpected reset failure classification from collector")
                        # No transition exists: neither replay nor success-rate denominator changes.
                        write_event(events, {**message, "updates": learner.updates,
                            "online_received": online_received, "online_size": len(online)})
                except queue.Empty:
                    pass
                failed = [(proc.name, proc.exitcode) for proc in workers if proc.exitcode is not None]
                if failed:
                    raise CollectorFailure(f"Collector exited unexpectedly: {failed}")
                if time.monotonic() - latest_message > cfg.idle_timeout:
                    raise CollectorFailure("No collector messages before idle_timeout; inspect collector logs")
                budget = min(cfg.max_updates, int(online_received * cfg.updates_per_transition))
                if len(online) >= cfg.learning_starts and len(demos) and start_online.is_set():
                    for _ in range(min(32, max(0, budget - learner.updates))):
                        metrics = learner.update(mix.sample(cfg.batch_size), mix.sample_success(cfg.batch_size))
                        if learner.updates % cfg.log_every == 0:
                            write_event(events, {"kind": "update", **metrics, "online_received": online_received})
                        if learner.updates % cfg.sync_every == 0:
                            broadcast_weights(weight_queues, learner.actor, learner.updates)
                        validating = cfg.validation_every and learner.updates % cfg.validation_every == 0
                        if learner.updates % cfg.checkpoint_every == 0 and not validating:
                            save()
                        if validating:
                            actor_path = save_actor()
                            write_event(events, {"kind": "validation_start", "updates": learner.updates,
                                                 "checkpoint": str(actor_path)})
                            try:
                                score, validation_dir = validate_actor(cfg, actor_path, learner.updates)
                            except ValidationEvaluationError as exc:
                                if (cfg.validation_scene_failure_policy != "continue_sampling_exhausted"
                                        or not exc.scene_sampling_exhausted):
                                    raise ValidationFailure(f"Residual validation failed: {exc}") from exc
                                # The evaluator still failed strictly. There is no score,
                                # no replay, and no change to best/guard history.
                                validation_invalid_count += 1
                                checkpoint_path = save()
                                write_event(events, {"kind": "validation_invalid", "updates": learner.updates,
                                    "valid": False, "macro_success_rate": None, "action": "continue_training",
                                    "output_dir": exc.output_dir, "validation_error": exc.details,
                                    "validation_invalid_count": validation_invalid_count,
                                    "validation_monitor": validation_monitor.state_dict(),
                                    "checkpoint": str(checkpoint_path), "checkpoint_saved": True})
                                latest_message = time.monotonic()
                                continue
                            except Exception as exc:
                                raise ValidationFailure(f"Residual validation failed: {exc}") from exc
                            record = validation_monitor.observe(score, learner.updates,
                                minimum=cfg.validation_min_success_rate, max_drop=cfg.validation_max_drop,
                                patience=cfg.validation_patience)
                            record["output_dir"] = validation_dir
                            write_event(events, record)
                            if record["improved"]:
                                atomic_save({"schema": "simplememvla-residual-policy-v1", "config": asdict(cfg),
                                    "provenance": provenance, "sac_config": asdict(sac_cfg),
                                    "actor": actor_weights(learner.actor), "updates": learner.updates,
                                    "validation": record}, checkpoint_dir / "best_actor.pt")
                            # Commit validation counters so a resumed run retains its guard history.
                            save()
                            latest_message = time.monotonic()
                            if record["stop"]:
                                raise ValidationFailure(f"Residual validation below threshold or regressed "
                                    f"for {record['bad_streak']} consecutive checks; inspect {validation_dir}")
            if last_saved_update != learner.updates:
                save()
            write_event(events, {"kind": "complete", "updates": learner.updates,
                "online_received": online_received, "validation_invalid_count": validation_invalid_count,
                "validation_monitor": validation_monitor.state_dict()})
    except CollectorFailure as exc:
        # These errors are observed outside learner.update(): optimizer state is
        # at a completed update. Never replace the last periodic checkpoint, and
        # never treat an environment error as a failed-task training sample.
        stop.set()
        start_online.set()
        failure = {"reason": str(exc), "traceback": traceback.format_exc(),
                   "updates": learner.updates, "online_received": online_received,
                   "unfinished_collector_episodes_saved": False}
        if isinstance(exc, ValidationFailure) and getattr(exc.__cause__, "details", None) is not None:
            failure["validation_error"] = exc.__cause__.details
        event = {"kind": "validation_stop" if isinstance(exc, ValidationFailure) else "collector_failure",
                 **failure, "checkpoint_saved": False}
        try:
            path = save("failure.pt", failure=failure)
            event.update(checkpoint_saved=True, checkpoint=str(path))
        except Exception as save_error:
            event["checkpoint_error"] = f"{type(save_error).__name__}: {save_error}"
        try:
            with (output_dir / "metrics.jsonl").open("a", buffering=1) as events:
                write_event(events, event)
        except Exception:
            # Reporting must not mask the original collector failure either.
            print(json.dumps(event, ensure_ascii=False), flush=True)
        raise
    finally:
        stop.set()
        start_online.set()
        # A simulator reset can be long. Give cooperative workers one shared grace period.
        deadline = time.monotonic() + 10
        for proc in workers:
            proc.join(timeout=max(0, deadline - time.monotonic()))
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        for channel in [output, *weight_queues]:
            channel.cancel_join_thread()
            channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/residual_rl/robomme_sac.json"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--gpus", help="Physical GPU IDs, learner first, comma-separated")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--resume", help="Trusted local full learner/replay checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Read-only config, metadata, dependency checks")
    args = parser.parse_args()
    cfg = load_config(args.config, checkpoint=args.checkpoint, output_dir=args.output_dir,
        tasks=args.tasks, gpus=args.gpus.split(",") if args.gpus else None,
        checkpoint_dir=args.checkpoint_dir, max_updates=args.max_updates)
    report = preflight(cfg)
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return
    if report["missing"]:
        raise RuntimeError("Preflight failed; resolve missing dependencies/checkpoint above")
    run(cfg, args.resume)


if __name__ == "__main__":
    main()
