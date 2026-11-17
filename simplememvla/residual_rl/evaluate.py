"""Held-out deterministic residual evaluation and zero-offset paired control."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from .compat import normalize_legacy_config, normalize_legacy_sac_config
from .config import TrainConfig, checkpoint_identity, file_sha256, metadata_specs


class EvaluationWorkerError(RuntimeError):
    def __init__(self, message):
        self.details = message
        super().__init__(message["traceback"])


def read_policy(path):
    """Only load artifacts from a trusted local training run (torch pickle)."""
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema") == "simplememvla-residual-policy-v1":
        result = {key: saved[key] for key in ("config", "provenance", "sac_config", "actor", "updates")}
        result["config"] = normalize_legacy_config(result["config"])
        result["sac_config"] = normalize_legacy_sac_config(result["sac_config"])
        return result
    if saved.get("schema") != "simplememvla-residual-training-v1":
        raise ValueError("Not a residual training/policy checkpoint")
    learner = saved["learner"]
    return {"config": normalize_legacy_config(saved["config"]), "provenance": saved["provenance"],
            "sac_config": normalize_legacy_sac_config(learner["config"]), "updates": learner["updates"],
            "actor": {key: tensor.numpy().copy() for key, tensor in learner["actor"].items()}}


def evaluation_worker(rank, gpu, artifact, specs, args_dict, output):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cfg = TrainConfig(**artifact["config"])
    group = None
    try:
        import numpy as np
        import torch
        from .networks import SquashedGaussianActor, actor_kwargs
        from .robomme import RoboMMEResidualGroup
        torch.set_num_threads(cfg.cpu_threads)
        torch.cuda.set_device(0)
        seed = args_dict["seed"] + rank * 1009
        torch.manual_seed(seed)
        actor = SquashedGaussianActor(**actor_kwargs(artifact["sac_config"])).to("cuda:0")
        actor.load_state_dict({key: torch.as_tensor(value) for key, value in artifact["actor"].items()})
        actor.eval().requires_grad_(False)
        group = RoboMMEResidualGroup(cfg.policy_args(seed),
            metadata_dir=Path(cfg.metadata_root) / args_dict["split"], split=args_dict["split"],
            num_envs=args_dict["group_size"], delta=cfg.delta, gain=cfg.gain,
            offset_bound=cfg.offset_bound, image_size=cfg.image_size,
            enabled_tasks=cfg.enabled_tasks, start_step=cfg.start_step, **cfg.controller_kwargs())
        modes = ["zero", "residual"] if args_dict["mode"] == "paired" else [args_dict["mode"]]
        for task in dict.fromkeys(spec["task"] for spec in specs):
            task_specs = [spec for spec in specs if spec["task"] == task]
            for index in range(0, len(task_specs), args_dict["group_size"]):
                selected = task_specs[index:index + args_dict["group_size"]]
                # Repeat the same base sampler start for the two arms of each pair.
                base_cpu_rng = group._base_cpu_rng.clone()
                base_cuda_rng = group._base_cuda_rng.clone()
                for mode in modes:
                    group._base_cpu_rng = base_cpu_rng.clone()
                    group._base_cuda_rng = base_cuda_rng.clone()
                    obs = group.reset(selected)
                    steps, action_sum, offset_sum = {}, {}, {}
                    active_sum, offset_max, bound_sum = {}, {}, {}
                    while obs:
                        slots = sorted(obs)
                        if mode == "zero":
                            actions = np.zeros((len(slots), 7), dtype=np.float32)
                        else:
                            batch = {key: torch.as_tensor(np.stack([obs[i][key] for i in slots]), device="cuda:0")
                                     for key in ("images", "state")}
                            with torch.no_grad():
                                actions = (actor.mean_action(batch) * batch["state"][:, 24:25]).cpu().numpy()
                        for transition in group.step(dict(zip(slots, actions))):
                            slot = transition["slot"]
                            steps[slot] = steps.get(slot, 0) + 1
                            action_sum[slot] = action_sum.get(slot, 0) + float(np.abs(transition["action"]).mean())
                            offset_sum[slot] = offset_sum.get(slot, 0) + float(np.abs(transition["obs"]["state"][16:23]).mean())
                            applied_offset = np.abs(transition["obs"]["state"][16:23])
                            active_sum[slot] = active_sum.get(slot, 0) + float(transition["obs"]["state"][24])
                            offset_max[slot] = max(offset_max.get(slot, 0), float(applied_offset.max()))
                            bound_sum[slot] = bound_sum.get(slot, 0) + float((applied_offset >= .95 * cfg.offset_bound).mean())
                            if transition["terminated"] or transition["truncated"]:
                                output.put({"kind": "episode", "mode": mode, **selected[slot],
                                    "worker": rank, "base_rng_seed": seed, "steps": steps[slot],
                                    "success": bool(transition["success"]), "status": transition["status"],
                                    "timeout_source": transition.get("timeout_source"),
                                    "mean_abs_residual": action_sum[slot] / steps[slot],
                                    "mean_abs_offset_rad": offset_sum[slot] / steps[slot],
                                    "max_abs_offset_rad": offset_max[slot],
                                    "active_fraction": active_sum[slot] / steps[slot],
                                    "offset_bound_fraction": bound_sum[slot] / steps[slot]})
                                obs.pop(slot)
                            else:
                                obs[slot] = transition["next_obs"]
        output.put({"kind": "done", "worker": rank})
    except BaseException as exc:
        output.put({"kind": "error", "worker": rank, "traceback": traceback.format_exc(),
                    "error_type": type(exc).__name__, "reset_failure": getattr(exc, "failure", None)})
        raise
    finally:
        if group is not None:
            group.close()


def summarize(rows):
    summary = {}
    for mode in dict.fromkeys(row["mode"] for row in rows):
        task_results = {}
        for task in dict.fromkeys(row["task"] for row in rows if row["mode"] == mode):
            task_rows = [row for row in rows if row["mode"] == mode and row["task"] == task]
            successes = sum(row["success"] for row in task_rows)
            task_results[task] = {"successes": successes, "episodes": len(task_rows),
                                  "success_rate": successes / len(task_rows)}
        summary[mode] = {"tasks": task_results,
                         "macro_success_rate": sum(row["success_rate"] for row in task_results.values()) / len(task_results)}
        mode_rows = [row for row in rows if row["mode"] == mode]
        if all("active_fraction" in row for row in mode_rows):
            steps = sum(row["steps"] for row in mode_rows)
            summary[mode]["control_diagnostics"] = {
                key: sum(row[key] * row["steps"] for row in mode_rows) / steps
                for key in ("active_fraction", "mean_abs_residual", "mean_abs_offset_rad", "offset_bound_fraction")}
            summary[mode]["control_diagnostics"]["max_abs_offset_rad"] = max(row["max_abs_offset_rad"] for row in mode_rows)
    if "zero" in summary and "residual" in summary:
        summary["delta_percentage_points"] = 100 * (summary["residual"]["macro_success_rate"]
                                                     - summary["zero"]["macro_success_rate"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--checkpoint", help="Relocate the identical frozen base checkpoint")
    parser.add_argument("--metadata-root")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--mode", choices=("paired", "zero", "residual"), default="residual")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--group-size", type=int, default=17)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--episode-selection", choices=("first", "difficulty_balanced"), default="first")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.group_size <= 0 or args.episodes_per_task <= 0:
        parser.error("group-size and episodes-per-task must be positive")
    gpus = args.gpus.split(",")
    if not all(gpus) or len(set(gpus)) != len(gpus):
        parser.error("gpus must list distinct nonempty IDs")
    artifact = read_policy(args.residual_checkpoint)
    cfg = TrainConfig(**artifact["config"]).validate()
    if cfg.protocol() != artifact["provenance"]["protocol"]:
        raise ValueError("Actor configuration and saved control protocol disagree")
    if args.checkpoint:
        cfg.checkpoint = str(Path(args.checkpoint).resolve())
    if args.metadata_root:
        cfg.metadata_root = str(Path(args.metadata_root).resolve())
    if checkpoint_identity(cfg.checkpoint) != artifact["provenance"]["base_checkpoint"]:
        raise ValueError("Frozen base checkpoint differs from training")
    tasks = args.tasks or cfg.tasks
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(cfg.tasks):
        raise ValueError("Evaluate a unique subset of the training task list")
    artifact["config"] = asdict(cfg)
    specs = [spec for spec in metadata_specs(cfg, args.split) if spec["task"] in tasks]
    if args.episode_selection == "difficulty_balanced":
        from .sampling import difficulty_balanced_specs
        selected = difficulty_balanced_specs(specs, args.episodes_per_task,
                    Path(cfg.metadata_root) / args.split, args.seed)
    else:
        selected = bootstrap_specs_for_eval(specs, args.episodes_per_task)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("Use a fresh evaluation output directory")
    manifest = {"evaluation": vars(args), "training_protocol": artifact["provenance"],
        "actor_updates": artifact["updates"], "episodes": selected,
        "metadata_hashes": {task: file_sha256(Path(cfg.metadata_root) / args.split /
            f"record_dataset_{task}_metadata.json") for task in tasks},
        "matching": "same base, split, groups, sampler start; differing termination can change later RNG assignment"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    context, processes = mp.get_context("spawn"), []
    messages = context.Queue()
    rows, done = [], set()
    n_workers = min(len(gpus), len(selected))
    latest = time.monotonic()
    try:
        for rank in range(n_workers):
            process = context.Process(target=evaluation_worker,
                args=(rank, gpus[rank], artifact, selected[rank::n_workers], vars(args), messages))
            process.start()
            processes.append(process)
        with (output_dir / "episodes.jsonl").open("w", buffering=1) as stream:
            while len(done) < n_workers:
                try:
                    message = messages.get(timeout=1)
                    latest = time.monotonic()
                    if message["kind"] == "error":
                        raise EvaluationWorkerError(message)
                    if message["kind"] == "done":
                        done.add(message["worker"])
                    else:
                        rows.append(message)
                        line = json.dumps(message)
                        stream.write(line + "\n")
                        print(line, flush=True)
                except queue.Empty:
                    pass
                if any(process.exitcode not in (None, 0) for process in processes):
                    raise RuntimeError("Evaluation worker failed; inspect its traceback")
                if time.monotonic() - latest > cfg.idle_timeout:
                    raise TimeoutError("Evaluation workers produced no output before idle_timeout")
        expected = len(selected) * (2 if args.mode == "paired" else 1)
        identities = {(row["mode"], row["task"], row["episode"]) for row in rows}
        if len(rows) != expected or len(identities) != expected:
            raise RuntimeError("Incomplete or duplicate evaluation results")
        result = {"summary": summarize(rows), "episodes": rows}
        (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["summary"], indent=2), flush=True)
    except BaseException as exc:
        # Failed/incomplete evaluations never get results.json or a success rate.
        # Preserve a typed cause for the training supervisor; do not infer reset
        # categories by matching substrings in a human-readable traceback.
        failure = {"kind": "evaluation_failed", "actor_updates": artifact["updates"],
                   "split": args.split, "mode": args.mode, "completed_episodes": len(rows),
                   "expected_episodes": len(selected) * (2 if args.mode == "paired" else 1),
                   "error_type": type(exc).__name__, "reason": str(exc),
                   "traceback": traceback.format_exc(), **getattr(exc, "details", {})}
        failure["kind"] = "evaluation_failed"
        try:
            temporary = output_dir / "failure.json.tmp"
            temporary.write_text(json.dumps(failure, indent=2) + "\n")
            temporary.replace(output_dir / "failure.json")
        except Exception as save_error:
            print(f"Could not save evaluation failure details: {save_error}", flush=True)
        raise
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        messages.close()


def bootstrap_specs_for_eval(specs, count):
    from .train import bootstrap_specs
    return bootstrap_specs(specs, count)


if __name__ == "__main__":
    main()
