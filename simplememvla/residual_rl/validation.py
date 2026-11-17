"""Periodic residual-only held-out checks, outside learner updates and replay."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import subprocess
import sys


class ValidationEvaluationError(RuntimeError):
    def __init__(self, output_dir, details):
        self.output_dir, self.details = str(output_dir), details
        reset = details.get("reset_failure")
        reset = reset if isinstance(reset, dict) else {}
        super().__init__(f"Residual validation incomplete in {output_dir}: "
                         f"{details.get('error_type', 'unknown error')}; "
                         f"reset={reset.get('spec')}, failure_kind={reset.get('failure_kind')}")

    @property
    def scene_sampling_exhausted(self):
        reset = self.details.get("reset_failure")
        reset = reset if isinstance(reset, dict) else {}
        return (self.details.get("split") == "val" and self.details.get("mode") == "residual"
                and self.details.get("error_type") == "RoboMMEResetError"
                and reset.get("split") == "val"
                and reset.get("failure_kind") == "scene_sampling_exhausted")


@dataclass
class ValidationMonitor:
    best_score: float | None = None
    best_updates: int | None = None
    bad_streak: int = 0

    def observe(self, score, updates, *, minimum, max_drop, patience):
        if not 0 <= score <= 1:
            raise ValueError("Validation success rate must be in [0, 1]")
        improved = self.best_score is None or score > self.best_score
        bad = score < minimum or (self.best_score is not None and score < self.best_score - max_drop)
        self.bad_streak = self.bad_streak + 1 if bad else 0
        if improved:
            self.best_score, self.best_updates = score, updates
        return {"kind": "validation", "updates": updates, "macro_success_rate": score,
                "improved": improved, "passed": not bad, "stop": self.bad_streak >= patience,
                **asdict(self)}

    def state_dict(self):
        return asdict(self)


def validate_actor(cfg, checkpoint, updates):
    # Learner waits; collectors retain their original async queues/backpressure.
    # GPU 0 normally hosts only the small learner, and is reused by one frozen
    # VLA evaluator. No second baseline run and no val transitions in replay.
    output_dir = Path(cfg.output_dir) / "validation" / f"actor_{updates:08d}"
    if output_dir.exists():
        # A resume can revisit a checkpoint; preserve every previous attempt.
        attempt = 1
        while output_dir.with_name(output_dir.name + f"_retry{attempt}").exists():
            attempt += 1
        output_dir = output_dir.with_name(output_dir.name + f"_retry{attempt}")
    command = [sys.executable, "-m", "simplememvla.residual_rl.evaluate",
        "--residual-checkpoint", str(checkpoint), "--split", "val", "--mode", "residual",
        "--gpus", cfg.gpus[0], "--group-size", str(cfg.validation_group_size),
        "--episodes-per-task", str(cfg.validation_episodes_per_task),
        "--episode-selection", "difficulty_balanced", "--seed", str(cfg.validation_seed),
        "--output-dir", str(output_dir)]
    try:
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[2])
    except subprocess.CalledProcessError as exc:
        details = {"error_type": "CalledProcessError", "reason": str(exc)}
        try:
            candidate = json.loads((output_dir / "failure.json").read_text())
            if (isinstance(candidate, dict) and candidate.get("actor_updates") == updates and candidate.get("kind") == "evaluation_failed"
                    and candidate.get("split") == "val" and candidate.get("mode") == "residual"):
                details = candidate
        except (OSError, ValueError, TypeError):
            pass
        raise ValidationEvaluationError(output_dir, details) from exc
    result = json.loads((output_dir / "results.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    if manifest["actor_updates"] != updates:
        raise ValueError("Validation loaded the wrong actor checkpoint")
    rows = result["episodes"]
    expected = len(cfg.tasks) * cfg.validation_episodes_per_task
    if (manifest["evaluation"]["split"] != "val" or manifest["evaluation"]["mode"] != "residual"
            or len(rows) != expected
            or len({(row["task"], row["episode"]) for row in rows}) != expected
            or set(row["task"] for row in rows) != set(cfg.tasks)
            or any(sum(row["task"] == task for row in rows) != cfg.validation_episodes_per_task for task in cfg.tasks)
            or any(row["mode"] != "residual" for row in rows)):
        raise ValueError("Validation must contain a complete, unique residual-only val result")
    if cfg.control_version == 2 and not any(row.get("active_fraction", 0) > 0 for row in rows):
        raise ValueError("No validation control step enabled the residual; inspect phase-gate thresholds")
    return result["summary"]["residual"]["macro_success_rate"], str(output_dir)
