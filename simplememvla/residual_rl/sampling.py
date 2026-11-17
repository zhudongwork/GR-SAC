"""Deterministic episode selection using benchmark metadata, never policy inputs."""

import json
from pathlib import Path
import random


def difficulty_balanced_specs(specs, per_task, metadata_dir, seed):
    """Round-robin difficulty strata after independent, seeded shuffling.

    Only selects existing IDs. Does not alter scenes, seeds, labels or records.
    Validation uses its own metadata directory and never supplies replay data.
    """
    if per_task < 3:
        raise ValueError("Select at least three episodes per task to cover all difficulties")
    rng = random.Random(seed)
    selected = []
    for task in dict.fromkeys(spec["task"] for spec in specs):
        records = json.loads((Path(metadata_dir) / f"record_dataset_{task}_metadata.json").read_text())["records"]
        metadata = {row["episode"]: row for row in records}
        buckets = {key: [] for key in ("easy", "medium", "hard")}
        for spec in specs:
            if spec["task"] != task:
                continue
            row = metadata[spec["episode"]]
            difficulty = row.get("difficulty")
            if difficulty not in buckets or int(row["seed"]) != int(spec["seed"]):
                raise ValueError(f"Invalid difficulty/seed metadata for {spec}")
            buckets[difficulty].append(spec)
        if any(not bucket for bucket in buckets.values()) or sum(map(len, buckets.values())) < per_task:
            raise ValueError(f"Not enough {task} episodes/difficulty coverage for {per_task} samples")
        for bucket in buckets.values():
            rng.shuffle(bucket)
        task_selected = []
        while len(task_selected) < per_task:
            for bucket in buckets.values():
                if bucket and len(task_selected) < per_task:
                    task_selected.append(bucket.pop())
        selected.extend(task_selected)
    return selected


def demo_task_counts(demos, task_order):
    """Count retained successful demo transitions by their observable task ID."""
    counts = {task: 0 for task in task_order}
    for item in demos.storage:
        onehot = item["obs"]["state"][26:26 + len(task_order)]
        if len(onehot) != len(task_order) or sum(onehot == 1) != 1 or not all((onehot == 0) | (onehot == 1)):
            raise ValueError("Demo observation has an invalid task one-hot")
        if item["success"]:
            counts[task_order[int(onehot.argmax())]] += 1
    return counts
