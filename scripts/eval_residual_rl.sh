#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
: "${RESIDUAL_CHECKPOINT:?Set RESIDUAL_CHECKPOINT to a saved actor or full training checkpoint}"
args=(--residual-checkpoint "$RESIDUAL_CHECKPOINT" --mode "${MODE:-residual}"
      --split "${DATASET_SPLIT:-test}" --gpus "${GPUS:-0,1,2,3,4,5,6,7}"
      --group-size "${GROUP_SIZE:-10}" --episodes-per-task "${EPISODES_PER_TASK:-50}"
      --output-dir "${OUT_DIR:-$REPO_ROOT/logs/residual_rl/eval_$(date +%Y%m%d_%H%M%S)}")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
if [[ -n "${TASKS:-}" ]]; then
  read -r -a task_list <<< "$TASKS"
  args+=(--tasks "${task_list[@]}")
fi
exec "${PYBIN:-python}" -m simplememvla.residual_rl.evaluate "${args[@]}" "$@"
