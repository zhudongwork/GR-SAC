#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
args=(--config "${CONFIG:-$REPO_ROOT/configs/residual_rl/robomme_sac.json}")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
[[ -z "${OUT_DIR:-}" ]] || args+=(--output-dir "$OUT_DIR")
[[ -z "${CHECKPOINT_DIR:-}" ]] || args+=(--checkpoint-dir "$CHECKPOINT_DIR")
[[ -z "${GPUS:-}" ]] || args+=(--gpus "$GPUS")
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")
if [[ -n "${TASKS:-}" ]]; then
  read -r -a task_list <<< "$TASKS"
  args+=(--tasks "${task_list[@]}")
fi
exec "${PYBIN:-python}" -m simplememvla.residual_rl.train "${args[@]}" "$@"
