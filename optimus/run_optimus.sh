#!/usr/bin/env bash
# Launch a NatureBench run with the OPTIMUS harness. Run it inside tmux.
#   ~/run_optimus.sh <task-list> <results-name> [gpu-ids] [extra run_naturebench.py args]
# Examples:
#   ~/run_optimus.sh task-set/naturebench-25/gpu_low.txt optimus_nb25_gpu_low 0,1,2,3
#   ~/run_optimus.sh task-set/naturebench-25/cpu.txt     optimus_nb25_cpu          (no GPU, as the reference)
# Model: set OPTIMUS_MODEL to override (default anthropic/claude-opus-5.5).
set -euo pipefail

TASKS="${1:?usage: run_optimus.sh <task-list> <results-name> [gpu-ids]}"
OUT="${2:?usage: run_optimus.sh <task-list> <results-name> [gpu-ids]}"
GPUS="${3:-}"
EXTRA=("${@:4}")
NB="$HOME/NatureBench"
MODEL="${OPTIMUS_MODEL:-anthropic/claude-opus-5.5}"

if [ ! -s "$HOME/.openrouter_key" ]; then
  echo "No OpenRouter key at ~/.openrouter_key. Save it first (see vm_setup.sh output)."; exit 1
fi
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naturebench
cd "$NB"

KEY="$(cat "$HOME/.openrouter_key")"
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="$KEY"
unset ANTHROPIC_API_KEY
export JUDGE_MODEL="openai/gpt-5.5"
export JUDGE_OPENAI_API_KEY="$KEY"
export JUDGE_OPENAI_BASE_URL="https://openrouter.ai/api"

ARGS=(--tasks "$TASKS" --agent optimus --model "$MODEL" --out-dir "./results/$OUT"
      --eval-env-mapping ./eval_env_mapping.json --ensure-base-image)
if [ -n "$GPUS" ]; then
  N=$(awk -F, '{print NF}' <<<"$GPUS")
  ARGS+=(--gpu-devices "$GPUS" --max-workers "$N")
fi
# Start the evaluation services only if no earlier run on this machine started them.
if ! pgrep -f "eval_service.py" >/dev/null 2>&1; then
  ARGS+=(--start-eval-services)
fi

echo "OPTIMUS run: tasks=$TASKS out=results/$OUT model=$MODEL gpus=${GPUS:-none}"
python run_naturebench.py "${ARGS[@]}" "${EXTRA[@]}"
