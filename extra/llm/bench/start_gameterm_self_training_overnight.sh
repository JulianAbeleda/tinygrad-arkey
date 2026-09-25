#!/usr/bin/env bash
set -euo pipefail

repo="${TINYGRAD_SELF_TRAIN_REPO:-/home/ubuntu/tinygrad-self-training}"
model="${TINYGRAD_SELF_TRAIN_MODEL:-/home/ubuntu/storage/models/Qwen3-8B-Q4_K_M.gguf}"
harness="${GAMETERM_HARNESS:-/home/ubuntu/gameterm_beta/target/debug/gameterm-harness}"
out="${1:-/home/ubuntu/storage/self-training/overnight}"
pid_file="$out/launcher.pid"

mkdir -p "$out"
if [[ -f "$pid_file" ]]; then
  old_pid="$(cat "$pid_file")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "self-training controller is already running as PID $old_pid"
    exit 1
  fi
fi

cd "$repo"
nohup setsid env DEV=NV python3 -m extra.llm.bench.gameterm_self_training_loop \
  --repo "$repo" \
  --model "$model" \
  --input "$repo/test/fixtures/llm/qwen_output_lora_overnight.jsonl" \
  --harness "$harness" \
  --out "$out" \
  --system-prompt 'Reply with exactly one uppercase action label: READ, WATCH, OPEN, or BLOCK. No punctuation. /no_think' \
  --server-max-tokens 1 \
  --max-hours "${SELF_TRAIN_MAX_HOURS:-8}" \
  --max-rounds "${SELF_TRAIN_MAX_ROUNDS:-12}" \
  --steps-per-round "${SELF_TRAIN_STEPS_PER_ROUND:-96}" \
  --init-adapter "$repo/docs/task_workflow/evidence/gameterm-qwen-self-training-20260907/adapter" \
  >"$out/controller.stdout" 2>"$out/controller.stderr" < /dev/null &
pid=$!
echo "$pid" > "$pid_file"
echo "started GameTerm self-training controller as PID $pid"
echo "ledger: $out/controller-ledger.json"
