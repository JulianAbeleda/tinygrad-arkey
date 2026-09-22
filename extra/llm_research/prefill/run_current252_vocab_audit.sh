#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. All three arms share one serialized GPU lease.
audit_out=${1:?evidence directory required}
audit_mode=${2:-bracket}
case "$audit_mode" in
  bracket) audit_arms=(control-a candidate-b control-c) ;;
  default) audit_arms=(candidate-default) ;;
  profile) audit_arms=(candidate-profile) ;;
  *) exit 2 ;;
esac
mkdir -p "$audit_out"
exec 9>/tmp/gpu-bench.lock
flock -n 9
export DEV=NV PYTHONPATH=. QK_PRIMITIVE=1
export NV_COMPILER_Q4_IMMA_PP512=1 NV_COMPILER_Q4_IMMA_K_PP512=1 NV_COMPILER_Q4_IMMA_QO_PP512=1
export NV_COMPILER_Q6_IMMA_PP512=1 NV_COMPILER_Q6_IMMA_PP512_ROLES=attn_v,ffn_down
export NV_COMPILER_Q4_DOWN_STREAMK=1 NV_COMPILER_Q4_GATE_STREAMK=1
export NV_COMPILER_Q4_STREAMK_FRAGMENT_LOAD_TO_USE=0
export HCQ_NUM_COMPUTE=2 HCQ_NV_READY_PLACEMENT=0
export HCQ_NV_MULTI_QUEUE_CUT_POLICY=docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/combined-flash-direct-deps-cut-v2.json
git rev-parse HEAD > "$audit_out/commit-$audit_mode.txt"
sha256sum tinygrad/llm/model.py extra/llm_research/prefill/nv_compiler_q4k_gkqo_model_arm.py \
  extra/llm_research/prefill/run_current252_vocab_audit.sh > "$audit_out/source-$audit_mode.sha256"
for audit_arm in "${audit_arms[@]}"; do
  export NV_COMPILER_Q6_VOCAB_PP512=0
  if [[ "$audit_arm" == candidate-b ]]; then export NV_COMPILER_Q6_VOCAB_PP512=1; fi
  if [[ "$audit_mode" != bracket ]]; then unset NV_COMPILER_Q6_VOCAB_PP512; fi
  audit_replay=20; audit_rounds=9; audit_warmups=10
  if [[ "$audit_mode" == profile ]]; then audit_replay=1; audit_rounds=3; audit_warmups=3; fi
  nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv > "$audit_out/$audit_arm-before.csv"
  .venv/bin/python extra/llm_research/prefill/nv_compiler_q4k_gkqo_model_arm.py \
    --arm candidate --q4-v --q6-v --q6-down --q4-down-streamk --gate-streamk \
    --warmups "$audit_warmups" --rounds "$audit_rounds" --replay-cycles "$audit_replay" \
    --out "$audit_out/$audit_arm.json" --logits-npz "$audit_out/$audit_arm.npz" \
    > "$audit_out/$audit_arm.log" 2>&1
  nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv > "$audit_out/$audit_arm-after.csv"
done
