#!/bin/bash
# The Nemotron-H GEMM scan, one command: derived strategy space (BoltBeam) vs promoted routes vs vLLM vs roofline,
# BoltBeam-owned NCU side-by-side with the research queue (unexplained residuals, unknown techniques), and the
# workload lifecycle comparison when BoltBeam has it.
#
#   docs/nemotron-vllm-parity/bench/scan.sh OUTDIR                       # CPU only: tables from committed evidence
#   MEASURE=1 NCU=1 docs/nemotron-vllm-parity/bench/scan.sh OUTDIR       # + GPU: derived-space climb and fresh ncu
#
# Env: ROLES (default all eight), MS (tokens, default 8..8192), BOLTBEAM_ROOT (default ../BoltBeam),
# GPU steps run under ~/scratchpad/bin/gpu-run time (exclusive lock; keep other agents in mind).
set -euo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd); B=$R/docs/nemotron-vllm-parity/bench; A=$B/audit
BB=${BOLTBEAM_ROOT:-$(cd "$R/.." && pwd)/BoltBeam}; export BOLTBEAM_ROOT=$BB
OUT=$(mkdir -p "${1:?usage: scan.sh OUTDIR}" && cd "$1" && pwd)
ROLES=${ROLES:-ssm_in,ssm_out,attn_q,attn_kv,attn_o,ffn_up,ffn_down,output}
MS=${MS:-8,16,32,64,128,512,1024,2048,4096,8192}
GPU="$HOME/scratchpad/bin/gpu-run time"
bb() { (cd "$BB" && python3 -m boltbeam.cli "$@"); }

# 1. the compiler's lowering facts (what tinygrad can emit), for BoltBeam's derivation
python3 "$R/extra/llm_research/gemm_lowering_facts.py" --out "$OUT/lowering.json"

# 2. (GPU) climb the derived space per shape from the promoted route + the model's top seeds; GEMM rows are 2 per token
MEASURED=()
if [[ "${MEASURE:-0}" == 1 ]]; then
  ROWS=$(python3 -c "print(','.join(str(2*int(m)) for m in '$MS'.split(',') if int(m) > 8))")
  (cd "$R" && DEV=NV $GPU python3 extra/llm_research/prefill/dense_bf16_geometry_search.py --scan --roles "$ROLES" --rows "$ROWS" --out "$OUT/scan.jsonl")
  MEASURED=(--measured "$OUT/scan.jsonl")
fi

# 3. per-shape strategy table: derived candidate (model + measured) vs promoted route vs vLLM vs roofline, limiter
bb gemm-scan --lowering "$OUT/lowering.json" --reference "$A/results/vllm_gemm.json" \
  --selection "$R/extra/llm_research/prefill/dense_bf16_sm120_selection.json" --ours "$A/results/ours.json" \
  --roles "$ROLES" "${MEASURED[@]}" --out "$OUT/gemm_scan.json" --markdown "$OUT/gemm_scan.md"

# 4. NCU side-by-side (BoltBeam collector; fresh with NCU=1, else the audit's committed captures) + research queue
if [[ "${NCU:-0}" == 1 ]]; then
  $GPU "$A/oncu.sh" "$OUT/oncu" "$ROLES" "$MS"
  $GPU "$A/vncu.sh" "$OUT/vncu" "$ROLES" "$MS"
  COUNTERS=(--counters "$OUT/oncu.json" --counters "$OUT/vncu.json")
else
  E=$BB/evidence/ncu/nemotron_kernel_audit_20260926
  COUNTERS=(--counters "$E/counters_ours.json" --counters "$E/counters_reference.json" --census "$E/census.json")
fi
bb ncu-audit "${COUNTERS[@]}" --dims-from "$A/results/vllm_gemm.json" --lowering "$OUT/lowering.json" \
  --out "$OUT/ncu_audit.json" --markdown "$OUT/ncu_audit.md"

# 5. workload lifecycle (kernel sum vs wall, launches) when this BoltBeam has the command
if bb lifecycle-compare --help >/dev/null 2>&1; then bb lifecycle-compare --out "$OUT/lifecycle.json" --markdown "$OUT/lifecycle.md" || true; fi

{ echo "# Nemotron-H GEMM scan ($(date -I), tinygrad $(git -C "$R" rev-parse --short HEAD), BoltBeam $(git -C "$BB" rev-parse --short HEAD))"
  echo; echo "## Strategy per shape"; echo; cat "$OUT/gemm_scan.md"
  echo; echo "## NCU side by side"; echo; cat "$OUT/ncu_audit.md"
  [[ -f "$OUT/lifecycle.md" ]] && { echo; echo "## Lifecycle"; echo; cat "$OUT/lifecycle.md"; }
} > "$OUT/report.md"
echo "wrote $OUT/report.md"
