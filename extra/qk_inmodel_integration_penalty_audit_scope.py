#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-inmodel-integration-penalty/inmodel_integration_penalty_audit_scope_result.json"


def main() -> int:
  source_rows = {
    "prefill_production": {
      "source": "docs/prefill-amd-LEARNINGS-BANKED-and-prefill-benchmark-20260620.md",
      "model": "Qwen3-8B-Q4_K_M",
      "path": "PREFILL_V2 warm full-forward",
      "tokens": 512,
      "wall_ms": 183.0,
      "tok_s": 2797,
      "effective_tflops": 45.0,
      "llama_pp512_tok_s": 3020,
      "llama_fraction": 0.93,
    },
    "prefill_isolated_kernel": {
      "source": "docs/prefill-amd-gemm-gputime-thorough-20260620.md",
      "kernel": "dependency-free RDNA3 GEMM build_gemm_lds2(BK=32,PAD=16,PLRA=1)",
      "median_tflops": 78.6,
      "tensile_median_tflops": 70.9,
      "method": "raw AMDProgram wait=True GPU signal timestamps",
    },
    "decode_lifecycle_current": {
      "source": "docs/decode-owned-q8-lifecycle-attribution-result-20260620.md",
      "mixed_lifecycle_us": 123.64,
      "target_lifecycle_us": 115.24,
      "standing": "producer timing provenance unstable; consumer near expected",
    },
  }
  questions = [
    "How much of the prefill 78.6 -> 45 TFLOPS gap is matmul coverage versus non-matmul/KV/attention?",
    "Which model roles dominate PREFILL_V2 wall time after the GEMM win is banked?",
    "Are timings measured through the same launch path and same clock policy before comparing isolated and in-model rows?",
    "Does decode q8 show the same component-to-lifecycle penalty pattern, especially producer timing instability?",
    "Which rows are production-promotable versus kernel-only research wins?",
  ]
  measurements = [
    {
      "id": "AUDIT-1",
      "name": "source-of-truth row ledger",
      "status": "ready",
      "gate": "all claims point to committed docs/probes and distinguish isolated GPU-time from in-model tok/s",
    },
    {
      "id": "AUDIT-2",
      "name": "prefill role wall-time split",
      "status": "needed",
      "gate": "PREFILL_V2 forward split into FFN GEMMs, attention, KV/update, norms/elementwise, scheduler/JIT residual",
    },
    {
      "id": "AUDIT-3",
      "name": "matmul coverage / Amdahl ledger",
      "status": "needed",
      "gate": "quantify maximum tok/s movement possible from further GEMM-only work",
    },
    {
      "id": "AUDIT-4",
      "name": "timing-method parity",
      "status": "needed",
      "gate": "isolated kernels use raw AMDProgram wait=True; in-model rows record warm JIT replay wall and optional GPU sum",
    },
    {
      "id": "AUDIT-5",
      "name": "decode lifecycle cross-check",
      "status": "needed",
      "gate": "decode producer/consumer rows get the same launch-path, batch-isolate, and lifecycle attribution treatment",
    },
  ]
  result = {
    "date": "2026-06-20",
    "phase": "INMODEL_INTEGRATION_PENALTY_AUDIT_SCOPE",
    "schema": "inmodel_integration_penalty_audit_scope_v1",
    "verdict": "PASS_INMODEL_INTEGRATION_PENALTY_AUDIT_SCOPED",
    "gate_pass": True,
    "default_behavior_changed": False,
    "performance_claim": False,
    "source_rows": source_rows,
    "topline": {
      "prefill_isolated_to_inmodel_tflops_ratio": round(source_rows["prefill_production"]["effective_tflops"] / source_rows["prefill_isolated_kernel"]["median_tflops"], 3),
      "prefill_kernel_win_over_tensile": round(source_rows["prefill_isolated_kernel"]["median_tflops"] / source_rows["prefill_isolated_kernel"]["tensile_median_tflops"], 3),
      "prefill_production_vs_llama": source_rows["prefill_production"]["llama_fraction"],
    },
    "questions": questions,
    "measurements": measurements,
    "kill_conditions": [
      "do not start another prefill GEMM microkernel unless AUDIT-3 shows material end-to-end headroom",
      "do not promote decode q8 from isolated component speed; require lifecycle/in-model proof",
      "do not compare rows across different launch paths without an explicit host/GPU separation",
    ],
    "next_doc": "docs/inmodel-integration-penalty-audit-scope-20260620.md",
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "topline": result["topline"],
    "measurements": measurements,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
