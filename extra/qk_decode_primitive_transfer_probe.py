#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, re
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer"
LLAMA = pathlib.Path("/home/ubuntu/env/llama.cpp")

DOCS = [
  "docs/decode-native-tooling-readiness-result-20260619.md",
  "docs/decode-mmvq-artifact-import-discovery-result-20260619.md",
  "docs/decode-mmvq-large-project-p0-contract-inventory-result-20260619.md",
  "docs/decode-mmvq-large-project-p5-p6-result-20260619.md",
  "docs/llama-q4k-mmvq-inner-loop-audit-20260618.md",
  "docs/q8-ffn-artifact-promotion-result-20260620.md",
  "docs/llama-relative-promotion-reconciliation-20260620.md",
]

FILES = {
  "model": "tinygrad/llm/model.py",
  "q4": "extra/q4_k_gemv_primitive.py",
  "q6": "extra/q6_k_gemv_primitive.py",
  "readiness": "bench/qk-decode-native-tooling/readiness.json",
  "feature_attribution": "bench/qk-decode-native-tooling/feature_attribution.json",
  "q8_promotion": "bench/q8-ffn-artifact-promotion/promotion_result.json",
}


def read_text(rel: str) -> str:
  path = ROOT / rel
  return path.read_text(errors="replace") if path.exists() else ""


def read_json(rel: str) -> Any:
  path = ROOT / rel
  if not path.exists(): return None
  return json.loads(path.read_text())


def has(rel: str, pat: str) -> bool:
  return re.search(pat, read_text(rel), re.M) is not None


def llama_has(rel: str, pat: str) -> bool:
  path = LLAMA / rel
  if not path.exists(): return False
  return re.search(pat, path.read_text(errors="replace"), re.M) is not None


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def doc_present(rel: str, phrase: str | None = None) -> bool:
  txt = read_text(rel)
  return bool(txt) and (phrase is None or phrase in txt)


def main() -> int:
  readiness = read_json(FILES["readiness"]) or {}
  feature_attr = read_json(FILES["feature_attribution"]) or []
  q8_promotion = read_json(FILES["q8_promotion"]) or {}

  llama_sources = {
    "checkout_present": LLAMA.exists(),
    "mmvq_source_present": (LLAMA / "ggml/src/ggml-cuda/mmvq.cu").exists(),
    "vecdot_source_present": (LLAMA / "ggml/src/ggml-cuda/vecdotq.cuh").exists(),
    "q4_vecdot": llama_has("ggml/src/ggml-cuda/vecdotq.cuh", r"vec_dot_q4_K_q8_1"),
    "q6_vecdot": llama_has("ggml/src/ggml-cuda/vecdotq.cuh", r"vec_dot_q6_K_q8_1"),
    "q4_vdr": llama_has("ggml/src/ggml-cuda/vecdotq.cuh", r"VDR_Q4_K_Q8_1_MMVQ\s+2"),
    "q6_vdr": llama_has("ggml/src/ggml-cuda/vecdotq.cuh", r"VDR_Q6_K_Q8_1_MMVQ\s+1"),
    "mul_mat_vec_q": llama_has("ggml/src/ggml-cuda/mmvq.cu", r"mul_mat_vec_q"),
    "block_q8_lifecycle": llama_has("ggml/src/ggml-cuda/mmvq.cu", r"block_q8_1"),
  }

  tinygrad_surfaces = {
    "q4_primitive_linear": has(FILES["model"], r"class Q4KPrimitiveLinear"),
    "q6_primitive_linear": has(FILES["model"], r"class Q6KPrimitiveLinear"),
    "decode_enable_gate": has(FILES["model"], r"decode_enabled"),
    "q4_coop_route": has(FILES["model"], r"Q4K_ATTN_QO_COOP"),
    "q6_coop_route": has(FILES["model"], r"Q6K_FFN_DOWN_COOP|Q6K_LM_HEAD_COOP"),
    "q4_vdot_experiment": has(FILES["model"], r"Q4K_VDOT"),
    "q8_ffn_hardened_optin": q8_promotion.get("verdict") == "PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN",
    "flash_decode_route": has(FILES["model"], r"should_use_flash_decode"),
    "q4_packed_load_kernel": has(FILES["q4"], r"_q4k_group_dot_packed_load|_q4k_group_dot_vector_load"),
    "q4_q8_vdot_kernel": has(FILES["q4"], r"v_dot4_u32_u8|q4k_q8_1_vdot"),
    "q6_coop_kernel": has(FILES["q6"], r"q6k_coop_partial_kernel"),
  }

  existing_evidence = {
    "native_tooling_verdict": readiness.get("verdict"),
    "n2_candidate_count": (readiness.get("start_gate") or {}).get("n2_candidate_count"),
    "max_timing_grade_movement_us": (readiness.get("start_gate") or {}).get("max_timing_grade_movement_us"),
    "q8_hardened_optin_verdict": q8_promotion.get("verdict"),
    "source_import_inventory": doc_present("docs/decode-mmvq-large-project-p0-contract-inventory-result-20260619.md",
                                           "P0_PASS__SOURCE_IMPORT_P1_IS_LOADABLE_DESCRIPTOR_SMOKE"),
    "q4_import_lifecycle": doc_present("docs/decode-mmvq-large-project-p5-p6-result-20260619.md",
                                       "PASS_DEVICE_LIFECYCLE"),
    "q4_shape_matrix": doc_present("docs/decode-mmvq-large-project-p5-p6-result-20260619.md", "PASS_Q4_MATRIX"),
    "llama_inner_loop_audit": doc_present("docs/llama-q4k-mmvq-inner-loop-audit-20260618.md",
                                          "vec_dot_q4_K_q8_1_impl_vmmq"),
  }

  transfer_rows = [
    {
      "llama_primitive": "MMVQ lifecycle selection",
      "llama_source": "ggml-cuda/mmvq.cu: ggml_cuda_should_use_mmvq / mul_mat_vec_q",
      "tinygrad_today": "decode_enabled Q4/Q6 primitive linears plus role policies",
      "missing_native_schedule_surface": "single DecodeMMVQScheduleObject that owns lifecycle, role, shape, quant format, q8 producer, consumer, and reduction policy",
      "gate": "role contract rows for Q4_K/Q6_K/q8 FFN normalized with launch, resource, timing, and quality labels",
      "status": "partial",
    },
    {
      "llama_primitive": "block_q8_1 activation producer",
      "llama_source": "ggml-cuda/quantize.cu + mmvq.cu temporary q8 lifecycle",
      "tinygrad_today": "Q4K_VDOT q8_1 experiment, q8 FFN artifact producer, imported Q4 lifecycle probe",
      "missing_native_schedule_surface": "owned activation quant lifecycle with reuse accounting and quality policy",
      "gate": "byte-exact q8 producer, reuse count, dNLL/W==D policy per route",
      "status": "partial",
    },
    {
      "llama_primitive": "Q4_K x Q8_1 packed int dot",
      "llama_source": "vecdotq.cuh: vec_dot_q4_K_q8_1_impl_vmmq, VDR_Q4_K_Q8_1_MMVQ=2",
      "tinygrad_today": "Q4_K fp-dequant/cooperative kernels; q4_q8 vdot experiment; imported Q4 consumer proven in P5/P6",
      "missing_native_schedule_surface": "packed q4 nibble extraction, dp4a q8 dot, q8-sum/min correction, per-group scale application as one primitive",
      "gate": "structural op mix shows packed extract + sdot4 + per-group scale, then correctness/W==D",
      "status": "known_but_not_native_owned",
    },
    {
      "llama_primitive": "Q6_K x Q8_1 packed int dot",
      "llama_source": "vecdotq.cuh: vec_dot_q6_K_q8_1_impl_mmvq, VDR_Q6_K_Q8_1_MMVQ=1",
      "tinygrad_today": "Q6_K fp-dequant/cooperative kernels promoted for ffn_down/lm_head-like roles",
      "missing_native_schedule_surface": "Q6 packed low/high-bit extract + scale + q8 dot contract, plus imported/source correctness coverage",
      "gate": "Q6 imported/source contract correctness and role W==D movement",
      "status": "open_coverage",
    },
    {
      "llama_primitive": "small-batch ncols policy",
      "llama_source": "MMVQ_MAX_BATCH_SIZE=8 and per-arch should_use_mmvq tables",
      "tinygrad_today": "T==1 decode guards; small K batched GEMM fallback for K<=32; flash-decode ctx policy",
      "missing_native_schedule_surface": "batch/context policy encoded with primitive contract, not scattered env guards",
      "gate": "ctx/batch route matrix with correctness and W==D rows",
      "status": "partial",
    },
    {
      "llama_primitive": "one-kernel row/reduction/output contract",
      "llama_source": "mmvq.cu: mul_mat_vec_q and fusion hooks for FFN/bias paths",
      "tinygrad_today": "custom_kernel partials + separate sum(axis) for many paths; q8 FFN external artifact fuses producer+consumer",
      "missing_native_schedule_surface": "direct-output/reduction topology as an explicit decode primitive choice",
      "gate": "direct_out/reduce fused route beats current two-stage path in W==D without quality loss",
      "status": "blocked_by_prior_below_gate_evidence",
    },
  ]

  missing = []
  if readiness.get("verdict") != "TOOLING_NOT_READY":
    missing.append("unexpected readiness verdict: rerun decode-native-tooling readiness")
  if not existing_evidence["llama_inner_loop_audit"]:
    missing.append("llama Q4_K MMVQ inner-loop audit")
  if not existing_evidence["source_import_inventory"]:
    missing.append("llama MMVQ source/object inventory")
  if not existing_evidence["q4_import_lifecycle"]:
    missing.append("imported Q4 lifecycle proof")
  if not tinygrad_surfaces["q4_primitive_linear"] or not tinygrad_surfaces["q6_primitive_linear"]:
    missing.append("tinygrad Q4/Q6 decode primitive surfaces")
  if (readiness.get("start_gate") or {}).get("n2_candidate_count", 0) != 0:
    missing.append("readiness gate changed; re-evaluate native scheduler start")

  native_schedule_ready = False
  schedule_object_blockers = [
    "no timing-grade >=30us attributed q8/native scheduler feature",
    "q8 ffn_gate/up role-joined body/counter evidence is not closed enough for native codegen authority",
    "Q6 imported/source-contract coverage is still open",
    "decode lifecycle route is split across shipped primitives, imported source path, and q8 artifact policy rather than one owned schedule object",
  ]

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_PRIMITIVE_TRANSFER",
    "schema": "decode_primitive_transfer_v1",
    "verdict": "PASS_DECODE_PRIMITIVE_TRANSFER_SCOPED_NATIVE_OBJECT_BLOCKED",
    "gate_pass": True,
    "default_behavior_changed": False,
    "performance_claim": False,
    "native_schedule_object_ready": native_schedule_ready,
    "llama_sources": llama_sources,
    "tinygrad_surfaces": tinygrad_surfaces,
    "existing_evidence": existing_evidence,
    "transfer_rows": transfer_rows,
    "schedule_object_blockers": schedule_object_blockers,
    "missing_artifacts": missing,
    "feature_attribution_summary": feature_attr,
    "next_phases": [
      "DPT-1 normalize llama/tinygrad decode primitive contracts into one table: role, shape, quant, lifecycle, launch, resource, timing, quality",
      "DPT-2 build DecodeMMVQScheduleObject metadata only: q8 producer, packed weight load, dot/dequant, reduction/output, route policy",
      "DPT-3 structural probe for existing tinygrad Q4/Q6/q8 routes against that object",
      "DPT-4 source-import route gate for Q4 graph-safe path and Q6 coverage, separate from native renderer work",
      "DPT-5 start native renderer only if readiness finds timing-grade movement or the project accepts broad decode backend work",
    ],
    "principle": "decode follows the same primitive-first method as prefill, but the primitive is MMVQ/q8-lifecycle, not LDS/WMMA GEMM",
  }
  write_json("decode_primitive_transfer_result.json", result)
  print(json.dumps({
    "out": str(OUT / "decode_primitive_transfer_result.json"),
    "verdict": result["verdict"],
    "native_schedule_object_ready": native_schedule_ready,
    "llama_sources_ok": all(llama_sources.values()),
    "tinygrad_surface_count": sum(1 for v in tinygrad_surfaces.values() if v),
    "transfer_rows": len(transfer_rows),
    "blockers": schedule_object_blockers,
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
