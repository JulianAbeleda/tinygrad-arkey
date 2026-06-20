#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

from tinygrad.renderer.amd.schedule import (
  DecodeMMVQRoleContract, DecodeMMVQStage, DecodeMMVQResourceGate, DecodeMMVQEvidence,
  DecodeMMVQScheduleObject, decode_mmvq_schedule_object_summary,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer"
TRANSFER = "bench/qk-decode-primitive-transfer/decode_primitive_transfer_result.json"


def read_json(rel: str) -> dict[str, Any]:
  path = ROOT / rel
  if not path.exists(): raise FileNotFoundError(f"required artifact missing: {rel}")
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def stages(imported_or_artifact: bool, q8: bool) -> tuple[DecodeMMVQStage, ...]:
  auth = "runtime" if imported_or_artifact else "source"
  return (
    DecodeMMVQStage(0, "activation_prepare", "contiguous_or_fp16_vec", "x", "tinygrad decode x_vec path", "runtime"),
    DecodeMMVQStage(1, "activation_q8_producer", "block_q8_1" if q8 else "fp16_or_optional_q8_1", "x",
                    "llama block_q8_1 lifecycle / tinygrad q8-vdot or q8 artifact where enabled", "source"),
    DecodeMMVQStage(2, "activation_reuse", "per_token_cache_or_single_consumer", "x",
                    "reuse explicit in q8 artifact/import path; limited in current tinygrad primitives", "static"),
    DecodeMMVQStage(3, "packed_weight_load", "packed_u32_or_halfword_load", "weight",
                    "Q4_K/Q6_K packed storage words/halfs", auth),
    DecodeMMVQStage(4, "packed_extract", "nibble_or_low_high_extract", "weight",
                    "llama packed extract; tinygrad fp paths still scalar/dequant-heavy", "source"),
    DecodeMMVQStage(5, "dot_or_dequant_dot", "sdot4_or_fp_dequant_dot", "x,weight",
                    "llama sdot4 target; tinygrad Q4/Q6 current routes are mixed fp/custom", auth),
    DecodeMMVQStage(6, "scale_apply", "per_group_scale_or_fp_affine", "scale",
                    "per-group in llama; per-weight fp-affine remains a native gap", "source"),
    DecodeMMVQStage(7, "partial_reduce", "partial_sum_or_direct_out", None,
                    "tinygrad partials.sum or direct artifact route", "runtime"),
    DecodeMMVQStage(8, "output_store", "float_output", "out",
                    "role output vector", "runtime"),
    DecodeMMVQStage(9, "route_policy", "role_flag_policy", None,
                    "decode_enabled, Q4/Q6 role policy, q8 hardened opt-in", "quality"),
  )


def build_objects(transfer: dict[str, Any]) -> list[DecodeMMVQScheduleObject]:
  src_ok = all((transfer.get("llama_sources") or {}).values())
  tiny = transfer.get("tinygrad_surfaces") or {}
  ev = transfer.get("existing_evidence") or {}

  blocked_common = (
    "native renderer lowering is not built",
    "no timing-grade >=30us attributed q8/native scheduler feature",
    "decode primitive lifecycle is not yet unified into production routing",
  )

  return [
    DecodeMMVQScheduleObject(
      DecodeMMVQRoleContract("ffn_gate/up", "Q4_K", 12288, 4096, 1, "q8_1", "row/group", "direct_or_partial", "imported_q4_proven_not_default"),
      stages(imported_or_artifact=True, q8=True),
      DecodeMMVQResourceGate(1, True, True, False, False),
      DecodeMMVQEvidence(src_ok, bool(tiny.get("q4_primitive_linear")), bool(ev.get("q4_shape_matrix")), False, True, False),
      blocked_common + ("graph-safe Q4 route not promoted",)),
    DecodeMMVQScheduleObject(
      DecodeMMVQRoleContract("attn_q/o", "Q4_K", 4096, 4096, 1, "fp16", "coop_lane4", "partials_sum", "promoted_default"),
      stages(imported_or_artifact=False, q8=False),
      DecodeMMVQResourceGate(1, False, False, True, False),
      DecodeMMVQEvidence(src_ok, bool(tiny.get("q4_coop_route")), True, False, False, True),
      blocked_common + ("native packed-int replacement not started",)),
    DecodeMMVQScheduleObject(
      DecodeMMVQRoleContract("ffn_down/lm_head", "Q6_K", 4096, 12288, 1, "fp16", "coop_pos", "partials_sum", "promoted_default_for_selected_roles"),
      stages(imported_or_artifact=False, q8=False),
      DecodeMMVQResourceGate(1, False, False, True, False),
      DecodeMMVQEvidence(src_ok, bool(tiny.get("q6_primitive_linear")), True, False, False, False),
      blocked_common + ("Q6 imported/source-contract parity still open",)),
    DecodeMMVQScheduleObject(
      DecodeMMVQRoleContract("ffn_gate/up_q8_artifact", "Q8_FFN_ARTIFACT", 12288, 4096, 1, "q8_artifact", "artifact_fused", "direct_out", "hardened_opt_in"),
      stages(imported_or_artifact=True, q8=True),
      DecodeMMVQResourceGate(1, True, True, False, False),
      DecodeMMVQEvidence(src_ok, bool(tiny.get("q8_ffn_hardened_optin")), bool(ev.get("q8_hardened_optin_verdict")), False, True, False),
      blocked_common + ("external lossy artifact remains default-off",)),
  ]


def main() -> int:
  transfer = read_json(TRANSFER)
  objects = build_objects(transfer)
  summaries = [decode_mmvq_schedule_object_summary(o) for o in objects]
  gates = [o.structural_gate() for o in objects]
  gate_pass = all(g["passed"] for g in gates)
  native_owned_count = sum(1 for o in objects if o.resource_gate.native_renderer_owned)
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_MMVQ_SCHEDULE_OBJECT_STRUCTURAL",
    "schema": "decode_mmvq_schedule_object_structural_v1",
    "verdict": "PASS_DECODE_MMVQ_SCHEDULE_OBJECT_STRUCTURAL_NATIVE_BLOCKED" if gate_pass else "BLOCKED_DECODE_MMVQ_SCHEDULE_OBJECT_STRUCTURAL",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "native_renderer_owned_count": native_owned_count,
    "native_renderer_ready": False,
    "objects": [o.to_dict() for o in objects],
    "summaries": summaries,
    "failed_checks": [s for s in summaries if s["failed_checks"]],
    "input_artifacts": [TRANSFER],
    "next_action": "normalize role contracts, then decide source-import graph route vs native renderer; no BEAM/search until native decode lowering exists",
  }
  write_json("decode_mmvq_schedule_object_result.json", result)
  print(json.dumps({
    "out": str(OUT / "decode_mmvq_schedule_object_result.json"),
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "object_count": len(objects),
    "native_renderer_owned_count": native_owned_count,
    "summaries": summaries,
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
