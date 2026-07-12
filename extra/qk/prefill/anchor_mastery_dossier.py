#!/usr/bin/env python3
"""Build the fixed 8B ffn_gate_up anchor mastery dossier from existing truth sources."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from extra.qk import route_manifest
from extra.qk.model_profiles import profile_by_id
from extra.qk.prefill_schedule_spec import describe_prefill_schedule
from extra.qk.pure_kernel_surface_audit import route_surface_row

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = "prefill-pure-anchor-mastery-dossier.v1"
PROFILE_ID, ROLE = "qwen3_8b_q4k_m_gfx1100", "ffn_gate_up"
PURE_ROUTE = "prefill_v2_scheduler_matmul_default"
STRUCTURAL_ORACLE_ROUTE = "prefill_wmma_pipe_lds_dbuf_primitive_generated"
DEFAULT_OUTPUT = ROOT / "bench/prefill-pure-machine-search/anchor-mastery-dossier.json"

ARTIFACTS = {
  "lds_primitive": "bench/prefill-pipe-mvp/ffn-gate-up-lds-primitive.json",
  "route_trace": "bench/prefill-s10-lds2-ownership/route-trace.json",
  "whole_route_authority": "bench/prefill-whole-synced/s10-composed-current-authority.json",
  "roofline_audit": "bench/prefill-lds2-s9/roofline-audit.json",
}

REQUIRED_EVIDENCE = {
  "exact_workload_contract": "Exact M/N/K, tensor patterns, quantization boundary, input/output dtype, strides, and epilogue semantics.",
  "lane_fragment_map": "Executable lane-to-global/LDS/WMMA-fragment/accumulator/output ownership map for the pure candidate.",
  "pipeline_dependency_graph": "Machine-readable load, LDS stage, wait, barrier, WMMA, and store dependency graph.",
  "generated_isa_capture": "Generated candidate ISA tied by hash to compiler IR and the executed binary.",
  "measured_resource_capture": "Compiler/runtime VGPR, SGPR, LDS, scratch/spill, waves, and occupancy evidence for that binary.",
  "full_shape_correctness": "Full M512 N12288 K4096 numerical comparison with declared tolerances and reference provenance.",
  "kernel_timing_authority": "Clock-pinned kernel-only samples for the exact pure candidate, excluding compile and whole-model aggregation.",
  "strict_pure_runtime_binding": "Runtime proof that the exact executable used no Ops.INS, ASM atom, embedded binary, or non-pure fallback.",
  "roofline_attribution": "Roofline inputs and diagnosis bound to the same pure executable and kernel-only timing.",
}


def _load_artifact(root: pathlib.Path, relative: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
  path = root / relative
  if not path.is_file(): return None, {"path": relative, "present": False, "error": "missing"}
  try: data = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError) as exc:
    return None, {"path": relative, "present": True, "error": f"invalid_json: {exc}"}
  if not isinstance(data, dict): return None, {"path": relative, "present": True, "error": "root_not_object"}
  return data, {"path": relative, "present": True, "schema": data.get("schema"), "verdict": data.get("verdict")}


def _route_summary(route_id: str) -> dict[str, Any]:
  route = route_manifest.ROUTES[route_id]
  surface = route_surface_row(route_id)
  return {
    "route_id": route_id,
    "status": route["status"],
    "provenance": route_manifest.route_provenance(route_id),
    "surface_class": surface["surface_class"],
    "strict_pure": surface["strict_pure"],
    "env": dict(route["env"]),
    "shape_guards": list(route["shape_guards"]),
    "authority_gate": route["authority_gate"],
    "source_paths": list(surface["writer_files"]),
  }


def build_dossier(*, root: pathlib.Path = ROOT) -> dict[str, Any]:
  profile = profile_by_id(PROFILE_ID)
  shape = profile.role_shape(ROLE)
  schedule = describe_prefill_schedule(shape.N, shape.K, role=shape.role)
  loaded, artifact_index = {}, {}
  for name, relative in ARTIFACTS.items():
    loaded[name], artifact_index[name] = _load_artifact(root, relative)

  lds = loaded["lds_primitive"] or {}
  correctness = lds.get("correctness") if isinstance(lds.get("correctness"), dict) else {}
  resources = lds.get("resource_counters") if isinstance(lds.get("resource_counters"), dict) else {}
  known_evidence = {
    "exact_profile_shape": {"state": "registry_fact", "source": "extra/qk/model_profiles.py", "value": shape.to_json()},
    "existing_schedule_spec": {"state": "spec_derived", "source": "extra/qk/prefill_schedule_spec.py", "value": schedule.to_json()},
    "existing_route_ownership": {"state": "registry_fact", "source": "extra/qk/route_manifest.py + extra/qk/pure_kernel_surface_audit.py",
                                 "value": {"pure_baseline": _route_summary(PURE_ROUTE),
                                           "structural_oracle": _route_summary(STRUCTURAL_ORACLE_ROUTE)}},
    "sample_correctness": {"state": correctness.get("status", "not_available"), "source": ARTIFACTS["lds_primitive"],
                           "scope": "sampled structural-oracle route; not full-shape pure-candidate proof",
                           "max_abs_error": correctness.get("max_abs_error"), "max_rel_error": correctness.get("max_rel_error")},
    "spec_resource_estimates": {"state": "spec_derived" if resources else "not_available", "source": ARTIFACTS["lds_primitive"],
                                "scope": "structural-oracle spec; not measured pure-candidate occupancy",
                                "lds_bytes": resources.get("lds_bytes"), "accum_vgprs": resources.get("accum_vgprs"),
                                "coop_temp_vgprs": resources.get("coop_temp_vgprs"), "spills": resources.get("spills")},
  }
  # Existing artifacts are useful context but none closes a full pure-candidate mastery requirement by itself.
  evidence_status = {key: {"status": "missing", "requirement": description} for key, description in REQUIRED_EVIDENCE.items()}
  evidence_status["exact_workload_contract"] = {
    "status": "partial", "requirement": REQUIRED_EVIDENCE["exact_workload_contract"],
    "known": ["M/N/K", "role", "phase", "model quant label", "tensor name patterns"],
    "missing": ["runtime input/output dtypes", "strides/layouts", "dequantization boundary", "bias/activation/epilogue semantics"],
  }
  missing = [{"id": key, "reason": row["requirement"], "status": row["status"]}
             for key, row in evidence_status.items() if row["status"] != "complete"]
  return {
    "schema": SCHEMA,
    "status": "BLOCKED_ON_MISSING_EVIDENCE",
    "mastery_complete": False,
    "anchor": {"profile_id": profile.id, "device_profile": profile.device_profile, "role": shape.role,
               "phase": shape.phase, "shape": {"M": shape.M, "N": shape.N, "K": shape.K}, "quant": shape.quant},
    "intent": {"candidate_origin": "strict pure Tinygrad scheduler route", "oracle_use": "evidence only; never candidate substrate",
               "non_duplication": "references existing profile/spec/manifest/audit/artifact sources; does not benchmark or emit kernels"},
    "known_evidence": known_evidence,
    "artifact_index": artifact_index,
    "evidence_status": evidence_status,
    "missing_evidence": missing,
    "blocking_gate": "Do not generalize or launch full machine search until every evidence_status entry is complete and bound to one pure executable.",
  }


def validate_dossier(report: Any) -> dict[str, Any]:
  if not isinstance(report, dict) or report.get("schema") != SCHEMA: raise ValueError("invalid anchor mastery dossier schema")
  if report.get("mastery_complete") is not False or report.get("status") != "BLOCKED_ON_MISSING_EVIDENCE":
    raise ValueError("v1 dossier must fail closed while required evidence is incomplete")
  anchor = report.get("anchor", {})
  if anchor.get("profile_id") != PROFILE_ID or anchor.get("role") != ROLE or anchor.get("shape") != {"M": 512, "N": 12288, "K": 4096}:
    raise ValueError("dossier is not bound to the fixed 8B ffn_gate_up anchor")
  statuses = report.get("evidence_status", {})
  if set(statuses) != set(REQUIRED_EVIDENCE): raise ValueError("dossier evidence requirements drifted")
  if not report.get("missing_evidence"): raise ValueError("incomplete dossier must name missing evidence")
  return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args(argv)
  report = validate_dossier(build_dossier())
  output = args.output if args.output.is_absolute() else ROOT / args.output
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=2) + "\n")
  if args.json: print(json.dumps(report, indent=2))
  else: print(output)
  return report


if __name__ == "__main__": main()
