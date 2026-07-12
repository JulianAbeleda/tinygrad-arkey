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
  "exact_isa_resources": "bench/prefill-pure-full-kernel/anchor-ffn-gate-up/mastery-v1/resources-isa.json",
  "exact_role_timing": "bench/prefill-pure-full-kernel/anchor-ffn-gate-up/mastery-v1/role-timing.json",
  "lane_fragment_map": "bench/prefill-pure-full-kernel/anchor-ffn-gate-up/mastery-v1/lane-fragment-map.json",
  "epoch_graph": "bench/prefill-pure-full-kernel/anchor-ffn-gate-up/mastery-v1/epoch-graph.json",
  "correctness_binding": "bench/prefill-pure-full-kernel/anchor-ffn-gate-up/mastery-v1/correctness-binding.json",
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
  exact_capture = loaded["exact_isa_resources"] or {}
  exact_timing = loaded["exact_role_timing"] or {}
  lane_map = loaded["lane_fragment_map"] or {}
  epoch_graph = loaded["epoch_graph"] or {}
  correctness_binding = loaded["correctness_binding"] or {}
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
    "exact_isa_resources": {"state": "captured" if exact_capture else "not_available",
                            "source": ARTIFACTS["exact_isa_resources"],
                            "git": exact_capture.get("git"), "binding_complete": exact_capture.get("binding_complete"),
                            "candidate_ids": [row.get("candidate_id") for row in exact_capture.get("captures", [])]},
    "exact_role_timing": {"state": "captured" if exact_timing else "not_available",
                          "source": ARTIFACTS["exact_role_timing"], "environment": exact_timing.get("environment"),
                          "complete": exact_timing.get("complete"),
                          "tflops": {row.get("regime"): (row.get("measurement") or {}).get("tflops")
                                     for row in exact_timing.get("rows", [])}},
    "lane_fragment_map": {"state": "calculated_oracle_map" if lane_map else "not_available",
                          "source": ARTIFACTS["lane_fragment_map"], "proof_class": lane_map.get("proof_class"),
                          "counts": lane_map.get("counts"), "mapping": lane_map.get("mapping")},
    "epoch_graph": {"state": "partial_structural_graph" if epoch_graph else "not_available",
                    "source": ARTIFACTS["epoch_graph"], "claims": epoch_graph.get("claims"),
                    "identity_loss": epoch_graph.get("identity_loss")},
    "correctness_binding": {"state": "captured" if correctness_binding else "not_available",
                            "source": ARTIFACTS["correctness_binding"], "passed": correctness_binding.get("passed"),
                            "binding": correctness_binding.get("binding"), "surface": correctness_binding.get("surface")},
  }
  # Existing artifacts are useful context but none closes a full pure-candidate mastery requirement by itself.
  evidence_status = {key: {"status": "missing", "requirement": description} for key, description in REQUIRED_EVIDENCE.items()}
  evidence_status["exact_workload_contract"] = {
    "status": "partial", "requirement": REQUIRED_EVIDENCE["exact_workload_contract"],
    "known": ["M/N/K", "role", "phase", "model quant label", "tensor name patterns"],
    "missing": ["runtime input/output dtypes", "strides/layouts", "dequantization boundary", "bias/activation/epilogue semantics"],
  }
  captures = {row.get("candidate_id"): row for row in exact_capture.get("captures", [])}
  pure_capture = captures.get("pure.default.m512n12288k4096") or {}
  clean_capture = (exact_capture.get("schema") == "prefill-pure-anchor-isa-resource-capture.v1" and
                   exact_capture.get("binding_complete") is True and (exact_capture.get("git") or {}).get("dirty") is False)
  pure_program = pure_capture.get("program") or {}
  pure_surface = pure_capture.get("surface") or {}
  if clean_capture and pure_surface.get("strict_pure") is True and all(pure_program.get(k) for k in
      ("program_key", "source_sha256", "binary_sha256", "isa_sha256")):
    evidence_status["generated_isa_capture"] = {
      "status": "complete", "requirement": REQUIRED_EVIDENCE["generated_isa_capture"],
      "source": ARTIFACTS["exact_isa_resources"], "candidate_id": pure_capture["candidate_id"],
    }
  pure_resources = pure_capture.get("resources") or {}
  if clean_capture and all(k in pure_resources for k in ("vgpr", "sgpr", "lds_bytes", "scratch_bytes")):
    evidence_status["measured_resource_capture"] = {
      "status": "complete", "requirement": REQUIRED_EVIDENCE["measured_resource_capture"],
      "source": ARTIFACTS["exact_isa_resources"], "authority": pure_resources.get("authority"),
    }
  timing_rows = {row.get("regime"): row for row in exact_timing.get("rows", [])}
  clean_timing = (exact_timing.get("schema") == "prefill-anchor-gemm-regime-timing.v1" and
                  exact_timing.get("complete") is True and (exact_timing.get("environment") or {}).get("git_dirty") is False)
  if clean_timing and set(timing_rows) == {"pure_scheduler", "spec_owned", "s9_oracle"} and \
      all(row.get("binding_pass") is True for row in timing_rows.values()):
    evidence_status["kernel_timing_authority"] = {
      "status": "complete", "requirement": REQUIRED_EVIDENCE["kernel_timing_authority"],
      "source": ARTIFACTS["exact_role_timing"],
      "tflops": {name: (row.get("measurement") or {}).get("tflops") for name, row in timing_rows.items()},
    }
  if evidence_status["generated_isa_capture"]["status"] == "complete" and \
      evidence_status["measured_resource_capture"]["status"] == "complete" and clean_timing:
    evidence_status["roofline_attribution"] = {
      "status": "partial", "requirement": REQUIRED_EVIDENCE["roofline_attribution"],
      "known": ["exact instruction identity", "compiled resources", "pinned kernel TFLOPS"],
      "missing": ["same-binary measured memory traffic", "same-binary achieved bandwidth and compute counters"],
    }
  if lane_map.get("schema") == "prefill-anchor-lane-fragment-evidence.v1" and \
      (lane_map.get("invariants") or {}).get("accumulator_to_c_is_bijective") is True:
    evidence_status["lane_fragment_map"] = {
      "status": "partial", "requirement": REQUIRED_EVIDENCE["lane_fragment_map"],
      "known": ["exhaustive S9 oracle cooperative-load map", "oracle WMMA fragment map", "oracle accumulator-to-C bijection"],
      "missing": ["same mapping derived from the pure candidate compiler IR/ISA", "measured LDS bank behavior"],
      "source": ARTIFACTS["lane_fragment_map"],
    }
  if epoch_graph.get("schema") == "ffn-gate-up-epoch-dependency-graph.v1" and \
      (epoch_graph.get("claims") or {}).get("structural_reaching_definitions_complete") is True:
    evidence_status["pipeline_dependency_graph"] = {
      "status": "partial", "requirement": REQUIRED_EVIDENCE["pipeline_dependency_graph"],
      "known": ["128 epochs", "two-slot structural dependency graph", "DBUF checker pass"],
      "missing": ["512 value keys", "lowered final-instruction correlation"],
      "source": ARTIFACTS["epoch_graph"],
    }
  binding = correctness_binding.get("binding") or {}
  surface = correctness_binding.get("surface") or {}
  clean_correctness = (correctness_binding.get("schema") == "prefill-pure-anchor-correctness-binding.v1" and
                       correctness_binding.get("passed") is True and
                       ((correctness_binding.get("environment") or {}).get("git") or {}).get("dirty") is False)
  if clean_correctness and all(row.get("passed") is True for row in
      ((correctness_binding.get("correctness") or {}).get("cases") or [])):
    evidence_status["full_shape_correctness"] = {
      "status": "complete", "requirement": REQUIRED_EVIDENCE["full_shape_correctness"],
      "source": ARTIFACTS["correctness_binding"], "comparison": "three adversarial full-output cases",
    }
  if clean_correctness and surface.get("strict_pure") is True and surface.get("ops_ins_count") == 0 and \
      binding.get("runtime_binary_matches_candidate") is True and binding.get("all_cases_same_binary") is True:
    evidence_status["strict_pure_runtime_binding"] = {
      "status": "complete", "requirement": REQUIRED_EVIDENCE["strict_pure_runtime_binding"],
      "source": ARTIFACTS["correctness_binding"], "binary_sha256": binding.get("binary_sha256"),
      "identity_scope": binding.get("identity_scope"),
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
