#!/usr/bin/env python3
"""S10.5 machine-search candidate serializer for ffn_gate_up.

This module only writes metadata. It does not import the runtime route and does
not lower or emit a kernel body.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

if __package__ in (None, ""):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from extra.qk.prefill.dbuf_s10_lds_spec_exporter import export_s10_lds_spec
from extra.qk.prefill_schedule_spec import describe_prefill_schedule
from extra.qk.wmma_lds_spec import LDS2_OWNERSHIP_CLASSIFICATION, extract_wmma_lds_spec, wmma_lds_slot_identity_proof


SCHEMA = "prefill-s10.5-machine-search-candidate.v1"
DEFAULT_OUTPUT = pathlib.Path("bench/prefill-s10_5-machine-search/ffn-gate-up-candidates.json")
DEFAULT_SEARCH_OUTPUT = pathlib.Path("bench/prefill-s10_5-machine-search/search-report.json")
DEFAULT_REPORT_OUTPUT = pathlib.Path("bench/prefill-s10_5-machine-search/final-report.json")
ROLE = "ffn_gate_up"
SHAPE = {"M": 512, "N": 12288, "K": 4096}
CLASSIFICATION = LDS2_OWNERSHIP_CLASSIFICATION
AUTHORITY_ROUTE = "prefill_pipe_role_selective_generated"
PP512_MIN_TOK_S = 4000.0

WAIT_VARIANTS: tuple[dict[str, Any], ...] = (
  {
    "candidate_id": "wait-default",
    "label": "default",
    "env_overrides": {},
    "wait_policy": {"vm_after_coop_load": 0, "lgkm_after_coop_store": 0, "lgkm_after_frag_load": 0},
    "prior_authority_artifact": "bench/prefill-whole-synced/s9-repeat-default-a.json",
  },
  {
    "candidate_id": "wait-lgkm-coop-store-2",
    "label": "LGKM_COOP_STORE=2",
    "env_overrides": {"PREFILL_LDS2_WAIT_LGKM_COOP_STORE": "2"},
    "wait_policy": {"vm_after_coop_load": 0, "lgkm_after_coop_store": 2, "lgkm_after_frag_load": 0},
    "prior_authority_artifact": "bench/prefill-whole-synced/s9-repeat-coop-store2-a.json",
  },
  {
    "candidate_id": "wait-lgkm-frag-load-2",
    "label": "LGKM_FRAG_LOAD=2",
    "env_overrides": {"PREFILL_LDS2_WAIT_LGKM_FRAG_LOAD": "2"},
    "wait_policy": {"vm_after_coop_load": 0, "lgkm_after_coop_store": 0, "lgkm_after_frag_load": 2},
    "prior_authority_artifact": "bench/prefill-whole-synced/s9-repeat-frag-load2-a.json",
  },
)


def _unsupported_record(schedule_json: dict[str, Any], errors: list[str]) -> dict[str, Any]:
  return {
    "schema": SCHEMA,
    "role": ROLE,
    "shape": dict(SHAPE),
    "schedule_spec": schedule_json,
    "lds_spec": None,
    "selected_backend_atom": {
      "name": "asm_backend_atom",
      "classification": CLASSIFICATION,
      "runtime_emission_changed": False,
    },
    "classification": CLASSIFICATION,
    "wait": None,
    "cadence": None,
    "lifecycle": None,
    "dbuf_epoch_primitive": None,
    "legality_errors": errors,
    "slot_identity_proof": {
      "schema": "wmma-lds-slot-identity-proof.v1",
      "active_buffers": 2,
      "ok": False,
      "errors": list(errors),
    },
    "dbuf_checker_metadata": {
      "ok": False,
      "active_buffers": 2,
      "errors": list(errors),
      "events": [],
    },
    "promotion_status": "blocked",
    "promotion_reason": "failed to extract legal WMMALDSSpec",
    "pure_generated": False,
  }


def _prior_authority_summary(path: str | None) -> dict[str, Any] | None:
  if path is None: return None
  p = pathlib.Path(path)
  try:
    data = json.loads(p.read_text())
  except FileNotFoundError:
    return {"path": path, "status": "missing"}
  whole = data.get("whole_tok_s") or {}
  return {
    "path": path,
    "status": "ok",
    "pin_clock": data.get("pin_clock"),
    "pp512": whole.get("512"),
    "pp1024": whole.get("1024"),
    "pp2048": whole.get("2048"),
    "pp4096": whole.get("4096"),
  }


def build_ffn_gate_up_candidate(*, active_buffers: int = 2, variant: dict[str, Any] | None = None) -> dict[str, Any]:
  """Return a machine-readable S10.5 candidate record for the existing LDS atom."""
  variant = WAIT_VARIANTS[0] if variant is None else variant
  schedule = describe_prefill_schedule(SHAPE["N"], SHAPE["K"], role=ROLE)
  schedule_json = schedule.to_json()
  lds_spec = extract_wmma_lds_spec(schedule)
  if lds_spec is None:
    return _unsupported_record(schedule_json, ["extract_wmma_lds_spec returned None"])

  lds_json = lds_spec.to_json()
  legality_errors = list(lds_json.get("legality_errors", []))
  slot_identity = wmma_lds_slot_identity_proof(lds_spec, active_buffers=active_buffers)
  dbuf_metadata = export_s10_lds_spec(lds_spec, active_buffers=active_buffers)
  promotion_ok = not legality_errors and slot_identity["ok"] and active_buffers == 2
  promotion_reason = (
    "candidate is serializable and DBUF slot identity is proven; cadence remains metadata/not promoted"
    if promotion_ok else
    "candidate is blocked by legality or slot identity errors"
  )

  return {
    "schema": SCHEMA,
    "role": ROLE,
    "shape": dict(SHAPE),
    "candidate_id": variant.get("candidate_id", "wait-default"),
    "candidate_label": variant.get("label", "default"),
    "env_overrides": dict(variant.get("env_overrides", {})),
    "search_knobs": {
      "class": "s9_safe_wait_policy",
      "wait_policy": dict(variant.get("wait_policy", {})),
      "runtime_emission_changed": False,
    },
    "schedule_spec": schedule_json,
    "lds_spec": lds_json,
    "selected_backend_atom": {
      "name": lds_spec.lifecycle.backend_atom,
      "classification": lds_spec.ownership_classification(),
      "source": "extra.qk.wmma_lds_spec.WMMALDSSpec.lifecycle.backend_atom",
      "runtime_emission_changed": False,
      "runtime_emission": "unchanged_existing_backend_atom",
    },
    "classification": lds_spec.ownership_classification(),
    "wait": lds_spec.wait.to_json(),
    "cadence": lds_spec.cadence.to_json(),
    "lifecycle": lds_spec.lifecycle.to_json(),
    "dbuf_epoch_primitive": lds_spec.dbuf_epoch_primitive.to_json(),
    "legality_errors": legality_errors,
    "slot_identity_proof": slot_identity,
    "dbuf_checker_metadata": dbuf_metadata,
    "promotion_status": "candidate" if promotion_ok else "blocked",
    "promotion_reason": promotion_reason,
    "prior_authority": _prior_authority_summary(variant.get("prior_authority_artifact")),
    "pure_generated": False,
    "not_pure_generated_reason": "ffn_gate_up is spec-owned around a reusable ASM backend atom; runtime emission is unchanged",
  }


def build_search_report(*, active_buffers: int = 2) -> dict[str, Any]:
  candidates = [build_ffn_gate_up_candidate(active_buffers=active_buffers, variant=v) for v in WAIT_VARIANTS]
  viable = []
  for cand in candidates:
    prior = cand.get("prior_authority") or {}
    try: pp512 = float(prior.get("pp512"))
    except (TypeError, ValueError): pp512 = float("nan")
    proof_ok = (cand.get("slot_identity_proof") or {}).get("ok") is True
    viable.append({
      "candidate_id": cand["candidate_id"],
      "label": cand["candidate_label"],
      "proof_ok": proof_ok,
      "prior_pp512": prior.get("pp512"),
      "prior_pp4096": prior.get("pp4096"),
      "prior_pin_clock": prior.get("pin_clock"),
      "authority_gate_estimate_ok": pp512 >= PP512_MIN_TOK_S,
      "recommended_for_authority": proof_ok and pp512 >= PP512_MIN_TOK_S,
      "env_overrides": cand["env_overrides"],
    })
  recommended = [v for v in viable if v["recommended_for_authority"]]
  return {
    "schema": "prefill-s10.5-machine-search-report.v1",
    "route": AUTHORITY_ROUTE,
    "classification": CLASSIFICATION,
    "pure_generated": False,
    "search_space": "s9_safe_wait_policy_over_backend_atom",
    "pp512_min_tok_s": PP512_MIN_TOK_S,
    "candidate_count": len(candidates),
    "recommended_candidate_ids": [v["candidate_id"] for v in recommended],
    "verdict": "S10_5_SEARCH_READY_FOR_AUTHORITY"
               if recommended else "S10_5_SEARCH_BLOCKED_NO_AUTHORITY_VIABLE_CANDIDATE",
    "summary": viable,
    "candidates": candidates,
  }


def write_candidate(path: pathlib.Path = DEFAULT_OUTPUT, *, active_buffers: int = 2) -> dict[str, Any]:
  record = build_ffn_gate_up_candidate(active_buffers=active_buffers)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(record, indent=2) + "\n")
  return record


def write_search_report(path: pathlib.Path = DEFAULT_SEARCH_OUTPUT, *, active_buffers: int = 2) -> dict[str, Any]:
  report = build_search_report(active_buffers=active_buffers)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(report, indent=2) + "\n")
  return report


def build_authority_gate(authority: dict[str, Any]) -> dict[str, Any]:
  route = (authority.get("route_attribution") or {}).get("prefill_route_family")
  whole = authority.get("whole_tok_s") or {}
  pp512 = whole.get("512")
  pp4096 = whole.get("4096")
  try: pp512_f = float(pp512)
  except (TypeError, ValueError): pp512_f = float("nan")
  route_ok = route == AUTHORITY_ROUTE
  perf_ok = pp512_f >= PP512_MIN_TOK_S
  return {
    "schema": "prefill-s10.5-authority-gate.v1",
    "required_route": AUTHORITY_ROUTE,
    "selected_route": route,
    "route_ok": route_ok,
    "pp512_min_tok_s": PP512_MIN_TOK_S,
    "pp512_tok_s": pp512,
    "pp4096_tok_s": pp4096,
    "pin_clock": authority.get("pin_clock"),
    "perf_ok": perf_ok,
    "binding_gate_verdict": (authority.get("prefill_route_binding_gate") or {}).get("verdict"),
    "binding_gate_note": (
      "generic pure-route binding may fail because S10.5 intentionally classifies this path as hybrid/backend-atom"
    ),
    "ok": route_ok and perf_ok,
  }


def build_final_report(*, candidate: dict[str, Any] | None = None,
                       authority: dict[str, Any] | None = None) -> dict[str, Any]:
  candidate = build_ffn_gate_up_candidate() if candidate is None else candidate
  authority_gate = build_authority_gate(authority) if authority is not None else {
    "schema": "prefill-s10.5-authority-gate.v1",
    "ok": False,
    "status": "not_run",
    "required_route": AUTHORITY_ROUTE,
    "pp512_min_tok_s": PP512_MIN_TOK_S,
  }
  candidate_ok = (
    candidate.get("classification") == CLASSIFICATION and
    candidate.get("pure_generated") is False and
    (candidate.get("slot_identity_proof") or {}).get("ok") is True and
    candidate.get("promotion_status") == "candidate"
  )
  ready = candidate_ok and authority_gate.get("ok") is True
  return {
    "schema": "prefill-s10.5-machine-search-final-report.v1",
    "verdict": "S10_5_HYBRID_SEARCH_OWNED_BACKEND_ATOM_READY" if ready
               else "S10_5_HYBRID_SEARCH_BLOCKED_WITH_EXACT_REASON",
    "classification": CLASSIFICATION,
    "pure_generated": False,
    "full_fine_tuned_hand_kernel": False,
    "candidate_ok": candidate_ok,
    "authority_gate": authority_gate,
    "candidate": candidate,
  }


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
  ap.add_argument("--search-output", type=pathlib.Path, default=DEFAULT_SEARCH_OUTPUT)
  ap.add_argument("--report-output", type=pathlib.Path, default=DEFAULT_REPORT_OUTPUT)
  ap.add_argument("--authority-artifact", type=pathlib.Path)
  ap.add_argument("--active-buffers", type=int, default=2)
  ap.add_argument("--search", action="store_true", help="write the S10.5 wait-policy search report")
  ap.add_argument("--json", action="store_true", help="print the candidate JSON to stdout")
  args = ap.parse_args(argv)

  record = write_search_report(args.search_output, active_buffers=args.active_buffers) if args.search else \
           write_candidate(args.output, active_buffers=args.active_buffers)
  if args.authority_artifact is not None:
    authority = json.loads(args.authority_artifact.read_text())
    candidate = record["candidates"][0] if args.search else record
    report = build_final_report(candidate=candidate, authority=authority)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2) + "\n")
    record = report
  if args.json:
    print(json.dumps(record, indent=2))
  else:
    print(f"wrote {args.report_output if args.authority_artifact is not None else args.output}")
  return record


if __name__ == "__main__":
  main()
