#!/usr/bin/env python3
"""hybrid machine-search candidate serializer for ffn_gate_up.

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
from extra.qk.model_profiles import prefill_role_shapes, profile_by_id
from extra.qk.wmma_lds_spec import LDS2_OWNERSHIP_CLASSIFICATION, extract_wmma_lds_spec, wmma_lds_slot_identity_proof


SCHEMA = "prefill-hybrid-machine-search-candidate.v1"
DEFAULT_OUTPUT = pathlib.Path("bench/prefill-hybrid-machine-search/ffn-gate-up-candidates.json")
DEFAULT_SEARCH_OUTPUT = pathlib.Path("bench/prefill-hybrid-machine-search/search-report.json")
DEFAULT_REPORT_OUTPUT = pathlib.Path("bench/prefill-hybrid-machine-search/final-report.json")
ROLE = "ffn_gate_up"
DEFAULT_PROFILE = "qwen3_8b_q4k_m_gfx1100"
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


def _profile_role_shapes(profile: str) -> tuple[Any, ...]:
  try: return prefill_role_shapes(profile_by_id(profile))
  except KeyError as e: raise ValueError(f"unsupported profile {profile!r}") from e


def _shape_value(row: Any, key: str) -> Any:
  if isinstance(row, dict): return row[key]
  return getattr(row, key)


def select_prefill_role_shape(*, profile: str = DEFAULT_PROFILE, role: str = ROLE) -> Any:
  matches = [row for row in _profile_role_shapes(profile)
             if _shape_value(row, "role") == role and _shape_value(row, "phase") == "prefill"]
  if not matches:
    raise ValueError(f"profile {profile!r} has no prefill role shape for role {role!r}")
  return matches[0]


def _shape_dict(row: Any) -> dict[str, int]:
  return {k: int(_shape_value(row, k)) for k in ("M", "N", "K")}


def _unsupported_record(role: str, shape: dict[str, int], schedule_json: dict[str, Any],
                        errors: list[str], variant: dict[str, Any]) -> dict[str, Any]:
  return {
    "schema": SCHEMA,
    "role": role,
    "shape": dict(shape),
    "candidate_id": variant.get("candidate_id", "wait-default"),
    "candidate_label": variant.get("label", "default"),
    "env_overrides": dict(variant.get("env_overrides", {})),
    "search_knobs": {
      "class": "s9_safe_wait_policy",
      "wait_policy": dict(variant.get("wait_policy", {})),
      "runtime_emission_changed": False,
    },
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


def build_role_shape_candidate(role_shape: Any, *, active_buffers: int = 2,
                               variant: dict[str, Any] | None = None) -> dict[str, Any]:
  """Return a machine-readable hybrid machine-search candidate record for the existing LDS atom."""
  variant = WAIT_VARIANTS[0] if variant is None else variant
  role = str(_shape_value(role_shape, "role"))
  shape = _shape_dict(role_shape)
  schedule = describe_prefill_schedule(shape["N"], shape["K"], role=role)
  schedule_json = schedule.to_json()
  lds_spec = extract_wmma_lds_spec(schedule)
  if lds_spec is None:
    route_family = schedule_json.get("route_family")
    return _unsupported_record(
      role, shape, schedule_json,
      [f"backend atom unsupported for role/shape: schedule route_family={route_family!r}; "
       "extract_wmma_lds_spec returned None"],
      variant,
    )

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
    "role": role,
    "shape": dict(shape),
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
    "not_pure_generated_reason": f"{role} is spec-owned around a reusable ASM backend atom; runtime emission is unchanged",
  }


def build_ffn_gate_up_candidate(*, active_buffers: int = 2, variant: dict[str, Any] | None = None,
                                profile: str = DEFAULT_PROFILE) -> dict[str, Any]:
  return build_role_shape_candidate(select_prefill_role_shape(profile=profile, role=ROLE),
                                    active_buffers=active_buffers, variant=variant)


def build_search_report(*, active_buffers: int = 2, profile: str = DEFAULT_PROFILE, role: str = ROLE) -> dict[str, Any]:
  role_shape = select_prefill_role_shape(profile=profile, role=role)
  candidates = [build_role_shape_candidate(role_shape, active_buffers=active_buffers, variant=v) for v in WAIT_VARIANTS]
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
    "schema": "prefill-hybrid-machine-search-report.v1",
    "route": AUTHORITY_ROUTE,
    "classification": CLASSIFICATION,
    "pure_generated": False,
    "search_space": "s9_safe_wait_policy_over_backend_atom",
    "pp512_min_tok_s": PP512_MIN_TOK_S,
    "candidate_count": len(candidates),
    "recommended_candidate_ids": [v["candidate_id"] for v in recommended],
    "verdict": "HYBRID_MACHINE_SEARCH_READY_FOR_AUTHORITY"
               if recommended else "HYBRID_MACHINE_SEARCH_BLOCKED_NO_AUTHORITY_VIABLE_CANDIDATE",
    "summary": viable,
    "candidates": candidates,
  }


def write_candidate(path: pathlib.Path = DEFAULT_OUTPUT, *, active_buffers: int = 2,
                    profile: str = DEFAULT_PROFILE, role: str = ROLE) -> dict[str, Any]:
  record = build_role_shape_candidate(select_prefill_role_shape(profile=profile, role=role),
                                      active_buffers=active_buffers)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(record, indent=2) + "\n")
  return record


def write_search_report(path: pathlib.Path = DEFAULT_SEARCH_OUTPUT, *, active_buffers: int = 2,
                        profile: str = DEFAULT_PROFILE, role: str = ROLE) -> dict[str, Any]:
  report = build_search_report(active_buffers=active_buffers, profile=profile, role=role)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(report, indent=2) + "\n")
  return report


def _as_float(v: Any) -> float:
  try: return float(v)
  except (TypeError, ValueError): return float("nan")


def measurement_regime(authority: dict[str, Any]) -> dict[str, Any]:
  """Name the measurement regime of a whole-synced authority artifact (F2).

  The `prefill-whole-synced-authority.v1` schema is reused for THREE incomparable
  regimes distinguished by route provenance: the pure tinygrad-generated scheduler
  path (~1.6k tok/s pp512), the S10 compiler-primitive spec-owned hybrid path
  (~1.5k), and the external hand-written kernel reference (~4.4k). Only the pure
  generated regime is authoritative for the generated-route promotion question;
  the hand-external number is an aspirational ceiling, NOT this route's speed.
  Cross-regime comparison is forbidden by the comparator check below.
  """
  ra = authority.get("route_attribution") or {}
  prov = ra.get("prefill_route_provenance")
  regime_id = {
    "tinygrad_scheduler_generated": "generated_pure",
    "machine_authored_generated": "generated_pure",
    "compiler_primitive_spec_owned": "spec_owned_hybrid",
    "external_handwritten_kernel": "hand_external_reference",
    "rollback_oracle": "hand_external_reference",
  }.get(prov, "unknown")
  return {
    "regime_id": regime_id,
    "provenance": prov,
    "route_pure": ra.get("prefill_route_pure"),
    "route_rolled_back": ra.get("prefill_route_rolled_back"),
    "mode": authority.get("mode"),
    "logits_only": authority.get("logits_only"),
    # only the pure generated regime may be cited as the generated route's promotion authority
    "authoritative_for_generated_promotion": regime_id == "generated_pure",
  }


def _route_classification(route: str | None) -> dict[str, Any]:
  """Authoritative shipped/research classification, sourced from route_manifest (read-only)."""
  try:
    from extra.qk import route_manifest as rm
    r = rm.route(route)
    prov, status = r.get("provenance"), r.get("status")
    return {
      "route_id": route, "status": status, "provenance": prov,
      "purity_status": rm.derive_purity_status(status, prov),
      "final_default_allowed": prov in rm.FINAL_DEFAULT_PROVENANCE,
      "source": "extra/qk/route_manifest.py",
    }
  except Exception as e:
    return {"route_id": route, "status": None, "provenance": None, "purity_status": "unknown",
            "final_default_allowed": False, "error": f"{type(e).__name__}: {e}"}


def _comparator_delta(authority: dict[str, Any], comparator: dict[str, Any] | None,
                      cand_regime: dict[str, Any]) -> dict[str, Any]:
  """Same-regime comparator vs the current default (F1b). A bare floor is NOT a comparator."""
  if comparator is None:
    return {"status": "MISSING", "ok": False,
            "note": "no same-regime current-default comparator supplied; the bare pp512>=floor is a "
                    "diagnostic, not a comparator delta (F1b) -> authority refused"}
  comp_regime = measurement_regime(comparator)
  cand_pp512 = _as_float((authority.get("whole_tok_s") or {}).get("512"))
  base_pp512 = _as_float((comparator.get("whole_tok_s") or {}).get("512"))
  same_regime = cand_regime["regime_id"] == comp_regime["regime_id"]
  delta = None
  if base_pp512 == base_pp512 and base_pp512 != 0 and cand_pp512 == cand_pp512:
    delta = round((cand_pp512 - base_pp512) / base_pp512 * 100.0, 2)
  ok = bool(same_regime and delta is not None and delta >= 0.0)
  return {
    "status": "OK" if same_regime else "CROSS_REGIME_FORBIDDEN",
    "candidate_regime": cand_regime["regime_id"], "comparator_regime": comp_regime["regime_id"],
    "candidate_pp512": None if cand_pp512 != cand_pp512 else cand_pp512,
    "comparator_pp512": None if base_pp512 != base_pp512 else base_pp512,
    "delta_pct": delta, "same_regime": same_regime, "ok": ok,
    "note": None if same_regime else
            f"candidate regime {cand_regime['regime_id']!r} != comparator regime {comp_regime['regime_id']!r}; "
            "cross-regime pp512 comparison is forbidden (F2)",
  }


def build_authority_gate(authority: dict[str, Any], *, comparator: dict[str, Any] | None = None,
                         quality_gate: dict[str, Any] | None = None) -> dict[str, Any]:
  route = (authority.get("route_attribution") or {}).get("prefill_route_family")
  whole = authority.get("whole_tok_s") or {}
  pp512, pp4096 = whole.get("512"), whole.get("4096")
  route_ok = route == AUTHORITY_ROUTE
  perf_floor_ok = _as_float(pp512) >= PP512_MIN_TOK_S  # DIAGNOSTIC bare floor only; never grants authority alone
  binding_verdict = (authority.get("prefill_route_binding_gate") or {}).get("verdict")
  binding_ok = binding_verdict == "PREFILL_ROUTE_BINDING_PASS"
  regime = measurement_regime(authority)
  classification = _route_classification(route)
  classification_ok = bool(classification.get("final_default_allowed"))
  comparator_result = _comparator_delta(authority, comparator, regime)
  comparator_ok = comparator_result.get("ok") is True
  qg = quality_gate if quality_gate is not None else {
    "status": "MISSING",
    "note": "no whole-model dNLL/greedy-parity quality gate supplied; promotion refused per F3 (honesty over invention)",
  }
  quality_ok = qg.get("status") == "PASS"
  # AUTHORITY requires ALL of: right route, binding PASS (not waved off), a same-regime comparator delta,
  # a passing quality/correctness gate, AND a shipped-promotable classification. perf_floor_ok is NOT sufficient.
  authority_ok = bool(route_ok and binding_ok and comparator_ok and quality_ok and classification_ok)
  return {
    "schema": "prefill-hybrid-machine-search-authority-gate.v2",
    "required_route": AUTHORITY_ROUTE,
    "selected_route": route,
    "route_ok": route_ok,
    "pp512_min_tok_s": PP512_MIN_TOK_S,
    "pp512_tok_s": pp512,
    "pp4096_tok_s": pp4096,
    "pin_clock": authority.get("pin_clock"),
    "perf_floor_ok": perf_floor_ok,
    "measurement_regime": regime,
    "route_classification": classification,
    "classification_ok": classification_ok,
    "comparator": comparator_result,
    "comparator_ok": comparator_ok,
    "quality_gate": qg,
    "quality_ok": quality_ok,
    "binding_gate_verdict": binding_verdict,
    "binding_ok": binding_ok,
    "binding_gate_note": (
      "hybrid_machine_search classifies this path as hybrid/backend-atom, but a FAILING binding gate is NOT waved off: a "
      "binding FAIL blocks authority. The route stays research until the binding gate passes."
    ),
    "authority_ok": authority_ok,
    "ok": authority_ok,
  }


def build_final_report(*, candidate: dict[str, Any] | None = None,
                       authority: dict[str, Any] | None = None,
                       comparator: dict[str, Any] | None = None,
                       quality_gate: dict[str, Any] | None = None) -> dict[str, Any]:
  candidate = build_ffn_gate_up_candidate() if candidate is None else candidate
  authority_gate = build_authority_gate(authority, comparator=comparator, quality_gate=quality_gate) \
    if authority is not None else {
      "schema": "prefill-hybrid-machine-search-authority-gate.v2",
      "ok": False, "authority_ok": False, "status": "not_run",
      "required_route": AUTHORITY_ROUTE, "pp512_min_tok_s": PP512_MIN_TOK_S,
    }
  candidate_ok = (
    candidate.get("classification") == CLASSIFICATION and
    candidate.get("pure_generated") is False and
    (candidate.get("slot_identity_proof") or {}).get("ok") is True and
    candidate.get("promotion_status") == "candidate"
  )
  ready = candidate_ok and authority_gate.get("authority_ok") is True
  blocking_reasons: list[str] = []
  if not candidate_ok:
    blocking_reasons.append("candidate is not serializable / slot identity not proven")
  for key, msg in (
    ("route_ok", "authority route does not match required route"),
    ("binding_ok", f"binding gate is not PASS (verdict={authority_gate.get('binding_gate_verdict')!r})"),
    ("comparator_ok", "no same-regime comparator delta vs the current default (bare floor is not a comparator)"),
    ("quality_ok", "quality/correctness gate is MISSING or failing"),
    ("classification_ok", "route is research / final_default_allowed:false per route_manifest (not shippable)"),
  ):
    if authority_gate.get(key) is False:
      blocking_reasons.append(msg)
  verdict = "HYBRID_MACHINE_SEARCH_OWNED_BACKEND_ATOM_READY" if ready \
    else "HYBRID_MACHINE_SEARCH_RESEARCH_CANDIDATE_NOT_PROMOTED"
  return {
    "schema": "prefill-hybrid-machine-search-final-report.v2",
    "verdict": verdict,
    "classification": CLASSIFICATION,
    "pure_generated": False,
    "full_fine_tuned_hand_kernel": False,
    "candidate_ok": candidate_ok,
    "promotion": {
      "ready": ready,
      "decision": "promote_hybrid_machine_search_backend_atom" if ready
                  else "keep_default_authority_and_treat_hybrid_machine_search_as_research",
      "blocking_reasons": blocking_reasons,
    },
    "authority_gate": authority_gate,
    "candidate": candidate,
  }


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
  ap.add_argument("--search-output", type=pathlib.Path, default=DEFAULT_SEARCH_OUTPUT)
  ap.add_argument("--report-output", type=pathlib.Path, default=DEFAULT_REPORT_OUTPUT)
  ap.add_argument("--authority-artifact", type=pathlib.Path)
  ap.add_argument("--comparator-artifact", type=pathlib.Path,
                  help="same-regime current-default whole-synced authority artifact for the comparator delta (F1b)")
  ap.add_argument("--quality-gate-artifact", type=pathlib.Path,
                  help="whole-model quality/correctness gate JSON with a 'status' field (F3)")
  ap.add_argument("--active-buffers", type=int, default=2)
  ap.add_argument("--profile", default=DEFAULT_PROFILE,
                  help="model profile id to select role-shape data from")
  ap.add_argument("--role", default=ROLE, help="prefill linear role to serialize")
  ap.add_argument("--search", action="store_true", help="write the hybrid machine-search wait-policy search report")
  ap.add_argument("--json", action="store_true", help="print the candidate JSON to stdout")
  args = ap.parse_args(argv)

  record = write_search_report(args.search_output, active_buffers=args.active_buffers,
                               profile=args.profile, role=args.role) if args.search else \
           write_candidate(args.output, active_buffers=args.active_buffers, profile=args.profile, role=args.role)
  if args.authority_artifact is not None:
    authority = json.loads(args.authority_artifact.read_text())
    comparator = json.loads(args.comparator_artifact.read_text()) if args.comparator_artifact is not None else None
    quality_gate = json.loads(args.quality_gate_artifact.read_text()) if args.quality_gate_artifact is not None else None
    candidate = record["candidates"][0] if args.search else record
    report = build_final_report(candidate=candidate, authority=authority,
                                comparator=comparator, quality_gate=quality_gate)
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
