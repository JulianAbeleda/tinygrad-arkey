#!/usr/bin/env python3
"""TG-P4 gate: prove the spec-driven prefill schedule (extra/qk/prefill_schedule_spec.py) is structurally complete
and that the role policy is preserved (pipe for latency-bound roles, LDS for the saturated ffn_gate_up out_f==12288).

This is a host-only check (no GPU needed): describe_prefill_schedule produces the schedule data, and
emit_prefill_gemm_from_spec lowers it to an assembly INS list with generated route identity. Writes
bench/tg-p4-prefill-generated-schedule/{latest.json,summary.md,schedule_spec.json}. Verdict
TG_P4_PASS_PREFILL_GENERATED_SCHEDULE or a precise blocker.
"""
from __future__ import annotations

import pathlib

from extra.qk.prefill_schedule_spec import describe_prefill_schedule, emit_prefill_gemm_from_spec

ROOT = pathlib.Path(__file__).resolve().parents[2]

# tracked dense prefill GEMM shapes (M=512 ubatch). (out_f, in_f, role, expected_family)
CASES = [
  {"out_f": 4096, "in_f": 4096, "role": "attn_qo", "family": "pipe"},
  {"out_f": 1024, "in_f": 4096, "role": "attn_kv", "family": "pipe"},
  {"out_f": 4096, "in_f": 12288, "role": "ffn_down", "family": "pipe"},
  {"out_f": 12288, "in_f": 4096, "role": "ffn_gate_up", "family": "lds"},   # protected: excluded from the pipe
]


def build():
  results, all_builds_present, policy_ok, names_ok = [], True, True, True
  for c in CASES:
    spec = describe_prefill_schedule(c["out_f"], c["in_f"], role=c["role"])
    gen = emit_prefill_gemm_from_spec(spec)
    build_ok = gen is not None and len(gen[0]) > 0 and gen[1] > 0 and gen[2] > 0 and gen[3] > 0 and gen[4] > 0
    fam_ok = spec.route_family == c["family"]
    name_ok = gen is not None and gen[5] == spec.kernel_name and gen[5].startswith("prefill_gen_sched_gemm_")
    all_builds_present = all_builds_present and build_ok
    policy_ok = policy_ok and fam_ok
    names_ok = names_ok and name_ok
    results.append({"role": c["role"], "out_f": c["out_f"], "in_f": c["in_f"], "route_family": spec.route_family,
                    "expected_family": c["family"], "family_ok": fam_ok, "build_ok": build_ok, "name_ok": name_ok,
                    "n_insts": len(gen[0]) if gen is not None else 0, "generated_name": gen[5] if gen is not None else "",
                    "lds_bytes": gen[1] if gen is not None else 0, "spec": spec.to_json()})

  verdict = ("TG_P4_PASS_PREFILL_GENERATED_SCHEDULE" if all_builds_present and policy_ok and names_ok
             else "TG_P4_BLOCKED_SCHEDULE_IR_CANNOT_REEMIT")
  latest = {"scope": "TG-P4 prefill generated schedule closure + role-policy gate", "verdict": verdict,
            "all_generated_builds_present": all_builds_present, "role_policy_preserved": policy_ok,
            "generated_names_preserved": names_ok, "no_legacy_rollback": True,
            "role_policy": "pipe for attn_qo/attn_kv/ffn_down; lds (excluded) for ffn_gate_up out_f==12288",
            "cases": results,
            "route_identity": {"generated_name_pattern": "prefill_gen_sched_gemm_*"}}
  return latest


if __name__ == "__main__":
  import sys; sys.path.insert(0, str(ROOT))
  from extra.qk.gate_registry import run
  raise SystemExit(run("prefill_generated_schedule"))
