#!/usr/bin/env python3
"""TG-P4 gate: prove the spec-driven prefill schedule (extra/qk/prefill_schedule_spec.py) reproduces the legacy
role-selective route byte-for-byte, and that the role policy is preserved (pipe for latency-bound roles, LDS for the
saturated ffn_gate_up out_f==12288).

Instruction-stream identity is a host-only check (no GPU needed): _emit_schedule returns the assembly INS list; the
generated route and the legacy _kernel resolve the same params, so the encoded instruction bytes must match. Writes
bench/tg-p4-prefill-generated-schedule/{latest.json,summary.md,schedule_spec.json}. Verdict
TG_P4_PASS_PREFILL_GENERATED_SCHEDULE or a precise blocker.
"""
from __future__ import annotations

import json, pathlib

from extra.qk.prefill_graph_gemm_route import _kernel, _resolve_schedule
from extra.qk.prefill_schedule_spec import describe_prefill_schedule, emit_prefill_gemm_from_spec

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/tg-p4-prefill-generated-schedule"

# tracked dense prefill GEMM shapes (M=512 ubatch). (out_f, in_f, role, expected_family)
CASES = [
  {"out_f": 4096, "in_f": 4096, "role": "attn_qo", "family": "pipe"},
  {"out_f": 1024, "in_f": 4096, "role": "attn_kv", "family": "pipe"},
  {"out_f": 4096, "in_f": 12288, "role": "ffn_down", "family": "pipe"},
  {"out_f": 12288, "in_f": 4096, "role": "ffn_gate_up", "family": "lds"},   # protected: excluded from the pipe
]


def _insts_sig(built):
  # deterministic instruction-stream signature: per-instruction repr + encoded size, plus the launch geometry.
  insts, lds_bytes, bm, bn, threads, name = built
  stream = tuple((repr(i), i.size()) for i in insts)
  return stream, (lds_bytes, bm, bn, threads)


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  results, all_identical, policy_ok = [], True, True
  for c in CASES:
    legacy = _kernel(c["out_f"], c["in_f"])
    spec = describe_prefill_schedule(c["out_f"], c["in_f"], role=c["role"])
    gen = emit_prefill_gemm_from_spec(spec)
    lstream, lgeom = _insts_sig(legacy)
    gstream, ggeom = _insts_sig(gen)
    identical = (lstream == gstream) and (lgeom == ggeom)
    fam_ok = spec.route_family == c["family"]
    all_identical = all_identical and identical
    policy_ok = policy_ok and fam_ok
    results.append({"role": c["role"], "out_f": c["out_f"], "in_f": c["in_f"], "route_family": spec.route_family,
                    "expected_family": c["family"], "family_ok": fam_ok, "instructions_identical": identical,
                    "n_insts": len(legacy[0]), "legacy_name": legacy[5], "generated_name": gen[5],
                    "lds_bytes": gen[1], "spec": spec.to_json()})

  verdict = ("TG_P4_PASS_PREFILL_GENERATED_SCHEDULE" if all_identical and policy_ok
             else "TG_P4_BLOCKED_SCHEDULE_IR_CANNOT_REEMIT")
  latest = {"scope": "TG-P4 prefill generated schedule lossless + role-policy gate", "verdict": verdict,
            "all_instructions_identical": all_identical, "role_policy_preserved": policy_ok,
            "role_policy": "pipe for attn_qo/attn_kv/ffn_down; lds (excluded) for ffn_gate_up out_f==12288",
            "cases": results,
            "route_identity": {"generated_name_pattern": "prefill_gen_sched_gemm_*",
                               "legacy_name_pattern": "prefill_graph_gemm_*"}}
  json.dump(latest, open(OUT / "latest.json", "w"), indent=2)
  json.dump([r["spec"] for r in results], open(OUT / "schedule_spec.json", "w"), indent=2)
  md = [f"# TG-P4 Prefill Generated Schedule\n", f"Verdict: **{verdict}**\n",
        f"All instructions identical: {all_identical}; role policy preserved: {policy_ok}\n",
        "| role | out_f | in_f | family | expected | family_ok | insts_identical | n_insts |",
        "|---|---|---|---|---|---|---|---|"]
  for r in results:
    md.append(f"| {r['role']} | {r['out_f']} | {r['in_f']} | {r['route_family']} | {r['expected_family']} | "
              f"{r['family_ok']} | {r['instructions_identical']} | {r['n_insts']} |")
  open(OUT / "summary.md", "w").write("\n".join(md) + "\n")
  print(verdict, "all_identical=", all_identical, "policy_ok=", policy_ok)
  return 0 if verdict == "TG_P4_PASS_PREFILL_GENERATED_SCHEDULE" else 1


if __name__ == "__main__":
  raise SystemExit(main())
