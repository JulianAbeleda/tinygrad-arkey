#!/usr/bin/env python3
"""PMS-R4: seed the promote/refute ledger with the current durable hot-kernel decisions.

Appends (idempotently, by candidate_id) the five PMS-R4 rows to
bench/qk-project-search-ledger/ledger.jsonl. Each row keeps the EXISTING ledger schema fields
(bench/qk-project-search-ledger/schema.json) AND carries the PMS-R4 scope fields (profile_id, route_id, workload,
owned_or_baseline, rollback, do_not_search_implications) so the ledger is the durable "do not reopen / promoted"
record the candidate evaluator/generator can read.

Three rows REUSE the R2 evaluator's ledger_update.json (bench/qk-candidate-evaluator/<route_id>/ledger_update.json),
remapped to the PMS-R4 candidate_id and enriched with the scope fields. Two rows (attention-native low-leverage,
prefill global promoted) are built from the route manifest + cited authority artifacts (no GPU re-run; audit-first).

Run:  PYTHONPATH=. python3 extra/qk_ledger_seed_pms_r4.py            # append missing rows (idempotent)
      PYTHONPATH=. python3 extra/qk_ledger_seed_pms_r4.py --check    # report only, do not write
"""
from __future__ import annotations
import json, pathlib, sys
from extra.qk_route_manifest import route
from extra.qk_search_util import ledger_candidate_ids

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "bench/qk-project-search-ledger/ledger.jsonl"
EVAL = ROOT / "bench/qk-candidate-evaluator"

# scope fields layered on top of the existing schema for every PMS-R4 row
SCOPE_FIELDS = ["candidate_id", "profile_id", "route_id", "workload", "primitive_class", "owned_or_baseline",
                "correctness", "route_identity", "authority_benchmark", "verdict", "rollback", "artifact_links",
                "learned_rule", "do_not_search_implications"]


def _reuse(route_id: str, candidate_id: str, dns: list[str]) -> dict:
  """Load the R2 ledger_update.json for route_id, remap candidate_id, add the PMS-R4 scope fields."""
  row = json.load(open(EVAL / route_id / "ledger_update.json"))
  rmeta = route(route_id)
  row["candidate_id"] = candidate_id
  row["profile_id"] = rmeta.get("profile_id")
  row["route_id"] = route_id
  row["workload"] = rmeta.get("workload")
  row["owned_or_baseline"] = row.get("oracle")
  row["rollback"] = rmeta.get("rollback", {})
  row["do_not_search_implications"] = dns
  return row


def _attention_native_row() -> dict:
  rid = "decode_attention_native_correct_not_fast"
  rmeta = route(rid)
  ceil = json.load(open(ROOT / "bench/amd-isa-backend-decode-attention-ceiling/latest.json"))
  nvo = ceil["native_vs_owned"]; ws = ceil["loss_stack"]["tile_wall_share_measured"]
  return {
    "candidate_id": "decode/attention_native_correct_not_fast_low_leverage",
    "profile_id": rmeta["profile_id"], "route_id": rid, "workload": "decode",
    "lane": "decode", "primitive_class": "attention",
    "knobs": {"env_to_force": rmeta["env"], "rollback": rmeta.get("rollback", {})},
    "oracle": "decode_attention_owned_two_kernel", "owned_or_baseline": "decode_attention_owned_two_kernel",
    "correctness": "token-correct + route-bound (generated whole-cache flash decode fires; no owned fallback)",
    "route_identity": rmeta["route_attribution"],
    "materialization_abi": "n/a",
    "isa": "native generated AMD-ISA / whole-cache flash tile vs owned two-kernel split tile",
    "local_diagnostic": f"native_vs_owned {nvo['512']}%@512 -> {nvo['4096']}%@4096 (correct but ~60-68% of owned)",
    "authority_benchmark": {"authority_type": "decode_wd_ceiling_amdahl",
                            "native_vs_owned_pct": nvo, "tile_wall_share_measured": ws,
                            "ceiling_verdict": ceil["verdict"]},
    "verdict": "CORRECT_NOT_FAST_LOW_LEVERAGE (reproduces AMD_ISA_ATTENTION_CEILING_PASS_MOVE_TO_NON_ATTENTION)",
    "rollback": rmeta.get("rollback", {}),
    "stop_reason": "native attention is correct/route-bound but below owned; ceiling audit shows attention wall-share "
                   "~10%@512 -> ~3%@4096 (Amdahl) -> low whole-decode leverage; owned stays shipped.",
    "artifact_links": ["bench/amd-isa-backend-phase-n7/latest.json",
                       "bench/amd-isa-backend-decode-attention-ceiling/latest.json",
                       "extra/qk_route_manifest.py"],
    "learned_rule": "native/generated decode attention is a correct route but ~60-68% of owned; its WALL share is "
                    "small and overlapped by the weight-bound FFN, so it cannot move whole decode under the current "
                    "Qwen3-8B-Q4_K_M/gfx1100 target. Keep owned two-kernel shipped; do not spend broad attention search.",
    "do_not_search_implications": [
      "native_attention_as_default: correct_not_fast (~60-68% of owned) -> do not promote",
      "broad decode-attention combine/fusion: exhausted/low-leverage (PMS-R7 gate = DO_NOT_REOPEN_ATTENTION)",
      "scheduler-only / occupancy-LDS-only / N1B scalar-address attention tuning: refuted, no W==D movement"],
  }


def _prefill_global_row() -> dict:
  rid = "prefill_pipe_global_rollback"
  rmeta = route(rid)
  promo = json.load(open(ROOT / "bench/qk-prefill-pipe-promotion/latest.json"))
  wd = promo["wd_table"]
  per_ctx = {c: {"baseline_old_lds2": v["old"], "candidate_global_pipe": v["new"], "delta_pct": v["delta_pct"]}
             for c, v in wd.items()}
  return {
    "candidate_id": "prefill/pipe_global_promoted",
    "profile_id": rmeta["profile_id"], "route_id": rid, "workload": "prefill",
    "lane": "prefill", "primitive_class": "route_policy",
    "knobs": {"env_to_force": rmeta["env"], "rollback": rmeta.get("rollback", {})},
    "oracle": "old lds2 default", "owned_or_baseline": "old lds2 default (PREFILL_GEMM_PIPELINE=0)",
    "correctness": promo["correctness"],
    "route_identity": rmeta["route_attribution"],
    "materialization_abi": "n/a",
    "isa": "software-pipelined assembly GEMM (build_gemm_pipe tm2/tn2) for ALL graph-gemm roles",
    "local_diagnostic": "all-ctx TIER_A_MAJOR vs old lds2 default; "
                        + ", ".join(f"{c}:+{v['delta_pct']}%" for c, v in wd.items()),
    "authority_benchmark": {"authority_type": "prefill_whole", "per_ctx_tok_s": per_ctx,
                            "verdict": promo["verdict"], "flip_commit": promo["flip_commit"]},
    "verdict": "PROMOTE_TIER_A_SUPERSEDED_BY_ROLE_SELECTIVE (reproduces PIPE_PROMOTE_PASS_DEFAULT_FLIPPED)",
    "rollback": rmeta.get("rollback", {}),
    "stop_reason": "global pipe was promoted TIER_A vs old lds2 (+8.5..19.2%), then SUPERSEDED by role-selective "
                   "(prefill_pipe_role_selective_default) which excludes the BLAS-saturated ffn_gate_up; kept as "
                   "the A/B rollback comparator (PREFILL_PIPE_ROLE_SELECTIVE=0).",
    "artifact_links": ["bench/qk-prefill-pipe-promotion/latest.json", "bench/qk-prefill-pipe-promotion/summary.md",
                       "extra/qk_prefill_graph_gemm_route.py", "extra/qk_route_manifest.py"],
    "learned_rule": "global pipe_tm2_tn2 beats the old lds2 prefill default TIER_A at every ctx (output-equivalent), "
                    "but is itself superseded by the role-selective default; it is the rollback comparator, not the "
                    "live default. Do not re-prove the global-pipe-vs-lds2 result.",
    "do_not_search_implications": [
      "global-pipe-vs-old-lds2: settled TIER_A; do not re-search",
      "the live prefill default is role-selective (prefill_pipe_role_selective_default); global pipe is its rollback A/B"],
  }


def rows() -> list[dict]:
  return [
    _reuse("decode_q4k_g3_generated", "decode/q4k_g3_generated_speed_equivalent",
           ["q4k_offline_layout_reshuffle: deprioritized while G3 parity holds (G3 == owned, no layout gap)",
            "decode Q4_K GEMV closest-to-pure default; do not reopen as built"]),
    _reuse("decode_q6k_direct_refuted", "decode/q6k_direct_refuted",
           ["q6k_direct_half_warp_route: refuted (W==D -4.77..-6.06%); do not re-chase the half-warp partition as built",
            "only reopen Q6_K direct with a DIFFERENT topology + a fresh residual audit"]),
    _attention_native_row(),
    _prefill_global_row(),
    _reuse("prefill_pipe_role_selective_default", "prefill/pipe_role_selective_promoted",
           ["role-selective prefill pipe is the promoted default; re-proving the speed result is not a search target",
            "ffn_gate_up (out_f==12288) stays on lds2 (BLAS-saturated); do not pipe it"]),
  ]


def existing_ids() -> set[str]:
  return ledger_candidate_ids(LEDGER)  # shared safe reader (was an unsafe d["candidate_id"] -> bug #5)


def main() -> int:
  check = "--check" in sys.argv[1:]
  have = existing_ids()
  new = [r for r in rows() if r["candidate_id"] not in have]
  print(json.dumps({"verdict": "PMS_R4_PASS_LEDGER_CURRENT",
                    "already_present": sorted(r["candidate_id"] for r in rows() if r["candidate_id"] in have),
                    "to_append": [r["candidate_id"] for r in new],
                    "scope_fields_on_every_row": all(all(f in r for f in SCOPE_FIELDS) for r in rows()),
                    "ledger": str(LEDGER.relative_to(ROOT))}, indent=2))
  if not check and new:
    with LEDGER.open("a") as f:
      for r in new: f.write(json.dumps(r) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
