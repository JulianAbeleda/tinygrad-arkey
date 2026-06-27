#!/usr/bin/env python3
"""Owned-ASM oracle parity audit for decode attention.

Treats the owned hand-coded AMDGCN tile (`extra/qk_owned_flash_decode.hip`, routed by DECODE_ATTN_AMDGCN_TILE=1,
captured as `owned_flash_tile_gqa_whole`) as the REFERENCE ORACLE, and compares the current generated candidate
against it. This is a CLOSED PARITY problem: the primitive set is known from the owned ASM; the loop's job is parity
closure across primitive | placement | topology | resource | schedule | lifecycle | W==D/token.

It does no GPU work -- it consumes the artifacts produced by the existing tools and emits a machine-readable parity
matrix. A row whose datum is absent/stale is `UNKNOWN` (a finding: improve that instrument before searching).

Owned-vs-generated sources:
  - qk_decode_attention_isa_diff_gate.py   -> owned_tile resources+markers (the ORACLE) and a generated xlane tile
  - qk_decode_occupancy_guardrail.py       -> current generated block-tile resources
  - qk_decode_hotloop_schedule_diff.py     -> owned vs generated schedule (waitcnt, shadow-fill, cross-lane)
  - qk_decode_isa_vectorization_gate.py    -> generated route topology (split/workgroups)
  - qk_split_kv_economics_audit.py         -> combine/lifecycle
  - qk_decode_runtime_overhead.py + qk_decode_token_match_check.py -> W==D + token (the only promotion authority)

Run: PYTHONPATH=. python3 extra/qk_owned_oracle_parity_audit.py
"""
from __future__ import annotations
import json, pathlib, glob

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-owned-oracle-parity"
ISA_DIFF = ROOT / "bench/qk-decode-attention-isa-diff/latest.json"
HOTLOOP = ROOT / "bench/qk-decode-hotloop-schedule-diff/latest.json"
OCC = ROOT / "bench/qk-decode-occupancy-guardrail/latest.json"
ISA_VEC = ROOT / "bench/qk-decode-isa-vectorization/latest.json"
ECON = ROOT / "bench/qk-split-kv-economics-audit/latest.json"
WD_THRESHOLD = 90.0

def load(p):
  try: return json.loads(p.read_text())
  except Exception: return {}

def latest_transfer():
  fs = sorted(glob.glob(str(ROOT / "bench/qk-pure-search-gap/transfer_snapshot_*.json")))
  return load(pathlib.Path(fs[-1])) if fs else {}

def st(owned, gen):
  if owned is None and gen is None: return "UNKNOWN"
  if owned is None or gen is None: return "MISSING"
  return "MATCH" if owned == gen else "MISMATCH"

def row(layer, name, owned_prop, owned_obs, gen_obs, status, blocker, tool, action, axis, gate):
  return {"layer": layer, "row": name, "owned_property": owned_prop, "owned_observation": owned_obs,
          "generated_observation": gen_obs, "status": status, "blocker_kind": blocker,
          "responsible_tool": tool, "required_action": action, "candidate_axis": axis, "gate_to_close": gate}

def main() -> int:
  isad = load(ISA_DIFF); hot = load(HOTLOOP); occ = load(OCC); isav = load(ISA_VEC); econ = load(ECON)
  tr = latest_transfer()

  o_tile = isad.get("owned_tile", {})
  o_res, o_mark = o_tile.get("resources", {}), o_tile.get("markers", {})
  g_occ = {k: v.get("value") for k, v in occ.get("checks", {}).items()}   # generated block-tile resources
  g_mark = (isav.get("capture", {}).get("tile", {}) or {}).get("markers", {})   # generated block-tile EVIDENCE (markers)
  def prim_status(o, g):   # primitive parity: MATCH only when BOTH owned and generated evidence show it present
    if g is None: return "UNKNOWN"      # no generated evidence captured -> cannot claim MATCH (do not trust prose)
    if (g or 0) > 0 and (o or 0) > 0: return "MATCH"
    return "MISSING"
  g_wide = sum((g_mark.get(k) or 0) for k in
               ("global_load_dwordx4", "global_load_b128", "global_load_b96", "global_load_b64", "global_load_d16"))
  cmp = hot.get("comparison", {})
  route = isav.get("route_cleanliness", {}).get("occupancy", {})
  arms = {a["arm"]: a for a in tr.get("arms", [])}
  o_wd, g_wd = arms.get("owned_baseline", {}), arms.get("block_tile_route_full_stack", {})
  def wd_pct(c):
    o, g = o_wd.get(f"ctx{c}_tok_s"), g_wd.get(f"ctx{c}_tok_s")
    return round(g / o * 100.0, 1) if o and g else None

  ISADIFF, HOT, OCCT, ISAVT = ("qk_decode_attention_isa_diff_gate.py", "qk_decode_hotloop_schedule_diff.py",
                               "qk_decode_occupancy_guardrail.py", "qk_decode_isa_vectorization_gate.py")
  ds_o, ds_g = cmp.get("ds_bpermute_owned_vs_generated", [None, None])
  wc_o, wc_g = cmp.get("s_waitcnt_owned_vs_generated", [None, None])
  sf_o, sf_g = cmp.get("ds_bpermute_shadow_fill_owned_vs_generated", [None, None])

  rows = [
    # PRIMITIVE parity: the known owned primitives must be expressed by the generated path -- judged from GENERATED
    # EVIDENCE (isa_vectorization markers), not a prose assertion. No generated evidence -> UNKNOWN (not MATCH).
    row("primitive", "v_dot2_score", "v_dot2 packed dot", o_mark.get("v_dot2"), g_mark.get("v_dot2"),
        prim_status(o_mark.get("v_dot2"), g_mark.get("v_dot2")), None if g_mark.get("v_dot2") else "MISSING_PRIMITIVE",
        ISAVT, "none if present; else expose v_dot2 lowering", "V_DOT2_LOWERING", "isa_vectorization v_dot2 > 0"),
    row("primitive", "cross_lane_reduce", "ds_bpermute cross-lane", o_mark.get("cross_lane"), g_mark.get("cross_lane"),
        prim_status(o_mark.get("cross_lane"), g_mark.get("cross_lane")),
        None if g_mark.get("cross_lane") else "MISSING_PRIMITIVE", ISAVT, "none if present", None,
        "isa_vectorization cross_lane > 0"),
    row("primitive", "lds_kv_staging", "8 KB LDS K/V tile", o_res.get("lds"), g_occ.get("lds"),
        st(o_res.get("lds"), g_occ.get("lds")), None, ISADIFF, "none (present)", "DECODE_STAGE_COALESCE",
        "isa shows LDS tile"),

    # PLACEMENT parity: do primitives land in owned-equivalent layout/vectorization? Now judged from generated
    # isa_vectorization load markers (b64/dwordx4/d16). MATCH iff generated emits vectorized (wide) loads.
    row("placement", "load_vectorization", "owned vectorized loads (d16 x22)", f"d16={o_mark.get('global_load_d16')}",
        f"b64={g_mark.get('global_load_b64')} dwordx4={g_mark.get('global_load_dwordx4')} d16={g_mark.get('global_load_d16')}",
        ("UNKNOWN" if not g_mark else "MATCH" if g_wide > 0 else "MISMATCH"),
        None if g_wide > 0 else "PRIMITIVE_PLACEMENT_BUG", ISAVT,
        "generated must emit wide/vectorized loads (b64/dwordx4); if scalar-only, fix coalescing placement",
        "COALESCED_LOAD_LOWERING", "generated emits vectorized loads"),
    row("placement", "reduce_placement", f"cross_lane {ds_o} per owned loop", ds_o,
        f"cross_lane {ds_g} per generated loop (loop-size-confounded)", "UNKNOWN", "INSTRUMENTATION_GAP", ISADIFF,
        "normalize cross-lane per logical reduction (owned/gen loop bodies differ) before judging placement",
        None, "per-reduction-normalized cross-lane parity"),

    # TOPOLOGY parity: workgroup count / split structure. Both S=48 / 384 wg -> MATCH.
    row("topology", "workgroup_count", "split S=48 -> 384 workgroups", 48, route.get("split_count"),
        st(48, route.get("split_count")), "TIMING_TRIGGER", ISAVT,
        "if MISMATCH, change split count (more workgroups); W==D-gated (combine tax in-model)",
        "DECODE_ATTN_FUSED_XLANE_SCORE_PV_S", "route split/wg parity"),

    # RESOURCE parity: owned IS captured by isa_diff (vgpr 64). Generated block tile = 88 -> MISMATCH (real gap).
    row("resource", "vgpr", "owned vgpr (from isa_diff)", o_res.get("vgpr"), g_occ.get("vgpr"),
        st(o_res.get("vgpr"), g_occ.get("vgpr")), "TIMING_TRIGGER", OCCT,
        "reduce generated VGPR toward owned (work-removal, NOT ILP-via-state which raises it); prove vgpr dropped",
        None, "vgpr within owned band (note: SCHED_UNROLL/splits RAISE vgpr -> wrong direction)"),
    row("resource", "lds_bytes", "8192 B", o_res.get("lds"), g_occ.get("lds"), st(o_res.get("lds"), g_occ.get("lds")),
        None, OCCT, "none", None, "lds parity"),
    row("resource", "scratch", "0", o_res.get("scratch"), g_occ.get("scratch"),
        st(o_res.get("scratch"), g_occ.get("scratch")), None, OCCT, "none", None, "scratch parity"),

    # SCHEDULE parity: waitcnt + latency shadow-fill (hotloop). Both MISMATCH -> the open SCHEDULE delta.
    row("schedule", "waitcnt", f"{wc_o} s_waitcnt", wc_o, wc_g, st(wc_o, wc_g), "TIMING_TRIGGER", HOT,
        "reduce generated waitcnt drains toward owned; re-run hotloop and prove the counter moved", "SCHED_UNROLL",
        "hotloop waitcnt within owned band"),
    row("schedule", "latency_shadow_fill", f"{sf_o} avg fill", sf_o, sf_g, st(sf_o, sf_g), "TIMING_TRIGGER", HOT,
        "improve latency hiding so fill approaches owned; prove the counter moved", "SCHED_LIST",
        "hotloop shadow-fill within owned band"),

    # LIFECYCLE parity: the split-KV combine. economics artifact absent -> UNKNOWN.
    row("lifecycle", "split_kv_combine", "efficient many-split combine", None,
        econ.get("verdict", "economics artifact absent"), "UNKNOWN" if not econ else "MISMATCH",
        "INSTRUMENTATION_GAP" if not econ else "TIMING_TRIGGER", "qk_split_kv_economics_audit.py",
        "run split-KV economics for the block-tile route; then pursue a cheaper/fused combine (W==D-gated)",
        "DECODE_ATTN_FUSED_XLANE_SCORE_PV_S", "combine economics not dominating"),

    # W==D / token parity: the bottom line; only the W==D harness + token-match can produce PROMOTABLE.
    row("wd_token", "wd_tok_s", f"owned {o_wd.get('ctx512_tok_s')}/{o_wd.get('ctx4096_tok_s')} tok/s",
        [o_wd.get("ctx512_tok_s"), o_wd.get("ctx4096_tok_s")], [g_wd.get("ctx512_tok_s"), g_wd.get("ctx4096_tok_s")],
        "MATCH" if (wd_pct(4096) or 0) >= WD_THRESHOLD else "MISMATCH", "TIMING_TRIGGER",
        "qk_decode_runtime_overhead.py + qk_decode_token_match_check.py",
        "the bottom-line parity; closes only when schedule+resource+lifecycle rows close", None,
        f"W==D >= {WD_THRESHOLD}% of owned AND token-match"),
  ]

  from collections import Counter
  summ = dict(Counter(r["status"] for r in rows))
  failed = [r for r in rows if r["status"] in ("MISMATCH", "MISSING")]
  unknown = [r for r in rows if r["status"] == "UNKNOWN"]
  wd_match = next((r for r in rows if r["row"] == "wd_tok_s"), {}).get("status") == "MATCH"

  searchable = [r for r in failed if r["candidate_axis"]]
  unk = [r["row"] for r in unknown]
  if not failed and not unknown and wd_match:
    verdict = "PARITY_CLOSED__PROMOTABLE_PENDING_WD_TOKEN_MATCH"
    rec = "run W==D + token-match to confirm PROMOTABLE"
  elif not failed and not wd_match:
    verdict = "INSTRUMENTATION_INCOMPLETE__ALL_VISIBLE_ROWS_MATCH_BUT_WD_FAILS"
    rec = "all visible rows MATCH but W==D fails -> a hidden delta is unmeasured; improve attribution"
  elif searchable:
    verdict = "PARITY_OPEN__FAILED_ROWS_TARGETABLE"
    rec = (f"search failed rows {[r['row'] for r in searchable]} via generator --failed-rows"
           + (f"; IN PARALLEL instrument UNKNOWN rows {unk} (they only block candidates that target them)" if unk else ""))
  elif unknown:
    verdict = "PARITY_OPEN__ONLY_UNKNOWN_ROWS__IMPROVE_INSTRUMENTATION"
    rec = f"no searchable failed row; instrument UNKNOWN rows {unk} first"
  else:
    verdict = "PARITY_OPEN__FAILED_ROWS_HAVE_NO_SEARCHABLE_AXIS"
    rec = f"failed rows {[r['row'] for r in failed]} have no knob axis -> add a capability (work-removal/primitive/topology), NOT more search"

  out = {
    "schema": "qk_owned_oracle_parity_audit_v1",
    "oracle": {"kernel": "extra/qk_owned_flash_decode.hip", "route_flag": "DECODE_ATTN_AMDGCN_TILE=1",
               "captured_as": "owned_flash_tile_gqa_whole"},
    "wd_threshold_pct_of_owned": WD_THRESHOLD,
    "summary": summ,
    "failed_rows": [r["row"] for r in failed],
    "unknown_rows": [r["row"] for r in unknown],
    "recommended_next": rec,
    "decision_unknown_rows": ("UNKNOWN rows block ONLY candidates that target them (they are excluded from "
                              "searchable_failed_rows). Measurable failed rows remain searchable. Prefer closing "
                              "instrumentation gaps for trust, but UNKNOWN is NOT a global hard block."),
    # the loop drives candidate generation off THESE: only a candidate whose targets_delta is a searchable failed row may run.
    "searchable_failed_rows": [{"row": r["row"], "candidate_axis": r["candidate_axis"], "gate_to_close": r["gate_to_close"]}
                               for r in failed if r["candidate_axis"]],
    "rules": [
      "every candidate must target a FAILED parity row (status MISMATCH/MISSING with a candidate_axis)",
      "a candidate that does not MOVE its target row -> SEARCH_SPACE_BUG (or TOOLING_BUG if unobservable)",
      "an UNKNOWN row -> do NOT search; improve that instrument first",
      "all rows MATCH but W==D fails -> instrumentation incomplete",
      "PROMOTABLE only when parity rows match enough AND W==D + token-match clears threshold",
    ],
    "matrix": rows,
    "verdict": verdict,
  }
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
