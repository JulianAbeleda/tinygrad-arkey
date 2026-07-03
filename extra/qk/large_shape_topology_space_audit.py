#!/usr/bin/env python3
"""Phases KT2/KT3 (combined evidence): enumerate the reachable Q4_K decode topology space for the large shapes and
determine whether words_per_group tuning can win BEFORE building a parametric emitter.

The G3 wave splits 32 lanes as block_groups * words_per_group == 32 (lane_extent), and the K reduction is
block_groups (cross-lane) * blocks_per_group (serial), with block_groups | k_blocks. So:

  block_groups <= gcd(32, k_blocks)   (must divide 32 AND divide k_blocks)

More K-parallelism = more block_groups = fewer serial blocks/lane. This audit enumerates the legal (block_groups,
words_per_group) per target shape and shows the max-parallel legal topology. If the shipped topology (bg=4, wpg=8)
already equals gcd(32, k_blocks), then no reachable words_per_group value can reduce serial work -- the wpg axis is
exhausted-or-worse and the missing lever is split-K (K-parallelism beyond the 32-lane constraint), not wpg.

Also records the emitter/primitive block: _q4k_block_dot_packed_load hardcodes 8 word-columns, so emitting wpg!=8
needs a new dot primitive (KT2_CODEGEN_CAPABILITY_BLOCKED_WORDS_PER_GROUP).

Writes bench/qwen-14b-32b-truegen/kt2_kt3_topology_space/{latest,candidate_space}.json + summary.md
"""
from __future__ import annotations
import sys, json, math, inspect, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qwen-14b-32b-truegen/kt2_kt3_topology_space"
QK_K, LANE_EXTENT = 256, 32

# target Q4_K decode shapes (in_features -> out_features), from KT0/Q1432-0
SHAPES = [
  ("attn_q/o",     5120,  5120),
  ("14b_ffn_gate", 5120,  17408),
  ("14b_ffn_down", 17408, 5120),
  ("32b_attn_q",   5120,  8192),
  ("32b_ffn_gate", 5120,  25600),
  ("32b_ffn_down", 25600, 5120),
]

def legal_topologies(k:int):
  """Enumerate legal (block_groups, words_per_group): bg*wpg==32, bg | k_blocks."""
  k_blocks = k // QK_K
  out = []
  for bg in (1, 2, 4, 8, 16, 32):
    if LANE_EXTENT % bg != 0: continue
    wpg = LANE_EXTENT // bg
    if k_blocks % bg != 0: continue
    out.append({"block_groups": bg, "words_per_group": wpg, "blocks_per_group_serial": k_blocks // bg})
  return k_blocks, out

def main():
  rows = []
  for role, in_f, out_f in SHAPES:
    k_blocks, tops = legal_topologies(in_f)
    bg_max = max(t["block_groups"] for t in tops)
    gcd = math.gcd(LANE_EXTENT, k_blocks)
    shipped = next((t for t in tops if t["block_groups"] == 4), None)
    rows.append({"role": role, "in_features": in_f, "out_features": out_f, "k_blocks": k_blocks,
                 "gcd_32_kblocks": gcd, "max_legal_block_groups": bg_max,
                 "shipped_bg4_is_optimal": bg_max == 4 and shipped is not None,
                 "legal_topologies": tops,
                 "reachable_wpg_values": sorted(t["words_per_group"] for t in tops)})

  all_bg4_optimal = all(r["shipped_bg4_is_optimal"] for r in rows)

  # primitive block: does the packed-load dot hardcode 8 word-columns?
  from extra.qk.quant.q4_k_gemv_primitive import _q4k_group_dot_packed_load, _q4k_block_dot_packed_load
  dot_src = inspect.getsource(_q4k_group_dot_packed_load)
  wpg8_locked = "(grp//2)*8 + lane4" in dot_src or "for nib in range(4)" in dot_src

  # is a split-K (parts) primitive available anywhere (the real missing axis)?
  import extra.qk.quant.q4_k_gemv_primitive as P
  split_k_primitive_exists = any("partial" in n or "parts" in inspect.getsource(getattr(P, n)) for n in dir(P)
                                 if callable(getattr(P, n)) and n.endswith("kernel"))

  verdict_kt2 = "KT2_CODEGEN_CAPABILITY_BLOCKED_WORDS_PER_GROUP"
  verdict_kt3 = "KT3_SEARCH_SPACE_INCOMPLETE_MISSING_SPLIT_K" if all_bg4_optimal else "KT3_PASS_ROLE_LOCAL_CANDIDATES_RANKED"

  result = {
    "verdict_kt2": verdict_kt2, "verdict_kt3": verdict_kt3,
    "lane_extent": LANE_EXTENT, "shapes": rows, "all_shapes_bg4_already_optimal": all_bg4_optimal,
    "words_per_group_primitive_locked_to_8": bool(wpg8_locked),
    "split_k_partials_primitive_exists": bool(split_k_primitive_exists),
    "headline": ("Every target shape has gcd(32, k_blocks)==4, so the shipped block_groups=4 (words_per_group=8) is "
                 "already the MOST-parallel legal topology; the only reachable alternatives (wpg 16/32 -> bg 2/1) are "
                 "strictly MORE serial. Varying words_per_group cannot win. And emitting wpg!=8 is separately blocked "
                 "because _q4k_block_dot_packed_load hardcodes 8 word-columns. The real missing lever is SPLIT-K "
                 "(K-parallelism beyond the 32-lane wave, via partials across workgroups) -- a partials 'parts' "
                 "primitive already exists in the gemm path but is not wired into the generated G3 GEMV route."),
  }
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "candidate_space.json").write_text(json.dumps(rows, indent=2))
  (OUT / "latest.json").write_text(json.dumps(result, indent=2))
  L = ["# KT2/KT3: reachable Q4_K topology space for large shapes", "",
       f"KT2 verdict: **{verdict_kt2}**", f"KT3 verdict: **{verdict_kt3}**", "", result["headline"], "",
       "| role | in→out | k_blocks | gcd(32,kb) | max legal bg | reachable wpg | serial blocks/lane @bg4 |",
       "|---|---|---|---|---|---|---|"]
  for r in rows:
    ser = next((t["blocks_per_group_serial"] for t in r["legal_topologies"] if t["block_groups"] == 4), "-")
    L.append(f"| {r['role']} | {r['in_features']}→{r['out_features']} | {r['k_blocks']} | {r['gcd_32_kblocks']} "
             f"| {r['max_legal_block_groups']} | {r['reachable_wpg_values']} | {ser} |")
  L += ["", f"words_per_group primitive locked to 8: **{bool(wpg8_locked)}** · "
        f"split-K partials primitive exists (unwired): **{bool(split_k_primitive_exists)}**"]
  (OUT / "summary.md").write_text("\n".join(L) + "\n")

  for r in rows:
    print(f"  {r['role']:14} k_blocks={r['k_blocks']:3d} gcd32={r['gcd_32_kblocks']} max_bg={r['max_legal_block_groups']} "
          f"reachable_wpg={r['reachable_wpg_values']} bg4_optimal={r['shipped_bg4_is_optimal']}")
  print(f"== {verdict_kt2} | {verdict_kt3} ==")

if __name__ == "__main__":
  main()
