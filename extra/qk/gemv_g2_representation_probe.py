#!/usr/bin/env python3
"""G2.0-G2.2 static representation probe for pure generated Q4_K GEMV."""
from __future__ import annotations

import pathlib
from datetime import datetime
from typing import Any

from tinygrad.uop.ops import UOp
from extra.qk.layout_coalesce_check import axis_stride, is_coalesced, vector_width
from extra.qk.gemv_g2_lanemap import Q4KGateUpLaneMap

ROOT = pathlib.Path(__file__).resolve().parents[2]


def eval_expr(expr:UOp, axes:dict[str, UOp], values:dict[str, int]) -> int:
  subs = {axes[k]: axes[k].const_like(v) for k, v in values.items() if k in axes}
  out = expr.substitute(subs).simplify()
  if out.arg is None: raise RuntimeError(f"expression did not fold to const: {out}")
  return int(out.arg)


def build() -> dict[str, Any]:
  lm = Q4KGateUpLaneMap()
  axes = lm.axis_uops()
  lane = lm.lane_expr(axes)
  word_idx = lm.packed_word_index_expr(axes)

  lane_stride_word_col = axis_stride(lane, axes["word_col"])
  lane_stride_block_group = axis_stride(lane, axes["block_group"])
  word_stride_word_col = axis_stride(word_idx, axes["word_col"])
  word_stride_block_group = axis_stride(word_idx, axes["block_group"])
  word_stride_local_block = axis_stride(word_idx, axes["local_block"])
  word_stride_group_pair = axis_stride(word_idx, axes["group_pair"])

  sample_rows = [0, 1, lm.n - 1]
  sample_local_blocks = [0, lm.blocks_per_group - 1]
  sample_group_pairs = [0, lm.group_pairs - 1]
  mismatches = []
  lane_mismatches = []
  samples = 0
  for row in sample_rows:
    for block_group in range(lm.block_groups):
      for local_block in sample_local_blocks:
        for group_pair in sample_group_pairs:
          for word_col in range(lm.words_per_group):
            values = {"row": row, "block_group": block_group, "local_block": local_block, "group_pair": group_pair, "word_col": word_col}
            got = eval_expr(word_idx, axes, values)
            exp = lm.packed_word_index_ref(row, block_group, local_block, group_pair, word_col)
            got_lane = eval_expr(lane, axes, values)
            exp_lane = block_group * lm.words_per_group + word_col
            samples += 1
            if got != exp: mismatches.append({"values": values, "got": got, "expected": exp})
            if got_lane != exp_lane: lane_mismatches.append({"values": values, "got": got_lane, "expected": exp_lane})

  checks = {
    "g2_0_lane_stride_word_col_is_1": lane_stride_word_col == 1,
    "g2_0_lane_stride_block_group_is_8": lane_stride_block_group == lm.words_per_group,
    "g2_0_word_stride_word_col_is_1": word_stride_word_col == 1,
    "g2_0_word_stride_block_group_is_blocks_per_group_times_36": word_stride_block_group == lm.blocks_per_group * lm.q4k_words_per_block,
    "g2_0_word_stride_local_block_is_36": word_stride_local_block == lm.q4k_words_per_block,
    "g2_0_word_stride_group_pair_is_8": word_stride_group_pair == lm.words_per_group,
    "g2_0_word_col_coalesced": is_coalesced(word_idx, axes["word_col"]),
    "g2_0_word_col_vector_width_4": vector_width(word_idx, axes["word_col"]) == 4,
    "g2_1_lanemap_serializable": bool(lm.serialize()["lane_formula"]),
    "g2_1_lane_equality": not lane_mismatches,
    "g2_2_sampled_index_equality": not mismatches,
  }

  verdict = "G2_LANEMAP_ADDRESS_BUILDER_PASS" if all(checks.values()) else "G2_LANEMAP_ADDRESS_BUILDER_FAIL"
  return {
    "date": "2026-06-25",
    "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "completed_phases": ["G2.0 static representation probe", "G2.1 minimal Q4_K LaneMap object", "G2.2 generated packed-address builder"],
    "profile": {"model": "Qwen3-8B-Q4_K_M", "device": "AMD", "arch": "gfx1100", "phase": "decode T==1", "projection": "FFN gate/up", "shape": f"{lm.k}x{lm.n}", "wave": lm.lane_extent},
    "lanemap": lm.serialize(),
    "strides": {"lane_wrt_word_col": lane_stride_word_col, "lane_wrt_block_group": lane_stride_block_group, "word_idx_wrt_word_col": word_stride_word_col, "word_idx_wrt_block_group": word_stride_block_group, "word_idx_wrt_local_block": word_stride_local_block, "word_idx_wrt_group_pair": word_stride_group_pair},
    "checks": checks,
    "samples_checked": samples,
    "mismatches": mismatches[:8],
    "lane_mismatches": lane_mismatches[:8],
    "decision": "G2.0-G2.2 are structurally passable after correcting Q4_K quant group-pairs to 4. Proceed to G2.3 generated dequant/reduce/store route; the remaining risk is runtime/codegen binding, not LaneMap or packed-address expression." if verdict == "G2_LANEMAP_ADDRESS_BUILDER_PASS" else "Do not proceed to runtime routing until the LaneMap/address-builder mismatch is classified.",
  }


if __name__ == "__main__":
  import sys; sys.path.insert(0, str(ROOT))
  from extra.qk.gate_registry import run
  raise SystemExit(run("gemv_g2_representation"))
