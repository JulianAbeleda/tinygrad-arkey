#!/usr/bin/env python3
"""G2.0 static representation probe for pure generated Q4_K GEMV.

This does not run the model. It answers the first G2 question: can the existing
UOp/RANGE address algebra express the lane-partition packed-word index used by
the fast Q4_K gate/up GEMV structure?
"""
from __future__ import annotations

import json, os
from datetime import datetime
from pathlib import Path
from typing import Any

from tinygrad.uop.ops import UOp, AxisType
from extra.qk_layout_coalesce_check import axis_stride, is_coalesced, vector_width

OUT_DIR = Path("bench/qk-gemv-g2-representation-probe")
K = 4096
N = 12288
QK_K = 256
K_BLOCKS = K // QK_K
WARP = 32
BLOCK_GROUPS = 4
WORDS_PER_GROUP = 8
BLOCKS_PER_GROUP = K_BLOCKS // BLOCK_GROUPS
Q4K_WORDS_PER_BLOCK = 36
Q4K_QUANT_WORD_BASE = 4
GROUP_PAIRS = 8


def q4k_word_index_ref(row:int, block_group:int, local_block:int, group_pair:int, word_col:int) -> int:
  blk = block_group * BLOCKS_PER_GROUP + local_block
  return (row * K_BLOCKS + blk) * Q4K_WORDS_PER_BLOCK + Q4K_QUANT_WORD_BASE + group_pair * WORDS_PER_GROUP + word_col


def build_uop_expr() -> tuple[dict[str, UOp], UOp]:
  row = UOp.range(N, 0, AxisType.GLOBAL)
  block_group = UOp.range(BLOCK_GROUPS, 1, AxisType.LOCAL)
  word_col = UOp.range(WORDS_PER_GROUP, 2, AxisType.LOCAL)
  local_block = UOp.range(BLOCKS_PER_GROUP, 3, AxisType.REDUCE)
  group_pair = UOp.range(GROUP_PAIRS, 4, AxisType.REDUCE)

  lane = block_group * WORDS_PER_GROUP + word_col
  blk = block_group * BLOCKS_PER_GROUP + local_block
  word_idx = (row * K_BLOCKS + blk) * Q4K_WORDS_PER_BLOCK + Q4K_QUANT_WORD_BASE + group_pair * WORDS_PER_GROUP + word_col
  return {"row": row, "block_group": block_group, "word_col": word_col, "local_block": local_block, "group_pair": group_pair, "lane": lane}, word_idx


def eval_expr(expr:UOp, axes:dict[str, UOp], values:dict[str, int]) -> int:
  subs = {axes[k]: axes[k].const_like(v) for k, v in values.items() if k in axes and k != "lane"}
  out = expr.substitute(subs).simplify()
  if out.arg is None: raise RuntimeError(f"expression did not fold to const: {out}")
  return int(out.arg)


def run_probe() -> dict[str, Any]:
  axes, word_idx = build_uop_expr()
  lane = axes["lane"]

  lane_stride_word_col = axis_stride(lane, axes["word_col"])
  lane_stride_block_group = axis_stride(lane, axes["block_group"])
  word_stride_word_col = axis_stride(word_idx, axes["word_col"])
  word_stride_block_group = axis_stride(word_idx, axes["block_group"])
  word_stride_local_block = axis_stride(word_idx, axes["local_block"])
  word_stride_group_pair = axis_stride(word_idx, axes["group_pair"])

  sample_rows = [0, 1, N - 1]
  sample_local_blocks = [0, BLOCKS_PER_GROUP - 1]
  sample_group_pairs = [0, GROUP_PAIRS - 1]
  mismatches = []
  samples = 0
  for row in sample_rows:
    for block_group in range(BLOCK_GROUPS):
      for local_block in sample_local_blocks:
        for group_pair in sample_group_pairs:
          for word_col in range(WORDS_PER_GROUP):
            values = {"row": row, "block_group": block_group, "local_block": local_block, "group_pair": group_pair, "word_col": word_col}
            got = eval_expr(word_idx, axes, values)
            exp = q4k_word_index_ref(row, block_group, local_block, group_pair, word_col)
            samples += 1
            if got != exp:
              mismatches.append({"values": values, "got": got, "expected": exp})

  checks = {
    "lane_stride_word_col_is_1": lane_stride_word_col == 1,
    "lane_stride_block_group_is_8": lane_stride_block_group == WORDS_PER_GROUP,
    "word_stride_word_col_is_1": word_stride_word_col == 1,
    "word_stride_block_group_is_blocks_per_group_times_36": word_stride_block_group == BLOCKS_PER_GROUP * Q4K_WORDS_PER_BLOCK,
    "word_stride_local_block_is_36": word_stride_local_block == Q4K_WORDS_PER_BLOCK,
    "word_stride_group_pair_is_8": word_stride_group_pair == WORDS_PER_GROUP,
    "word_col_coalesced": is_coalesced(word_idx, axes["word_col"]),
    "word_col_vector_width_4": vector_width(word_idx, axes["word_col"]) == 4,
    "sampled_index_equality": not mismatches,
  }

  verdict = "G2_REPRESENTATION_PROBE_PASS" if all(checks.values()) else "G2_REPRESENTATION_PROBE_FAIL"
  return {
    "date": "2026-06-25",
    "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "scope": "G2.0 static representation probe",
    "profile": {"model": "Qwen3-8B-Q4_K_M", "device": "AMD", "arch": "gfx1100", "phase": "decode T==1", "projection": "FFN gate/up", "shape": f"{K}x{N}", "wave": WARP},
    "lane_map": "lane = block_group * 8 + word_col",
    "packed_word_index": "(row * k_blocks + (block_group * blocks_per_group + local_block)) * 36 + 4 + group_pair * 8 + word_col",
    "constants": {"k": K, "n": N, "qk_k": QK_K, "k_blocks": K_BLOCKS, "block_groups": BLOCK_GROUPS, "blocks_per_group": BLOCKS_PER_GROUP, "words_per_group": WORDS_PER_GROUP, "q4k_words_per_block": Q4K_WORDS_PER_BLOCK, "group_pairs": GROUP_PAIRS},
    "strides": {"lane_wrt_word_col": lane_stride_word_col, "lane_wrt_block_group": lane_stride_block_group, "word_idx_wrt_word_col": word_stride_word_col, "word_idx_wrt_block_group": word_stride_block_group, "word_idx_wrt_local_block": word_stride_local_block, "word_idx_wrt_group_pair": word_stride_group_pair},
    "checks": checks,
    "samples_checked": samples,
    "mismatches": mismatches[:8],
    "decision": "The UOp/RANGE algebra can express the Q4_K gate/up lane map and packed-word index locally; proceed to G2.1 minimal Q4_K LaneMap object." if verdict == "G2_REPRESENTATION_PROBE_PASS" else "Static representation failed; do not proceed to runtime routing until the missing expression is classified.",
  }


def main() -> None:
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  result = run_probe()
  stamp_path = OUT_DIR / f"g2-representation-probe-{result['timestamp']}.json"
  latest_path = OUT_DIR / "latest.json"
  stamp_path.write_text(json.dumps(result, indent=2) + "\n")
  latest_path.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  if result["verdict"] != "G2_REPRESENTATION_PROBE_PASS":
    raise SystemExit(1)


if __name__ == "__main__":
  os.chdir(Path(__file__).resolve().parents[1])
  main()
