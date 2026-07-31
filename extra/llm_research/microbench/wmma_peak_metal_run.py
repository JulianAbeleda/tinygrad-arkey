#!/usr/bin/env python3
"""Runner for wmma_peak_metal.py: grid-size sweep, NACC sweep, disassembly verification.
Dumps raw JSON to /tmp so every number reported can be traced back to an actual run.

    python3 extra/llm_research/microbench/wmma_peak_metal_run.py
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp/extra/llm_research/microbench")

import wmma_peak_metal as W


def main():
  out = {}

  print("===== STAGE 1: grid-size sweep (find where simdgroup count stops raising throughput) =====")
  print("nacc=8, tpb=256 (8 simdgroups/threadgroup, matches AMD wave32 1:1), iters=4000")
  block_counts = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
  grid_rows = W.sweep_grid(nacc=8, tpb=256, iters=4000, block_counts=block_counts, warmup=2, reps=5)
  out["grid_sweep"] = grid_rows

  best_grid = max(grid_rows, key=lambda r: r["mean_gflops"])
  plateau_blocks = best_grid["blocks"]
  print(f"\nbest mean_gflops in grid sweep: {best_grid['mean_gflops']:.1f} at blocks={plateau_blocks} "
        f"simdgroups={best_grid['simdgroups']}")

  print("\n===== STAGE 2: NACC sweep at the plateaued grid size =====")
  # iters chosen per-nacc so each run does comparable *total* wall time (larger NACC does more
  # FLOP/iter, so fewer iters needed) -- keeps every run in the same ~tens-of-ms timing regime.
  nacc_list = [2, 4, 8, 16]
  iters_for_nacc = {2: 16000, 4: 8000, 8: 4000, 16: 2000}
  nacc_rows = W.sweep_nacc(nacc_list, blocks=plateau_blocks, tpb=256, iters_for_nacc=iters_for_nacc,
                           warmup=2, reps=5)
  out["nacc_sweep"] = nacc_rows

  best_nacc_row = max(nacc_rows, key=lambda r: r["mean_gflops"])
  print(f"\nbest mean_gflops in nacc sweep: {best_nacc_row['mean_gflops']:.1f} at nacc={best_nacc_row['nacc']}")

  # confirm the grid-size plateau still holds at the winning NACC (in case the right NACC needs a
  # different occupancy point -- do not assume the nacc=8 plateau transfers). Pushed out to 32768
  # blocks (262144 simdgroups) because nacc=2/1 need a bigger grid than nacc=8 to fully saturate.
  print(f"\n===== STAGE 3: re-confirm grid-size plateau at nacc={best_nacc_row['nacc']} =====")
  block_counts2 = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
  grid_rows2 = W.sweep_grid(nacc=best_nacc_row["nacc"], tpb=256, iters=iters_for_nacc[best_nacc_row["nacc"]],
                            block_counts=block_counts2, warmup=2, reps=5)
  out["grid_sweep_at_best_nacc"] = grid_rows2

  # nacc=1 sanity: true back-to-back dependency, measures latency not throughput (per task spec).
  print("\n===== STAGE 3b: nacc=1 sanity check at the same plateaued grid point =====")
  plateau_blocks2 = grid_rows2[-1]["blocks"]
  nacc1_rows = W.sweep_nacc([1], blocks=plateau_blocks2, tpb=256,
                            iters_for_nacc={1: iters_for_nacc[best_nacc_row["nacc"]]}, warmup=2, reps=5)
  out["nacc1_sanity"] = nacc1_rows

  # threadgroup-shape sensitivity at matched total simdgroup count.
  print("\n===== STAGE 3c: tpb sensitivity at matched total simdgroup count =====")
  target_simdgroups = grid_rows2[-1]["simdgroups"]
  tpb_rows = []
  for tpb in (32, 128, 256, 512, 1024):
    blocks = target_simdgroups // (tpb // W.SIMD_WIDTH)
    tpb_rows += W.sweep_grid(nacc=best_nacc_row["nacc"], tpb=tpb, iters=iters_for_nacc[best_nacc_row["nacc"]],
                             block_counts=[blocks], warmup=2, reps=5)
  out["tpb_sensitivity"] = tpb_rows

  best_overall = max(grid_rows + nacc_rows + grid_rows2 + nacc1_rows + tpb_rows, key=lambda r: r["mean_gflops"])
  print(f"\nBEST OVERALL mean_gflops = {best_overall['mean_gflops']:.1f} "
        f"(nacc={best_overall['nacc']} blocks={best_overall['blocks']} simdgroups={best_overall['simdgroups']})")
  out["best_overall"] = best_overall

  print("\n===== STAGE 4: disassembly verification (xcrun metal -c && metal-objdump) =====")
  dis = W.disassemble_check(best_overall["nacc"])
  out["disassembly"] = {"ok": dis["ok"]}
  if dis["ok"]:
    text = dis["disassembly"]
    n_lines = text.count("\n")
    # count mnemonics of interest across the whole disassembly (loop body is not separately
    # delimited in AIR/LLVM-IR-level disassembly, so this is a whole-function census -- see report)
    counts = {}
    for kw in ["simdgroup_multiply_accumulate", "call", "load", "store", "getelementptr"]:
      counts[kw] = text.count(kw)
    out["disassembly"]["counts"] = counts
    out["disassembly"]["air_len"] = dis["air_len"]
    print(f"air_len={dis['air_len']} lines={n_lines}")
    print(f"mnemonic counts: {counts}")
    dis_path = "/tmp/wmma_peak_metal_disassembly.txt"
    with open(dis_path, "w") as f:
      f.write(text)
    print(f"full disassembly written to {dis_path}")
  else:
    print(f"DISASSEMBLY FAILED at stage {dis['stage']}: {dis['stderr']}")

  with open("/tmp/wmma_peak_metal_result.json", "w") as f:
    json.dump(out, f, indent=2)
  print("\nfull result JSON written to /tmp/wmma_peak_metal_result.json")


if __name__ == "__main__":
  main()
