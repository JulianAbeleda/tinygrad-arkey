#!/usr/bin/env python3
"""Runner for fma_peak_metal.py: for each of 3 precision variants (f16f16, f16f32, f32f32),
vector-width sweep, NACC sweep, grid-size sweep (plateau), disassembly verification. Dumps raw
JSON to /tmp so every number reported can be traced back to an actual run.

    python3 extra/llm_research/microbench/fma_peak_metal_run.py

Decides: is Apple's simdgroup_multiply_accumulate (R ~= 3781 GFLOPS, wmma_peak_metal.py) a separate
matrix unit, or does it lower onto the same ALUs plain FMA uses? The primary comparator is
"f16f32" (half x half -> float accumulate), because that is what simdgroup_float8x8 actually
computes -- fp16 operands, fp32 accumulate, not fp16 -> fp16. "f16f16" and "f32f32" are measured
alongside to (a) catch a dtype artifact and (b) show whether fp16 gets packed 2x over fp32.

Verdict rule: compare max(f16f16, f16f32, f32f32) against R, not any single variant in isolation.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp/extra/llm_research/microbench")

import fma_peak_metal as F

R_GFLOPS = 3781.3  # from wmma_peak_metal.py / README.md, same 10-core M4, same session family

WIDTHS = (1, 2, 4, 8)
NACC_LIST = (1, 2, 4, 8, 16)
GRID_BLOCK_COUNTS = (64, 256, 1024, 4096)
GRID_BLOCK_COUNTS_RECONFIRM = (1024, 4096, 8192, 16384)
TPB = 256
TARGET_TIME = 0.3


def run_variant(variant: str) -> dict:
  out = {}
  print(f"\n########## VARIANT {variant} ({F.VARIANTS[variant][0]} x {F.VARIANTS[variant][0]} -> "
        f"{F.VARIANTS[variant][1]}) ##########")

  print(f"\n===== [{variant}] STAGE 1: vector-width sweep, nacc=2, grid swept per width =====")
  width_sweep = {}
  for width in WIDTHS:
    rows = F.sweep_grid(nacc=2, width=width, variant=variant, tpb=TPB,
                         block_counts=list(GRID_BLOCK_COUNTS), warmup=2, reps=5, target_time=TARGET_TIME)
    width_sweep[width] = rows
  out["width_sweep"] = width_sweep

  best_per_width = {w: max(rows, key=lambda r: r["mean_gflops"]) for w, rows in width_sweep.items()}
  for w, r in best_per_width.items():
    print(f"  width={w}: best mean_gflops={r['mean_gflops']:.1f} at blocks={r['blocks']}")
  best_width = max(best_per_width, key=lambda w: best_per_width[w]["mean_gflops"])
  print(f"  best width @ nacc=2: {best_width} ({best_per_width[best_width]['mean_gflops']:.1f} GFLOPS)")

  print(f"\n===== [{variant}] STAGE 2: NACC sweep at width={best_width}, blocks={GRID_BLOCK_COUNTS[-1]} =====")
  nacc_rows = F.sweep_nacc(list(NACC_LIST), width=best_width, variant=variant,
                           blocks=GRID_BLOCK_COUNTS[-1], tpb=TPB, warmup=2, reps=5, target_time=TARGET_TIME)
  out["nacc_sweep"] = nacc_rows
  best_nacc_row = max(nacc_rows, key=lambda r: r["mean_gflops"])
  best_nacc = best_nacc_row["nacc"]
  print(f"  best nacc: {best_nacc} ({best_nacc_row['mean_gflops']:.1f} GFLOPS)")

  print(f"\n===== [{variant}] STAGE 3: re-confirm grid-size plateau at width={best_width}, nacc={best_nacc} =====")
  grid_rows2 = F.sweep_grid(nacc=best_nacc, width=best_width, variant=variant, tpb=TPB,
                            block_counts=list(GRID_BLOCK_COUNTS_RECONFIRM), warmup=2, reps=5,
                            target_time=TARGET_TIME)
  out["grid_sweep_reconfirm"] = grid_rows2
  best_plateau = max(grid_rows2, key=lambda r: r["mean_gflops"])
  print(f"  PLATEAU: {best_plateau['mean_gflops']:.1f} GFLOPS at width={best_width} nacc={best_nacc} "
        f"blocks={best_plateau['blocks']}")

  print(f"\n===== [{variant}] STAGE 3b: re-confirm width ranking at nacc={best_nacc}, blocks={best_plateau['blocks']} =====")
  width_recheck_rows = []
  for width in WIDTHS:
    width_recheck_rows += F.sweep_grid(nacc=best_nacc, width=width, variant=variant, tpb=TPB,
                                        block_counts=[best_plateau["blocks"]], warmup=2, reps=5,
                                        target_time=TARGET_TIME)
  out["width_recheck"] = width_recheck_rows
  best_overall = max(grid_rows2 + width_recheck_rows, key=lambda r: r["mean_gflops"])
  out["best_overall"] = best_overall
  print(f"  BEST OVERALL for {variant}: {best_overall['mean_gflops']:.1f} GFLOPS "
        f"(width={best_overall['width']} nacc={best_overall['nacc']} blocks={best_overall['blocks']})")
  return out


def main():
  all_out = {}
  for variant in ("f16f16", "f16f32", "f32f32"):
    all_out[variant] = run_variant(variant)

  print("\n===== STAGE 4: disassembly verification (xcrun metal -fno-fast-math -c && metal-objdump) =====")
  dis_results = {}
  for variant in ("f16f16", "f16f32", "f32f32"):
    bo = all_out[variant]["best_overall"]
    dis = F.disassemble_check(bo["nacc"], bo["width"], variant)
    key = f"{variant}_w{bo['width']}_n{bo['nacc']}"
    entry = {"ok": dis["ok"]}
    if dis["ok"]:
      text = dis["disassembly"]
      counts = {}
      for kw in ["fma", "call", "load", "store", "getelementptr", "simdgroup", "fmul", "fadd",
                 "fast", "fpext", "fptrunc", "f16", "f32"]:
        counts[kw] = text.count(kw)
      entry["counts"] = counts
      entry["air_len"] = dis["air_len"]
      print(f"{key}: air_len={dis['air_len']} mnemonic counts: {counts}")
      dis_path = f"/tmp/fma_peak_metal_disassembly_{key}.txt"
      with open(dis_path, "w") as f:
        f.write(text)
      entry["dis_path"] = dis_path
      print(f"  full disassembly written to {dis_path}")
      src_path = f"/tmp/fma_peak_metal_src_{key}.metal"
      with open(src_path, "w") as f:
        f.write(dis["src"])
      entry["src_path"] = src_path
    else:
      print(f"{key}: DISASSEMBLY FAILED at stage {dis['stage']}: {dis['stderr']}")
    dis_results[key] = entry
  all_out["disassembly"] = dis_results

  with open("/tmp/fma_peak_metal_result.json", "w") as f:
    json.dump(all_out, f, indent=2)
  print("\nfull result JSON written to /tmp/fma_peak_metal_result.json")

  print("\n===== SUMMARY =====")
  print(f"R (simdgroup_multiply_accumulate, from wmma_peak_metal.py) = {R_GFLOPS:.1f} GFLOPS")
  bests = {}
  for variant in ("f16f16", "f16f32", "f32f32"):
    bo = all_out[variant]["best_overall"]
    bests[variant] = bo["mean_gflops"]
    print(f"  {variant}: plateau = {bo['mean_gflops']:.1f} GFLOPS (width={bo['width']} nacc={bo['nacc']} "
          f"blocks={bo['blocks']}) -- {bo['mean_gflops']/R_GFLOPS:.3f}x R")
  max_variant = max(bests, key=lambda v: bests[v])
  print(f"\nmax over variants: {max_variant} = {bests[max_variant]:.1f} GFLOPS "
        f"= {bests[max_variant]/R_GFLOPS:.3f}x R")
  if bests["f32f32"] > 0:
    print(f"f16f32 / f32f32 ratio (packed-math check): {bests['f16f32']/bests['f32f32']:.3f}x")
    print(f"f16f16 / f32f32 ratio: {bests['f16f16']/bests['f32f32']:.3f}x")


if __name__ == "__main__":
  main()
