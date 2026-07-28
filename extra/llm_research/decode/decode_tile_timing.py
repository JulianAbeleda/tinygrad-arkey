#!/usr/bin/env python3
"""Synchronized direct flash-decode tile replay (AMD timing harness).

This is intentionally separate from the production model W/D authority: it
allocates the exact executor tensors and measures only the captured tile path.
"""
from __future__ import annotations
import argparse, json, os, re, statistics, time
from pathlib import Path

import numpy as np

H_KV, H_D, MAX_CONTEXT, DEFAULT_SPLITS = 8, 128, 4608, 48

def normalize_config(hq: int, tc: int, splits: int = DEFAULT_SPLITS) -> dict[str, int]:
  hq, tc, splits = int(hq), int(tc), int(splits)
  if hq not in (32, 40): raise ValueError("Hq must be 32 or 40")
  if tc <= 0 or tc > MAX_CONTEXT: raise ValueError("Tc must be in 1..MAX_CONTEXT")
  if splits <= 0: raise ValueError("splits must be positive")
  if hq % H_KV: raise ValueError("Hq must be divisible by Hkv")
  aligned = ((tc + splits - 1) // splits + 15) // 16 * 16
  blocks = aligned // 16
  return {"Hq": hq, "Hkv": H_KV, "Hd": H_D, "MAXC": MAX_CONTEXT, "Tc": tc,
          "S": splits, "blocks_per_split": blocks, "tiles": H_KV * splits * blocks}

def power_state() -> dict[str, str | None]:
  paths = ["/sys/class/drm/card0/device/power_dpm_force_performance_level",
           "/sys/class/drm/card0/device/pp_dpm_sclk", "/sys/class/drm/card0/device/pp_dpm_mclk"]
  return {Path(p).name: (Path(p).read_text().strip() if Path(p).exists() else None) for p in paths}

def run(hq: int, tc: int, *, splits: int = DEFAULT_SPLITS, warmups: int = 3, samples: int = 15) -> dict:
  cfg = normalize_config(hq, tc, splits)
  os.environ.setdefault("DEV", "AMD")
  from tinygrad import Device, Tensor, TinyJit
  from extra.llm_research.decode.flash_decode_attention_executor import flash_decode_live_split_block_tile
  rng = np.random.default_rng(20260726 + cfg["Hq"] + cfg["Tc"])
  q = Tensor(rng.normal(0, .04, (1, cfg["Hq"], 1, H_D)).astype(np.float16), device="AMD")
  cache_np = np.zeros((2, 1, H_KV, MAX_CONTEXT, H_D), dtype=np.float16)
  cache_np[:, :, :, :cfg["Tc"], :] = rng.normal(0, .04, (2, 1, H_KV, cfg["Tc"], H_D)).astype(np.float16)
  cache = Tensor(cache_np, device="AMD")
  fn = TinyJit(lambda: flash_decode_live_split_block_tile(q, cache, cfg["Tc"], H_D, cfg["Hq"], H_KV,
                                                           MAX_CONTEXT, cfg["S"], staging="KV_BOTH", fused_combine=True))
  out = fn().numpy().astype(np.float32)
  for _ in range(warmups): fn().realize()
  dev = Device["AMD"]
  dev.synchronize()
  linear = str(fn.captured.linear)
  names = sorted(set(re.findall(r"(?:flash_block_tiled_xlane_score_pv_tile_whole_cache|flash_fused_gmax_combine)_\\d+(?:_\\d+)?", linear)))
  expected = [f"flash_block_tiled_xlane_score_pv_tile_whole_cache_{cfg['Hq']}_{H_D}", "flash_fused_gmax_combine"]
  matches = [x for x in expected if x in linear]
  raw = []
  for _ in range(samples):
    dev.synchronize(); started = time.perf_counter_ns(); fn().realize(); dev.synchronize()
    raw.append((time.perf_counter_ns() - started) / 1e6)
  med = statistics.median(raw)
  return {**cfg, "warmups": warmups, "samples": samples, "jit_runner_count": len(names),
          "jit_runner_names": names, "expected_match_count": len(matches), "raw_ms": raw,
          "median_ms": med, "ns_per_16tok_kv_tile": med * 1e6 / cfg["tiles"],
          "ns_per_tile_per_query_head": med * 1e6 / cfg["tiles"] / cfg["Hq"],
          "output_finite": bool(np.isfinite(out).all()), "output_nonzero": bool(np.any(out != 0)),
          "power_state": power_state()}

def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--Hq", type=int, choices=(32, 40), required=True)
  ap.add_argument("--Tc", type=int, choices=(512, 4096), required=True)
  ap.add_argument("--splits", type=int, default=DEFAULT_SPLITS)
  ap.add_argument("--warmups", type=int, default=3); ap.add_argument("--samples", type=int, default=15)
  args = ap.parse_args(argv)
  print(json.dumps(run(args.Hq, args.Tc, splits=args.splits,
                       warmups=args.warmups, samples=args.samples), sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
