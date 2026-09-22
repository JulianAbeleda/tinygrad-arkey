#!/usr/bin/env python3
"""NV flash-geometry wall bracket: control / candidate / control, same session, bitwise-token gate.

The P3 search selected ``stage_width=4, reduce_structure=inline, dot_pair_width=4`` as the fastest
score body (3.968 us vs 4.19 us pinned control; nv-flash-geometry-search-20260819.json).  This harness
proves the route-level wall effect and token identity by leasing the searched tile geometry on the
model and every block (``_flash_decode_tile_geometry_lease``), then running a serialized reverse
bracket under one fresh process per arm.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
GEOMETRY = {"stage_width": 4, "reduce_structure": "inline", "dot_pair_width": 4}
TILE_PREFIX = "flash_block_tiled_xlane_score_pv_tile_whole_cache_"
PROMOTION_US = 50.0


def _install_lease(model, geometry: dict | None) -> None:
  setattr(model, "_flash_decode_tile_geometry_lease", geometry)
  for block in model.blk:
    setattr(block, "_flash_decode_tile_geometry_lease", geometry)


def _tile_names(model, depth: int) -> list[str]:
  """Capture the flash tile kernel names rendered during the first decode token."""
  from tinygrad.helpers import Context
  import tinygrad.llm.model as tgm
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  gen = model.generate([1] * depth, chunk_size=32, temperature=0.0)
  with Context(DEBUG=0):
    next(gen)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    with Context(DEBUG=2):
      next(gen)
  gen.close()
  names = sorted({line.split()[3] for line in buf.getvalue().splitlines()
                  if TILE_PREFIX in line})
  return names


def run_arm(arm: str, depth: int, nmeas: int, reps: int, geometry: dict | None = None) -> dict:
  if arm not in ("control", "candidate"):
    raise ValueError(f"unknown arm {arm!r}")
  from tinygrad import Device
  from tinygrad.helpers import Context
  from tinygrad.llm.model import Transformer
  import tinygrad.llm.model as tgm
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  model, _kv = Transformer.from_gguf(MODEL, 4608)
  _install_lease(model, geometry if arm == "candidate" else None)
  tile_names = _tile_names(model, depth)
  dev = Device[Device.DEFAULT]
  # Capture the decode graph once before any timed window so JIT/capture cost is not charged to tok/s.
  gen = model.generate([1] * depth, chunk_size=32, temperature=0.0)
  with Context(DEBUG=0):
    next(gen)
  dev.synchronize()
  gen.close()
  tok_s, shas, firsts = [], [], []
  for _ in range(reps):
    model.reset_generation_state()
    gen = model.generate([1] * depth, chunk_size=32, temperature=0.0)
    next(gen)
    dev.synchronize()
    lat, toks = [], []
    for _ in range(nmeas):
      t0 = time.perf_counter()
      toks.append(int(next(gen)))
      lat.append(time.perf_counter() - t0)
    gen.close()
    tok_s.append(nmeas / sum(lat))
    shas.append(hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest())
    firsts.append(toks[0])
  return {"arm": arm, "depth": depth, "nmeas": nmeas, "reps": reps,
          "tile_names": tile_names,
          "tok_s_median": statistics.median(tok_s), "tok_s_samples": tok_s,
          "token_sha_reps": shas, "first_token_reps": firsts,
          "geometry": geometry if arm == "candidate" else None}


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", choices=("control", "candidate"), required=True)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--nmeas", type=int, default=8)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--out", type=Path)
  ap.add_argument("--geometry-json", default=json.dumps(GEOMETRY),
                  help="JSON object for the candidate tile geometry override")
  args = ap.parse_args(argv)
  geometry = json.loads(args.geometry_json)
  result = run_arm(args.arm, args.depth, args.nmeas, args.reps, geometry)
  encoded = json.dumps(result, indent=2, sort_keys=True)
  print(encoded)
  if args.out:
    args.out.write_text(encoded + "\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
