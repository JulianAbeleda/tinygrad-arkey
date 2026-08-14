#!/usr/bin/env python3
"""Device-side (CUPTI/nsys) timing of the decode flash score kernels.

The python-loop microgate (``nv_flash_vec_llama_microgate.py``) timed
``fn(q,cache).realize()`` in a Python loop.  That is dispatch-dominated
(~65 us/graph of Python + launch, against a ~3-8 us kernel body), so it
cannot resolve a body difference.  This probe instead launches each score
kernel back-to-back on the GPU and lets ``nsys`` report the true CUPTI
``CUPTI_ACTIVITY_KIND_KERNEL`` duration, the same metric as the pinned
llama/tinygrad node ledgers.

It also closes the config trap in the old microgate: that file compared the
research tile at S=4 against the vec at S=4, but *production* decode runs the
tile at S=48 (384 blocks) while llama runs S=4 (32 blocks).  So the old "flat"
was comparing two S=4 bodies, not the production body.  This probe measures
all three.
"""
from __future__ import annotations

import argparse, json, subprocess, time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tinygrad import Device, Tensor, dtypes
from tinygrad.llm.flash_decode_attention import (
  flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel,
  flash_vec_llama_score_pv_kernel)
from tinygrad.llm.kernel_program import (
  KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program)
from tinygrad.uop.ops import UOp

Hq, Hkv, Hd, MAXC, Tc = 32, 8, 128, 4608, 513
W = Hd + 2


def _score_program(name: str, emitter, S: int) -> KernelProgram:
  return KernelProgram("research.nv_flash_body_device_timing", name,
                       KernelProgramProvenance.RESEARCH_ONLY, emitter,
                       output_spec=OutputSpec((Hq * S * W,), dtypes.float32))


def build_programs() -> dict[str, tuple[KernelProgram, int, int]]:
  tc = UOp.const(dtypes.int, Tc)
  return {
    # Production decode route: FLASH_DECODE_G4 -> S=48, per-split length 16 at Tc=513.
    "tile_s48_prod": (_score_program("tile_s48_prod",
      flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, 16, 48, tc, stage_width=1), 48),
      48, MAXC),
    # The config the old microgate actually compared.
    "tile_s4": (_score_program("tile_s4",
      flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, 144, 4, tc, stage_width=1), 4),
      4, MAXC),
    # llama's single-pass substrate (llama uses S=4 at context 512).
    "vec_s4": (_score_program("vec_s4",
      flash_vec_llama_score_pv_kernel(Hd, Hq, Hkv, MAXC, 4, tc), 4),
      4, MAXC),
    # Loop-bound control: NCHUNK=ceil(1024/512)=2 instead of ceil(4608/512)=9.
    "vec_s4_tcbound": (_score_program("vec_s4_tcbound",
      flash_vec_llama_score_pv_kernel(Hd, Hq, Hkv, 1024, 4, tc), 4),
      4, 1024),
  }


def _inputs(maxc: int = MAXC):
  rng = np.random.default_rng(20260813)
  q = rng.normal(0, .2, Hq * Hd).astype(np.float16)
  cache = rng.normal(0, .2, (2, 1, Hkv, maxc, Hd)).astype(np.float16)
  return (Tensor(q, device="CUDA").contiguous().realize(),
          Tensor(cache, device="CUDA").contiguous().realize())


def run(replays: int = 400, warmup: int = 20, only: str | None = None) -> dict:
  if Device.DEFAULT != "CUDA":
    raise RuntimeError(f"DEV=CUDA required for CUPTI timing, got {Device.DEFAULT}")
  programs = build_programs()
  if only is not None:
    if only not in programs:
      raise ValueError(f"unknown kernel {only!r}; choose from {sorted(programs)}")
    programs = {only: programs[only]}
  out: dict = {"schema": "tinygrad.nv_flash_body_device_timing.v1",
               "device": str(Device.DEFAULT),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "Tc": Tc},
               "replays": replays, "warmup": warmup, "runs": {}}
  for name, (program, S, maxc) in programs.items():
    q, cache = _inputs(maxc)
    dst = Tensor.empty(Hq * S * W, dtype=dtypes.float32, device="CUDA")
    # Warm up and validate numerics once.
    res = execute_research_program(dst, q, cache, program=program)
    res.realize(); Device["CUDA"].synchronize()
    arr = np.asarray(res.numpy()).reshape(Hq * S, W)
    # The per-split max lane legitimately stays -inf for fully-masked splits; only PV+den must be finite.
    finite = bool(np.isfinite(arr[:, :Hd]).all() and np.isfinite(arr[:, Hd]).all())
    for _ in range(warmup):
      execute_research_program(dst, q, cache, program=program).realize()
    Device["CUDA"].synchronize()
    # Back-to-back launches for the profiler; one sync at the end.
    for _ in range(replays):
      execute_research_program(dst, q, cache, program=program).realize()
    Device["CUDA"].synchronize()
    out["runs"][name] = {"S": S, "MAXC": maxc, "NCHUNK": -(-maxc // (S * 128)), "finite": finite, "replays": replays,
                         "nsys_marker": f"START_{name}_x{replays}_END"}
    # A tiny sleep separates the three loops so the nsys timeline is unambiguous.
    time.sleep(0.05)
  return out


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--replays", type=int, default=400)
  ap.add_argument("--warmup", type=int, default=20)
  ap.add_argument("--only", choices=("tile_s48_prod", "tile_s4", "vec_s4", "vec_s4_tcbound"))
  ap.add_argument("--out", type=Path)
  args = ap.parse_args()
  got = run(args.replays, args.warmup, args.only)
  print(json.dumps(got, indent=2, sort_keys=True))
  if args.out:
    args.out.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
