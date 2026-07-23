#!/usr/bin/env python3
"""Opt-in compiler/resource probe for the phase_abi_v1 attention representative.

This is evidence collection only.  It neither imports nor modifies production routing.
"""
from __future__ import annotations

import argparse, json, traceback
from pathlib import Path

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.support.compiler_amd import compile_hip
from tinygrad.uop.ops import KernelInfo, Ops
from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention

from extra.qk.amdgpu_metadata import parse_amdgpu_metadata

SCHEMA = "tinygrad.shared_attention.phase_abi_v1_resource_probe.v1"


def _write(path: Path, value: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def probe() -> dict:
  row = {"schema": SCHEMA, "phase_abi": "phase_abi_v1", "target": "AMD:HIP:gfx1100",
         "geometry": {"q_tokens": 16, "q_heads": 4, "kv_heads": 2, "kv_tokens": 64, "head_dim": 128},
         "compile": {"status": "unavailable"}}
  try:
    q, hq, hkv, kv = 16, 4, 2, 64
    qi = Tensor.empty(hq*q*128, dtype=dtypes.half, device="AMD")
    ki = Tensor.empty(hkv*kv*128, dtype=dtypes.half, device="AMD")
    vi = Tensor.empty(hkv*kv*128, dtype=dtypes.half, device="AMD")
    out = Tensor.empty(hq*q*128, dtype=dtypes.half, device="AMD")
    def kernel(o, qv, kvv, vv):
      return amd_gfx1100_q16_grid_hd128_loop_attention(qv, kvv, vv, o, q_tokens=q, q_heads=hq, kv_heads=hkv,
        kv_tokens=kv, scale=1/(128**.5), causal=True, kernel_info=KernelInfo(name="phase_abi_v1_probe"), phase_abi_v1=True)
    scheduled = out.custom_kernel(qi, ki, vi, fxn=kernel)[0].schedule_linear()
    calls = [x for x in scheduled.src if x.op is Ops.CALL]
    if len(calls) != 1: raise RuntimeError(f"expected one phase ABI call, got {len(calls)}")
    program = to_program(calls[0].src[0], HIPRenderer(Target.parse("AMD:HIP:gfx1100")))
    source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
    binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
    row["compile"] = {"status": "PASS", "command": "tinygrad.runtime.support.compiler_amd.compile_hip",
                      "program_name": program.arg.name, "source_sha256": __import__("hashlib").sha256(source.encode()).hexdigest(),
                      "metadata": parse_amdgpu_metadata(binary)}
  except Exception as exc:
    row["compile"] = {"status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc(limit=4)}
  return row


def main() -> None:
  ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
  args = ap.parse_args(); _write(args.output, probe())


if __name__ == "__main__": main()
