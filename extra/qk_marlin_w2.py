#!/usr/bin/env python3
"""W2: parametrize the W1b' Marlin fused-dequant->WMMA kernel for real throughput.

W2.0 -- grid parallelism: block the output over M-rows into one workgroup per BLOCK_M tile (whole N,
whole K per workgroup). The W1b' kernel ran as ONE workgroup, hence tiny TFLOPS; this should scale
~linearly with workgroup count. Keeps whole-K-in-LDS (BLOCK_M*K*2 <= ~64KB), so K<=2048 for BLOCK_M=16.

Run: DEV=AMD DEBUG=2 PYTHONPATH=. .venv/bin/python extra/qk_marlin_w2.py --gate w20
"""
from __future__ import annotations
import os
os.environ.setdefault("TC", "1")

import argparse, json, pathlib, sys
from tinygrad import Tensor, dtypes, Device
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import GlobalCounters
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, UOp
from extra.q4_k_gemv_primitive import _q4k_weight
from extra.qk_layout import (read_metadata, pick_tensor, tensor_shape, q4_k_reference,
                             Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS)

TC_OPT_ARG = (-1, 2, 1)
MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf").expanduser()
PEAK_TFLOPS = 83.64
ART = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/wmma-w2")


def _ci(v:int) -> UOp: return UOp.const(dtypes.weakint, v)


def marlin_grid_kernel(BLOCK_M:int, M:int, K:int, N:int, opts:tuple[Opt, ...]):
  """Grid over M-rows: one workgroup per BLOCK_M tile, computing BLOCK_M x N over the full K.
  Each workgroup dequants its BLOCK_M x K weight tile ONCE into LDS, barriers, then WMMA over N."""
  k_blocks = K // Q4_K_BLOCK_ELEMS
  n_blocks = M // BLOCK_M
  def kernel(out:UOp, words:UOp, B:UOp) -> UOp:
    block_m = UOp.range(n_blocks, 3, AxisType.GLOBAL)
    Alds = UOp.placeholder((BLOCK_M*K,), dtypes.float16, slot=0, addrspace=AddrSpace.LOCAL)
    # --- dequant-stage this block's BLOCK_M rows x K weights into LDS (once) ---
    rl = UOp.range(BLOCK_M, 10, AxisType.LOOP)
    blk = UOp.range(k_blocks, 11, AxisType.LOOP)
    pos = UOp.range(32, 12, AxisType.LOOP)
    grow = block_m*_ci(BLOCK_M) + rl
    base = (grow * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    stores = []
    for grp in range(8):
      kidx = blk*_ci(Q4_K_BLOCK_ELEMS) + _ci(grp*32) + pos
      w = _q4k_weight(words, base, grp, pos).cast(dtypes.float16)
      stores.append(Alds.index(rl*_ci(K) + kidx, ptr=True).store(w))
    Alds = Alds.after(UOp.barrier(UOp.group(*stores).end(rl, blk, pos)))
    # --- matmul: BLOCK_M x N over K ---
    m = UOp.range(BLOCK_M, 1, AxisType.LOOP)
    n = UOp.range(N, 2, AxisType.LOOP)
    k = UOp.range(K, 0, AxisType.REDUCE)
    mul = (Alds.index(m*_ci(K)+k) * B.flatten().index(k*_ci(N)+n)).cast(dtypes.float32)
    red = mul.reduce(k, arg=Ops.ADD, dtype=dtypes.float32).cast(out.dtype.base)
    grow_o = block_m*_ci(BLOCK_M) + m
    store = out.flatten().index(grow_o*_ci(N)+n, ptr=True).store(red).end(m, n)
    return store.end(block_m).sink(arg=KernelInfo(name=f"marlin_w20_{M}_{K}_{N}_bm{BLOCK_M}", opts_to_apply=opts))
  return kernel


def ceiling_grid_kernel(BLOCK_M:int, M:int, K:int, N:int, opts:tuple[Opt, ...]):
  """Same grid structure with a pre-dequanted fp16 weight (the materialized-fp16 WMMA ceiling)."""
  n_blocks = M // BLOCK_M
  def kernel(out:UOp, W:UOp, B:UOp) -> UOp:
    block_m = UOp.range(n_blocks, 3, AxisType.GLOBAL)
    Alds = UOp.placeholder((BLOCK_M*K,), dtypes.float16, slot=0, addrspace=AddrSpace.LOCAL)
    rl = UOp.range(BLOCK_M, 10, AxisType.LOOP)
    ck = UOp.range(K, 11, AxisType.LOOP)
    grow = block_m*_ci(BLOCK_M) + rl
    store_lds = Alds.index(rl*_ci(K)+ck, ptr=True).store(W.flatten().index(grow*_ci(K)+ck)).end(rl, ck)
    Alds = Alds.after(UOp.barrier(store_lds))
    m = UOp.range(BLOCK_M, 1, AxisType.LOOP)
    n = UOp.range(N, 2, AxisType.LOOP)
    k = UOp.range(K, 0, AxisType.REDUCE)
    mul = (Alds.index(m*_ci(K)+k) * B.flatten().index(k*_ci(N)+n)).cast(dtypes.float32)
    red = mul.reduce(k, arg=Ops.ADD, dtype=dtypes.float32).cast(out.dtype.base)
    grow_o = block_m*_ci(BLOCK_M) + m
    store = out.flatten().index(grow_o*_ci(N)+n, ptr=True).store(red).end(m, n)
    return store.end(block_m).sink(arg=KernelInfo(name=f"ceil_w20_{M}_{K}_{N}_bm{BLOCK_M}", opts_to_apply=opts))
  return kernel


def _load(M:int, K:int, tensor:str):
  meta = read_metadata(MODEL); info = pick_tensor(meta.infos, tensor); rows, Kfull = tensor_shape(info)
  k_blocks_full = Kfull // Q4_K_BLOCK_ELEMS
  nb = K // Q4_K_BLOCK_ELEMS
  assert M <= rows and nb <= k_blocks_full
  bs = meta.data_start + info.off
  full = Tensor(MODEL)[bs:bs + M*k_blocks_full*Q4_K_BLOCK_BYTES].to("AMD").realize()
  raw = full.reshape(M, k_blocks_full, Q4_K_BLOCK_BYTES)[:, :nb, :].flatten().contiguous().realize()
  words = raw.bitcast(dtypes.uint32).realize()
  wf16 = q4_k_reference(raw, M*K).reshape(M, K).cast(dtypes.float16).realize()
  return words, wf16


def _time(fn, warmup=5, iters=30):
  for _ in range(warmup): fn().realize()
  Device[Device.DEFAULT].synchronize()
  ts = []
  for _ in range(iters):
    GlobalCounters.reset(); fn().realize(); Device[Device.DEFAULT].synchronize()
    ts.append(GlobalCounters.time_sum_s)
  ts.sort(); return ts[len(ts)//2]


def measure(M:int, K:int, N:int, BLOCK_M:int, tensor:str):
  words, wf16 = _load(M, K, tensor)
  Tensor.manual_seed(1337)
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ref = (wf16.cast(dtypes.float32) @ B.cast(dtypes.float32)).realize()
  opts = (Opt(OptOps.TC, 1, TC_OPT_ARG),)
  out = Tensor.empty(M, N, dtype=dtypes.float32, device="AMD")
  marlin = lambda: Tensor.custom_kernel(out, words, B, fxn=marlin_grid_kernel(BLOCK_M, M, K, N, opts))[0]
  ceiling = lambda: Tensor.custom_kernel(out, wf16, B, fxn=ceiling_grid_kernel(BLOCK_M, M, K, N, opts))[0]
  rel_m = (marlin().realize() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
  rel_c = (ceiling().realize() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
  t_m, t_c = _time(marlin), _time(ceiling)
  flops = 2*M*K*N
  return {
    "shape": {"M": M, "K": K, "N": N}, "BLOCK_M": BLOCK_M, "workgroups": M//BLOCK_M, "lds_bytes": BLOCK_M*K*2,
    "marlin_correct": rel_m < 1e-2, "ceiling_correct": rel_c < 1e-2,
    "marlin_us": round(t_m*1e6, 2), "ceiling_us": round(t_c*1e6, 2),
    "marlin_tflops": round(flops/t_m/1e12, 3), "ceiling_tflops": round(flops/t_c/1e12, 3),
    "marlin_pct_peak": round(flops/t_m/1e12/PEAK_TFLOPS*100, 2),
    "marlin_vs_ceiling": round(t_c/t_m, 3),
  }


def run_w20(tensor:str):
  # grid over M (BLOCK_M=16), whole N, whole K (<=2048 to fit LDS). M large -> many workgroups.
  shapes = [(256, 1024, 512), (1024, 1024, 512), (4096, 1024, 512), (4096, 2048, 256), (4096, 1024, 2048)]
  curve = [measure(M, K, N, 16, tensor) for (M, K, N) in shapes]
  out = {"kind": "qk_marlin_w2", "phase": "Phase W2.0", "tensor": tensor, "peak_tflops": PEAK_TFLOPS,
         "lever": "grid parallelism over M-rows (BLOCK_M=16), whole N + whole K per workgroup",
         "curve": curve}
  ART.mkdir(parents=True, exist_ok=True)
  (ART / "w20_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print(json.dumps(out, indent=2, sort_keys=True), file=sys.__stdout__)
  return all(c["marlin_correct"] and c["ceiling_correct"] for c in curve)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--gate", default="w20", choices=["w20"])
  p.add_argument("--tensor", default="blk.20.attn_q.weight")
  args = p.parse_args()
  return 0 if run_w20(args.tensor) else 1


if __name__ == "__main__":
  raise SystemExit(main())
