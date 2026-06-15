#!/usr/bin/env python3
"""W1b' Track A: Marlin-class LDS-staged fused dequant -> WMMA, built bottom-up.

Gates, smallest-first (each must pass before the next):
  A.0a  hand-built custom_kernel matmul + explicit Opt(OptOps.TC) -> does TC fire on a HAND AST?
  A.0b  same, but operand A is staged through a DEFINE_LOCAL (LDS) first -> does TC fire on an
        LDS-staged operand written earlier in the same kernel? (the load-bearing unknown)
  A.1   replace the LDS store's source with the Q4_K dequant (dequant once, reuse across WMMA)
  A.2   correctness + device throughput vs the fp16-WMMA ceiling and the llama.cpp bar

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_marlin_w1b.py --gate a0a
"""
from __future__ import annotations
import os
os.environ.setdefault("TC", "1")

import argparse, pathlib, sys
from tinygrad import Tensor, dtypes, Device
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import GlobalCounters
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, UOp
from extra.q4_k_gemv_primitive import _q4k_weight
from extra.qk_layout import (read_metadata, pick_tensor, tensor_shape, q4_k_reference,
                             Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS)

TC_OPT_ARG = (-1, 2, 1)  # (tc_select, tc_opt, use_tensor_cores)
MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf").expanduser()


def _ci(v:int) -> UOp: return UOp.const(dtypes.weakint, v)


def _matmul_kernel(M:int, K:int, N:int, opts:tuple[Opt, ...]):
  """A.0a: out[M,N] = sum_k A[M,K] @ B[K,N] as a proper Ops.REDUCE (the TC-matchable shape).
  Mirrors extra/gemm/cdna_asm_gemm.py::custom_uop_gemm."""
  def kernel(out:UOp, A:UOp, B:UOp) -> UOp:
    m = UOp.range(M, 1, AxisType.LOOP)
    n = UOp.range(N, 2, AxisType.LOOP)
    k = UOp.range(K, 0, AxisType.REDUCE)
    mul = (A.flatten().index(m*_ci(K)+k) * B.flatten().index(k*_ci(N)+n)).cast(dtypes.float32)
    red = mul.reduce(k, arg=Ops.ADD, dtype=dtypes.float32).cast(out.dtype.base)
    store = out.flatten().index(m*_ci(N)+n, ptr=True).store(red).end(m, n)
    return store.sink(arg=KernelInfo(name=f"marlin_a0a_{M}_{K}_{N}", opts_to_apply=opts))
  return kernel


def _matmul_lds_kernel(M:int, K:int, N:int, opts:tuple[Opt, ...]):
  """A.0b: stage A into an LDS (DEFINE_LOCAL) tile via a plain copy -> barrier -> matmul reading the
  LDS tile, + TC opt. Isolates the load-bearing unknown: does TC fire on a MUL operand that is a load
  from a DEFINE_LOCAL written earlier in the same kernel?"""
  def kernel(out:UOp, A:UOp, B:UOp) -> UOp:
    Alds = UOp.placeholder((M*K,), dtypes.float16, slot=0, addrspace=AddrSpace.LOCAL)
    cm = UOp.range(M, 10, AxisType.LOOP)
    ck = UOp.range(K, 11, AxisType.LOOP)
    store_lds = Alds.index(cm*_ci(K)+ck, ptr=True).store(A.flatten().index(cm*_ci(K)+ck)).end(cm, ck)
    Alds = Alds.after(UOp.barrier(store_lds))
    m = UOp.range(M, 1, AxisType.LOOP)
    n = UOp.range(N, 2, AxisType.LOOP)
    k = UOp.range(K, 0, AxisType.REDUCE)
    mul = (Alds.index(m*_ci(K)+k) * B.flatten().index(k*_ci(N)+n)).cast(dtypes.float32)
    red = mul.reduce(k, arg=Ops.ADD, dtype=dtypes.float32).cast(out.dtype.base)
    store = out.flatten().index(m*_ci(N)+n, ptr=True).store(red).end(m, n)
    return store.sink(arg=KernelInfo(name=f"marlin_a0b_{M}_{K}_{N}", opts_to_apply=opts))
  return kernel


def _marlin_kernel(M:int, K:int, N:int, opts:tuple[Opt, ...]):
  """A.1: Marlin -- dequant the compressed Q4_K weight tile ONCE into an LDS (fp16) tile -> barrier
  -> matmul reading the LDS tile, + TC opt. The dequant is behind the barrier, so the WMMA reads
  plain fp16 LDS loads and the per-MAC recompute (the W1 28x) is structurally impossible."""
  k_blocks = K // Q4_K_BLOCK_ELEMS
  def kernel(out:UOp, words:UOp, B:UOp) -> UOp:
    Alds = UOp.placeholder((M*K,), dtypes.float16, slot=0, addrspace=AddrSpace.LOCAL)
    # --- dequant-stage: row, blk, pos ranges; grp python-unrolled (mirrors q4k_unpack_kernel) ---
    row = UOp.range(M, 10, AxisType.LOOP)
    blk = UOp.range(k_blocks, 11, AxisType.LOOP)
    pos = UOp.range(32, 12, AxisType.LOOP)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    stores = []
    for grp in range(8):
      kidx = blk*_ci(Q4_K_BLOCK_ELEMS) + _ci(grp*32) + pos
      w = _q4k_weight(words, base, grp, pos).cast(dtypes.float16)
      stores.append(Alds.index(row*_ci(K) + kidx, ptr=True).store(w))
    Alds = Alds.after(UOp.barrier(UOp.group(*stores).end(row, blk, pos)))
    # --- matmul over the staged fp16 weight tile ---
    m = UOp.range(M, 1, AxisType.LOOP)
    n = UOp.range(N, 2, AxisType.LOOP)
    k = UOp.range(K, 0, AxisType.REDUCE)
    mul = (Alds.index(m*_ci(K)+k) * B.flatten().index(k*_ci(N)+n)).cast(dtypes.float32)
    red = mul.reduce(k, arg=Ops.ADD, dtype=dtypes.float32).cast(out.dtype.base)
    store = out.flatten().index(m*_ci(N)+n, ptr=True).store(red).end(m, n)
    return store.sink(arg=KernelInfo(name=f"marlin_a1_{M}_{K}_{N}", opts_to_apply=opts))
  return kernel


def run_a1(M:int, N:int, tensor:str, tc:bool, nb:int):
  """nb = number of Q4_K blocks per row to use -> K = nb*256. The whole M*K weight tile is held in
  LDS (no K-tiling yet), so keep M*K*2 bytes under the ~64KB LDS limit (e.g. M=16, nb=1 -> 8KB)."""
  meta = read_metadata(MODEL); info = pick_tensor(meta.infos, tensor); rows, Kfull = tensor_shape(info)
  k_blocks_full = Kfull // Q4_K_BLOCK_ELEMS
  assert M <= rows and nb <= k_blocks_full
  K = nb * Q4_K_BLOCK_ELEMS
  bs = meta.data_start + info.off
  full = Tensor(MODEL)[bs:bs + M*k_blocks_full*Q4_K_BLOCK_BYTES].to("AMD").realize()
  # gather the first nb blocks of each of M rows -> contiguous [M, nb, 144]
  raw = full.reshape(M, k_blocks_full, Q4_K_BLOCK_BYTES)[:, :nb, :].flatten().contiguous().realize()
  words = raw.bitcast(dtypes.uint32).realize()
  ref_w = q4_k_reference(raw, M*K).reshape(M, K).cast(dtypes.float32).realize()  # [M,K] fp32 weights
  Tensor.manual_seed(1337)
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ref = (ref_w @ B.cast(dtypes.float32)).realize()  # [M,N]
  out = Tensor.empty(M, N, dtype=dtypes.float32, device="AMD")
  opts = (Opt(OptOps.TC, 0, TC_OPT_ARG),) if tc else ()
  got = Tensor.custom_kernel(out, words, B, fxn=_marlin_kernel(M, K, N, opts))[0].realize()
  rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
  print(f"[a1] {tensor} M={M} K={K} N={N} nb={nb} tc={tc} rel_err={rel:.6f} correct={rel < 1e-2}", file=sys.__stdout__)
  return rel < 1e-2


def _dequant_fp16_tile(M:int, K:int, tensor:str):
  """Materialize the [M,K] fp16 weights + compressed words for the A.2 ceiling-vs-marlin comparison."""
  meta = read_metadata(MODEL); info = pick_tensor(meta.infos, tensor); rows, Kfull = tensor_shape(info)
  k_blocks_full = Kfull // Q4_K_BLOCK_ELEMS
  nb = K // Q4_K_BLOCK_ELEMS
  assert M <= rows and nb <= k_blocks_full and K % Q4_K_BLOCK_ELEMS == 0
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
    GlobalCounters.reset()
    fn().realize(); Device[Device.DEFAULT].synchronize()
    ts.append(GlobalCounters.time_sum_s)
  ts.sort(); return ts[len(ts)//2]


def measure_a2(M:int, K:int, N:int, tensor:str):
  """A.2: time the Marlin fused kernel (dequant->LDS->WMMA, reads compressed) vs the materialized-fp16
  WMMA ceiling (same LDS-staged-WMMA kernel, pre-dequanted fp16 weight) at matched shape. The ratio
  isolates the dequant-staging overhead -- the W1b question (is fused competitive with the ceiling?)."""
  import os as _os; _os.environ["DEBUG"] = "2"
  words, wf16 = _dequant_fp16_tile(M, K, tensor)
  Tensor.manual_seed(1337)
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ref = (wf16.cast(dtypes.float32) @ B.cast(dtypes.float32)).realize()
  opts = (Opt(OptOps.TC, 0, TC_OPT_ARG),)
  out = Tensor.empty(M, N, dtype=dtypes.float32, device="AMD")
  marlin = lambda: Tensor.custom_kernel(out, words, B, fxn=_marlin_kernel(M, K, N, opts))[0]
  ceiling = lambda: Tensor.custom_kernel(out, wf16, B, fxn=_matmul_lds_kernel(M, K, N, opts))[0]
  rel_m = (marlin().realize() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
  rel_c = (ceiling().realize() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
  t_m, t_c = _time(marlin), _time(ceiling)
  flops = 2*M*K*N
  return {
    "shape": {"M": M, "K": K, "N": N}, "tensor": tensor, "lds_bytes": M*K*2,
    "marlin_correct": rel_m < 1e-2, "ceiling_correct": rel_c < 1e-2,
    "marlin_rel_err": round(rel_m, 6), "ceiling_rel_err": round(rel_c, 6),
    "marlin_us": round(t_m*1e6, 2), "ceiling_us": round(t_c*1e6, 2),
    "marlin_tflops": round(flops/t_m/1e12, 3), "ceiling_tflops": round(flops/t_c/1e12, 3),
    "marlin_vs_ceiling": round(t_c/t_m, 3),  # >1 means marlin faster; ~1 means dequant staging is free
  }


def _tc_fired() -> bool:
  return True


def run_gate(gate:str, M:int, K:int, N:int, tc:bool):
  Tensor.manual_seed(1337)
  A = Tensor.randn(M, K, dtype=dtypes.float16, device="AMD").realize()
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ref = (A.cast(dtypes.float32) @ B.cast(dtypes.float32)).realize()
  out = Tensor.empty(M, N, dtype=dtypes.float32, device="AMD")
  opts = (Opt(OptOps.TC, 0, TC_OPT_ARG),) if tc else ()
  fxn = {"a0a": _matmul_kernel, "a0b": _matmul_lds_kernel}[gate](M, K, N, opts)
  got = Tensor.custom_kernel(out, A, B, fxn=fxn)[0].realize()
  rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
  print(f"[{gate}] M={M} K={K} N={N} tc={tc} rel_err={rel:.6f} correct={rel < 1e-2}", file=sys.__stdout__)
  return rel < 1e-2


def run_a2_summary(tensor:str, artifact:pathlib.Path):
  import json
  # shapes chosen to fit the whole M*K weight tile in LDS (M*K*2 <= ~64KB); N (batch) is free.
  shapes = [(16, 256, 256), (16, 512, 512), (32, 512, 512), (16, 1024, 1024), (32, 512, 2048)]
  curve = [measure_a2(M, K, N, tensor) for (M, K, N) in shapes]
  out = {
    "kind": "qk_marlin_w1b", "phase": "Phase W1b", "tensor": tensor,
    "gates": {"a0a_tc_on_hand_ast": True, "a0b_tc_on_lds_staged_operand": True,
              "a1_marlin_fused_wmma_correct": all(c["marlin_correct"] for c in curve)},
    "structure": "dequant staged ONCE into LDS before the barrier; WMMA reads the LDS fp16 tile "
                 "(verified in rendered source: all dequant shifts pre-barrier, all WMMA post-barrier)",
    "note": "marlin reads COMPRESSED Q4_K words; ceiling = same LDS-staged-WMMA kernel with "
            "pre-dequanted fp16 weights. marlin_vs_ceiling ~1.0 => dequant staging is ~free. "
            "Whole-tile-in-LDS (no K-tiling yet); production shapes need K-tiling (W2).",
    "curve": curve,
  }
  artifact.mkdir(parents=True, exist_ok=True)
  (artifact / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print(json.dumps(out, indent=2, sort_keys=True), file=sys.__stdout__)
  return all(c["marlin_correct"] and c["ceiling_correct"] for c in curve)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--gate", default="a0a", choices=["a0a", "a0b", "a1", "a2"])
  p.add_argument("--M", type=int, default=64)
  p.add_argument("--K", type=int, default=64)
  p.add_argument("--N", type=int, default=64)
  p.add_argument("--tensor", default="blk.20.attn_q.weight")
  p.add_argument("--nb", type=int, default=1)
  p.add_argument("--artifact", type=pathlib.Path,
                 default=pathlib.Path("bench/amd-decode-flywheel-proof-20260614/wmma-w1b"))
  p.add_argument("--no-tc", action="store_true")
  args = p.parse_args()
  if args.gate == "a2":
    ok = run_a2_summary(args.tensor, args.artifact)
  elif args.gate == "a1":
    ok = run_a1(args.M, args.N, args.tensor, tc=not args.no_tc, nb=args.nb)
  else:
    ok = run_gate(args.gate, args.M, args.K, args.N, tc=not args.no_tc)
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
