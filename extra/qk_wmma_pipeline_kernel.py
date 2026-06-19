#!/usr/bin/env python3
"""CG-1 — UOp-expressibility test for a software-pipelined (double-buffered global->reg->LDS prefetch) WMMA GEMM.

Base = amd_copy_matmul (single-buffer, CG-0 = ~48 TFLOPS, global_load on the critical path each K-tile). Here the
k_tile loop PREFETCHES tile k+1's global data into registers during tile k, so the global-load latency can overlap the
WMMA. Question (the fork): does tinygrad's linearizer/renderer PRESERVE the overlap (issue the prefetch loads before
the WMMA, defer vmcnt) -> fork A buildable; or serialize it (barrier-per-iteration keeps global_load after the prior
barrier, no overlap) -> fork B renderer-capability.

  run: DEV=AMD WMMA=1 N=4096 PYTHONPATH=. .venv/bin/python extra/qk_wmma_pipeline_kernel.py
"""
import os, sys
sys.path.insert(0, "extra/gemm")
from tinygrad import UOp, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.dtype import AddrSpace, dtypes

N = getenv("N", 4096); M = getenv("M", N); K = getenv("K", N)
WARP_SIZE = 32; BLOCK_M, BLOCK_N = 128, 128; BLOCK_K = getenv("BK", 16)
WAVES_M, WAVES_N = 2, 2; LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 16
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16; WMMA_ACC = WMMA_M // LANES_PER_WAVE_M
THREADS_PER_BLOCK = WARP_SIZE * WAVES_M * WAVES_N
TM = BLOCK_M // (WAVES_M * LANES_PER_WAVE_M); TN = BLOCK_N // (WAVES_N * LANES_PER_WAVE_N)
NK = K // BLOCK_K

def pipelined_gemm(c:UOp, a:UOp, b:UOp) -> UOp:
  wave_m = UOp.range(WAVES_M, 2, AxisType.LOCAL); wave_n = UOp.range(WAVES_N, 3, AxisType.LOCAL)
  lane = UOp.range(WARP_SIZE, -1, AxisType.WARP); tid = (wave_m*WAVES_N+wave_n)*WARP_SIZE + lane
  A_local = UOp.placeholder((BLOCK_M, BLOCK_K), a.dtype.base, slot=0, addrspace=AddrSpace.LOCAL)
  B_local = UOp.placeholder((BLOCK_N, BLOCK_K), b.dtype.base, slot=1, addrspace=AddrSpace.LOCAL)
  a = a.reshape(NK, BLOCK_K, BLOCK_M); b = b.reshape(NK, BLOCK_K, BLOCK_N)
  k_tile = UOp.range(NK, 100, AxisType.REDUCE)
  # PREFETCH: stage NEXT tile's global -> register during this iteration (independent of this iter's WMMA).
  knext = (k_tile + 1) % NK
  ECNT = (BLOCK_M*BLOCK_K)//THREADS_PER_BLOCK
  a_pf = UOp.placeholder((ECNT,), a.dtype.base, slot=3, addrspace=AddrSpace.REG)
  b_pf = UOp.placeholder((ECNT,), b.dtype.base, slot=4, addrspace=AddrSpace.REG)
  a_pf = a_pf.after(a_pf.store(a[knext].reshape(-1, THREADS_PER_BLOCK)[:, tid]))   # global -> reg (prefetch)
  b_pf = b_pf.after(b_pf.store(b[knext].reshape(-1, THREADS_PER_BLOCK)[:, tid]))
  A_copy = A_local.permute((1,0)); B_copy = B_local.permute((1,0))
  A_store = A_copy.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(a[k_tile].reshape(-1, THREADS_PER_BLOCK)[:, tid])
  B_store = B_copy.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(b[k_tile].reshape(-1, THREADS_PER_BLOCK)[:, tid])
  barrier = UOp.barrier(A_store, B_store, a_pf, b_pf)   # prefetch issued alongside the LDS store, before WMMA
  A_local, B_local = A_local.after(barrier), B_local.after(barrier)
  lane_m, lane_n = lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
  acc = UOp.placeholder((TM, TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.zeros_like()))
  k = UOp.range(BLOCK_K // WMMA_K, 101, AxisType.REDUCE)
  tile_m = UOp.range(TM // WMMA_ACC, 200, AxisType.LOOP); tile_n = UOp.range(TN, 201, AxisType.LOOP)
  acc_view = acc.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0,2,1)
  acc_frag = acc_view[tile_m, tile_n]; acc_frag_after = acc_view.after(k)[tile_m, tile_n]
  a_frag = A_local.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_K // WMMA_K, WMMA_K)[wave_m, tile_m, lane_n, k]
  b_frag = B_local.reshape(WAVES_N, TN, WMMA_N, BLOCK_K // WMMA_K, WMMA_K)[wave_n, tile_n, lane_n, k]
  wmma = UOp(Ops.SHAPED_WMMA, dtypes.float, (a_frag, b_frag, acc_frag_after), arg=((16,16,16),'AMD',32))
  acc_store = acc_frag.store(wmma).end(tile_m, tile_n)
  acc = acc.after(acc_store.end(k).barrier().end(k_tile))
  c = c.reshape(WAVES_M, TM//1, LANES_PER_WAVE_M, 1, WAVES_N, TN//1, LANES_PER_WAVE_N, 1)
  c = c.permute((0,4,2,6,1,3,5,7)).reshape(THREADS_PER_BLOCK, TM, TN)
  return c[tid].store(acc).end(wave_m, wave_n, lane)

def pipelined_matmul(c:UOp, a:UOp, b:UOp) -> UOp:
  bm = UOp.range(M // BLOCK_M, 0, AxisType.GLOBAL); bn = UOp.range(N // BLOCK_N, 1, AxisType.GLOBAL)
  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[bm, :, bn, :]
  a = a.T.reshape(K, M // BLOCK_M, BLOCK_M)[:, bm, :]; b = b.reshape(K, N // BLOCK_N, BLOCK_N)[:, bn, :]
  return pipelined_gemm(c, a, b).end(bn, bm).sink(arg=KernelInfo(opts_to_apply=()))

if __name__ == "__main__":
  os.environ.setdefault("WMMA","1")
  from amd_uop_matmul import eval_custom_matmul
  eval_custom_matmul(pipelined_matmul, dtypes.half)
