#!/usr/bin/env python3
"""CG-R1 — CORRECTLY-WIRED software-pipelined (double-buffered) WMMA GEMM. Pure tinygrad, no dependency.

CG-1's prefetch was dead code (stored a[k_tile], not the prefetched a[k+1]) -> byte-identical ISA. This redoes it
right: 2 LDS buffers A_local[2]/B_local[2]; PROLOGUE loads tile 0 -> buf0; loop k: store tile (k+1) global->buf[(k+1)%2]
(the prefetch, WIRED) while WMMA computes tile k <- buf[k%2] (loaded last iter). next-tile load and current WMMA use
different buffers -> can overlap (the lever the oracle showed Tensile uses). Measure TFLOPS + disasm overlap.

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

def dbuf_gemm(c:UOp, a:UOp, b:UOp) -> UOp:
  wave_m = UOp.range(WAVES_M, 2, AxisType.LOCAL); wave_n = UOp.range(WAVES_N, 3, AxisType.LOCAL)
  lane = UOp.range(WARP_SIZE, -1, AxisType.WARP); tid = (wave_m*WAVES_N+wave_n)*WARP_SIZE + lane
  # 2-deep LDS double buffer
  A_local = UOp.placeholder((2, BLOCK_M, BLOCK_K), a.dtype.base, slot=0, addrspace=AddrSpace.LOCAL)
  B_local = UOp.placeholder((2, BLOCK_N, BLOCK_K), b.dtype.base, slot=1, addrspace=AddrSpace.LOCAL)
  a = a.reshape(NK, BLOCK_K, BLOCK_M); b = b.reshape(NK, BLOCK_K, BLOCK_N)

  def store_tile(kk, slot):   # global tile kk -> LDS buffer `slot` (transpose for wmma)
    As = A_local[slot].permute((1,0)).reshape(-1, THREADS_PER_BLOCK)[:, tid].store(a[kk].reshape(-1, THREADS_PER_BLOCK)[:, tid])
    Bs = B_local[slot].permute((1,0)).reshape(-1, THREADS_PER_BLOCK)[:, tid].store(b[kk].reshape(-1, THREADS_PER_BLOCK)[:, tid])
    return As, Bs

  # PROLOGUE: load tile 0 into buffer 0
  pA, pB = store_tile(0, 0)
  prologue = UOp.barrier(pA, pB)
  Al, Bl = A_local.after(prologue), B_local.after(prologue)

  lane_m, lane_n = lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
  acc = UOp.placeholder((TM, TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.zeros_like()))

  k_tile = UOp.range(NK, 100, AxisType.REDUCE)
  cur = k_tile % 2; nxt = (k_tile + 1) % 2
  # PREFETCH tile k+1 (wraps; harmless extra load on last iter) into the alternate buffer
  nA, nB = store_tile((k_tile + 1) % NK, nxt)
  # COMPUTE tile k_tile from the current buffer (loaded last iteration / prologue)
  Ak = Al.after(nA)[cur]; Bk = Bl.after(nB)[cur]   # after the prefetch store so the barrier orders both
  k = UOp.range(BLOCK_K // WMMA_K, 101, AxisType.REDUCE)
  tile_m = UOp.range(TM // WMMA_ACC, 200, AxisType.LOOP); tile_n = UOp.range(TN, 201, AxisType.LOOP)
  acc_view = acc.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0,2,1)
  acc_frag = acc_view[tile_m, tile_n]; acc_frag_after = acc_view.after(k)[tile_m, tile_n]
  a_frag = Ak.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_K // WMMA_K, WMMA_K)[wave_m, tile_m, lane_n, k]
  b_frag = Bk.reshape(WAVES_N, TN, WMMA_N, BLOCK_K // WMMA_K, WMMA_K)[wave_n, tile_n, lane_n, k]
  wmma = UOp(Ops.SHAPED_WMMA, dtypes.float, (a_frag, b_frag, acc_frag_after), arg=((16,16,16),'AMD',32))
  acc_store = acc_frag.store(wmma).end(tile_m, tile_n)
  acc = acc.after(acc_store.end(k).barrier().end(k_tile))   # barrier per k_tile (orders prefetch vs next-iter wmma)

  c = c.reshape(WAVES_M, TM, LANES_PER_WAVE_M, 1, WAVES_N, TN, LANES_PER_WAVE_N, 1)
  c = c.permute((0,4,2,6,1,3,5,7)).reshape(THREADS_PER_BLOCK, TM, TN)
  return c[tid].store(acc).end(wave_m, wave_n, lane)

def dbuf_matmul(c:UOp, a:UOp, b:UOp) -> UOp:
  bm = UOp.range(M // BLOCK_M, 0, AxisType.GLOBAL); bn = UOp.range(N // BLOCK_N, 1, AxisType.GLOBAL)
  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[bm, :, bn, :]
  a = a.T.reshape(K, M // BLOCK_M, BLOCK_M)[:, bm, :]; b = b.reshape(K, N // BLOCK_N, BLOCK_N)[:, bn, :]
  return dbuf_gemm(c, a, b).end(bn, bm).sink(arg=KernelInfo(opts_to_apply=()))

if __name__ == "__main__":
  os.environ.setdefault("WMMA","1")
  from amd_uop_matmul import eval_custom_matmul
  eval_custom_matmul(dbuf_matmul, dtypes.half)
