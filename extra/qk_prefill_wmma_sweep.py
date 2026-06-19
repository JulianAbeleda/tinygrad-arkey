#!/usr/bin/env python3
"""POWN-0/1 — WMMA prefill config sweep (pure tinygrad, no deps).

PWLT-A2 finding: tinygrad WMMA matmul == ALU matmul (~34% WMMA peak) -> WMMA units stalled. Hypothesis: 128 fp32
accumulators/thread at 128 threads/block = high register pressure -> low occupancy -> WMMA latency not hidden. Lever:
more waves (more threads, fewer acc/thread) -> higher occupancy. Sweep WAVES_M x WAVES_N x BLOCK on the ffn shape.

Parameterizes the proven SHAPED_WMMA kernel from extra/gemm/amd_copy_matmul.py. DEBUG=2 device time, correctness vs
fp16 oracle. No route. Gate: any config >=50% WMMA peak (~62 TFLOPS, >=1.5x current).

  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_wmma_sweep.py
"""
from __future__ import annotations
import statistics
from tinygrad import Tensor, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from tinygrad.dtype import AddrSpace

WARP=32; PEAK=122.0
M,K,N = 512,4096,12288

def wmma_kernel(BLOCK_M:int, BLOCK_N:int, BLOCK_K:int, WAVES_M:int, WAVES_N:int):
  LM, LN = 2, 16                       # lanes/wave for WMMA 16x16 on wave32
  WMMA_M=WMMA_N=WMMA_K=16
  WMMA_ACC = WMMA_M // LM              # 8
  THREADS = WARP*WAVES_M*WAVES_N
  TM = BLOCK_M//(WAVES_M*LM); TN = BLOCK_N//(WAVES_N*LN)
  assert TM % WMMA_ACC == 0, f"TM={TM} not mult of {WMMA_ACC}"
  assert N%BLOCK_N==0 and M%BLOCK_M==0 and K%BLOCK_K==0
  def block_gemm(c:UOp, a:UOp, b:UOp) -> UOp:
    wave_m = UOp.range(WAVES_M, 2, AxisType.LOCAL); wave_n = UOp.range(WAVES_N, 3, AxisType.LOCAL)
    lane = UOp.range(WARP, -1, AxisType.WARP); tid = (wave_m*WAVES_N+wave_n)*WARP+lane
    A_local = UOp.placeholder((BLOCK_M, BLOCK_K), a.dtype.base, slot=0, addrspace=AddrSpace.LOCAL)
    B_local = UOp.placeholder((BLOCK_N, BLOCK_K), b.dtype.base, slot=1, addrspace=AddrSpace.LOCAL)
    a = a.reshape(K//BLOCK_K, BLOCK_K, BLOCK_M); b = b.reshape(K//BLOCK_K, BLOCK_K, BLOCK_N)
    k_tile = UOp.range(K//BLOCK_K, 100, AxisType.REDUCE)
    A_store = A_local.permute((1,0)).reshape(-1,THREADS)[:,tid].store(a[k_tile].reshape(-1,THREADS)[:,tid])
    B_store = B_local.permute((1,0)).reshape(-1,THREADS)[:,tid].store(b[k_tile].reshape(-1,THREADS)[:,tid])
    barrier = UOp.barrier(A_store, B_store); A_local,B_local = A_local.after(barrier), B_local.after(barrier)
    lane_m, lane_n = lane//LN, lane%LN
    acc = UOp.placeholder((TM,TN), dtypes.float, slot=2, addrspace=AddrSpace.REG); acc = acc.after(acc.store(acc.zeros_like()))
    k = UOp.range(BLOCK_K//WMMA_K, 101, AxisType.REDUCE)
    tile_m = UOp.range(TM//WMMA_ACC, 200, AxisType.LOOP); tile_n = UOp.range(TN, 201, AxisType.LOOP)
    acc_view = acc.reshape(TM//WMMA_ACC, WMMA_ACC, TN).permute(0,2,1)
    acc_frag = acc_view[tile_m, tile_n]; acc_frag_after = acc_view.after(k)[tile_m, tile_n]
    a_frag = A_local.reshape(WAVES_M, TM//WMMA_ACC, WMMA_M, BLOCK_K//WMMA_K, WMMA_K)[wave_m, tile_m, lane_n, k]
    b_frag = B_local.reshape(WAVES_N, TN, WMMA_N, BLOCK_K//WMMA_K, WMMA_K)[wave_n, tile_n, lane_n, k]
    wmma = UOp(Ops.SHAPED_WMMA, dtypes.float, (a_frag, b_frag, acc_frag_after), arg=((16,16,16),'AMD',32))
    acc_store = acc_frag.store(wmma).end(tile_m, tile_n)
    acc = acc.after(acc_store.end(k).barrier().end(k_tile))
    c = c.reshape(WAVES_M, TM, LM, 1, WAVES_N, TN, LN, 1).permute((0,4,2,6,1,3,5,7)).reshape(THREADS,TM,TN)
    return c[tid].store(acc).end(wave_m, wave_n, lane)
  def kernel(c:UOp, a:UOp, b:UOp) -> UOp:
    bm = UOp.range(M//BLOCK_M, 0, AxisType.GLOBAL); bn = UOp.range(N//BLOCK_N, 1, AxisType.GLOBAL)
    c = c.reshape(M//BLOCK_M, BLOCK_M, N//BLOCK_N, BLOCK_N)[bm, :, bn, :]
    a = a.T.reshape(K, M//BLOCK_M, BLOCK_M)[:, bm, :]; b = b.reshape(K, N//BLOCK_N, BLOCK_N)[:, bn, :]
    return block_gemm(c, a, b).end(bn, bm).sink(arg=KernelInfo(opts_to_apply=()))
  return kernel

def wmma_kernel_nolds(BLOCK_M:int, BLOCK_N:int, BLOCK_K:int, WAVES_M:int, WAVES_N:int):
  # PWLT-A2 learning: LDS is IC-served (useless). Remove LDS + per-K-tile barriers; load WMMA frags direct from global.
  LM, LN = 2, 16; WMMA_M=WMMA_N=WMMA_K=16; WMMA_ACC=WMMA_M//LM; THREADS=WARP*WAVES_M*WAVES_N
  TM=BLOCK_M//(WAVES_M*LM); TN=BLOCK_N//(WAVES_N*LN); assert TM%WMMA_ACC==0
  def block_gemm(c:UOp, a:UOp, b:UOp) -> UOp:
    wave_m=UOp.range(WAVES_M,2,AxisType.LOCAL); wave_n=UOp.range(WAVES_N,3,AxisType.LOCAL)
    lane=UOp.range(WARP,-1,AxisType.WARP); lane_n=lane%LN
    a=a.reshape(K//BLOCK_K, BLOCK_K, BLOCK_M); b=b.reshape(K//BLOCK_K, BLOCK_K, BLOCK_N)
    k_tile=UOp.range(K//BLOCK_K,100,AxisType.REDUCE)
    acc=UOp.placeholder((TM,TN),dtypes.float,slot=2,addrspace=AddrSpace.REG); acc=acc.after(acc.store(acc.zeros_like()))
    k=UOp.range(BLOCK_K//WMMA_K,101,AxisType.REDUCE)
    tile_m=UOp.range(TM//WMMA_ACC,200,AxisType.LOOP); tile_n=UOp.range(TN,201,AxisType.LOOP)
    acc_view=acc.reshape(TM//WMMA_ACC,WMMA_ACC,TN).permute(0,2,1); acc_frag=acc_view[tile_m,tile_n]; acc_frag_after=acc_view.after(k)[tile_m,tile_n]
    # frags direct from global tile a[k_tile]=[BK,BLOCK_M], b[k_tile]=[BK,BLOCK_N]
    a_frag=a[k_tile].reshape(BLOCK_K//WMMA_K, WMMA_K, WAVES_M, TM//WMMA_ACC, WMMA_M)[k, :, wave_m, tile_m, lane_n].permute(())  if False else \
           a[k_tile].permute((1,0)).reshape(WAVES_M, TM//WMMA_ACC, WMMA_M, BLOCK_K//WMMA_K, WMMA_K)[wave_m, tile_m, lane_n, k]
    b_frag=b[k_tile].permute((1,0)).reshape(WAVES_N, TN, WMMA_N, BLOCK_K//WMMA_K, WMMA_K)[wave_n, tile_n, lane_n, k]
    wmma=UOp(Ops.SHAPED_WMMA,dtypes.float,(a_frag,b_frag,acc_frag_after),arg=((16,16,16),'AMD',32))
    acc=acc.after(acc_frag.store(wmma).end(tile_m,tile_n).end(k).end(k_tile))
    c=c.reshape(WAVES_M,TM,LM,1,WAVES_N,TN,LN,1).permute((0,4,2,6,1,3,5,7)).reshape(THREADS,TM,TN)
    tid=(wave_m*WAVES_N+wave_n)*WARP+lane
    return c[tid].store(acc).end(wave_m,wave_n,lane)
  def kernel(c:UOp,a:UOp,b:UOp)->UOp:
    bm=UOp.range(M//BLOCK_M,0,AxisType.GLOBAL); bn=UOp.range(N//BLOCK_N,1,AxisType.GLOBAL)
    c=c.reshape(M//BLOCK_M,BLOCK_M,N//BLOCK_N,BLOCK_N)[bm,:,bn,:]
    a=a.T.reshape(K,M//BLOCK_M,BLOCK_M)[:,bm,:]; b=b.reshape(K,N//BLOCK_N,BLOCK_N)[:,bn,:]
    return block_gemm(c,a,b).end(bn,bm).sink(arg=KernelInfo(opts_to_apply=()))
  return kernel

def _tf(fxn, a, b, iters=8):
  def run(): return Tensor.custom_kernel(Tensor.empty(M,N,dtype=dtypes.float), a, b, fxn=fxn)[0]
  with Context(DEBUG=0):
    for _ in range(3): run().realize()
  ets=[]
  with Context(DEBUG=2):
    for _ in range(iters):
      GlobalCounters.reset(); run().realize(); ets.append(GlobalCounters.time_sum_s)
  return M*N*K*2/min(ets)*1e-12, run

def main():
  Tensor.manual_seed(0)
  a=Tensor.randn(M,K,dtype=dtypes.half).realize(); b=Tensor.randn(K,N,dtype=dtypes.half).realize()
  ref=(a.float()@b.float()).realize()
  print(f"=== POWN-1 WMMA config sweep, ffn shape M={M} K={K} N={N} (peak~{PEAK}) ===")
  # (BLOCK_M, BLOCK_N, BLOCK_K, WAVES_M, WAVES_N) -> threads, acc/thread
  cfgs = [
    (128,128,16,2,2),  # current amd_copy_matmul (128 threads, 128 acc/thr)
    (128,128,16,4,2),  # 256 threads, fewer acc/thr
    (128,128,16,2,4),  # 256 threads
    (128,128,16,4,4),  # 512 threads
    (128,256,16,2,2),  # MORE acc/thr (256): bigger N, same waves
    (256,128,16,2,2),  # MORE acc/thr (256): bigger M, same waves
    (256,256,16,2,2),  # MORE acc/thr (512): big tile
    (128,128,32,2,2),  # 2 WMMA-K iters/tile (fewer barriers/K-flop)
    (128,128,16,1,1),  # 1 wave (32 thr): max acc/thr but tiny occupancy
  ]
  best=0
  for cfg in cfgs:
    bm,bn,bk,wm,wn=cfg; th=WARP*wm*wn; tm=bm//(wm*2); tn=bn//(wn*16); accpt=tm*tn
    try:
      fxn=wmma_kernel(*cfg); tf,run=_tf(fxn,a,b)
      err=(run().realize()-ref).square().mean().item()
      ok = err < 1e-2
      best=max(best, tf if ok else 0)
      print(f"  B{bm}x{bn}x{bk} W{wm}x{wn} ({th:3d}thr, {accpt:3d}acc/thr): {tf:6.2f} TFLOPS ({100*tf/PEAK:2.0f}% peak) mse={err:.1e} {'OK' if ok else 'WRONG'}")
    except Exception as e:
      print(f"  B{bm}x{bn}x{bk} W{wm}x{wn} ({th:3d}thr): FAIL {type(e).__name__}: {str(e)[:90]}")
  print("  --- no-LDS WMMA (drop the IC-served LDS + per-K-tile barriers; frags direct from global) ---")
  for cfg in [(128,128,16,2,2),(128,128,16,2,4)]:
    bm,bn,bk,wm,wn=cfg; th=WARP*wm*wn
    try:
      fxn=wmma_kernel_nolds(*cfg); tf,run=_tf(fxn,a,b)
      err=(run().realize()-ref).square().mean().item(); ok=err<1e-2; best=max(best, tf if ok else 0)
      print(f"  noLDS B{bm}x{bn}x{bk} W{wm}x{wn} ({th:3d}thr): {tf:6.2f} TFLOPS ({100*tf/PEAK:2.0f}% peak) mse={err:.1e} {'OK' if ok else 'WRONG'}")
    except Exception as e:
      print(f"  noLDS B{bm}x{bn}x{bk} W{wm}x{wn}: FAIL {type(e).__name__}: {str(e)[:90]}")
  print(f"\nbest: {best:.1f} TFLOPS ({100*best/PEAK:.0f}% peak)  | current ~41 (34%)  | gate >=62 (50%, 1.5x) | ext ceiling ~70")

if __name__=="__main__": main()
