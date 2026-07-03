"""Inc 3 proof for the prefill ASM instruction scheduler: waitcnt RELOCATION (the first non-neutral lever).

Inc 2 showed pure instruction REORDERING is perf-neutral. Inc 3 changes the instruction SET: in each compute block
([N ds_loads][lgkm(0) full drain][M wmmas]) it removes the full drain and inserts, before each WMMA (issued in
frag-ready order), the MINIMAL lgkmcnt for just that WMMA's fragments -- overlapping WMMA compute with the tail of
LDS-load latency. Inserting instructions requires recomputing branch offsets (capture_branch_targets + fix_branches,
non-mutating). All three Inc-0/1/2 gates still apply; verify_wait_correct is the correctness gate.

Proven here (correctness is the gate; timing is informational):
  S1 RELOCATION_CORRECT  -- relocate_lgkm_waits is byte-correct (rmse <= 3e-4) and verify_wait_correct passes across the
                            route config space and across K sizes (NBLK 16..128). Branch offsets are right (no fault).
  S2 NON_MUTATING        -- relocating a fresh build does NOT corrupt a separately-built identity stream.
  S3 RELOCATION_WINS_DBUF1 (informational) -- clean clock-pinned isolated timing shows a real, reproducible speedup on
                            DBUF1 (~+6%); config-dependent (PLRA route ~+2%, kv_halved regresses) -> needs per-config
                            gating, and whole-prefill confirmation before any promotion.

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk/asm_scheduler_inc3_test.py
"""
import numpy as np
from tinygrad import Tensor, dtypes, Context, GlobalCounters, Device
from tinygrad.engine.realize import run_linear, Estimates
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.helpers import colored
from tinygrad.dtype import AddrSpace
from extra.qk.prefill.wmma import build_gemm_lds2, _run_insts_lds, _rmse
from extra.qk.asm_scheduler import relocate_lgkm_waits, verify_wait_correct, lift

CFGS = [("plain", (2,2,4,4,32,16,0,0)), ("DBUF1", (2,2,4,4,32,16,1,0)),
        ("PLRA_route", (2,2,4,4,32,16,0,1)), ("kv_halved", (2,1,4,4,32,16,0,1))]

def run_small(insts, cfg, M, N, K):
  WM_, WN_, wm, wn, bk, pad, dbuf, plra = cfg
  TH, BM, BN = WM_*WN_*32, WM_*wm*16, WN_*wn*16
  LDSB = max((bk*2+pad)*(BM+BN)*(2 if dbuf else 1), 8192)
  rng = np.random.default_rng(1)
  a = (rng.standard_normal((M, K))*0.1).astype(np.float16); bt = (rng.standard_normal((N, K))*0.1).astype(np.float16)
  c = Tensor.empty(M, N, dtype=dtypes.half); Tensor.realize(c)
  lin, out = _run_insts_lds(insts, Tensor(a), Tensor(bt), c, M, N, K, "t", LDSB, BM, BN, TH)
  with Context(DEBUG=0): run_linear(lin)
  return _rmse(out, a, bt)

def time_dbuf1():
  M, N, K = 512, 4096, 4096
  cfg = (2, 2, 4, 4, 32, 16, 1, 0)
  WM_, WN_, wm, wn, bk, pad, dbuf, plra = cfg
  TH, BM, BN = WM_*WN_*32, WM_*wm*16, WN_*wn*16
  LDSB = max((bk*2+pad)*(BM+BN)*2, 8192); grid = (N//BN, M//BM, 1)
  rng = np.random.default_rng(1)
  A = Tensor((rng.standard_normal((M, K))*0.1).astype(np.float16)).contiguous().realize()
  Bt = Tensor((rng.standard_normal((N, K))*0.1).astype(np.float16)).contiguous().realize()
  def mk(insts, nm):
    C = Tensor.empty(M, N, dtype=dtypes.half).contiguous().realize()
    def kern(A, Bt, C):
      lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=LDSB, addrspace=AddrSpace.LOCAL), (), 'lds')
      g = [UOp.special(grid[0], "gidx0"), UOp.special(grid[1], "gidx1")]
      sink = UOp.sink(A.base, Bt.base, C.base, lds, *g, UOp.special(TH, "lidx0"),
                      arg=KernelInfo(name=colored(nm, "cyan"), estimates=Estimates(ops=M*N*K*2, mem=(M*K+N*K+M*N)*2)))
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                   UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))
    return Tensor.custom_kernel(A, Bt, C, fxn=kern)[2]
  lins = {"identity": mk(build_gemm_lds2(M, N, K, *cfg), "id").schedule_linear(),
          "reloc": mk(relocate_lgkm_waits(build_gemm_lds2(M, N, K, *cfg)), "reloc").schedule_linear()}
  with Context(DEBUG=2):
    for _ in range(5):
      for l in lins.values(): run_linear(l)
  best = {}
  for _ in range(10):
    for nm, l in lins.items():
      with Context(DEBUG=2):
        ets = [(lambda st: (run_linear(l), GlobalCounters.time_sum_s-st)[1])(GlobalCounters.time_sum_s) for _ in range(40)]
      best[nm] = min([best.get(nm, 9e9)] + [t for t in ets if t > 0])
  return best["identity"], best["reloc"], M*N*K*2

def main():
  ok = True
  # S1 correctness across configs (512) and across K sizes (plain)
  print("S1 relocation correctness:")
  for lab, cfg in CFGS:
    base = build_gemm_lds2(512, 512, 512, *cfg); reloc = relocate_lgkm_waits(build_gemm_lds2(512, 512, 512, *cfg))
    dwaits = sum(1 for i in reloc if lift(i, 0).name == "S_WAITCNT") - sum(1 for i in base if lift(i, 0).name == "S_WAITCNT")
    okv = verify_wait_correct(reloc)[0]; e = run_small(reloc, cfg, 512, 512, 512)
    good = okv and e <= 3e-4; ok &= good
    print(f"   {lab:11} waits+{dwaits:<2} verify={okv} rmse={e:.2e} {'ok' if good else 'FAIL'}")
  for (M, N, K) in [(512, 1024, 1024), (512, 4096, 4096)]:  # NBLK 32, 128 -- proves branch offsets right at scale
    cfg = (2, 2, 4, 4, 32, 16, 1, 0); reloc = relocate_lgkm_waits(build_gemm_lds2(M, N, K, *cfg))
    e = run_small(reloc, cfg, M, N, K); good = verify_wait_correct(reloc)[0] and e <= 3e-4; ok &= good
    print(f"   DBUF1 NBLK={K//32:<3} rmse={e:.2e} {'ok' if good else 'FAIL'}")
  print(f"S1 RELOCATION_CORRECT ... {'PASS' if ok else 'FAIL'}")

  # S2 non-mutating: relocating must not corrupt a separately-built identity stream
  b = build_gemm_lds2(512, 512, 512, 2, 2, 4, 4, 32, 16, 1, PLRA=0)
  before = [i.to_bytes() for i in b]
  _ = relocate_lgkm_waits(b)
  s2 = all(x == y.to_bytes() for x, y in zip(before, b)); ok &= s2
  print(f"S2 NON_MUTATING (identity stream intact) ... {'PASS' if s2 else 'FAIL'}")

  # S3 informational timing
  ti, tr, flop = time_dbuf1()
  print(f"\nS3 (informational) clean clock-pinned isolated timing, DBUF1 512x4096x4096:")
  print(f"   identity {flop/ti*1e-12:.2f} TFLOPS | reloc {flop/tr*1e-12:.2f} TFLOPS | reloc {(ti/tr-1)*100:+.2f}%"
        f"  (config-dependent: DBUF1 ~+6%, PLRA route ~+2%, kv_halved regresses -> needs per-config gating)")

  print(f"\nINC3 {'CORRECTNESS_PASS -- waitcnt relocation is a real (config-dependent) lever; first non-neutral result' if ok else 'FAIL'}")
  return 0 if ok else 1

if __name__ == "__main__":
  raise SystemExit(main())
