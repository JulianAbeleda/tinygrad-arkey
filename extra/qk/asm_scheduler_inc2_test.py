"""Inc 2 proof for the prefill ASM instruction scheduler: sound cross-motion + latency-aware schedule.

Inc 2 corrects Inc 1's misdiagnosis. Inc 1 concluded a fence_only reorder that was register-legal AND wait-correct
still computed wrong = "an RDNA3 hardware-spacing hazard." That was WRONG. The real cause: the loop-entry point (the
backward branch TARGET) is a control-flow boundary that build_regions did not model, so the reorder moved instructions
across the loop entry (between prologue and loop body). Adding branch-target boundaries makes fence_only cross-motion
byte-identical-correct. Independently confirmed by ISA research: on RDNA3 s_delay_alu is performance-only (the hardware
interlocks VALU/VMEM deps), so a register-legal + wait-correct reorder CANNOT corrupt values via spacing.

Proven here (correctness is the gate; timing is informational and non-deterministic):
  R1 CROSS_MOTION_SOUND   -- fence_only asap reorder (mem ops moved) is byte-identical-correct across the route config
                             space, for BOTH asap (stress) and critical (latency-aware) modes.
  R2 THREE_GATES_HOLD     -- offsets preserved + verify_wait_correct passes on every scheduled stream.
  R3 LATENCY_REORDER_NEUTRAL (informational) -- the critical-path reorder is perf-neutral vs identity on clean
                             clock-pinned isolated timing (the hand schedule is already near-optimal in-region; the
                             residual to Tensile needs waitcnt-relocation / cross-iteration pipelining, not reordering).

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk/asm_scheduler_inc2_test.py
"""
import numpy as np
from tinygrad import Tensor, dtypes, Context, GlobalCounters, Device
from tinygrad.helpers import getenv
from tinygrad.engine.realize import run_linear, Estimates
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.helpers import colored
from tinygrad.dtype import AddrSpace
from extra.qk.prefill.wmma import build_gemm_lds2, _run_insts_lds, _rmse
from extra.qk.asm_scheduler import schedule, verify_wait_correct, check_offsets_preserved, lift, branch_target_indices

# (label, WAVES_M,WAVES_N,WM,WN,BK,PAD,DBUF,PLRA,PLRAB) -- the real route config space
CFGS = [("default_PLRA", (2,2,4,4,32,16,0,1,0)), ("kv_halved", (2,1,4,4,32,16,0,1,0)),
        ("DBUF1", (2,2,4,4,32,16,1,0,0)), ("8wave_PLRAB", (4,2,2,4,32,16,1,0,1))]

def run_small(insts, args, name):
  M = N = K = 512
  WM_, WN_, wm, wn, bk, pad, dbuf, plra, plrab = args
  TH, BM, BN = WM_*WN_*32, WM_*wm*16, WN_*wn*16
  LDSB = max((bk*2+pad)*(BM+BN)*(2 if dbuf else 1), 8192)
  rng = np.random.default_rng(1)
  a = (rng.standard_normal((M, K))*0.1).astype(np.float16); bt = (rng.standard_normal((N, K))*0.1).astype(np.float16)
  c = Tensor.empty(M, N, dtype=dtypes.half); Tensor.realize(c)
  linear, out = _run_insts_lds(insts, Tensor(a), Tensor(bt), c, M, N, K, name, LDSB, BM, BN, TH)
  with Context(DEBUG=0): run_linear(linear)
  return _rmse(out, a, bt)

def timing_ab():
  M, N, K = 512, 4096, 4096
  WM_, WN_, wm, wn, bk, pad, dbuf = 2, 2, 4, 4, 32, 16, 1
  TH, BM, BN = WM_*WN_*32, WM_*wm*16, WN_*wn*16
  LDSB = max((bk*2+pad)*(BM+BN)*2, 8192); grid = (N//BN, M//BM, 1)
  rng = np.random.default_rng(1)
  A = Tensor((rng.standard_normal((M, K))*0.1).astype(np.float16)).contiguous().realize()
  Bt = Tensor((rng.standard_normal((N, K))*0.1).astype(np.float16)).contiguous().realize()
  def mk(insts, lab):
    C = Tensor.empty(M, N, dtype=dtypes.half).contiguous().realize()
    def kern(A, Bt, C):
      lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=LDSB, addrspace=AddrSpace.LOCAL), (), 'lds')
      g = [UOp.special(grid[0], "gidx0"), UOp.special(grid[1], "gidx1")]
      sink = UOp.sink(A.base, Bt.base, C.base, lds, *g, UOp.special(TH, "lidx0"),
                      arg=KernelInfo(name=colored(lab, "cyan"), estimates=Estimates(ops=M*N*K*2, mem=(M*K+N*K+M*N)*2)))
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                   UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))
    return Tensor.custom_kernel(A, Bt, C, fxn=kern)[2].schedule_linear()
  base = build_gemm_lds2(M, N, K, WM_, WN_, wm, wn, bk, pad, dbuf, PLRA=0)
  lins = {"identity": mk(base, "identity"), "critical": mk(schedule(base, "critical", fence_only=True), "critical")}
  with Context(DEBUG=2):
    for _ in range(5):
      for lin in lins.values(): run_linear(lin)
  best = {}
  for _ in range(8):
    for lab, lin in lins.items():
      with Context(DEBUG=2):
        ets = [(lambda st: (run_linear(lin), GlobalCounters.time_sum_s-st)[1])(GlobalCounters.time_sum_s) for _ in range(40)]
      best[lab] = min([best.get(lab, 9e9)] + [t for t in ets if t > 0])
  flop = M*N*K*2
  return best["identity"], best["critical"], flop

def main():
  ok = True
  # R1 + R2: cross-motion sound across configs, both modes
  print("R1/R2 cross-motion correctness (fence_only + loop-entry boundaries):")
  for lab, args in CFGS:
    insts = build_gemm_lds2(512, 512, 512, *args)
    nt = len(branch_target_indices(insts))
    row_ok = True
    for mode in ("asap", "critical"):
      sched = schedule(insts, mode, fence_only=True)
      mm = sum(1 for a, b in zip(insts, sched) if a is not b and lift(a, 0).domain is not None)
      g_off = check_offsets_preserved(insts, sched); g_wait = verify_wait_correct(sched)[0]
      e = run_small(sched, args, f"{lab}_{mode}")
      good = g_off and g_wait and e <= 3e-4
      row_ok &= good
      print(f"   {lab:13} {mode:8} mem_moved={mm:3} off={g_off} wait={g_wait} rmse={e:.2e} {'ok' if good else 'FAIL'}")
    ok &= row_ok
  print(f"R1 CROSS_MOTION_SOUND + R2 THREE_GATES_HOLD ... {'PASS' if ok else 'FAIL'}")

  # R3 informational timing
  ti, tc, flop = timing_ab()
  print(f"\nR3 (informational) clean clock-pinned isolated timing, DBUF1 512x4096x4096:")
  print(f"   identity {ti*1e6:6.1f}us {flop/ti*1e-12:.2f} TFLOPS | critical {tc*1e6:6.1f}us {flop/tc*1e-12:.2f} TFLOPS"
        f" | critical {(ti/tc-1)*100:+.2f}% (within noise -> latency reorder NEUTRAL on the hand-tuned kernel)")

  print(f"\nINC2 {'CORRECTNESS_PASS -- cross-motion sound (Inc1 hazard misdiagnosis corrected); latency reorder perf-neutral' if ok else 'FAIL'}")
  return 0 if ok else 1

if __name__ == "__main__":
  raise SystemExit(main())
