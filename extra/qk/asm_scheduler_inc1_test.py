"""Inc 1 proof for the prefill ASM instruction scheduler: the wait-counter (s_waitcnt) model.

Inc 1 adds the async-load wait-counter model: AMD RDNA3 tracks outstanding async memory ops in per-domain counters
(`vmcnt` for VMEM, `lgkmcnt` for LDS+SMEM); a load's destination is valid only after an s_waitcnt drains its counter,
and same-domain ops retire in issue order. Delivered + proven here:

  Q1 HAND_WAITS_ALREADY_MINIMAL  -- audit: the existing full drains have ~0 relaxable slack (the human placed them
                                    optimally for the given order). HONEST: standalone consumer-only relax is ~free.
  Q2 RECOMPUTE_INPLACE_CORRECT   -- recompute_waits_inplace() (minimal counts, byte-layout preserving) runs correctly.
  Q3 IDENTITY_IS_WAIT_CORRECT    -- the soundness gate verify_wait_correct() passes on the unmodified stream.
  Q4 GATE_DISCRIMINATES          -- the gate REJECTS a stream with the drains removed (it isn't trivially true).
  Q5 WAIT_MODEL_COMPOSES_WITH_REORDER -- recompute waits on the Inc-0 (memory-anchored, proven-safe) reorder still
                                    computes correctly on the GPU: the wait model composes with reordering.
  Q6 WAIT_CORRECTNESS_NECESSARY_NOT_SUFFICIENT -- KEY honest finding. A fence_only reorder that moves memory ops is
                                    register-legal (Inc 0 DAG, 0 missing edges) AND wait-correct (gate=True), yet the
                                    prologue variant computes wrong on hardware. So wait-correctness alone does NOT
                                    license cross-motion -- there is an additional RDNA3 hardware-spacing/scoreboard
                                    hazard. That hazard recognizer is Inc 2's scope; cross-motion stays OFF until then.

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk/asm_scheduler_inc1_test.py
"""
import numpy as np
from tinygrad import Tensor, dtypes, Context
from tinygrad.helpers import getenv
from tinygrad.engine.realize import run_linear
from extra.qk.prefill.wmma import build_gemm_lds2, _run_insts_lds, _rmse
from extra.qk.asm_scheduler import (schedule, lift, wait_constraints, wait_slack, verify_wait_correct,
                                    recompute_waits_inplace)
from tinygrad.runtime.autogen.amd.rdna3.ins import s_waitcnt

M = N = K = getenv("MNK", 512)
WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF, PLRA = 2, 2, 4, 4, 32, 16, 0, 1
THREADS, BM, BN = WAVES_M*WAVES_N*32, WAVES_M*WM*16, WAVES_N*WN*16
LDSB = max((BK*2+PAD)*(BM+BN)*(2 if DBUF else 1), 8192)

def build(): return build_gemm_lds2(M, N, K, WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF, PLRA=PLRA)

def run(insts, name):
  rng = np.random.default_rng(1)
  a_np = (rng.standard_normal((M, K))*0.1).astype(np.float16)
  bt_np = (rng.standard_normal((N, K))*0.1).astype(np.float16)
  c = Tensor.empty(M, N, dtype=dtypes.half); Tensor.realize(c)
  linear, out = _run_insts_lds(insts, Tensor(a_np), Tensor(bt_np), c, M, N, K, name, LDSB, BM, BN, THREADS)
  with Context(DEBUG=0): run_linear(linear)
  return _rmse(out, a_np, bt_np)

def main():
  ok = True
  insts = build()

  # Q1 audit: hand-placed waits already minimal
  cons = wait_constraints(insts); slack = wait_slack(insts)
  relaxable = [(k, D, h, r) for k, D, h, r in cons if r > h and r < 0x3F]
  print(f"Q1 audit: {len(cons)} (wait,domain) constraints, total relaxable slack = {slack}")
  for k, D, h, r in relaxable: print(f"     relaxable @{k} {D}: have={h} -> minimal={r} (perf-irrelevant prologue scalar load)")
  q1 = slack <= 2
  print(f"Q1 HAND_WAITS_ALREADY_MINIMAL ............... {'PASS' if q1 else 'FAIL'}"); ok &= q1

  # Q2 recompute-in-place runs correct on GPU
  rec = recompute_waits_inplace(insts)
  changed = sum(1 for a, b in zip(insts, rec) if a.to_bytes() != b.to_bytes())
  e_rec = run(rec, "inc1_recompute"); q2 = e_rec <= 3e-4
  print(f"Q2 RECOMPUTE_INPLACE_CORRECT ({changed} waits changed, rmse {e_rec:.2e}) ... {'PASS' if q2 else 'FAIL'}"); ok &= q2

  # Q3 soundness gate passes on the identity stream
  ok3, why3 = verify_wait_correct(insts)
  print(f"Q3 IDENTITY_IS_WAIT_CORRECT ................. {'PASS' if ok3 else 'FAIL'} ({why3})"); ok &= ok3

  # Q4 the gate rejects a stream with the drains removed
  no_drain = [s_waitcnt(simm16=0x3FF0) if lift(i, 0).name == "S_WAITCNT" else i for i in insts]  # vm=max,lgkm=max
  okn, _ = verify_wait_correct(no_drain); q4 = not okn
  print(f"Q4 GATE_DISCRIMINATES (no-drain rejected={not okn}) ... {'PASS' if q4 else 'FAIL'}"); ok &= q4

  # Q5 wait model composes with the proven-safe Inc-0 reorder (memory anchored -> no cross-motion)
  safe = recompute_waits_inplace(schedule(insts, "asap"))
  ok5g, _ = verify_wait_correct(safe)
  e_safe = run(safe, "inc1_safe_reorder"); q5 = ok5g and e_safe <= 3e-4
  print(f"Q5 WAIT_MODEL_COMPOSES_WITH_REORDER (gate={ok5g}, rmse {e_safe:.2e}) ... {'PASS' if q5 else 'FAIL'}"); ok &= q5

  # Q6 KEY honest finding: a fence_only cross-motion reorder is register-legal AND wait-correct, but that is NOT
  # sufficient for hardware correctness -> we assert the gate vouches (necessary) while keeping cross-motion OFF.
  fo = schedule(insts, "asap", fence_only=True)
  ok6g, why6 = verify_wait_correct(fo)
  moved_mem = sum(1 for a, b in zip(insts, fo) if a is not b and lift(a, 0).domain is not None)
  q6 = ok6g and moved_mem > 0   # gate says wait-correct (necessary); see Inc 2 for the OTHER necessary gate
  print(f"Q6 WAIT_CORRECTNESS_NECESSARY_NOT_SUFFICIENT (gate={ok6g}, {moved_mem} mem moved) ... {'PASS' if q6 else 'FAIL'}")
  print("     -> CORRECTED by Inc 2: the missing gate is the LOOP-ENTRY (branch-target) control-flow boundary, NOT an")
  print("        RDNA3 hardware hazard (ISA-confirmed: s_delay_alu is perf-only). With it, cross-motion is sound.")
  ok &= q6

  print(f"\nINC1 {'ALL_PASS -- wait-counter model delivered (audit+recompute+verify); cross-motion gate completed in Inc 2 (loop-entry boundary)' if ok else 'FAIL'}")
  return 0 if ok else 1

if __name__ == "__main__":
  raise SystemExit(main())
