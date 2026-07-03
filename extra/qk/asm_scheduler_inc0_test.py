"""Inc 0 proof for the prefill ASM instruction scheduler (extra/qk/asm_scheduler.py).

Proves the instruction IR + dependency DAG is FAITHFUL, so later increments can reorder safely:

  P1 IDENTITY BYTE-IDENTICAL   -- schedule(identity) reproduces the build_gemm_lds2 stream bit-for-bit (no info loss).
  P2 DECODE COVERAGE + RANGE   -- every operand of every instruction is classified; all decoded regs are in range.
  P3 DAG LEGALITY              -- every dependency edge is backward in program order (original order is a valid topo sort).
  P4 LAYOUT PRESERVED         -- per-region reorder keeps total byte size (branch offsets baked by the builder stay valid).
  P5 IDENTITY RUNS CORRECT    -- the unmodified kernel computes the GEMM (control; rel_rmse <= 3e-4).
  P6 LEGAL REORDER RUNS CORRECT-- a non-trivial dependency-respecting 'asap' reorder STILL computes correctly. This is
                                  the strong empirical test: a missing real dependency would let asap reorder a true
                                  hazard and corrupt the result.

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk/asm_scheduler_inc0_test.py
"""
import numpy as np
from tinygrad import Tensor, dtypes, Context, GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.engine.realize import run_linear
from extra.qk.prefill.wmma import build_gemm_lds2, _run_insts_lds, _rmse
from extra.qk.asm_scheduler import (schedule, dag_stats, lift, build_regions,
                                    check_identity_byte_identical, check_offsets_preserved)

# Dense-prefill-representative config: W2x2 T4x4 -> 128x128 tile, BK32, PAD16, plain compute (max independent ds/wmma).
M = N = K = getenv("MNK", 512)
WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF = 2, 2, 4, 4, 32, 16, 0
THREADS, BM, BN = WAVES_M*WAVES_N*32, WAVES_M*WM*16, WAVES_N*WN*16
LDSB = max((BK*2+PAD)*(BM+BN)*(2 if DBUF else 1), 8192)

def build(): return build_gemm_lds2(M, N, K, WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF)

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
  st = dag_stats(insts)
  print(f"config W{WAVES_M}x{WAVES_N} T{WM}x{WN} BK{BK} PAD{PAD} DBUF{DBUF}  M=N=K={M}")
  print(f"DAG: {st['insts']} insts | {st['fences']} fences | {st['regions']} regions | "
        f"max_region={st['max_region']} | {st['dep_edges']} dep edges")

  # P1 identity byte-identical
  p1 = check_identity_byte_identical(insts)
  print(f"P1 IDENTITY_BYTE_IDENTICAL ........ {'PASS' if p1 else 'FAIL'}"); ok &= p1

  # P2 decode coverage + range (lift() asserts on unknown optype; here we assert every reg is in physical range)
  p2 = True
  for k, i in enumerate(insts):
    nd = lift(i, k)
    for cls, num in (nd.defs | nd.uses):
      if not ((cls == "v" and 0 <= num <= 255) or (cls == "s" and 0 <= num <= 105)):
        print(f"   out-of-range reg ({cls},{num}) in {nd.name}@{k}"); p2 = False
  print(f"P2 DECODE_COVERAGE_AND_RANGE ...... {'PASS' if p2 else 'FAIL'}"); ok &= p2

  # P3 DAG legality: every edge backward in program order
  nodes = [lift(i, k) for k, i in enumerate(insts)]
  p3 = all(all(j < a for j in r.deps[a]) for r in build_regions(nodes) for a in r.deps)
  print(f"P3 DAG_LEGALITY_BACKWARD_EDGES .... {'PASS' if p3 else 'FAIL'}"); ok &= p3

  # P4 layout preserved under asap reorder
  asap = schedule(insts, "asap")
  p4 = check_offsets_preserved(insts, asap)
  permuted = sum(1 for a, b in zip(insts, asap) if a is not b)
  print(f"P4 LAYOUT_PRESERVED ({permuted} insts moved) ... {'PASS' if p4 else 'FAIL'}"); ok &= p4
  if permuted == 0:
    print("   WARNING: asap did not permute anything -- the reorder test below is not exercising the DAG"); ok = False

  # P5 identity correctness on GPU (control)
  e_id = run(schedule(insts, "identity"), "inc0_identity")
  p5 = e_id <= 3e-4
  print(f"P5 IDENTITY_RUNS_CORRECT (rmse {e_id:.2e}) ... {'PASS' if p5 else 'FAIL'}"); ok &= p5

  # P6 legal reorder correctness on GPU (the strong faithfulness test)
  e_as = run(asap, "inc0_asap")
  p6 = e_as <= 3e-4
  print(f"P6 LEGAL_REORDER_RUNS_CORRECT (rmse {e_as:.2e}) ... {'PASS' if p6 else 'FAIL'}"); ok &= p6

  print(f"\nINC0 {'ALL_PASS -- IR+DAG faithful, ready for Inc 1 (waitcnt lever)' if ok else 'FAIL'}")
  return 0 if ok else 1

if __name__ == "__main__":
  raise SystemExit(main())
