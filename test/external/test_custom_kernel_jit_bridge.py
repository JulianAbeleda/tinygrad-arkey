#!/usr/bin/env python3
"""Bridge proof: a custom kernel (Tensor.custom_kernel) is captured by TinyJit as an Ops.PROGRAM and replayed
correctly. This is the prerequisite for a future fused flash-prefill kernel -- the only sanctioned way a custom
kernel can join graph/JIT execution is as a compiler-visible program (Tensor.custom_kernel -> Ops.PROGRAM ->
TinyJit capture/replay), NOT a raw HIP launch hidden from the graph (which would not be captured/ordered/replayed).

If these FAIL, the next task is NOT flash-prefill -- it is the custom-kernel/JIT integration boundary itself.

No model load, no GGUF, no hardware benchmark. Device-agnostic UOp custom kernels (run on Device.DEFAULT).
"""
import unittest

from tinygrad import Tensor, UOp, TinyJit, dtypes, Device
from tinygrad.helpers import JIT
from tinygrad.uop.ops import KernelInfo, Ops

# The capture/Ops.PROGRAM/replay logic proven here is device-agnostic, but we validate on AMD -- it's the
# flash-prefill target, and CPU custom-kernel compilation isn't reliably available in this environment.
_BRIDGE_DEV = Device.DEFAULT == "AMD"

# --- tiny custom kernels (UOp style, per test/backend/test_custom_kernel.py) -----------------------------
# y[i] = x[i]*2 + 1 -- a fused affine that cannot be mistaken for a default elementwise lowering.
def double_plus_one_kernel(B:UOp, A:UOp) -> UOp:
  A, B = A.flatten(), B.flatten()
  i = UOp.range(A.numel(), 0)
  return B[i].store(A[i] * 2 + 1).end(i).sink(arg=KernelInfo(name=f"double_plus_one_{A.numel()}"))

# y[i] = x[i]*3 -- a distinct second stage (simulates flash-prefill needing multiple custom stages).
def triple_kernel(B:UOp, A:UOp) -> UOp:
  A, B = A.flatten(), B.flatten()
  i = UOp.range(A.numel(), 0)
  return B[i].store(A[i] * 3).end(i).sink(arg=KernelInfo(name=f"triple_{A.numel()}"))

def _dpo(x:Tensor) -> Tensor: return Tensor.empty(x.shape, dtype=x.dtype).custom_kernel(x, fxn=double_plus_one_kernel)[0]
def _tri(x:Tensor) -> Tensor: return Tensor.empty(x.shape, dtype=x.dtype).custom_kernel(x, fxn=triple_kernel)[0]

def _captured_program_names(jit:TinyJit) -> list[str]:
  # Each compiled kernel is an Ops.CALL whose src[0] is its Ops.PROGRAM. A single custom kernel sits directly
  # in linear.src; chained custom kernels get wrapped (the CALLs nest under an Ops.CUSTOM_FUNCTION), so we walk
  # the full toposort. Toposort is dependency order, so a kernel that consumes another's output appears after it.
  assert jit.captured is not None, "TinyJit captured nothing -- custom kernel did not enter the graph"
  names = []
  for u in jit.captured.linear.toposort():
    if u.op is Ops.CALL and len(u.src) and u.src[0].op is Ops.PROGRAM: names.append(u.src[0].arg.name)
  return names

@unittest.skipUnless(JIT and _BRIDGE_DEV, "bridge proof requires JIT (capture/replay) on AMD")
class TestCustomKernelJitBridge(unittest.TestCase):
  N = 64

  def test_single_kernel_captured_as_program_and_replayed(self):
    jf = TinyJit(_dpo)
    # 3 calls cross the cnt transitions: 0=eager, 1=capture, >=2=replay. Use a value at REPLAY that differs
    # from the CAPTURE value, so a correct replay proves real input substitution (not a frozen captured result).
    valsA = [i * 0.5 for i in range(self.N)]
    valsB = [i - 17.0 for i in range(self.N)]              # capture value (call #2), distinct from A
    for call_i, vals in enumerate((valsA, valsB, valsA)):  # eager / capture / replay
      x = Tensor(vals, dtype=dtypes.float32).realize()
      got = jf(x).tolist()
      exp = [v * 2 + 1 for v in vals]
      for g, e in zip(got, exp):
        self.assertAlmostEqual(g, e, places=4, msg=f"call#{call_i}: custom kernel wrong (g={g} e={e})")

    names = _captured_program_names(jf)
    matches = [n for n in names if n.startswith("double_plus_one")]
    self.assertEqual(len(matches), 1, f"expected exactly ONE custom-kernel PROGRAM captured (no drop/dup), got {names}")

  def test_program_actually_present_negative_guard(self):
    # The capture must contain a real Ops.PROGRAM. If custom_kernel ever stopped being graphable this is the
    # canary: an empty/absent program list fails here loudly rather than silently no-op'ing.
    jf = TinyJit(_dpo)
    for _ in range(3): jf(Tensor([1.0] * self.N, dtype=dtypes.float32).realize())
    self.assertTrue(any(n.startswith("double_plus_one") for n in _captured_program_names(jf)),
                    "no Ops.PROGRAM for the custom kernel in the captured JIT -- bridge broken")

  def test_two_kernels_in_sequence_captured_in_order(self):
    # simulates flash-prefill needing >1 custom stage: y = triple(double_plus_one(x)) = (x*2+1)*3
    def f(x:Tensor) -> Tensor: return _tri(_dpo(x))
    jf = TinyJit(f)
    valsA = [float(i) for i in range(self.N)]
    valsB = [i * -0.25 for i in range(self.N)]
    for vals in (valsA, valsB, valsA):
      x = Tensor(vals, dtype=dtypes.float32).realize()
      got = jf(x).tolist()
      exp = [(v * 2 + 1) * 3 for v in vals]
      for g, e in zip(got, exp): self.assertAlmostEqual(g, e, places=4)

    names = _captured_program_names(jf)
    dpo_idx = next((i for i, n in enumerate(names) if n.startswith("double_plus_one")), None)
    tri_idx = next((i for i, n in enumerate(names) if n.startswith("triple")), None)
    self.assertIsNotNone(dpo_idx, f"first custom kernel missing from capture: {names}")
    self.assertIsNotNone(tri_idx, f"second custom kernel missing from capture: {names}")
    self.assertLess(dpo_idx, tri_idx, f"custom kernels captured out of order: {names}")
    self.assertEqual(sum(n.startswith("double_plus_one") for n in names), 1, f"stage 1 dropped/duplicated: {names}")
    self.assertEqual(sum(n.startswith("triple") for n in names), 1, f"stage 2 dropped/duplicated: {names}")

if __name__ == "__main__":
  unittest.main()
