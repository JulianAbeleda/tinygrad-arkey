#!/usr/bin/env python3
"""Phase 1 capability probes: does the proven custom_kernel -> Ops.PROGRAM -> TinyJit bridge support the
buffer/shape features a fused flash-prefill kernel needs? (test_custom_kernel_jit_bridge.py proved the bridge
itself for simple kernels.)

Probes (tiny shapes, no model, no benchmark):
  A. sliced KV-cache read   -- custom kernel reads a slice of a cache-shaped tensor; correct + JIT replay.
  B. strided/non-contiguous -- transposed q-like view; establish whether contiguous is required (invariant).
  C. SYMBOLIC start_pos     -- the critical one. flash-decode's mechanism: a BOUND symbolic slices an input
     (carries the value into var_vals), an UNBOUND DEFINE_VAR twin goes in the kernel range (a BIND in a
     custom-kernel AST fails type_verify). Replay with different start_pos -> different correct outputs.
  D. multiple outputs       -- one custom kernel writing two output buffers; both captured + correct + replay.

If A/C/D pass, the full fused flash-prefill kernel is technically unblocked. If any fail: STOP and report
(do not fix runtime/codegen here). Device-agnostic logic; gated to AMD (the target; CPU custom-kernel
compilation isn't reliable in this env).
"""
import unittest

from tinygrad import Tensor, UOp, TinyJit, dtypes, Device
from tinygrad.helpers import JIT
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops

_DEV_OK = Device.DEFAULT == "AMD"

def _program_names(jit:TinyJit) -> list[str]:
  assert jit.captured is not None, "TinyJit captured nothing"
  return [u.src[0].arg.name for u in jit.captured.linear.toposort()
          if u.op is Ops.CALL and len(u.src) and u.src[0].op is Ops.PROGRAM]

# ---- kernels ------------------------------------------------------------------------------------------
def col_sum_kernel(B:UOp, A:UOp) -> UOp:
  # A:[L,Hd] -> B:[Hd], column sums (proves the kernel reads exactly the sliced region).
  L, Hd = A.shape
  j = UOp.range(Hd, 0)
  i = UOp.range(L, 1, axis_type=AxisType.REDUCE)
  C = B[j].set(0.0)
  C = B[j].set(C.after(i)[j] + A[i, j], end=i)
  return C.end(j).sink(arg=KernelInfo(name=f"col_sum_{L}_{Hd}", opts_to_apply=()))

def flat_sum_kernel(B:UOp, A:UOp) -> UOp:
  # B[0] = sum(A) over a flattened view (probes strided/non-contiguous flatten behavior).
  A = A.flatten()
  i = UOp.range(A.numel(), 0, axis_type=AxisType.REDUCE)
  C = B[0].set(0.0)
  C = B[0].set(C.after(i)[0] + A[i], end=i)
  return C.sink(arg=KernelInfo(name=f"flat_sum_{A.numel()}", opts_to_apply=()))

def sum_prefix_kernel(MAXC:int, sp_unbound:UOp):
  # B[0] = sum_{i < sp} A[i]. Range over the CONCRETE MAXC, mask by the UNBOUND symbolic sp; index-clamp the
  # masked lanes so they read a written cell (the flash-decode NaN trap: a masked lane must read finite data).
  def kernel(B:UOp, A:UOp) -> UOp:
    i = UOp.range(MAXC, 0, axis_type=AxisType.REDUCE)
    m = i < sp_unbound
    safe = m.where(i, i.const_like(0))
    val = m.where(A[safe], UOp.const(dtypes.float32, 0.0))
    C = B[0].set(0.0)
    C = B[0].set(C.after(i)[0] + val, end=i)
    return C.sink(arg=KernelInfo(name=f"sum_prefix_{MAXC}", opts_to_apply=()))
  return kernel

def addmul_kernel(C:UOp, D:UOp, A:UOp, B:UOp) -> UOp:
  # two outputs from ONE kernel: C=A+B, D=A*B (the multi-output shape: out + lse/max/scratch).
  C, D, A, B = C.flatten(), D.flatten(), A.flatten(), B.flatten()
  i = UOp.range(C.numel(), 0)
  return UOp.group(C[i].store(A[i] + B[i]), D[i].store(A[i] * B[i])).end(i).sink(
    arg=KernelInfo(name=f"addmul_{C.numel()}")).simplify()

@unittest.skipUnless(JIT and _DEV_OK, "probe requires JIT capture/replay on AMD")
class TestFlashPrefillBridgeCaps(unittest.TestCase):
  def test_A_sliced_kv_read(self):
    MAXC, L, Hd = 64, 24, 16
    def f(cache:Tensor) -> Tensor:
      return Tensor.empty(Hd, dtype=dtypes.float32).custom_kernel(cache[0:L, :], fxn=col_sum_kernel)[0]
    jf = TinyJit(f)
    for fill in (1.0, 3.0, 2.0):  # eager / capture / replay; replay value != capture value
      cache = Tensor.full((MAXC, Hd), fill, dtype=dtypes.float32).contiguous().realize()
      got = jf(cache).tolist()
      for g in got: self.assertAlmostEqual(g, L * fill, places=3)
    self.assertEqual(sum(n.startswith("col_sum") for n in _program_names(jf)), 1, _program_names(jf))

  def test_B_strided_noncontiguous(self):
    # a transposed q-like view [T,H,Hd] -> [H,T,Hd] is non-contiguous; record whether the kernel handles it
    # via flatten or needs an explicit .contiguous() (the invariant flash-prefill must encode). Use a Python
    # reference (sum of arange is stride-invariant) -- a tinygrad reduce on the strided VIEW routes to the CPU
    # backend here (clang), which is unrelated to the custom_kernel capability under test.
    T, H, Hd = 8, 4, 8; N = T * H * Hd
    ref = N * (N - 1) / 2.0
    base = Tensor.arange(N, dtype=dtypes.float32).reshape(T, H, Hd).realize()
    view = base.transpose(0, 1)  # [H,T,Hd], non-contiguous (same elements -> same sum)
    def try_kernel(x):
      try:
        o = Tensor.empty(1, dtype=dtypes.float32).custom_kernel(x, fxn=flat_sum_kernel)[0]
        return abs(float(o.item()) - ref) < 1e-1
      except Exception:
        return None  # backend couldn't compile/run this form in this env
    noncontig_ok = try_kernel(view)
    contig_ok = try_kernel(view.contiguous())
    if contig_ok is None: self.skipTest("custom-kernel compile unavailable in this env")
    self.assertTrue(contig_ok, "custom kernel failed on a contiguous input")
    # INVARIANT: contiguous works; flash-prefill already .contiguous()-isolates q/k/v in _prefill_v2, so
    # requiring contiguous is fine. Record (don't fail on) the strided case.
    verdict = {True: "OK (strided supported)", False: "NOT SUPPORTED -> .contiguous() required",
               None: "untestable in this env"}[noncontig_ok]
    print(f"[probe B] strided/non-contiguous custom_kernel: {verdict}", flush=True)

  def test_C_symbolic_start_pos(self):
    # THE critical probe: bound symbolic carries var_vals via a slice; unbound twin drives the kernel range.
    MAXC = 64
    src = Tensor.ones(MAXC, dtype=dtypes.float32).contiguous().realize()
    v = UOp.variable("sp", 1, MAXC)                          # unbound DEFINE_VAR (used in the kernel range)
    def f(sp) -> Tensor:                                     # sp = v.bind(N) (carries N into var_vals)
      buf = Tensor.empty(MAXC, dtype=dtypes.float32)
      a = Tensor(buf.uop.after(buf[0:sp].uop.store(src[0:sp].uop)))   # store ones into [0:N]
      return Tensor.empty(1, dtype=dtypes.float32).custom_kernel(a, fxn=sum_prefix_kernel(MAXC, v))[0]
    jf = TinyJit(f)
    # replay value (15) differs from capture value (20) -> proves symbolic var substitution, not a frozen result
    for n in (10, 20, 15):                                   # eager / capture / replay
      got = float(jf(v.bind(n)).item())
      self.assertAlmostEqual(got, float(n), places=3, msg=f"symbolic start_pos: sp={n} gave {got}")
    self.assertTrue(any(p.startswith("sum_prefix") for p in _program_names(jf)),
                    f"symbolic-start_pos kernel not captured as a PROGRAM: {_program_names(jf)}")

  def test_D_multiple_outputs(self):
    N = 32
    def f(a:Tensor, b:Tensor):
      c, d = Tensor.custom_kernel(Tensor.empty(N, dtype=dtypes.float32), Tensor.empty(N, dtype=dtypes.float32),
                                  a, b, fxn=addmul_kernel)[:2]
      return c, d
    jf = TinyJit(f)
    for s in (1.0, 4.0, 2.0):
      a = Tensor.full((N,), s, dtype=dtypes.float32).contiguous().realize()
      b = Tensor.full((N,), s + 1.0, dtype=dtypes.float32).contiguous().realize()
      c, d = jf(a, b)
      cl, dl = c.tolist(), d.tolist()
      for cv, dv in zip(cl, dl):
        self.assertAlmostEqual(cv, s + (s + 1.0), places=3)   # C = A+B
        self.assertAlmostEqual(dv, s * (s + 1.0), places=3)   # D = A*B
    self.assertEqual(sum(n.startswith("addmul") for n in _program_names(jf)), 1,
                     f"multi-output kernel should be one captured PROGRAM: {_program_names(jf)}")

if __name__ == "__main__":
  unittest.main()
