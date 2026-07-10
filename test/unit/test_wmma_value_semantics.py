"""VALUE-semantics gate for the fp16 RDNA3 WMMA lowering (16x16x16, K=64 reduce chain, 64x64 multi-tile).

The structural gates in test_amd_isa_wmma.py only count instructions / inspect the emitted source; their own
comments defer numeric correctness to a "parent DEV=AMD bit-exact gate". That gate did not exist as a runnable
pytest, which is exactly the hole that let the unsigned-dot4 bug ship. This file closes it: it emits the real
AMDISARenderer kernel and RUNS it through the remu RDNA3 functional emulator (models v_wmma_f32_16x16x16_f16),
comparing the emitted-kernel output against a numpy `(a @ b)` reference over signed / negative / zero /
edge-magnitude fp16 lanes.

remu is a pure-software emulator (no GPU needed), so this runs in CI as long as libremu.so is present. When it is
absent the tests SKIP honestly (machine-enforced), never silently pass. The invariant is enforced by execution,
not by a comment.
"""
import ctypes, os, unittest
import numpy as np

os.environ.setdefault("ALLOW_DEVICE_USAGE", "1")

from tinygrad import Tensor
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.codegen import to_program, to_program_cache

# remu 0.1.2 (github.com/Qazalin/remu) prebuilt shared object. Overridable via $LIBREMU. It models real RDNA3
# instruction semantics including v_wmma_f32_16x16x16_f16, so it is a faithful numeric oracle for the WMMA path.
LIBREMU = os.environ.get("LIBREMU", "/home/ubuntu/.claude/jobs/2f995982/tmp/libremu.so")
HAVE_REMU = os.path.exists(LIBREMU)
WMMA_MN = "v_wmma_f32_16x16x16_f16"


def _emit_wmma_kernel(M: int, N: int, K: int) -> tuple[bytes, int]:
  """Render the FINAL resolved AMDISARenderer instruction stream for a half A[M,K] @ B[K,N] matmul.

  Returns the raw .text bytes and the number of v_wmma instructions actually emitted, so the caller can assert it
  is really exercising the WMMA path (and the specific multi-tile subtile count) rather than a scalar fallback.
  """
  ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  to_program_cache.clear()   # else a repeat shape short-circuits rendering and _resolve_labels never runs
  cap: dict = {}
  orig = ren._resolve_labels
  def wrap(insts):
    r = orig(insts); cap["final"] = list(r); return r
  ren._resolve_labels = wrap
  a = Tensor.empty(M, K, dtype="half"); b = Tensor.empty(K, N, dtype="half")
  ast = [u for u in (a @ b).schedule_linear().toposort() if u.op is Ops.SINK][0]
  prg = to_program(ast, ren)
  lin = [u for u in prg.src if u.op is Ops.LINEAR][0]
  n_wmma = sum(1 for u in lin.src if not isinstance(u.arg, tuple) and str(u.arg).startswith(WMMA_MN))
  raw = b"".join(u.arg.to_bytes() for u in cap["final"])
  assert len(raw) % 4 == 0, len(raw)
  return raw, n_wmma


def _run_remu(A: np.ndarray, B: np.ndarray) -> tuple[int, np.ndarray, int]:
  """Emit the WMMA kernel for A@B and execute it through remu. Returns (rc, out_fp32[M,N], n_wmma)."""
  A = np.ascontiguousarray(A, dtype=np.float16); B = np.ascontiguousarray(B, dtype=np.float16)
  (M, K), (K2, N) = A.shape, B.shape
  assert K == K2, (A.shape, B.shape)
  OUT = np.zeros((M, N), dtype=np.float16)
  text, n_wmma = _emit_wmma_kernel(M, N, K)
  # kernarg buffer: [out_ptr, A_ptr, B_ptr] at offsets 0x0/0x8/0x10 (tinygrad order [out, *ins]).
  args = (ctypes.c_uint64 * 3)(OUT.ctypes.data, A.ctypes.data, B.ctypes.data)
  lib = ctypes.CDLL(LIBREMU)
  lib.run_asm.restype = ctypes.c_int
  lib.run_asm.argtypes = [ctypes.c_char_p, ctypes.c_uint32] + [ctypes.c_uint32] * 6 + [ctypes.POINTER(ctypes.c_uint64)]
  rc = lib.run_asm(ctypes.c_char_p(text), len(text), 1, 1, 1, 32, 1, 1, args)  # single workgroup, wave32
  return rc, OUT.astype(np.float32), n_wmma


def _rel_rmse(got: np.ndarray, ref: np.ndarray) -> float:
  return float(np.sqrt(np.mean((got - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-6))


@unittest.skipUnless(HAVE_REMU, f"remu emulator not found at {LIBREMU} (set $LIBREMU); fp16 WMMA value gate needs it")
class TestAMDISAWmmaValueSemantics(unittest.TestCase):
  # DEV-free numeric gate: the emitted v_wmma kernel, run through remu, must match numpy (a @ b). This is the
  # bit-exact-ish parent gate the structural gates in test_amd_isa_wmma.py defer their correctness to.
  def _check(self, A, B, *, expect_wmma: int, rtol: float = 4e-3, atol: float = 5e-2):
    rc, got, n_wmma = _run_remu(A, B)
    self.assertEqual(rc, 0, f"remu run_asm returned rc={rc}")
    self.assertEqual(n_wmma, expect_wmma, f"expected {expect_wmma} v_wmma in the emitted kernel, got {n_wmma}")
    ref = A.astype(np.float32) @ B.astype(np.float32)
    self.assertEqual(int(np.isnan(got).sum()), 0, "WMMA output contains NaN (hardware/logic hazard)")
    max_abs = float(np.max(np.abs(got - ref)))
    rel = _rel_rmse(got, ref)
    # zero/exact cases pass on max_abs; general fp16 accumulation passes on relative rmse.
    self.assertTrue(rel < rtol or max_abs < atol,
                    f"WMMA value mismatch: rel_rmse={rel:.4e} max_abs={max_abs:.4e}\n got[0,:6]={got.ravel()[:6]}\n ref[0,:6]={ref.ravel()[:6]}")
    return got, ref

  # ---- 16x16x16: the single-tile base case (1 WMMA) ----
  def test_16x16x16_signed_random(self):
    rng = np.random.default_rng(0)
    self._check(rng.standard_normal((16, 16)), rng.standard_normal((16, 16)), expect_wmma=1)

  def test_16x16x16_all_negative(self):
    # every lane strictly negative -> product terms positive; a sign-handling bug (unsigned dot) would diverge.
    rng = np.random.default_rng(1)
    A = -np.abs(rng.standard_normal((16, 16))) - 0.5
    B = -np.abs(rng.standard_normal((16, 16))) - 0.5
    self._check(A, B, expect_wmma=1)

  def test_16x16x16_zeros_and_negatives(self):
    # structured zeros interleaved with negative lanes: exercises zero-lane accumulation + sign together.
    rng = np.random.default_rng(2)
    A = rng.standard_normal((16, 16)); A[::2, :] = 0.0; A[:, 1::3] = -np.abs(A[:, 1::3])
    B = rng.standard_normal((16, 16)); B[1::2, :] = 0.0
    self._check(A, B, expect_wmma=1)

  def test_16x16x16_all_zero_is_exact_zero(self):
    got, ref = self._check(np.zeros((16, 16)), np.zeros((16, 16)), expect_wmma=1)
    self.assertTrue(np.array_equal(got, ref), "all-zero WMMA output must be exactly zero")

  def test_16x16x16_edge_magnitude(self):
    # large fp16 lane magnitudes (well above the usual ~N(0,1)); output stays inside fp16 range for K=16.
    rng = np.random.default_rng(3)
    A = (rng.uniform(-1, 1, (16, 16)) * 30.0).astype(np.float16)
    B = (rng.uniform(-1, 1, (16, 16)) * 30.0).astype(np.float16)
    self._check(A, B, expect_wmma=1, rtol=6e-3)

  # ---- 16x16x64: the rolled K-reduction chain (still 1 in-place WMMA accumulating over 4 K-tiles) ----
  def test_k64_reduction_signed(self):
    rng = np.random.default_rng(4)
    self._check(rng.standard_normal((16, 64)), rng.standard_normal((64, 16)), expect_wmma=1)

  def test_k64_reduction_zeros_and_negatives(self):
    rng = np.random.default_rng(5)
    A = rng.standard_normal((16, 64)); A[:, ::4] = 0.0; A[:, 1::4] = -np.abs(A[:, 1::4])
    B = rng.standard_normal((64, 16)); B[::4, :] = 0.0
    self._check(A, B, expect_wmma=1)

  # ---- 64x64x64: THE audit-flagged multi-output-tile register model (WM=WN=4 -> 16 subtiles / accumulators) ----
  def test_64x64_multitile_signed_random(self):
    rng = np.random.default_rng(6)
    self._check(rng.standard_normal((64, 64)), rng.standard_normal((64, 64)), expect_wmma=16)

  def test_64x64_multitile_zeros_and_negatives(self):
    # per-subtile residency bug would cross-wire A-rows/B-cols; structured zeros/negatives localize any misroute.
    rng = np.random.default_rng(7)
    A = rng.standard_normal((64, 64)); A[16:32, :] = 0.0; A[:, ::5] = -np.abs(A[:, ::5])
    B = rng.standard_normal((64, 64)); B[:, 32:48] = 0.0
    self._check(A, B, expect_wmma=16)

  def test_64x64_multitile_edge_magnitude(self):
    rng = np.random.default_rng(8)
    A = (rng.uniform(-1, 1, (64, 64)) * 6.0).astype(np.float16)
    B = (rng.uniform(-1, 1, (64, 64)) * 6.0).astype(np.float16)
    self._check(A, B, expect_wmma=16, rtol=6e-3)


if __name__ == "__main__":
  unittest.main()
