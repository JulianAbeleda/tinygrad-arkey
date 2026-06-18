#!/usr/bin/env python3
"""Value-level + emission test for the _sdot4 renderer helper (signed*unsigned dot4).

_sdot4(a,b,c) = a(SIGNED int8 lanes) . b(UNSIGNED int8 lanes) + c, lowered via __builtin_amdgcn_sudot4 ->
native v_dot4_i32_iu8 with neg_lo signedness modifier (llama's RDNA3 ggml_cuda_dp4a path). For Q4_K MMVQ:
a=q8 (signed activations), b=q4 nibbles (0..15).

History/lesson: a prior version used a BARE `v_dot4_i32_iu8` asm (no modifier), which silently computes
UNSIGNED*UNSIGNED -- and the test only checked instruction *emission*, never the computed *value*, so the bug
shipped mislabeled "signed". This test now checks the VALUE.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python -m pytest test/external/test_sdot4_lowering.py -q
"""
import os, re, subprocess, tempfile, unittest
import numpy as np

class TestSdot4Lowering(unittest.TestCase):
  def test_value_signed_times_unsigned(self):
    # the DECISIVE test: a treated SIGNED (int8), b treated UNSIGNED (uint8), incl negatives in a.
    from tinygrad import Tensor, dtypes
    from tinygrad.uop.ops import UOp, KernelInfo
    from extra.q4_k_gemv_primitive import _sdot4_op
    def kfn(N):
      def kern(out, a, b):
        gid = UOp.special(N, "gidx0")
        return out[gid].store(_sdot4_op(a[gid], b[gid], UOp.const(dtypes.int32, 0))).sink(arg=KernelInfo(name="sdv"))
      return kern
    cases = np.array([[[255,0,0,0],[1,0,0,0]],      # a=-1 signed, b=1  -> -1
                      [[128,0,0,0],[3,0,0,0]],      # a=-128, b=3       -> -384
                      [[1,2,3,255],[1,1,1,255]],    # a3=-1, b3=255(u)  -> 1+2+3-255
                      [[5,10,15,2],[253,4,255,7]]], dtype=np.uint8)
    N = len(cases)
    a = Tensor(cases[:,0].copy().view(np.int32).reshape(N)).realize()
    b = Tensor(cases[:,1].copy().view(np.int32).reshape(N)).realize()
    got = Tensor.empty(N, dtype=dtypes.int32).custom_kernel(a, b, fxn=kfn(N))[0].numpy()
    a_s = cases[:,0].astype(np.int8).astype(int); b_u = cases[:,1].astype(int)
    want = np.array([int((a_s[i]*b_u[i]).sum()) for i in range(N)])
    self.assertTrue(np.array_equal(got, want), f"_sdot4 != signed*unsigned: got {got.tolist()} want {want.tolist()}")
    # guard against the old unsigned*unsigned bug
    a_u = cases[:,0].astype(int)
    self.assertFalse(np.array_equal(got, np.array([int((a_u[i]*b_u[i]).sum()) for i in range(N)])),
                     "_sdot4 computed unsigned*unsigned (the regressed bug)")

  def test_renderer_helper_emits_native_v_dot4(self):
    # the renderer-emitted _sdot4 (sudot4) must lower to a native v_dot4 instruction, not scalarize
    from tinygrad.runtime.support.compiler_amd import compile_hip
    helper = '__attribute__((device)) int _sdot4(int a,int b,int c){ return __builtin_amdgcn_sudot4(true,a,false,b,c,false); }'
    src = helper + '\nextern "C" __attribute__((global)) void k(int* o,int* a,int* b){ o[0]=_sdot4(a[0],b[0],0); }'
    lib = compile_hip(src, "gfx1100")
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f: f.write(lib); fn = f.name
    try:
      objdump = "/opt/rocm-7.2.4/llvm/bin/llvm-objdump"
      if not os.path.exists(objdump): self.skipTest("llvm-objdump not found")
      dis = subprocess.run([objdump, "-d", fn], capture_output=True, text=True).stdout
    finally:
      os.unlink(fn)
    self.assertTrue([l for l in dis.splitlines() if "v_dot4" in l], "no native v_dot4 (scalar fallback)")

  def test_builtin_sdot4_scalar_fallbacks_on_gfx1100(self):
    # ISA fact: __builtin_amdgcn_sdot4 (dot1-insts, GCN-era) scalar-fallbacks on RDNA3 -> use sudot4 instead
    from tinygrad.runtime.support.compiler_amd import compile_hip
    src = ('__attribute__((device)) __attribute__((target("dot1-insts"))) int s(int a,int b,int c){ '
           'return __builtin_amdgcn_sdot4(a,b,c,false); }\n'
           'extern "C" __attribute__((global)) void k(int* o,int* a,int* b){ o[0]=s(a[0],b[0],0); }')
    lib = compile_hip(src, "gfx1100")
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f: f.write(lib); fn = f.name
    try:
      objdump = "/opt/rocm-7.2.4/llvm/bin/llvm-objdump"
      if not os.path.exists(objdump): self.skipTest("llvm-objdump not found")
      dis = subprocess.run([objdump, "-d", fn], capture_output=True, text=True).stdout
    finally:
      os.unlink(fn)
    self.assertEqual([l for l in dis.splitlines() if "v_dot4" in l], [], "builtin sdot4 unexpectedly emitted v_dot4 (ISA changed)")

if __name__ == "__main__":
  unittest.main()
