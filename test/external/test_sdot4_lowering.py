#!/usr/bin/env python3
"""Codegen test for the first-class signed dot4 renderer helper (_sdot4).

Validates the LOWERING (Phase 3 of the MMVQ deep-linearizer arc): a kernel that references `_sdot4(` must emit
the renderer-owned helper using the native RDNA3 dot4 instruction (v_dot4_i32_iu8), NOT scalarize to 4 muls, and
the compiled object must contain a real v_dot4 instruction (disassembly).

Run: DEV=AMD PYTHONPATH=. .venv/bin/python -m pytest test/external/test_sdot4_lowering.py -q
"""
import os, re, subprocess, tempfile, unittest

class TestSdot4Lowering(unittest.TestCase):
  def test_helper_emits_native_v_dot4(self):
    from tinygrad.runtime.support.compiler_amd import compile_hip
    # a kernel body that references _sdot4 -> renderer emits the helper (cstyle.py)
    helper = ('__attribute__((device)) int _sdot4(int a, int b, int c){ int r = c; '
              'asm("v_dot4_i32_iu8 %0, %1, %2, %0" : "+v"(r) : "v"(a), "v"(b)); return r; }')
    src = helper + '\nextern "C" __attribute__((global)) void k(int* o, int* a, int* b){ o[0] = _sdot4(a[0], b[0], 0); }'
    lib = compile_hip(src, "gfx1100")
    self.assertGreater(len(lib), 0)
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f: f.write(lib); fn = f.name
    try:
      objdump = "/opt/rocm-7.2.4/llvm/bin/llvm-objdump"
      if not os.path.exists(objdump): self.skipTest("llvm-objdump not found")
      dis = subprocess.run([objdump, "-d", fn], capture_output=True, text=True).stdout
    finally:
      os.unlink(fn)
    v_dot4 = [l for l in dis.splitlines() if "v_dot4" in l]
    self.assertTrue(v_dot4, "no native v_dot4 instruction emitted (scalar fallback)")
    # must be a real dot4 (iu8 variant on RDNA3), not the scalar-fallback builtin
    self.assertTrue(any("v_dot4_i32_iu8" in l or "v_dot4_u32_u8" in l for l in v_dot4), f"unexpected dot4 form: {v_dot4[:1]}")

  def test_builtin_sdot4_scalar_fallbacks_on_gfx1100(self):
    # documents the ISA finding: __builtin_amdgcn_sdot4 compiles (with dot1-insts) but emits NO v_dot4 on RDNA3
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
    self.assertEqual([l for l in dis.splitlines() if "v_dot4" in l], [],
                     "builtin sdot4 unexpectedly emitted v_dot4 on gfx1100 (ISA finding changed)")

if __name__ == "__main__":
  unittest.main()
