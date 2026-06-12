#!/usr/bin/env python3
from __future__ import annotations

import argparse, contextlib, io
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.support.compiler_amd import amdgpu_disassemble, compile_hip
from tinygrad.uop.ops import KernelInfo, Ops, UOp

VDOT_HIP_SMOKE = r'''
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 64)))
amd_vdot_smoke(unsigned int* out, unsigned int* a, unsigned int* b) {
  unsigned int av = *a;
  unsigned int bv = *b;
  unsigned int acc = 0;
  asm volatile("v_dot4_u32_u8 %0, %1, %2, %0" : "+v"(acc) : "v"(av), "v"(bv));
  *out = acc;
}
'''

def _pack_u8(vals) -> np.uint32:
  out = 0
  for i, x in enumerate(vals): out |= (int(x) & 0xff) << (8*i)
  return np.uint32(out)

def _vdot_group_kernel(out:UOp, q4_words:UOp, q8_bias_words:UOp) -> UOp:
  srcs = (out,) + tuple(q8_bias_words[i] for i in range(8)) + tuple(q4_words[i] for i in range(8))
  lines = ["{{ int dot = 0; int q4sum = 0; int q8sum = 0;"]
  for i in range(8):
    q8, q4 = 1+i, 1+8+i
    lines += [
      f"  unsigned int q8_{i} = (unsigned int)({{{q8}}}); unsigned int q4_{i} = (unsigned int)({{{q4}}});",
      f"  asm volatile(\"v_dot4_u32_u8 %0, %1, %2, %0\" : \"+v\"(dot) : \"v\"(q8_{i}), \"v\"(q4_{i}));",
    ]
    for shift in (0, 8, 16, 24):
      lines.append(f"  q4sum += (int)((q4_{i} >> {shift}) & 255u); q8sum += (int)((q8_{i} >> {shift}) & 255u);")
  lines.append("  {0}[0] = dot - q4sum * 128; {0}[1] = q8sum - 4096; {0}[2] = q4sum; }}")
  return UOp(Ops.CUSTOM, dtypes.void, srcs, arg=" ".join(lines)).sink(
    arg=KernelInfo(name="q4_q8_vdot_group_smoke", opts_to_apply=()))

def disassemble_smoke(arch:str) -> str:
  lib = compile_hip(VDOT_HIP_SMOKE, arch)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf): amdgpu_disassemble(lib)
  asm = buf.getvalue()
  dot_lines = [line.strip() for line in asm.splitlines() if "v_dot4" in line]
  if not dot_lines: raise AssertionError("compiled smoke did not disassemble to a v_dot4 instruction")
  print(f"instruction_smoke: PASS arch={arch} line={dot_lines[0]}")
  return asm

def run_group_correctness(device:str, seed:int) -> None:
  rng = np.random.default_rng(seed)
  q4 = rng.integers(0, 16, size=32, dtype=np.int32)
  q8 = rng.integers(-127, 128, size=32, dtype=np.int32)
  q4_words = np.array([_pack_u8(q4[i:i+4]) for i in range(0, 32, 4)], dtype=np.uint32)
  q8_bias_words = np.array([_pack_u8(q8[i:i+4] + 128) for i in range(0, 32, 4)], dtype=np.uint32)
  ref = np.array([int((q4*q8).sum()), int(q8.sum()), int(q4.sum())], dtype=np.int32)

  out = Tensor.empty(3, dtype=dtypes.int32, device=device)
  q4t = Tensor(q4_words, device=device).realize()
  q8t = Tensor(q8_bias_words, device=device).realize()
  got = out.custom_kernel(q4t, q8t, fxn=_vdot_group_kernel)[0].realize().numpy()
  if got.tolist() != ref.tolist():
    raise AssertionError(f"biased vdot group mismatch: got={got.tolist()} ref={ref.tolist()} q4={q4.tolist()} q8={q8.tolist()}")
  print(f"group_correctness: PASS seed={seed} dot={got[0]} q8_sum={got[1]} q4_sum={got[2]}")

def main() -> None:
  parser = argparse.ArgumentParser(description="AMD RDNA packed-dot smoke for Q4 x q8_1 experiments")
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--arch", default=None)
  parser.add_argument("--seed", type=int, default=1337)
  parser.add_argument("--print-asm", action="store_true")
  args = parser.parse_args()

  arch = args.arch or getattr(Device[args.device], "arch", "gfx1100")
  asm = disassemble_smoke(arch)
  run_group_correctness(args.device, args.seed)
  if args.print_asm: print(asm)

if __name__ == "__main__":
  main()
