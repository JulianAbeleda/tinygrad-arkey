#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib

import numpy as np

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from tinygrad.renderer.amd.dsl import NULL, s, v
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  global_load_b32, global_store_b32, s_cmp_eq_u32, s_cselect_b32, s_endpgm, s_load_b128, s_mul_i32,
  s_waitcnt, v_lshlrev_b32_e32, v_mov_b32_e32,
)

HIDDEN, Q4_WORDS, Q8_BYTES = 12288, 7077888, 4608
Q4_WORDS_PER_ROW = 576

def build_q4_load_skeleton(gate:UOp, up:UOp, gate_words:UOp, up_words:UOp, q8:UOp) -> UOp:
  gidxs = [UOp.special(n, f"gidx{i}") for i, n in enumerate((HIDDEN, 2, 1))]
  lidxs = [UOp.special(n, f"lidx{i}") for i, n in enumerate((32, 4, 1))]
  insts = [
    s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL),       # gate/up output pointers
    s_load_b128(sdata=s[12:15], sbase=s[0:1], offset=0x10, soffset=NULL),  # gate_words/up_words pointers
    s_waitcnt(simm16=0),
    # Pick output pointer and weight pointer from role.
    s_cmp_eq_u32(ssrc0=s[3], ssrc1=0),
    s_cselect_b32(sdst=s[8], ssrc0=s[4], ssrc1=s[6]),
    s_cselect_b32(sdst=s[9], ssrc0=s[5], ssrc1=s[7]),
    s_cselect_b32(sdst=s[16], ssrc0=s[12], ssrc1=s[14]),
    s_cselect_b32(sdst=s[17], ssrc0=s[13], ssrc1=s[15]),
    # output offset = row * 4
    v_mov_b32_e32(vdst=v[2], src0=s[2]),
    v_lshlrev_b32_e32(vdst=v[2], src0=2, vsrc1=v[2]),
    # q4 byte offset = row * 576 words * 4 bytes.
    s_mul_i32(sdst=s[10], ssrc0=s[2], ssrc1=Q4_WORDS_PER_ROW * 4),
    v_mov_b32_e32(vdst=v[6], src0=s[10]),
    global_load_b32(vdst=v[4], addr=v[6], saddr=s[16:17]),
    v_mov_b32_e32(vdst=v[3], src0=0),
    s_waitcnt(simm16=0),
    global_store_b32(addr=v[2], data=v[4], saddr=s[8:9]),
    s_endpgm(),
  ]
  sink = UOp.sink(gate.base, up.base, gate_words.base, up_words.base, q8.base, *gidxs, *lidxs,
                  arg=KernelInfo(name="q8_b2b_q4_load_skeleton"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))

def main() -> None:
  ap = argparse.ArgumentParser(description="B2b3 Q4_K load skeleton for AMD DSL fused q8 gate/up consumer")
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-codegen-transfer/asm_q4_load_skeleton.json"))
  args = ap.parse_args()

  gate = Tensor.empty(HIDDEN, dtype=dtypes.uint32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.uint32, device="AMD").contiguous()
  idx = np.arange(Q4_WORDS, dtype=np.uint32)
  gate_host = (idx * np.uint32(17) + np.uint32(3)).astype(np.uint32)
  up_host = (idx * np.uint32(29) + np.uint32(0x80000011)).astype(np.uint32)
  gate_words = Tensor(gate_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate, up, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=build_q4_load_skeleton)[:2]
  run_linear(gate.schedule_linear())

  got_gate, got_up = gate.numpy().astype(np.uint32), up.numpy().astype(np.uint32)
  rows = np.arange(HIDDEN, dtype=np.int64)
  exp_gate = gate_host[rows * Q4_WORDS_PER_ROW]
  exp_up = up_host[rows * Q4_WORDS_PER_ROW]
  bad_gate = np.flatnonzero(got_gate != exp_gate)
  bad_up = np.flatnonzero(got_up != exp_up)
  result = {
    "date": "2026-06-19",
    "phase": "B2b3_q4_load_skeleton",
    "route": "tinygrad_Ops.PROGRAM_AMD_DSL_q4_weight_pointer_and_row_stride",
    "q4_words_per_row": Q4_WORDS_PER_ROW,
    "gate_mismatches": int(bad_gate.size),
    "up_mismatches": int(bad_up.size),
    "first_gate_bad": int(bad_gate[0]) if bad_gate.size else None,
    "first_up_bad": int(bad_up[0]) if bad_up.size else None,
    "samples": {
      "gate": [int(x) for x in got_gate[:8]],
      "up": [int(x) for x in got_up[:8]],
      "expected_gate": [int(x) for x in exp_gate[:8]],
      "expected_up": [int(x) for x in exp_up[:8]],
    },
  }
  result["verdict"] = "PASS" if result["gate_mismatches"] == 0 and result["up_mismatches"] == 0 else "FAIL"
  result["next"] = "If PASS, proceed to Q4_K field/nibble diagnostics on real block layout, then one-block dot."
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))

if __name__ == "__main__":
  main()
