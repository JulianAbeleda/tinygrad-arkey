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
  global_load_b32, global_load_u8, global_store_b32, s_add_u32, s_and_b32, s_cmp_eq_u32, s_cselect_b32,
  s_endpgm, s_load_b128, s_lshr_b32, s_mul_i32, s_waitcnt, v_and_b32_e32, v_lshlrev_b32_e32,
  v_mov_b32_e32, v_or_b32_e32,
)

HIDDEN, Q4_WORDS, Q8_BYTES = 12288, 7077888, 4608
Q4_WORDS_PER_ROW, Q4_BYTES_PER_ROW, Q4_BYTES_PER_BLOCK = 576, 2304, 144

def build_q4_field_skeleton(gate:UOp, up:UOp, gate_words:UOp, up_words:UOp, q8:UOp) -> UOp:
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
    # Diagnostic coverage: kb = row & 15, sub = (row >> 4) & 7.
    s_mul_i32(sdst=s[20], ssrc0=s[2], ssrc1=Q4_BYTES_PER_ROW),
    s_and_b32(sdst=s[21], ssrc0=s[2], ssrc1=15),
    s_mul_i32(sdst=s[21], ssrc0=s[21], ssrc1=Q4_BYTES_PER_BLOCK),
    s_add_u32(sdst=s[20], ssrc0=s[20], ssrc1=s[21]),
    s_lshr_b32(sdst=s[22], ssrc0=s[2], ssrc1=4),
    s_and_b32(sdst=s[22], ssrc0=s[22], ssrc1=7),
    # scale byte offset = rowbase + kb*144 + 4 + sub
    s_add_u32(sdst=s[23], ssrc0=s[20], ssrc1=4),
    s_add_u32(sdst=s[23], ssrc0=s[23], ssrc1=s[22]),
    v_mov_b32_e32(vdst=v[6], src0=s[23]),
    global_load_u8(vdst=v[4], addr=v[6], saddr=s[16:17]),
    # qs word offset = rowbase + kb*144 + 16 + (sub >> 1) * 32
    s_lshr_b32(sdst=s[24], ssrc0=s[22], ssrc1=1),
    s_mul_i32(sdst=s[24], ssrc0=s[24], ssrc1=32),
    s_add_u32(sdst=s[24], ssrc0=s[24], ssrc1=16),
    s_add_u32(sdst=s[24], ssrc0=s[24], ssrc1=s[20]),
    v_mov_b32_e32(vdst=v[7], src0=s[24]),
    global_load_b32(vdst=v[5], addr=v[7], saddr=s[16:17]),
    v_mov_b32_e32(vdst=v[3], src0=0),
    s_waitcnt(simm16=0),
    # Pack diagnostic: high byte is scale byte, low 24 bits are first qs word low bytes.
    v_lshlrev_b32_e32(vdst=v[4], src0=24, vsrc1=v[4]),
    v_and_b32_e32(vdst=v[5], src0=0x00ffffff, vsrc1=v[5]),
    v_or_b32_e32(vdst=v[4], src0=v[5], vsrc1=v[4]),
    global_store_b32(addr=v[2], data=v[4], saddr=s[8:9]),
    s_endpgm(),
  ]
  sink = UOp.sink(gate.base, up.base, gate_words.base, up_words.base, q8.base, *gidxs, *lidxs,
                  arg=KernelInfo(name="q8_b2b_q4_field_skeleton"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))

def make_q4_words(mult:int, add:int) -> np.ndarray:
  # Synthetic bytes expose byte-addressing bugs better than word-patterned data.
  b = ((np.arange(Q4_WORDS * 4, dtype=np.uint32) * np.uint32(mult) + np.uint32(add)) & np.uint32(0xff)).astype(np.uint8)
  return np.frombuffer(b.tobytes(), dtype=np.uint32).copy()

def expected(words:np.ndarray) -> np.ndarray:
  b = words.view(np.uint8)
  out = np.empty(HIDDEN, dtype=np.uint32)
  for row in range(HIDDEN):
    kb, sub = row & 15, (row >> 4) & 7
    base = row * Q4_BYTES_PER_ROW + kb * Q4_BYTES_PER_BLOCK
    scale = int(b[base + 4 + sub])
    qoff = base + 16 + (sub >> 1) * 32
    qword = int(np.frombuffer(b[qoff:qoff+4].tobytes(), dtype=np.uint32)[0])
    out[row] = np.uint32((scale << 24) | (qword & 0x00ffffff))
  return out

def main() -> None:
  ap = argparse.ArgumentParser(description="B2b4 Q4_K field skeleton for AMD DSL fused q8 gate/up consumer")
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-codegen-transfer/asm_q4_field_skeleton.json"))
  args = ap.parse_args()

  gate = Tensor.empty(HIDDEN, dtype=dtypes.uint32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.uint32, device="AMD").contiguous()
  gate_host, up_host = make_q4_words(37, 5), make_q4_words(53, 19)
  gate_words = Tensor(gate_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate, up, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=build_q4_field_skeleton)[:2]
  run_linear(gate.schedule_linear())

  got_gate, got_up = gate.numpy().astype(np.uint32), up.numpy().astype(np.uint32)
  exp_gate, exp_up = expected(gate_host), expected(up_host)
  bad_gate = np.flatnonzero(got_gate != exp_gate)
  bad_up = np.flatnonzero(got_up != exp_up)
  result = {
    "date": "2026-06-19",
    "phase": "B2b4_q4_field_skeleton",
    "route": "tinygrad_Ops.PROGRAM_AMD_DSL_q4_scale_and_qs_field_loads",
    "coverage": "kb=row&15, sub=(row>>4)&7, both gate/up pointers",
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
  result["next"] = "If PASS, proceed to one-subblock dot diagnostic."
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))

if __name__ == "__main__":
  main()
