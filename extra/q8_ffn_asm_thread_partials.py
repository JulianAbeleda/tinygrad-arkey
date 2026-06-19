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
  global_load_b32, global_store_b32, s_add_u32, s_cmp_eq_u32, s_cselect_b32, s_endpgm, s_load_b128, s_load_b64,
  s_mul_i32, s_waitcnt, v_add_nc_u32_e32, v_and_b32_e32, v_cmp_ne_u32_e32, v_cndmask_b32_e32, v_dot4_i32_iu8,
  v_lshlrev_b32_e32, v_lshrrev_b32_e32, v_mov_b32_e32, v_mul_lo_u32, v_or_b32_e32,
)

HIDDEN, Q4_WORDS, Q8_BYTES = 12288, 7077888, 4608
Q4_BYTES_PER_ROW, Q4_BYTES_PER_BLOCK = 2304, 144

def build_thread_partials(gate:UOp, up:UOp, gate_words:UOp, up_words:UOp, q8:UOp) -> UOp:
  gidxs = [UOp.special(n, f"gidx{i}") for i, n in enumerate((HIDDEN, 2, 1))]
  lidxs = [UOp.special(n, f"lidx{i}") for i, n in enumerate((128, 1, 1))]
  insts = [
    s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL),
    s_load_b128(sdata=s[12:15], sbase=s[0:1], offset=0x10, soffset=NULL),
    s_load_b64(sdata=s[18:19], sbase=s[0:1], offset=0x20, soffset=NULL),
    s_waitcnt(simm16=0),
    s_cmp_eq_u32(ssrc0=s[3], ssrc1=0),
    s_cselect_b32(sdst=s[8], ssrc0=s[4], ssrc1=s[6]),
    s_cselect_b32(sdst=s[9], ssrc0=s[5], ssrc1=s[7]),
    s_cselect_b32(sdst=s[16], ssrc0=s[12], ssrc1=s[14]),
    s_cselect_b32(sdst=s[17], ssrc0=s[13], ssrc1=s[15]),
    # output offset = (row*128 + tid) * 4
    s_mul_i32(sdst=s[20], ssrc0=s[2], ssrc1=128),
    v_mov_b32_e32(vdst=v[2], src0=s[20]),
    v_add_nc_u32_e32(vdst=v[2], src0=v[0], vsrc1=v[2]),
    v_lshlrev_b32_e32(vdst=v[2], src0=2, vsrc1=v[2]),
    # tid decomposition: kb=tid>>3, sub=tid&7
    v_lshrrev_b32_e32(vdst=v[20], src0=3, vsrc1=v[0]),        # kb
    v_and_b32_e32(vdst=v[21], src0=7, vsrc1=v[0]),            # sub
    v_lshrrev_b32_e32(vdst=v[22], src0=1, vsrc1=v[21]),       # sub/2
    # q4 addr = row*2304 + kb*144 + 16 + (sub/2)*32
    s_mul_i32(sdst=s[21], ssrc0=s[2], ssrc1=Q4_BYTES_PER_ROW),
    v_mul_lo_u32(vdst=v[6], src0=v[20], src1=Q4_BYTES_PER_BLOCK),
    v_add_nc_u32_e32(vdst=v[6], src0=s[21], vsrc1=v[6]),
    v_lshlrev_b32_e32(vdst=v[22], src0=5, vsrc1=v[22]),
    v_add_nc_u32_e32(vdst=v[6], src0=v[22], vsrc1=v[6]),
    v_add_nc_u32_e32(vdst=v[6], src0=16, vsrc1=v[6]),
    # q8 addr = (kb*8 + sub)*36 + 4
    v_lshlrev_b32_e32(vdst=v[7], src0=3, vsrc1=v[20]),
    v_add_nc_u32_e32(vdst=v[7], src0=v[21], vsrc1=v[7]),
    v_mul_lo_u32(vdst=v[7], src0=v[7], src1=36),
    v_add_nc_u32_e32(vdst=v[7], src0=4, vsrc1=v[7]),
    v_mov_b32_e32(vdst=v[4], src0=0),
    v_mov_b32_e32(vdst=v[5], src0=0),
  ]
  for _ in range(8):
    insts += [
      global_load_b32(vdst=v[8], addr=v[6], saddr=s[16:17]),
      global_load_b32(vdst=v[9], addr=v[7], saddr=s[18:19]),
      s_waitcnt(simm16=0),
      v_lshrrev_b32_e32(vdst=v[10], src0=4, vsrc1=v[8]),
      v_and_b32_e32(vdst=v[10], src0=0x0f0f0f0f, vsrc1=v[10]),
      v_and_b32_e32(vdst=v[8], src0=0x0f0f0f0f, vsrc1=v[8]),
      v_and_b32_e32(vdst=v[11], src0=1, vsrc1=v[21]),
      v_cmp_ne_u32_e32(src0=0, vsrc1=v[11]),
      v_cndmask_b32_e32(vdst=v[8], src0=v[8], vsrc1=v[10]),
      v_dot4_i32_iu8(vdst=v[4], src0=v[8], src1=v[9], src2=v[4], neg=2),
      v_dot4_i32_iu8(vdst=v[5], src0=0x01010101, src1=v[9], src2=v[5], neg=2),
      v_add_nc_u32_e32(vdst=v[6], src0=4, vsrc1=v[6]),
      v_add_nc_u32_e32(vdst=v[7], src0=4, vsrc1=v[7]),
    ]
  insts += [
    v_and_b32_e32(vdst=v[4], src0=0xffff, vsrc1=v[4]),
    v_lshlrev_b32_e32(vdst=v[5], src0=16, vsrc1=v[5]),
    v_or_b32_e32(vdst=v[4], src0=v[5], vsrc1=v[4]),
    v_mov_b32_e32(vdst=v[3], src0=0),
    global_store_b32(addr=v[2], data=v[4], saddr=s[8:9]),
    s_endpgm(),
  ]
  sink = UOp.sink(gate.base, up.base, gate_words.base, up_words.base, q8.base, *gidxs, *lidxs,
                  arg=KernelInfo(name="q8_b2b_thread_partials"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))

def make_q4_words(mult:int, add:int) -> np.ndarray:
  b = ((np.arange(Q4_WORDS * 4, dtype=np.uint32) * np.uint32(mult) + np.uint32(add)) & np.uint32(0xff)).astype(np.uint8)
  return np.frombuffer(b.tobytes(), dtype=np.uint32).copy()

def make_q8() -> np.ndarray:
  vals = np.zeros(Q8_BYTES, dtype=np.uint8)
  for block in range(128):
    base = block * 36
    vals[base:base+2] = np.array([1.0], dtype=np.float16).view(np.uint8)
    signed = (((np.arange(32, dtype=np.int32) * 11 + block * 3) % 255) - 127).astype(np.int8)
    vals[base+4:base+36] = signed.view(np.uint8)
  return vals

def expected(words:np.ndarray, q8:np.ndarray, rows_check:int) -> np.ndarray:
  b, q8s = words.view(np.uint8), q8.view(np.int8)
  out = np.empty(rows_check * 128, dtype=np.uint32)
  for row in range(rows_check):
    for tid in range(128):
      kb, sub = tid >> 3, tid & 7
      q4 = row * Q4_BYTES_PER_ROW + kb * Q4_BYTES_PER_BLOCK + 16 + (sub >> 1) * 32
      q8o = (kb * 8 + sub) * 36 + 4
      sumi = sumq = 0
      for k in range(8):
        qword = b[q4 + k*4:q4 + k*4 + 4]
        qbytes = q8s[q8o + k*4:q8o + k*4 + 4].astype(np.int32)
        nib = ((qword >> (4 if sub & 1 else 0)) & 0x0f).astype(np.int32)
        sumi += int(np.sum(nib * qbytes))
        sumq += int(np.sum(qbytes))
      out[row*128 + tid] = np.uint32(((sumq & 0xffff) << 16) | (sumi & 0xffff))
  return out

def main() -> None:
  ap = argparse.ArgumentParser(description="B2b8 per-thread partial diagnostic for AMD DSL q8 gate/up consumer")
  ap.add_argument("--rows-check", type=int, default=128)
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-codegen-transfer/asm_thread_partials.json"))
  args = ap.parse_args()

  gate = Tensor.empty(HIDDEN * 128, dtype=dtypes.uint32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN * 128, dtype=dtypes.uint32, device="AMD").contiguous()
  gate_host, up_host = make_q4_words(37, 5), make_q4_words(53, 19)
  q8_host = make_q8()
  gate_words = Tensor(gate_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate, up, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=build_thread_partials)[:2]
  run_linear(gate.schedule_linear())

  n = args.rows_check * 128
  got_gate, got_up = gate.numpy().astype(np.uint32)[:n], up.numpy().astype(np.uint32)[:n]
  exp_gate, exp_up = expected(gate_host, q8_host, args.rows_check), expected(up_host, q8_host, args.rows_check)
  bad_gate = np.flatnonzero(got_gate != exp_gate)
  bad_up = np.flatnonzero(got_up != exp_up)
  result = {
    "date": "2026-06-19",
    "phase": "B2b8_thread_partials",
    "route": "tinygrad_Ops.PROGRAM_AMD_DSL_128_thread_q4k_q8_partials",
    "coverage": f"first {args.rows_check} rows, all 128 tid lanes, signed q8, low/high nibbles",
    "gate_mismatches": int(bad_gate.size),
    "up_mismatches": int(bad_up.size),
    "first_gate_bad": int(bad_gate[0]) if bad_gate.size else None,
    "first_up_bad": int(bad_up[0]) if bad_up.size else None,
    "samples": {
      "gate": [int(x) for x in got_gate[:16]],
      "up": [int(x) for x in got_up[:16]],
      "expected_gate": [int(x) for x in exp_gate[:16]],
      "expected_up": [int(x) for x in exp_up[:16]],
    },
  }
  result["verdict"] = "PASS" if result["gate_mismatches"] == 0 and result["up_mismatches"] == 0 else "FAIL"
  result["next"] = "If PASS, proceed to workgroup reduction/full-row output."
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))

if __name__ == "__main__":
  main()
