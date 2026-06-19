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
  global_load_b32, global_load_u16, global_load_u8, global_store_b32, s_add_u32, s_and_b32, s_cmp_eq_u32,
  s_cselect_b32, s_endpgm, s_load_b128, s_load_b64, s_mul_i32, s_waitcnt, v_and_b32_e32, v_cvt_f32_f16_e32,
  v_cvt_f32_i32_e32, v_cvt_f32_u32_e32, v_dot4_i32_iu8, v_lshlrev_b32_e32, v_lshrrev_b32_e32, v_mov_b32_e32,
  v_mul_f32_e32, v_sub_f32_e32,
)

HIDDEN, Q4_WORDS, Q8_BYTES = 12288, 7077888, 4608
Q4_BYTES_PER_ROW, Q4_BYTES_PER_BLOCK = 2304, 144

def build_scaled_subblock(gate:UOp, up:UOp, gate_words:UOp, up_words:UOp, q8:UOp) -> UOp:
  gidxs = [UOp.special(n, f"gidx{i}") for i, n in enumerate((HIDDEN, 2, 1))]
  lidxs = [UOp.special(n, f"lidx{i}") for i, n in enumerate((32, 4, 1))]
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
    v_mov_b32_e32(vdst=v[2], src0=s[2]),
    v_lshlrev_b32_e32(vdst=v[2], src0=2, vsrc1=v[2]),
    # kb = row & 15, sub fixed to 1.
    s_and_b32(sdst=s[20], ssrc0=s[2], ssrc1=15),
    s_mul_i32(sdst=s[21], ssrc0=s[2], ssrc1=Q4_BYTES_PER_ROW),
    s_mul_i32(sdst=s[22], ssrc0=s[20], ssrc1=Q4_BYTES_PER_BLOCK),
    s_add_u32(sdst=s[21], ssrc0=s[21], ssrc1=s[22]),          # q4 block base
    s_mul_i32(sdst=s[23], ssrc0=s[20], ssrc1=8 * 36),
    s_add_u32(sdst=s[23], ssrc0=s[23], ssrc1=36),             # q8 block base for sub=1
    # Load d, dmin, sc, mn, d8.
    v_mov_b32_e32(vdst=v[6], src0=s[21]),
    global_load_u16(vdst=v[24], addr=v[6], saddr=s[16:17]),
    s_add_u32(sdst=s[24], ssrc0=s[21], ssrc1=2),
    v_mov_b32_e32(vdst=v[6], src0=s[24]),
    global_load_u16(vdst=v[25], addr=v[6], saddr=s[16:17]),
    s_add_u32(sdst=s[24], ssrc0=s[21], ssrc1=5),              # scales[1]
    v_mov_b32_e32(vdst=v[6], src0=s[24]),
    global_load_u8(vdst=v[26], addr=v[6], saddr=s[16:17]),
    s_add_u32(sdst=s[24], ssrc0=s[21], ssrc1=9),              # scales[1+4]
    v_mov_b32_e32(vdst=v[6], src0=s[24]),
    global_load_u8(vdst=v[27], addr=v[6], saddr=s[16:17]),
    v_mov_b32_e32(vdst=v[7], src0=s[23]),
    global_load_u16(vdst=v[28], addr=v[7], saddr=s[18:19]),
    s_waitcnt(simm16=0),
    v_cvt_f32_f16_e32(vdst=v[24], src0=v[24]),
    v_cvt_f32_f16_e32(vdst=v[25], src0=v[25]),
    v_cvt_f32_u32_e32(vdst=v[26], src0=v[26]),
    v_cvt_f32_u32_e32(vdst=v[27], src0=v[27]),
    v_cvt_f32_f16_e32(vdst=v[28], src0=v[28]),
    s_add_u32(sdst=s[21], ssrc0=s[21], ssrc1=16),             # q4 qs offset
    s_add_u32(sdst=s[23], ssrc0=s[23], ssrc1=4),              # q8 qs offset
    v_mov_b32_e32(vdst=v[4], src0=0),
    v_mov_b32_e32(vdst=v[5], src0=0),
  ]
  for _ in range(8):
    insts += [
      v_mov_b32_e32(vdst=v[6], src0=s[21]),
      v_mov_b32_e32(vdst=v[7], src0=s[23]),
      global_load_b32(vdst=v[8], addr=v[6], saddr=s[16:17]),
      global_load_b32(vdst=v[9], addr=v[7], saddr=s[18:19]),
      s_waitcnt(simm16=0),
      v_lshrrev_b32_e32(vdst=v[8], src0=4, vsrc1=v[8]),
      v_and_b32_e32(vdst=v[8], src0=0x0f0f0f0f, vsrc1=v[8]),
      v_dot4_i32_iu8(vdst=v[4], src0=v[8], src1=v[9], src2=v[4], neg=2),
      v_dot4_i32_iu8(vdst=v[5], src0=0x01010101, src1=v[9], src2=v[5], neg=2),
      s_add_u32(sdst=s[21], ssrc0=s[21], ssrc1=4),
      s_add_u32(sdst=s[23], ssrc0=s[23], ssrc1=4),
    ]
  insts += [
    v_cvt_f32_i32_e32(vdst=v[4], src0=v[4]),                 # sumi
    v_cvt_f32_i32_e32(vdst=v[5], src0=v[5]),                 # sumq
    v_mul_f32_e32(vdst=v[10], src0=v[24], vsrc1=v[26]),      # d * sc
    v_mul_f32_e32(vdst=v[10], src0=v[10], vsrc1=v[4]),       # * sumi
    v_mul_f32_e32(vdst=v[11], src0=v[25], vsrc1=v[27]),      # dmin * mn
    v_mul_f32_e32(vdst=v[11], src0=v[11], vsrc1=v[5]),       # * sumq
    v_sub_f32_e32(vdst=v[10], src0=v[10], vsrc1=v[11]),      # term0 - term1
    v_mul_f32_e32(vdst=v[10], src0=v[28], vsrc1=v[10]),      # * d8
    v_mov_b32_e32(vdst=v[3], src0=0),
    global_store_b32(addr=v[2], data=v[10], saddr=s[8:9]),
    s_endpgm(),
  ]
  sink = UOp.sink(gate.base, up.base, gate_words.base, up_words.base, q8.base, *gidxs, *lidxs,
                  arg=KernelInfo(name="q8_b2b_scaled_subblock"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))

def make_q4_words(mult:int, add:int) -> np.ndarray:
  b = ((np.arange(Q4_WORDS * 4, dtype=np.uint32) * np.uint32(mult) + np.uint32(add)) & np.uint32(0xff)).astype(np.uint8)
  # Keep half scale fields finite and small.
  for row in range(HIDDEN):
    for kb in range(16):
      base = row * Q4_BYTES_PER_ROW + kb * Q4_BYTES_PER_BLOCK
      b[base:base+2] = np.array([0.03125 + ((row + kb) % 5) * 0.00390625], dtype=np.float16).view(np.uint8)
      b[base+2:base+4] = np.array([0.015625 + ((row + kb) % 3) * 0.001953125], dtype=np.float16).view(np.uint8)
  return np.frombuffer(b.tobytes(), dtype=np.uint32).copy()

def make_q8() -> np.ndarray:
  vals = np.zeros(Q8_BYTES, dtype=np.uint8)
  for block in range(128):
    base = block * 36
    vals[base:base+2] = np.array([0.0078125 + (block % 4) * 0.001953125], dtype=np.float16).view(np.uint8)
    signed = (((np.arange(32, dtype=np.int32) * 11 + block * 3) % 255) - 127).astype(np.int8)
    vals[base+4:base+36] = signed.view(np.uint8)
  return vals

def expected(words:np.ndarray, q8:np.ndarray) -> np.ndarray:
  b = words.view(np.uint8)
  q8s = q8.view(np.int8)
  out = np.empty(HIDDEN, dtype=np.float32)
  for row in range(HIDDEN):
    kb, sumi, sumq = row & 15, 0, 0
    base = row * Q4_BYTES_PER_ROW + kb * Q4_BYTES_PER_BLOCK
    q8base = (kb * 8 + 1) * 36
    d = np.frombuffer(b[base:base+2].tobytes(), dtype=np.float16).astype(np.float32)[0]
    dmin = np.frombuffer(b[base+2:base+4].tobytes(), dtype=np.float16).astype(np.float32)[0]
    sc, mn = np.float32(b[base+5]), np.float32(b[base+9])
    d8 = np.frombuffer(q8[q8base:q8base+2].tobytes(), dtype=np.float16).astype(np.float32)[0]
    q4, q8o = base + 16, q8base + 4
    for k in range(8):
      qword = b[q4 + k*4:q4 + k*4 + 4]
      qbytes = q8s[q8o + k*4:q8o + k*4 + 4].astype(np.int32)
      nibbles = ((qword >> 4) & 0x0f).astype(np.int32)
      sumi += int(np.sum(nibbles * qbytes))
      sumq += int(np.sum(qbytes))
    out[row] = np.float32(d8 * (d * sc * np.float32(sumi) - dmin * mn * np.float32(sumq)))
  return out

def main() -> None:
  ap = argparse.ArgumentParser(description="B2b7 scaled one-subblock Q4_K x signed q8 contribution")
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-codegen-transfer/asm_scaled_subblock.json"))
  args = ap.parse_args()

  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_host, up_host = make_q4_words(37, 5), make_q4_words(53, 19)
  q8_host = make_q8()
  gate_words = Tensor(gate_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate, up, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=build_scaled_subblock)[:2]
  run_linear(gate.schedule_linear())

  got_gate, got_up = gate.numpy().astype(np.float32), up.numpy().astype(np.float32)
  exp_gate, exp_up = expected(gate_host, q8_host), expected(up_host, q8_host)
  gate_abs, up_abs = np.abs(got_gate - exp_gate), np.abs(got_up - exp_up)
  result = {
    "date": "2026-06-19",
    "phase": "B2b7_scaled_subblock",
    "route": "tinygrad_Ops.PROGRAM_AMD_DSL_scaled_q4k_subblock_contribution",
    "coverage": "sub=1, kb=row&15, d/dmin/sc/mn/d8, signed q8, both gate/up",
    "gate_max_abs": float(gate_abs.max()),
    "up_max_abs": float(up_abs.max()),
    "gate_mean_abs": float(gate_abs.mean()),
    "up_mean_abs": float(up_abs.mean()),
    "samples": {
      "gate": [float(x) for x in got_gate[:8]],
      "up": [float(x) for x in got_up[:8]],
      "expected_gate": [float(x) for x in exp_gate[:8]],
      "expected_up": [float(x) for x in exp_up[:8]],
    },
  }
  result["verdict"] = "PASS" if result["gate_max_abs"] <= 1e-5 and result["up_max_abs"] <= 1e-5 else "FAIL"
  result["next"] = "If PASS, proceed to full-row single-role dot with all kb/sub contributions."
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))

if __name__ == "__main__":
  main()
