#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib

import numpy as np

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from tinygrad.renderer.amd.dsl import NULL, s, v
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  ds_bpermute_b32, ds_load_b32, ds_store_b32, global_load_b32, global_load_u16, global_load_u8, global_store_b32,
  s_add_u32, s_barrier, s_cmp_eq_u32, s_cselect_b32, s_endpgm, s_load_b128, s_load_b64, s_mul_i32, s_waitcnt,
  v_add_f32_e32, v_add_nc_u32_e32, v_and_b32_e32, v_cmp_gt_u32_e32, v_cmp_ne_u32_e32, v_cndmask_b32_e32,
  v_cvt_f32_f16_e32, v_cvt_f32_i32_e32, v_cvt_f32_u32_e32, v_dot4_i32_iu8, v_lshlrev_b32_e32, v_lshrrev_b32_e32,
  v_mov_b32_e32, v_mul_f32_e32, v_mul_lo_u32, v_or_b32_e32, v_sub_f32_e32, v_xor_b32_e32,
)

HIDDEN, Q4_WORDS, Q8_BYTES = 12288, 7077888, 4608
Q4_BYTES_PER_ROW, Q4_BYTES_PER_BLOCK = 2304, 144

def add_scaled_partial(insts:list) -> None:
  # tid decomposition: kb=v20, sub=v21, sub/2=v22, q4 blockbase=v23, q8 blockbase=v24.
  insts += [
    v_lshrrev_b32_e32(vdst=v[20], src0=3, vsrc1=v[0]),
    v_and_b32_e32(vdst=v[21], src0=7, vsrc1=v[0]),
    v_lshrrev_b32_e32(vdst=v[22], src0=1, vsrc1=v[21]),
    s_mul_i32(sdst=s[21], ssrc0=s[2], ssrc1=Q4_BYTES_PER_ROW),
    v_mul_lo_u32(vdst=v[23], src0=v[20], src1=Q4_BYTES_PER_BLOCK),
    v_add_nc_u32_e32(vdst=v[23], src0=s[21], vsrc1=v[23]),
    v_lshlrev_b32_e32(vdst=v[24], src0=3, vsrc1=v[20]),
    v_add_nc_u32_e32(vdst=v[24], src0=v[21], vsrc1=v[24]),
    v_mul_lo_u32(vdst=v[24], src0=v[24], src1=36),
    # Load d/dmin/d8.
    global_load_u16(vdst=v[30], addr=v[23], saddr=s[16:17]),
    v_add_nc_u32_e32(vdst=v[6], src0=2, vsrc1=v[23]),
    global_load_u16(vdst=v[31], addr=v[6], saddr=s[16:17]),
    global_load_u16(vdst=v[32], addr=v[24], saddr=s[18:19]),
    # Load qsub, qsub4, qsubm4 from scales.
    v_add_nc_u32_e32(vdst=v[6], src0=4, vsrc1=v[23]),
    v_add_nc_u32_e32(vdst=v[6], src0=v[21], vsrc1=v[6]),
    global_load_u8(vdst=v[33], addr=v[6], saddr=s[16:17]),      # q[sub]
    v_add_nc_u32_e32(vdst=v[6], src0=8, vsrc1=v[23]),
    v_add_nc_u32_e32(vdst=v[6], src0=v[21], vsrc1=v[6]),
    global_load_u8(vdst=v[34], addr=v[6], saddr=s[16:17]),      # q[sub+4]
    v_and_b32_e32(vdst=v[35], src0=3, vsrc1=v[21]),
    v_add_nc_u32_e32(vdst=v[6], src0=4, vsrc1=v[23]),
    v_add_nc_u32_e32(vdst=v[6], src0=v[35], vsrc1=v[6]),
    global_load_u8(vdst=v[35], addr=v[6], saddr=s[16:17]),      # q[sub&3]
    s_waitcnt(simm16=0),
    # sc/mn candidates.
    v_and_b32_e32(vdst=v[36], src0=63, vsrc1=v[33]),            # sc_lt4
    v_and_b32_e32(vdst=v[37], src0=63, vsrc1=v[34]),            # mn_lt4
    v_and_b32_e32(vdst=v[38], src0=15, vsrc1=v[34]),
    v_lshrrev_b32_e32(vdst=v[39], src0=6, vsrc1=v[35]),
    v_lshlrev_b32_e32(vdst=v[39], src0=4, vsrc1=v[39]),
    v_or_b32_e32(vdst=v[38], src0=v[39], vsrc1=v[38]),          # sc_ge4
    v_lshrrev_b32_e32(vdst=v[39], src0=4, vsrc1=v[34]),
    v_lshrrev_b32_e32(vdst=v[40], src0=6, vsrc1=v[33]),
    v_lshlrev_b32_e32(vdst=v[40], src0=4, vsrc1=v[40]),
    v_or_b32_e32(vdst=v[39], src0=v[40], vsrc1=v[39]),          # mn_ge4
    v_mov_b32_e32(vdst=v[41], src0=3),
    v_cmp_gt_u32_e32(src0=v[21], vsrc1=v[41]),
    v_cndmask_b32_e32(vdst=v[36], src0=v[36], vsrc1=v[38]),     # sc
    v_cndmask_b32_e32(vdst=v[37], src0=v[37], vsrc1=v[39]),     # mn
    v_cvt_f32_f16_e32(vdst=v[30], src0=v[30]),
    v_cvt_f32_f16_e32(vdst=v[31], src0=v[31]),
    v_cvt_f32_f16_e32(vdst=v[32], src0=v[32]),
    v_cvt_f32_u32_e32(vdst=v[36], src0=v[36]),
    v_cvt_f32_u32_e32(vdst=v[37], src0=v[37]),
    # q4 qs addr and q8 qs addr.
    v_lshlrev_b32_e32(vdst=v[22], src0=5, vsrc1=v[22]),
    v_add_nc_u32_e32(vdst=v[23], src0=16, vsrc1=v[23]),
    v_add_nc_u32_e32(vdst=v[23], src0=v[22], vsrc1=v[23]),
    v_add_nc_u32_e32(vdst=v[24], src0=4, vsrc1=v[24]),
    v_mov_b32_e32(vdst=v[4], src0=0),
    v_mov_b32_e32(vdst=v[5], src0=0),
  ]
  for _ in range(8):
    insts += [
      global_load_b32(vdst=v[8], addr=v[23], saddr=s[16:17]),
      global_load_b32(vdst=v[9], addr=v[24], saddr=s[18:19]),
      s_waitcnt(simm16=0),
      v_lshrrev_b32_e32(vdst=v[10], src0=4, vsrc1=v[8]),
      v_and_b32_e32(vdst=v[10], src0=0x0f0f0f0f, vsrc1=v[10]),
      v_and_b32_e32(vdst=v[8], src0=0x0f0f0f0f, vsrc1=v[8]),
      v_and_b32_e32(vdst=v[11], src0=1, vsrc1=v[21]),
      v_cmp_ne_u32_e32(src0=0, vsrc1=v[11]),
      v_cndmask_b32_e32(vdst=v[8], src0=v[8], vsrc1=v[10]),
      v_dot4_i32_iu8(vdst=v[4], src0=v[8], src1=v[9], src2=v[4], neg=2),
      v_dot4_i32_iu8(vdst=v[5], src0=0x01010101, src1=v[9], src2=v[5], neg=2),
      v_add_nc_u32_e32(vdst=v[23], src0=4, vsrc1=v[23]),
      v_add_nc_u32_e32(vdst=v[24], src0=4, vsrc1=v[24]),
    ]
  insts += [
    v_cvt_f32_i32_e32(vdst=v[4], src0=v[4]),
    v_cvt_f32_i32_e32(vdst=v[5], src0=v[5]),
    v_mul_f32_e32(vdst=v[10], src0=v[30], vsrc1=v[36]),
    v_mul_f32_e32(vdst=v[10], src0=v[10], vsrc1=v[4]),
    v_mul_f32_e32(vdst=v[11], src0=v[31], vsrc1=v[37]),
    v_mul_f32_e32(vdst=v[11], src0=v[11], vsrc1=v[5]),
    v_sub_f32_e32(vdst=v[10], src0=v[10], vsrc1=v[11]),
    v_mul_f32_e32(vdst=v[10], src0=v[32], vsrc1=v[10]),
  ]

def build_fullrow_reduce(gate:UOp, up:UOp, gate_words:UOp, up_words:UOp, q8:UOp) -> UOp:
  gidxs = [UOp.special(n, f"gidx{i}") for i, n in enumerate((HIDDEN, 2, 1))]
  lidxs = [UOp.special(n, f"lidx{i}") for i, n in enumerate((128, 1, 1))]
  lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=16, addrspace=AddrSpace.LOCAL), (), "lds")
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
  ]
  add_scaled_partial(insts)
  # Wave32 reduction using ds_bpermute. v10 holds partial.
  insts += [v_and_b32_e32(vdst=v[50], src0=31, vsrc1=v[0])]
  for off in [16, 8, 4, 2, 1]:
    insts += [
      v_xor_b32_e32(vdst=v[51], src0=off, vsrc1=v[50]),
      v_lshlrev_b32_e32(vdst=v[51], src0=2, vsrc1=v[51]),
      ds_bpermute_b32(vdst=v[52], addr=v[51], data0=v[10]),
      s_waitcnt(simm16=0),
      v_add_f32_e32(vdst=v[10], src0=v[52], vsrc1=v[10]),
    ]
  # Store per-wave sums to LDS slots. All lanes in the wave store the same value to the same slot.
  insts += [
    v_lshrrev_b32_e32(vdst=v[53], src0=5, vsrc1=v[0]),
    v_lshlrev_b32_e32(vdst=v[53], src0=2, vsrc1=v[53]),
    ds_store_b32(addr=v[53], data0=v[10]),
    s_waitcnt(simm16=0),
    s_barrier(),
    v_mov_b32_e32(vdst=v[54], src0=0),
    ds_load_b32(vdst=v[10], addr=v[54]),
    v_mov_b32_e32(vdst=v[54], src0=4),
    ds_load_b32(vdst=v[11], addr=v[54]),
    v_mov_b32_e32(vdst=v[54], src0=8),
    ds_load_b32(vdst=v[12], addr=v[54]),
    v_mov_b32_e32(vdst=v[54], src0=12),
    ds_load_b32(vdst=v[13], addr=v[54]),
    s_waitcnt(simm16=0),
    v_add_f32_e32(vdst=v[10], src0=v[11], vsrc1=v[10]),
    v_add_f32_e32(vdst=v[12], src0=v[13], vsrc1=v[12]),
    v_add_f32_e32(vdst=v[10], src0=v[12], vsrc1=v[10]),
    v_mov_b32_e32(vdst=v[2], src0=s[2]),
    v_lshlrev_b32_e32(vdst=v[2], src0=2, vsrc1=v[2]),
    v_mov_b32_e32(vdst=v[3], src0=0),
    global_store_b32(addr=v[2], data=v[10], saddr=s[8:9]),
    s_endpgm(),
  ]
  sink = UOp.sink(gate.base, up.base, gate_words.base, up_words.base, q8.base, lds, *gidxs, *lidxs,
                  arg=KernelInfo(name="q8_b2b_fullrow_reduce"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))

def make_q4_words(mult:int, add:int) -> np.ndarray:
  b = ((np.arange(Q4_WORDS * 4, dtype=np.uint32) * np.uint32(mult) + np.uint32(add)) & np.uint32(0xff)).astype(np.uint8)
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

def get_scale_min(j:int, q:np.ndarray) -> tuple[int, int]:
  if j < 4: return int(q[j] & 63), int(q[j+4] & 63)
  return int((q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)), int((q[j+4] >> 4) | ((q[j] >> 6) << 4))

def expected(words:np.ndarray, q8:np.ndarray, rows_check:int) -> np.ndarray:
  b, q8s = words.view(np.uint8), q8.view(np.int8)
  out = np.empty(rows_check, dtype=np.float32)
  for row in range(rows_check):
    total = np.float32(0.0)
    for kb in range(16):
      base = row * Q4_BYTES_PER_ROW + kb * Q4_BYTES_PER_BLOCK
      d = np.frombuffer(b[base:base+2].tobytes(), dtype=np.float16).astype(np.float32)[0]
      dmin = np.frombuffer(b[base+2:base+4].tobytes(), dtype=np.float16).astype(np.float32)[0]
      scales = b[base+4:base+16]
      for sub in range(8):
        sc, mn = get_scale_min(sub, scales)
        q4 = base + 16 + (sub >> 1) * 32
        q8base = (kb * 8 + sub) * 36
        d8 = np.frombuffer(q8[q8base:q8base+2].tobytes(), dtype=np.float16).astype(np.float32)[0]
        sumi = sumq = 0
        for k in range(8):
          qword = b[q4 + k*4:q4 + k*4 + 4]
          qbytes = q8s[q8base + 4 + k*4:q8base + 8 + k*4].astype(np.int32)
          nib = ((qword >> (4 if sub & 1 else 0)) & 0x0f).astype(np.int32)
          sumi += int(np.sum(nib * qbytes)); sumq += int(np.sum(qbytes))
        total = np.float32(total + np.float32(d8 * (d * np.float32(sc) * np.float32(sumi) - dmin * np.float32(mn) * np.float32(sumq))))
    out[row] = total
  return out

def main() -> None:
  ap = argparse.ArgumentParser(description="B2b9 full-row reduction diagnostic for AMD DSL q8 gate/up consumer")
  ap.add_argument("--rows-check", type=int, default=128)
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-codegen-transfer/asm_fullrow_reduce.json"))
  args = ap.parse_args()

  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_host, up_host = make_q4_words(37, 5), make_q4_words(53, 19)
  q8_host = make_q8()
  gate_words = Tensor(gate_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate, up, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=build_fullrow_reduce)[:2]
  run_linear(gate.schedule_linear())

  got_gate, got_up = gate.numpy().astype(np.float32)[:args.rows_check], up.numpy().astype(np.float32)[:args.rows_check]
  exp_gate, exp_up = expected(gate_host, q8_host, args.rows_check), expected(up_host, q8_host, args.rows_check)
  gate_abs, up_abs = np.abs(got_gate - exp_gate), np.abs(got_up - exp_up)
  result = {
    "date": "2026-06-19",
    "phase": "B2b9_fullrow_reduce",
    "route": "tinygrad_Ops.PROGRAM_AMD_DSL_fullrow_q4k_q8_reduce",
    "coverage": f"first {args.rows_check} rows, all 128 tid lanes reduced, both gate/up",
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
  result["verdict"] = "PASS" if result["gate_max_abs"] <= 1e-3 and result["up_max_abs"] <= 1e-3 else "FAIL"
  result["next"] = "If PASS, proceed to real-GGUF full fused gate/up correctness and timing."
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))

if __name__ == "__main__":
  main()
