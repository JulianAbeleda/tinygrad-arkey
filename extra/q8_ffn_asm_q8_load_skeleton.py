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
  global_load_u8, global_store_b32, s_add_u32, s_and_b32, s_cmp_eq_u32, s_cselect_b32, s_endpgm,
  s_load_b128, s_load_b64, s_mul_i32, s_waitcnt, v_lshlrev_b32_e32, v_mov_b32_e32,
)

HIDDEN, Q4_WORDS, Q8_BYTES = 12288, 7077888, 4608

def build_q8_load_skeleton(gate:UOp, up:UOp, gate_words:UOp, up_words:UOp, q8:UOp) -> UOp:
  gidxs = [UOp.special(n, f"gidx{i}") for i, n in enumerate((HIDDEN, 2, 1))]
  lidxs = [UOp.special(n, f"lidx{i}") for i, n in enumerate((32, 4, 1))]
  insts = [
    s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL),      # gate/up pointers
    s_load_b64(sdata=s[12:13], sbase=s[0:1], offset=0x20, soffset=NULL),  # q8 pointer
    s_waitcnt(simm16=0),
    # Pick output pointer from role.
    s_cmp_eq_u32(ssrc0=s[3], ssrc1=0),
    s_cselect_b32(sdst=s[8], ssrc0=s[4], ssrc1=s[6]),
    s_cselect_b32(sdst=s[9], ssrc0=s[5], ssrc1=s[7]),
    # output offset = row * 4
    v_mov_b32_e32(vdst=v[2], src0=s[2]),
    v_lshlrev_b32_e32(vdst=v[2], src0=2, vsrc1=v[2]),
    # q8 byte offset = (row & 127) * 36 + 4 + role. This samples qs[0] for gate and qs[1] for up.
    s_and_b32(sdst=s[10], ssrc0=s[2], ssrc1=127),
    s_mul_i32(sdst=s[10], ssrc0=s[10], ssrc1=36),
    s_add_u32(sdst=s[10], ssrc0=s[10], ssrc1=4),
    s_add_u32(sdst=s[10], ssrc0=s[10], ssrc1=s[3]),
    v_mov_b32_e32(vdst=v[6], src0=s[10]),
    global_load_u8(vdst=v[4], addr=v[6], saddr=s[12:13]),
    v_mov_b32_e32(vdst=v[3], src0=0),
    s_waitcnt(simm16=0),
    global_store_b32(addr=v[2], data=v[4], saddr=s[8:9]),
    s_endpgm(),
  ]
  sink = UOp.sink(gate.base, up.base, gate_words.base, up_words.base, q8.base, *gidxs, *lidxs,
                  arg=KernelInfo(name="q8_b2b_q8_load_skeleton"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))

def main() -> None:
  ap = argparse.ArgumentParser(description="B2b2 q8 load skeleton for AMD DSL fused q8 gate/up consumer")
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-codegen-transfer/asm_q8_load_skeleton.json"))
  args = ap.parse_args()

  gate = Tensor.empty(HIDDEN, dtype=dtypes.uint32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.uint32, device="AMD").contiguous()
  gate_words = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8_host = (np.arange(Q8_BYTES, dtype=np.uint32) % 251).astype(np.uint8)
  q8 = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate, up, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=build_q8_load_skeleton)[:2]
  run_linear(gate.schedule_linear())

  got_gate, got_up = gate.numpy().astype(np.uint32), up.numpy().astype(np.uint32)
  rows = np.arange(HIDDEN, dtype=np.uint32)
  exp_gate = q8_host[((rows & 127) * 36 + 4).astype(np.int64)].astype(np.uint32)
  exp_up = q8_host[((rows & 127) * 36 + 5).astype(np.int64)].astype(np.uint32)
  bad_gate = np.flatnonzero(got_gate != exp_gate)
  bad_up = np.flatnonzero(got_up != exp_up)
  result = {
    "date": "2026-06-19",
    "phase": "B2b2_q8_load_skeleton",
    "route": "tinygrad_Ops.PROGRAM_AMD_DSL_q8_global_load_u8",
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
  result["next"] = "If PASS, proceed to Q4_K load skeleton."
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))

if __name__ == "__main__":
  main()
