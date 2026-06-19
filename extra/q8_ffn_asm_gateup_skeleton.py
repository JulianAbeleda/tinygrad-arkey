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
  global_store_b32, s_cmp_eq_u32, s_cselect_b32, s_endpgm, s_load_b128, s_waitcnt,
  v_add_nc_u32_e32, v_lshlrev_b32_e32, v_mov_b32_e32,
)

HIDDEN, Q4_WORDS, Q8_BYTES = 12288, 7077888, 4608

def build_address_skeleton(gate:UOp, up:UOp, gate_words:UOp, up_words:UOp, q8:UOp) -> UOp:
  gidxs = [UOp.special(n, f"gidx{i}") for i, n in enumerate((HIDDEN, 2, 1))]
  lidxs = [UOp.special(n, f"lidx{i}") for i, n in enumerate((32, 4, 1))]
  insts = [
    # Kernarg order: gate, up, gate_words, up_words, q8. This slice only needs the first two output pointers.
    s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL),
    s_waitcnt(simm16=0),
    # s2 = gidx0 row, s3 = gidx1 role. Pick output pointer: role 0 -> gate, role 1 -> up.
    s_cmp_eq_u32(ssrc0=s[3], ssrc1=0),
    s_cselect_b32(sdst=s[8], ssrc0=s[4], ssrc1=s[6]),
    s_cselect_b32(sdst=s[9], ssrc0=s[5], ssrc1=s[7]),
    # Address offset is row * sizeof(uint32). Keep high offset 0 because HIDDEN*4 fits in 32 bits.
    v_mov_b32_e32(vdst=v[2], src0=s[2]),
    v_lshlrev_b32_e32(vdst=v[2], src0=2, vsrc1=v[2]),
    v_mov_b32_e32(vdst=v[3], src0=0),
    # Store deterministic pattern: gate[row] = row, up[row] = row + HIDDEN.
    v_mov_b32_e32(vdst=v[4], src0=s[2]),
    s_cmp_eq_u32(ssrc0=s[3], ssrc1=0),
    s_cselect_b32(sdst=s[10], ssrc0=0, ssrc1=HIDDEN),
    v_add_nc_u32_e32(vdst=v[4], src0=s[10], vsrc1=v[4]),
    global_store_b32(addr=v[2], data=v[4], saddr=s[8:9]),
    s_endpgm(),
  ]
  sink = UOp.sink(gate.base, up.base, gate_words.base, up_words.base, q8.base, *gidxs, *lidxs,
                  arg=KernelInfo(name="q8_b2b_gateup_address_skeleton"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))

def main() -> None:
  ap = argparse.ArgumentParser(description="B2b1 address/control skeleton for AMD DSL fused q8 gate/up consumer")
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-codegen-transfer/asm_gateup_address_skeleton.json"))
  args = ap.parse_args()

  gate = Tensor.empty(HIDDEN, dtype=dtypes.uint32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.uint32, device="AMD").contiguous()
  gate_words = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate, up, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=build_address_skeleton)[:2]
  run_linear(gate.schedule_linear())

  got_gate, got_up = gate.numpy().astype(np.uint32), up.numpy().astype(np.uint32)
  exp_gate = np.arange(HIDDEN, dtype=np.uint32)
  exp_up = np.arange(HIDDEN, dtype=np.uint32) + np.uint32(HIDDEN)
  bad_gate = np.flatnonzero(got_gate != exp_gate)
  bad_up = np.flatnonzero(got_up != exp_up)
  result = {
    "date": "2026-06-19",
    "phase": "B2b1_address_control_skeleton",
    "route": "tinygrad_Ops.PROGRAM_AMD_DSL_real_5_buffer_gateup_contract",
    "launch": {"global": [HIDDEN, 2, 1], "local": [32, 4, 1]},
    "gate_mismatches": int(bad_gate.size),
    "up_mismatches": int(bad_up.size),
    "first_gate_bad": int(bad_gate[0]) if bad_gate.size else None,
    "first_up_bad": int(bad_up[0]) if bad_up.size else None,
    "samples": {"gate": [int(x) for x in got_gate[:8]], "up": [int(x) for x in got_up[:8]]},
  }
  result["verdict"] = "PASS" if result["gate_mismatches"] == 0 and result["up_mismatches"] == 0 else "FAIL"
  result["next"] = "If PASS, proceed to q8 load skeleton; if FAIL, fix SGPR workgroup id or kernarg pointer contract first."
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))

if __name__ == "__main__":
  main()
