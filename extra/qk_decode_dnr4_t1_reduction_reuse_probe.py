#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from tinygrad.renderer.amd.dsl import NULL, s, v
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  ds_bpermute_b32, ds_load_b32, ds_store_b32, global_store_b32,
  s_barrier, s_cmp_eq_u32, s_cselect_b32, s_endpgm, s_load_b128, s_load_b64, s_waitcnt,
  v_add_f32_e32, v_and_b32_e32, v_lshlrev_b32_e32, v_lshrrev_b32_e32, v_mov_b32_e32, v_xor_b32_e32,
)
from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, Q8_BYTES, add_scaled_partial, expected, make_q4_words, make_q8
from extra.qk_decode_native_renderer_dnr3b_compound_emitter_probe import grouped, insts_from_program


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_dnr4_t1_reduction_reuse_result.json"


def max_vgpr_from_insts(insts: list[Any]) -> int:
  import re
  regs: set[int] = set()
  for inst in insts:
    text = str(inst)
    for a, b in re.findall(r"v\[(\d+):(\d+)\]", text):
      regs.update(range(int(a), int(b) + 1))
    for a in re.findall(r"(?<![A-Za-z0-9_\[])v\[(\d+)\]", text):
      regs.add(int(a))
  return max(regs) if regs else -1


def build_fullrow_reduce_dnr4_t1(gate: UOp, up: UOp, gate_words: UOp, up_words: UOp, q8: UOp) -> UOp:
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

  # DNR4-T1: after add_scaled_partial, v10 is the only value needed by the reduction.
  # Reuse low dead temporaries instead of reserving v50-v54.
  insts += [v_and_b32_e32(vdst=v[4], src0=31, vsrc1=v[0])]
  for off in [16, 8, 4, 2, 1]:
    insts += [
      v_xor_b32_e32(vdst=v[5], src0=off, vsrc1=v[4]),
      v_lshlrev_b32_e32(vdst=v[5], src0=2, vsrc1=v[5]),
      ds_bpermute_b32(vdst=v[6], addr=v[5], data0=v[10]),
      s_waitcnt(simm16=0),
      v_add_f32_e32(vdst=v[10], src0=v[6], vsrc1=v[10]),
    ]
  insts += [
    v_lshrrev_b32_e32(vdst=v[2], src0=5, vsrc1=v[0]),
    v_lshlrev_b32_e32(vdst=v[2], src0=2, vsrc1=v[2]),
    ds_store_b32(addr=v[2], data0=v[10]),
    s_waitcnt(simm16=0),
    s_barrier(),
    v_mov_b32_e32(vdst=v[1], src0=0),
    ds_load_b32(vdst=v[10], addr=v[1]),
    v_mov_b32_e32(vdst=v[1], src0=4),
    ds_load_b32(vdst=v[11], addr=v[1]),
    v_mov_b32_e32(vdst=v[1], src0=8),
    ds_load_b32(vdst=v[12], addr=v[1]),
    v_mov_b32_e32(vdst=v[1], src0=12),
    ds_load_b32(vdst=v[13], addr=v[1]),
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
                  arg=KernelInfo(name="q8_b2b_fullrow_reduce_dnr4_t1_reduction_reuse"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))


def run_candidate(rows_check: int = 128) -> dict[str, Any]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_host, up_host = make_q4_words(37, 5), make_q4_words(53, 19)
  q8_host = make_q8()
  gate_words = Tensor(gate_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()

  program = build_fullrow_reduce_dnr4_t1(gate.uop, up.uop, gate_words.uop, up_words.uop, q8.uop)
  insts = insts_from_program(program)
  result: dict[str, Any] = {
    "instruction_count": len(insts),
    "grouped": grouped(insts),
    "max_vgpr_index_static": max_vgpr_from_insts(insts),
  }
  try:
    gate_out, up_out, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=build_fullrow_reduce_dnr4_t1)[:2]
    run_linear(gate_out.schedule_linear())
    got_gate = gate_out.numpy().astype(np.float32)[:rows_check]
    got_up = up_out.numpy().astype(np.float32)[:rows_check]
    exp_gate = expected(gate_host, q8_host, rows_check)
    exp_up = expected(up_host, q8_host, rows_check)
    gate_abs, up_abs = np.abs(got_gate - exp_gate), np.abs(got_up - exp_up)
    result.update({
      "launch": "PASS",
      "correctness": {
        "rows_check": rows_check,
        "gate_max_abs": float(gate_abs.max()),
        "gate_mean_abs": float(gate_abs.mean()),
        "up_max_abs": float(up_abs.max()),
        "up_mean_abs": float(up_abs.mean()),
      },
    })
  except Exception as e:
    result.update({"launch": "FAIL", "error": repr(e)})
  return result


def main() -> int:
  probe = run_candidate()
  correctness = probe.get("correctness", {})
  correct = probe.get("launch") == "PASS" and correctness.get("gate_max_abs", 1.0) <= 1e-3 and correctness.get("up_max_abs", 1.0) <= 1e-3
  gates = {
    "launches": probe.get("launch") == "PASS",
    "correct": correct,
    "dot4_preserved": probe["grouped"].get("dot4") == 16,
    "reduction_topology_preserved": probe["grouped"].get("shuffle") == 5 and probe["grouped"].get("barrier") == 1,
    "max_vgpr_index_lte_41": probe["max_vgpr_index_static"] <= 41,
    "global_load_count_unchanged": probe["grouped"].get("global_load") == 22,
  }
  result = {
    "date": "2026-06-20",
    "phase": "DNR4_T1_REDUCTION_BAND_REUSE",
    "schema": "decode_dnr4_t1_reduction_reuse_v1",
    "verdict": "PASS_DNR4_T1_REDUCTION_REUSE_STRUCTURAL_CORRECT" if all(gates.values()) else "BLOCKED_DNR4_T1_REDUCTION_REUSE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "probe": probe,
    "gates": gates,
    "next": {
      "if_pass": "Run same-harness timing against native/best-static/C7C; if timing does not move, proceed to DNR4-T2 dot-body compression.",
      "if_blocked": "Fix register reuse only if correctness failed; do not alter S3 issue order in this pass.",
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "grouped": probe["grouped"],
    "max_vgpr_index_static": probe["max_vgpr_index_static"],
    "correctness": correctness,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())

