#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from tinygrad.renderer.amd.dsl import NULL, s, v
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  ds_bpermute_b32, ds_load_b32, ds_store_b32, global_load_b128, global_store_b32,
  s_barrier, s_cmp_eq_u32, s_cselect_b32, s_endpgm, s_load_b128, s_load_b64, s_waitcnt,
  v_add_f32_e32, v_and_b32_e32, v_cmp_ne_u32_e32, v_cndmask_b32_e32,
  v_dot4_i32_iu8, v_lshlrev_b32_e32, v_lshrrev_b32_e32, v_mov_b32_e32, v_xor_b32_e32,
)
from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, Q4_WORDS, Q8_BYTES, add_scaled_partial, build_fullrow_reduce, expected, make_q4_words, make_q8
from extra.qk_decode_native_renderer_dnr3b_compound_emitter_probe import grouped, insts_from_program
from extra.qk_decode_native_renderer_dnr3c6_attribution_scope import build_b128_dsload_b128_no_markers
from extra.qk_decode_native_renderer_dnr3c7c_issue_interleaving_probe import (
  ROOT, build_inputs, build_unpack_all_then_dot_dsload_b128, correctness,
  prepare_kernel, static_grouped, time_interleaved,
)


OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_dnr4_t2_lowband_preload_result.json"


def regs_from_insts(insts: list[Any]) -> dict[str, Any]:
  regs: set[int] = set()
  for inst in insts:
    text = str(inst)
    for a, b in re.findall(r"v\[(\d+):(\d+)\]", text):
      regs.update(range(int(a), int(b) + 1))
    for a in re.findall(r"(?<![A-Za-z0-9_\[])v\[(\d+)\]", text):
      regs.add(int(a))
  return {"max_vgpr_index_static": max(regs) if regs else -1, "unique_vgpr_static": len(regs), "vgpr_set": sorted(regs)}


def _lowband_dot_body() -> list[Any]:
  q4_regs = [v[i] for i in range(12, 20)]
  q8_regs = [v[i] for i in [25, 26, 27, 28, 38, 39, 40, 41]]
  insts: list[Any] = [
    global_load_b128(vdst=v[12:15], addr=v[23], saddr=s[16:17], offset=0),
    global_load_b128(vdst=v[16:19], addr=v[23], saddr=s[16:17], offset=16),
    global_load_b128(vdst=v[25:28], addr=v[24], saddr=s[18:19], offset=0),
    global_load_b128(vdst=v[38:41], addr=v[24], saddr=s[18:19], offset=16),
    s_waitcnt(simm16=0),
    v_and_b32_e32(vdst=v[11], src0=1, vsrc1=v[21]),
    v_cmp_ne_u32_e32(src0=0, vsrc1=v[11]),
  ]
  for lane, (q4_word, q8_word) in enumerate(zip(q4_regs, q8_regs)):
    insts += [
      v_lshrrev_b32_e32(vdst=v[10], src0=4, vsrc1=q4_word),
      v_and_b32_e32(vdst=v[10], src0=0x0f0f0f0f, vsrc1=v[10]),
      v_and_b32_e32(vdst=q4_word, src0=0x0f0f0f0f, vsrc1=q4_word),
      v_cndmask_b32_e32(vdst=q4_word, src0=q4_word, vsrc1=v[10]),
      v_dot4_i32_iu8(vdst=v[4], src0=q4_word, src1=q8_word, src2=v[4], neg=2),
      v_dot4_i32_iu8(vdst=v[5], src0=0x01010101, src1=q8_word, src2=v[5], neg=2),
    ]
  return insts


def _t1_low_reduction_tail() -> list[Any]:
  insts: list[Any] = [v_and_b32_e32(vdst=v[4], src0=31, vsrc1=v[0])]
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
  return insts


def build_fullrow_reduce_dnr4_t2_lowband(gate: UOp, up: UOp, gate_words: UOp, up_words: UOp, q8: UOp) -> UOp:
  gidxs = [UOp.special(n, f"gidx{i}") for i, n in enumerate((HIDDEN, 2, 1))]
  lidxs = [UOp.special(n, f"lidx{i}") for i, n in enumerate((128, 1, 1))]
  lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=16, addrspace=AddrSpace.LOCAL), (), "lds")
  prologue = [
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
  tmp: list[Any] = []
  add_scaled_partial(tmp)
  scale_setup = tmp[:49]
  scale_apply = tmp[153:161]
  insts = prologue + scale_setup + _lowband_dot_body() + scale_apply + _t1_low_reduction_tail()
  sink = UOp.sink(gate.base, up.base, gate_words.base, up_words.base, q8.base, lds, *gidxs, *lidxs,
                  arg=KernelInfo(name="q8_b2b_fullrow_reduce_dnr4_t2_lowband_preload"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))


def synthetic_correctness(rows_check: int = 128) -> dict[str, Any]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_host, up_host = make_q4_words(37, 5), make_q4_words(53, 19)
  q8_host = make_q8()
  gate_words = Tensor(gate_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8_tensor = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  program = build_fullrow_reduce_dnr4_t2_lowband(gate.uop, up.uop, gate_words.uop, up_words.uop, q8_tensor.uop)
  insts = insts_from_program(program)
  ret: dict[str, Any] = {"grouped": grouped(insts), "registers": regs_from_insts(insts), "instruction_count": len(insts)}
  try:
    gate_out, up_out, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8_tensor, fxn=build_fullrow_reduce_dnr4_t2_lowband)[:2]
    run_linear(gate_out.schedule_linear())
    got_gate = gate_out.numpy().astype(np.float32)[:rows_check]
    got_up = up_out.numpy().astype(np.float32)[:rows_check]
    exp_gate = expected(gate_host, q8_host, rows_check)
    exp_up = expected(up_host, q8_host, rows_check)
    gate_abs, up_abs = np.abs(got_gate - exp_gate), np.abs(got_up - exp_up)
    ret.update({"launch": "PASS", "correctness": {
      "rows_check": rows_check,
      "gate_max_abs": float(gate_abs.max()),
      "gate_mean_abs": float(gate_abs.mean()),
      "up_max_abs": float(up_abs.max()),
      "up_mean_abs": float(up_abs.mean()),
    }})
  except Exception as e:
    ret.update({"launch": "FAIL", "error": repr(e)})
  return ret


def real_timing(gguf: Path, seed: int, warmups: int, iters: int) -> list[dict[str, Any]]:
  gate_words_host, up_words_host, q8_host, ref0, ref1 = build_inputs(gguf, seed)
  variants: list[tuple[str, Callable]] = [
    ("native_dnr2", build_fullrow_reduce),
    ("best_static_dnr3c6", build_b128_dsload_b128_no_markers),
    ("c7c_best_unpack_dot_dsload_b128", build_unpack_all_then_dot_dsload_b128),
    ("dnr4_t2_lowband_preload", build_fullrow_reduce_dnr4_t2_lowband),
  ]
  rows: list[dict[str, Any]] = []
  for name, fxn in variants:
    gate, up, linear = prepare_kernel(fxn, gate_words_host, up_words_host, q8_host)
    run_linear(linear)
    Device["AMD"].synchronize()
    rows.append({"name": name, "linear": linear, "correctness": correctness(gate, up, ref0, ref1), "grouped": static_grouped(fxn, gate_words_host.size)})
  for row in rows:
    row["correct"] = row["correctness"]["gate_max_abs"] <= 2e-3 and row["correctness"]["up_max_abs"] <= 2e-3
  time_interleaved(rows, warmups, iters)
  for row in rows:
    del row["linear"]
  by_name = {row["name"]: row for row in rows}
  native_us, best_us, c7c_us, t2_us = (by_name[k]["median_us"] for k in ["native_dnr2", "best_static_dnr3c6", "c7c_best_unpack_dot_dsload_b128", "dnr4_t2_lowband_preload"])
  for row in rows:
    row["delta_vs_native_us"] = row["median_us"] - native_us
    row["delta_vs_best_static_us"] = row["median_us"] - best_us
    row["delta_vs_c7c_us"] = row["median_us"] - c7c_us
    row["delta_vs_t2_us"] = row["median_us"] - t2_us
  return rows


def main() -> int:
  ap = argparse.ArgumentParser(description="DNR4-T2 low-band b128 preload candidate")
  ap.add_argument("--gguf", type=Path, default=Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=4)
  ap.add_argument("--iters", type=int, default=12)
  ap.add_argument("--out", type=Path, default=OUT)
  args = ap.parse_args()

  syn = synthetic_correctness()
  syn_corr = syn.get("correctness", {})
  syn_correct = syn.get("launch") == "PASS" and syn_corr.get("gate_max_abs", 1.0) <= 1e-3 and syn_corr.get("up_max_abs", 1.0) <= 1e-3
  timing_rows = real_timing(args.gguf, args.seed, args.warmups, args.iters) if syn_correct else []
  by_name = {row["name"]: row for row in timing_rows}
  t2 = by_name.get("dnr4_t2_lowband_preload", {})
  native = by_name.get("native_dnr2", {})
  best = by_name.get("best_static_dnr3c6", {})
  c7c = by_name.get("c7c_best_unpack_dot_dsload_b128", {})
  all_correct = bool(timing_rows) and all(row["correct"] for row in timing_rows)
  gain_native = (native.get("median_us", 0) - t2.get("median_us", 0)) if t2 else 0.0
  gain_best = (best.get("median_us", 0) - t2.get("median_us", 0)) if t2 else 0.0
  gain_c7c = (c7c.get("median_us", 0) - t2.get("median_us", 0)) if t2 else 0.0
  material = gain_native >= 30.0 or gain_best >= 15.0 or gain_c7c >= 10.0
  gates = {
    "synthetic_launches": syn.get("launch") == "PASS",
    "synthetic_correct": syn_correct,
    "dot4_preserved": syn["grouped"].get("dot4") == 16,
    "global_load_lte_11": syn["grouped"].get("global_load", 999) <= 11,
    "no_high_v80_band": max((r for r in syn["registers"]["vgpr_set"] if r >= 80), default=-1) == -1,
    "max_vgpr_index_lte_41": syn["registers"]["max_vgpr_index_static"] <= 41,
    "real_timing_all_correct": all_correct,
    "real_timing_material": material,
  }
  if not syn_correct:
    verdict = "BLOCKED_DNR4_T2_LOWBAND_SYNTHETIC_INCORRECT"
  elif not all_correct:
    verdict = "BLOCKED_DNR4_T2_LOWBAND_REAL_GGUF_INCORRECT"
  elif material:
    verdict = "PASS_DNR4_T2_LOWBAND_TIMING_MATERIAL_SCOPE_PROMOTION"
  else:
    verdict = "BLOCKED_DNR4_T2_LOWBAND_CORRECT_TIMING_NOT_MATERIAL"

  result = {
    "date": "2026-06-20",
    "phase": "DNR4_T2_LOWBAND_B128_PRELOAD",
    "schema": "decode_dnr4_t2_lowband_preload_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "synthetic": syn,
    "timing_harness": {"warmups": args.warmups, "iters": args.iters, "method": "same-process interleaved timing"},
    "timing_rows": timing_rows,
    "timing_context": {
      "native_us": native.get("median_us"),
      "best_static_us": best.get("median_us"),
      "c7c_best_us": c7c.get("median_us"),
      "dnr4_t2_us": t2.get("median_us"),
      "t2_gain_vs_native_us": gain_native,
      "t2_gain_vs_best_static_us": gain_best,
      "t2_gain_vs_c7c_us": gain_c7c,
    },
    "gates": gates,
    "decision": {
      "if_blocked_timing": "Low-band packing is structurally correct but not enough; next needs issue/latency attribution or ATT, not more count matching.",
      "if_blocked_correctness": "The selected low registers alias live scale/min or address state; revise the map before timing.",
      "if_pass": "Run PMC confirmation before any promotion.",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "gates": gates,
    "synthetic_grouped": syn["grouped"],
    "synthetic_registers": {k: syn["registers"][k] for k in ["max_vgpr_index_static", "unique_vgpr_static"]},
    "timing_context": result["timing_context"],
    "out": str(args.out.relative_to(ROOT) if args.out.is_absolute() and args.out.is_relative_to(ROOT) else args.out),
  }, indent=2))
  return 0 if syn_correct and all_correct else 1


if __name__ == "__main__":
  raise SystemExit(main())
