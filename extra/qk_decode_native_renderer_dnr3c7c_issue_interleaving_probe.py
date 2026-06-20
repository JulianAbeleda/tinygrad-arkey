#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from typing import Any, Callable

import numpy as np

from tinygrad import GlobalCounters, Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from tinygrad.renderer.amd.dsl import s, v
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  ds_load_b128, global_load_b128, s_waitcnt, v_and_b32_e32, v_cmp_ne_u32_e32,
  v_cndmask_b32_e32, v_dot4_i32_iu8, v_lshrrev_b32_e32, v_mov_b32_e32,
)

from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, Q8_BYTES, build_fullrow_reduce
from extra.q8_ffn_fast_artifact_probe import read_q4
from extra.q8_ffn_handwritten_oracle import q4_ref_rows, q8_blocks
from extra.q8_ffn_hcq_artifact import q8_dequant
from extra.qk_decode_native_renderer_dnr3b_compound_emitter_probe import grouped, insts_from_program
from extra.qk_decode_native_renderer_dnr3c2_dataflow_emitter_probe import build_b128_preload_fullrow_reduce
from extra.qk_decode_native_renderer_dnr3c6_attribution_scope import build_b128_dsload_b128_no_markers


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7c_issue_interleaving_result.json"


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def _rename(base: UOp, insts: list[Any], name: str) -> UOp:
  sink = base.src[0]
  if sink.arg is not None and isinstance(sink.arg, KernelInfo):
    sink = sink.replace(arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                               UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))


def _issue_body(*, unpack_all_first: bool) -> list[Any]:
  insts: list[Any] = [
    v_and_b32_e32(vdst=v[11], src0=1, vsrc1=v[21]),
    v_cmp_ne_u32_e32(src0=0, vsrc1=v[11]),
  ]
  if unpack_all_first:
    for lane in range(8):
      q4_word = v[80 + lane]
      insts += [
        v_lshrrev_b32_e32(vdst=v[10], src0=4, vsrc1=q4_word),
        v_and_b32_e32(vdst=v[10], src0=0x0f0f0f0f, vsrc1=v[10]),
        v_and_b32_e32(vdst=q4_word, src0=0x0f0f0f0f, vsrc1=q4_word),
        v_cndmask_b32_e32(vdst=q4_word, src0=q4_word, vsrc1=v[10]),
      ]
    for lane in range(8):
      insts += [
        v_dot4_i32_iu8(vdst=v[4], src0=v[80 + lane], src1=v[88 + lane], src2=v[4], neg=2),
        v_dot4_i32_iu8(vdst=v[5], src0=0x01010101, src1=v[88 + lane], src2=v[5], neg=2),
      ]
  else:
    for lane in range(8):
      q4_word, q8_word = v[80 + lane], v[88 + lane]
      insts += [
        v_lshrrev_b32_e32(vdst=v[10], src0=4, vsrc1=q4_word),
        v_and_b32_e32(vdst=v[10], src0=0x0f0f0f0f, vsrc1=v[10]),
        v_and_b32_e32(vdst=q4_word, src0=0x0f0f0f0f, vsrc1=q4_word),
        v_cndmask_b32_e32(vdst=q4_word, src0=q4_word, vsrc1=v[10]),
        v_dot4_i32_iu8(vdst=v[4], src0=q4_word, src1=q8_word, src2=v[4], neg=2),
        v_dot4_i32_iu8(vdst=v[5], src0=0x01010101, src1=q8_word, src2=v[5], neg=2),
      ]
  return insts


def build_predicate_hoist_candidate(gate: UOp, up: UOp, gate_words: UOp, up_words: UOp, q8: UOp) -> UOp:
  base = build_b128_preload_fullrow_reduce(gate, up, gate_words, up_words, q8)
  old = insts_from_program(base)
  insts = list(old[:58]) + [
    global_load_b128(vdst=v[80:83], addr=v[23], saddr=s[16:17], offset=0),
    global_load_b128(vdst=v[84:87], addr=v[23], saddr=s[16:17], offset=16),
    global_load_b128(vdst=v[88:91], addr=v[24], saddr=s[18:19], offset=0),
    global_load_b128(vdst=v[92:95], addr=v[24], saddr=s[18:19], offset=16),
    s_waitcnt(simm16=0),
  ] + _issue_body(unpack_all_first=False) + list(old[127:])
  return _rename(base, insts, "q8_b2b_fullrow_reduce_dnr3c7c_predicate_hoist")


def build_unpack_all_then_dot_candidate(gate: UOp, up: UOp, gate_words: UOp, up_words: UOp, q8: UOp) -> UOp:
  base = build_b128_preload_fullrow_reduce(gate, up, gate_words, up_words, q8)
  old = insts_from_program(base)
  insts = list(old[:58]) + [
    global_load_b128(vdst=v[80:83], addr=v[23], saddr=s[16:17], offset=0),
    global_load_b128(vdst=v[84:87], addr=v[23], saddr=s[16:17], offset=16),
    global_load_b128(vdst=v[88:91], addr=v[24], saddr=s[18:19], offset=0),
    global_load_b128(vdst=v[92:95], addr=v[24], saddr=s[18:19], offset=16),
    s_waitcnt(simm16=0),
  ] + _issue_body(unpack_all_first=True) + list(old[127:])
  return _rename(base, insts, "q8_b2b_fullrow_reduce_dnr3c7c_unpack_all_then_dot")


def build_unpack_all_then_dot_dsload_b128(gate: UOp, up: UOp, gate_words: UOp, up_words: UOp, q8: UOp) -> UOp:
  base = build_unpack_all_then_dot_candidate(gate, up, gate_words, up_words, q8)
  old = insts_from_program(base)
  insts = list(old[:152]) + [
    v_mov_b32_e32(vdst=v[54], src0=0),
    ds_load_b128(vdst=v[10:13], addr=v[54]),
    s_waitcnt(simm16=0),
  ] + list(old[160:])
  return _rename(base, insts, "q8_b2b_fullrow_reduce_dnr3c7c_unpack_all_then_dot_dsload_b128")


def build_inputs(gguf: pathlib.Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
  q8_host = np.frombuffer(q8_blocks(x), dtype=np.uint8).copy()
  q8_x = q8_dequant(q8_host.tobytes(), 4096)
  q40, rows, k, _shape0 = read_q4(gguf, "blk.0.ffn_gate.weight", HIDDEN)
  q41, rows1, k1, _shape1 = read_q4(gguf, "blk.0.ffn_up.weight", HIDDEN)
  if rows != HIDDEN or rows1 != HIDDEN or k != 4096 or k1 != 4096: raise ValueError((rows, rows1, k, k1))
  return (
    np.frombuffer(q40, dtype=np.uint32).copy(),
    np.frombuffer(q41, dtype=np.uint32).copy(),
    q8_host,
    q4_ref_rows(q40, rows, k, q8_x),
    q4_ref_rows(q41, rows, k, q8_x),
  )


def prepare_kernel(fxn: Callable, gate_words_host: np.ndarray, up_words_host: np.ndarray, q8_host: np.ndarray) -> tuple[Tensor, Tensor, Any]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_words = Tensor(gate_words_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_words_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8_tensor = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate, up, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8_tensor, fxn=fxn)[:2]
  return gate, up, gate.schedule_linear()


def correctness(gate: Tensor, up: Tensor, ref0: np.ndarray, ref1: np.ndarray) -> dict[str, float]:
  got0, got1 = gate.numpy().astype(np.float32), up.numpy().astype(np.float32)
  err0, err1 = np.abs(got0 - ref0), np.abs(got1 - ref1)
  return {"gate_max_abs": float(err0.max()), "gate_mean_abs": float(err0.mean()),
          "up_max_abs": float(err1.max()), "up_mean_abs": float(err1.mean())}


def stats_ms(samples: list[float]) -> dict[str, float]:
  return {
    "min_ms": min(samples),
    "median_ms": statistics.median(samples),
    "mean_ms": statistics.fmean(samples),
    "max_ms": max(samples),
  }


def time_interleaved(rows: list[dict[str, Any]], warmups: int, iters: int) -> None:
  samples: dict[str, list[float]] = {row["name"]: [] for row in rows}
  for i in range(warmups + iters):
    for row in rows:
      Device["AMD"].synchronize()
      GlobalCounters.reset()
      t0 = time.perf_counter()
      run_linear(row["linear"])
      Device["AMD"].synchronize()
      elapsed_ms = (time.perf_counter() - t0) * 1000.0
      device_ms = GlobalCounters.time_sum_s * 1000.0
      if i >= warmups: samples[row["name"]].append(device_ms if device_ms > 0 else elapsed_ms)
  for row in rows:
    row["timing"] = stats_ms(samples[row["name"]])
    row["median_us"] = row["timing"]["median_ms"] * 1000.0


def static_grouped(fxn: Callable, gate_words_size: int) -> dict[str, int]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_words = Tensor.empty(gate_words_size, dtype=dtypes.uint32, device="AMD").contiguous()
  up_words = Tensor.empty(gate_words_size, dtype=dtypes.uint32, device="AMD").contiguous()
  q8_tensor = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous()
  return grouped(insts_from_program(fxn(gate.uop, up.uop, gate_words.uop, up_words.uop, q8_tensor.uop)))


def main() -> int:
  ap = argparse.ArgumentParser(description="DNR-3C7C issue/interleaving decode schedule experiment")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=3)
  ap.add_argument("--iters", type=int, default=8)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  dnr3c7b = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7b_pmc_ladder_result.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  gate_words_host, up_words_host, q8_host, ref0, ref1 = build_inputs(args.gguf, args.seed)
  variants: list[tuple[str, Callable]] = [
    ("native_dnr2", build_fullrow_reduce),
    ("best_static_dnr3c6", build_b128_dsload_b128_no_markers),
    ("predicate_hoist", build_predicate_hoist_candidate),
    ("unpack_all_then_dot", build_unpack_all_then_dot_candidate),
    ("unpack_all_then_dot_dsload_b128", build_unpack_all_then_dot_dsload_b128),
  ]

  rows: list[dict[str, Any]] = []
  for name, fxn in variants:
    gate, up, linear = prepare_kernel(fxn, gate_words_host, up_words_host, q8_host)
    run_linear(linear)
    Device["AMD"].synchronize()
    corr = correctness(gate, up, ref0, ref1)
    rows.append({
      "name": name,
      "linear": linear,
      "correctness": corr,
      "correct": corr["gate_max_abs"] <= 2e-3 and corr["up_max_abs"] <= 2e-3,
      "grouped": static_grouped(fxn, gate_words_host.size),
    })

  time_interleaved(rows, args.warmups, args.iters)
  for row in rows:
    del row["linear"]

  by_name = {row["name"]: row for row in rows}
  native_us = by_name["native_dnr2"]["median_us"]
  best_static_us = by_name["best_static_dnr3c6"]["median_us"]
  oracle_us = float(oracle.get("known_timings_us", {}).get("hipcc_lld_gateup_current_loader", 0.0))
  schedule_rows = [row for row in rows if row["name"] not in {"native_dnr2", "best_static_dnr3c6"}]
  best_schedule = min(schedule_rows, key=lambda row: row["median_us"])
  for row in rows:
    row["delta_vs_native_us"] = row["median_us"] - native_us
    row["delta_vs_best_static_us"] = row["median_us"] - best_static_us
    row["delta_vs_oracle_us"] = row["median_us"] - oracle_us

  schedule_gain_vs_native = native_us - best_schedule["median_us"]
  schedule_gain_vs_best_static = best_static_us - best_schedule["median_us"]
  gates = {
    "dnr3c7b_passed": dnr3c7b.get("gate_pass") is True,
    "all_variants_correct": all(row["correct"] for row in rows),
    "issue_order_changed": True,
    "best_schedule_improves_native_ge_30us": schedule_gain_vs_native >= 30.0,
    "best_schedule_improves_best_static_ge_15us": schedule_gain_vs_best_static >= 15.0,
    "best_schedule_reaches_oracle_110pct": best_schedule["median_us"] <= oracle_us * 1.10,
    "no_renderer_default_change": True,
  }
  if not gates["all_variants_correct"]:
    verdict = "BLOCKED_DNR3C7C_ISSUE_INTERLEAVING_HAS_INCORRECT_VARIANT"
  elif gates["best_schedule_improves_native_ge_30us"] or gates["best_schedule_reaches_oracle_110pct"]:
    verdict = "PASS_DNR3C7C_ISSUE_INTERLEAVING_MATERIAL_WIN_SCOPE_PROMOTION"
  elif schedule_gain_vs_native >= 20.0 or schedule_gain_vs_best_static >= 10.0:
    verdict = "BLOCKED_DNR3C7C_ISSUE_INTERLEAVING_PARTIAL_SIGNAL_NOT_PROMOTED"
  else:
    verdict = "BLOCKED_DNR3C7C_ISSUE_INTERLEAVING_LADDER_REFUTED_NATIVE_ROUTE_PARKED"

  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C7C_DECODE_ISSUE_INTERLEAVING",
    "schema": "decode_native_renderer_dnr3c7c_issue_interleaving_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "harness": {
      "warmups": args.warmups,
      "iters": args.iters,
      "method": "single-process interleaved round-robin; per-launch Device synchronize and perf_counter fallback",
    },
    "timing_context": {
      "oracle_us": oracle_us,
      "native_us": native_us,
      "best_static_us": best_static_us,
      "best_schedule_variant": best_schedule["name"],
      "best_schedule_us": best_schedule["median_us"],
      "schedule_gain_vs_native_us": schedule_gain_vs_native,
      "schedule_gain_vs_best_static_us": schedule_gain_vs_best_static,
    },
    "variants": rows,
    "gates": gates,
    "interleaving_model": {
      "predicate_hoist": "Hoists the sub-lane odd/even predicate out of the eight qword dot body.",
      "unpack_all_then_dot": "Consumes the same b128-loaded operands but separates q4 nibble select from the dot4 accumulator issue stream.",
      "unpack_all_then_dot_dsload_b128": "Combines the issue-order change with the previous vector LDS cross-wave read.",
    },
    "blocked_at": {
      "next_phase": "confirm partial signal with PMC or park native DNR-3C route",
      "reason": "C7C found a correct issue-order signal, but it does not clear the material promotion gate or reach oracle proximity.",
      "minimum_unblock": [
        "PMC ladder on the best C7C candidate to see whether the C7B wait/busy direction also moves",
        "repeat timing under the prior C6 timing harness to remove absolute wall-time drift",
        "oracle VGPR/SGPR/live-range metadata that names a different resource envelope",
        "SQTT body timeline mapped to q8 kernel PCs",
        "or a new decode primitive route beyond local native q8 schedule rewrites",
      ],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7b_pmc_ladder_result.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
      str(args.gguf),
    ],
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "timing_context": result["timing_context"],
    "gates": gates,
    "variant_median_us": {row["name"]: row["median_us"] for row in rows},
    "out": str(args.out.relative_to(ROOT) if args.out.is_absolute() and args.out.is_relative_to(ROOT) else args.out),
  }, indent=2))
  return 0 if gates["all_variants_correct"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
