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
from tinygrad.renderer.amd.dsl import v
from tinygrad.runtime.autogen.amd.rdna3.ins import ds_load_b128, s_waitcnt, v_mov_b32_e32

from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, Q4_WORDS, Q8_BYTES, build_fullrow_reduce
from extra.q8_ffn_asm_gateup_full import stats_ms
from extra.q8_ffn_fast_artifact_probe import read_q4
from extra.q8_ffn_handwritten_oracle import q4_ref_rows, q8_blocks
from extra.q8_ffn_hcq_artifact import q8_dequant
from extra.qk_decode_native_renderer_dnr3b_compound_emitter_probe import grouped, insts_from_program
from extra.qk_decode_native_renderer_dnr3c2_dataflow_emitter_probe import build_b128_preload_fullrow_reduce
from extra.qk_decode_native_renderer_dnr3c3_compound_shape_probe import build_marker_count_candidate
from extra.qk_decode_native_renderer_dnr3c4_semantic_reduction_probe import build_dnr3c4_candidate


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def build_b128_dsload_b128_no_markers(gate: UOp, up: UOp, gate_words: UOp, up_words: UOp, q8: UOp) -> UOp:
  base = build_b128_preload_fullrow_reduce(gate, up, gate_words, up_words, q8)
  old = insts_from_program(base)
  insts = list(old[:166]) + [
    v_mov_b32_e32(vdst=v[54], src0=0),
    ds_load_b128(vdst=v[10:13], addr=v[54]),
    s_waitcnt(simm16=0),
  ] + list(old[175:])
  sink = base.src[0]
  if sink.arg is not None and isinstance(sink.arg, KernelInfo):
    sink = sink.replace(arg=KernelInfo(name="q8_b2b_fullrow_reduce_dnr3c6_b128_dsload_b128_no_markers"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                               UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))


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
    np.frombuffer(q8_host, dtype=np.uint8).copy(),
    q4_ref_rows(q40, rows, k, q8_x),
    q4_ref_rows(q41, rows, k, q8_x),
  )


def prepare_kernel(fxn: Callable, gate_words_host: np.ndarray, up_words_host: np.ndarray, q8_host: np.ndarray) -> tuple[Tensor, Tensor, Any]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_words = Tensor(gate_words_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_words_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate, up, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=fxn)[:2]
  return gate, up, gate.schedule_linear()


def time_linear(linear: Any, warmups: int, iters: int) -> dict[str, Any]:
  samples: list[float] = []
  for i in range(warmups + iters):
    GlobalCounters.reset()
    t0 = time.perf_counter()
    run_linear(linear)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    device_ms = GlobalCounters.time_sum_s * 1000.0
    if i >= warmups: samples.append(device_ms if device_ms > 0 else elapsed_ms)
  return stats_ms(samples)


def correctness(gate: Tensor, up: Tensor, ref0: np.ndarray, ref1: np.ndarray) -> dict[str, float]:
  got0, got1 = gate.numpy().astype(np.float32), up.numpy().astype(np.float32)
  err0, err1 = np.abs(got0 - ref0), np.abs(got1 - ref1)
  return {
    "gate_max_abs": float(err0.max()),
    "gate_mean_abs": float(err0.mean()),
    "up_max_abs": float(err1.max()),
    "up_mean_abs": float(err1.mean()),
  }


def static_grouped(fxn: Callable, gate_words_size: int) -> dict[str, int]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_words = Tensor.empty(gate_words_size, dtype=dtypes.uint32, device="AMD").contiguous()
  up_words = Tensor.empty(gate_words_size, dtype=dtypes.uint32, device="AMD").contiguous()
  q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous()
  return grouped(insts_from_program(fxn(gate.uop, up.uop, gate_words.uop, up_words.uop, q8.uop)))


def run_variant(name: str, fxn: Callable, gate_words_host: np.ndarray, up_words_host: np.ndarray, q8_host: np.ndarray,
                ref0: np.ndarray, ref1: np.ndarray, warmups: int, iters: int) -> dict[str, Any]:
  gate, up, linear = prepare_kernel(fxn, gate_words_host, up_words_host, q8_host)
  run_linear(linear)
  Device["AMD"].synchronize()
  timing = time_linear(linear, warmups, iters)
  Device["AMD"].synchronize()
  corr = correctness(gate, up, ref0, ref1)
  return {
    "name": name,
    "timing": timing,
    "median_us": timing["median_ms"] * 1000.0,
    "correctness": corr,
    "correct": corr["gate_max_abs"] <= 2e-3 and corr["up_max_abs"] <= 2e-3,
    "grouped": static_grouped(fxn, gate_words_host.size),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description="DNR-3C6 attribution scope and same-harness static-feature timing ladder")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=2)
  ap.add_argument("--iters", type=int, default=6)
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c6_attribution_scope_result.json"))
  args = ap.parse_args()

  dnr3c5 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c5_timing_result.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  gate_words_host, up_words_host, q8_host, ref0, ref1 = build_inputs(args.gguf, args.seed)
  variants: list[tuple[str, Callable]] = [
    ("native_dnr2", build_fullrow_reduce),
    ("load_b128_dnr3c2", build_b128_preload_fullrow_reduce),
    ("load_b128_markers_dnr3c3", build_marker_count_candidate),
    ("load_b128_dsload_b128_no_markers", build_b128_dsload_b128_no_markers),
    ("load_b128_dsload_b128_markers_dnr3c4", build_dnr3c4_candidate),
  ]
  rows = [
    run_variant(name, fxn, gate_words_host, up_words_host, q8_host, ref0, ref1, args.warmups, args.iters)
    for name, fxn in variants
  ]
  by_name = {row["name"]: row for row in rows}
  native_us = by_name["native_dnr2"]["median_us"]
  c4_us = by_name["load_b128_dsload_b128_markers_dnr3c4"]["median_us"]
  oracle_us = float(oracle.get("known_timings_us", {}).get("hipcc_lld_gateup_current_loader", 0.0))
  for row in rows:
    row["delta_vs_native_us"] = row["median_us"] - native_us
    row["delta_vs_oracle_us"] = row["median_us"] - oracle_us

  improvements = {row["name"]: native_us - row["median_us"] for row in rows if row["name"] != "native_dnr2"}
  best = min(rows, key=lambda x: x["median_us"])
  static_local_best_us = max(improvements.values()) if improvements else 0.0
  gates = {
    "dnr3c5_blocked": dnr3c5.get("verdict") == "BLOCKED_DNR3C5_C4_IMPROVES_BUT_REMAINS_BEHIND_ORACLE",
    "all_variants_correct": all(row["correct"] for row in rows),
    "c4_improves_native": c4_us < native_us,
    "best_static_variant_close_to_oracle": best["median_us"] <= oracle_us * 1.10,
    "static_local_features_explain_30us": static_local_best_us >= 30.0,
    "performance_measured": True,
  }
  if not gates["all_variants_correct"]:
    verdict = "BLOCKED_DNR3C6_ATTRIBUTION_LADDER_HAS_INCORRECT_VARIANT"
  elif gates["best_static_variant_close_to_oracle"]:
    verdict = "PASS_DNR3C6_STATIC_LADDER_CLOSE_TO_ORACLE_SCOPE_PROMOTION"
  elif not gates["static_local_features_explain_30us"]:
    verdict = "BLOCKED_DNR3C6_STATIC_LADDER_REFUTES_LOCAL_COUNT_ATTRIBUTION"
  else:
    verdict = "BLOCKED_DNR3C6_STATIC_LADDER_PARTIAL_MOVEMENT_NEEDS_DEEPER_ATTRIBUTION"

  scope = [
    {
      "phase": "DNR-3C6A same-harness static-feature ladder",
      "question": "Do load shape, vector LDS reduction, and markers explain a material fraction of the oracle gap?",
      "exit_gate": ">=30us improvement or <=110% oracle; otherwise local count matching is refuted.",
      "status": "executed",
    },
    {
      "phase": "DNR-3C6B marker placement attribution",
      "question": "Are s_clause/s_delay_alu counts neutral, helpful, or harmful when placed by the current policy?",
      "exit_gate": "marker rows beat matching unmarked rows by >=10us, or marker path is deprioritized.",
      "status": "pending" if "REFUTES" in verdict else "conditional",
    },
    {
      "phase": "DNR-3C6C issue/resource attribution",
      "question": "Is the remaining gap from instruction mix, issue interleaving, VGPR/resource occupancy, or branch/wait control?",
      "exit_gate": "one cause has credible >=30us movement or native renderer route is parked.",
      "status": "pending",
    },
    {
      "phase": "DNR-3C6D branch/wait experiment",
      "question": "Only if attribution points there: do branch/wait semantics recover timing without breaking correctness?",
      "exit_gate": "correct candidate materially closes gap; no dead branch count matching.",
      "status": "blocked_on_C6C",
    },
  ]
  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C6_DECODE_ATTRIBUTION_SCOPE",
    "schema": "decode_native_renderer_dnr3c6_attribution_scope_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "timing_context": {
      "warmups": args.warmups,
      "iters": args.iters,
      "oracle_us": oracle_us,
      "native_us": native_us,
      "best_variant": best["name"],
      "best_variant_us": best["median_us"],
      "best_static_improvement_us": static_local_best_us,
    },
    "scope": scope,
    "variants": rows,
    "gates": gates,
    "blocked_at": {
      "next_phase": "DNR-3C6C issue/resource attribution or route pause",
      "reason": "The same-harness ladder decides whether local static feature matching is still worth pursuing.",
      "minimum_unblock": [
        "if static ladder is refuted, stop adding branch/wait/count patches without hardware or issue/resource attribution",
        "collect issue/resource evidence or construct a new schedule with changed interleaving, not just changed counts",
        "decide whether native decode renderer remains worth pursuing versus keeping the q8 artifact oracle path",
      ],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c5_timing_result.json",
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
    "out": str(args.out),
  }, indent=2))
  return 0 if gates["all_variants_correct"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
