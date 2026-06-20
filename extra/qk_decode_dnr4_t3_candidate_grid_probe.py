#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from tinygrad.renderer.amd.dsl import NULL, s, v
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  global_load_b128, s_cmp_eq_u32, s_cselect_b32, s_endpgm, s_load_b128, s_load_b64, s_waitcnt,
  v_and_b32_e32, v_cmp_ne_u32_e32, v_cndmask_b32_e32, v_dot4_i32_iu8, v_lshrrev_b32_e32,
)

from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, Q8_BYTES, add_scaled_partial, build_fullrow_reduce
from extra.qk_decode_dnr4_t2_lowband_preload_probe import (
  build_fullrow_reduce_dnr4_t2_lowband, regs_from_insts, _t1_low_reduction_tail,
)
from extra.qk_decode_native_renderer_dnr3b_compound_emitter_probe import grouped, insts_from_program
from extra.qk_decode_native_renderer_dnr3c6_attribution_scope import build_b128_dsload_b128_no_markers
from extra.qk_decode_native_renderer_dnr3c7c_issue_interleaving_probe import (
  ROOT, build_inputs, build_unpack_all_then_dot_dsload_b128, correctness,
  prepare_kernel, static_grouped, time_interleaved,
)


OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_dnr4_t3_candidate_grid_result.json"

COUNTER_PASSES = {
  "issue_wait_cache": "SQ_BUSY_CYCLES,SQ_WAIT_ANY,SQ_INSTS_VALU,SQ_INSTS_SALU,GRBM_GUI_ACTIVE,GL2C_HIT,GL2C_MISS,SQ_INSTS_TEX_LOAD",
  "lds_memory": "SQ_BUSY_CYCLES,SQC_LDS_IDX_ACTIVE,SQC_LDS_BANK_CONFLICT,SQ_INSTS_LDS,GRBM_GUI_ACTIVE,GL2C_HIT,GL2C_MISS,SQ_INSTS_SMEM",
}


def _lowband_unpack_all_then_dot_body() -> list[Any]:
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
  for q4_word in q4_regs:
    insts += [
      v_lshrrev_b32_e32(vdst=v[10], src0=4, vsrc1=q4_word),
      v_and_b32_e32(vdst=v[10], src0=0x0f0f0f0f, vsrc1=v[10]),
      v_and_b32_e32(vdst=q4_word, src0=0x0f0f0f0f, vsrc1=q4_word),
      v_cndmask_b32_e32(vdst=q4_word, src0=q4_word, vsrc1=v[10]),
    ]
  for q4_word, q8_word in zip(q4_regs, q8_regs):
    insts += [
      v_dot4_i32_iu8(vdst=v[4], src0=q4_word, src1=q8_word, src2=v[4], neg=2),
      v_dot4_i32_iu8(vdst=v[5], src0=0x01010101, src1=q8_word, src2=v[5], neg=2),
    ]
  return insts


def build_dnr4_t3_lowband_unpack_all_then_dot(gate: UOp, up: UOp, gate_words: UOp, up_words: UOp, q8: UOp) -> UOp:
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
  insts = prologue + scale_setup + _lowband_unpack_all_then_dot_body() + scale_apply + _t1_low_reduction_tail()
  sink = UOp.sink(gate.base, up.base, gate_words.base, up_words.base, q8.base, lds, *gidxs, *lidxs,
                  arg=KernelInfo(name="q8_b2b_fullrow_reduce_dnr4_t3_lowband_unpack_all_then_dot"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))


def candidate_variants() -> list[tuple[str, Callable]]:
  return [
    ("native_dnr2", build_fullrow_reduce),
    ("best_static_dnr3c6", build_b128_dsload_b128_no_markers),
    ("c7c_best_unpack_dot_dsload_b128", build_unpack_all_then_dot_dsload_b128),
    ("dnr4_t2_lowband_preload", build_fullrow_reduce_dnr4_t2_lowband),
    ("dnr4_t3_lowband_unpack_all_then_dot", build_dnr4_t3_lowband_unpack_all_then_dot),
  ]


def static_shape(fxn: Callable, gate_words_size: int) -> dict[str, Any]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_words = Tensor.empty(gate_words_size, dtype=dtypes.uint32, device="AMD").contiguous()
  up_words = Tensor.empty(gate_words_size, dtype=dtypes.uint32, device="AMD").contiguous()
  q8_tensor = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous()
  insts = insts_from_program(fxn(gate.uop, up.uop, gate_words.uop, up_words.uop, q8_tensor.uop))
  return {"grouped": grouped(insts), "registers": regs_from_insts(insts), "instruction_count": len(insts)}


def timing_grid(gguf: pathlib.Path, seed: int, warmups: int, iters: int) -> list[dict[str, Any]]:
  gate_words_host, up_words_host, q8_host, ref0, ref1 = build_inputs(gguf, seed)
  rows: list[dict[str, Any]] = []
  for name, fxn in candidate_variants():
    gate, up, linear = prepare_kernel(fxn, gate_words_host, up_words_host, q8_host)
    run_linear(linear)
    Device["AMD"].synchronize()
    corr = correctness(gate, up, ref0, ref1)
    rows.append({
      "name": name,
      "linear": linear,
      "correctness": corr,
      "correct": corr["gate_max_abs"] <= 2e-3 and corr["up_max_abs"] <= 2e-3,
      "static": static_shape(fxn, gate_words_host.size),
    })
  time_interleaved(rows, warmups, iters)
  for row in rows:
    del row["linear"]
  by_name = {row["name"]: row for row in rows}
  bases = {k: by_name[k]["median_us"] for k in ["native_dnr2", "best_static_dnr3c6", "c7c_best_unpack_dot_dsload_b128", "dnr4_t2_lowband_preload"]}
  for row in rows:
    row["delta_vs_native_us"] = row["median_us"] - bases["native_dnr2"]
    row["delta_vs_best_static_us"] = row["median_us"] - bases["best_static_dnr3c6"]
    row["delta_vs_c7c_us"] = row["median_us"] - bases["c7c_best_unpack_dot_dsload_b128"]
    row["delta_vs_t2_us"] = row["median_us"] - bases["dnr4_t2_lowband_preload"]
  return rows


CHILD = r"""
import argparse, json, pathlib, time
import numpy as np

from tinygrad import Device, GlobalCounters, Tensor
from tinygrad.device import Compiled
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import run_linear

from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, build_fullrow_reduce
from extra.q8_ffn_fast_artifact_probe import read_q4
from extra.q8_ffn_handwritten_oracle import q4_ref_rows, q8_blocks
from extra.q8_ffn_hcq_artifact import q8_dequant
from extra.qk_decode_dnr4_t2_lowband_preload_probe import build_fullrow_reduce_dnr4_t2_lowband
from extra.qk_decode_dnr4_t3_candidate_grid_probe import build_dnr4_t3_lowband_unpack_all_then_dot
from extra.qk_decode_native_renderer_dnr3c6_attribution_scope import build_b128_dsload_b128_no_markers
from extra.qk_decode_native_renderer_dnr3c7c_issue_interleaving_probe import build_unpack_all_then_dot_dsload_b128
from extra.qk_pmc_capture import decode_pmc

def stats(c):
  act = c.get("GRBM_GUI_ACTIVE", 0) or 1
  hit, miss = c.get("GL2C_HIT", 0), c.get("GL2C_MISS", 0)
  def div(k): return c.get(k, 0) / act
  return {
    "raw": c,
    "active": c.get("GRBM_GUI_ACTIVE", 0),
    "l2_hit_pct": 100.0 * hit / (hit + miss + 1e-9),
    "norm_per_active": {
      "sq_busy": div("SQ_BUSY_CYCLES"),
      "sq_wait_any": div("SQ_WAIT_ANY"),
      "valu": div("SQ_INSTS_VALU"),
      "salu": div("SQ_INSTS_SALU"),
      "tex_load": div("SQ_INSTS_TEX_LOAD"),
      "smem": div("SQ_INSTS_SMEM"),
      "lds_inst": div("SQ_INSTS_LDS"),
      "lds_active": div("SQC_LDS_IDX_ACTIVE"),
      "lds_bank_conflict": div("SQC_LDS_BANK_CONFLICT"),
    },
  }

def build_child_inputs(gguf, seed):
  rng = np.random.default_rng(seed)
  x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
  q8_host = np.frombuffer(q8_blocks(x), dtype=np.uint8).copy()
  q8_x = q8_dequant(q8_host.tobytes(), 4096)
  q40, rows, k, _shape0 = read_q4(gguf, "blk.0.ffn_gate.weight", HIDDEN)
  q41, rows1, k1, _shape1 = read_q4(gguf, "blk.0.ffn_up.weight", HIDDEN)
  if rows != HIDDEN or rows1 != HIDDEN or k != 4096 or k1 != 4096: raise ValueError((rows, rows1, k, k1))
  return np.frombuffer(q40, dtype=np.uint32).copy(), np.frombuffer(q41, dtype=np.uint32).copy(), q8_host, q4_ref_rows(q40, rows, k, q8_x), q4_ref_rows(q41, rows, k, q8_x)

def prepare(fxn, gate_words_host, up_words_host, q8_host):
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_words = Tensor(gate_words_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_words_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate, up, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=fxn)[:2]
  return gate, up, gate.schedule_linear()

def correctness(gate, up, ref0, ref1):
  got0, got1 = gate.numpy().astype(np.float32), up.numpy().astype(np.float32)
  err0, err1 = np.abs(got0 - ref0), np.abs(got1 - ref1)
  return {"gate_max_abs": float(err0.max()), "gate_mean_abs": float(err0.mean()),
          "up_max_abs": float(err1.max()), "up_mean_abs": float(err1.mean())}

def capture(label, linear, gate, up, ref0, ref1, warmups):
  for _ in range(warmups):
    GlobalCounters.reset()
    run_linear(linear)
  Device["AMD"].synchronize()
  base = len([e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"])
  GlobalCounters.reset()
  t0 = time.perf_counter()
  run_linear(linear)
  Device["AMD"].synchronize()
  elapsed_ms = (time.perf_counter() - t0) * 1000.0
  device_ms = GlobalCounters.time_sum_s * 1000.0
  Device["AMD"]._at_profile_finalize()
  evs = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"][base:]
  decoded = [decode_pmc(ev) for ev in evs]
  dominant = max(decoded, key=lambda row: row.get("GRBM_GUI_ACTIVE", row.get("SQ_BUSY_CYCLES", 0))) if decoded else {}
  corr = correctness(gate, up, ref0, ref1)
  return {
    "label": label,
    "pmc_event_count": len(evs),
    "elapsed_ms": elapsed_ms,
    "device_ms": device_ms,
    "dominant": stats(dominant),
    "correctness": corr,
    "correct": corr["gate_max_abs"] <= 2e-3 and corr["up_max_abs"] <= 2e-3,
  }

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--gguf", type=pathlib.Path, required=True)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=1)
  ap.add_argument("--pass-name", required=True)
  args = ap.parse_args()
  gate_words_host, up_words_host, q8_host, ref0, ref1 = build_child_inputs(args.gguf, args.seed)
  variants = [
    ("native_dnr2", build_fullrow_reduce),
    ("best_static_dnr3c6", build_b128_dsload_b128_no_markers),
    ("c7c_best_unpack_dot_dsload_b128", build_unpack_all_then_dot_dsload_b128),
    ("dnr4_t2_lowband_preload", build_fullrow_reduce_dnr4_t2_lowband),
    ("dnr4_t3_lowband_unpack_all_then_dot", build_dnr4_t3_lowband_unpack_all_then_dot),
  ]
  prepared = [(name, *prepare(fxn, gate_words_host, up_words_host, q8_host)) for name, fxn in variants]
  for _name, _gate, _up, linear in prepared: run_linear(linear)
  rows = [capture(name, linear, gate, up, ref0, ref1, args.warmups) for name, gate, up, linear in prepared]
  print("DNR4T3_CHILD_JSON=" + json.dumps({"pass_name": args.pass_name, "rows": rows}, sort_keys=True))

if __name__ == "__main__":
  main()
"""


def run_child(pass_name: str, counters: str, gguf: pathlib.Path, seed: int, warmups: int, timeout_s: int) -> dict[str, Any]:
  env = os.environ.copy()
  env.update({
    "PYTHONPATH": str(ROOT),
    "DEV": "AMD",
    "PROFILE": "1",
    "PMC": "1",
    "SQTT": "0",
    "VIZ": "0",
    "DEBUG": "0",
    "PMC_COUNTERS": counters,
  })
  cmd = [sys.executable, "-c", CHILD, "--gguf", str(gguf), "--seed", str(seed), "--warmups", str(warmups), "--pass-name", pass_name]
  t0 = time.perf_counter()
  cp = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                      timeout=timeout_s, check=False)
  parsed = None
  for line in cp.stdout.splitlines():
    if line.startswith("DNR4T3_CHILD_JSON="):
      parsed = json.loads(line.split("=", 1)[1])
  return {
    "pass_name": pass_name,
    "counters": counters.split(","),
    "returncode": cp.returncode,
    "elapsed_s": round(time.perf_counter() - t0, 3),
    "stdout_tail": cp.stdout.splitlines()[-20:],
    "stderr_tail": cp.stderr.splitlines()[-20:],
    "parsed": parsed,
  }


def summarize_pass(run: dict[str, Any]) -> dict[str, Any]:
  parsed = run.get("parsed") or {}
  rows = {row["label"]: row for row in parsed.get("rows", [])}
  base_names = ["native_dnr2", "best_static_dnr3c6", "c7c_best_unpack_dot_dsload_b128", "dnr4_t2_lowband_preload"]
  base_norm = {name: ((rows.get(name, {}).get("dominant") or {}).get("norm_per_active") or {}) for name in base_names}
  ret: dict[str, Any] = {"pass_name": run["pass_name"], "ok": run["returncode"] == 0 and parsed != {}, "variants": {}}
  for name, row in rows.items():
    dominant = row.get("dominant") or {}
    norm = dominant.get("norm_per_active") or {}
    ret["variants"][name] = {
      "correct": row.get("correct"),
      "pmc_event_count": row.get("pmc_event_count"),
      "device_ms": row.get("device_ms"),
      "active": dominant.get("active"),
      "l2_hit_pct": dominant.get("l2_hit_pct"),
      "norm_per_active": norm,
      "delta_norm": {
        base: {k: norm.get(k, 0.0) - base_norm[base].get(k, 0.0) for k in sorted(set(norm) | set(base_norm[base]))}
        for base in base_names
      },
    }
  return ret


def monotonic_score(timing_rows: list[dict[str, Any]], pmc_summaries: list[dict[str, Any]]) -> dict[str, Any]:
  timings = {row["name"]: row["median_us"] for row in timing_rows}
  out: dict[str, Any] = {}
  for summary in pmc_summaries:
    for metric in ["sq_wait_any", "sq_busy", "valu", "salu", "tex_load", "lds_inst", "lds_active", "lds_bank_conflict"]:
      pairs: list[tuple[float, float]] = []
      for name, item in summary.get("variants", {}).items():
        norm = item.get("norm_per_active") or {}
        if name in timings and metric in norm:
          pairs.append((timings[name], float(norm[metric])))
      if len(pairs) < 3:
        continue
      time_order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
      metric_order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
      agreement = sum(1 for a, b in zip(time_order, metric_order) if a == b) / len(pairs)
      out[f"{summary['pass_name']}:{metric}"] = {
        "rank_position_agreement": agreement,
        "best_time_variant_metric_rank": metric_order.index(time_order[0]) if time_order[0] in metric_order else None,
        "pairs": pairs,
      }
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description="DNR4-T3 decode candidate grid and PMC correlation")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--timing-warmups", type=int, default=4)
  ap.add_argument("--timing-iters", type=int, default=12)
  ap.add_argument("--pmc-warmups", type=int, default=1)
  ap.add_argument("--timeout-s", type=int, default=420)
  ap.add_argument("--skip-pmc", action="store_true")
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  timing_rows = timing_grid(args.gguf, args.seed, args.timing_warmups, args.timing_iters)
  timing = {row["name"]: row for row in timing_rows}
  t3 = timing["dnr4_t3_lowband_unpack_all_then_dot"]
  native, best, c7c, t2 = (timing[k] for k in ["native_dnr2", "best_static_dnr3c6", "c7c_best_unpack_dot_dsload_b128", "dnr4_t2_lowband_preload"])

  pmc_runs = [] if args.skip_pmc else [run_child(name, counters, args.gguf, args.seed, args.pmc_warmups, args.timeout_s) for name, counters in COUNTER_PASSES.items()]
  pmc_summaries = [summarize_pass(run) for run in pmc_runs]
  all_pmc_ok = args.skip_pmc or all(s["ok"] for s in pmc_summaries)
  all_correct = all(row["correct"] for row in timing_rows) and (
    args.skip_pmc or all(v.get("correct") is True for s in pmc_summaries for v in s.get("variants", {}).values())
  )

  t3_gain_native = native["median_us"] - t3["median_us"]
  t3_gain_best = best["median_us"] - t3["median_us"]
  t3_gain_c7c = c7c["median_us"] - t3["median_us"]
  t3_gain_t2 = t2["median_us"] - t3["median_us"]
  best_variant = min(timing_rows, key=lambda row: row["median_us"])
  material = t3_gain_native >= 30.0 or t3_gain_best >= 15.0 or t3_gain_c7c >= 10.0
  correlation = monotonic_score(timing_rows, pmc_summaries)
  predictive_metrics = {k: v for k, v in correlation.items() if v["rank_position_agreement"] >= 0.8 or v["best_time_variant_metric_rank"] == 0}

  gates = {
    "all_variants_correct": all_correct,
    "pmc_runs_ok": all_pmc_ok,
    "t3_candidate_buildable": t3["correct"] is True,
    "t3_static_no_high_v80_band": max((r for r in t3["static"]["registers"]["vgpr_set"] if r >= 80), default=-1) == -1,
    "t3_static_dot4_16": t3["static"]["grouped"].get("dot4") == 16,
    "t3_material_timing": material,
    "best_variant_is_t3": best_variant["name"] == "dnr4_t3_lowband_unpack_all_then_dot",
    "counter_predictive_signal": len(predictive_metrics) > 0,
    "no_renderer_default_change": True,
  }
  if not all_correct:
    verdict = "BLOCKED_DNR4_T3_CANDIDATE_GRID_INCORRECT"
  elif not all_pmc_ok:
    verdict = "BLOCKED_DNR4_T3_CANDIDATE_GRID_PMC_FAILED"
  elif material and best_variant["name"] == "dnr4_t3_lowband_unpack_all_then_dot":
    verdict = "PASS_DNR4_T3_COMBINED_CANDIDATE_MATERIAL_SCOPE_PROMOTION"
  elif len(predictive_metrics) > 0:
    verdict = "BLOCKED_DNR4_T3_COUNTER_SIGNAL_PRESENT_TIMING_NOT_MATERIAL"
  else:
    verdict = "BLOCKED_DNR4_T3_NO_MATERIAL_NATIVE_LEVER_UNBLOCK_ATT"

  result = {
    "date": "2026-06-20",
    "phase": "DNR4_T3_CANDIDATE_GRID_AND_PMC_CORRELATION",
    "schema": "decode_dnr4_t3_candidate_grid_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "counter_passes": COUNTER_PASSES,
    "timing_harness": {
      "warmups": args.timing_warmups,
      "iters": args.timing_iters,
      "method": "same-process interleaved candidate grid timing",
    },
    "timing_context": {
      "native_us": native["median_us"],
      "best_static_us": best["median_us"],
      "c7c_us": c7c["median_us"],
      "dnr4_t2_us": t2["median_us"],
      "dnr4_t3_us": t3["median_us"],
      "best_variant": best_variant["name"],
      "best_variant_us": best_variant["median_us"],
      "t3_gain_vs_native_us": t3_gain_native,
      "t3_gain_vs_best_static_us": t3_gain_best,
      "t3_gain_vs_c7c_us": t3_gain_c7c,
      "t3_gain_vs_t2_us": t3_gain_t2,
    },
    "timing_rows": timing_rows,
    "pmc_runs": pmc_runs,
    "pmc_summaries": pmc_summaries,
    "counter_correlation": correlation,
    "predictive_metrics": predictive_metrics,
    "gates": gates,
    "decision": {
      "if_pass": "Run a confirmation pass before promotion.",
      "if_counter_signal_only": "Use the predictive counter family to design one final targeted candidate; do not start BEAM/search yet.",
      "if_no_signal": "Stop native schedule rewrites and unblock ATT PC timeline before further decode work.",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "timing_context": result["timing_context"],
    "gates": gates,
    "predictive_metrics": sorted(predictive_metrics),
    "out": str(args.out.relative_to(ROOT) if args.out.is_absolute() and args.out.is_relative_to(ROOT) else args.out),
  }, indent=2))
  return 0 if all_correct and all_pmc_ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
