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

from tinygrad.engine.realize import run_linear

from extra.q8_ffn_asm_fullrow_reduce import build_fullrow_reduce
from extra.qk_decode_native_renderer_dnr3c6_attribution_scope import build_b128_dsload_b128_no_markers
from extra.qk_decode_native_renderer_dnr3c7c_issue_interleaving_probe import (
  ROOT, build_inputs, build_unpack_all_then_dot_dsload_b128, correctness,
  prepare_kernel, static_grouped, time_interleaved,
)


OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7d_confirmation_result.json"

COUNTER_PASSES = {
  "issue_wait_cache": "SQ_BUSY_CYCLES,SQ_WAIT_ANY,SQ_INSTS_VALU,SQ_INSTS_SALU,GRBM_GUI_ACTIVE,GL2C_HIT,GL2C_MISS,SQ_INSTS_TEX_LOAD",
  "lds_memory": "SQ_BUSY_CYCLES,SQC_LDS_IDX_ACTIVE,SQC_LDS_BANK_CONFLICT,SQ_INSTS_LDS,GRBM_GUI_ACTIVE,GL2C_HIT,GL2C_MISS,SQ_INSTS_SMEM",
}

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
    "all_decoded": decoded,
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
  ]
  prepared = [(name, *prepare(fxn, gate_words_host, up_words_host, q8_host)) for name, fxn in variants]
  for _name, _gate, _up, linear in prepared: run_linear(linear)
  rows = [capture(name, linear, gate, up, ref0, ref1, args.warmups) for name, gate, up, linear in prepared]
  print("DNR3C7D_CHILD_JSON=" + json.dumps({"pass_name": args.pass_name, "rows": rows}, sort_keys=True))

if __name__ == "__main__":
  main()
"""


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def row_by_name(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
  parsed = run.get("parsed") or {}
  return {row["label"]: row for row in parsed.get("rows", [])}


def run_child(pass_name: str, counters: str, gguf: pathlib.Path, warmups: int, timeout_s: int) -> dict[str, Any]:
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
  cmd = [sys.executable, "-c", CHILD, "--gguf", str(gguf), "--warmups", str(warmups), "--pass-name", pass_name]
  t0 = time.perf_counter()
  cp = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                      timeout=timeout_s, check=False)
  parsed = None
  for line in cp.stdout.splitlines():
    if line.startswith("DNR3C7D_CHILD_JSON="):
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
  rows = row_by_name(run)
  native = rows.get("native_dnr2", {})
  best_static = rows.get("best_static_dnr3c6", {})
  ret: dict[str, Any] = {"pass_name": run["pass_name"], "ok": run["returncode"] == 0 and run.get("parsed") is not None, "variants": {}}
  bases = {
    "native": native,
    "best_static": best_static,
  }
  base_norm = {k: ((v.get("dominant") or {}).get("norm_per_active") or {}) for k, v in bases.items()}
  base_hit = {k: (v.get("dominant") or {}).get("l2_hit_pct") for k, v in bases.items()}
  for name, row in rows.items():
    dominant = row.get("dominant") or {}
    norm = dominant.get("norm_per_active") or {}
    item = {
      "correct": row.get("correct"),
      "pmc_event_count": row.get("pmc_event_count"),
      "device_ms": row.get("device_ms"),
      "active": dominant.get("active"),
      "l2_hit_pct": dominant.get("l2_hit_pct"),
      "norm_per_active": norm,
      "delta_vs_native": {
        "l2_hit_pct": None if base_hit["native"] is None else dominant.get("l2_hit_pct", 0) - base_hit["native"],
        "norm": {k: norm.get(k, 0) - base_norm["native"].get(k, 0) for k in sorted(set(norm) | set(base_norm["native"]))},
      },
      "delta_vs_best_static": {
        "l2_hit_pct": None if base_hit["best_static"] is None else dominant.get("l2_hit_pct", 0) - base_hit["best_static"],
        "norm": {k: norm.get(k, 0) - base_norm["best_static"].get(k, 0) for k in sorted(set(norm) | set(base_norm["best_static"]))},
      },
    }
    ret["variants"][name] = item
  return ret


def build_timing_rows(gguf: pathlib.Path, seed: int, warmups: int, iters: int) -> list[dict[str, Any]]:
  gate_words_host, up_words_host, q8_host, ref0, ref1 = build_inputs(gguf, seed)
  variants: list[tuple[str, Callable]] = [
    ("native_dnr2", build_fullrow_reduce),
    ("best_static_dnr3c6", build_b128_dsload_b128_no_markers),
    ("c7c_best_unpack_dot_dsload_b128", build_unpack_all_then_dot_dsload_b128),
  ]
  rows: list[dict[str, Any]] = []
  for name, fxn in variants:
    gate, up, linear = prepare_kernel(fxn, gate_words_host, up_words_host, q8_host)
    run_linear(linear)
    rows.append({
      "name": name,
      "linear": linear,
      "correctness": correctness(gate, up, ref0, ref1),
      "grouped": static_grouped(fxn, gate_words_host.size),
    })
  for row in rows:
    row["correct"] = row["correctness"]["gate_max_abs"] <= 2e-3 and row["correctness"]["up_max_abs"] <= 2e-3
  time_interleaved(rows, warmups, iters)
  for row in rows:
    del row["linear"]
  by_name = {row["name"]: row for row in rows}
  native_us = by_name["native_dnr2"]["median_us"]
  best_static_us = by_name["best_static_dnr3c6"]["median_us"]
  for row in rows:
    row["delta_vs_native_us"] = row["median_us"] - native_us
    row["delta_vs_best_static_us"] = row["median_us"] - best_static_us
  return rows


def main() -> int:
  ap = argparse.ArgumentParser(description="DNR-3C7D confirmation for the C7C partial issue-order signal")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--timing-warmups", type=int, default=4)
  ap.add_argument("--timing-iters", type=int, default=12)
  ap.add_argument("--pmc-warmups", type=int, default=1)
  ap.add_argument("--timeout-s", type=int, default=360)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  dnr3c7c = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7c_issue_interleaving_result.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  timing_rows = build_timing_rows(args.gguf, args.seed, args.timing_warmups, args.timing_iters)
  timing = {row["name"]: row for row in timing_rows}
  c7c = timing["c7c_best_unpack_dot_dsload_b128"]
  native = timing["native_dnr2"]
  best_static = timing["best_static_dnr3c6"]

  pmc_runs = [run_child(name, counters, args.gguf, args.pmc_warmups, args.timeout_s) for name, counters in COUNTER_PASSES.items()]
  pmc_summaries = [summarize_pass(run) for run in pmc_runs]
  all_pmc_ok = all(s["ok"] for s in pmc_summaries)
  all_correct = all(row["correct"] for row in timing_rows) and all(
    v.get("correct") is True for s in pmc_summaries for v in s.get("variants", {}).values()
  )

  issue = next((s for s in pmc_summaries if s["pass_name"] == "issue_wait_cache"), {"variants": {}})
  lds = next((s for s in pmc_summaries if s["pass_name"] == "lds_memory"), {"variants": {}})
  c7c_issue = issue.get("variants", {}).get("c7c_best_unpack_dot_dsload_b128", {})
  c7c_lds = lds.get("variants", {}).get("c7c_best_unpack_dot_dsload_b128", {})
  c7c_issue_delta_best = c7c_issue.get("delta_vs_best_static", {}).get("norm", {})
  c7c_issue_delta_native = c7c_issue.get("delta_vs_native", {}).get("norm", {})
  c7c_lds_delta_best = c7c_lds.get("delta_vs_best_static", {}).get("norm", {})

  c7c_gain_vs_native = -c7c["delta_vs_native_us"]
  c7c_gain_vs_best_static = -c7c["delta_vs_best_static_us"]
  oracle_us = float(oracle.get("known_timings_us", {}).get("hipcc_lld_gateup_current_loader", 0.0))
  pmc_confirms_wait_or_busy = (
    c7c_issue_delta_best.get("sq_wait_any", 0.0) < -10.0 or
    c7c_issue_delta_best.get("sq_busy", 0.0) < -0.25 or
    c7c_issue_delta_native.get("sq_wait_any", 0.0) < -30.0
  )
  timing_material = c7c_gain_vs_native >= 30.0 or c7c_gain_vs_best_static >= 15.0 or c7c["median_us"] <= oracle_us * 1.10
  gates = {
    "dnr3c7c_partial_signal_input": dnr3c7c.get("verdict") == "BLOCKED_DNR3C7C_ISSUE_INTERLEAVING_PARTIAL_SIGNAL_NOT_PROMOTED",
    "all_variants_correct": all_correct,
    "pmc_runs_ok": all_pmc_ok,
    "timing_material": timing_material,
    "pmc_confirms_wait_or_busy": pmc_confirms_wait_or_busy,
    "oracle_proximity": c7c["median_us"] <= oracle_us * 1.10,
    "no_renderer_default_change": True,
  }
  if not all_correct or not all_pmc_ok:
    verdict = "BLOCKED_DNR3C7D_CONFIRMATION_FAILED_OR_INCORRECT"
  elif timing_material and pmc_confirms_wait_or_busy:
    verdict = "PASS_DNR3C7D_C7C_SIGNAL_CONFIRMED_SCOPE_PROMOTION"
  elif c7c_gain_vs_native >= 20.0 or c7c_gain_vs_best_static >= 10.0:
    verdict = "BLOCKED_DNR3C7D_C7C_SIGNAL_PARTIAL_BUT_NOT_CONFIRMED"
  else:
    verdict = "BLOCKED_DNR3C7D_C7C_SIGNAL_NOT_REPRODUCED_PARK_NATIVE_ROUTE"

  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C7D_DECODE_C7C_CONFIRMATION",
    "schema": "decode_native_renderer_dnr3c7d_confirmation_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "counter_passes": COUNTER_PASSES,
    "timing_harness": {
      "warmups": args.timing_warmups,
      "iters": args.timing_iters,
      "method": "same-process interleaved native/best-static/C7C-best timing",
    },
    "timing_context": {
      "oracle_us": oracle_us,
      "native_us": native["median_us"],
      "best_static_us": best_static["median_us"],
      "c7c_best_us": c7c["median_us"],
      "c7c_gain_vs_native_us": c7c_gain_vs_native,
      "c7c_gain_vs_best_static_us": c7c_gain_vs_best_static,
      "c7c_gap_to_oracle_us": c7c["median_us"] - oracle_us,
    },
    "timing_rows": timing_rows,
    "pmc_runs": pmc_runs,
    "pmc_summaries": pmc_summaries,
    "pmc_attribution": {
      "c7c_issue_delta_vs_best_static": c7c_issue_delta_best,
      "c7c_issue_delta_vs_native": c7c_issue_delta_native,
      "c7c_lds_delta_vs_best_static": c7c_lds_delta_best,
      "pmc_confirms_wait_or_busy": pmc_confirms_wait_or_busy,
    },
    "gates": gates,
    "blocked_at": {
      "next_phase": "park native DNR-3C or require oracle/SQTT/resource data",
      "reason": "The best C7C schedule is correct and may move timing directionally, but confirmation requires both material timing and matching PMC movement.",
      "minimum_unblock": [
        "material timing gate: >=30us vs native or >=15us vs best static",
        "PMC confirmation in SQ wait/busy family",
        "or new oracle resource/SQTT body data that explains a remaining schedule lever",
      ],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7c_issue_interleaving_result.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
      str(args.gguf),
    ],
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "timing_context": result["timing_context"],
    "pmc_attribution": result["pmc_attribution"],
    "gates": gates,
    "out": str(args.out.relative_to(ROOT) if args.out.is_absolute() and args.out.is_relative_to(ROOT) else args.out),
  }, indent=2))
  return 0 if all_correct and all_pmc_ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
