#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7b_pmc_ladder_result.json"

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
from extra.qk_decode_native_renderer_dnr3c2_dataflow_emitter_probe import build_b128_preload_fullrow_reduce
from extra.qk_decode_native_renderer_dnr3c4_semantic_reduction_probe import build_dnr3c4_candidate
from extra.qk_decode_native_renderer_dnr3c6_attribution_scope import build_b128_dsload_b128_no_markers
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

def build_inputs(gguf, seed):
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
  ap.add_argument("--warmups", type=int, default=2)
  ap.add_argument("--pass-name", required=True)
  args = ap.parse_args()
  gate_words_host, up_words_host, q8_host, ref0, ref1 = build_inputs(args.gguf, args.seed)
  variants = [
    ("native_dnr2", build_fullrow_reduce),
    ("load_b128_dnr3c2", build_b128_preload_fullrow_reduce),
    ("best_static_no_markers", build_b128_dsload_b128_no_markers),
    ("dnr3c4_marked", build_dnr3c4_candidate),
  ]
  prepared = [(name, *prepare(fxn, gate_words_host, up_words_host, q8_host)) for name, fxn in variants]
  for _name, _gate, _up, linear in prepared: run_linear(linear)
  rows = [capture(name, linear, gate, up, ref0, ref1, args.warmups) for name, gate, up, linear in prepared]
  print("DNR3C7B_CHILD_JSON=" + json.dumps({"pass_name": args.pass_name, "rows": rows}, sort_keys=True))

if __name__ == "__main__":
  main()
"""


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


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
    if line.startswith("DNR3C7B_CHILD_JSON="):
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


def row_by_name(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
  parsed = run.get("parsed") or {}
  return {row["label"]: row for row in parsed.get("rows", [])}


def summarize_pass(run: dict[str, Any]) -> dict[str, Any]:
  rows = row_by_name(run)
  native = rows.get("native_dnr2", {})
  ret: dict[str, Any] = {"pass_name": run["pass_name"], "ok": run["returncode"] == 0 and run.get("parsed") is not None, "variants": {}}
  native_norm = ((native.get("dominant") or {}).get("norm_per_active") or {})
  native_raw = ((native.get("dominant") or {}).get("raw") or {})
  native_hit = (native.get("dominant") or {}).get("l2_hit_pct")
  for name, row in rows.items():
    norm = (row.get("dominant") or {}).get("norm_per_active") or {}
    raw = (row.get("dominant") or {}).get("raw") or {}
    ret["variants"][name] = {
      "correct": row.get("correct"),
      "pmc_event_count": row.get("pmc_event_count"),
      "device_ms": row.get("device_ms"),
      "active": (row.get("dominant") or {}).get("active"),
      "l2_hit_pct": (row.get("dominant") or {}).get("l2_hit_pct"),
      "l2_hit_pct_delta_vs_native": None if native_hit is None else (row.get("dominant") or {}).get("l2_hit_pct", 0) - native_hit,
      "norm_per_active": norm,
      "norm_delta_vs_native": {k: norm.get(k, 0) - native_norm.get(k, 0) for k in sorted(set(norm) | set(native_norm))},
      "raw": raw,
      "raw_delta_vs_native": {k: raw.get(k, 0) - native_raw.get(k, 0) for k in sorted(set(raw) | set(native_raw))},
    }
  return ret


def main() -> int:
  ap = argparse.ArgumentParser(description="DNR-3C7B same-harness native PMC counter ladder for q8 decode candidates")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--warmups", type=int, default=2)
  ap.add_argument("--timeout-s", type=int, default=360)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  dnr3c7a = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7a_resource_ledger_result.json")
  runs = [run_child(name, counters, args.gguf, args.warmups, args.timeout_s) for name, counters in COUNTER_PASSES.items()]
  summaries = [summarize_pass(run) for run in runs]
  all_ok = all(s["ok"] for s in summaries)
  all_correct = all(v.get("correct") is True for s in summaries for v in s.get("variants", {}).values())

  # Conservative classification. PMC is perturbing; use direction only.
  issue = next((s for s in summaries if s["pass_name"] == "issue_wait_cache"), {"variants": {}})
  lds = next((s for s in summaries if s["pass_name"] == "lds_memory"), {"variants": {}})
  best_issue = issue.get("variants", {}).get("best_static_no_markers", {})
  c4_issue = issue.get("variants", {}).get("dnr3c4_marked", {})
  best_lds = lds.get("variants", {}).get("best_static_no_markers", {})
  attribution = {
    "best_static_vs_native_busy_delta": (best_issue.get("norm_delta_vs_native") or {}).get("sq_busy"),
    "best_static_vs_native_wait_delta": (best_issue.get("norm_delta_vs_native") or {}).get("sq_wait_any"),
    "best_static_vs_native_l2_hit_delta": best_issue.get("l2_hit_pct_delta_vs_native"),
    "best_static_vs_native_lds_active_delta": (best_lds.get("norm_delta_vs_native") or {}).get("lds_active"),
    "best_static_vs_native_bank_conflict_delta": (best_lds.get("norm_delta_vs_native") or {}).get("lds_bank_conflict"),
    "c4_marker_vs_best_busy_delta": ((c4_issue.get("norm_per_active") or {}).get("sq_busy", 0) -
                                     (best_issue.get("norm_per_active") or {}).get("sq_busy", 0)),
    "c4_marker_vs_best_wait_delta": ((c4_issue.get("norm_per_active") or {}).get("sq_wait_any", 0) -
                                     (best_issue.get("norm_per_active") or {}).get("sq_wait_any", 0)),
  }

  names_30us_cause = False
  verdict = (
    "PASS_DNR3C7B_PMC_LADDER_CAPTURED_BLOCKED_NO_30US_COUNTER_CAUSE"
    if all_ok and all_correct and not names_30us_cause else
    "BLOCKED_DNR3C7B_PMC_LADDER_CAPTURE_FAILED_OR_INCORRECT"
  )
  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C7B_DECODE_PMC_COUNTER_LADDER",
    "schema": "decode_native_renderer_dnr3c7b_pmc_ladder_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": False,
    "counter_passes": COUNTER_PASSES,
    "runs": runs,
    "summaries": summaries,
    "attribution": attribution,
    "gates": {
      "dnr3c7a_passed": dnr3c7a.get("gate_pass") is True,
      "pmc_runs_ok": all_ok,
      "all_variants_correct": all_correct,
      "counter_ladder_names_30us_cause": names_30us_cause,
      "no_renderer_default_change": True,
    },
    "blocked_at": {
      "next_phase": "DNR-3C7C issue/interleaving model or route pause",
      "reason": "PMC ladder provides counter direction but does not by itself produce a proven >=30us native scheduling lever.",
      "minimum_unblock": [
        "if counters show the same bottleneck class across native and static variants, stop local native rewrites",
        "if a counter family moves strongly, build an issue/interleaving schedule objective around that family",
        "otherwise keep q8 artifact oracle as the practical decode route",
      ],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7a_resource_ledger_result.json",
      str(args.gguf),
    ],
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "gates": result["gates"],
    "attribution": attribution,
    "out": str(args.out.relative_to(ROOT) if args.out.is_absolute() and args.out.is_relative_to(ROOT) else args.out),
  }, indent=2))
  return 0 if all_ok and all_correct else 1


if __name__ == "__main__":
  raise SystemExit(main())
