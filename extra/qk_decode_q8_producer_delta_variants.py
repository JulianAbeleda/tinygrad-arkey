#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, random, statistics, subprocess
from typing import Any

import numpy as np

from tinygrad import dtypes
from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_codegen_transfer_audit import inspect_blob
from extra.q8_ffn_fast_artifact_probe import compile_hipcc_linked, hip_norm_source, ms_stats
from extra.q8_ffn_hcq_artifact import NORM_SOURCE, copyin_array, copyout_array, make_buffer, q8_dequant


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_producer_delta_variants_result.json"


def rel(p: pathlib.Path) -> str:
  return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def comgr_norm_source(nt: int) -> str:
  if nt == 256: return NORM_SOURCE
  src = NORM_SOURCE
  src = src.replace("amdgpu_flat_work_group_size(1, 256)", f"amdgpu_flat_work_group_size(1, {nt})")
  src = src.replace("float red[256]", f"float red[{nt}]")
  src = src.replace("i += 256", f"i += {nt}")
  src = src.replace("b += 256", f"b += {nt}")
  src = src.replace("for (int off = 128;", f"for (int off = {nt // 2};")
  return src


def clock_sample() -> dict[str, Any]:
  cmd = ["rocm-smi", "--showgpuclocks"]
  try:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=2)
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-2000:]}
  except Exception as e:
    return {"cmd": cmd, "error": repr(e)}


class Fixture:
  def __init__(self, seed: int):
    rng = np.random.default_rng(seed)
    self.x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
    self.w = (0.7 + rng.random(4096).astype(np.float32) * 0.2).astype(np.float32)
    rinv = np.float32(1.0 / np.sqrt(np.sum(self.x * self.x, dtype=np.float32) / np.float32(4096.0) + np.float32(1.0e-6)))
    self.ref_norm = (self.x * rinv * self.w).astype(np.float32)
    self.xbuf, self.wbuf = make_buffer(4096, dtypes.float32), make_buffer(4096, dtypes.float32)
    self.norm_out, self.q8buf = make_buffer(4096, dtypes.float32), make_buffer(128 * 36, dtypes.uint8)
    copyin_array(self.xbuf, self.x)
    copyin_array(self.wbuf, self.w)

  def correctness(self) -> dict[str, Any]:
    got_norm = copyout_array(self.norm_out, np.empty(4096, dtype=np.float32))
    got_q8 = bytearray(128 * 36)
    self.q8buf.copyout(memoryview(got_q8))
    q8_x = q8_dequant(bytes(got_q8), 4096)
    fp_err = np.abs(got_norm - self.ref_norm)
    q8_err = np.abs(q8_x - self.ref_norm)
    return {
      "producer_fp_max_abs": float(fp_err.max()),
      "producer_fp_mean_abs": float(fp_err.mean()),
      "q8_dequant_max_abs": float(q8_err.max()),
      "q8_dequant_mean_abs": float(q8_err.mean()),
      "producer_correct": float(fp_err.max()) <= 1e-5,
      "q8_dequant_bounded": float(q8_err.max()) <= 0.02,
    }


def run_once(prg: FixedLaunchRunner, fx: Fixture) -> float:
  return float(prg(fx.norm_out._buf, fx.q8buf._buf, fx.xbuf._buf, fx.wbuf._buf, wait=True)) * 1000.0 * 1000.0


def summarize_us(samples: list[float]) -> dict[str, Any]:
  ms = [x / 1000.0 for x in samples]
  out = ms_stats(ms)
  out["median_us"] = statistics.median(samples)
  out["mean_us"] = statistics.fmean(samples)
  out["n"] = len(samples)
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description="Decode q8 producer delta variant probe")
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--seed", type=int, default=91)
  ap.add_argument("--warmups", type=int, default=12)
  ap.add_argument("--rounds", type=int, default=40)
  ap.add_argument("--material-delta-us", type=float, default=2.0)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  dev = Device["AMD"]
  variants: dict[str, dict[str, Any]] = {
    "comgr_nt256": {"compiler": "tinygrad_COMGR", "threads": 256, "blob": dev.compiler.compile(comgr_norm_source(256))},
    "comgr_nt512": {"compiler": "tinygrad_COMGR", "threads": 512, "blob": dev.compiler.compile(comgr_norm_source(512))},
    "comgr_nt1024": {"compiler": "tinygrad_COMGR", "threads": 1024, "blob": dev.compiler.compile(comgr_norm_source(1024))},
    "hipcc_lld_nt1024": {"compiler": "hipcc_lld", "threads": 1024, "blob": compile_hipcc_linked(hip_norm_source(1024), args.arch)},
  }
  for name, v in variants.items():
    v["inspect"] = inspect_blob(f"decode_q8_producer_delta_{name}", v["blob"], f"decode_q8_producer_delta_{name}")
    v["runner"] = FixedLaunchRunner(dev.runtime(f"decode_q8_producer_delta_run_{name}", v["blob"]), (1, 1, 1), (v["threads"], 1, 1))

  fx = Fixture(args.seed)
  for _ in range(args.warmups):
    for v in variants.values():
      run_once(v["runner"], fx)

  correctness: dict[str, Any] = {}
  for name, v in variants.items():
    run_once(v["runner"], fx)
    correctness[name] = fx.correctness()

  rng = random.Random(args.seed)
  samples = {name: [] for name in variants}
  rows: list[dict[str, Any]] = []
  labels = list(variants)
  clock_before = clock_sample()
  for r in range(args.rounds):
    order = labels[:]
    rng.shuffle(order)
    for label in order:
      us = run_once(variants[label]["runner"], fx)
      samples[label].append(us)
      rows.append({"round": r, "label": label, "producer_us": us})
  clock_after = clock_sample()

  summaries = {name: summarize_us(vals) for name, vals in samples.items()}
  inspected = {
    name: {
      "compiler": v["compiler"],
      "threads": v["threads"],
      "runtime": (v["inspect"].get("runtime") or {}),
      "instruction_count": (v["inspect"].get("disasm") or {}).get("instruction_count"),
      "grouped_counts": (v["inspect"].get("disasm") or {}).get("grouped_counts"),
      "top_mnemonics": ((v["inspect"].get("disasm") or {}).get("top_mnemonics") or [])[:20],
      "elf_bytes": (v["inspect"].get("elf") or {}).get("bytes"),
    }
    for name, v in variants.items()
  }
  best_comgr_name = min((n for n in variants if n.startswith("comgr_")), key=lambda n: summaries[n]["median_us"])
  best_comgr_us = summaries[best_comgr_name]["median_us"]
  hip_us = summaries["hipcc_lld_nt1024"]["median_us"]
  delta_us = best_comgr_us - hip_us
  nt256_us = summaries["comgr_nt256"]["median_us"]
  nt1024_us = summaries["comgr_nt1024"]["median_us"]
  nt_improvement_us = nt256_us - nt1024_us

  all_correct = all(c["producer_correct"] and c["q8_dequant_bounded"] for c in correctness.values())
  all_load = all((inspected[name]["runtime"] or {}).get("loads_in_amdprogram") is True for name in inspected)
  rows_present = all(summaries[name]["n"] == args.rounds for name in variants)
  if not all_correct:
    verdict, classification = "BLOCKED_DECODE_Q8_PRODUCER_DELTA_VARIANT_INCORRECT", "INCORRECT"
  elif delta_us <= args.material_delta_us:
    verdict, classification = "PASS_DECODE_Q8_PRODUCER_DELTA_VARIANT_ATTRIBUTED", "THREAD_SHAPE_CLOSES_DELTA"
  elif nt_improvement_us > args.material_delta_us:
    verdict, classification = "PASS_DECODE_Q8_PRODUCER_DELTA_VARIANT_ATTRIBUTED", "THREAD_SHAPE_PARTIAL"
  else:
    verdict, classification = "PASS_DECODE_Q8_PRODUCER_DELTA_VARIANT_ATTRIBUTED", "CODEGEN_SHAPE_DEBT"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_PRODUCER_DELTA_VARIANTS",
    "schema": "decode_q8_producer_delta_variants_v1",
    "verdict": verdict,
    "classification": classification,
    "gate_pass": verdict.startswith("PASS_") and all_correct and all_load and rows_present,
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "summaries": summaries,
    "correctness": correctness,
    "inspect": inspected,
    "analysis": {
      "best_comgr": best_comgr_name,
      "best_comgr_us": best_comgr_us,
      "hipcc_lld_nt1024_us": hip_us,
      "best_comgr_minus_hipcc_lld_us": delta_us,
      "comgr_nt256_us": nt256_us,
      "comgr_nt1024_us": nt1024_us,
      "nt256_minus_nt1024_us": nt_improvement_us,
      "material_delta_us": args.material_delta_us,
    },
    "gates": {
      "all_correct": all_correct,
      "all_load_in_amdprogram": all_load,
      "rows_present": rows_present,
    },
    "clock": {"before": clock_before, "after": clock_after},
    "rows": rows,
    "decision": {
      "THREAD_SHAPE_CLOSES_DELTA": "use the larger workgroup producer shape as the bounded native fix",
      "THREAD_SHAPE_PARTIAL": "thread shape helps but does not close hipcc/LLD parity; inspect remaining codegen deltas",
      "CODEGEN_SHAPE_DEBT": "producer delta is not mainly NT; focus codegen/source instruction shape or artifact import",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "classification": classification,
    "medians_us": {k: summaries[k]["median_us"] for k in summaries},
    "analysis": result["analysis"],
    "gates": result["gates"],
    "out": rel(args.out),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
