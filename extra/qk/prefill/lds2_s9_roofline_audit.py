#!/usr/bin/env python3
"""Roofline audit for the LDS2 S9 promotion decision.

This is intentionally artifact-driven: it reads the S9 micro search plus whole-prefill
authority outputs and answers whether the S9 candidate is worth default promotion.
"""
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

DEFAULT_S9_DIR = pathlib.Path("bench/prefill-lds2-s9")
DEFAULT_WHOLE_DIR = pathlib.Path("bench/prefill-whole-synced")
DEFAULT_OUTPUT = DEFAULT_S9_DIR / "roofline-audit.json"


def _load(path: pathlib.Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def _float(value: Any, default: float = 0.0) -> float:
  try: return float(value)
  except (TypeError, ValueError): return default


def _whole(path: pathlib.Path) -> dict[str, float]:
  data = _load(path)
  whole = data.get("whole_tok_s", {})
  return {str(k): _float(v) for k, v in whole.items()} if isinstance(whole, dict) else {}


def _pct(new: float, old: float) -> float | None:
  return new / old - 1.0 if old > 0 else None


def _shape_from_combined(combined: dict[str, Any]) -> dict[str, int]:
  shape = combined.get("shape", {})
  return {k: int(shape[k]) for k in ("m", "n", "k") if k in shape}


def build_audit(s9_dir: pathlib.Path = DEFAULT_S9_DIR, whole_dir: pathlib.Path = DEFAULT_WHOLE_DIR,
                fp16_peak_tflops: float = 122.8, hbm_bandwidth_gbs: float = 960.0,
                promotion_threshold: float = 0.01, roofline_efficiency_point_threshold: float = 0.01) -> dict[str, Any]:
  combined = _load(s9_dir / "combined-search.json")
  final = _load(s9_dir / "final-report.json") if (s9_dir / "final-report.json").exists() else {}
  shape = _shape_from_combined(combined)
  m, n, k = shape["m"], shape["n"], shape["k"]
  ops = 2 * m * n * k
  compulsory_bytes = (m * k + n * k + m * n) * 2
  operational_intensity = ops / compulsory_bytes
  memory_roof_tflops = operational_intensity * hbm_bandwidth_gbs / 1000.0
  roofline_tflops = min(fp16_peak_tflops, memory_roof_tflops)

  baseline_tflops = _float(combined.get("baseline_tflops"))
  best_tflops = _float(combined.get("best_tflops"))
  baseline_eff = baseline_tflops / roofline_tflops if roofline_tflops else 0.0
  best_eff = best_tflops / roofline_tflops if roofline_tflops else 0.0
  efficiency_point_gain = best_eff - baseline_eff

  default_whole = _whole(whole_dir / "raw-hand-s9-combined-default-authority.json")
  best_whole = _whole(whole_dir / "raw-hand-s9-combined-best-authority.json")
  whole_speedups = {length: _pct(best_whole.get(length, 0.0), default_whole.get(length, 0.0))
                    for length in sorted(set(default_whole) & set(best_whole), key=int)}
  max_whole_speedup = max((v for v in whole_speedups.values() if v is not None), default=None)

  compute_bound = fp16_peak_tflops <= memory_roof_tflops
  micro_speedup = _pct(best_tflops, baseline_tflops)
  promote = bool(
    max_whole_speedup is not None and max_whole_speedup >= promotion_threshold and
    efficiency_point_gain >= roofline_efficiency_point_threshold
  )
  if promote:
    verdict = "S9_ROOFLINE_PROMOTE_DEFAULT"
    rationale = "combined candidate clears whole-prefill and roofline-efficiency promotion thresholds"
  else:
    verdict = "S9_ROOFLINE_KEEP_OPT_IN"
    rationale = "active GEMM is compute-bound, but S9 recovers too little roofline efficiency and whole-prefill throughput to default"

  return {
    "schema": "prefill-lds2-s9-roofline-audit.v1",
    "hardware_model": {
      "name": "AMD Radeon RX 7900 XTX / gfx1100",
      "fp16_peak_tflops": fp16_peak_tflops,
      "hbm_bandwidth_gbs": hbm_bandwidth_gbs,
      "notes": "FP16 peak and HBM bandwidth are SKU-level theoretical inputs; measured clocks can make practical peak lower.",
    },
    "shape": shape,
    "work": {
      "flop": ops,
      "compulsory_fp16_bytes": compulsory_bytes,
      "operational_intensity_flop_per_byte": operational_intensity,
    },
    "roofline": {
      "memory_roof_tflops": memory_roof_tflops,
      "compute_roof_tflops": fp16_peak_tflops,
      "active_roof_tflops": roofline_tflops,
      "bound": "compute" if compute_bound else "memory",
    },
    "micro": {
      "baseline_tflops": baseline_tflops,
      "best_tflops": best_tflops,
      "speedup": micro_speedup,
      "baseline_roofline_efficiency": baseline_eff,
      "best_roofline_efficiency": best_eff,
      "efficiency_point_gain": efficiency_point_gain,
      "best_candidate": (final.get("axes", {}).get("combined", {}).get("best_correct_candidate")
                         if isinstance(final.get("axes"), dict) else None),
    },
    "whole_prefill": {
      "default_tok_s": default_whole,
      "best_tok_s": best_whole,
      "speedups": whole_speedups,
      "max_speedup": max_whole_speedup,
    },
    "thresholds": {
      "promotion_speedup": promotion_threshold,
      "roofline_efficiency_point_gain": roofline_efficiency_point_threshold,
    },
    "verdict": verdict,
    "rationale": rationale,
  }


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--s9-dir", type=pathlib.Path, default=DEFAULT_S9_DIR)
  ap.add_argument("--whole-dir", type=pathlib.Path, default=DEFAULT_WHOLE_DIR)
  ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
  ap.add_argument("--fp16-peak-tflops", type=float, default=122.8)
  ap.add_argument("--hbm-bandwidth-gbs", type=float, default=960.0)
  ap.add_argument("--promotion-threshold", type=float, default=0.01)
  ap.add_argument("--roofline-efficiency-point-threshold", type=float, default=0.01)
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args(argv)

  audit = build_audit(args.s9_dir, args.whole_dir, args.fp16_peak_tflops, args.hbm_bandwidth_gbs,
                      args.promotion_threshold, args.roofline_efficiency_point_threshold)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(audit, indent=2) + "\n")
  if args.json:
    print(json.dumps(audit, indent=2))
  else:
    roof = audit["roofline"]
    micro = audit["micro"]
    whole = audit["whole_prefill"]
    print(f"{audit['verdict']} bound={roof['bound']} roof={roof['active_roof_tflops']:.2f} TFLOPS "
          f"micro={micro['best_tflops']:.2f}/{micro['baseline_tflops']:.2f} "
          f"eff={micro['best_roofline_efficiency']:.3f} "
          f"whole_max_speedup={(whole['max_speedup'] or 0.0):.4f} output={args.output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
