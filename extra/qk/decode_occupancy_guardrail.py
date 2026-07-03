#!/usr/bin/env python3
"""Decode-attention occupancy/resource guardrail.

Consumes the ISA-vectorization/resource artifact and split-aware hotloop audit. This is a preflight gate for any
new decode-attention candidate: if a candidate increases register pressure, scratch, or cross-lane/waitcnt pressure
relative to the current generated best-stack baseline, it must be rejected before W==D unless explicitly overridden.
"""
from __future__ import annotations
import json, pathlib, datetime
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
ISA = ROOT / "bench/qk-decode-isa-vectorization/latest.json"
HOTLOOP = ROOT / "bench/qk-decode-hotloop-schedule-diff/latest.json"
BASELINE = {"vgpr_max": 88, "scratch_max": 0, "lds_max": 8192, "min_wg_per_cu": 4.0, "cross_lane_max": 40, "waitcnt_max": 50}

def load(path: pathlib.Path) -> dict[str, Any]:
  return json.loads(path.read_text()) if path.exists() else {}

class _Args:
  def __init__(self):
    self.isa, self.hotloop = str(ISA), str(HOTLOOP)
    self.vgpr_max, self.scratch_max, self.lds_max = BASELINE["vgpr_max"], BASELINE["scratch_max"], BASELINE["lds_max"]
    self.min_wg_per_cu, self.cross_lane_max, self.waitcnt_max = BASELINE["min_wg_per_cu"], BASELINE["cross_lane_max"], BASELINE["waitcnt_max"]

def build() -> dict:
  args = _Args()
  isa, hot = load(pathlib.Path(args.isa)), load(pathlib.Path(args.hotloop))
  tile = isa.get("tile") or isa.get("capture", {}).get("tile", {})
  resources = tile.get("resources", {})
  markers = tile.get("markers", {})
  route_occ = isa.get("route_cleanliness", {}).get("occupancy", {})
  gen_loop = hot.get("generated", {}).get("selected_loop", {})
  gen_mix = gen_loop.get("metrics", {}).get("mix", {})
  checks = {
    "vgpr": {"value": resources.get("vgpr"), "max": args.vgpr_max, "pass": (resources.get("vgpr", 10**9) <= args.vgpr_max)},
    "scratch": {"value": resources.get("scratch"), "max": args.scratch_max, "pass": (resources.get("scratch", 10**9) <= args.scratch_max)},
    "lds": {"value": resources.get("lds"), "max": args.lds_max, "pass": (resources.get("lds", 10**9) <= args.lds_max)},
    "wg_per_cu": {"value": route_occ.get("wg_per_cu"), "min": args.min_wg_per_cu, "pass": (route_occ.get("wg_per_cu", args.min_wg_per_cu) >= args.min_wg_per_cu)},
    "cross_lane": {"value": markers.get("cross_lane", gen_mix.get("ds_bpermute")), "max": args.cross_lane_max, "pass": ((markers.get("cross_lane", gen_mix.get("ds_bpermute", 10**9))) <= args.cross_lane_max)},
    "selected_loop_waitcnt": {"value": gen_mix.get("s_waitcnt"), "max": args.waitcnt_max, "pass": (gen_mix.get("s_waitcnt", 10**9) <= args.waitcnt_max)},
  }
  passed = all(v["pass"] for v in checks.values())
  out = {
    "schema": "qk_decode_occupancy_guardrail_v1",
    "date": datetime.date.today().isoformat(),
    "inputs": {"isa": str(pathlib.Path(args.isa)), "hotloop": str(pathlib.Path(args.hotloop))},
    "baseline_policy": BASELINE,
    "checks": checks,
    "decision_rule": "Abort pressure-increasing candidates before W==D unless an explicit override records why occupancy loss is expected to win.",
    "verdict": "OCCUPANCY_GUARDRAIL_PASS" if passed else "OCCUPANCY_GUARDRAIL_FAIL__PRESSURE_INCREASE",
    "pass": passed,
  }
  return out

if __name__ == "__main__":
  import sys; sys.path.insert(0, str(ROOT))
  from extra.qk.gate_registry import run
  raise SystemExit(run("occupancy_guardrail"))
