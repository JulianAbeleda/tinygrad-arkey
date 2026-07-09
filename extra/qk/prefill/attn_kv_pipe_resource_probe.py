#!/usr/bin/env python3
"""Structural probe for the generated attn_kv pipe LDS overflow.

This intentionally does not alter route selection.  It compares the smallest
candidate fixes for the known S10 composed-route failure where generated pipe
local staging for M=512,N=1024,K=4096 declares 69632 bytes of LDS.
"""
from __future__ import annotations

import argparse, json, pathlib, sys
from dataclasses import dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.qk.prefill.s10_compile_capture import analyze_amd_source

LDS_LIMIT_BYTES = 65536
ATTN_KV_SHAPE = {"m": 512, "n": 1024, "k": 4096}
FAIL_SOURCE_SKETCH = """
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 32))) r_attn_kv(half* data0_524288, half* data1_2097152, half* data2_4194304) {
  __attribute__((shared, aligned(16)))half buf0[2048];
  __attribute__((shared, aligned(16)))half buf1[32768];
  float8 x = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
"""


@dataclass(frozen=True)
class StageComponent:
  name: str
  elements: int
  elem_bytes: int = 2

  @property
  def bytes(self) -> int:
    return self.elements * self.elem_bytes

  def row(self) -> dict[str, Any]:
    return {"name": self.name, "elements": self.elements, "elem_bytes": self.elem_bytes, "bytes": self.bytes}


BASE_COMPONENTS = (
  StageComponent("a_fragment_local_stage", 2048),
  StageComponent("b_tile_local_stage", 32768),
)


def _candidate(name: str, description: str, components: tuple[StageComponent, ...],
               *, route_change: str, implementation_scope: str) -> dict[str, Any]:
  shared_bytes = sum(c.bytes for c in components)
  return {
    "name": name,
    "description": description,
    "components": [c.row() for c in components],
    "shared_bytes": shared_bytes,
    "shared_limit_bytes": LDS_LIMIT_BYTES,
    "headroom_bytes": LDS_LIMIT_BYTES - shared_bytes,
    "fits_lds": shared_bytes <= LDS_LIMIT_BYTES,
    "preserves_pipe_route": route_change == "none",
    "route_change": route_change,
    "implementation_scope": implementation_scope,
  }


def _rank(c: dict[str, Any]) -> tuple[int, int, int]:
  scope_cost = {"tiny": 0, "small": 1, "medium": 2, "large": 3}.get(c["implementation_scope"], 9)
  route_cost = 0 if c["route_change"] == "none" else 1
  return (0 if c["fits_lds"] else 1, route_cost, scope_cost)


def build_report() -> dict[str, Any]:
  failure = analyze_amd_source(FAIL_SOURCE_SKETCH)
  baseline = _candidate(
    "current_generated_pipe_local_staging",
    "Current generated attn_kv pipe local staging reproduces the 69632 byte LDS declaration.",
    BASE_COMPONENTS,
    route_change="none",
    implementation_scope="baseline",
  )
  candidates = [
    _candidate(
      "disable_attn_kv_local_staging",
      "Gate attn_kv pipe local staging off and leave this role on raw/global operand fallback.",
      (),
      route_change="none",
      implementation_scope="tiny",
    ),
    _candidate(
      "retile_n_1024_to_512",
      "Keep local staging but split the N tile so the B-side local stage is halved.",
      (BASE_COMPONENTS[0], StageComponent("b_tile_local_stage", BASE_COMPONENTS[1].elements // 2)),
      route_change="tile_shape",
      implementation_scope="medium",
    ),
    _candidate(
      "byte_budgeted_local_staging",
      "Keep only the local-stage components that fit under the LDS byte budget for the role/shape.",
      (BASE_COMPONENTS[0],),
      route_change="none",
      implementation_scope="small",
    ),
  ]
  legal = [c for c in candidates if c["fits_lds"]]
  ranked = sorted(legal, key=_rank)
  recommendation = ranked[0]["name"] if ranked else None
  return {
    "schema": "attn-kv-pipe-resource-probe.v1",
    "role": "attn_kv",
    "shape": dict(ATTN_KV_SHAPE),
    "route_family": "pipe",
    "active_route_changed": False,
    "failure_source_analysis": failure,
    "baseline": baseline,
    "candidates": candidates,
    "legal_candidate_names": [c["name"] for c in legal],
    "smallest_legal_no_route_change": recommendation,
    "next_primitive_fix": recommendation,
    "interpretation": (
      "The smallest legal path that does not change the active route is to disable attn_kv local staging. "
      "Byte-budgeted local staging is the next general primitive because it also preserves the pipe route while "
      "retaining the legal A-side stage. Retiling N fixes the byte count but changes tile shape."
    ),
  }


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--json", action="store_true", help="print the full JSON report")
  args = ap.parse_args(argv)
  report = build_report()
  if args.json:
    print(json.dumps(report, indent=2, allow_nan=False))
  else:
    print(f"{report['schema']} role={report['role']} shape={report['shape']}")
    print(f"baseline_shared_bytes={report['baseline']['shared_bytes']} limit={LDS_LIMIT_BYTES}")
    for row in report["candidates"]:
      print(f"{row['name']}: shared={row['shared_bytes']} fits={row['fits_lds']} route_change={row['route_change']}")
    print(f"next_primitive_fix={report['next_primitive_fix']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
