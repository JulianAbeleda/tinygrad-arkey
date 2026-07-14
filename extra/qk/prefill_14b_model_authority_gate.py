#!/usr/bin/env python3
"""Compatibility entry point for the profile-driven 14B prefill authority gate."""
from __future__ import annotations

import argparse, json
from typing import Any

from extra.qk.model_profiles import qwen3_14b_q4k_m_gfx1100_profile
from extra.qk.prefill_model_authority_gate import build as build_profile_authority

DEFAULT_TARGET_ROUTE_IDS = (
  "prefill_q4k_int8_wmma_generated_research",
  "prefill_q4k_int8_wmma_tiled_research",
)


def build(*, target_route_ids:tuple[str,...]=DEFAULT_TARGET_ROUTE_IDS,
          representative_shapes:tuple[tuple[str,int,int,int], ...]|None=None, scope:str|None=None) -> dict[str,Any]:
  out = build_profile_authority(qwen3_14b_q4k_m_gfx1100_profile(), target_route_ids=target_route_ids,
                                representative_shapes=representative_shapes, scope=scope)
  out.update(schema="prefill_14b_model_authority_gate.v1", route="prefill_14b_model_authority",
    verdict="PREFILL_14B_MODEL_AUTHORITY_BLOCKED" if out["classified_blocker"] else "PREFILL_14B_MODEL_AUTHORITY_PASS")
  out["representative_q4k_shapes"] = out.pop("representative_shapes")
  return out


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--target-route", action="append", default=[])
  args = ap.parse_args()
  report = build(target_route_ids=tuple(args.target_route) or DEFAULT_TARGET_ROUTE_IDS)
  print(json.dumps(report, indent=2))
  raise SystemExit(0 if not report["classified_blocker"] else 1)
