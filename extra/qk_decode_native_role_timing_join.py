#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-native-tooling/role_timing_join.json"


def load(rel: str) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else None


def row(role: str, data: dict[str, Any]) -> dict[str, Any]:
  variants = (((data.get("programs") or {}).get("variants")) or [])
  main = next((p for p in variants if p.get("program_name") == "q4k_gemv_partial_12288_4096_1"), {})
  return {
    "role": role,
    "program_name": main.get("program_name"),
    "lib_sha16": main.get("lib_sha16"),
    "timing_us": None,
    "att_wall_ms": ((data.get("interval") or {}).get("target_wall_ms")),
    "timing_authority": "att_wall_not_authority",
    "usable_for_projection": False,
  }


def main() -> int:
  gate = load("bench/qk-att-inmodel-role-join/ffn_gate.json") or {}
  up = load("bench/qk-att-inmodel-role-join/ffn_up.json") or {}
  contract = load("bench/q8-ffn-amd-scheduler-project/oracle_contract.json") or {}
  timings = contract.get("known_timings_us") or {}
  rows = [row("ffn_gate", gate), row("ffn_up", up)]
  result = {
    "schema": "decode_native_role_timing_join_v1",
    "date": "2026-06-19",
    "verdict": "PROXY_ONLY",
    "rows": rows,
    "proxy_timing": {
      "native_q8_us": timings.get("tinygrad_asm_gateup_full"),
      "artifact_oracle_us": timings.get("hipcc_lld_gateup_current_loader"),
      "gap_us": round(timings.get("tinygrad_asm_gateup_full", 0) - timings.get("hipcc_lld_gateup_current_loader", 0), 3),
      "authority": "standalone_proxy",
    },
    "blocker": "blocked_same_binary_timing",
    "blocker_detail": (
      "Role ATT captures identify the in-model native Q4_K program, but only profiler-wall timing exists for the "
      "same interval. Existing q8 native/oracle timings are standalone proxy artifacts, not same-binary role timing."
    ),
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
