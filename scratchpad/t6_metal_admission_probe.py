#!/usr/bin/env python3
"""T6 -- Metal m4_10c admission-level probe through the canonical lane.

The m4_10c candidate set is minted through the typed path (T6). This driver is
admission-level only by default (pure Python, no GPU touch), so it runs on any
box; pass ``--gpu`` on a Mac to run the full lane (compile + guarded
execution). Expected admission outcomes: buffer2 is rejected by
``capability_lds`` because the 128x128x32 / waves 4x2 geometry needs 40960
bytes and the m4_10c row declares 32768; buffer1 admits (20480 <= 32768).
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import extra.llm_research.prefill.precontract_probe_lane as lane
from extra.llm_research.prefill.precontract_probe_lane import ProbeConfig, admit_probe_config, run_precontract_probe

MINT_PATH = pathlib.Path(
  "bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-m4-10c-v1/candidate-set.json")


def _mint_payload() -> dict:
  artifact = json.loads(MINT_PATH.read_text())
  return next(e["payload"] for e in artifact["entries"]
              if e["payload"]["workload"]["role"] == "attn_kv")


def main() -> int:
  gpu = "--gpu" in sys.argv
  state = {"payload": None}

  def _base_payload_for_shape(profile: str, role: str, shape: tuple[int, int, int]) -> dict:
    return copy.deepcopy(state["payload"])

  lane._base_payload_for_shape = _base_payload_for_shape

  for name, geometry in (("m4_10c-typed-buffer2", (128, 128, 32, 4, 2, 2)),
                         ("m4_10c-typed-buffer1", (128, 128, 32, 4, 2, 1))):
    state["payload"] = _mint_payload()
    cfg = ProbeConfig("Q4_K", "attn_kv", (512, 1024, 4096), geometry,
                      device="Metal", rounds=3, warmups=1, timeout_seconds=240)
    if not gpu:
      try:
        entry, admission = admit_probe_config(cfg)
        print(f"== {name} -> admission ADMITS active_lds={admission.active_lds_bytes} "
              f"capability={admission.capability.capability_id}")
      except (ValueError, Exception) as exc:  # noqa: BLE001 -- admission outcomes are the report
        print(f"== {name} -> admission rejected: {exc}")
      continue
    result = run_precontract_probe(cfg)
    print(f"== {name} -> status={result.status}")
    if result.skip_reason:
      print(f"   skip_reason: {result.skip_reason}")
    print("   " + json.dumps(result.to_json())[:1800])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
