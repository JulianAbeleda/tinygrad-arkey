#!/usr/bin/env python3
"""C5/T6 -- NV probe on the 5090 through the canonical lane (scratchpad run, not a commit).

The sm120 candidate set is now minted through the typed path (T6), so the
driver is one call per `(mint_path, device)` and the old `_nv_typed` retyping
step is deleted: the mint's schedule already carries the NV-typed facts
(`cuda_mma_*` fragment, `wmma_f32_8x16x16_f16`, `max_lds_bytes: 49152`,
null waitcnt), so `run_precontract_probe(ProbeConfig(..., device=device))`
is the whole probe. Expected outcomes on the 5090: admission passes for both
buffer shapes (the old capability_tc skip is gone) and the child compile fails
at the two known lowering boundaries (kernel_pipeline.py:181 accumulator
carrier vec(8) vs vec(4); kernel_lds.py:171 operand lane layout).

`load_candidate_payloads()` has no env override (manifest policy only), so the
driver hands the mint payload to the lane by monkeypatching the lane's own
`_base_payload_for_shape` accessor. Everything else is the public API.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import extra.llm_research.prefill.precontract_probe_lane as lane
from extra.llm_research.prefill.precontract_probe_lane import ProbeConfig, run_precontract_probe

MINT_PATH = pathlib.Path(
  "bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-sm120-v1/candidate-set.json")


def _mint_payload() -> dict:
  artifact = json.loads(MINT_PATH.read_text())
  return next(e["payload"] for e in artifact["entries"]
              if e["payload"]["workload"]["role"] == "attn_kv")


def main() -> None:
  state = {"payload": None}

  def _base_payload_for_shape(profile: str, role: str, shape: tuple[int, int, int]) -> dict:
    return copy.deepcopy(state["payload"])

  lane._base_payload_for_shape = _base_payload_for_shape

  variants = {
    "nv-typed-buffer2": ("CUDA", (128, 128, 32, 4, 2, 2)),
    "nv-typed-buffer1": ("CUDA", (128, 128, 32, 4, 2, 1)),
  }

  for name, (device, geometry) in variants.items():
    state["payload"] = _mint_payload()
    cfg = ProbeConfig("Q4_K", "attn_kv", (512, 1024, 4096), geometry,
                      device=device, rounds=3, warmups=1, timeout_seconds=240)
    result = run_precontract_probe(cfg)
    print(f"== {name} -> status={result.status}")
    if result.skip_reason:
      print(f"   skip_reason: {result.skip_reason}")
    print("   " + json.dumps(result.to_json())[:1800])


if __name__ == "__main__":
  main()
