#!/usr/bin/env python3
"""Emit a per-target typed schedule template for BoltBeam's candidate-set builder.

T6 of ``target-schedule-derivation-scope-20260801.md``: the mint stops cloning
AMD's schedule wholesale. This script computes the target's typed schedule via
``derive_target_schedule`` (declared row + the promoted AMD buffer2 geometry +
workload shape/dtypes) and writes the ``{"schedule", "static_constraints"}``
template BoltBeam's ``build_qwen3_8b_buffer2_candidate_set`` stamps identity
onto. The AMD gfx1100 control template is byte-identical to the promoted
schedule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extra.llm_research.runtime_specs import (
  GFX1100_TWO_BUFFER_STAGE1_CAPABILITY, METAL_M4_10C_TWO_BUFFER_STAGE1_CAPABILITY,
  NV_SM120_TWO_BUFFER_STAGE1_CAPABILITY,
)
from extra.llm_research.target_schedule import derive_target_schedule

_TARGET_ROWS = {
  "gfx1100": GFX1100_TWO_BUFFER_STAGE1_CAPABILITY,
  "sm120": NV_SM120_TWO_BUFFER_STAGE1_CAPABILITY,
  "m4_10c": METAL_M4_10C_TWO_BUFFER_STAGE1_CAPABILITY,
}
_PROMOTED_SET = Path(__file__).resolve().parents[2] / "tinygrad" / "llm" / "generated" / \
  "prefill_wmma_lds_dbuf_candidate_set.json"
_AMD_BUFFER2_SET = Path(__file__).resolve().parents[2] / "bench" / "prefill-pure-full-kernel" / \
  "multirole-buffer2-candidate-set-v1" / "candidate-set.json"


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("target", choices=tuple(_TARGET_ROWS), help="capability row to derive from")
  parser.add_argument("--out", required=True, help="write the typed schedule template JSON here")
  args = parser.parse_args()

  promoted = json.loads(_PROMOTED_SET.read_text())
  template = promoted["template"]
  geometry = {"tile": dict(template["schedule"]["tile"]), "waves": dict(template["schedule"]["waves"]),
              "buffer_count": template["schedule"]["pipeline"]["buffer_count"],
              "stage_count": template["schedule"]["pipeline"]["stage_count"]}
  gate_up = next(e["payload"]["workload"] for e in json.loads(_AMD_BUFFER2_SET.read_text())["entries"]
                 if e["payload"]["workload"]["role"] == "ffn_gate_up")
  shape = {"m": gate_up["shape"]["m"], "n": gate_up["shape"]["n"], "k": gate_up["shape"]["k"],
           "dtypes": dict(template["dtypes"])}
  out = derive_target_schedule(_TARGET_ROWS[args.target], geometry, shape)
  Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
