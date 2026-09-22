#!/usr/bin/env python3
"""Emit a per-target typed schedule template, or a full compact promoted artifact.

T6 of ``target-schedule-derivation-scope-20260801.md``: the mint stops cloning
AMD's schedule wholesale. This script computes the target's typed schedule via
``derive_target_schedule`` (declared row + the promoted AMD buffer2 geometry +
workload shape/dtypes) and writes the ``{"schedule", "static_constraints"}``
template BoltBeam's ``build_qwen3_8b_buffer2_candidate_set`` stamps identity
onto. The AMD gfx1100 control template is byte-identical to the promoted
schedule.

``--compact-out`` additionally mints the full checked-in compact artifact
(``tinygrad.prefill_wmma_lds_compact.v1``) for the target, in the exact shape
``tinygrad/llm/prefill_candidate_runtime.py`` expands: same template
vocabulary, the four production roles at their exact m=512 shapes, and the
canonical/legacy/candidate-set identities computed by the runtime's own
identity functions (imported from production, never reimplemented). For
gfx1100 the minted compact artifact expands to the exact checked-in AMD
candidate set (semantic equality, byte-format independent).
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
from tinygrad.llm.prefill_candidate_runtime import (
  ROUTE_ID, _candidate_identity, _legacy_candidate_identity,
  canonical_candidate_set_identity,
)

_TARGET_ROWS = {
  "gfx1100": GFX1100_TWO_BUFFER_STAGE1_CAPABILITY,
  "sm120": NV_SM120_TWO_BUFFER_STAGE1_CAPABILITY,
  "m4_10c": METAL_M4_10C_TWO_BUFFER_STAGE1_CAPABILITY,
}
_PROMOTED_SET = Path(__file__).resolve().parents[2] / "tinygrad" / "llm" / "generated" / \
  "prefill_wmma_lds_dbuf_candidate_set.json"
_AMD_BUFFER2_SET = Path(__file__).resolve().parents[2] / "bench" / "prefill-pure-full-kernel" / \
  "multirole-buffer2-candidate-set-v1" / "candidate-set.json"
_COMPACT_TARGETS = {
  "gfx1100": ({"backend": "AMD", "arch": "gfx1100", "wave_size": 32}, "gfx1100"),
  "sm120": ({"backend": "NV", "arch": "sm_120", "wave_size": 32}, "sm120"),
}
_ROLES_SHAPES = (("attn_kv", (512, 1024, 4096)), ("attn_qo", (512, 4096, 4096)),
                 ("ffn_down", (512, 4096, 12288)), ("ffn_gate_up", (512, 12288, 4096)))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("target", choices=tuple(_TARGET_ROWS), help="capability row to derive from")
  parser.add_argument("--out", required=True, help="write the typed schedule template JSON here")
  parser.add_argument("--compact-out", help="additionally mint the full compact promoted artifact here")
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
  if args.compact_out:
    if args.target not in _COMPACT_TARGETS:
      raise SystemExit(f"--compact-out has no declared target facts for {args.target}")
    target, profile_suffix = _COMPACT_TARGETS[args.target]
    compact = {
      "schema": "tinygrad.prefill_wmma_lds_compact.v1",
      "route_id": ROUTE_ID,
      "candidate_set_identity": "unset",
      "profile": f"qwen3_8b_q4k_m_{profile_suffix}",
      "target": dict(target),
      "template": {"schema_version": template["schema_version"], "dtypes": dict(template["dtypes"]),
                   "layout": dict(template["layout"]), "schedule": json.loads(json.dumps(out["schedule"])),
                   "static_constraints": dict(out["static_constraints"])},
      "entries": [],
    }
    expanded = {"schema": "boltbeam.full_kernel_candidate_set.v1", "entries": []}
    for role, (m, n, k) in _ROLES_SHAPES:
      payload = {"schema_version": compact["template"]["schema_version"],
                 "workload": {"profile": compact["profile"], "role": role,
                              "shape": {"m": m, "n": n, "k": k},
                              "dtypes": dict(compact["template"]["dtypes"]),
                              "layout": dict(compact["template"]["layout"]), "target": dict(target)},
                 "schedule": json.loads(json.dumps(compact["template"]["schedule"])),
                 "static_constraints": dict(compact["template"]["static_constraints"]),
                 "applicability": {"exact_shape": True, "profiles": [compact["profile"]],
                                   "roles": [role],
                                   "targets": [f"{target['backend']}:{target['arch']}:wave{target['wave_size']}"]}}
      canonical, legacy = _candidate_identity(payload), _legacy_candidate_identity(payload)
      compact["entries"].append({"role": role, "shape": {"m": m, "n": n, "k": k},
                                 "canonical_identity": canonical, "legacy_identity": legacy})
      expanded["entries"].append({"canonical_identity": canonical, "payload": payload})
    compact["candidate_set_identity"] = canonical_candidate_set_identity(expanded)
    Path(args.compact_out).write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
