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
# Dense bf16 reuse: the promoted NV sm_120 schedule stamped verbatim onto Nemotron-H 4B BF16's exact projections
# ((N, K) from the GGUF; N padded up to the 128-row tile) at every padded row chunk the router emits.
_NV_PROMOTED_SET = _PROMOTED_SET.with_name("prefill_sm120_lds_dbuf_candidate_set.json")
_DENSE_BF16_ARTIFACT = _PROMOTED_SET.with_name("dense_bf16_sm120_candidate_set.json")
_DENSE_BF16_PROFILE = "nemotron_h_4b_bf16_sm120"
_DENSE_BF16_ROLES = (("ssm_in", 17504, 3136), ("ssm_out", 3136, 7680), ("attn_q", 5120, 3136), ("attn_kv", 1024, 3136),
                     ("attn_o", 3136, 5120), ("ffn_up", 12544, 3136), ("ffn_down", 3136, 12544), ("output", 131072, 3136))
_DENSE_BF16_ROWS = (128, 256, 384, 512)
# Rows that lost to the safe tensor-core path in the promotion gate (32 CTAs on 170 SMs); left unpromoted.
_DENSE_BF16_LOSERS = {("attn_kv", 128), ("attn_kv", 256), ("attn_kv", 384)}
_ROLES_SHAPES = (("attn_kv", (512, 1024, 4096)), ("attn_qo", (512, 4096, 4096)),
                 ("ffn_down", (512, 4096, 12288)), ("ffn_gate_up", (512, 12288, 4096)))


def _compact(profile:str, target:dict, template:dict, roles_shapes) -> dict:
  compact = {"schema": "tinygrad.prefill_wmma_lds_compact.v1", "route_id": ROUTE_ID, "candidate_set_identity": "unset",
             "profile": profile, "target": dict(target), "template": template, "entries": []}
  expanded = {"schema": "boltbeam.full_kernel_candidate_set.v1", "entries": []}
  for role, (m, n, k) in roles_shapes:
    payload = {"schema_version": template["schema_version"],
               "workload": {"profile": profile, "role": role, "shape": {"m": m, "n": n, "k": k},
                            "dtypes": dict(template["dtypes"]), "layout": dict(template["layout"]), "target": dict(target)},
               "schedule": json.loads(json.dumps(template["schedule"])),
               "static_constraints": dict(template["static_constraints"]),
               "applicability": {"exact_shape": True, "profiles": [profile], "roles": [role],
                                 "targets": [f"{target['backend']}:{target['arch']}:wave{target['wave_size']}"]}}
    canonical, legacy = _candidate_identity(payload), _legacy_candidate_identity(payload)
    compact["entries"].append({"role": role, "shape": {"m": m, "n": n, "k": k},
                               "canonical_identity": canonical, "legacy_identity": legacy})
    expanded["entries"].append({"canonical_identity": canonical, "payload": payload})
  compact["candidate_set_identity"] = canonical_candidate_set_identity(expanded)
  return compact


def mint_dense_bf16() -> dict:
  """Reuse, not search: the promoted NV schedule and constraints verbatim, only the workload dtypes and shapes change."""
  nv = json.loads(_NV_PROMOTED_SET.read_text())
  template = {**json.loads(json.dumps(nv["template"])), "dtypes": {"a": "bf16", "b": "bf16", "accumulator": "fp32", "c": "fp32"}}
  tile_m, tile_n = template["schedule"]["tile"]["m"], template["schedule"]["tile"]["n"]
  assert all(rows % tile_m == 0 for rows in _DENSE_BF16_ROWS)
  roles_shapes = [(role, (rows, -(-n // tile_n) * tile_n, k)) for role, n, k in _DENSE_BF16_ROLES for rows in _DENSE_BF16_ROWS
                  if (role, rows) not in _DENSE_BF16_LOSERS]
  return _compact(_DENSE_BF16_PROFILE, nv["target"], template, roles_shapes)


def main() -> int:
  if sys.argv[1:] == ["--dense-bf16"]:
    _DENSE_BF16_ARTIFACT.write_text(json.dumps(mint_dense_bf16(), indent=2, sort_keys=True) + "\n")
    return 0
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
