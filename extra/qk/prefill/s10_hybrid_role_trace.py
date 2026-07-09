#!/usr/bin/env python3
"""S10 hybrid role trace over the S9 graph-GEMM backend atoms.

This artifact is intentionally non-invasive: it does not enable generated S10
primitive flags and does not lower or launch kernels. It records the current
S9 graph-GEMM role policy as S10-owned metadata around the existing backend
atoms.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from extra.qk.prefill_schedule_spec import describe_prefill_schedule
from extra.qk.wmma_lds_spec import extract_wmma_lds_spec

ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "bench/prefill-s10-lds2-ownership/hybrid-s9-s10-role-trace.json"
HYBRID_CLASSIFICATION = "hybrid_machine_searched_over_hand_tuned_backend_atoms"
S9_ENV = {"PREFILL_V2": "1", "PREFILL_GRAPH_GEMM": "1"}
FORBIDDEN_PRIMITIVE_ENVS = ("PREFILL_WMMA_PIPE_PRIMITIVE", "PREFILL_WMMA_LDS_PRIMITIVE", "PREFILL_DBUF")

PREFILL_8B_ROLES: tuple[tuple[str, int, int], ...] = (
  ("attn_qo", 4096, 4096),
  ("attn_kv", 1024, 4096),
  ("ffn_down", 4096, 12288),
  ("ffn_gate_up", 12288, 4096),
)


def _lds_summary(lds_spec) -> dict[str, Any] | None:
  if lds_spec is None: return None
  data = lds_spec.to_json()
  keep = (
    "m", "n", "k", "tile_m", "tile_n", "tile_k", "waves_m", "waves_n", "wm", "wn", "threads",
    "pad", "dbuf", "plra", "plrab", "leanaddr", "selection_label", "ownership_classification",
    "lds_total_bytes", "legality_errors",
  )
  return {k: data[k] for k in keep if k in data}


def role_row(role: str, out_f: int, in_f: int) -> dict[str, Any]:
  spec = describe_prefill_schedule(out_f, in_f, role=role)
  lds_spec = extract_wmma_lds_spec(spec) if spec.route_family == "lds" else None
  backend_atom = "build_gemm_pipe" if spec.route_family == "pipe" else "lower_lds2_gemm_kernel/build_gemm_lds2"
  row = {
    "role": role,
    "shape": {"m": 512, "n": out_f, "k": in_f},
    "route_family": spec.route_family,
    "kernel_name": spec.kernel_name,
    "backend_atom": backend_atom,
    "classification": HYBRID_CLASSIFICATION,
    "s10_ownership_claim": "records schedule/spec/search metadata; S9 backend atom emits instruction lifecycle",
    "schedule_spec": spec.to_json(),
  }
  if lds_spec is not None: row["lds_spec_summary"] = _lds_summary(lds_spec)
  if lds_spec is not None: row["hand_coded_epoch_primitive"] = lds_spec.dbuf_epoch_primitive.to_json()
  return row


def build_trace() -> dict[str, Any]:
  rows = [role_row(role, out_f, in_f) for role, out_f, in_f in PREFILL_8B_ROLES]
  return {
    "schema": "prefill-s10-hybrid-s9-s10-role-trace.v1",
    "env": dict(S9_ENV),
    "forbidden_env": list(FORBIDDEN_PRIMITIVE_ENVS),
    "classification": HYBRID_CLASSIFICATION,
    "acceptance_gate": {
      "pp512_min_tok_s": 4000,
      "authority_command": "PREFILL_V2=1 PREFILL_GRAPH_GEMM=1 python3 extra/qk/prefill_whole_synced.py "
                           "--mode authority -K 8 --warmups 4 --rounds 3 --whole-lengths 512 --pin-clock",
      "primitive_flags_allowed": False,
    },
    "rows": rows,
  }


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args(argv)

  report = build_trace()
  out = args.output
  if not out.is_absolute(): out = ROOT / out
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2) + "\n")
  if args.json: print(json.dumps(report, indent=2))
  else: print(out)
  return report


if __name__ == "__main__":
  main()
