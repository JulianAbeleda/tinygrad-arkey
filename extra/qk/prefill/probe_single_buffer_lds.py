"""Compile-only A/B probe for the generated prefill LDS stage count.

This intentionally does not modify the promoted candidate set or route manifest.
It derives a new content-addressed one-buffer payload from one admitted default
candidate, compiles both code objects, and writes only final compiler evidence.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from extra.qk.prefill.current_prefill_execution_adapter import prepare_current_prefill_compile
from extra.qk.prefill.packed_wmma_correctness_canary import candidate_payload
from extra.qk.runtime_specs import FullKernelCandidateSetEntry


def _entry(payload: dict[str, Any]) -> FullKernelCandidateSetEntry:
  # The constructor is the canonical full-kernel payload/hash authority.
  provisional = FullKernelCandidateSetEntry.__new__(FullKernelCandidateSetEntry)
  from extra.qk.runtime_specs import _canonical_full_kernel_identity
  return FullKernelCandidateSetEntry(_canonical_full_kernel_identity(payload), payload)


def one_buffer_payload(payload: dict[str, Any]) -> dict[str, Any]:
  probe = deepcopy(payload)
  pipeline = probe["schedule"]["pipeline"]
  if (pipeline.get("buffer_count"), pipeline.get("stage_count")) != (2, 1):
    raise ValueError("probe requires the admitted two-buffer stage-1 baseline")
  pipeline["buffer_count"] = 1
  return probe


def _record(label: str, entry: FullKernelCandidateSetEntry) -> dict[str, Any]:
  _, evidence = prepare_current_prefill_compile(entry.to_json()["payload"], entry.canonical_identity, device="AMD")
  resources = evidence["resource_summary"]
  return {"label": label, "canonical_identity": entry.canonical_identity,
          "source_sha256": evidence["source_sha256"], "binary_sha256": evidence["binary_sha256"],
          "resources": {name: resources[name] for name in ("vgpr", "allocated_vgpr", "sgpr", "lds_bytes",
                        "scratch_bytes", "vgpr_spills", "sgpr_spills", "workgroup_threads", "wavefront_size")},
          "isa_artifact_status": evidence["artifacts"]["final_isa_manifest"]["status"]}


def run(profile: str, role: str) -> dict[str, Any]:
  baseline = _entry(candidate_payload(profile, role))
  reduced = _entry(one_buffer_payload(baseline.to_json()["payload"]))
  before, after = _record("two_buffer_baseline", baseline), _record("one_buffer_probe", reduced)
  return {"schema": "tinygrad.prefill_lds_single_buffer_compile_probe.v1", "mode": "compile_only_no_dispatch",
          "profile": profile, "role": role, "baseline": before, "probe": after,
          "delta": {name: after["resources"][name] - before["resources"][name]
                    for name in ("vgpr", "allocated_vgpr", "sgpr", "lds_bytes", "scratch_bytes",
                                 "vgpr_spills", "sgpr_spills")},
          "promotion": "not_registered_not_promoted"}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--profile", default="qwen3_8b_q4k_m_gfx1100")
  parser.add_argument("--role", default="attn_qo")
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  result = run(args.profile, args.role)
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
