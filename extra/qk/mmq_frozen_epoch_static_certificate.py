#!/usr/bin/env python3
"""CPU-only C1/C2/C3 certificate for one exact frozen epoch family.

This module composes the existing strict frozen-family loader, native AMD
resource authority, and exhaustive memory-semantics certificate.  It does not
compile programs or create a Device, runtime, allocator, or queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import prod
from pathlib import Path
from typing import Any, Mapping

from extra.qk.mmq_capability import GFX11_MMQ_CAPABILITY
from extra.qk.mmq_exact_role_spec import DEFAULT_INVENTORY, ExactRoleSpec, exact_role_spec
from extra.qk.mmq_frozen_epoch_memory_certificate import certify_frozen_epoch_program_family
from extra.qk.mmq_frozen_epoch_program_set import (
  PROVENANCE_SCHEMA, SCHEMA as FROZEN_SCHEMA, load_frozen_epoch_program_set_binding,
)
from extra.qk.mmq_llama_candidate_plan import llama_mmq_candidate_plan
from extra.qk.mmq_llama_five_buffer_full_kernel import AMD_ISA_TARGET
from extra.qk.prefill.amd_native_program_resources import amd_native_program_resources


SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_epoch_static_certificate.v1"
MAX_VGPR_PER_THREAD = 256


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
  options = {"sort_keys": True, "allow_nan": False}
  if pretty: options["indent"] = 2
  else: options["separators"] = (",", ":")
  return (json.dumps(value, **options) + "\n").encode()


def _sha256_json(value: Any) -> str:
  return hashlib.sha256(_json_bytes(value)).hexdigest()


def _reject(epoch: int, message: str) -> None:
  raise ValueError(f"C2 epoch {epoch} native resource certificate rejected: {message}")


def _resource_constraints(role_spec: ExactRoleSpec) -> dict[str, Any]:
  capability = GFX11_MMQ_CAPABILITY
  capability.validate()
  geometry = llama_mmq_candidate_plan().geometry
  if (capability.backend, capability.arch, capability.wave_width) != ("AMD", "gfx1100", 32):
    raise ValueError("C2 declared gfx1100 MMQ capability changed")
  if geometry.threads != prod((256, 1, 1)) or geometry.lds_bytes > capability.max_lds_bytes:
    raise ValueError("C2 declared llama MMQ geometry exceeds its gfx1100 capability")
  return {
    "target": AMD_ISA_TARGET,
    "max_vgpr_per_thread": MAX_VGPR_PER_THREAD,
    "max_lds_bytes": capability.max_lds_bytes,
    "expected_lds_bytes": geometry.lds_bytes,
    "allow_scratch": False,
    "allow_spills": False,
    "wavefront_size": capability.wave_width,
    "global_size": list(role_spec.program.grid),
    "local_size": [geometry.threads, 1, 1],
  }


def _certify_resource(epoch: int, program_key: str, resources: Mapping[str, Any],
                      constraints: Mapping[str, Any]) -> dict[str, Any]:
  if resources.get("schema") != "tinygrad.amd.native_program_resources.v1":
    _reject(epoch, "native resource schema changed")
  if resources.get("target") != str(constraints["target"]).rsplit(":", 1)[-1]:
    _reject(epoch, "target differs from the declared architecture")
  for field in ("scratch_bytes", "vgpr_spills", "sgpr_spills"):
    if resources.get(field) != 0: _reject(epoch, f"{field} is not zero")
  allocated_vgpr, used_vgpr = resources.get("allocated_vgpr"), resources.get("used_vgpr")
  if not isinstance(allocated_vgpr, int) or isinstance(allocated_vgpr, bool) or \
     not 0 < allocated_vgpr <= constraints["max_vgpr_per_thread"]:
    _reject(epoch, "allocated VGPR count exceeds the declared limit")
  if not isinstance(used_vgpr, int) or isinstance(used_vgpr, bool) or not 0 <= used_vgpr <= allocated_vgpr:
    _reject(epoch, "used VGPR count is invalid or exceeds its descriptor allocation")
  lds_bytes = resources.get("lds_bytes")
  if lds_bytes != constraints["expected_lds_bytes"] or lds_bytes > constraints["max_lds_bytes"]:
    _reject(epoch, "LDS count differs from the declared llama geometry or exceeds the target limit")
  if resources.get("wavefront_size") != constraints["wavefront_size"]:
    _reject(epoch, "wavefront size differs from the declared capability")
  if resources.get("global_size") != constraints["global_size"] or \
     resources.get("local_size") != constraints["local_size"]:
    _reject(epoch, "launch geometry differs from the exact role contract")
  if resources.get("workgroup_threads") != prod(constraints["local_size"]) or \
     resources.get("max_workgroup_threads") != prod(constraints["local_size"]):
    _reject(epoch, "workgroup thread count differs from the exact launch")
  return {"epoch": epoch, "program_key": program_key, "resources": dict(resources)}


def certify_frozen_epoch_static(role_spec: ExactRoleSpec, bundle: str | Path, *,
                                inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
                                target: str = AMD_ISA_TARGET) -> dict[str, Any]:
  """Strictly certify C1 provenance, C2 resources, and C3 memory semantics.

  Every retained epoch PROGRAM is independently inspected.  The returned
  report is deterministic and content-addressed; callers may serialize it with
  :func:`static_certificate_json`.
  """
  if target != AMD_ISA_TARGET:
    raise ValueError(f"C2 target drift: expected {AMD_ISA_TARGET!r}")
  binding = load_frozen_epoch_program_set_binding(
    role_spec, bundle, inventory=inventory, require_c1=True)
  artifact, admitted = binding.artifact, binding.role_spec
  manifest = artifact.manifest
  if manifest.get("schema") != FROZEN_SCHEMA or manifest.get("c1_certification", {}).get("certified") is not True:
    raise ValueError("C1 strict frozen-family certification is absent")
  provenance = manifest.get("generation_provenance")
  if not isinstance(provenance, Mapping) or provenance.get("schema") != PROVENANCE_SCHEMA:
    raise ValueError("C1 generation provenance is absent or malformed")
  if len(artifact.programs) != admitted.epochs or len(artifact.sinks) != admitted.epochs:
    raise ValueError("C1 artifact does not contain every exact-role epoch")

  constraints = _resource_constraints(admitted)
  program_keys = tuple(program.key.hex() for program in artifact.programs)
  if program_keys != binding.program_keys:
    raise ValueError("C1 loaded PROGRAM order differs from the frozen binding")
  resource_rows = tuple(
    _certify_resource(epoch, program_key, amd_native_program_resources(program, target=target), constraints)
    for epoch, (program_key, program) in enumerate(zip(program_keys, artifact.programs))
  )
  memory = certify_frozen_epoch_program_family(admitted, artifact.sinks, artifact.programs)
  if memory.get("state") != "PASS" or memory.get("cpu_only") is not True:
    raise ValueError("C3 frozen-family memory certificate did not pass CPU-only")

  body = {
    "schema": SCHEMA,
    "state": "PASS",
    "cpu_only": True,
    "role": {
      "name": admitted.role,
      "shape": list(admitted.shape),
      "epochs": admitted.epochs,
      "candidate_identity": admitted.candidate_canonical_identity,
    },
    "gates": {"C1": "PASS", "C2": "PASS", "C3": "PASS"},
    "c1": {
      "frozen_schema": manifest["schema"],
      "provenance_schema": provenance["schema"],
      "generation_provenance": dict(provenance),
      "family_identity": binding.family_identity,
      "content_address": manifest["content_address"],
      "ordered_program_keys": list(binding.program_keys),
    },
    "c2": {
      "schema": "tinygrad.mmq_q4k_q8_1.frozen_epoch_native_resources.v1",
      "state": "PASS",
      "constraints": constraints,
      "variant_count": len(resource_rows),
      "variants": list(resource_rows),
    },
    "c3": memory,
  }
  return {**body, "certificate_sha256": _sha256_json(body)}


def static_certificate_json(certificate: Mapping[str, Any]) -> bytes:
  """Return the canonical, human-readable deterministic report encoding."""
  return _json_bytes(certificate, pretty=True)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--role", required=True, help="exact inventory-admitted role name")
  parser.add_argument("--bundle", required=True, type=Path, help="frozen v3 directory or ustar archive")
  parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
  parser.add_argument("--output", type=Path, help="write report here instead of stdout")
  args = parser.parse_args(argv)
  role_spec = exact_role_spec(args.role, inventory=args.inventory)
  payload = static_certificate_json(
    certify_frozen_epoch_static(role_spec, args.bundle, inventory=args.inventory))
  if args.output is None:
    import sys
    sys.stdout.buffer.write(payload)
  else:
    args.output.write_bytes(payload)
  return 0


if __name__ == "__main__": raise SystemExit(main())


__all__ = [
  "MAX_VGPR_PER_THREAD", "SCHEMA", "certify_frozen_epoch_static", "static_certificate_json",
]
