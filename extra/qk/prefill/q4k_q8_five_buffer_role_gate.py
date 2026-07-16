"""Fail-closed, compile-only validation for admitted real-role five-buffer candidates.

The caller supplies one canonical candidate-inventory artifact.  This module does
not discover models or inventories and does not participate in route selection.
"""
from __future__ import annotations

from math import prod
from typing import Any, Callable, Mapping

from extra.qk.prefill.q4k_q8_five_buffer_compile_adapter import admit_q4k_q8_five_buffer_compile
from extra.qk.prefill.q4k_q8_five_buffer_execution_adapter import prepare_q4k_q8_five_buffer_compile
from extra.qk.prefill.workload_inventory import CANDIDATE_INVENTORY_SCHEMA
from extra.qk.runtime_specs import (FullKernelCandidateSet, admit_full_kernel_candidate_set,
  derive_q4k_q8_1_five_buffer_candidate, full_kernel_workload, rebind_full_kernel_workload)

SCHEMA = "tinygrad.q4k_q8_five_buffer_real_role_gate.v1"


def _fail(message: str) -> ValueError:
  return ValueError(f"five-buffer role gate: {message}")


def admitted_q4k_non_fitting_roles(artifact: Mapping[str, Any]):
  """Return every admitted Q4_K entry, ordered by structural role and M/N/K."""
  if not isinstance(artifact, Mapping) or artifact.get("schema") != CANDIDATE_INVENTORY_SCHEMA:
    raise _fail("unsupported candidate inventory schema")
  raw_set = artifact.get("candidate_sets", {}).get("Q4_K")
  if not isinstance(raw_set, Mapping): raise _fail("Q4_K candidate set is missing")
  candidate_set = FullKernelCandidateSet.from_json(dict(raw_set))
  admit_full_kernel_candidate_set(candidate_set)

  bindings = artifact.get("bindings")
  if not isinstance(bindings, list): raise _fail("bindings are missing")
  q4_bindings = {}
  for binding in bindings:
    try: key, identity = binding["inventory_key"], binding["canonical_identity"]
    except (KeyError, TypeError): raise _fail("malformed binding") from None
    if key.get("quant_format") != "Q4_K": continue
    if identity in q4_bindings: raise _fail("duplicate Q4_K binding identity")
    q4_bindings[identity] = key

  rows = []
  for inventory_entry in candidate_set.entries:
    # The committed inventory admits the packed workload.  Derive its physical
    # five-buffer candidate through the canonical primitive; do not reinterpret
    # profile labels or role names here.
    workload = full_kernel_workload(inventory_entry.payload)
    binding = q4_bindings.pop(inventory_entry.canonical_identity, None)
    if binding is None: raise _fail("admitted Q4_K candidate lacks an exact binding")
    shape = tuple(binding.get("shape", {}).get(x) for x in ("m", "n", "k"))
    if (binding.get("role"), shape) != (workload.role, workload.shape):
      raise _fail("binding role/shape differs from admission")
    template = rebind_full_kernel_workload(inventory_entry.payload, profile=workload.profile, role=workload.role,
      shape=workload.shape, target=workload.target)
    entry = derive_q4k_q8_1_five_buffer_candidate(template.payload)
    admission = admit_q4k_q8_five_buffer_compile(entry.payload, entry.canonical_identity)
    tile = admission.plan.tile
    if not any(dim > extent for dim, extent in zip(workload.shape, tile)):
      raise _fail(f"candidate {admission.canonical_identity} is fitting, not a real-role obligation")
    rows.append((workload.role, *workload.shape, entry, admission))
  if q4_bindings: raise _fail("Q4_K bindings exist outside the admitted candidate set")
  if not rows: raise _fail("no admitted non-fitting Q4_K roles")
  return tuple((entry, admission) for _, _, _, _, entry, admission in sorted(rows, key=lambda x:x[:4]))


def _validated_row(entry, admission, program, evidence: Mapping[str, Any]) -> dict[str, Any]:
  workload = full_kernel_workload(admission.normalized_payload)
  schedule = admission.normalized_payload["schedule"]
  tile = schedule["tile"]
  if schedule["lane_ownership"] != "rdna3_wave32_direct_wmma_output_tile":
    raise _fail("admitted lane ownership is not the direct one-wave output tile contract")
  if schedule["tail_policy"] != "aligned_only_no_tails":
    raise _fail("admitted tail policy is not aligned-only")
  if any(extent % tile[axis] for extent, axis in zip(workload.shape, ("m", "n", "k"))):
    raise _fail("aligned-only workload has an uncovered M/N/K tail")
  if evidence.get("passed") is not True: raise _fail("compile evidence did not pass")
  identity = admission.canonical_identity
  if evidence.get("canonical_identity") != identity: raise _fail("compile/admission context identity drift")
  context = getattr(program.src[0].arg, "candidate_context", None)
  # The compiler performs its own fail-closed admission, so object identity is
  # intentionally not required across that API boundary; canonical identity is.
  if getattr(context, "canonical_identity", None) != identity:
    raise _fail("PROGRAM context identity drift")
  abi = evidence.get("abi")
  abi_identity = evidence.get("abi_digest")
  if not isinstance(abi, Mapping) or not isinstance(abi_identity, str) or len(abi_identity) != 64:
    raise _fail("five-buffer ABI identity/evidence is missing")
  if abi.get("argument_order") != ["output", "q4_packed_words", "q8_ds4_values", "q8_scales", "q8_weighted_sums"]:
    raise _fail("five-buffer ABI order drift")
  resources = evidence.get("resource_summary")
  if not isinstance(resources, Mapping): raise _fail("final resource evidence is missing")
  required = ("lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills", "workgroup", "workgroup_threads", "grid")
  if any(key not in resources for key in required): raise _fail("final resource evidence is incomplete")
  workgroup, grid = resources["workgroup"], resources["grid"]
  if not isinstance(workgroup, list) or not workgroup or prod(workgroup) != admission.plan.threads:
    raise _fail("workgroup differs from admission")
  if not isinstance(grid, list) or not grid or any(type(x) is not int or x <= 0 for x in grid):
    raise _fail("grid evidence is invalid")
  expected_grid = [workload.shape[1] // tile["n"], workload.shape[0] // tile["m"], 1]
  if grid != expected_grid:
    raise _fail(f"final outer grid {grid!r} does not exactly own output tiles {expected_grid!r}")
  if resources["lds_bytes"] != admission.active_lds_bytes or any(resources[x] != 0 for x in
      ("scratch_bytes", "vgpr_spills", "sgpr_spills")):
    raise _fail("LDS/scratch/spill evidence violates admission")
  return {"role": workload.role, "M": workload.shape[0], "N": workload.shape[1], "K": workload.shape[2],
    "compile_status": "pass", "canonical_identity": identity, "context_identity": context.canonical_identity,
    "abi_identity": abi_identity, "abi": dict(abi), "workgroup": workgroup, "grid": grid,
    "coverage": {"tile": dict(tile), "output_tiles": expected_grid[:2],
      "owned_output_tiles": expected_grid[0] * expected_grid[1], "k_tiles": workload.shape[2] // tile["k"],
      "lane_ownership": schedule["lane_ownership"], "tail_policy": schedule["tail_policy"]},
    "resources": {key: resources[key] for key in ("lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills",
      "vgpr", "sgpr", "workgroup_threads", "wavefront_size") if key in resources}}


def build_role_gate(artifact: Mapping[str, Any], *,
                    compiler: Callable[[dict[str, Any], str], tuple[Any, Mapping[str, Any]]] = prepare_q4k_q8_five_buffer_compile) -> dict[str, Any]:
  """Compile all obligations and return a deterministic report; any failure closes the gate."""
  obligations = admitted_q4k_non_fitting_roles(artifact)
  rows, blockers = [], []
  for entry, admission in obligations:
    workload = full_kernel_workload(admission.normalized_payload)
    base = {"role": workload.role, "M": workload.shape[0], "N": workload.shape[1], "K": workload.shape[2],
            "canonical_identity": admission.canonical_identity}
    try:
      program, evidence = compiler(entry.payload, entry.canonical_identity)
    except Exception as exc:  # compilation is evidence, so unavailable compilation fails closed
      blocker = {**base, "compile_status": "blocked", "blocker_kind": "compiler",
                 "error_type": type(exc).__name__, "error": str(exc)}
      rows.append(blocker); blockers.append(blocker)
      continue
    try: rows.append(_validated_row(entry, admission, program, evidence))
    except Exception as exc:
      blocker = {**base, "compile_status": "fail", "blocker_kind": "gate_contract",
                 "error_type": type(exc).__name__, "error": str(exc)}
      rows.append(blocker); blockers.append(blocker)
  return {"schema": SCHEMA, "status": "pass" if not blockers else "fail", "passed": not blockers,
          "role_count": len(rows), "rows": rows, "blockers": blockers}


__all__ = ["SCHEMA", "admitted_q4k_non_fitting_roles", "build_role_gate"]
