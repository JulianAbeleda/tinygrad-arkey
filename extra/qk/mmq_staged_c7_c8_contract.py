"""Reusable CPU-only C7/C8 evidence contracts for frozen staged MMQ roles.

C7 joins the logical byte requirements in a frozen staged-family manifest to
per-queue physical allocation lifetimes.  C8 accepts synchronized-wall timing
rounds whose candidate side enumerates every stage copy, synchronization, and
epoch dispatch.  Neither builder allocates a device, opens a queue, or changes
route policy.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import random
import statistics
from typing import Any

from extra.qk.mmq_frozen_staged_family import (
  QUEUE_MODES, SCHEMA as STAGED_FAMILY_SCHEMA, STATE as STAGED_FAMILY_STATE,
  FrozenStagedFamily, _manifest_payload, _validate_provenance,
)
from extra.qk.physical_memory_ledger import PhysicalMemoryEvidence
from extra.qk.prefill.frozen_exact_role_runtime import ABI_DTYPES, FrozenExactRoleBinding


C7_SCHEMA = "tinygrad.mmq_q4k_q8_1.staged_c7_memory_ledger.v1"
C8_SCHEMA = "tinygrad.mmq_q4k_q8_1.staged_c8_complete_role_timing.v1"
DIRECT_FALLBACK = "direct_packed"

_LOGICAL_CATEGORIES = (
  "full_q4_source", "full_q8_values_source", "full_q8_scales_source", "full_q8_sums_source",
  "compact_q4_stage", "compact_q8_values_stage", "compact_q8_scales_stage", "compact_q8_sums_stage",
  "output",
)
_INFRASTRUCTURE_CATEGORIES = ("code_object", "runtime", "kernarg", "queue_state")
_TEMPORARY_CATEGORIES = ("temporary_gather", "temporary_transfer")
_REQUIRED_CATEGORIES = _LOGICAL_CATEGORIES + _INFRASTRUCTURE_CATEGORIES + _TEMPORARY_CATEGORIES
_FULL_WINDOW_CATEGORIES = _LOGICAL_CATEGORIES + ("code_object", "runtime", "queue_state")
_ALLOWED_CATEGORIES = frozenset(_REQUIRED_CATEGORIES + ("co_resident_model",))
_DTYPE_BYTES = {str(dtype): dtype.itemsize for dtype in ABI_DTYPES}
_HEX = frozenset("0123456789abcdef")
_ROW_SOURCES = frozenset(("physical_memory_ledger", "runtime_allocation_census", "explicit_zero_measurement"))


def _canonical(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(payload: Mapping[str, Any]) -> str:
  return "sha256:" + hashlib.sha256(_canonical(payload)).hexdigest()


def _digest(value: Any, label: str) -> str:
  value = _nonempty(value, label)
  if not value.startswith("sha256:") or len(value) != 71 or any(char not in _HEX for char in value[7:]):
    raise ValueError(f"{label} must be a sha256 content identity")
  return value


def _nonempty(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value: raise ValueError(f"{label} must be a non-empty string")
  return value


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < (1 if positive else 0):
    qualifier = "positive" if positive else "non-negative"
    raise ValueError(f"{label} must be a {qualifier} integer")
  return value


def _number(value: Any, label: str) -> float:
  if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
    raise ValueError(f"{label} must be a finite non-negative number")
  return float(value)


def _positive_number(value: Any, label: str) -> float:
  result = _number(value, label)
  if result <= 0: raise ValueError(f"{label} must be positive")
  return result


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
  if set(value) != keys:
    raise ValueError(f"{label} fields differ: expected {sorted(keys)!r}, got {sorted(value)!r}")


def _family(family: FrozenStagedFamily) -> tuple[Mapping[str, Any], str, int, dict[str, int]]:
  if not isinstance(family, FrozenStagedFamily) or not isinstance(family.binding, FrozenExactRoleBinding):
    raise TypeError("family must be a loader-validated FrozenStagedFamily")
  family_manifest = family.manifest
  if not isinstance(family_manifest, Mapping):
    raise ValueError("frozen staged-family manifest must be a mapping")
  _exact_keys(family_manifest, {
    "schema", "state", "family_identity", "role", "artifact", "program",
    "staging", "queue_modes", "provenance",
  }, "frozen staged-family manifest")
  if family_manifest.get("schema") != STAGED_FAMILY_SCHEMA or \
     family_manifest.get("state") != STAGED_FAMILY_STATE:
    raise ValueError("frozen staged-family schema or state differs")
  family_identity = _nonempty(family_manifest.get("family_identity"), "family_identity")
  if family.family_identity != family_identity:
    raise ValueError("typed staged-family identity differs from its manifest")
  provenance = _validate_provenance(family_manifest["provenance"], family.binding)
  expected = _manifest_payload(family.binding.role_spec, family.binding, provenance)
  observed = {key: value for key, value in family_manifest.items() if key != "family_identity"}
  if observed != expected or family_identity != _identity(expected):
    raise ValueError("staged-family content differs from its exact binding or content identity")
  source = family.binding.artifact.source.encode()
  binary = family.binding.artifact.binary
  if hashlib.sha256(source).hexdigest() != family.binding.source_sha256 or \
     hashlib.sha256(binary).hexdigest() != family.binding.binary_sha256:
    raise ValueError("staged-family binding payload digest differs")
  role, program, staging = (family_manifest.get(key) for key in ("role", "program", "staging"))
  if not all(isinstance(value, Mapping) for value in (role, program, staging)):
    raise ValueError("frozen staged-family role/program/staging contract is incomplete")
  epochs = _integer(role.get("epoch_count"), "role.epoch_count", positive=True)
  if program.get("dispatch_count") != epochs:
    raise ValueError("program dispatch_count differs from role epoch_count")
  inputs = staging.get("inputs")
  abi = program.get("abi")
  if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)) or len(inputs) != 4:
    raise ValueError("staging.inputs must enumerate the four staged inputs")
  if not isinstance(abi, Sequence) or isinstance(abi, (str, bytes)) or len(abi) != 5:
    raise ValueError("program.abi must enumerate the five-buffer ABI")
  abi_by_slot = {row.get("slot"): row for row in abi if isinstance(row, Mapping)}
  if set(abi_by_slot) != set(range(5)): raise ValueError("program.abi slots differ from the five-buffer ABI")
  exact_abi = [{
    "slot": slot, "name": name, "dtype": f"{dtype}.ptr({elements})", "elements": elements,
    "nbytes": elements * dtype.itemsize, "direction": "inout" if slot == 0 else "in",
  } for slot, (name, dtype, elements) in enumerate(zip(
    ("output", "q4", "q8_values", "q8_scales", "q8_original_sums"),
    ABI_DTYPES, family.binding.role_spec.program.abi_elements))]
  if list(abi) != exact_abi:
    raise ValueError("program ABI dtype/elements/nbytes differs from the exact role binding")

  requirements: dict[str, int] = {}
  names = (
    ("full_q4_source", "compact_q4_stage"),
    ("full_q8_values_source", "compact_q8_values_stage"),
    ("full_q8_scales_source", "compact_q8_scales_stage"),
    ("full_q8_sums_source", "compact_q8_sums_stage"),
  )
  for expected_slot, (row, (source_category, stage_category)) in enumerate(zip(inputs, names), start=1):
    if not isinstance(row, Mapping) or row.get("slot") != expected_slot:
      raise ValueError("staging input order or slot differs from the five-buffer ABI")
    dtype = abi_by_slot[expected_slot].get("dtype")
    dtype_name = next((name for name in _DTYPE_BYTES if isinstance(dtype, str) and dtype.startswith(name + ".ptr(")), None)
    if dtype_name is None: raise ValueError(f"program.abi slot {expected_slot} has an unsupported dtype")
    source, stage = row.get("source"), row.get("stage")
    if not isinstance(source, Mapping) or not isinstance(stage, Mapping):
      raise ValueError(f"staging input slot {expected_slot} source/stage contract is incomplete")
    source_elements = _integer(source.get("elements"), f"staging.inputs[{expected_slot-1}].source.elements", positive=True)
    stage_elements = _integer(stage.get("elements"), f"staging.inputs[{expected_slot-1}].stage.elements", positive=True)
    requirements[source_category] = source_elements * _DTYPE_BYTES[dtype_name]
    requirements[stage_category] = stage_elements * _DTYPE_BYTES[dtype_name]
  requirements["output"] = _integer(abi_by_slot[0].get("nbytes"), "program.abi[0].nbytes", positive=True)
  return family_manifest, family_identity, epochs, requirements


def staged_logical_memory_requirements(family: FrozenStagedFamily) -> dict[str, Any]:
  """Return manifest-derived source, compact-stage, and output byte requirements."""
  _, family_identity, epochs, requirements = _family(family)
  return {
    "family_identity": family_identity,
    "epoch_count": epochs,
    "components": dict(requirements),
    "full_source_bytes": sum(requirements[name] for name in _LOGICAL_CATEGORIES if name.startswith("full_")),
    "compact_stage_bytes": sum(requirements[name] for name in _LOGICAL_CATEGORIES if name.startswith("compact_")),
    "output_bytes": requirements["output"],
  }


def physical_lifetime_rows(evidence: PhysicalMemoryEvidence, *,
                           category_by_semantic_owner: Mapping[str, str]) -> list[dict[str, Any]]:
  """Adapt the existing physical ledger to C7 rows without guessing ownership."""
  if not isinstance(evidence, PhysicalMemoryEvidence):
    raise TypeError("evidence must be PhysicalMemoryEvidence")
  if not evidence.complete or evidence.blockers:
    raise ValueError("physical memory evidence must be complete and blocker-free")
  source_identity = _identity(evidence.to_json())
  rows = []
  for lifetime in evidence.lifetimes:
    # Cross-device mappings are zero-byte aliases of another physical base.
    # Their source allocation remains in the ledger and is the row C7 counts.
    if lifetime.mapped: continue
    owner = lifetime.owner
    semantic = None if owner is None else owner.semantic_owner_id
    if semantic is None or semantic not in category_by_semantic_owner:
      raise ValueError(f"physical allocation {lifetime.allocation_id} has no explicit C7 semantic category")
    if lifetime.physical_nbytes is None or lifetime.free_sequence is None:
      raise ValueError(f"physical allocation {lifetime.allocation_id} has an incomplete exact lifetime")
    rows.append({
      "allocation_id": f"physical:{lifetime.device}:{lifetime.allocation_id}",
      "physical_base_identity": f"{lifetime.device}:base:{lifetime.physical_base_id}",
      "category": category_by_semantic_owner[semantic],
      "requested_bytes": lifetime.requested_nbytes,
      "physical_bytes": lifetime.physical_nbytes,
      "live_from": lifetime.alloc_sequence,
      "live_until": lifetime.free_sequence,
      "provenance": f"physical_memory_ledger:{semantic}",
      "source": "physical_memory_ledger",
      "source_evidence_identity": source_identity,
    })
  return rows


def _lifetime(row: Any, label: str, *, route_start: int, route_end: int) -> dict[str, Any]:
  if not isinstance(row, Mapping): raise ValueError(f"{label} must be a mapping")
  keys = {"allocation_id", "physical_base_identity", "category", "requested_bytes", "physical_bytes",
          "live_from", "live_until", "provenance", "source", "source_evidence_identity"}
  _exact_keys(row, keys, label)
  allocation_id = _nonempty(row["allocation_id"], f"{label}.allocation_id")
  category = _nonempty(row["category"], f"{label}.category")
  if category == "dense_fp16_weight" or category not in _ALLOWED_CATEGORIES:
    raise ValueError(f"{label}.category is not admitted by the staged C7 contract")
  requested = _integer(row["requested_bytes"], f"{label}.requested_bytes")
  physical = _integer(row["physical_bytes"], f"{label}.physical_bytes")
  if physical < requested: raise ValueError(f"{label}.physical_bytes must cover requested_bytes")
  physical_base = row["physical_base_identity"]
  if physical:
    physical_base = _nonempty(physical_base, f"{label}.physical_base_identity")
  elif physical_base is not None:
    raise ValueError(f"{label}.physical_base_identity must be null for explicit zero storage")
  start = _integer(row["live_from"], f"{label}.live_from")
  end = _integer(row["live_until"], f"{label}.live_until")
  if start < route_start or end > route_end or start >= end:
    raise ValueError(f"{label} lifetime must be a non-empty subset of the route window")
  source = _nonempty(row["source"], f"{label}.source")
  if source not in _ROW_SOURCES: raise ValueError(f"{label}.source is not an admitted census authority")
  if physical == 0 and source != "explicit_zero_measurement":
    raise ValueError(f"{label} zero-byte rows require explicit_zero_measurement provenance")
  return {
    "allocation_id": allocation_id, "physical_base_identity": physical_base,
    "category": category, "requested_bytes": requested,
    "physical_bytes": physical, "live_from": start, "live_until": end,
    "provenance": _nonempty(row["provenance"], f"{label}.provenance"), "source": source,
    "source_evidence_identity": _digest(row["source_evidence_identity"], f"{label}.source_evidence_identity"),
  }


def _memory_authority(value: Any, label: str) -> dict[str, Any]:
  if not isinstance(value, Mapping): raise ValueError(f"{label} must be a mapping")
  keys = {"device_identity", "software_identity", "allocator_identity",
          "allocation_granularity_bytes", "budget_identity"}
  _exact_keys(value, keys, label)
  return {
    "device_identity": _nonempty(value["device_identity"], f"{label}.device_identity"),
    "software_identity": _nonempty(value["software_identity"], f"{label}.software_identity"),
    "allocator_identity": _nonempty(value["allocator_identity"], f"{label}.allocator_identity"),
    "allocation_granularity_bytes":
      _integer(value["allocation_granularity_bytes"], f"{label}.allocation_granularity_bytes", positive=True),
    "budget_identity": _digest(value["budget_identity"], f"{label}.budget_identity"),
  }


def staged_c7_budget_identity(*, device_identity: str, software_identity: str, allocator_identity: str,
                              allocation_granularity_bytes: int, admitted_budget_bytes: int,
                              budget_provenance: str) -> str:
  """Bind the admitted byte limit to its device/software/allocator authority."""
  payload = {
    "device_identity": _nonempty(device_identity, "device_identity"),
    "software_identity": _nonempty(software_identity, "software_identity"),
    "allocator_identity": _nonempty(allocator_identity, "allocator_identity"),
    "allocation_granularity_bytes":
      _integer(allocation_granularity_bytes, "allocation_granularity_bytes", positive=True),
    "admitted_budget_bytes": _integer(admitted_budget_bytes, "admitted_budget_bytes", positive=True),
    "budget_provenance": _nonempty(budget_provenance, "budget_provenance"),
  }
  return _identity(payload)


def staged_c7_census_identity(*, authority: Mapping[str, Any], route_start: int, route_end: int,
                              lifetimes: Sequence[Mapping[str, Any]]) -> str:
  """Content identity callers must record beside a complete raw allocation census."""
  normalized_authority = _memory_authority(authority, "authority")
  start, end = _integer(route_start, "route_start"), _integer(route_end, "route_end", positive=True)
  if start >= end: raise ValueError("route window must be non-empty")
  rows = [_lifetime(row, f"lifetimes[{index}]", route_start=start, route_end=end)
          for index, row in enumerate(lifetimes)]
  return _identity({"authority": normalized_authority, "route_start": start, "route_end": end, "lifetimes": rows})


def _queue_memory(queue: str, value: Any, requirements: Mapping[str, int],
                  admitted_budget_bytes: int, admitted_authority: Mapping[str, Any]) -> dict[str, Any]:
  if not isinstance(value, Mapping): raise ValueError(f"{queue} memory observation must be a mapping")
  _exact_keys(value, {
    "route_start", "route_end", "allocation_census_identity", "allocation_census_complete",
    "dense_fp16_weight_materialization", "authority", "lifetimes",
  }, f"{queue} memory observation")
  start = _integer(value["route_start"], f"{queue}.route_start")
  end = _integer(value["route_end"], f"{queue}.route_end", positive=True)
  if start >= end: raise ValueError(f"{queue} route window must be non-empty")
  if type(value["allocation_census_complete"]) is not bool:
    raise ValueError(f"{queue}.allocation_census_complete must be a bool")
  if type(value["dense_fp16_weight_materialization"]) is not bool:
    raise ValueError(f"{queue}.dense_fp16_weight_materialization must be a bool")
  authority = _memory_authority(value["authority"], f"{queue}.authority")
  if authority != admitted_authority:
    raise ValueError(f"{queue} allocation census authority differs from the admitted memory authority")
  raw_rows = value["lifetimes"]
  if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
    raise ValueError(f"{queue}.lifetimes must be a sequence")
  rows = [_lifetime(row, f"{queue}.lifetimes[{index}]", route_start=start, route_end=end)
          for index, row in enumerate(raw_rows)]
  if len({row["allocation_id"] for row in rows}) != len(rows):
    raise ValueError(f"{queue} allocation_id values must be unique")
  positive_bases = [row["physical_base_identity"] for row in rows if row["physical_bytes"]]
  if len(set(positive_bases)) != len(positive_bases):
    raise ValueError(f"{queue} physical base identities must be unique; duplicate/alias accounting is forbidden")
  granularity = authority["allocation_granularity_bytes"]
  if any(row["physical_bytes"] % granularity for row in rows):
    raise ValueError(f"{queue} physical bytes must be aligned to the admitted allocator granularity")
  census_identity = _digest(value["allocation_census_identity"], f"{queue}.allocation_census_identity")
  expected_census = _identity({
    "authority": authority, "route_start": start, "route_end": end, "lifetimes": rows,
  })
  if census_identity != expected_census:
    raise ValueError(f"{queue} allocation census content identity differs")

  category_requested = {name: sum(row["requested_bytes"] for row in rows if row["category"] == name)
                        for name in sorted(_ALLOWED_CATEGORIES)}
  category_physical = {name: sum(row["physical_bytes"] for row in rows if row["category"] == name)
                       for name in sorted(_ALLOWED_CATEGORIES)}
  blockers, failures = [], []
  if not value["allocation_census_complete"]:
    blockers.append("allocation census is incomplete")
  if value["dense_fp16_weight_materialization"]:
    failures.append("dense FP16 weight materialization was observed")
  missing = [name for name in _REQUIRED_CATEGORIES if not any(row["category"] == name for row in rows)]
  if missing: blockers.append("missing exact C7 categories: " + ", ".join(missing))
  for category, expected in requirements.items():
    if category_requested.get(category) != expected:
      blockers.append(
        f"{category} requested bytes {category_requested.get(category, 0)} differ from manifest {expected}")
  for category in _FULL_WINDOW_CATEGORIES:
    if not any(row["category"] == category and row["live_from"] == start and row["live_until"] == end for row in rows):
      blockers.append(f"{category} has no allocation covering the complete route window")
  for category in _INFRASTRUCTURE_CATEGORIES:
    if category_physical.get(category, 0) <= 0:
      blockers.append(f"{category} has no positive measured physical bytes")

  points = sorted({start, *(row["live_from"] for row in rows), *(row["live_until"] for row in rows if row["live_until"] < end)})
  overlap = []
  for point in points:
    live = [row for row in rows if row["live_from"] <= point < row["live_until"]]
    overlap.append({"event": point, "physical_bytes": sum(row["physical_bytes"] for row in live),
                    "allocation_ids": sorted(row["allocation_id"] for row in live)})
  peak = max((row["physical_bytes"] for row in overlap), default=0)
  admitted = not blockers and not failures and peak <= admitted_budget_bytes
  if peak > admitted_budget_bytes:
    failures.append(f"peak physical bytes {peak} exceed admitted budget {admitted_budget_bytes} by "
                    f"{peak-admitted_budget_bytes}")
  status = "BLOCKED" if blockers else "FAIL" if failures else "PASS"
  return {
    "queue_mode": queue, "status": status, "admitted": admitted,
    "allocation_census_identity": census_identity, "authority": authority,
    "allocation_census_complete": value["allocation_census_complete"],
    "dense_fp16_weight_materialization": value["dense_fp16_weight_materialization"],
    "route_window": {"start": start, "end_exclusive": end},
    "lifetimes": rows, "requested_bytes_by_category": category_requested,
    "physical_bytes_by_category": category_physical,
    "peak_physical_bytes": peak, "peak_overlap": overlap, "blockers": blockers, "failures": failures,
  }


def build_staged_c7_memory_ledger(*, family: FrozenStagedFamily,
                                  queue_observations: Mapping[str, Any],
                                  admitted_budget_bytes: int,
                                  budget_provenance: str,
                                  memory_authority: Mapping[str, Any]) -> dict[str, Any]:
  """Build a content-addressed, per-queue C7 route-memory admission ledger."""
  _, family_identity, epochs, requirements = _family(family)
  budget = _integer(admitted_budget_bytes, "admitted_budget_bytes", positive=True)
  provenance = _nonempty(budget_provenance, "budget_provenance")
  authority = _memory_authority(memory_authority, "memory_authority")
  expected_budget_identity = staged_c7_budget_identity(
    device_identity=authority["device_identity"], software_identity=authority["software_identity"],
    allocator_identity=authority["allocator_identity"],
    allocation_granularity_bytes=authority["allocation_granularity_bytes"],
    admitted_budget_bytes=budget, budget_provenance=provenance)
  if authority["budget_identity"] != expected_budget_identity:
    raise ValueError("memory authority budget identity differs from the admitted byte limit")
  if not isinstance(queue_observations, Mapping) or set(queue_observations) != set(QUEUE_MODES):
    raise ValueError(f"queue_observations must contain exactly {QUEUE_MODES}")
  queues = {queue: _queue_memory(queue, queue_observations[queue], requirements, budget, authority)
            for queue in QUEUE_MODES}
  status = "BLOCKED" if any(row["status"] == "BLOCKED" for row in queues.values()) else \
    "FAIL" if any(row["status"] == "FAIL" for row in queues.values()) else "PASS"
  payload = {
    "schema": C7_SCHEMA, "family_identity": family_identity, "epoch_count": epochs,
    "dense_fp16_weight_materialization":
      any(row["dense_fp16_weight_materialization"] for row in queues.values()),
    "logical_requirements": staged_logical_memory_requirements(family),
    "budget": {"admitted_bytes": budget, "provenance": provenance, "authority": authority},
    "queues": queues, "status": status, "production_dispatch_changed": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def validate_staged_c7_memory_ledger(bundle: Any, *, family: FrozenStagedFamily) -> dict[str, Any]:
  """Deeply recompute C7 logical requirements, queue ledgers, overlap, and admission."""
  if not isinstance(bundle, Mapping): raise ValueError("C7 memory ledger must be a mapping")
  _exact_keys(bundle, {
    "schema", "family_identity", "epoch_count", "dense_fp16_weight_materialization",
    "logical_requirements", "budget", "queues", "status", "production_dispatch_changed",
    "evidence_identity",
  }, "C7 memory ledger")
  if bundle.get("schema") != C7_SCHEMA or bundle.get("production_dispatch_changed") is not False:
    raise ValueError("C7 memory ledger schema or production state differs")
  budget, queues = bundle.get("budget"), bundle.get("queues")
  if not isinstance(budget, Mapping) or not isinstance(queues, Mapping) or set(queues) != set(QUEUE_MODES):
    raise ValueError("C7 budget or queue ledger is incomplete")
  _exact_keys(budget, {"admitted_bytes", "provenance", "authority"}, "C7 budget")
  observations = {}
  for queue in QUEUE_MODES:
    row = queues[queue]
    if not isinstance(row, Mapping): raise ValueError(f"C7 {queue} queue ledger must be a mapping")
    window = row.get("route_window")
    if not isinstance(window, Mapping) or set(window) != {"start", "end_exclusive"}:
      raise ValueError(f"C7 {queue} route window is malformed")
    observations[queue] = {
      "route_start": window["start"], "route_end": window["end_exclusive"],
      "allocation_census_identity": row.get("allocation_census_identity"),
      "allocation_census_complete": row.get("allocation_census_complete"),
      "dense_fp16_weight_materialization": row.get("dense_fp16_weight_materialization"),
      "authority": row.get("authority"), "lifetimes": row.get("lifetimes"),
    }
  rebuilt = build_staged_c7_memory_ledger(
    family=family, queue_observations=observations,
    admitted_budget_bytes=budget.get("admitted_bytes"), budget_provenance=budget.get("provenance"),
    memory_authority=budget.get("authority"),
  )
  if dict(bundle) != rebuilt:
    raise ValueError("C7 memory ledger differs from recomputed logical, physical, or admission evidence")
  return rebuilt


def _candidate_round(value: Any, *, queue: str, round_index: int, epochs: int,
                     compact_requirements: Mapping[str, int]) -> dict[str, Any]:
  label = f"{queue}.candidate_rounds[{round_index}]"
  if not isinstance(value, Mapping): raise ValueError(f"{label} must be a mapping")
  _exact_keys(value, {"output_initialization_ms", "epochs", "final_sync_ms", "complete_role_ms"}, label)
  epoch_rows = value["epochs"]
  if not isinstance(epoch_rows, Sequence) or isinstance(epoch_rows, (str, bytes)) or len(epoch_rows) != epochs:
    raise ValueError(f"{label}.epochs must enumerate all {epochs} dispatches")
  normalized_epochs, phase_sum = [], 0.0
  epoch_keys = {"ordinal", "gather_ms", "transfer_ms", "staging_sync_ms", "dispatch_ms",
                "dispatch_sync_ms", "staged_bytes", "staged_components", "copy_count"}
  compact_stage_bytes = sum(compact_requirements.values())
  for ordinal, epoch in enumerate(epoch_rows):
    epoch_label = f"{label}.epochs[{ordinal}]"
    if not isinstance(epoch, Mapping): raise ValueError(f"{epoch_label} must be a mapping")
    _exact_keys(epoch, epoch_keys, epoch_label)
    if epoch["ordinal"] != ordinal: raise ValueError(f"{epoch_label}.ordinal differs from the complete epoch order")
    if _integer(epoch["staged_bytes"], f"{epoch_label}.staged_bytes") != compact_stage_bytes:
      raise ValueError(f"{epoch_label}.staged_bytes differs from the complete compact stage")
    if not isinstance(epoch["staged_components"], Mapping) or dict(epoch["staged_components"]) != compact_requirements:
      raise ValueError(f"{epoch_label}.staged_components differ from all four compact inputs")
    if _integer(epoch["copy_count"], f"{epoch_label}.copy_count", positive=True) != 4:
      raise ValueError(f"{epoch_label}.copy_count must cover all four staged inputs")
    normalized = {"ordinal": ordinal, "staged_bytes": compact_stage_bytes,
                  "staged_components": dict(compact_requirements), "copy_count": 4}
    for field in ("gather_ms", "transfer_ms", "staging_sync_ms", "dispatch_ms", "dispatch_sync_ms"):
      normalized[field] = _number(epoch[field], f"{epoch_label}.{field}")
      phase_sum += normalized[field]
    normalized_epochs.append(normalized)
  initialization = _number(value["output_initialization_ms"], f"{label}.output_initialization_ms")
  final_sync = _number(value["final_sync_ms"], f"{label}.final_sync_ms")
  complete = _positive_number(value["complete_role_ms"], f"{label}.complete_role_ms")
  expected = initialization + phase_sum + final_sync
  if not math.isclose(complete, expected, rel_tol=1e-12, abs_tol=1e-9):
    raise ValueError(f"{label}.complete_role_ms differs from the exhaustive phase sum")
  return {"output_initialization_ms": initialization, "epochs": normalized_epochs,
          "final_sync_ms": final_sync, "complete_role_ms": complete}


def _fallback_round(value: Any, *, queue: str, round_index: int) -> dict[str, float]:
  label = f"{queue}.fallback_rounds[{round_index}]"
  if not isinstance(value, Mapping): raise ValueError(f"{label} must be a mapping")
  _exact_keys(value, {"complete_role_ms"}, label)
  return {"complete_role_ms": _positive_number(value["complete_role_ms"], f"{label}.complete_role_ms")}


def _evidence(value: Any, *, label: str, schema: str, keys: set[str]) -> dict[str, Any]:
  if not isinstance(value, Mapping): raise ValueError(f"{label} must be a mapping")
  _exact_keys(value, keys | {"evidence_identity"}, label)
  if value.get("schema") != schema: raise ValueError(f"{label}.schema differs")
  payload = {key: item for key, item in value.items() if key != "evidence_identity"}
  if value.get("evidence_identity") != _identity(payload):
    raise ValueError(f"{label} content identity differs")
  return dict(value)


def _c6_evidence(value: Any, *, family_manifest: Mapping[str, Any], family_identity: str) -> dict[str, Any]:
  keys = {
    "schema", "status", "family_identity", "candidate_executable_identity", "candidate_binary_sha256",
    "workload_identity", "input_identity", "device_identity", "software_identity",
    "queue_correctness", "queue_comparators",
  }
  row = _evidence(value, label="C6 correctness evidence",
                  schema="tinygrad.mmq_q4k_q8_1.staged_c6_correctness_binding.v1", keys=keys)
  if row["status"] != "PASS" or row["family_identity"] != family_identity:
    raise ValueError("C6 correctness evidence did not pass for the exact staged family")
  _digest(row["candidate_executable_identity"], "C6 candidate_executable_identity")
  binary = row["candidate_binary_sha256"]
  if not isinstance(binary, str) or len(binary) != 64 or any(char not in _HEX for char in binary) or \
     binary != family_manifest["program"]["binary_sha256"]:
    raise ValueError("C6 candidate binary differs from the frozen staged PROGRAM")
  for field in ("workload_identity", "input_identity", "device_identity", "software_identity"):
    _nonempty(row[field], f"C6 {field}")
  queue_correctness = row["queue_correctness"]
  if not isinstance(queue_correctness, Mapping) or set(queue_correctness) != set(QUEUE_MODES):
    raise ValueError(f"C6 queue correctness must contain exactly {QUEUE_MODES}")
  for queue in QUEUE_MODES: _digest(queue_correctness[queue], f"C6 {queue} correctness identity")
  queue_comparators = row["queue_comparators"]
  if not isinstance(queue_comparators, Mapping) or set(queue_comparators) != set(QUEUE_MODES):
    raise ValueError(f"C6 queue comparators must contain exactly {QUEUE_MODES}")
  for queue in QUEUE_MODES: _digest(queue_comparators[queue], f"C6 {queue} comparator identity")
  return row


def _fallback_evidence(value: Any, *, queue: str) -> dict[str, Any]:
  keys = {
    "schema", "status", "route_id", "queue_mode", "artifact_schema", "artifact_identity",
    "binary_sha256", "executable_identity", "comparator_identity", "workload_identity",
    "input_identity", "device_identity",
    "software_identity", "clock_identity",
  }
  row = _evidence(value, label=f"{queue} direct_packed fallback evidence",
                  schema="tinygrad.direct_packed.complete_role_fallback.v1", keys=keys)
  if row["status"] != "PASS" or row["route_id"] != DIRECT_FALLBACK or row["queue_mode"] != queue:
    raise ValueError(f"{queue} fallback evidence is not an exact passing direct_packed artifact")
  _nonempty(row["artifact_schema"], f"{queue} fallback artifact_schema")
  binary = row["binary_sha256"]
  if not isinstance(binary, str) or len(binary) != 64 or any(char not in _HEX for char in binary):
    raise ValueError(f"{queue} fallback binary_sha256 must be a lowercase hexadecimal digest")
  for field in ("artifact_identity", "executable_identity", "comparator_identity"):
    _digest(row[field], f"{queue} fallback {field}")
  for field in ("workload_identity", "input_identity", "device_identity", "software_identity", "clock_identity"):
    _nonempty(row[field], f"{queue} fallback {field}")
  artifact_payload = {
    "artifact_schema": row["artifact_schema"], "binary_sha256": binary,
    "executable_identity": row["executable_identity"], "comparator_identity": row["comparator_identity"],
    "queue_mode": queue, "workload_identity": row["workload_identity"],
  }
  if row["artifact_identity"] != _identity(artifact_payload):
    raise ValueError(f"{queue} fallback artifact identity differs from its exact executable content")
  return row


def staged_c8_randomized_orders(*, seed: int, round_count: int) -> list[list[str]]:
  """Return the required balanced, seeded pair-order schedule."""
  seed = _integer(seed, "seed")
  rounds = _integer(round_count, "round_count", positive=True)
  if rounds < 10: raise ValueError("round_count must be at least ten")
  orders = [
    ["staged_candidate", DIRECT_FALLBACK] if index % 2 == 0
    else [DIRECT_FALLBACK, "staged_candidate"] for index in range(rounds)
  ]
  random.Random(seed).shuffle(orders)
  return orders


def _queue_timing(queue: str, value: Any, *, family_identity: str, epochs: int,
                  compact_requirements: Mapping[str, int], required_speedup: float,
                  c6: Mapping[str, Any], memory_authority: Mapping[str, Any]) -> dict[str, Any]:
  if not isinstance(value, Mapping): raise ValueError(f"{queue} timing observation must be a mapping")
  keys = {
    "family_identity", "candidate_executable_identity", "candidate_c6_queue_evidence_identity",
    "candidate_comparator_identity",
    "fallback_evidence", "session_identity",
    "workload_identity", "input_identity", "device_identity", "software_identity", "clock_identity",
    "measurement_source", "candidate_warmups", "fallback_warmups", "randomization_seed",
    "randomization_algorithm", "randomized_order_identity", "paired_rounds",
  }
  _exact_keys(value, keys, f"{queue} timing observation")
  if value["family_identity"] != family_identity: raise ValueError(f"{queue} timing family identity differs")
  candidate_executable = _digest(value["candidate_executable_identity"], f"{queue}.candidate_executable_identity")
  if candidate_executable != c6["candidate_executable_identity"]:
    raise ValueError(f"{queue} timing candidate executable differs from C6")
  c6_queue = _digest(value["candidate_c6_queue_evidence_identity"],
                     f"{queue}.candidate_c6_queue_evidence_identity")
  if c6_queue != c6["queue_correctness"][queue]:
    raise ValueError(f"{queue} timing correctness identity differs from C6")
  candidate_comparator = _digest(value["candidate_comparator_identity"],
                                 f"{queue}.candidate_comparator_identity")
  if candidate_comparator != c6["queue_comparators"][queue]:
    raise ValueError(f"{queue} timing comparator identity differs from C6")
  identities = {field: _nonempty(value[field], f"{queue}.{field}") for field in (
    "session_identity", "workload_identity", "input_identity",
    "device_identity", "software_identity", "clock_identity")}
  for field in ("workload_identity", "input_identity", "device_identity", "software_identity"):
    if identities[field] != c6[field]: raise ValueError(f"{queue} timing {field} differs from C6")
  for field in ("device_identity", "software_identity"):
    if identities[field] != memory_authority[field]:
      raise ValueError(f"{queue} timing {field} differs from C7 memory authority")
  fallback_evidence = _fallback_evidence(value["fallback_evidence"], queue=queue)
  if fallback_evidence["comparator_identity"] != candidate_comparator:
    raise ValueError(f"{queue} fallback comparator identity differs from candidate/C6")
  for field in ("workload_identity", "input_identity", "device_identity", "software_identity", "clock_identity"):
    if fallback_evidence[field] != identities[field]:
      raise ValueError(f"{queue} fallback {field} differs from the matched timing execution")
  if value["measurement_source"] != "synchronized_wall":
    raise ValueError(f"{queue}.measurement_source must be synchronized_wall")
  candidate_warmups = _integer(value["candidate_warmups"], f"{queue}.candidate_warmups", positive=True)
  fallback_warmups = _integer(value["fallback_warmups"], f"{queue}.fallback_warmups", positive=True)
  if candidate_warmups != fallback_warmups or candidate_warmups < 3:
    raise ValueError(f"{queue} requires equal candidate/fallback warmups of at least three")
  seed = _integer(value["randomization_seed"], f"{queue}.randomization_seed")
  if value["randomization_algorithm"] != "python_random_v1_balanced_pair_order":
    raise ValueError(f"{queue}.randomization_algorithm differs")
  paired_raw = value["paired_rounds"]
  if not isinstance(paired_raw, Sequence) or isinstance(paired_raw, (str, bytes)) or len(paired_raw) < 10:
    raise ValueError(f"{queue}.paired_rounds requires at least ten promotion-grade pairs")
  pairs, orders = [], []
  for index, pair in enumerate(paired_raw):
    label = f"{queue}.paired_rounds[{index}]"
    if not isinstance(pair, Mapping): raise ValueError(f"{label} must be a mapping")
    _exact_keys(pair, {"pair_index", "order", "candidate", "fallback"}, label)
    if pair["pair_index"] != index: raise ValueError(f"{label}.pair_index differs from paired order")
    order = pair["order"]
    if not isinstance(order, Sequence) or isinstance(order, (str, bytes)) or \
       tuple(order) not in (("staged_candidate", DIRECT_FALLBACK), (DIRECT_FALLBACK, "staged_candidate")):
      raise ValueError(f"{label}.order must contain each route exactly once")
    orders.append(list(order))
    pairs.append({
      "pair_index": index, "order": list(order),
      "candidate": _candidate_round(pair["candidate"], queue=queue, round_index=index, epochs=epochs,
                                    compact_requirements=compact_requirements),
      "fallback": _fallback_round(pair["fallback"], queue=queue, round_index=index),
    })
  if orders != staged_c8_randomized_orders(seed=seed, round_count=len(pairs)):
    raise ValueError(f"{queue} paired order differs from the seeded balanced randomization")
  order_identity = _digest(value["randomized_order_identity"], f"{queue}.randomized_order_identity")
  if order_identity != _identity({"seed": seed, "orders": orders}):
    raise ValueError(f"{queue} randomized order content identity differs")
  candidate = [pair["candidate"] for pair in pairs]
  fallback = [pair["fallback"] for pair in pairs]
  candidate_median = statistics.median(row["complete_role_ms"] for row in candidate)
  fallback_median = statistics.median(row["complete_role_ms"] for row in fallback)
  speedup = fallback_median / candidate_median
  candidate_wins = candidate_median < fallback_median and speedup >= required_speedup
  return {
    "queue_mode": queue, "matched_session": True, **identities,
    "candidate_executable_identity": candidate_executable,
    "candidate_c6_queue_evidence_identity": c6_queue,
    "candidate_comparator_identity": candidate_comparator,
    "fallback_evidence": fallback_evidence,
    "measurement_source": "synchronized_wall",
    "warmups": {"staged_candidate": candidate_warmups, DIRECT_FALLBACK: fallback_warmups},
    "round_count": len(pairs), "randomization_seed": seed,
    "randomization_algorithm": "python_random_v1_balanced_pair_order",
    "randomized_order_identity": order_identity, "paired_rounds": pairs,
    "statistics": {"statistic": "median", "candidate_complete_role_ms": candidate_median,
                   "fallback_complete_role_ms": fallback_median, "speedup": speedup},
    "winner": "staged_candidate" if candidate_wins else DIRECT_FALLBACK,
  }


def build_staged_c8_timing_contract(*, family: FrozenStagedFamily,
                                    c7_memory_ledger: Mapping[str, Any],
                                    c6_correctness_evidence: Mapping[str, Any],
                                    queue_observations: Mapping[str, Any],
                                    required_speedup: int | float = 1.0) -> dict[str, Any]:
  """Build matched PM4/AQL complete-role C8 timing and an explicit route decision."""
  family_manifest, family_identity, epochs, requirements = _family(family)
  c7 = validate_staged_c7_memory_ledger(c7_memory_ledger, family=family)
  if c7.get("status") != "PASS" or c7.get("dense_fp16_weight_materialization") is not False or \
     any(row.get("status") != "PASS" or row.get("admitted") is not True or
         row.get("peak_physical_bytes", math.inf) > c7["budget"]["admitted_bytes"]
         for row in c7["queues"].values()):
    raise ValueError("C8 requires a passing C7 memory admission")
  c6 = _c6_evidence(c6_correctness_evidence, family_manifest=family_manifest,
                    family_identity=family_identity)
  threshold = _number(required_speedup, "required_speedup")
  if threshold < 1.0: raise ValueError("required_speedup must be at least 1.0")
  if not isinstance(queue_observations, Mapping) or set(queue_observations) != set(QUEUE_MODES):
    raise ValueError(f"queue_observations must contain exactly {QUEUE_MODES}")
  compact_requirements = {name: nbytes for name, nbytes in requirements.items() if name.startswith("compact_")}
  compact_stage_bytes = sum(compact_requirements.values())
  queues = {queue: _queue_timing(queue, queue_observations[queue], family_identity=family_identity,
                                  epochs=epochs, compact_requirements=compact_requirements,
                                  required_speedup=threshold, c6=c6,
                                  memory_authority=c7["budget"]["authority"])
            for queue in QUEUE_MODES}
  all_win = all(row["winner"] == "staged_candidate" for row in queues.values())
  decision = {
    "status": "CERTIFIED_WIN" if all_win else "CERTIFIED_FALLBACK",
    "selected_route": "staged_candidate" if all_win else DIRECT_FALLBACK,
    "fallback_route": DIRECT_FALLBACK,
    "rule": "staged candidate must beat the exact fallback by the required speedup in both PM4 and AQL",
    "required_speedup": threshold,
  }
  payload = {
    "schema": C8_SCHEMA, "family_identity": family_identity, "epoch_count": epochs,
    "c7_evidence_identity": c7["evidence_identity"],
    "c6_evidence_identity": c6["evidence_identity"],
    "complete_role_definition": {
      "candidate": "output initialization + every epoch gather + four staged transfers + staging sync + "
                   "dispatch + dispatch sync + final sync",
      "fallback": "exact direct_packed route over the same synchronized-wall boundary",
      "compact_stage_bytes_per_epoch": compact_stage_bytes,
    },
    "queues": queues, "decision": decision, "production_dispatch_changed": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def validate_staged_certification_bundle(bundle: Any) -> dict[str, Any]:
  """Validate immutable envelope and content identity for either C7 or C8."""
  if not isinstance(bundle, Mapping): raise ValueError("certification bundle must be a mapping")
  schema = bundle.get("schema")
  if schema not in (C7_SCHEMA, C8_SCHEMA): raise ValueError("certification bundle schema is unsupported")
  if bundle.get("production_dispatch_changed") is not False:
    raise ValueError("production_dispatch_changed must be False")
  if set(bundle.get("queues", ())) != set(QUEUE_MODES):
    raise ValueError(f"certification bundle must contain exactly {QUEUE_MODES}")
  identity = bundle.get("evidence_identity")
  payload = {key: value for key, value in bundle.items() if key != "evidence_identity"}
  if identity != _identity(payload): raise ValueError("certification bundle content identity differs")
  return dict(bundle)


__all__ = [
  "C7_SCHEMA", "C8_SCHEMA", "DIRECT_FALLBACK", "build_staged_c7_memory_ledger",
  "build_staged_c8_timing_contract", "physical_lifetime_rows",
  "staged_c7_budget_identity", "staged_c7_census_identity", "staged_c8_randomized_orders",
  "staged_logical_memory_requirements",
  "validate_staged_c7_memory_ledger", "validate_staged_certification_bundle",
]
