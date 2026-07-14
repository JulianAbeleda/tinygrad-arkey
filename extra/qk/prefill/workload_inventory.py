"""Model-agnostic packed-prefill inventory and canonical candidate-set generation.

Tensor facts remain authoritative for identity, quant, role and shape.  Candidate
construction/admission is delegated to runtime_specs, including packed geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from tinygrad.codegen.opt.packed_weight import PackedWeightTransform
from tinygrad.llm.model_facts import ModelFacts, TensorFact, model_facts_from_gguf_metadata
from extra.qk.runtime_specs import (FullKernelCandidateSet, admit_full_kernel_candidate_set,
  derive_packed_weight_candidate, rebind_full_kernel_workload)

INVENTORY_SCHEMA = "qk.packed_prefill_workload_inventory.v1"
CANDIDATE_INVENTORY_SCHEMA = "qk.packed_prefill_candidate_inventory.v1"
SUPPORTED_FORMATS = ("Q4_K", "Q6_K")


@dataclass(frozen=True, order=True)
class MeasuredRow:
  role: str
  quant_format: str
  m: int
  n: int
  k: int

  @property
  def key(self) -> tuple[str, str, int, int, int]: return (self.role, self.quant_format, self.m, self.n, self.k)


def _measured_row(row: MeasuredRow | dict[str, Any]) -> MeasuredRow:
  if isinstance(row, MeasuredRow): return row
  try: out = MeasuredRow(row["role"], row["quant_format"], *(int(row[x]) for x in ("m", "n", "k")))
  except (KeyError, TypeError, ValueError) as exc: raise ValueError("malformed measured row") from exc
  if not isinstance(out.role, str) or not out.role or any(x <= 0 for x in (out.m, out.n, out.k)):
    raise ValueError("malformed measured row")
  return out


def build_workload_inventory(facts: ModelFacts, measured_rows: Iterable[MeasuredRow | dict[str, Any]], *,
                             profile: str, memory_lifetime: str = "model_resident") -> dict[str, Any]:
  """Reconcile measured rows with exact tensor facts, failing closed on ambiguity."""
  measured = tuple(_measured_row(x) for x in measured_rows)
  if len({x.key for x in measured}) != len(measured): raise ValueError("duplicate measured workload mapping")
  unsupported = sorted({x.quant_format for x in measured if x.quant_format not in SUPPORTED_FORMATS})
  if unsupported: raise ValueError(f"unsupported packed formats {unsupported!r}")
  if not profile or not memory_lifetime: raise ValueError("profile and memory_lifetime must be non-empty")

  grouped: dict[tuple[str, str, int, int], list[TensorFact]] = {}
  for tensor in facts.tensors:
    if tensor.role is None: continue
    grouped.setdefault((tensor.role, tensor.quant_label, tensor.rows, tensor.cols), []).append(tensor)

  measured_roles = {x.role for x in measured}
  expected_fact_keys = {(x.role, x.quant_format, x.n, x.k) for x in measured}
  discovered_fact_keys = {key for key in grouped if key[0] in measured_roles and key[1] in SUPPORTED_FORMATS}
  unknown = discovered_fact_keys - expected_fact_keys
  if unknown: raise ValueError(f"unknown packed tensor mappings {sorted(unknown)!r}")

  rows = []
  for expected in measured:
    tensors = grouped.get((expected.role, expected.quant_format, expected.n, expected.k), [])
    if not tensors: raise ValueError(f"unknown measured workload mapping {expected.key!r}")
    transform = PackedWeightTransform(expected.quant_format, expected.n, expected.k)
    names = sorted(t.name for t in tensors)
    if len(names) != len(set(names)): raise ValueError(f"duplicate tensor mapping for {expected.key!r}")
    rows.append({
      "tensor_identities": names, "quant_format": expected.quant_format, "role": expected.role,
      "shape": {"m": expected.m, "n": expected.n, "k": expected.k},
      "layout": {"logical": "transposed_row_major", "packed": "ggml_k_blocks",
                 "block_elems": transform.block_elems, "block_bytes": transform.block_bytes},
      "call_count": len(tensors), "source_bytes": len(tensors) * transform.packed_bytes,
      "logical_flop": len(tensors) * 2 * expected.m * expected.n * expected.k,
      "memory_lifetime": memory_lifetime,
    })
  return {"schema": INVENTORY_SCHEMA, "profile": profile, "rows": rows}


def inventory_from_gguf_metadata(kv: dict[str, Any], meta: dict[str, Any], measured_rows: Iterable[MeasuredRow | dict[str, Any]],
                                 *, profile: str) -> dict[str, Any]:
  return build_workload_inventory(model_facts_from_gguf_metadata(kv, meta), measured_rows, profile=profile)


def generate_candidate_inventory(inventory: dict[str, Any], templates: dict[str, dict[str, Any]], *,
                                 target: dict[str, Any] | None = None) -> dict[str, Any]:
  """Rebind role templates and create admitted canonical sets, partitioned by quant.

  Partitioning preserves runtime_specs' fail-closed warmstart key authority: mixed
  formats with identical M/N/K are never placed in one warmstart namespace.
  """
  if inventory.get("schema") != INVENTORY_SCHEMA or not isinstance(inventory.get("rows"), list):
    raise ValueError("unsupported workload inventory schema")
  by_quant: dict[str, list[Any]] = {}
  seen: set[tuple[str, str, int, int, int]] = set()
  rows = []
  for row in inventory["rows"]:
    shape = tuple(row["shape"][x] for x in ("m", "n", "k"))
    key = (row["role"], row["quant_format"], *shape)
    if key in seen: raise ValueError(f"duplicate exact inventory key {key!r}")
    seen.add(key)
    if row["quant_format"] not in SUPPORTED_FORMATS: raise ValueError(f"unsupported packed format {row['quant_format']!r}")
    transform = PackedWeightTransform(row["quant_format"], shape[1], shape[2])
    count = len(row.get("tensor_identities", ()))
    if count <= 0 or row.get("call_count") != count or row.get("source_bytes") != count * transform.packed_bytes or \
       row.get("logical_flop") != count * 2 * shape[0] * shape[1] * shape[2]:
      raise ValueError(f"candidate/tensor shape mismatch for {key!r}")
    template = templates.get(row["role"])
    if template is None: raise ValueError(f"unknown schedule-template mapping for role {row['role']!r}")
    rebound = rebind_full_kernel_workload(template, profile=inventory["profile"], role=row["role"], shape=shape, target=target)
    entry = derive_packed_weight_candidate(rebound.to_json()["payload"], row["quant_format"])
    workload = entry.payload["workload"]
    if tuple(workload["shape"][x] for x in ("m", "n", "k")) != shape:
      raise ValueError(f"candidate/tensor shape mismatch for {key!r}")
    by_quant.setdefault(row["quant_format"], []).append(entry)
    rows.append({"inventory_key": list(key), "canonical_identity": entry.canonical_identity})

  candidate_sets = {}
  for quant, entries in sorted(by_quant.items()):
    candidate_set = FullKernelCandidateSet(tuple(entries))
    admit_full_kernel_candidate_set(candidate_set)
    candidate_sets[quant] = candidate_set.to_json()
  return {"schema": CANDIDATE_INVENTORY_SCHEMA, "inventory": inventory,
          "candidate_sets": candidate_sets, "bindings": rows}


def write_canonical_json(artifact: dict[str, Any], path: str | Path) -> None:
  Path(path).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
