"""Model-agnostic packed-prefill inventory and canonical candidate-set generation.

Tensor facts remain authoritative for identity, quant, role and shape.  Candidate
construction/admission is delegated to runtime_specs, including packed geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from tinygrad.codegen.opt.packed_weight import PackedWeightTransform
from tinygrad.llm.model_facts import ModelFacts, TensorFact, model_facts_from_gguf_metadata
from extra.qk.runtime_specs import (FullKernelCandidateSet, admit_full_kernel_candidate_set,
  derive_packed_weight_candidate, derive_q4k_q8_1_five_buffer_candidate, rebind_full_kernel_workload)

INVENTORY_SCHEMA = "qk.packed_prefill_workload_inventory.v1"
CANDIDATE_INVENTORY_SCHEMA = "qk.packed_prefill_candidate_inventory.v1"
SUPPORTED_FORMATS = ("Q4_K", "Q6_K")
RUNTIME_INVENTORY_SCHEMA = "tinygrad.model_runtime_prefill_inventory.v2"


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


def _canonical_inventory_identity(rows: list[dict[str, Any]]) -> str:
  """Digest only exact routed tensor content; labels and paths are provenance."""
  canonical = []
  for row in rows:
    try:
      shape = {x: row["shape"][x] for x in ("m", "n", "k")}
      layout = {x: row["layout"][x] for x in ("logical", "packed", "block_elems", "block_bytes")}
      canonical.append({"role": row["role"], "quant_format": row["quant_format"], "shape": shape,
        "layout": layout, "tensor_identities": sorted(row["tensor_identities"]), "call_count": row["call_count"],
        "source_bytes": row["source_bytes"]})
    except (KeyError, TypeError) as exc: raise ValueError("malformed workload inventory row") from exc
  encoded = json.dumps(sorted(canonical, key=lambda x: (x["role"], x["quant_format"],
    x["shape"]["m"], x["shape"]["n"], x["shape"]["k"])), sort_keys=True, separators=(",", ":"),
    ensure_ascii=True, allow_nan=False).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def _canonical_json_hash(value: Any) -> str:
  return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
    ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _projected_inventory_identity(inventory: dict[str, Any]) -> str:
  return _canonical_json_hash({key:inventory[key] for key in ("parent_inventory_identity", "rows", "fixed_obligations")})


def project_runtime_prefill_inventory(parent: dict[str, Any]) -> dict[str, Any]:
  """Losslessly group the selected-model v2 invocation inventory for candidate construction.

  The parent remains authoritative: exact invocation semantics are retained in each
  group (or fixed obligation), and no ModelFacts/profile/measured-row facts are used.
  """
  if parent.get("schema") != RUNTIME_INVENTORY_SCHEMA or not isinstance(parent.get("rows"), list):
    raise ValueError("unsupported runtime prefill inventory schema")
  parent_identity = parent.get("inventory_identity")
  if not isinstance(parent_identity, str) or not parent_identity: raise ValueError("missing parent inventory identity")
  rows = parent["rows"]
  if parent_identity != _canonical_json_hash(sorted(rows, key=lambda x: x.get("invocation_id", ""))):
    raise ValueError("parent inventory identity mismatch")

  seen_invocations, seen_tensors, seen_sources = set(), set(), set()
  grouped: dict[tuple[str, str, str, int, int, int], list[dict[str, Any]]] = {}
  fixed = []
  for raw in rows:
    if not isinstance(raw, dict): raise ValueError("malformed runtime inventory row")
    row = dict(raw)
    invocation_id, tensor_identity = row.get("invocation_id"), row.get("tensor_identity")
    if not isinstance(invocation_id, str) or not invocation_id or not isinstance(tensor_identity, str) or not tensor_identity:
      raise ValueError("missing invocation/source tensor identity")
    semantic = {key:value for key,value in row.items() if key != "invocation_id"}
    if invocation_id != "invocation:sha256:" + _canonical_json_hash(semantic):
      raise ValueError("invocation identity mismatch")
    source_identity = row.get("source_tensor_identity", tensor_identity)
    if not isinstance(source_identity, str) or not source_identity: raise ValueError("missing invocation/source tensor identity")
    if invocation_id in seen_invocations: raise ValueError("duplicate invocation identity")
    if tensor_identity in seen_tensors: raise ValueError("duplicate tensor identity")
    if source_identity in seen_sources: raise ValueError("duplicate source tensor identity")
    seen_invocations.add(invocation_id); seen_tensors.add(tensor_identity); seen_sources.add(source_identity)
    try:
      phase, role, quant = row.get("phase", "prefill"), row["role"], row["quant_format"]
      shape = tuple(row["shape"][x] for x in ("m", "n", "k"))
    except (KeyError, TypeError): raise ValueError("malformed runtime inventory row") from None
    if not all(isinstance(x, str) and x for x in (phase, role, quant)) or \
       not all(isinstance(x, int) and x > 0 for x in shape): raise ValueError("malformed runtime inventory row")
    controlled = row.get("candidate_controlled")
    if controlled is True:
      if quant not in SUPPORTED_FORMATS: raise ValueError(f"unsupported packed format {quant!r}")
      if "fixed_route_id" in row: raise ValueError("candidate invocation has fixed call semantics")
      grouped.setdefault((phase, role, quant, *shape), []).append(row)
    elif controlled is False:
      if not isinstance(row.get("fixed_route_id"), str) or not row["fixed_route_id"]:
        raise ValueError("fixed invocation lacks fixed call semantics")
      fixed.append(row)
    else: raise ValueError("missing candidate call semantics")

  projected = []
  for (phase, role, quant, m, n, k), invocations in sorted(grouped.items()):
    invocations.sort(key=lambda x: x["invocation_id"])
    transform = PackedWeightTransform(quant, n, k)
    projected.append({"phase":phase, "role":role, "quant_format":quant, "shape":{"m":m, "n":n, "k":k},
      "layout":{"logical":"transposed_row_major", "packed":"ggml_k_blocks",
                "block_elems":transform.block_elems, "block_bytes":transform.block_bytes},
      "tensor_identities":[x["tensor_identity"] for x in invocations],
      "source_tensor_identities":[x.get("source_tensor_identity", x["tensor_identity"]) for x in invocations],
      "invocation_ids":[x["invocation_id"] for x in invocations], "invocations":invocations,
      "call_count":len(invocations), "source_bytes":len(invocations) * transform.packed_bytes,
      "logical_flop":len(invocations) * 2 * m * n * k, "memory_lifetime":"model_resident"})
  fixed.sort(key=lambda x: x["invocation_id"])
  semantic = {"parent_inventory_identity":parent_identity, "rows":projected, "fixed_obligations":fixed}
  return {"schema":INVENTORY_SCHEMA, **semantic, "inventory_identity":_canonical_json_hash(semantic)}


def build_workload_inventory(facts: ModelFacts, measured_rows: Iterable[MeasuredRow | dict[str, Any]], *,
                             profile: str | None = None, model_path: str | None = None,
                             memory_lifetime: str = "model_resident") -> dict[str, Any]:
  """Reconcile measured rows with exact tensor facts, failing closed on ambiguity."""
  measured = tuple(_measured_row(x) for x in measured_rows)
  if len({x.key for x in measured}) != len(measured): raise ValueError("duplicate measured workload mapping")
  unsupported = sorted({x.quant_format for x in measured if x.quant_format not in SUPPORTED_FORMATS})
  if unsupported: raise ValueError(f"unsupported packed formats {unsupported!r}")
  if profile is not None and (not isinstance(profile, str) or not profile): raise ValueError("profile must be non-empty")
  if model_path is not None and (not isinstance(model_path, str) or not model_path): raise ValueError("model_path must be non-empty")
  if not memory_lifetime: raise ValueError("memory_lifetime must be non-empty")

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
  out = {"schema": INVENTORY_SCHEMA, "inventory_identity": _canonical_inventory_identity(rows), "rows": rows}
  provenance = {key:value for key,value in (("profile", profile), ("model_path", model_path)) if value is not None}
  if provenance: out["provenance"] = provenance
  return out


def inventory_from_gguf_metadata(kv: dict[str, Any], meta: dict[str, Any], measured_rows: Iterable[MeasuredRow | dict[str, Any]],
                                 *, profile: str | None = None, model_path: str | None = None) -> dict[str, Any]:
  return build_workload_inventory(model_facts_from_gguf_metadata(kv, meta), measured_rows,
    profile=profile, model_path=model_path)


def generate_candidate_inventory(inventory: dict[str, Any], templates: dict[str, dict[str, Any]], *,
                                 target: dict[str, Any] | None = None) -> dict[str, Any]:
  """Rebind role templates and create admitted canonical sets, partitioned by quant.

  Partitioning preserves runtime_specs' fail-closed warmstart key authority: mixed
  formats with identical M/N/K are never placed in one warmstart namespace.
  """
  if inventory.get("schema") != INVENTORY_SCHEMA or not isinstance(inventory.get("rows"), list):
    raise ValueError("unsupported workload inventory schema")
  inventory_identity = _projected_inventory_identity(inventory) if "parent_inventory_identity" in inventory else \
    _canonical_inventory_identity(inventory["rows"])
  # v1 artifacts had only a profile. They remain readable, but all newly generated
  # bindings derive their semantic namespace from exact inventory content.
  recorded_identity = inventory.get("inventory_identity")
  if recorded_identity is not None and recorded_identity != inventory_identity:
    raise ValueError("workload inventory identity mismatch")
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
    rebound = rebind_full_kernel_workload(template, profile=inventory_identity, role=row["role"], shape=shape, target=target)
    entry = derive_q4k_q8_1_five_buffer_candidate(rebound.to_json()["payload"]) if row["quant_format"] == "Q4_K" else \
      derive_packed_weight_candidate(rebound.to_json()["payload"], row["quant_format"])
    workload = entry.payload["workload"]
    if tuple(workload["shape"][x] for x in ("m", "n", "k")) != shape:
      raise ValueError(f"candidate/tensor shape mismatch for {key!r}")
    by_quant.setdefault(row["quant_format"], []).append(entry)
    rows.append({"inventory_key": {"inventory_identity": inventory_identity, "role": row["role"],
      "quant_format": row["quant_format"], "shape": dict(row["shape"]),
      "packed_abi": {x:row["layout"][x] for x in ("logical", "packed", "block_elems", "block_bytes")},
      "tensor_identities": sorted(row["tensor_identities"]), "call_count": row["call_count"],
      "source_bytes": row["source_bytes"]}, "canonical_identity": entry.canonical_identity})

  candidate_sets = {}
  for quant, entries in sorted(by_quant.items()):
    candidate_set = FullKernelCandidateSet(tuple(entries))
    admit_full_kernel_candidate_set(candidate_set)
    candidate_sets[quant] = candidate_set.to_json()
  return {"schema": CANDIDATE_INVENTORY_SCHEMA, "inventory_identity": inventory_identity, "inventory": inventory,
          "candidate_sets": candidate_sets, "bindings": rows}


def write_canonical_json(artifact: dict[str, Any], path: str | Path) -> None:
  Path(path).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
