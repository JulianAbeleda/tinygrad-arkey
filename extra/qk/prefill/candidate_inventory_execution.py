#!/usr/bin/env python3
"""Strictly sequential execution of canonical packed-prefill inventories."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from extra.qk.prefill.current_prefill_execution_adapter import (ADAPTER_ID, prepare_current_prefill_compile,
  register_current_prefill_adapter)
from extra.qk.prefill.q4k_q8_five_buffer_execution_adapter import (ADAPTER_ID as FIVE_BUFFER_ADAPTER_ID,
  prepare_q4k_q8_five_buffer_compile, register_q4k_q8_five_buffer_adapter)
from extra.qk.prefill.q4k_q8_five_buffer_artifact import (build_q4k_q8_five_buffer_artifact,
  save_q4k_q8_five_buffer_artifact)
from extra.qk.prefill.operand_path_execution_worker import AdapterRegistry, execute as execute_request
from extra.qk.prefill.packed_wmma_correctness_canary import build_artifact
from extra.qk.prefill.workload_inventory import CANDIDATE_INVENTORY_SCHEMA, INVENTORY_SCHEMA
from extra.qk.runtime_specs import (FullKernelCandidateSet, FullKernelCandidateSetEntry, capability_transport,
  Q4K_Q8_1_FIVE_BUFFER_ABI, full_kernel_candidate_capability, full_kernel_workload)
from extra.qk.prefill.execution_bridge_contracts import (CorrectnessProtocol, ExecutionRequest,
  GuardProtocol, TimingProtocol, TransportPlan, canonical_digest)

OUTPUT_SCHEMA = "qk.packed_prefill_candidate_inventory_execution.v1"


@dataclass(frozen=True)
class JoinedCandidate:
  order: int
  role: str
  quant_format: str
  shape: tuple[int, int, int]
  tensor_identities: tuple[str, ...]
  canonical_identity: str
  payload: dict[str, Any]

  @property
  def inventory_key(self) -> tuple[str, str, int, int, int]:
    return (self.role, self.quant_format, *self.shape)


def _key(value: Any, name: str) -> tuple[str, str, int, int, int]:
  if not isinstance(value, Mapping): raise ValueError(f"{name} must be a mapping")
  try: role, quant, shape = value["role"], value["quant_format"], value["shape"]
  except KeyError as exc: raise ValueError(f"{name} is malformed") from exc
  if not isinstance(shape, Mapping): raise ValueError(f"{name} is malformed")
  try: shape = [shape[x] for x in ("m", "n", "k")]
  except KeyError as exc: raise ValueError(f"{name} is malformed") from exc
  if not isinstance(role, str) or not role or not isinstance(quant, str) or not quant or \
     any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in shape):
    raise ValueError(f"{name} is malformed")
  return role, quant, *shape


def validate_and_join(artifact: Mapping[str, Any]) -> tuple[JoinedCandidate, ...]:
  """Validate the complete inventory/binding/partition join and return inventory order."""
  if not isinstance(artifact, Mapping) or artifact.get("schema") != CANDIDATE_INVENTORY_SCHEMA:
    raise ValueError("unsupported candidate inventory schema")
  if set(artifact) != {"schema", "inventory_identity", "inventory", "candidate_sets", "bindings"}:
    raise ValueError("candidate inventory has unknown or missing top-level fields")
  inventory, sets, bindings = artifact["inventory"], artifact["candidate_sets"], artifact["bindings"]
  inventory_identity = artifact.get("inventory_identity")
  if not isinstance(inventory_identity, str) or not inventory_identity or \
     not isinstance(inventory, Mapping) or inventory.get("schema") != INVENTORY_SCHEMA or \
     inventory.get("inventory_identity") != inventory_identity or not isinstance(inventory.get("rows"), list) or \
     not isinstance(sets, Mapping) or not isinstance(bindings, list):
    raise ValueError("candidate inventory containers are malformed")

  rows: dict[tuple[str, str, int, int, int], tuple[int, tuple[str, ...]]] = {}
  order: list[tuple[str, str, int, int, int]] = []
  for idx, row in enumerate(inventory["rows"]):
    try:
      key = _key(row, "inventory key")
      identities = tuple(row["tensor_identities"])
    except (KeyError, TypeError) as exc: raise ValueError("malformed inventory row") from exc
    if key in rows: raise ValueError(f"duplicate inventory key {key!r}")
    if not identities or any(not isinstance(x, str) or not x for x in identities) or len(set(identities)) != len(identities):
      raise ValueError(f"invalid tensor identities for {key!r}")
    rows[key] = (idx, identities); order.append(key)

  binding_by_key: dict[tuple[str, str, int, int, int], tuple[str, Mapping[str, Any]]] = {}
  for binding in bindings:
    if not isinstance(binding, Mapping) or set(binding) != {"inventory_key", "canonical_identity"}:
      raise ValueError("binding has unknown or missing fields")
    key = _key(binding["inventory_key"], "binding inventory_key")
    identity = binding["canonical_identity"]
    if key in binding_by_key: raise ValueError(f"duplicate binding key {key!r}")
    if not isinstance(identity, str) or not identity: raise ValueError("binding canonical identity is malformed")
    inventory_key = binding["inventory_key"]
    if set(inventory_key) != {"inventory_identity", "role", "quant_format", "shape", "packed_abi",
                              "tensor_identities", "call_count", "source_bytes"} or \
       inventory_key.get("inventory_identity") != inventory_identity:
      raise ValueError(f"binding inventory_key is malformed for {key!r}")
    binding_by_key[key] = (identity, inventory_key)

  candidates: dict[tuple[str, str, int, int, int], FullKernelCandidateSetEntry] = {}
  for quant, raw_set in sets.items():
    if not isinstance(quant, str): raise ValueError("candidate-set quant key is malformed")
    candidate_set = FullKernelCandidateSet.from_json(raw_set)
    for entry in candidate_set.entries:
      workload = full_kernel_workload(entry.payload)
      actual_quant = entry.payload.get("kernel_abi", {}).get("quant_format") if quant == "Q4_K" else \
        entry.payload.get("operand_sources", {}).get("b", {}).get("quant_format")
      key = (workload.role, quant, *workload.shape)
      if actual_quant != quant: raise ValueError(f"quant partition drift for {entry.canonical_identity}")
      if workload.profile != inventory_identity: raise ValueError(f"profile drift for {entry.canonical_identity}")
      if key in candidates: raise ValueError(f"duplicate candidate key {key!r}")
      candidates[key] = entry

  expected = set(rows)
  for label, actual in (("bindings", set(binding_by_key)), ("candidate sets", set(candidates))):
    unknown, missing = actual - expected, expected - actual
    if unknown: raise ValueError(f"unknown {label} keys {sorted(unknown)!r}")
    if missing: raise ValueError(f"missing {label} keys {sorted(missing)!r}")
  joined = []
  for key in order:
    entry = candidates[key]
    binding_identity, bound = binding_by_key[key]
    row = inventory["rows"][rows[key][0]]
    expected_bound = {"inventory_identity":inventory_identity, "role":row["role"],
      "quant_format":row["quant_format"], "shape":row["shape"],
      "packed_abi":{x:row["layout"][x] for x in ("logical", "packed", "block_elems", "block_bytes")},
      "tensor_identities":sorted(row["tensor_identities"]), "call_count":row["call_count"],
      "source_bytes":row["source_bytes"]}
    if dict(bound) != expected_bound: raise ValueError(f"binding inventory_key drift for {key!r}")
    if binding_identity != entry.canonical_identity:
      raise ValueError(f"canonical identity drift for {key!r}")
    joined.append(JoinedCandidate(rows[key][0], key[0], key[1], key[2:], rows[key][1],
                                  entry.canonical_identity, entry.to_json()["payload"]))
  return tuple(joined)


def select_candidates(candidates: Iterable[JoinedCandidate], *, roles: Iterable[str] = (),
                      quant_formats: Iterable[str] = ()) -> tuple[JoinedCandidate, ...]:
  roles, quant_formats = frozenset(roles), frozenset(quant_formats)
  selected = tuple(x for x in candidates if (not roles or x.role in roles) and
                   (not quant_formats or x.quant_format in quant_formats))
  known_roles, known_quants = {x.role for x in candidates}, {x.quant_format for x in candidates}
  if roles - known_roles: raise ValueError(f"unknown role filters {sorted(roles-known_roles)!r}")
  if quant_formats - known_quants: raise ValueError(f"unknown quant filters {sorted(quant_formats-known_quants)!r}")
  if not selected: raise ValueError("filters selected no candidates")
  return selected


def make_request(candidate: JoinedCandidate, input_npz: str, *, phase: str,
                 timeout_seconds: float = 30.0, warmups: int = 1, rounds: int = 5) -> ExecutionRequest:
  if phase not in ("compile-only", "correctness", "timing"): raise ValueError("invalid GPU phase")
  workload_facts = {"role": candidate.role, "quant_format": candidate.quant_format,
    "shape": dict(zip(("m", "n", "k"), candidate.shape)), "tensor_identities": list(candidate.tensor_identities),
    "workload": candidate.payload["workload"]}
  workload_digest = canonical_digest(workload_facts, "workload facts")
  schedule_digest = canonical_digest({"schedule": candidate.payload["schedule"],
    "canonical_identity": candidate.canonical_identity}, "schedule facts")
  transport = capability_transport(full_kernel_candidate_capability(candidate.payload))
  adapter_id = (FIVE_BUFFER_ADAPTER_ID if candidate.payload.get("kernel_abi", {}).get("family") == Q4K_Q8_1_FIVE_BUFFER_ABI
                else ADAPTER_ID)
  return ExecutionRequest(experiment_id="sha256:" + canonical_digest({"workload": workload_digest,
      "schedule": schedule_digest, "candidate": candidate.canonical_identity}),
    candidate_id=candidate.canonical_identity, comparator_id=candidate.canonical_identity,
    workload_digest=workload_digest, schedule_digest=schedule_digest,
    transport_plan=TransportPlan(transport, schedule_digest),
    target_context={"workload": workload_facts}, compiler_context={"adapter_id": adapter_id,
      "candidate_payload": candidate.payload, "canonical_identity": candidate.canonical_identity, "input_npz": input_npz},
    candidate_knobs=dict(candidate.payload["schedule"]), fixed_invariants=workload_facts,
    correctness=None if phase == "compile-only" else CorrectnessProtocol("packed_wmma_npz", atol=2e-2, rtol=2e-2),
    guard=None if phase == "compile-only" else GuardProtocol(max(1, int(timeout_seconds * 1000))),
    timing=TimingProtocol(warmups, rounds, 0) if phase == "timing" else None)


def _failed_or_unhealthy(result: Any) -> bool:
  phases = getattr(result, "phases", ())
  if not phases or any(getattr(x, "status", None) != "passed" for x in phases): return True
  for phase in phases:
    health = getattr(phase, "evidence", {}).get("health", {})
    if health.get("preflight") is False or health.get("postflight") is False or health.get("device_fault") is True: return True
  return False


def _is_five_buffer(candidate: JoinedCandidate) -> bool:
  return candidate.payload.get("kernel_abi", {}).get("family") == Q4K_Q8_1_FIVE_BUFFER_ABI


def _build_default_input(candidate: JoinedCandidate, path: str) -> Mapping[str, Any]:
  if _is_five_buffer(candidate):
    return save_q4k_q8_five_buffer_artifact(path, build_q4k_q8_five_buffer_artifact(*candidate.shape))
  return build_artifact(candidate.quant_format, path, candidate.shape)


def run_inventory(artifact: Mapping[str, Any], *, phase: str, artifact_dir: str,
                  roles: Iterable[str] = (), quant_formats: Iterable[str] = (), timeout_seconds: float = 30.0,
                  warmups: int = 1, rounds: int = 5, build_fn: Callable[..., Mapping[str, Any]] | None = None,
                  prepare_fn: Callable[..., tuple[Any, Mapping[str, Any]]] | None = None,
                  execute_fn: Callable[[ExecutionRequest], Any] | None = None) -> dict[str, Any]:
  """Run selected candidates serially. Injected functions keep orchestration CPU-testable."""
  if phase not in ("build-input", "compile-only", "correctness", "timing"): raise ValueError("invalid phase")
  selected = select_candidates(validate_and_join(artifact), roles=roles, quant_formats=quant_formats)
  root = Path(artifact_dir)
  outputs = []
  executor = execute_fn
  if phase in ("correctness", "timing") and executor is None:
    registry = AdapterRegistry()
    register_current_prefill_adapter(registry)
    register_q4k_q8_five_buffer_adapter(registry)
    executor = lambda request: execute_request(request, registry=registry)
  for candidate in selected:
    path = str(root / f"{candidate.order:04d}-{candidate.canonical_identity}.npz")
    identity = {"inventory_key": list(candidate.inventory_key), "canonical_identity": candidate.canonical_identity}
    if phase == "build-input":
      value = dict(build_fn(candidate.quant_format, path, candidate.shape) if build_fn is not None
                   else _build_default_input(candidate, path))
      outputs.append({"identity": identity, "phase": phase, "status": "passed", "artifact": value})
      continue
    request = make_request(candidate, path, phase=phase, timeout_seconds=timeout_seconds, warmups=warmups, rounds=rounds)
    if phase == "compile-only":
      try:
        compile_prepare = prepare_fn or (prepare_q4k_q8_five_buffer_compile if _is_five_buffer(candidate)
                                         else prepare_current_prefill_compile)
        compile_kwargs = ({"device": "AMD"} if prepare_fn is not None or not _is_five_buffer(candidate)
                          else {"target": "AMD:ISA:gfx1100"})
        _, evidence = compile_prepare(candidate.payload, candidate.canonical_identity, **compile_kwargs)
        value = {"compile_evidence": dict(evidence)}
        outputs.append({"identity": identity, "request": request.to_dict(), "phase": phase, "status": "passed", **value})
      except Exception as exc:
        outputs.append({"identity": identity, "request": request.to_dict(), "phase": phase, "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}"})
        break
      continue
    try:
      if executor is None: raise RuntimeError("execution provider is unavailable")
      result = executor(request)
    except Exception as exc:
      outputs.append({"identity": identity, "request": request.to_dict(), "phase": phase, "status": "failed",
                      "error": f"{type(exc).__name__}: {exc}"})
      break
    outputs.append({"identity": identity, "request": request.to_dict(), "result": result.to_dict()})
    if _failed_or_unhealthy(result): break
  return {"schema": OUTPUT_SCHEMA, "phase": phase, "selected_count": len(selected), "completed_count": len(outputs),
          "selection": {"roles": sorted(set(roles)), "quant_formats": sorted(set(quant_formats))}, "results": outputs}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("inventory"); parser.add_argument("--artifact-dir", required=True)
  parser.add_argument("--phase", choices=("build-input", "compile-only", "correctness", "timing"), required=True)
  parser.add_argument("--role", action="append", default=[]); parser.add_argument("--quant", action="append", default=[])
  parser.add_argument("--timeout", type=float, default=30.0); parser.add_argument("--warmups", type=int, default=1)
  parser.add_argument("--rounds", type=int, default=5); parser.add_argument("--output")
  args = parser.parse_args()
  artifact = json.loads(Path(args.inventory).read_text())
  result = run_inventory(artifact, phase=args.phase, artifact_dir=args.artifact_dir, roles=args.role,
    quant_formats=args.quant, timeout_seconds=args.timeout, warmups=args.warmups, rounds=args.rounds)
  encoded = json.dumps(result, sort_keys=True)
  if args.output: Path(args.output).write_text(encoded + "\n")
  else: print(encoded)


if __name__ == "__main__": main()
