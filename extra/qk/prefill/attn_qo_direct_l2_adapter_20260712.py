"""CPU-only adapter for the guarded attn_qo direct-L2/LDS experiment.

This module is deliberately an adapter, not an execution authority.  Payloads
and execution/timing evidence are supplied by the existing single-buffer
authorities (or a future caller); this file only validates joins and exposes
callback shapes for the canary and paired benchmark runners.
"""
from __future__ import annotations

from typing import Any, Callable

from extra.qk.prefill.pure_register_direct_l2_decision import candidate
from extra.qk.prefill.pure_single_buffer_evaluation_gate import canonical_candidate_hash
from extra.qk.prefill.register_hardware_promotion import EXACT_ROLE, TARGET
from extra.qk.runtime_specs import admit_full_kernel_candidate
from tinygrad.runtime.execution_bridge_contracts import dispatch_state

SCHEMA = "attn-qo-direct-l2-adapter.v1"
PROFILE = "qwen3_8b_q4k_m_gfx1100"
SHAPE = {"m": 512, "n": 4096, "k": 4096}
_HEX = set("0123456789abcdef")


def _sha(value: Any) -> bool:
  return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _blocked(*reasons: str, **extra: Any) -> dict[str, Any]:
  return {"schema": SCHEMA, "status": "blocked", "decision": "blocked",
          "dispatch_state": dispatch_state("not_attempted"), "blockers": list(reasons), **extra}


def prepare_exact_pair(*, direct_payload: dict[str, Any] | None,
                       lds_payload: dict[str, Any] | None,
                       direct_binary_sha256: str | None,
                       lds_binary_sha256: str | None,
                       pair_key: str | None = None,
                       profile: str = PROFILE,
                       role: str = EXACT_ROLE,
                       shape: dict[str, int] = SHAPE,
                       target: dict[str, Any] = TARGET) -> dict[str, Any]:
  """Prepare identity-joined candidate metadata without compiling or dispatching.

  The expected workload (profile/role/shape/target) is row data supplied by the
  experiment (P2-3); this adapter validates the candidates against it and stays
  workload-neutral rather than hardcoding a single role/shape/target.
  """
  if direct_payload is None or lds_payload is None:
    return _blocked("exact direct_l2 and lds candidate payloads are required")
  if direct_binary_sha256 == lds_binary_sha256:
    return _blocked("paired binaries must be distinct")
  if not _sha(direct_binary_sha256) or not _sha(lds_binary_sha256):
    return _blocked("exact direct_l2 and lds binary SHA-256 identities are required")
  if not isinstance(pair_key, str) or not pair_key:
    return _blocked("semantic pair key is required")
  try:
    direct_id, lds_id = canonical_candidate_hash(direct_payload), canonical_candidate_hash(lds_payload)
  except Exception as exc:
    return _blocked(f"candidate payload identity cannot be proven: {type(exc).__name__}: {exc}")
  shape_tuple = (shape["m"], shape["n"], shape["k"])
  errors = []
  for name, payload in (("direct_l2", direct_payload), ("lds", lds_payload)):
    workload = payload["workload"]
    if workload.get("profile") != profile: errors.append(f"{name} workload profile is not exact")
    if workload.get("role") != role: errors.append(f"{name} workload role is not {role}")
    if workload.get("shape") != shape: errors.append(f"{name} workload shape is not {shape_tuple}")
    if workload.get("target") != target: errors.append(f"{name} workload target is not the row target")
    identity = direct_id if name == "direct_l2" else lds_id
    try: admit_full_kernel_candidate(payload, identity, profile=profile, role=role,
                                     shape=shape_tuple, target=target)
    except Exception as exc: errors.append(f"{name} is not admitted by single_buffer authority: {exc}")
  if errors: return _blocked(*errors, canonical_identity=direct_id, pair_key=pair_key)
  return {"schema": SCHEMA, "status": "prepared", "decision": "pending_external_evidence",
          "dispatch_state": dispatch_state("not_attempted"), "canonical_identity": direct_id, "pair_key": pair_key,
          "role": role, "shape": dict(shape), "target": dict(target), "profile": profile,
          "candidates": {"direct_l2": {"storage": "direct_l2", "canonical_identity": direct_id, "binary_sha256": direct_binary_sha256},
                          "lds": {"storage": "lds", "canonical_identity": lds_id, "binary_sha256": lds_binary_sha256}}}


def make_benchmark_callback(pair: dict[str, Any], capture: Callable[[str, str, int], dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
  """Adapt an external, already-authorized paired capture to the canary callback."""
  def callback(contract: dict[str, Any]) -> dict[str, Any]:
    if pair.get("status") != "prepared": return _blocked(*pair.get("blockers", ()))
    if contract.get("canonical_identity") != pair["canonical_identity"]:
      return _blocked("canary benchmark contract identity differs from prepared pair")
    # P0-3: each benchmark row keeps its OWN distinct candidate identity; the
    # shared pair identity must never overwrite the LDS candidate identity.
    rows = {name: {"role": pair["role"], "shape": dict(pair["shape"]), "canonical_identity": row["canonical_identity"],
                   "binary_sha256": row["binary_sha256"], "storage": row["storage"], "pair_key": pair["pair_key"]}
            for name, row in pair["candidates"].items()}
    for storage in ("direct_l2", "lds"):
      rows[storage].update(capture(storage, "timed", 0))
    return rows
  return callback


def make_paired_runner_callbacks(pair: dict[str, Any], *, artifact: Callable[[str], dict[str, Any]],
                                 route_binding: Callable[[str], dict[str, Any]],
                                 correctness: Callable[[str], dict[str, Any]],
                                 benchmark: Callable[[str, str, int], dict[str, Any]]) -> dict[str, Callable[..., Any]]:
  """Return the callback bundle expected by ``run_paired_direct_l2_benchmark``.

  The supplied callbacks remain external owners of compilation, dispatch, and
  measurement.  A blocked preparation yields blocked prerequisites and the
  benchmark callback is never called.
  """
  def guarded(name: str, callback: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    if pair.get("status") != "prepared": return _blocked(*pair.get("blockers", ()))
    return callback(name)
  def timed(storage: str, phase: str, index: int) -> dict[str, Any]:
    if pair.get("status") != "prepared": return {"samples_ms": [], "counters": {}}
    return benchmark(storage, phase, index)
  return {"artifact": lambda storage: guarded(storage, artifact),
          "route_binding": lambda storage: guarded(storage, route_binding),
          "correctness": lambda storage: guarded(storage, correctness),
          "benchmark": timed}
