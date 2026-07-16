"""Private model-load authority for exact-fact memory-adaptive policies.

The public boundary intentionally has one input: the selected model path.  The
controller owns discovery, cache identity, candidate enumeration, and guarded
measurement. Normal model loading only reads a previously completed cache
record. Search and persistence require the explicit refresh entry point.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA = "tinygrad.memory_adaptive_runtime_authority.v1"
_runtime_policy_adapter: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None] | None = None
_memory_evidence_validator: Callable[..., Mapping[str, Any] | None] | None = None
_candidate_set_decoder: Callable[[Mapping[str, Any]], object | None] | None = None

def register_memory_adaptive_adapters(*, policy_adapter=None, evidence_validator=None, candidate_set_decoder=None) -> None:
  """Install optional policy-data decoders without coupling the model to their package."""
  global _runtime_policy_adapter, _memory_evidence_validator, _candidate_set_decoder
  if policy_adapter is not None:
    if not callable(policy_adapter): raise TypeError("policy adapter must be callable")
    _runtime_policy_adapter = policy_adapter
  if evidence_validator is not None:
    if not callable(evidence_validator): raise TypeError("evidence validator must be callable")
    _memory_evidence_validator = evidence_validator
  if candidate_set_decoder is not None:
    if not callable(candidate_set_decoder): raise TypeError("candidate-set decoder must be callable")
    _candidate_set_decoder = candidate_set_decoder

def adapt_cached_memory_policy(request: Mapping[str, Any], source: Mapping[str, Any]) -> Mapping[str, Any] | None:
  return None if _runtime_policy_adapter is None else _runtime_policy_adapter(request, source)

def validate_memory_evidence(evidence, *, candidate_id: str):
  return None if _memory_evidence_validator is None else _memory_evidence_validator(evidence, candidate_id=candidate_id)

def decode_candidate_set(candidate_set: Mapping[str, Any]):
  return None if _candidate_set_decoder is None else _candidate_set_decoder(candidate_set)


def _cache_path(selected_model_source: str) -> Path:
  root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tinygrad"
  identity = hashlib.sha256(os.path.realpath(selected_model_source).encode()).hexdigest()
  return root / "memory-adaptive" / (identity + ".json")


def _read(path: Path) -> Mapping[str, Any] | None:
  try:
    value = json.loads(path.read_text())
    return value if isinstance(value, Mapping) else None
  except (OSError, ValueError, TypeError):
    return None


def _write(path: Path, value: Mapping[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with os.fdopen(fd, "w") as handle:
      json.dump(value, handle, sort_keys=True, separators=(",", ":"))
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, path)
  except BaseException:
    try: os.unlink(temporary)
    except FileNotFoundError: pass
    raise


def _resolve_for_test(selected_model_source: str, *, runner: Callable[..., Mapping[str, Any]],
                      cache: Mapping[str, Any] | None = None) -> Mapping[str, Any] | None:
  """Private seam used by unit tests; never part of the production API."""
  try:
    result = runner(model_path=selected_model_source, cache_record=cache)
  except Exception:
    return None
  if (not isinstance(result, Mapping) or result.get("decision") != "SELECTED" or
      result.get("interrupted") is not False): return None
  record, policy = result.get("cache_record"), result.get("policy")
  if not isinstance(record, Mapping) or not isinstance(policy, Mapping): return None
  cached = record.get("result")
  if (not isinstance(cached, Mapping) or cached != policy or
      result.get("selected_candidate_id") != cached.get("selected_candidate_id")): return None
  # Runtime inventory/device/workload validation belongs to
  # collect_runtime_policy, after from_gguf has opened the selected model and
  # taken its one live DeviceFacts snapshot. This boundary only admits a
  # completed, persistent controller result and preserves the whole envelope.
  return dict(result)


def resolve_memory_adaptive_policy(selected_model_source: str) -> Mapping[str, Any] | None:
  """Read a completed cached policy, or return ``None`` for the direct baseline.

  This model-load path is deliberately read-only: it never imports or launches
  the search controller and never creates or updates cache state.
  """
  if not isinstance(selected_model_source, str) or not selected_model_source: return None
  try:
    record = _read(_cache_path(selected_model_source))
    policy = record.get("result") if isinstance(record, Mapping) else None
    if not isinstance(policy, Mapping): return None
    candidate = policy.get("selected_candidate_id")
    if not isinstance(candidate, str) or not candidate: return None
    raw = {"decision": "SELECTED", "selected_candidate_id": candidate, "interrupted": False,
           "from_cache": True, "policy": policy, "cache_record": record}
    return _resolve_for_test(selected_model_source, runner=lambda **_: raw, cache=record)
  except Exception:
    return None


__all__ = ["SCHEMA", "resolve_memory_adaptive_policy", "register_memory_adaptive_adapters", "adapt_cached_memory_policy",
           "validate_memory_evidence", "decode_candidate_set"]
