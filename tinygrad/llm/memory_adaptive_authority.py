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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA = "tinygrad.memory_adaptive_runtime_authority.v1"

@dataclass(frozen=True)
class MemoryAdaptiveAdapters:
  policy_adapter: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None]
  evidence_validator: Callable[..., Mapping[str, Any] | None]
  candidate_set_decoder: Callable[[Mapping[str, Any]], object | None]

  def __post_init__(self) -> None:
    if not all(callable(x) for x in (self.policy_adapter, self.evidence_validator, self.candidate_set_decoder)):
      raise TypeError("memory-adaptive adapters must all be callable")

_adapters: MemoryAdaptiveAdapters | None = None

def activate_memory_adaptive_adapters(adapters: MemoryAdaptiveAdapters) -> None:
  """Explicitly activate one complete production adapter bundle."""
  if not isinstance(adapters, MemoryAdaptiveAdapters): raise TypeError("expected a MemoryAdaptiveAdapters bundle")
  global _adapters
  _adapters = adapters

def memory_adaptive_adapters_active() -> bool: return _adapters is not None

def register_memory_adaptive_adapters(*, policy_adapter=None, evidence_validator=None, candidate_set_decoder=None) -> None:
  """Compatibility registration; activation is deliberately all-or-nothing."""
  current = _adapters
  values = (policy_adapter or (current.policy_adapter if current else None),
            evidence_validator or (current.evidence_validator if current else None),
            candidate_set_decoder or (current.candidate_set_decoder if current else None))
  if not all(callable(x) for x in values): raise TypeError("registration requires one complete adapter bundle")
  activate_memory_adaptive_adapters(MemoryAdaptiveAdapters(*values))

def adapt_cached_memory_policy(request: Mapping[str, Any], source: Mapping[str, Any]) -> Mapping[str, Any] | None:
  if _adapters is None: raise RuntimeError("cached memory-adaptive policy requires explicit adapter activation")
  return _adapters.policy_adapter(request, source)

def validate_memory_evidence(evidence, *, candidate_id: str):
  if _adapters is None: raise RuntimeError("memory-adaptive evidence requires explicit adapter activation")
  return _adapters.evidence_validator(evidence, candidate_id=candidate_id)

def decode_candidate_set(candidate_set: Mapping[str, Any]):
  if _adapters is None: raise RuntimeError("memory-adaptive candidate set requires explicit adapter activation")
  return _adapters.candidate_set_decoder(candidate_set)


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


__all__ = ["SCHEMA", "MemoryAdaptiveAdapters", "activate_memory_adaptive_adapters", "memory_adaptive_adapters_active",
           "resolve_memory_adaptive_policy", "register_memory_adaptive_adapters", "adapt_cached_memory_policy",
           "validate_memory_evidence", "decode_candidate_set"]
