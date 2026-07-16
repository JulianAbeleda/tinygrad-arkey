"""Explicit search refresh entry point for the memory-adaptive experiments."""
from __future__ import annotations

from typing import Any, Mapping

from extra.qk import memory_adaptive_search_controller
from tinygrad.llm.memory_adaptive_authority import _cache_path, _read, _resolve_for_test, _write

def refresh_memory_adaptive_policy(selected_model_source: str, *, min_samples: int = 3) -> Mapping[str, Any] | None:
  if not isinstance(selected_model_source, str) or not selected_model_source: return None
  path = _cache_path(selected_model_source)
  cache = _read(path)
  raw = memory_adaptive_search_controller.run_controller(model_path=selected_model_source, cache_record=cache, min_samples=min_samples)
  resolved = _resolve_for_test(selected_model_source, runner=lambda **_: raw, cache=cache)
  if resolved is not None: _write(path, resolved["cache_record"])
  return resolved

search_memory_adaptive_policy = refresh_memory_adaptive_policy
__all__ = ["refresh_memory_adaptive_policy", "search_memory_adaptive_policy"]
