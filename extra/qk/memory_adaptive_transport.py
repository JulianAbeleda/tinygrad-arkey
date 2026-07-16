"""Import-light data transported across memory-adaptive process boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SelectedModelScan:
  """Exact facts discovered from the model selected by the user."""
  facts: Mapping[str, Any]
  inventory: Mapping[str, Any]
  base_terms: Sequence[Any]
  workload: Mapping[str, Any]
  compiler_runtime_revision: Mapping[str, Any]


__all__ = ["SelectedModelScan"]
