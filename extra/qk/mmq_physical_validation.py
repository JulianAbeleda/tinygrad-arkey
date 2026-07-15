"""Static checks for physical MMQ launch contracts.

This module consumes lowered facts; it does not choose a schedule or inspect
the bounded atom.  A missing fact is an error so probe metadata cannot become
an accidental promotion claim.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
import re


def validate_physical_contract(*, local_size: Iterable[int], consumed_local_dims: Iterable[int],
                               lane_map: Mapping[str, Any], barriers: Iterable[Mapping[str, Any]],
                               owners: Iterable[Mapping[str, Any]],
                               expected_outputs: Iterable[tuple[int, int]]) -> dict[str, Any]:
  """Validate lowered launch/lane facts and return an audit-friendly summary."""
  local = tuple(int(x) for x in local_size)
  consumed_values = tuple(int(x) for x in consumed_local_dims)
  consumed = frozenset(consumed_values)
  errors: list[str] = []
  if len(consumed_values) != len(consumed): errors.append("consumed local dimensions contain duplicates")
  if not local or any(x <= 0 for x in local): errors.append("local dimensions must be positive")
  if any(d < 0 or d >= len(local) for d in consumed): errors.append("consumed local dimension is out of range")
  unused = tuple(i for i, size in enumerate(local) if size > 1 and i not in consumed)
  if unused: errors.append(f"unused non-unit local dimensions: {unused}")
  if not lane_map or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in lane_map.items()): errors.append("physical lane mapping is incomplete")
  lane_dims = {int(match.group(1)) for value in lane_map.values() if (match := re.fullmatch(r"lidx([0-9]+)", value))}
  if any(d >= len(local) for d in lane_dims): errors.append("lane mapping references an out-of-range local dimension")
  if not lane_dims.issubset(consumed): errors.append("lane mapping uses a local dimension not declared consumed")
  barrier_rows = tuple(barriers)
  for i, barrier in enumerate(barrier_rows):
    if barrier.get("uniform") is not True: errors.append(f"barrier {i} is not proven uniform")
    if barrier.get("scope", "workgroup") != "workgroup": errors.append(f"barrier {i} does not have an explicit workgroup scope")
  expected = {(int(m), int(n)) for m, n in expected_outputs}
  owner_rows = tuple(owners)
  actual = {(int(row["m"]), int(row["n"])) for row in owner_rows if "m" in row and "n" in row}
  if len(actual) != len(owner_rows): errors.append("owner map contains duplicate or incomplete output coordinates")
  if actual != expected: errors.append("owner map is not exactly one-to-one over outputs")
  if any(not isinstance(row, Mapping) or "owner" not in row or not isinstance(row["owner"], Mapping) for row in owner_rows): errors.append("owner map lacks explicit owner facts")
  return {"passed": not errors, "errors": errors, "local_size": list(local),
          "consumed_local_dims": sorted(consumed), "unused_local_dims": list(unused),
          "barrier_count": len(barrier_rows), "owner_count": len(owner_rows)}
