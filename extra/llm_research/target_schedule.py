"""One derive authority for the typed prefill schedule (T2).

``derive_target_schedule`` assembles a target's v1 schedule from three inputs
instead of cloning another target's literal (target-schedule-derivation-scope-
20260801.md):

* the **declared row** (``FullKernelCapability``, extended with the C-class
  emitter contracts it does not yet carry): fragment/lane/accumulator
  vocabulary, waitcnt/barrier vocabulary, lds banks/padding, residency,
  numerical mode, register/spill constraints. Values are carried verbatim;
  the derive function never fabricates vocabulary (in particular it never
  invents a ``wmma_`` prefix -- the family string is read from the row).
* the **geometry** (search output, caller-supplied): tile, waves,
  buffer_count/stage_count.
* the **shape** (workload facts, caller-supplied): exact M/N/K and the operand
  dtypes the schedule transports (the vector widths are
  ``vector_bytes // itemsize``, so itemsize is an input, not a literal).

Derived fields (computed, never stored as a literal): ``threads``
(``wm*wn*wave_size``), cooperative-load and LDS vector widths/alignment from
``vector_bytes``, ``lds.strides`` (``tk*itemsize + padding``), ``lds.windows``
(``tile*stride``), and ``static_constraints.max_lds_bytes`` (transport-
dependent: ``Renderer.shared_max`` for LDS transport; the direct-global and
register schedules are their own emitter authorities and fail closed here).

The v1 schema shape does not change: values are retyped, keys stay exactly as
today. AMD byte-identity with the promoted template is a test, not an
aspiration.
"""
from __future__ import annotations

from typing import Any

from tinygrad.dtype import dtypes

from extra.llm_research.runtime_specs import FullKernelCapability

_GEOMETRY_KEYS = {"tile", "waves", "buffer_count", "stage_count"}
_TILE_KEYS = {"m", "n", "k"}
_WAVES_KEYS = {"m", "n"}
_SHAPE_KEYS = {"m", "n", "k", "dtypes"}
_DTYPES_KEYS = {"a", "b", "c", "accumulator"}


def _strict_keys(value: Any, required: set[str], label: str) -> None:
  if not isinstance(value, dict):
    raise ValueError(f"{label} must be an object")
  missing, unknown = required - set(value), set(value) - required
  if missing: raise ValueError(f"{label} missing fields {sorted(missing)}")
  if unknown: raise ValueError(f"{label} has unknown fields {sorted(unknown)}")


def _positive_int(value: Any, label: str) -> None:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f"{label} must be a positive int, got {value!r}")


def _nonempty_str(value: Any, label: str) -> None:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a non-empty string")


def derive_target_schedule(row: FullKernelCapability, geometry: dict[str, Any],
                           shape: dict[str, Any]) -> dict[str, Any]:
  """Assemble a target's typed v1 schedule from declared row + geometry + shape.

  Returns ``{"schedule": ..., "static_constraints": ...}`` with the exact v1
  key shape. ``geometry`` carries ``tile``/``waves``/``buffer_count``/
  ``stage_count``; ``shape`` carries exact ``m``/``n``/``k`` plus ``dtypes``.
  The output never contains vocabulary the row does not declare.
  """
  if row.transport != "lds":
    raise ValueError(
      f"derive_target_schedule supports the LDS buffer family only, got transport {row.transport!r}; "
      "register/direct-global schedules are their own emitter authorities")
  _strict_keys(geometry, _GEOMETRY_KEYS, "geometry")
  _strict_keys(geometry["tile"], _TILE_KEYS, "geometry.tile")
  _strict_keys(geometry["waves"], _WAVES_KEYS, "geometry.waves")
  for dim in ("m", "n", "k"): _positive_int(geometry["tile"][dim], f"geometry.tile.{dim}")
  for dim in ("m", "n"): _positive_int(geometry["waves"][dim], f"geometry.waves.{dim}")
  for key in ("buffer_count", "stage_count"): _positive_int(geometry[key], f"geometry.{key}")
  _strict_keys(shape, _SHAPE_KEYS, "shape")
  for dim in ("m", "n", "k"): _positive_int(shape[dim], f"shape.{dim}")
  _strict_keys(shape["dtypes"], _DTYPES_KEYS, "shape.dtypes")
  for key in ("a", "b"): _nonempty_str(shape["dtypes"][key], f"shape.dtypes.{key}")
  if any(shape["dtypes"][key] != "fp16" for key in ("a", "b")):
    raise ValueError("derive_target_schedule transports fp16 operands only")

  itemsize = dtypes.half.itemsize
  if row.vector_bytes % itemsize:
    raise ValueError(f"row vector_bytes {row.vector_bytes} is not a multiple of the fp16 itemsize")
  vector_width = row.vector_bytes // itemsize
  tile, waves = geometry["tile"], geometry["waves"]
  tm, tn, tk = tile["m"], tile["n"], tile["k"]
  wm, wn = waves["m"], waves["n"]
  stride = tk * itemsize + row.lds_padding
  a_end, b_end = tm * stride, (tm + tn) * stride

  epoch_graph = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in row.epoch_graph}]
  schedule: dict[str, Any] = {
    "tile": {"m": tm, "n": tn, "k": tk},
    "waves": {"m": wm, "n": wn},
    "threads": wm * wn * row.wave_size,
    "lane_ownership": row.lane_ownership,
    "cooperative_load": {
      operand: {"lane_mapping": row.cooperative_lane_mapping, "vector_width": vector_width,
                "alignment": row.vector_bytes}
      for operand in ("a", "b")
    },
    "lds": {
      "windows": {"a": [0, a_end], "b": [a_end, b_end]},
      "strides": {"a": stride, "b": stride},
      "padding": row.lds_padding,
      "banks": row.lds_banks,
      "store_vector_width": vector_width,
      "load_vector_width": vector_width,
    },
    "pipeline": {"buffer_count": geometry["buffer_count"], "stage_count": geometry["stage_count"],
                 "epoch_graph": epoch_graph},
    "wmma": {"instruction_family": row.instruction_family, "fragment_layout": row.fragment_layout,
             "accumulator_ownership": row.accumulator_ownership},
    "dependency_policy": {"waitcnt": dict(row.waitcnt), "barriers": list(row.dependency_barriers)},
    "residency": {"preload": list(row.residency_preload), "resident": list(row.residency_resident),
                  "reuse": dict(row.residency_reuse)},
    "epilogue": {"lane_mapping": row.epilogue_lane_mapping, "vector_width": row.epilogue_vector_width},
    "numerical_mode": row.numerical_mode,
  }
  static_constraints = {"max_lds_bytes": row.max_lds_bytes,
                        "max_vgpr_per_thread": row.max_vgpr_per_thread,
                        "allow_spill": row.allow_spill}
  return {"schedule": schedule, "static_constraints": static_constraints}


__all__ = ["derive_target_schedule"]
