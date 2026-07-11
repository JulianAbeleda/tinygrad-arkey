from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS
from extra.qk.mmq_q4k_q8_reference import Q8_1_MMQ_DS4_LAYOUT
from extra.qk.q4k_tile_loader import Q4K_TILE_LOAD_LAYOUT


SCHEMA = "tinygrad.mmq_staging_evidence.v1"
SUM_SLOT_SCHEMA = "tinygrad.mmq_sum_slot_map.v1"
STATUSES = ("PASS", "FAIL", "BLOCKED")
STAGING_STATUSES = ("present", "missing", "blocked")
SHAPE_FIELDS = ("M", "N", "K")
TOP_LEVEL_FIELDS = (
  "schema", "evidence_kind", "candidate_id", "backend", "shape", "q4k_tile_staging", "q8_1_ds4_staging",
  "sum_slot_map", "status", "exact_blocker", "production_dispatch_changed", "notes",
)
STAGING_FIELDS = ("status", "layout", "tile_shape", "bytes_per_tile", "block_elems", "panels", "source")
SUM_SLOT_TOP_LEVEL_FIELDS = ("schema", "tile_shape", "slots_per_thread", "total_slots", "mapping")
SUM_SLOT_ENTRY_FIELDS = ("m", "n", "slot", "thread", "lane")


def _validate_non_empty_string(value: Any, path: str) -> None:
  if not isinstance(value, str) or value == "":
    raise ValueError(f"{path} must be a non-empty string")


def _validate_positive_int(value: Any, path: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f"{path} must be a positive integer")
  return value


def _validate_non_negative_int(value: Any, path: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise ValueError(f"{path} must be a non-negative integer")
  return value


def _validate_shape(value: Any, path: str) -> dict[str, int]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{path} must be a dict")
  unknown = set(value) - set(SHAPE_FIELDS)
  if unknown:
    raise ValueError(f"{path} contains unknown fields: {sorted(unknown)}")
  missing = set(SHAPE_FIELDS) - set(value)
  if missing:
    raise ValueError(f"{path} missing required fields: {sorted(missing)}")
  shape = {field: _validate_positive_int(value[field], f"{path}.{field}") for field in SHAPE_FIELDS}
  if shape["M"] % 16:
    raise ValueError(f"{path}.M must be 16-aligned for this bounded R4 evidence")
  if shape["N"] % 16:
    raise ValueError(f"{path}.N must be 16-aligned for this bounded R4 evidence")
  if shape["K"] % Q4_K_BLOCK_ELEMS:
    raise ValueError(f"{path}.K must be {Q4_K_BLOCK_ELEMS}-aligned for Q4_K/Q8_1 MMQ evidence")
  return shape


def _validate_tile_shape(value: Any, path: str) -> dict[str, int]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{path} must be a dict")
  allowed = {"M", "N", "K"}
  unknown = set(value) - allowed
  if unknown:
    raise ValueError(f"{path} contains unknown fields: {sorted(unknown)}")
  if not value:
    raise ValueError(f"{path} must not be empty")
  return {str(field): _validate_positive_int(value[field], f"{path}.{field}") for field in value}


def _validate_staging(value: Any, path: str) -> dict[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{path} must be a dict")
  unknown = set(value) - set(STAGING_FIELDS)
  if unknown:
    raise ValueError(f"{path} contains unknown fields: {sorted(unknown)}")
  if "status" not in value:
    raise ValueError(f"{path}.status missing")
  if value["status"] not in STAGING_STATUSES:
    raise ValueError(f"{path}.status must be one of {STAGING_STATUSES}")
  out: dict[str, Any] = {"status": value["status"]}
  for field in ("layout", "source"):
    if field in value:
      _validate_non_empty_string(value[field], f"{path}.{field}")
      out[field] = value[field]
  if "tile_shape" in value:
    out["tile_shape"] = _validate_tile_shape(value["tile_shape"], f"{path}.tile_shape")
  for field in ("bytes_per_tile", "block_elems", "panels"):
    if field in value:
      out[field] = _validate_positive_int(value[field], f"{path}.{field}")
  return out


def build_bounded_16x16_sum_slot_map(*, waves: int = 8, wave_size: int = 64) -> dict[str, Any]:
  _validate_positive_int(waves, "waves")
  _validate_positive_int(wave_size, "wave_size")
  mapping = []
  for m in range(16):
    for n in range(16):
      slot = m * 16 + n
      thread = slot % (waves * wave_size)
      mapping.append({"m": m, "n": n, "slot": slot, "thread": thread, "lane": thread % wave_size})
  return {
    "schema": SUM_SLOT_SCHEMA,
    "tile_shape": {"M": 16, "N": 16},
    "slots_per_thread": 1,
    "total_slots": 16 * 16,
    "mapping": mapping,
  }


def _validate_sum_slot_map(value: Any, path: str) -> dict[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{path} must be a dict")
  unknown = set(value) - set(SUM_SLOT_TOP_LEVEL_FIELDS)
  if unknown:
    raise ValueError(f"{path} contains unknown fields: {sorted(unknown)}")
  if value.get("schema") != SUM_SLOT_SCHEMA:
    raise ValueError(f"{path}.schema must be {SUM_SLOT_SCHEMA}")
  tile_shape = _validate_tile_shape(value.get("tile_shape"), f"{path}.tile_shape")
  if tile_shape != {"M": 16, "N": 16}:
    raise ValueError(f"{path}.tile_shape must be {{'M': 16, 'N': 16}}")
  slots_per_thread = _validate_positive_int(value.get("slots_per_thread"), f"{path}.slots_per_thread")
  total_slots = _validate_positive_int(value.get("total_slots"), f"{path}.total_slots")
  mapping = value.get("mapping")
  if not isinstance(mapping, Sequence) or isinstance(mapping, (str, bytes, bytearray)):
    raise ValueError(f"{path}.mapping must be a sequence")
  if len(mapping) != total_slots:
    raise ValueError(f"{path}.mapping length must equal total_slots")
  seen_outputs: set[tuple[int, int]] = set()
  seen_slots: set[int] = set()
  out_mapping: list[dict[str, int]] = []
  for idx, entry in enumerate(mapping):
    if not isinstance(entry, Mapping):
      raise ValueError(f"{path}.mapping[{idx}] must be a dict")
    unknown_entry = set(entry) - set(SUM_SLOT_ENTRY_FIELDS)
    if unknown_entry:
      raise ValueError(f"{path}.mapping[{idx}] contains unknown fields: {sorted(unknown_entry)}")
    normalized = {field: _validate_non_negative_int(entry.get(field), f"{path}.mapping[{idx}].{field}") for field in SUM_SLOT_ENTRY_FIELDS}
    if normalized["m"] >= 16 or normalized["n"] >= 16:
      raise ValueError(f"{path}.mapping[{idx}] output index outside 16x16 tile")
    output = (normalized["m"], normalized["n"])
    if output in seen_outputs:
      raise ValueError(f"{path}.mapping duplicates output {output}")
    if normalized["slot"] in seen_slots:
      raise ValueError(f"{path}.mapping duplicates slot {normalized['slot']}")
    seen_outputs.add(output)
    seen_slots.add(normalized["slot"])
    out_mapping.append(normalized)
  if len(seen_outputs) != 16 * 16:
    raise ValueError(f"{path}.mapping must cover every 16x16 output")
  return {"schema": SUM_SLOT_SCHEMA, "tile_shape": tile_shape, "slots_per_thread": slots_per_thread,
          "total_slots": total_slots, "mapping": out_mapping}


def build_mmq_staging_evidence_bundle(
  *,
  candidate_id: str,
  backend: str,
  shape: Mapping[str, Any],
  q4k_tile_staging: Mapping[str, Any] | None = None,
  q8_1_ds4_staging: Mapping[str, Any] | None = None,
  sum_slot_map: Mapping[str, Any] | None = None,
  status: str | None = None,
  exact_blocker: str | None = None,
  production_dispatch_changed: bool = False,
  notes: str | None = None,
) -> dict[str, Any]:
  _validate_non_empty_string(candidate_id, "candidate_id")
  _validate_non_empty_string(backend, "backend")
  shape_out = _validate_shape(shape, "shape")
  if production_dispatch_changed is not False:
    raise ValueError("production_dispatch_changed must be False")

  q4k = _validate_staging(q4k_tile_staging or {
    "status": "present", "layout": Q4K_TILE_LOAD_LAYOUT, "tile_shape": {"N": 16, "K": Q4_K_BLOCK_ELEMS},
    "bytes_per_tile": 16 * Q4_K_BLOCK_BYTES, "block_elems": Q4_K_BLOCK_ELEMS, "source": "extra.qk.q4k_tile_loader",
  }, "q4k_tile_staging")
  q8 = _validate_staging(q8_1_ds4_staging or {
    "status": "present", "layout": Q8_1_MMQ_DS4_LAYOUT, "tile_shape": {"M": 16, "K": Q4_K_BLOCK_ELEMS},
    "bytes_per_tile": 16 * Q4_K_BLOCK_ELEMS, "block_elems": Q8_1_BLOCK_ELEMS, "panels": 2,
    "source": "extra.qk.mmq_q4k_q8_reference",
  }, "q8_1_ds4_staging")
  slots = _validate_sum_slot_map(sum_slot_map or build_bounded_16x16_sum_slot_map(), "sum_slot_map")

  inferred_status = "PASS"
  inferred_blocker = None
  if q4k["status"] != "present":
    inferred_status, inferred_blocker = "BLOCKED", "missing Q4_K tile staging data"
  elif q8["status"] != "present":
    inferred_status, inferred_blocker = "BLOCKED", "missing Q8_1 DS4 staging data"
  if status is None:
    status = inferred_status
  if status not in STATUSES:
    raise ValueError(f"status must be one of {STATUSES}")
  if exact_blocker is None:
    exact_blocker = inferred_blocker
  if status == "PASS" and exact_blocker is not None:
    raise ValueError("exact_blocker must be None when status is PASS")
  if status in ("FAIL", "BLOCKED"):
    _validate_non_empty_string(exact_blocker, "exact_blocker")
  if notes is not None:
    _validate_non_empty_string(notes, "notes")

  bundle: dict[str, Any] = {
    "schema": SCHEMA,
    "evidence_kind": "staging_sum_slots",
    "candidate_id": candidate_id,
    "backend": backend,
    "shape": shape_out,
    "q4k_tile_staging": q4k,
    "q8_1_ds4_staging": q8,
    "sum_slot_map": slots,
    "status": status,
    "exact_blocker": exact_blocker,
    "production_dispatch_changed": False,
  }
  if notes is not None:
    bundle["notes"] = notes
  return bundle


def validate_mmq_staging_evidence_bundle(bundle: Any) -> dict[str, Any]:
  if not isinstance(bundle, Mapping):
    raise ValueError("bundle must be a dict")
  unknown = set(bundle) - set(TOP_LEVEL_FIELDS)
  if unknown:
    raise ValueError(f"bundle contains unknown fields: {sorted(unknown)}")
  if bundle.get("schema") != SCHEMA:
    raise ValueError(f"schema must be {SCHEMA}")
  if bundle.get("production_dispatch_changed") is not False:
    raise ValueError("production_dispatch_changed must be False")
  required = ("evidence_kind", "candidate_id", "backend", "shape", "q4k_tile_staging", "q8_1_ds4_staging", "sum_slot_map", "status", "exact_blocker")
  missing = [field for field in required if field not in bundle]
  if missing:
    raise ValueError(f"bundle missing required fields: {missing}")
  if bundle["evidence_kind"] != "staging_sum_slots":
    raise ValueError("evidence_kind must be staging_sum_slots")
  return build_mmq_staging_evidence_bundle(
    candidate_id=bundle["candidate_id"],
    backend=bundle["backend"],
    shape=bundle["shape"],
    q4k_tile_staging=bundle["q4k_tile_staging"],
    q8_1_ds4_staging=bundle["q8_1_ds4_staging"],
    sum_slot_map=bundle["sum_slot_map"],
    status=bundle["status"],
    exact_blocker=bundle["exact_blocker"],
    production_dispatch_changed=bundle["production_dispatch_changed"],
    notes=bundle.get("notes"),
  )
