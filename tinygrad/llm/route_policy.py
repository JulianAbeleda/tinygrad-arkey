from __future__ import annotations

import json, pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from tinygrad import UOp, getenv


def qk_generated_policy_entry(policy:dict|None, typ:int, rows:int, cols:int, name:str|None=None) -> dict|None:
  if policy is None: return None
  if name is not None and (entry:=policy.get("by_tensor", {}).get((name, typ, rows, cols))) is not None: return entry
  return policy.get("by_shape", {}).get((typ, rows, cols))


_ROUTE_KIND_DECODE_FLASH = "decode_flash"
_ROUTE_KIND_Q4K_G3 = "decode_q4k_g3"
_ROUTE_KIND_Q6K_GEN = "decode_q6k_generated"
_ROUTE_KIND_PREFILL_GEN = "prefill_generated"
_ROUTE_POLICY_LOCAL = {
  "decode_flash_block_tile_g5_konly": {"kind": _ROUTE_KIND_DECODE_FLASH, "compat_params": ({"DECODE_LIVE_SPLIT": "1"},)},
  "decode_flash_live_split_g4_kvboth": {"kind": _ROUTE_KIND_DECODE_FLASH, "compat_params": ({"DECODE_LIVE_SPLIT": "1"},)},
  "decode_q4k_g3_generated": {"kind": _ROUTE_KIND_Q4K_G3, "compat_params": ({"BUBBLEBEAM_FUTURESIGHT": "1"},)},
  "decode_q6k_coop_generated": {"kind": _ROUTE_KIND_Q6K_GEN, "compat_params": ({"DECODE_Q6K_GENERATED": "1"},)},
}

def _freeze(value):
  if isinstance(value, dict): return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
  if isinstance(value, (list, tuple)): return tuple(_freeze(v) for v in value)
  return value

@dataclass(frozen=True)
class RoutePolicyRow:
  artifact: Mapping

  def get(self, key, default=None): return self.artifact.get(key, default)
  def __getitem__(self, key): return self.artifact[key]

@dataclass(frozen=True)
class ValidatedRoutePolicy(Mapping):
  path: str
  provenance: Mapping
  selected: Mapping
  q4k_g3: tuple[RoutePolicyRow, ...]
  q6k_gen: tuple[RoutePolicyRow, ...]
  prefill_gen: tuple[RoutePolicyRow, ...]

  def __getitem__(self, key):
    if key not in ("path", "provenance", "selected", "q4k_g3", "q6k_gen", "prefill_gen"): raise KeyError(key)
    return getattr(self, key)
  def __iter__(self): return iter(("path", "provenance", "selected", "q4k_g3", "q6k_gen", "prefill_gen"))
  def __len__(self): return 6

def _manifest_routes(manifest_registry:Mapping|None = None) -> dict[str, dict]:
  """Return caller-owned manifest facts; no manifest global is semantic authority."""
  if manifest_registry is None: return {}
  return {rid: {"env": dict(row.get("env", {})), "status": row.get("status")}
          for rid, row in manifest_registry.items() if row.get("workload") in ("decode", "prefill")}

def _qk_route_specs(manifest_registry:Mapping|None = None) -> dict[str, dict]:
  specs = {rid: dict(spec) for rid, spec in _ROUTE_POLICY_LOCAL.items()}
  for rid, manifest in _manifest_routes(manifest_registry).items():
    if manifest.get("status") not in ("promoted_default", "default_shipped"): continue
    kind = specs.get(rid, {}).get("kind", _ROUTE_KIND_PREFILL_GEN if rid.startswith("prefill_") else _ROUTE_KIND_DECODE_FLASH)
    specs.setdefault(rid, {})["kind"] = kind
  for rid, manifest in _manifest_routes(manifest_registry).items():
    if rid in specs: specs[rid]["manifest_env"] = dict(manifest.get("env", {}))
  return specs

def _supported_qk_route_ids(manifest_registry:Mapping|None = None) -> set[str]:
  return set(_qk_route_specs(manifest_registry))

class _LazySupportedQKRouteIds:
  def __iter__(self): return iter(_supported_qk_route_ids())
  def __contains__(self, route_id): return route_id in _supported_qk_route_ids()
  def __len__(self): return len(_supported_qk_route_ids())

_SUPPORTED_QK_ROUTE_IDS = _LazySupportedQKRouteIds()

def _route_policy_params_allowed(route_id:str, manifest_registry:Mapping|None = None) -> set[tuple[tuple[str, str], ...]]:
  spec = _qk_route_specs(manifest_registry)[route_id]
  allowed = {tuple(sorted((str(k), str(v)) for k, v in spec.get("manifest_env", {}).items()))}
  allowed.update(tuple(sorted((str(k), str(v)) for k, v in p.items())) for p in spec.get("compat_params", ()))
  return allowed

def _validate_route_params(policy_path:pathlib.Path, route_id:str, params:dict, manifest_registry:Mapping|None = None) -> None:
  actual = tuple(sorted((str(k), str(v)) for k, v in params.items()))
  allowed = _route_policy_params_allowed(route_id, manifest_registry)
  allowed_keys = {k for item in allowed for k, _ in item}
  if set(params) - allowed_keys:
    raise ValueError(f"{policy_path} route {route_id!r} has unsupported params {sorted(set(params)-allowed_keys)}")
  if actual not in allowed:
    expected = [dict(item) for item in sorted(allowed)]
    raise ValueError(f"{policy_path} route {route_id!r} route_params must match manifest env/compat params {expected}, got {params}")

def _route_shape_rows_cols(policy_path:pathlib.Path, route_id:str, row:dict) -> tuple[int, int]:
  shape = row.get("shape", {})
  try:
    rows_i, cols_i = int(shape["rows"]), int(shape["cols"])
  except (KeyError, TypeError, ValueError):
    raise ValueError(f"{policy_path} route {route_id!r} has malformed shape {shape!r}; expected integer rows/cols")
  if rows_i <= 0 or cols_i <= 0:
    raise ValueError(f"{policy_path} route {route_id!r} has non-positive shape rows={rows_i} cols={cols_i}")
  return rows_i, cols_i

def load_qk_route_policy(path:str, *, manifest_registry:Mapping|None = None) -> ValidatedRoutePolicy:
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1":
    raise ValueError(f"{policy_path} is not a boltbeam.route_policy.v1 route policy")
  selected: dict[str, dict] = {}
  q4k_g3_rows: list[dict] = []
  q6k_gen_rows: list[dict] = []
  prefill_gen_rows: list[dict] = []
  route_specs = _qk_route_specs(manifest_registry)
  supported_route_ids = set(route_specs)
  for row in data.get("routes", []):
    artifact_route_id = row.get("selected_route")
    if not artifact_route_id: continue
    route_id = artifact_route_id
    row = dict(row)
    if route_id not in supported_route_ids:
      raise ValueError(f"{policy_path} selects unsupported route {route_id!r}; supported={sorted(supported_route_ids)}")
    params = dict(row.get("route_params", {}))
    _validate_route_params(policy_path, route_id, params, manifest_registry)
    route_kind = route_specs[route_id]["kind"]
    if route_kind == _ROUTE_KIND_DECODE_FLASH:
      selected[route_id] = RoutePolicyRow(_freeze(row))
    elif route_kind == _ROUTE_KIND_Q4K_G3:
      _route_shape_rows_cols(policy_path, route_id, row)
      frozen_row = RoutePolicyRow(_freeze(row))
      q4k_g3_rows.append(frozen_row)
      selected.setdefault(route_id, frozen_row)
    elif route_kind == _ROUTE_KIND_Q6K_GEN:
      _route_shape_rows_cols(policy_path, route_id, row)
      frozen_row = RoutePolicyRow(_freeze(row))
      q6k_gen_rows.append(frozen_row)
      selected.setdefault(route_id, frozen_row)
    elif route_kind == _ROUTE_KIND_PREFILL_GEN:
      _route_shape_rows_cols(policy_path, route_id, row)
      frozen_row = RoutePolicyRow(_freeze(row))
      prefill_gen_rows.append(frozen_row)
      selected.setdefault(route_id, frozen_row)
    else:
      selected[route_id] = RoutePolicyRow(_freeze(row))
  provenance = _freeze({k: v for k, v in data.items() if k != "routes"})
  return ValidatedRoutePolicy(str(policy_path), provenance, MappingProxyType(selected), tuple(q4k_g3_rows),
                              tuple(q6k_gen_rows), tuple(prefill_gen_rows))

def qk_route_policy_selected(route_id:str, shape:dict[str, int]|None=None, *, policy:ValidatedRoutePolicy|None=None) -> bool:
  if policy is None: return False
  row = policy.get("selected", {}).get(route_id)
  if row is None: return False
  if shape is not None:
    rows = [cand for cand in policy.get("prefill_gen", []) if cand.get("selected_route") == route_id]
    if not rows: rows = [row]
    for cand in rows:
      policy_shape = dict(cand.get("shape", {}))
      if set(policy_shape) == set(shape) and all(int(policy_shape[k]) == int(v) for k, v in shape.items()): return True
    return False
  return True

def qk_route_policy_selects_q4k_g3(out_features:int, in_features:int, *, policy:ValidatedRoutePolicy|None=None) -> bool:
  if policy is None: return False
  for row in policy.get("q4k_g3", []):
    shape = row.get("shape", {})
    if "rows" in shape and "cols" in shape and int(shape["rows"]) == out_features and int(shape["cols"]) == in_features:
      return True
  return False

def qk_route_policy_selects_q6k_generated(out_features:int, in_features:int, *, policy:ValidatedRoutePolicy|None=None) -> bool:
  if policy is None: return False
  for row in policy.get("q6k_gen", []):
    shape = row.get("shape", {})
    if "rows" in shape and "cols" in shape and int(shape["rows"]) == out_features and int(shape["cols"]) == in_features:
      return True
  return False

def should_use_flash_decode(start_pos, T, use_flash:bool=False, getenv_fn=getenv) -> bool:
  if not (isinstance(start_pos, UOp) and isinstance(T, int) and T == 1): return False
  mode = str(getenv_fn("FLASH_DECODE", "auto")).lower()
  if mode in ("0", "false", "off"): return False
  if use_flash or mode in ("1", "true", "on"): return True
  if mode != "auto": return False
  try: ctx = start_pos.unbind()[1] + T
  except Exception: return False
  return ctx >= getenv_fn("FLASH_DECODE_THRESHOLD", 512)

_qk_generated_policy_entry = qk_generated_policy_entry
_load_qk_route_policy = load_qk_route_policy
_qk_route_policy_selected = qk_route_policy_selected
_qk_route_policy_selects_q4k_g3 = qk_route_policy_selects_q4k_g3
_qk_route_policy_selects_q6k_generated = qk_route_policy_selects_q6k_generated
