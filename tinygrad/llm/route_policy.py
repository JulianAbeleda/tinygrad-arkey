from __future__ import annotations

import json, pathlib
from tinygrad import UOp, getenv
from tinygrad.llm import route_ops


def qk_policy_value(entry:dict) -> dict:
  cand = entry.get("candidate") or {}
  return {
    "winner": entry.get("winner"), "parts": int(cand.get("parts", 0)),
    "opts": tuple(cand.get("opts", ())), "family": cand.get("family", ""),
    "reduction": cand.get("reduction", ""),
    "policy_reason": entry.get("policy_reason", ""), "storage": entry.get("storage", {}),
  }

def load_qk_generated_policy(path:str) -> dict:
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("kind") != "qk_generated_policy": raise ValueError(f"{policy_path} is not a QK generated policy cache")
  if data.get("generator_version") not in (0, 1):
    raise ValueError(f"{policy_path} has unsupported generator_version={data.get('generator_version')}")
  by_shape: dict[tuple[int, int, int], dict] = {}
  by_tensor: dict[tuple[str, int, int, int], dict] = {}
  for entry in data.get("entries", []):
    desc = entry.get("descriptor", {})
    key = (int(desc["ggml_type"]), int(desc["rows"]), int(desc["cols"]))
    value = qk_policy_value(entry)
    if entry.get("scope") == "tensor":
      tensor = str(desc.get("tensor", ""))
      if not tensor: raise ValueError(f"{policy_path} has tensor-scoped entry without descriptor.tensor")
      tensor_key = (tensor, *key)
      if tensor_key in by_tensor and by_tensor[tensor_key] != value:
        raise ValueError(f"{policy_path} has conflicting tensor generated policy entries for key={tensor_key}: "
                         f"{by_tensor[tensor_key]} vs {value}")
      by_tensor[tensor_key] = value
    else:
      if key in by_shape and by_shape[key] != value:
        raise ValueError(f"{policy_path} has conflicting generated policy entries for key={key}: {by_shape[key]} vs {value}")
      by_shape[key] = value
  if not by_shape and not by_tensor: raise ValueError(f"{policy_path} contains no generated policy entries")
  return {"by_shape": by_shape, "by_tensor": by_tensor}

def qk_generated_policy_len(policy:dict|None) -> int:
  if policy is None: return 0
  return len(policy.get("by_shape", {})) + len(policy.get("by_tensor", {}))

def qk_generated_policy_entry(policy:dict|None, typ:int, rows:int, cols:int, name:str|None=None) -> dict|None:
  if policy is None: return None
  if name is not None and (entry:=policy.get("by_tensor", {}).get((name, typ, rows, cols))) is not None: return entry
  return policy.get("by_shape", {}).get((typ, rows, cols))


_QK_ROUTE_POLICY: dict|None = None
_QK_ROUTE_POLICY_STRICT = False
_QK_ROUTE_POLICY_DEBUG = False
_ROUTE_KIND_DECODE_FLASH = "decode_flash"
_ROUTE_KIND_Q4K_G3 = "decode_q4k_g3"
_ROUTE_KIND_Q6K_GEN = "decode_q6k_generated"
_ROUTE_KIND_PREFILL_GEN = "prefill_generated"
_ROUTE_KIND_PREFILL_MMQ_ATOM = "prefill_mmq_atom"
_PREFILL_Q4K_MMQ_ATOM_ROUTE = "prefill_14b_q4k_q8_1_hybrid_mmq_atom"
_ROUTE_POLICY_LOCAL = {
  "decode_flash_block_tile_g5_konly": {"kind": _ROUTE_KIND_DECODE_FLASH, "compat_params": ({"DECODE_LIVE_SPLIT": "1"},)},
  "decode_flash_live_split_g4_8b_kvboth": {"kind": _ROUTE_KIND_DECODE_FLASH, "compat_params": ({"DECODE_LIVE_SPLIT": "1"},)},
  "decode_q4k_g3_generated": {"kind": _ROUTE_KIND_Q4K_G3, "compat_params": ({"BUBBLEBEAM_FUTURESIGHT": "1"},)},
  "decode_q6k_coop_generated": {"kind": _ROUTE_KIND_Q6K_GEN, "compat_params": ({"DECODE_Q6K_GENERATED": "1"},)},
  "prefill_q4k_int8_wmma_generated_research": {"kind": _ROUTE_KIND_PREFILL_GEN, "compat_params": ({},)},
  "prefill_q4k_int8_wmma_tiled_research": {"kind": _ROUTE_KIND_PREFILL_GEN, "compat_params": ({},)},
  _PREFILL_Q4K_MMQ_ATOM_ROUTE: {"kind": _ROUTE_KIND_PREFILL_MMQ_ATOM, "compat_params": ()},
}
_MANIFEST_ROUTE_CACHE: dict[str, dict]|None = None

def _manifest_routes() -> dict[str, dict]:
  global _MANIFEST_ROUTE_CACHE
  if _MANIFEST_ROUTE_CACHE is None:
    try:
      routes = route_ops.qk_route_manifest_attr("ROUTES")
      _MANIFEST_ROUTE_CACHE = {rid: {"env": dict(row.get("env", {})), "status": row.get("status")}
                               for rid, row in routes.items()
                               if row.get("workload") in ("decode", "prefill")}
    except Exception:
      _MANIFEST_ROUTE_CACHE = {}
  return _MANIFEST_ROUTE_CACHE

def _qk_route_specs() -> dict[str, dict]:
  specs = {rid: dict(spec) for rid, spec in _ROUTE_POLICY_LOCAL.items()}
  for rid, manifest in _manifest_routes().items():
    if manifest.get("status") not in ("promoted_default", "default_shipped"): continue
    kind = specs.get(rid, {}).get("kind", _ROUTE_KIND_PREFILL_GEN if rid.startswith("prefill_") else _ROUTE_KIND_DECODE_FLASH)
    specs.setdefault(rid, {})["kind"] = kind
  for rid, manifest in _manifest_routes().items():
    if rid in specs: specs[rid]["manifest_env"] = dict(manifest.get("env", {}))
  return specs

def _supported_qk_route_ids() -> set[str]:
  return set(_qk_route_specs())

class _LazySupportedQKRouteIds:
  def __iter__(self): return iter(_supported_qk_route_ids())
  def __contains__(self, route_id): return route_id in _supported_qk_route_ids()
  def __len__(self): return len(_supported_qk_route_ids())

_SUPPORTED_QK_ROUTE_IDS = _LazySupportedQKRouteIds()

def _route_policy_params_allowed(route_id:str) -> set[tuple[tuple[str, str], ...]]:
  spec = _qk_route_specs()[route_id]
  allowed = {tuple(sorted((str(k), str(v)) for k, v in spec.get("manifest_env", {}).items()))}
  allowed.update(tuple(sorted((str(k), str(v)) for k, v in p.items())) for p in spec.get("compat_params", ()))
  return allowed

def _validate_route_params(policy_path:pathlib.Path, route_id:str, params:dict) -> None:
  actual = tuple(sorted((str(k), str(v)) for k, v in params.items()))
  allowed = _route_policy_params_allowed(route_id)
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

def _validate_prefill_mmq_atom_row(policy_path:pathlib.Path, route_id:str, row:dict) -> None:
  rows_i, cols_i = _route_shape_rows_cols(policy_path, route_id, row)
  role = str(row.get("role", ""))
  quant = str(row.get("quant", ""))
  if route_id == _PREFILL_Q4K_MMQ_ATOM_ROUTE:
    if not bool(row.get("atom_available", False)):
      raise ValueError(f"{policy_path} route {route_id!r} is fail-closed until atom_available=true")
    if (role, rows_i, cols_i, quant) != ("ffn_gate_up", 17408, 5120, "Q4_K"):
      raise ValueError(f"{policy_path} route {route_id!r} only supports role='ffn_gate_up', "
                       f"shape rows=17408 cols=5120, quant='Q4_K'; got role={role!r}, "
                       f"rows={rows_i}, cols={cols_i}, quant={quant!r}")

def load_qk_route_policy(path:str) -> dict:
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1":
    raise ValueError(f"{policy_path} is not a boltbeam.route_policy.v1 route policy")
  selected: dict[str, dict] = {}
  q4k_g3_rows: list[dict] = []
  q6k_gen_rows: list[dict] = []
  prefill_gen_rows: list[dict] = []
  prefill_mmq_atom_rows: list[dict] = []
  route_specs = _qk_route_specs()
  supported_route_ids = set(route_specs)
  for row in data.get("routes", []):
    route_id = row.get("selected_route")
    if not route_id: continue
    if route_id not in supported_route_ids:
      raise ValueError(f"{policy_path} selects unsupported route {route_id!r}; supported={sorted(supported_route_ids)}")
    params = dict(row.get("route_params", {}))
    _validate_route_params(policy_path, route_id, params)
    route_kind = route_specs[route_id]["kind"]
    if route_kind == _ROUTE_KIND_DECODE_FLASH:
      selected[route_id] = row
    elif route_kind == _ROUTE_KIND_Q4K_G3:
      _route_shape_rows_cols(policy_path, route_id, row)
      q4k_g3_rows.append(row)
      selected.setdefault(route_id, row)
    elif route_kind == _ROUTE_KIND_Q6K_GEN:
      _route_shape_rows_cols(policy_path, route_id, row)
      q6k_gen_rows.append(row)
      selected.setdefault(route_id, row)
    elif route_kind == _ROUTE_KIND_PREFILL_GEN:
      _route_shape_rows_cols(policy_path, route_id, row)
      prefill_gen_rows.append(row)
      selected.setdefault(route_id, row)
    elif route_kind == _ROUTE_KIND_PREFILL_MMQ_ATOM:
      _validate_prefill_mmq_atom_row(policy_path, route_id, row)
      prefill_mmq_atom_rows.append(row)
      selected.setdefault(route_id, row)
    else:
      selected[route_id] = row
  return {"path": str(policy_path), "selected": selected, "q4k_g3": q4k_g3_rows, "q6k_gen": q6k_gen_rows,
          "prefill_gen": prefill_gen_rows, "prefill_mmq_atom": prefill_mmq_atom_rows}

def set_qk_route_policy(policy:dict|None, strict:bool=False, debug:bool=False) -> None:
  global _QK_ROUTE_POLICY, _QK_ROUTE_POLICY_STRICT, _QK_ROUTE_POLICY_DEBUG
  _QK_ROUTE_POLICY, _QK_ROUTE_POLICY_STRICT, _QK_ROUTE_POLICY_DEBUG = policy, strict, debug

def has_qk_route_policy() -> bool: return _QK_ROUTE_POLICY is not None
def qk_route_policy_strict() -> bool: return _QK_ROUTE_POLICY_STRICT

def qk_route_policy_selected(route_id:str, shape:dict[str, int]|None=None) -> bool:
  if _QK_ROUTE_POLICY is None: return False
  row = _QK_ROUTE_POLICY.get("selected", {}).get(route_id)
  if row is None: return False
  if shape is not None:
    rows = [cand for cand in _QK_ROUTE_POLICY.get("prefill_gen", []) if cand.get("selected_route") == route_id]
    if not rows: rows = [row]
    for cand in rows:
      policy_shape = dict(cand.get("shape", {}))
      if all(k not in policy_shape or int(policy_shape[k]) == int(v) for k, v in shape.items()): return True
    return False
  return True

def qk_route_policy_selects_q4k_g3(out_features:int, in_features:int) -> bool:
  if _QK_ROUTE_POLICY is None: return False
  for row in _QK_ROUTE_POLICY.get("q4k_g3", []):
    shape = row.get("shape", {})
    if "rows" in shape and "cols" in shape and int(shape["rows"]) == out_features and int(shape["cols"]) == in_features:
      return True
  return False

def qk_route_policy_selects_q6k_generated(out_features:int, in_features:int) -> bool:
  if _QK_ROUTE_POLICY is None: return False
  for row in _QK_ROUTE_POLICY.get("q6k_gen", []):
    shape = row.get("shape", {})
    if "rows" in shape and "cols" in shape and int(shape["rows"]) == out_features and int(shape["cols"]) == in_features:
      return True
  return False

def qk_route_policy_selects_prefill_generated(out_features:int, in_features:int) -> bool:
  if _QK_ROUTE_POLICY is None: return False
  for row in _QK_ROUTE_POLICY.get("prefill_gen", []):
    shape = row.get("shape", {})
    if "rows" in shape and "cols" in shape and int(shape["rows"]) == out_features and int(shape["cols"]) == in_features:
      return True
  return False

def validate_qk_route_policy_for_config(policy:dict|None, config) -> None:
  if policy is None: return
  expected = {"Hq": config.n_heads, "Hkv": config.n_kv_heads, "Hd": config.head_dim}
  for rid in ("decode_flash_block_tile_g5_konly", "decode_flash_live_split_g4_8b_kvboth"):
    row = policy.get("selected", {}).get(rid)
    if row is None: continue
    shape = dict(row.get("shape", {}))
    mismatches = {k: (shape.get(k), v) for k, v in expected.items() if k in shape and int(shape[k]) != int(v)}
    if mismatches and _QK_ROUTE_POLICY_STRICT:
      raise ValueError(f"QK_ROUTE_POLICY selects {rid} for incompatible model shape: {mismatches}")
    if _QK_ROUTE_POLICY_DEBUG:
      print(f"QK_ROUTE_POLICY_DEBUG path={policy.get('path')} route={rid} selected={sorted(policy.get('selected', {}))} "
            f"shape={shape} model={expected} compatible={not mismatches}")

def should_use_flash_decode(start_pos, T, use_flash:bool=False, getenv_fn=getenv) -> bool:
  if not (isinstance(start_pos, UOp) and isinstance(T, int) and T == 1): return False
  mode = str(getenv_fn("FLASH_DECODE", "auto")).lower()
  if mode in ("0", "false", "off"): return False
  if use_flash or mode in ("1", "true", "on"): return True
  if mode != "auto": return False
  try: ctx = start_pos.unbind()[1] + T
  except Exception: return False
  return ctx >= getenv_fn("FLASH_DECODE_THRESHOLD", 512)

_qk_policy_value = qk_policy_value
_load_qk_generated_policy = load_qk_generated_policy
_qk_generated_policy_len = qk_generated_policy_len
_qk_generated_policy_entry = qk_generated_policy_entry
_load_qk_route_policy = load_qk_route_policy
_set_qk_route_policy = set_qk_route_policy
_qk_route_policy_selected = qk_route_policy_selected
_qk_route_policy_selects_q4k_g3 = qk_route_policy_selects_q4k_g3
_qk_route_policy_selects_q6k_generated = qk_route_policy_selects_q6k_generated
_qk_route_policy_selects_prefill_generated = qk_route_policy_selects_prefill_generated
_validate_qk_route_policy_for_config = validate_qk_route_policy_for_config
