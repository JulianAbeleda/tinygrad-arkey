from __future__ import annotations

import json, pathlib
from tinygrad import UOp, getenv


def q4k_policy(name:str) -> tuple[int, tuple[str, ...]]|None:
  if ".ffn_gate.weight" in name or ".ffn_up.weight" in name: return 1, ("LOCAL:0:64",)
  if ".ffn_down.weight" in name: return 4, ("LOCAL:0:32",)
  if ".attn_q.weight" in name or ".attn_output.weight" in name: return 1, ("LOCAL:0:64",)
  if ".attn_k.weight" in name and getenv("DECODE_ROUTE_ATTN_K", 1): return 1, ("LOCAL:0:64",)
  # PROMOTED default-ON 2026-07-03 (rollback DECODE_ROUTE_ATTN_V=0): the Q4_K attn_v tensors were omitted here, so
  # they fell to the generic nn.Linear lazy-dequant GEMV at 2.8% of peak (24 GB/s) = 12.5% of 14B decode -- the same
  # route-miss class as attn_k, one tensor over. attn_v is 5120->1024 (same shape as attn_k) so it takes the same
  # primitive route (q4k_g3_lanemap_gemv, ~35% peak). Byte-identical; W==D token-identical; 14B ctx512 +13.3%, 8B
  # +8.7% (no regression -> global default-on). Found by the 2026-07-03 in-context lifecycle profile.
  if ".attn_v.weight" in name and getenv("DECODE_ROUTE_ATTN_V", 1): return 1, ("LOCAL:0:64",)
  return None

def q6k_policy(name:str) -> tuple[int, tuple[str, ...]]|None:
  if ".ffn_down.weight" in name: return 1, ("LOCAL:0:64",)
  if getenv("Q6K_COVER_MORE", 1):
    if ".attn_v.weight" in name: return 4, ("LOCAL:0:32",)
    if name == "output.weight": return 1, ("LOCAL:0:64",)
  return None

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
_SUPPORTED_QK_ROUTE_IDS = {"decode_flash_block_tile_g5_konly", "decode_flash_live_split_g4_8b_kvboth",
                           "decode_q4k_g3_generated", "decode_q6k_coop_generated",
                           "prefill_pipe_role_selective_generated"}

def load_qk_route_policy(path:str) -> dict:
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1":
    raise ValueError(f"{policy_path} is not a boltbeam.route_policy.v1 route policy")
  selected: dict[str, dict] = {}
  q4k_g3_rows: list[dict] = []
  q6k_gen_rows: list[dict] = []
  prefill_gen_rows: list[dict] = []
  for row in data.get("routes", []):
    route_id = row.get("selected_route")
    if not route_id: continue
    if route_id not in _SUPPORTED_QK_ROUTE_IDS:
      raise ValueError(f"{policy_path} selects unsupported route {route_id!r}; supported={sorted(_SUPPORTED_QK_ROUTE_IDS)}")
    params = dict(row.get("route_params", {}))
    if route_id == "decode_flash_block_tile_g5_konly":
      allowed = {"DECODE_LIVE_SPLIT"}
      if set(params) - allowed:
        raise ValueError(f"{policy_path} route {route_id!r} has unsupported params {sorted(set(params)-allowed)}")
      if params and params != {"DECODE_LIVE_SPLIT": "1"}:
        raise ValueError(f"{policy_path} route {route_id!r} must select the generated live-split route, got {params}")
      selected[route_id] = row
    elif route_id == "decode_flash_live_split_g4_8b_kvboth":
      allowed = {"DECODE_LIVE_SPLIT"}
      if set(params) - allowed:
        raise ValueError(f"{policy_path} route {route_id!r} has unsupported params {sorted(set(params)-allowed)}")
      if params and params != {"DECODE_LIVE_SPLIT": "1"}:
        raise ValueError(f"{policy_path} route {route_id!r} must select the generated 8B live-split route, got {params}")
      selected[route_id] = row
    elif route_id == "decode_q4k_g3_generated":
      allowed = {"BUBBLEBEAM_FUTURESIGHT"}
      if set(params) - allowed:
        raise ValueError(f"{policy_path} route {route_id!r} has unsupported params {sorted(set(params)-allowed)}")
      if params and params != {"BUBBLEBEAM_FUTURESIGHT": "1"}:
        raise ValueError(f"{policy_path} route {route_id!r} must select the generated G3 route (BUBBLEBEAM_FUTURESIGHT=1), got {params}")
      shape = row.get("shape", {})
      try:
        rows_i, cols_i = int(shape["rows"]), int(shape["cols"])
      except (KeyError, TypeError, ValueError):
        raise ValueError(f"{policy_path} route {route_id!r} has malformed shape {shape!r}; expected integer rows/cols")
      if rows_i <= 0 or cols_i <= 0:
        raise ValueError(f"{policy_path} route {route_id!r} has non-positive shape rows={rows_i} cols={cols_i}")
      q4k_g3_rows.append(row)
      selected.setdefault(route_id, row)
    elif route_id == "decode_q6k_coop_generated":
      allowed = {"DECODE_Q6K_GENERATED"}
      if set(params) - allowed:
        raise ValueError(f"{policy_path} route {route_id!r} has unsupported params {sorted(set(params)-allowed)}")
      if params and params != {"DECODE_Q6K_GENERATED": "1"}:
        raise ValueError(f"{policy_path} route {route_id!r} must select the generated Q6_K route (DECODE_Q6K_GENERATED=1), got {params}")
      shape = row.get("shape", {})
      try:
        rows_i, cols_i = int(shape["rows"]), int(shape["cols"])
      except (KeyError, TypeError, ValueError):
        raise ValueError(f"{policy_path} route {route_id!r} has malformed shape {shape!r}; expected integer rows/cols")
      if rows_i <= 0 or cols_i <= 0:
        raise ValueError(f"{policy_path} route {route_id!r} has non-positive shape rows={rows_i} cols={cols_i}")
      q6k_gen_rows.append(row)
      selected.setdefault(route_id, row)
    elif route_id == "prefill_pipe_role_selective_generated":
      allowed = {"PREFILL_GENERATED_SCHEDULE"}
      if set(params) - allowed:
        raise ValueError(f"{policy_path} route {route_id!r} has unsupported params {sorted(set(params)-allowed)}")
      if params and params != {"PREFILL_GENERATED_SCHEDULE": "1"}:
        raise ValueError(f"{policy_path} route {route_id!r} must select the generated prefill schedule (PREFILL_GENERATED_SCHEDULE=1), got {params}")
      shape = row.get("shape", {})
      try:
        rows_i, cols_i = int(shape["rows"]), int(shape["cols"])
      except (KeyError, TypeError, ValueError):
        raise ValueError(f"{policy_path} route {route_id!r} has malformed shape {shape!r}; expected integer rows/cols")
      if rows_i <= 0 or cols_i <= 0:
        raise ValueError(f"{policy_path} route {route_id!r} has non-positive shape rows={rows_i} cols={cols_i}")
      prefill_gen_rows.append(row)
      selected.setdefault(route_id, row)
    else:
      selected[route_id] = row
  return {"path": str(policy_path), "selected": selected, "q4k_g3": q4k_g3_rows, "q6k_gen": q6k_gen_rows,
          "prefill_gen": prefill_gen_rows}

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
    policy_shape = dict(row.get("shape", {}))
    for k, v in shape.items():
      if k in policy_shape and int(policy_shape[k]) != int(v): return False
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

# Compatibility aliases used by existing tests and callers during the extraction.
_q4k_policy = q4k_policy
_q6k_policy = q6k_policy
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
