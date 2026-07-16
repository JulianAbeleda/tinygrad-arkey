from __future__ import annotations

import json, pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
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


_ROUTE_KIND_DECODE_FLASH = "decode_flash"
_ROUTE_KIND_Q4K_G3 = "decode_q4k_g3"
_ROUTE_KIND_Q6K_GEN = "decode_q6k_generated"
_ROUTE_KIND_PREFILL_GEN = "prefill_generated"
_ROUTE_POLICY_LOCAL = {
  "decode_flash_block_tile_g5_konly": {"kind": _ROUTE_KIND_DECODE_FLASH, "compat_params": ({"DECODE_LIVE_SPLIT": "1"},)},
  "decode_flash_live_split_g4_kvboth": {"kind": _ROUTE_KIND_DECODE_FLASH, "compat_params": ({"DECODE_LIVE_SPLIT": "1"},)},
  "decode_q4k_g3_generated": {"kind": _ROUTE_KIND_Q4K_G3, "compat_params": ({"BUBBLEBEAM_FUTURESIGHT": "1"},)},
  "decode_q6k_coop_generated": {"kind": _ROUTE_KIND_Q6K_GEN, "compat_params": ({"DECODE_Q6K_GENERATED": "1"},)},
  "prefill_q4k_int8_wmma_generated_research": {"kind": _ROUTE_KIND_PREFILL_GEN, "compat_params": ({},)},
  "prefill_q4k_int8_wmma_tiled_research": {"kind": _ROUTE_KIND_PREFILL_GEN, "compat_params": ({},)},
}
_MANIFEST_ALIAS_CACHE: dict[str, str]|None = None

def _route_aliases() -> dict[str, str]:
  global _MANIFEST_ALIAS_CACHE
  if _MANIFEST_ALIAS_CACHE is None:
    try:
      rows = route_ops.qk_route_manifest_attr("ROUTE_COMPATIBILITY_ALIASES")
      _MANIFEST_ALIAS_CACHE = {alias: row["canonical_route_id"] for row in rows for alias in row["compatibility_aliases"]}
    except Exception: _MANIFEST_ALIAS_CACHE = {}
  return _MANIFEST_ALIAS_CACHE

def _canonical_route_id(route_id:str) -> str:
  return _route_aliases().get(route_id, route_id)

def _freeze(value):
  if isinstance(value, dict): return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
  if isinstance(value, (list, tuple)): return tuple(_freeze(v) for v in value)
  return value

def _plain(value):
  if isinstance(value, Mapping): return {k: _plain(v) for k, v in value.items()}
  if isinstance(value, tuple): return tuple(_plain(v) for v in value)
  return value

@dataclass(frozen=True)
class RoutePolicyRow:
  """One immutable artifact row.  ``facts`` is empty for provenance-only legacy rows."""
  artifact: Mapping
  facts: Mapping
  incomplete_reason: str|None = None

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

def _canonical_row(policy_path:pathlib.Path, route_id:str, row:dict) -> RoutePolicyRow:
  artifact = _freeze(row)
  missing = [name for name in ("phase", "role", "quant") if not isinstance(row.get(name), str) or not row[name]]
  try: rows, cols = _route_shape_rows_cols(policy_path, route_id, row)
  except ValueError as exc:
    return RoutePolicyRow(artifact, MappingProxyType({}), str(exc))
  if missing:
    return RoutePolicyRow(artifact, MappingProxyType({}), f"incomplete policy facts: missing {', '.join(missing)}")
  identities = {name: _freeze(row[name]) for name in ("target", "capability", "candidate") if name in row}
  facts = {"phase": row["phase"], "role": row["role"], "quant": row["quant"],
           "shape": _freeze({"rows": rows, "cols": cols}), **identities}
  return RoutePolicyRow(artifact, MappingProxyType(facts))

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

def _integrated_loop_evidence_passes(row:dict) -> bool:
  """Require proof for the row's exact scanned/candidate target before admitting fused Q4 loops."""
  if row.get("implementation") != "integrated_loop": return True
  evidence = row.get("evidence")
  target = row.get("target")
  if not isinstance(evidence, dict) or not isinstance(target, dict) or not target or evidence.get("target") != target: return False
  candidate = row.get("candidate")
  if isinstance(candidate, dict) and candidate.get("target") != target: return False
  if evidence.get("real_device") is not True: return False
  if evidence.get("fallback_used") is not False: return False
  return all(isinstance(evidence.get(name), dict) and evidence[name].get("passed") is True
             for name in ("compile", "correctness", "instruction"))

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
    route_id = _canonical_route_id(artifact_route_id)
    row = dict(row)
    row["selected_route"] = route_id
    if artifact_route_id != route_id: row["legacy_selected_route"] = artifact_route_id
    # integrated_loop is a research implementation, never a policy default.
    # It becomes selectable only with a same-candidate real-AMD proof covering
    # compile, numeric correctness, instructions, and no fallback.
    if not _integrated_loop_evidence_passes(row): continue
    if route_id not in supported_route_ids:
      raise ValueError(f"{policy_path} selects unsupported route {route_id!r}; supported={sorted(supported_route_ids)}")
    params = dict(row.get("route_params", {}))
    _validate_route_params(policy_path, route_id, params, manifest_registry)
    route_kind = route_specs[route_id]["kind"]
    if route_kind == _ROUTE_KIND_DECODE_FLASH:
      selected[route_id] = RoutePolicyRow(_freeze(row), MappingProxyType({}), "legacy decode compatibility row")
    elif route_kind == _ROUTE_KIND_Q4K_G3:
      _route_shape_rows_cols(policy_path, route_id, row)
      frozen_row = _canonical_row(policy_path, route_id, row)
      q4k_g3_rows.append(frozen_row)
      selected.setdefault(route_id, frozen_row)
    elif route_kind == _ROUTE_KIND_Q6K_GEN:
      _route_shape_rows_cols(policy_path, route_id, row)
      frozen_row = _canonical_row(policy_path, route_id, row)
      q6k_gen_rows.append(frozen_row)
      selected.setdefault(route_id, frozen_row)
    elif route_kind == _ROUTE_KIND_PREFILL_GEN:
      _route_shape_rows_cols(policy_path, route_id, row)
      frozen_row = _canonical_row(policy_path, route_id, row)
      prefill_gen_rows.append(frozen_row)
      selected.setdefault(route_id, frozen_row)
    else:
      selected[route_id] = RoutePolicyRow(_freeze(row), MappingProxyType({}), "legacy compatibility row")
  provenance = _freeze({k: v for k, v in data.items() if k != "routes"})
  return ValidatedRoutePolicy(str(policy_path), provenance, MappingProxyType(selected), tuple(q4k_g3_rows),
                              tuple(q6k_gen_rows), tuple(prefill_gen_rows))

def set_qk_route_policy(policy:dict|None, strict:bool=False, debug:bool=False) -> None:
  """Deprecated compatibility no-op. Policy state is never stored module-globally."""
  return None

def has_qk_route_policy() -> bool: return False
def qk_route_policy_strict() -> bool: return False

def qk_route_policy_selected(route_id:str, shape:dict[str, int]|None=None, *, policy:ValidatedRoutePolicy|None=None) -> bool:
  route_id = _canonical_route_id(route_id)
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

def qk_route_policy_role_admission(route_id:str, *, phase:str, role:str, quant:str, shape:dict[str, int],
                                   target=None, capability=None, candidate=None,
                                   policy:ValidatedRoutePolicy|None=None) -> dict:
  """Bind complete canonical invocation facts to one immutable policy row."""
  route_id, rollback = _canonical_route_id(route_id), "direct_packed"
  result = {"admitted": False, "route": rollback, "role": role,
            "provenance": "rollback", "rollback_route": rollback,
            "coverage": "exact_structural", "errors": []}
  bound_policy = policy
  if bound_policy is None:
    result["errors"].append("no route policy loaded")
    return result
  rows = [row for row in bound_policy.get("prefill_gen", [])
          if row.get("selected_route") == route_id and row.get("role") == role]
  invocation = {"phase": phase, "role": role, "quant": quant, "shape": shape,
                **({"target": target} if target is not None else {}),
                **({"capability": capability} if capability is not None else {}),
                **({"candidate": candidate} if candidate is not None else {})}
  for row in rows:
    if row.incomplete_reason:
      result["errors"].append(row.incomplete_reason)
      continue
    facts = _plain(row.facts)
    if set(shape) != {"rows", "cols"}:
      result["errors"].append("partial or non-canonical invocation shape; expected exactly rows/cols")
      continue
    required = set(facts)
    if set(invocation) != required:
      result["errors"].append(f"incomplete invocation facts: expected {sorted(required)}, got {sorted(invocation)}")
      continue
    if _plain(invocation) != facts:
      result["errors"].append("canonical invocation facts do not exactly match policy row")
      continue
    rollback = str(row.get("rollback_route") or rollback)
    result.update({"admitted": True, "route": route_id, "provenance": str(row.get("provenance") or "policy_row"),
                   "rollback_route": rollback, "coverage": "exact_structural", "errors": []})
    return result
  if not result["errors"]: result["errors"].append("no matching selected exact-fact row")
  return result

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

def qk_route_policy_selects_prefill_generated(out_features:int, in_features:int, *, policy:ValidatedRoutePolicy|None=None) -> bool:
  if policy is None: return False
  for row in policy.get("prefill_gen", []):
    shape = row.get("shape", {})
    if "rows" in shape and "cols" in shape and int(shape["rows"]) == out_features and int(shape["cols"]) == in_features:
      return True
  return False

def validate_qk_route_policy_for_config(policy:dict|None, config, *, strict:bool=False, debug:bool=False) -> None:
  if policy is None: return
  expected = {"Hq": config.n_heads, "Hkv": config.n_kv_heads, "Hd": config.head_dim}
  for rid in ("decode_flash_block_tile_g5_konly", "decode_flash_live_split_g4_kvboth"):
    row = policy.get("selected", {}).get(rid)
    if row is None: continue
    shape = dict(row.get("shape", {}))
    mismatches = {k: (shape.get(k), v) for k, v in expected.items() if k in shape and int(shape[k]) != int(v)}
    if mismatches and strict:
      raise ValueError(f"QK_ROUTE_POLICY selects {rid} for incompatible model shape: {mismatches}")
    if debug:
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
_qk_route_policy_role_admission = qk_route_policy_role_admission
_validate_qk_route_policy_for_config = validate_qk_route_policy_for_config
