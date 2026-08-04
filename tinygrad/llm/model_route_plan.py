from __future__ import annotations
import json, pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable
from tinygrad.llm.model_facts import normalize_route_role

# Serialized route-policy compatibility API.  This consumes only caller-supplied
# manifest facts; static model planning below remains independent of it.
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
  path: str; provenance: Mapping; selected: Mapping
  q4k_g3: tuple[RoutePolicyRow, ...]; q6k_gen: tuple[RoutePolicyRow, ...]; prefill_gen: tuple[RoutePolicyRow, ...]
  def __getitem__(self, key):
    if key not in ("path", "provenance", "selected", "q4k_g3", "q6k_gen", "prefill_gen"): raise KeyError(key)
    return getattr(self, key)
  def __iter__(self): return iter(("path", "provenance", "selected", "q4k_g3", "q6k_gen", "prefill_gen"))
  def __len__(self): return 6

def _manifest_routes(manifest_registry:Mapping|None = None) -> dict[str, dict]:
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

def _supported_qk_route_ids(manifest_registry:Mapping|None = None) -> set[str]: return set(_qk_route_specs(manifest_registry))
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
  actual, allowed = tuple(sorted((str(k), str(v)) for k, v in params.items())), _route_policy_params_allowed(route_id, manifest_registry)
  allowed_keys = {k for item in allowed for k, _ in item}
  if set(params) - allowed_keys: raise ValueError(f"{policy_path} route {route_id!r} has unsupported params {sorted(set(params)-allowed_keys)}")
  if actual not in allowed:
    raise ValueError(f"{policy_path} route {route_id!r} route_params must match manifest env/compat params {[dict(item) for item in sorted(allowed)]}, got {params}")

def _route_shape_rows_cols(policy_path:pathlib.Path, route_id:str, row:dict) -> tuple[int, int]:
  shape = row.get("shape", {})
  try: rows_i, cols_i = int(shape["rows"]), int(shape["cols"])
  except (KeyError, TypeError, ValueError): raise ValueError(f"{policy_path} route {route_id!r} has malformed shape {shape!r}; expected integer rows/cols")
  if rows_i <= 0 or cols_i <= 0: raise ValueError(f"{policy_path} route {route_id!r} has non-positive shape rows={rows_i} cols={cols_i}")
  return rows_i, cols_i

def load_qk_route_policy(path:str, *, manifest_registry:Mapping|None = None) -> ValidatedRoutePolicy:
  policy_path, data = pathlib.Path(path).expanduser(), None
  data = json.loads(policy_path.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1": raise ValueError(f"{policy_path} is not a boltbeam.route_policy.v1 route policy")
  selected, q4k_g3_rows, q6k_gen_rows, prefill_gen_rows = {}, [], [], []
  route_specs = _qk_route_specs(manifest_registry)
  for row in data.get("routes", []):
    route_id = row.get("selected_route")
    if not route_id: continue
    row = dict(row)
    if route_id not in route_specs: raise ValueError(f"{policy_path} selects unsupported route {route_id!r}; supported={sorted(route_specs)}")
    _validate_route_params(policy_path, route_id, dict(row.get("route_params", {})), manifest_registry)
    route_kind = route_specs[route_id]["kind"]
    if route_kind == _ROUTE_KIND_DECODE_FLASH: selected[route_id] = RoutePolicyRow(_freeze(row)); continue
    if route_kind in (_ROUTE_KIND_Q4K_G3, _ROUTE_KIND_Q6K_GEN, _ROUTE_KIND_PREFILL_GEN):
      _route_shape_rows_cols(policy_path, route_id, row)
      frozen_row = RoutePolicyRow(_freeze(row))
      { _ROUTE_KIND_Q4K_G3: q4k_g3_rows, _ROUTE_KIND_Q6K_GEN: q6k_gen_rows, _ROUTE_KIND_PREFILL_GEN: prefill_gen_rows }[route_kind].append(frozen_row)
      selected.setdefault(route_id, frozen_row)
    else: selected[route_id] = RoutePolicyRow(_freeze(row))
  provenance = _freeze({k: v for k, v in data.items() if k != "routes"})
  return ValidatedRoutePolicy(str(policy_path), provenance, MappingProxyType(selected), tuple(q4k_g3_rows), tuple(q6k_gen_rows), tuple(prefill_gen_rows))

def qk_route_policy_selected(route_id:str, shape:dict[str, int]|None=None, *, policy:ValidatedRoutePolicy|None=None) -> bool:
  if policy is None or (row := policy.get("selected", {}).get(route_id)) is None: return False
  if shape is not None:
    rows = [cand for cand in policy.get("prefill_gen", []) if cand.get("selected_route") == route_id] or [row]
    for cand in rows:
      policy_shape = dict(cand.get("shape", {}))
      if set(policy_shape) == set(shape) and all(int(policy_shape[k]) == int(v) for k, v in shape.items()): return True
    return False
  return True

def qk_route_policy_selects_q4k_g3(out_features:int, in_features:int, *, policy:ValidatedRoutePolicy|None=None) -> bool:
  return policy is not None and any("rows" in (shape:=row.get("shape", {})) and "cols" in shape and int(shape["rows"]) == out_features and int(shape["cols"]) == in_features for row in policy.get("q4k_g3", []))
def qk_route_policy_selects_q6k_generated(out_features:int, in_features:int, *, policy:ValidatedRoutePolicy|None=None) -> bool:
  return policy is not None and any("rows" in (shape:=row.get("shape", {})) and "cols" in shape and int(shape["rows"]) == out_features and int(shape["cols"]) == in_features for row in policy.get("q6k_gen", []))

_load_qk_route_policy = load_qk_route_policy
_qk_route_policy_selected = qk_route_policy_selected
_qk_route_policy_selects_q4k_g3 = qk_route_policy_selects_q4k_g3
_qk_route_policy_selects_q6k_generated = qk_route_policy_selects_q6k_generated

# TG3 (docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md): "is this candidate
# promoted for this resolved target" is a policy question and this is its one authority -- reuse, not a
# second table. `promoted_targets` is None until a BoltBeam-sourced promotion record has been loaded (TG4
# packages the canonical manifest; the EXP snapshot is the only artifact that may ever populate this) -- that
# is deliberately NOT the same as an explicit denial, so every target is admitted by capability alone until a
# real record exists, matching today's production default. No target string is hardcoded here: the only way
# a target is ever excluded is by an explicit, loaded promotion record naming (backend, architecture) pairs.
Target = tuple[str | None, str | None]

def load_qk_target_promotion(path:str) -> frozenset[Target] | None:
  """Read a Q4_K/Q6_K target-promotion record from an explicit BoltBeam-sourced route-policy snapshot path
  (the same boltbeam.route_policy.v1 schema `load_qk_route_policy` already validates -- the Boundary Rule
  forbids tinygrad/** importing extra/llm_research/route_manifest.py directly, so this explicit-path JSON
  parser is the only channel target-promotion facts may reach production through). A document with no
  `promoted_targets` key returns None (no restriction recorded); an explicit list -- including an empty one --
  is a real, enforced boundary once loaded (see ModelRoutePlan.target_promoted)."""
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1": raise ValueError(f"{policy_path} is not a boltbeam.route_policy.v1 route policy")
  targets = data.get("promoted_targets")
  if targets is None: return None
  return frozenset((t.get("backend"), t.get("architecture")) for t in targets)

_load_qk_target_promotion = load_qk_target_promotion

def load_decode_epilogue_fusion_promotion(path:str) -> frozenset[Target]:
  """Read the L1 decode epilogue-fusion promotion record (boltbeam.route_policy.v1, same schema family as
  `load_qk_target_promotion`). CLOSED default, deliberately the inverse of TG3's "no record -> open": the
  fused decode variants are additive emitter changes and must not move AMD/Metal admitted routes until each
  target opts in with a measured record (l1-decode-plumbing-fusion-design-20260802.md section 5). A document
  without `promoted_targets` -- or with an empty list -- promotes nothing."""
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1": raise ValueError(f"{policy_path} is not a boltbeam.route_policy.v1 route policy")
  targets = data.get("promoted_targets")
  if targets is None: return frozenset()
  return frozenset((t.get("backend"), t.get("architecture")) for t in targets)

_DECODE_EPILOGUE_FUSION_PROMOTION_RECORD = pathlib.Path(__file__).with_name("generated") / "decode-epilogue-fusion-route-policy.json"
_DECODE_EPILOGUE_FUSION_PROMOTED_TARGETS: frozenset[Target] = load_decode_epilogue_fusion_promotion(_DECODE_EPILOGUE_FUSION_PROMOTION_RECORD)

def decode_epilogue_fusion_promoted(target:Target) -> bool:
  """Policy authority for the L1 decode epilogue-fusion route (closed default, see the loader above)."""
  return target in _DECODE_EPILOGUE_FUSION_PROMOTED_TARGETS

def load_decode_q4k_epilogue_fusion_promotion(path:str) -> frozenset[Target]:
  """Read the L1 M4 q4k GEMV epilogue-fusion promotion record (boltbeam.route_policy.v1, same schema
  family as `load_decode_epilogue_fusion_promotion`). CLOSED default; deliberately a SEPARATE record from
  M2's `decode_epilogue_fusion` so the measured Q6K in-kernel merge stays NV-promoted while the measured
  non-landing q4k epilogue variants stay off on every target (m4-q4k-epilogue-measurement-record-20260802.md).
  A document without `promoted_targets` -- or with an empty list -- promotes nothing."""
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1": raise ValueError(f"{policy_path} is not a boltbeam.route_policy.v1 route policy")
  targets = data.get("promoted_targets")
  if targets is None: return frozenset()
  return frozenset((t.get("backend"), t.get("architecture")) for t in targets)

_DECODE_Q4K_EPILOGUE_FUSION_PROMOTION_RECORD = pathlib.Path(__file__).with_name("generated") / "decode-q4k-epilogue-fusion-route-policy.json"
_DECODE_Q4K_EPILOGUE_FUSION_PROMOTED_TARGETS: frozenset[Target] = load_decode_q4k_epilogue_fusion_promotion(_DECODE_Q4K_EPILOGUE_FUSION_PROMOTION_RECORD)

def decode_q4k_epilogue_fusion_promoted(target:Target) -> bool:
  """Policy authority for the L1 M4 q4k GEMV epilogue-fusion route (closed default, see the loader above)."""
  return target in _DECODE_Q4K_EPILOGUE_FUSION_PROMOTED_TARGETS

def load_decode_q4k_w1w3_fusion_promotion(path:str) -> frozenset[Target]:
  """Read the w1+w3 fused gate/up decode GEMV promotion record (boltbeam.route_policy.v1, same schema
  family as `load_decode_q4k_epilogue_fusion_promotion`). CLOSED default; deliberately a SEPARATE record
  from M4's q4k epilogue record and M2's Q6K merge record: the fused w1+w3 kernel replaces the gate GEMV
  + silu + up GEMV + mul chain (72 -> 36 kernels/token) with one 12288-row kernel, and must not fire on
  any target until a measured record opts it in (mc3-w1w3-fusion-measurement-record-20260803.md,
  q4k-w1w3-fused-qv-implementation-record-20260803.md). A document without `promoted_targets` -- or with
  an empty list -- promotes nothing."""
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1": raise ValueError(f"{policy_path} is not a boltbeam.route_policy.v1 route policy")
  targets = data.get("promoted_targets")
  if targets is None: return frozenset()
  return frozenset((t.get("backend"), t.get("architecture")) for t in targets)

_DECODE_Q4K_W1W3_FUSION_PROMOTION_RECORD = pathlib.Path(__file__).with_name("generated") / "decode-q4k-w1w3-fusion-route-policy.json"
_DECODE_Q4K_W1W3_FUSION_PROMOTED_TARGETS: frozenset[Target] = load_decode_q4k_w1w3_fusion_promotion(_DECODE_Q4K_W1W3_FUSION_PROMOTION_RECORD)

def decode_q4k_w1w3_fusion_promoted(target:Target) -> bool:
  """Policy authority for the fused w1+w3 gate/up decode GEMV route (closed default, see the loader above)."""
  return target in _DECODE_Q4K_W1W3_FUSION_PROMOTED_TARGETS

def load_decode_kv_store_fusion_promotion(path:str) -> frozenset[Target]:
  """Read the decode kv-store chain fusion promotion record (boltbeam.route_policy.v1, same schema
  family as `load_decode_q4k_w1w3_fusion_promotion`). CLOSED default; deliberately a SEPARATE record
  from every other decode record: the fused kv-store kernel replaces the k-rope + k-cast + v-cast +
  Tensor.stack(k, v) + cache store chain (5 kernels/layer, decode-kv-store-chain-fusion-scope-
  20260803.md), and must not fire on any target until a measured same-session record opts it in
  (nv-decode-gap-decomposition-record-20260803.md section 5, lever 1). A document without
  `promoted_targets` -- or with an empty list -- promotes nothing."""
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1": raise ValueError(f"{policy_path} is not a boltbeam.route_policy.v1 route policy")
  targets = data.get("promoted_targets")
  if targets is None: return frozenset()
  return frozenset((t.get("backend"), t.get("architecture")) for t in targets)

_DECODE_KV_STORE_FUSION_PROMOTION_RECORD = pathlib.Path(__file__).with_name("generated") / "decode-kv-store-fusion-route-policy.json"
_DECODE_KV_STORE_FUSION_PROMOTED_TARGETS: frozenset[Target] = load_decode_kv_store_fusion_promotion(_DECODE_KV_STORE_FUSION_PROMOTION_RECORD)

def decode_kv_store_fusion_promoted(target:Target) -> bool:
  """Policy authority for the fused decode kv-store route (closed default, see the loader above)."""
  return target in _DECODE_KV_STORE_FUSION_PROMOTED_TARGETS

def load_decode_flash_combine_fusion_promotion(path:str) -> frozenset[Target]:
  """Read the L1 M5 flash-decode combine fp16 absorption promotion record (boltbeam.route_policy.v1, same
  schema family as `load_decode_epilogue_fusion_promotion`). CLOSED default; deliberately a SEPARATE record
  from M2's `decode_epilogue_fusion` (which stays NV-promoted for the Q6K in-kernel merge only): the fp16
  combine variant (flash_fused_gmax_combine_f16_*) absorbs the post-combine E_32_32_4_0a5e fp32->fp16 cast
  and must not fire under the M2 record or on any target until a measured record opts it in
  (m5-flash-combine-normalization-measurement-record-20260802.md). A document without `promoted_targets`
  -- or with an empty list -- promotes nothing."""
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1": raise ValueError(f"{policy_path} is not a boltbeam.route_policy.v1 route policy")
  targets = data.get("promoted_targets")
  if targets is None: return frozenset()
  return frozenset((t.get("backend"), t.get("architecture")) for t in targets)

_DECODE_FLASH_COMBINE_FUSION_PROMOTION_RECORD = pathlib.Path(__file__).with_name("generated") / "decode-flash-combine-route-policy.json"
_DECODE_FLASH_COMBINE_FUSION_PROMOTED_TARGETS: frozenset[Target] = load_decode_flash_combine_fusion_promotion(_DECODE_FLASH_COMBINE_FUSION_PROMOTION_RECORD)

def decode_flash_combine_fusion_promoted(target:Target) -> bool:
  """Policy authority for the L1 M5 flash-decode combine fp16 absorption route (closed default, see the
  loader above)."""
  return target in _DECODE_FLASH_COMBINE_FUSION_PROMOTED_TARGETS

def load_decode_norm_fusion_promotion(path:str) -> frozenset[Target]:
  """Read the L1 M3 fused decode RMSNorm promotion record (boltbeam.route_policy.v1, same schema family
  as `load_decode_epilogue_fusion_promotion`). CLOSED default with the same semantics: a document without
  `promoted_targets` -- or with an empty list -- promotes nothing. The fused norm emitter is a measured
  NON-LANDING on every target so far (m3-fused-norm-measurement-record-20260802.md): the opaque
  custom-kernel boundary materializes one contiguous copy per norm call, so the family regresses the M2
  decode baseline despite byte-identical tokens. The record reopens only when a target has a measured
  copy-free or launch-overhead-equivalent number behind it."""
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1": raise ValueError(f"{policy_path} is not a boltbeam.route_policy.v1 route policy")
  targets = data.get("promoted_targets")
  if targets is None: return frozenset()
  return frozenset((t.get("backend"), t.get("architecture")) for t in targets)

_DECODE_NORM_FUSION_PROMOTION_RECORD = pathlib.Path(__file__).with_name("generated") / "decode-norm-fusion-route-policy.json"
_DECODE_NORM_FUSION_PROMOTED_TARGETS: frozenset[Target] = load_decode_norm_fusion_promotion(_DECODE_NORM_FUSION_PROMOTION_RECORD)

def decode_norm_fusion_promoted(target:Target) -> bool:
  """Policy authority for the L1 M3 fused decode RMSNorm route (closed default, measured non-landing)."""
  return target in _DECODE_NORM_FUSION_PROMOTED_TARGETS

def load_decode_rmsnorm_native_lowering_promotion(path:str) -> frozenset[Target]:
  """Read the Path 3 semantic RMSNorm native-lowering promotion record (boltbeam.route_policy.v1,
  same schema family as `load_decode_norm_fusion_promotion`). CLOSED default: a document without
  `promoted_targets` -- or with an empty list -- promotes nothing. The semantic marker lowers to a
  scheduler-owned kernel only for a target with a measured fixed-depth win behind it
  (path3-semantic-rmsnorm-task-20260802.md section 3); decode evidence never authorizes prefill."""
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1": raise ValueError(f"{policy_path} is not a boltbeam.route_policy.v1 route policy")
  targets = data.get("promoted_targets")
  if targets is None: return frozenset()
  return frozenset((t.get("backend"), t.get("architecture")) for t in targets)

_DECODE_RMSNORM_NATIVE_LOWERING_PROMOTION_RECORD = pathlib.Path(__file__).with_name("generated") / "decode-rmsnorm-native-lowering-route-policy.json"
_DECODE_RMSNORM_NATIVE_LOWERING_PROMOTED_TARGETS: frozenset[Target] = load_decode_rmsnorm_native_lowering_promotion(_DECODE_RMSNORM_NATIVE_LOWERING_PROMOTION_RECORD)

def decode_rmsnorm_native_lowering_promoted(target:Target) -> bool:
  """Policy authority for the Path 3 semantic RMSNorm native-lowering route (closed default)."""
  return target in _DECODE_RMSNORM_NATIVE_LOWERING_PROMOTED_TARGETS

@dataclass(frozen=True)
class PrimitiveRouteEntry:
  name:str; module_path:str; quant_label:str; rows:int; cols:int; role:str; parts:int; opts:tuple[str, ...]; family:str
  kernel_mode:str = "partial"

class ModelRoutePlan:
  def __init__(self, entries:Iterable[PrimitiveRouteEntry]=(), promoted_targets:frozenset[Target]|None=None):
    self._entries = {entry.name: entry for entry in entries}
    self.promoted_targets = promoted_targets
  def primitive(self, name:str) -> PrimitiveRouteEntry|None: return self._entries.get(name)
  def target_promoted(self, target:Target) -> bool:
    """Policy authority for TG3's second question. See the module-level note above `load_qk_target_promotion`
    for why `None` here means "undecided by target", not "denied"."""
    return self.promoted_targets is None or target in self.promoted_targets
  def __len__(self) -> int: return len(self._entries)
  def __iter__(self): return iter(self._entries.values())

def _module_path(name:str) -> str: return name[:-len(".weight")] if name.endswith(".weight") else name
def _role_family(role:str) -> str: return normalize_route_role(role)

def _default(name:str, quant:str, role:str, rows:int, cols:int) -> tuple[int, tuple[str, ...]]|None:
  if rows <= 0 or cols <= 0 or cols % 256: return None
  family = _role_family(role)
  if quant == "Q4_K":
    if family in ("ffn_gate_up", "attn_qo", "attn_kv"): return 1, ("LOCAL:0:64",)
    if family == "ffn_down": return 4, ("LOCAL:0:32",)
  if quant == "Q6_K":
    if family == "ffn_down" or family == "lm_head" or name == "output.weight": return 1, ("LOCAL:0:64",)
    if family == "attn_kv" and _module_path(name).rsplit(".", 1)[-1] == "attn_v": return 4, ("LOCAL:0:32",)
  return None

def primitive_route_entry_for_tensor(name:str, typ:int, rows:int, cols:int, *, module_path:str|None=None,
                                     quant_label:str|None=None, role:str|None=None, **_unused) -> PrimitiveRouteEntry|None:
  module_path = module_path or _module_path(name); role = role or module_path.rsplit(".", 1)[-1]
  quant = quant_label or ("Q4_K" if typ == 12 else "Q6_K" if typ == 14 else "")
  if not quant or (policy := _default(name, quant, role, rows, cols)) is None: return None
  parts, opts = policy
  return PrimitiveRouteEntry(name, module_path, quant, rows, cols, role, parts, opts,
                             "q4_k_packed_u32" if typ == 12 else "q6_k_packed_u16")

def _record_entry(record:Any) -> PrimitiveRouteEntry|None:
  if isinstance(record, PrimitiveRouteEntry): return record
  get = record.get if isinstance(record, dict) else lambda key, default=None: getattr(record, key, default)
  name, typ = get("name", get("tensor")), get("ggml_type", get("typ"))
  rows, cols, dims = get("rows"), get("cols"), get("dims")
  if (rows is None or cols is None) and dims is not None and len(dims) == 2: rows, cols = reversed(dims)
  if name is None or typ is None or rows is None or cols is None: return None
  return primitive_route_entry_for_tensor(str(name), int(typ), int(rows), int(cols), module_path=get("module_path"),
                                          quant_label=get("quant_label"), role=get("role"))

def build_model_route_plan(meta:dict|None=None, model_facts:Any=None, promoted_targets:frozenset[Target]|None=None) -> ModelRoutePlan:
  records = getattr(model_facts, "tensors", ()) if model_facts is not None else ()
  entries = [entry for record in records if (entry := _record_entry(record)) is not None]
  if not entries and meta is not None:
    for name, dims, typ, _off in meta.get("tensor_infos", ()):
      if str(name).endswith(".weight") and len(dims) == 2 and (entry := primitive_route_entry_for_tensor(str(name), int(typ), int(dims[1]), int(dims[0]))) is not None:
        entries.append(entry)
  return ModelRoutePlan(entries, promoted_targets)

try: from tinygrad.llm.model_facts import ModelFacts as ModelFacts
except Exception: ModelFacts = None
