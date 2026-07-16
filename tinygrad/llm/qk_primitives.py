from __future__ import annotations
import collections, pathlib
from dataclasses import dataclass
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.llm.gguf import MODEL_PARAMETER_ALLOCATION_OWNER
from tinygrad.llm.memory_semantics import MODEL_PARAMETER, memory_semantic_owner, model_parameter
from tinygrad.llm.physical_memory_ledger import allocation_owner, bind_allocation_owner
from tinygrad.llm import route_ops as qk_ops
from tinygrad.llm.decode_routes import q4k_primitive_linear_call, q6k_primitive_linear_call
from tinygrad.llm.route_policy import _qk_generated_policy_entry
from tinygrad.llm.model_route_plan import ModelRoutePlan, build_model_route_plan

@dataclass(frozen=True)
class QKPrimitiveEligibility:
  """Structural target facts retained by an installed AMD gfx1100 primitive."""
  backend: str|None = None
  architecture: str|None = None
  wave_size: int|None = None

  @property
  def eligible(self) -> bool:
    return (self.backend, self.architecture, self.wave_size) == ("AMD", "gfx1100", 32)

def qk_primitive_eligibility_from_device_facts(device_facts:object|None) -> QKPrimitiveEligibility:
  """Copy only immutable candidate-relevant fields from the load-entry DeviceFacts scan."""
  if device_facts is None: return QKPrimitiveEligibility()
  capabilities = getattr(device_facts, "capabilities", None)
  return QKPrimitiveEligibility(getattr(device_facts, "backend", None), getattr(device_facts, "architecture", None),
                                getattr(capabilities, "wave_size", None))

def _model_parameter_alias(source:Tensor|None, derived:Tensor) -> Tensor:
  """Attach model ownership to derived storage which already aliases a backing."""
  source = derived if source is None else source  # permits isolated primitive test doubles
  if memory_semantic_owner(source) is None: model_parameter(source)
  if memory_semantic_owner(source) != MODEL_PARAMETER:
    raise ValueError("packed model storage source must have MODEL_PARAMETER semantics")
  model_parameter(derived)
  try: bind_allocation_owner(derived.uop.buffer, MODEL_PARAMETER_ALLOCATION_OWNER)
  except AssertionError: pass
  return derived

def _model_parameter_materialization(source:Tensor|None, derived:Tensor) -> Tensor:
  """Materialize storage derived from a selected parameter with model ownership.

  The caller supplies the parameter by semantic role.  Ownership deliberately does
  not depend on tensor names, geometry, byte count, device, or route tier.
  """
  _model_parameter_alias(source, derived)
  owner = MODEL_PARAMETER_ALLOCATION_OWNER
  with allocation_owner(kind=owner.kind, lifetime=owner.lifetime, candidate_id=owner.candidate_id,
                        semantic_owner_id=owner.semantic_owner_id):
    derived.realize()
  bind_allocation_owner(derived.uop.buffer, owner)
  return derived

class _QKPrimitiveStorage:
  __slots__ = ("packed", "source_bytes", "persistent_bytes", "shared_bytes", "nonpersistent_bytes", "mode")
  def __init__(self, packed:Tensor, source_bytes:int, persistent_bytes:int, mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0):
    self.packed, self.source_bytes, self.persistent_bytes = packed, source_bytes, persistent_bytes
    self.shared_bytes, self.nonpersistent_bytes, self.mode = shared_bytes, nonpersistent_bytes, mode

class Q4KPrimitiveStorage(_QKPrimitiveStorage):
  __slots__ = ()
  @property
  def words(self) -> Tensor: return self.packed
  @words.setter
  def words(self, value:Tensor) -> None: self.packed = value

class Q6KPrimitiveStorage(_QKPrimitiveStorage):
  __slots__ = ()
  @property
  def halfs(self) -> Tensor: return self.packed
  @halfs.setter
  def halfs(self, value:Tensor) -> None: self.packed = value

class QKPrimitiveBudget:
  __slots__ = ("cap_bytes", "used_bytes", "strict")
  def __init__(self, cap_bytes:int|None=None, strict:bool=False):
    self.cap_bytes, self.used_bytes, self.strict = cap_bytes, 0, strict

  def reserve(self, name:str, bytes_needed:int, kind:str) -> bool:
    if bytes_needed < 0: raise ValueError(f"{kind} storage bytes must be non-negative for {name}: {bytes_needed}")
    if self.cap_bytes is not None and self.used_bytes + bytes_needed > self.cap_bytes:
      if self.strict:
        raise MemoryError(f"{kind} primitive storage cap exceeded for {name}: used={self.used_bytes} "
                          f"need={bytes_needed} cap={self.cap_bytes}")
      return False
    self.used_bytes += bytes_needed
    return True

class Q4KPrimitiveRegistry:
  __slots__ = ("linears",)
  def __init__(self, linears:list[Q4KPrimitiveLinear|Q6KPrimitiveLinear]|None=None): self.linears = linears or []

class _QKPrimitiveLinear:
  _storage_attr = ""
  _prefill_attr = ""
  _ggml_type = -1
  def _init_common(self, weight:Tensor, bias:Tensor|None, packed:Tensor, out_features:int, in_features:int, parts:int, opts:tuple,
                   name:str, storage, route_role:str, eligibility:QKPrimitiveEligibility|None):
    self.weight, self.bias = weight, bias
    _model_parameter_alias(weight, packed)
    setattr(self, self._storage_attr, storage)
    self.out_features, self.in_features, self.parts, self.opts, self.name = out_features, in_features, parts, opts, name
    self.route_role = route_role
    self.eligibility = eligibility or QKPrimitiveEligibility()
    self.decode_enabled = False

  def _fallback(self, x:Tensor) -> Tensor:
    return x.linear(self.weight.transpose(), self.bias)

  def prefill_packed_weight(self) -> Tensor:
    storage = getattr(self, self._storage_attr)
    if storage.mode in ("sidecar", "shared"): return storage.packed
    if not hasattr(self, self._prefill_attr): setattr(self, self._prefill_attr, _model_parameter_materialization(self.weight, storage.packed.clone()))
    return getattr(self, self._prefill_attr)

class Q4KPrimitiveLinear(_QKPrimitiveLinear):
  _storage_attr, _prefill_attr, _ggml_type = "q4k_storage", "_prefill_q4k_words", 12
  def __init__(self, weight:Tensor, bias:Tensor|None, words:Tensor, out_features:int, in_features:int, parts:int, opts:tuple,
               name:str, source_bytes:int, persistent_bytes:int, storage_mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0, kernel_mode:str="partial", route_role:str="",
               eligibility:QKPrimitiveEligibility|None=None):
    if kernel_mode not in ("partial", "direct_out"): raise ValueError(f"unsupported Q4_K primitive kernel mode {kernel_mode!r}")
    if kernel_mode == "direct_out" and parts != 1: raise ValueError("Q4_K direct_out primitive requires parts=1")
    self._init_common(weight, bias, words, out_features, in_features, parts, opts, name,
      Q4KPrimitiveStorage(words, source_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes), route_role, eligibility)
    self.kernel_mode = kernel_mode

  def __call__(self, x:Tensor) -> Tensor:
    return q4k_primitive_linear_call(self, x, self._fallback, self.eligibility.eligible)

class Q6KPrimitiveLinear(_QKPrimitiveLinear):
  _storage_attr, _prefill_attr, _ggml_type = "q6k_storage", "_prefill_q6k_halfs", 14
  def __init__(self, weight:Tensor, bias:Tensor|None, halfs:Tensor, out_features:int, in_features:int, parts:int, opts:tuple,
               name:str, source_bytes:int, persistent_bytes:int, storage_mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0, route_role:str="", eligibility:QKPrimitiveEligibility|None=None):
    self._init_common(weight, bias, halfs, out_features, in_features, parts, opts, name,
      Q6KPrimitiveStorage(halfs, source_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes), route_role, eligibility)

  def __call__(self, x:Tensor) -> Tensor:
    return q6k_primitive_linear_call(self, x, self._fallback, self.eligibility.eligible)

def _q6k_effective_storage_mode(requested_mode:str) -> str:
  # q4_ondemand is a Q4_K-only experiment. Q6_K stays persistent unless storage is shared.
  if requested_mode not in ("sidecar", "q4_ondemand", "shared"):
    raise ValueError(f"unsupported QK primitive storage mode {requested_mode!r}")
  return "shared" if requested_mode == "shared" else "sidecar"

@dataclass(frozen=True)
class QKConfig:
  """Explicit QK primitive install configuration.

  Scope is deliberately the install config, not everything QK-shaped:
  - Activation gating (`Q4K_PRIMITIVE`/`Q6K_PRIMITIVE`/`QK_GENERATED_POLICY`) stays at
    the from_gguf gate -- its auto-default is coupled to the gguf source + device, a
    separate (runtime, not env-only) concern.
  - Forward-pass probe flags (`Q4K_VDOT`/`Q4K_VDOT_AMORT`/`Q6K_COVER_MORE`/`Q4K_UNFUSE`/
    `FLASH_DECODE`/`FLASH_L`) are read per-call at their own sites; folding them here
    would change when they are read.

  Production callers must construct this value from their already-selected runtime
  policy, budget, route plan, and device facts. Environment parsing belongs in a
  separate research/configuration layer and is intentionally not provided here."""
  generated_policy_strict: bool
  max_storage_bytes: int | None
  storage_mode: str
  q6_storage_mode: str
  policy_debug: bool
  storage_debug: bool

  def __post_init__(self):
    if self.storage_mode not in ("sidecar", "q4_ondemand", "shared"):
      raise ValueError(f"unsupported QK primitive storage mode {self.storage_mode!r}")
    if self.q6_storage_mode != _q6k_effective_storage_mode(self.storage_mode):
      raise ValueError("q6_storage_mode must match the explicit storage_mode")

def _qk_storage_summary(linears:list[Q4KPrimitiveLinear|Q6KPrimitiveLinear]) -> dict:
  by_kind: collections.Counter[str] = collections.Counter()
  by_mode: collections.Counter[str] = collections.Counter()
  source_bytes = persistent_bytes = shared_bytes = nonpersistent_bytes = 0
  for linear in linears:
    storage = linear.q4k_storage if isinstance(linear, Q4KPrimitiveLinear) else linear.q6k_storage
    kind = "Q4K" if isinstance(linear, Q4KPrimitiveLinear) else "Q6K"
    by_kind[kind] += storage.persistent_bytes
    by_mode[storage.mode] += 1
    source_bytes += storage.source_bytes
    persistent_bytes += storage.persistent_bytes
    shared_bytes += getattr(storage, "shared_bytes", 0)
    nonpersistent_bytes += getattr(storage, "nonpersistent_bytes", 0)
  return {
    "source_bytes": source_bytes, "persistent_bytes": persistent_bytes,
    "shared_bytes": shared_bytes, "nonpersistent_bytes": nonpersistent_bytes,
    "by_kind": dict(sorted(by_kind.items())), "by_mode": dict(sorted(by_mode.items())),
  }

def _shared_packed_view(meta:dict, byte_start:int, nbytes:int, dtype) -> Tensor:
  raw = meta.get("raw_tensor")
  if raw is None: raise ValueError("shared QK primitive storage requires gguf_load_with_metadata raw_tensor")
  if byte_start % dtype.itemsize != 0 or nbytes % dtype.itemsize != 0:
    raise ValueError(f"shared QK primitive storage requires uint{dtype.itemsize*8}-aligned range: "
                     f"byte_start={byte_start} nbytes={nbytes}")
  raw_view = raw[byte_start:byte_start+nbytes].uop.buffer
  return Tensor(UOp.from_buffer(raw_view.view(nbytes//dtype.itemsize, dtype, 0), raw.device))

def _module_at(root, path:str):
  obj = root
  for part in path.split("."):
    obj = obj[int(part)] if isinstance(obj, list) and part.isdigit() else getattr(obj, part)
  return obj

def _set_module_at(root, path:str, value) -> None:
  if "." not in path: setattr(root, path, value); return  # top-level module (e.g. output)
  parent_path, attr = path.rsplit(".", 1)
  parent = _module_at(root, parent_path)
  if isinstance(parent, list) and attr.isdigit(): parent[int(attr)] = value
  else: setattr(parent, attr, value)

@dataclass(frozen=True)
class _QKInstallChoice:
  module_path: str; parts: int; opt_specs: tuple; route_role: str
  kernel_mode: str = "partial"

@dataclass(frozen=True)
class _QKInstallSpec:
  ggml_type: int; label: str; not_kind_counter: str; dtype: object; block_bytes: int
  generated_policy: object; route_choice: object; parse_opt: object; make_linear: object; installed_detail: object
  allowed_storage_modes: tuple[str, ...]

def _route_choice(label:str, route_plan:ModelRoutePlan|None, name:str, rows:int, cols:int,
                  skipped:collections.Counter[str], q4:bool=False) -> _QKInstallChoice|None:
  if route_plan is None or (entry := route_plan.primitive(name)) is None: skipped["policy_fallback"] += 1; return None
  if entry.quant_label != label or entry.rows != rows or entry.cols != cols: skipped["policy_unsupported"] += 1; return None
  return _QKInstallChoice(entry.module_path, entry.parts, entry.opts, entry.role, entry.kernel_mode if q4 else "partial")

def _q4_route_choice(route_plan, name, rows, cols, skipped): return _route_choice("Q4_K", route_plan, name, rows, cols, skipped, True)
def _q6_route_choice(route_plan, name, rows, cols, skipped): return _route_choice("Q6_K", route_plan, name, rows, cols, skipped)

def _generated_choice(policy:dict, typ:int, rows:int, cols:int, name:str, skipped:collections.Counter[str], families:set[str]):
  if (entry := _qk_generated_policy_entry(policy, typ, rows, cols, name)) is None: skipped["policy_missing"] += 1; return None
  if entry["winner"] == "fused_graph": skipped["policy_fused"] += 1; return None
  if entry["family"] not in families: skipped["policy_unsupported"] += 1; return None
  return entry

def _q4_generated_choice(policy, typ, rows, cols, name, skipped):
  if (entry := _generated_choice(policy, typ, rows, cols, name, skipped, {"q4_k_packed_u32", "q4_k_packed_u32_direct"})) is None: return None
  mode = "direct_out" if entry["family"] == "q4_k_packed_u32_direct" or entry.get("reduction") == "direct_out" else "partial"
  if mode == "direct_out" and entry["parts"] != 1: skipped["policy_invalid_direct_parts"] += 1; return None
  return _QKInstallChoice(name[:-len(".weight")], entry["parts"], entry["opts"], "", mode)

def _q6_generated_choice(policy, typ, rows, cols, name, skipped):
  if (entry := _generated_choice(policy, typ, rows, cols, name, skipped, {"q6_k_packed_u16"})) is None: return None
  return _QKInstallChoice(name[:-len(".weight")], entry["parts"], entry["opts"], "")

def _make_q4(module, packed, rows, cols, choice, opts, name, sizes, storage_mode, eligibility):
  return Q4KPrimitiveLinear(module.weight, module.bias, packed, rows, cols, choice.parts, opts, name, sizes[0], sizes[1], storage_mode, sizes[2], sizes[3], kernel_mode=choice.kernel_mode, route_role=choice.route_role, eligibility=eligibility)

def _make_q6(module, packed, rows, cols, choice, opts, name, sizes, storage_mode, eligibility):
  return Q6KPrimitiveLinear(module.weight, module.bias, packed, rows, cols, choice.parts, opts, name, sizes[0], sizes[1], storage_mode, sizes[2], sizes[3], route_role=choice.route_role, eligibility=eligibility)

def _q4_detail(x): return f"{x.name}:mode={x.kernel_mode}:parts={x.parts}:opts={[str(o) for o in x.opts]}"
def _q6_detail(x): return f"{x.name}:parts={x.parts}:opts={[str(o) for o in x.opts]}"

_Q4_INSTALL_SPEC = _QKInstallSpec(12, "Q4_K", "not_q4_k", dtypes.uint32, 144, _q4_generated_choice, _q4_route_choice,
  qk_ops.q4k_parse_opt, _make_q4, _q4_detail, ("sidecar", "q4_ondemand", "shared"))
_Q6_INSTALL_SPEC = _QKInstallSpec(14, "Q6_K", "not_q6_k", dtypes.uint16, 210, _q6_generated_choice, _q6_route_choice,
  qk_ops.q6k_parse_opt, _make_q6, _q6_detail, ("sidecar", "shared"))

def _install_qk_primitives(model, gguf:pathlib.Path, meta:dict, spec:_QKInstallSpec, generated_policy:dict|None, budget:QKPrimitiveBudget|None,
                           storage_mode:str, route_plan:ModelRoutePlan|None, device_facts:object|None, debug:bool):
  if storage_mode not in spec.allowed_storage_modes: raise ValueError(f"unsupported {spec.label} primitive storage mode {storage_mode!r}")
  raw, installed, skipped = Tensor(gguf, dtype=spec.dtype), [], collections.Counter()
  budget, eligibility = budget or QKPrimitiveBudget(), qk_primitive_eligibility_from_device_facts(device_facts)
  if generated_policy is None and route_plan is None: route_plan = build_model_route_plan(meta)
  for name, dims, typ, off in meta["tensor_infos"]:
    if typ != spec.ggml_type: skipped[spec.not_kind_counter] += 1; continue
    if len(dims) != 2: skipped["not_2d"] += 1; continue
    if not name.endswith(".weight"): skipped["not_weight"] += 1; continue
    rows, cols = tuple(reversed(dims))
    choice = (spec.route_choice(route_plan, name, rows, cols, skipped) if generated_policy is None else
              spec.generated_policy(generated_policy, typ, rows, cols, name, skipped))
    if choice is None: continue
    byte_start = meta["data_start"] + off
    if byte_start % spec.dtype.itemsize != 0: skipped["misaligned"] += 1; continue
    try: module = _module_at(model, choice.module_path)
    except (AttributeError, IndexError, ValueError): skipped["missing_module"] += 1; continue
    if not hasattr(module, "weight"): skipped["missing_weight"] += 1; continue
    if getattr(module, "bias", None) is not None: skipped["bias"] += 1; continue
    source_bytes = prod(dims) // 256 * spec.block_bytes
    persistent_bytes = source_bytes if storage_mode == "sidecar" else 0
    if not budget.reserve(name, persistent_bytes, spec.label): skipped["runtime_storage_cap"] += 1; continue
    if storage_mode == "shared": packed, shared_bytes = _shared_packed_view(meta, byte_start, source_bytes, spec.dtype), source_bytes
    else:
      source = raw[byte_start//spec.dtype.itemsize:byte_start//spec.dtype.itemsize+source_bytes//spec.dtype.itemsize]
      packed = source.contiguous() if storage_mode == "q4_ondemand" else _model_parameter_materialization(module.weight, source.to(None).contiguous())
      shared_bytes = 0
    sizes = (source_bytes, persistent_bytes, shared_bytes, source_bytes if storage_mode == "q4_ondemand" else 0)
    linear = spec.make_linear(module, packed, rows, cols, choice, tuple(spec.parse_opt(x) for x in choice.opt_specs), name, sizes, storage_mode, eligibility)
    _set_module_at(model, choice.module_path, linear)
    installed.append(linear)
  if debug:
    skipped_s = " ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
    summary, cap = _qk_storage_summary(installed), -1 if budget.cap_bytes is None else budget.cap_bytes
    print(f"{spec.label.replace('_', '')}_PRIMITIVE_DEBUG installed={len(installed)} skipped_total={sum(skipped.values())} {skipped_s} "
          f"source_bytes={summary['source_bytes']} storage_bytes={summary['persistent_bytes']} shared_bytes={summary['shared_bytes']} "
          f"nonpersistent_bytes={summary['nonpersistent_bytes']} runtime_cap_bytes={cap} runtime_cap_used_bytes={budget.used_bytes} storage_mode={storage_mode}")
    if installed: print(f"{spec.label.replace('_', '')}_PRIMITIVE_DEBUG installed_linears {' '.join(spec.installed_detail(x) for x in installed[:8])}"
                        f"{f' ...+{len(installed)-8}' if len(installed) > 8 else ''}")
  return installed

def _install_q4k_primitives(model, gguf:pathlib.Path, meta:dict, generated_policy:dict|None=None,
                            budget:QKPrimitiveBudget|None=None, storage_mode:str="sidecar",
                            route_plan:ModelRoutePlan|None=None, device_facts:object|None=None,
                            debug:bool=False) -> list[Q4KPrimitiveLinear]:
  return _install_qk_primitives(model, gguf, meta, _Q4_INSTALL_SPEC, generated_policy, budget, storage_mode, route_plan, device_facts, debug)

def _install_q6k_primitives(model, gguf:pathlib.Path, meta:dict, generated_policy:dict|None=None,
                            budget:QKPrimitiveBudget|None=None, storage_mode:str="sidecar",
                            route_plan:ModelRoutePlan|None=None, device_facts:object|None=None,
                            debug:bool=False) -> list[Q6KPrimitiveLinear]:
  return _install_qk_primitives(model, gguf, meta, _Q6_INSTALL_SPEC, generated_policy, budget, storage_mode, route_plan, device_facts, debug)
