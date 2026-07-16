from __future__ import annotations
import collections, pathlib
from dataclasses import dataclass
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.llm.gguf import MODEL_PARAMETER_ALLOCATION_OWNER, ggml_data_to_tensor
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

class Q4KPrimitiveStorage:
  __slots__ = ("words", "source_bytes", "persistent_bytes", "shared_bytes", "nonpersistent_bytes", "mode")
  def __init__(self, words:Tensor, source_bytes:int, persistent_bytes:int, mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0):
    self.words, self.source_bytes, self.persistent_bytes = words, source_bytes, persistent_bytes
    self.shared_bytes, self.nonpersistent_bytes, self.mode = shared_bytes, nonpersistent_bytes, mode

class Q6KPrimitiveStorage:
  __slots__ = ("halfs", "source_bytes", "persistent_bytes", "shared_bytes", "nonpersistent_bytes", "mode")
  def __init__(self, halfs:Tensor, source_bytes:int, persistent_bytes:int, mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0):
    self.halfs, self.source_bytes, self.persistent_bytes = halfs, source_bytes, persistent_bytes
    self.shared_bytes, self.nonpersistent_bytes, self.mode = shared_bytes, nonpersistent_bytes, mode

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

class Q4KPrimitiveLinear:
  def __init__(self, weight:Tensor, bias:Tensor|None, words:Tensor, out_features:int, in_features:int, parts:int, opts:tuple,
               name:str, source_bytes:int, persistent_bytes:int, storage_mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0, kernel_mode:str="partial", route_role:str="",
               eligibility:QKPrimitiveEligibility|None=None):
    if kernel_mode not in ("partial", "direct_out"): raise ValueError(f"unsupported Q4_K primitive kernel mode {kernel_mode!r}")
    if kernel_mode == "direct_out" and parts != 1: raise ValueError("Q4_K direct_out primitive requires parts=1")
    self.weight, self.bias = weight, bias
    _model_parameter_alias(weight, words)
    self.q4k_storage = Q4KPrimitiveStorage(words, source_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes)
    self.out_features, self.in_features, self.parts, self.opts, self.name = out_features, in_features, parts, opts, name
    self.kernel_mode = kernel_mode
    self.route_role = route_role
    self.eligibility = eligibility or QKPrimitiveEligibility()
    self.decode_enabled = False

  def _fallback(self, x:Tensor) -> Tensor:
    return x.linear(self.weight.transpose(), self.bias)

  def prefill_packed_weight(self) -> Tensor:
    if self.q4k_storage.mode in ("sidecar", "shared"): return self.q4k_storage.words
    if not hasattr(self, "_prefill_q4k_words"):
      self._prefill_q4k_words = _model_parameter_materialization(self.weight, self.q4k_storage.words.clone())
    return self._prefill_q4k_words

  def prefill_fp16_weight(self) -> Tensor:
    raw = self.q4k_storage.words.bitcast(dtypes.uint8).reshape(-1)
    return _model_parameter_materialization(self.weight, ggml_data_to_tensor(raw, self.out_features * self.in_features, 12).reshape(
      self.out_features, self.in_features).cast(dtypes.float16).contiguous())

  def __call__(self, x:Tensor) -> Tensor:
    return q4k_primitive_linear_call(self, x, self._fallback, self.eligibility.eligible)

class Q6KPrimitiveLinear:
  def __init__(self, weight:Tensor, bias:Tensor|None, halfs:Tensor, out_features:int, in_features:int, parts:int, opts:tuple,
               name:str, source_bytes:int, persistent_bytes:int, storage_mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0, route_role:str="", eligibility:QKPrimitiveEligibility|None=None):
    self.weight, self.bias = weight, bias
    _model_parameter_alias(weight, halfs)
    self.q6k_storage = Q6KPrimitiveStorage(halfs, source_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes)
    self.out_features, self.in_features, self.parts, self.opts, self.name = out_features, in_features, parts, opts, name
    self.route_role = route_role
    self.eligibility = eligibility or QKPrimitiveEligibility()
    self.decode_enabled = False

  def _fallback(self, x:Tensor) -> Tensor:
    return x.linear(self.weight.transpose(), self.bias)

  def prefill_packed_weight(self) -> Tensor:
    if self.q6k_storage.mode in ("sidecar", "shared"): return self.q6k_storage.halfs
    if not hasattr(self, "_prefill_q6k_halfs"):
      self._prefill_q6k_halfs = _model_parameter_materialization(self.weight, self.q6k_storage.halfs.clone())
    return self._prefill_q6k_halfs

  def prefill_fp16_weight(self) -> Tensor:
    raw = self.q6k_storage.halfs.bitcast(dtypes.uint8).reshape(-1)
    return _model_parameter_materialization(self.weight, ggml_data_to_tensor(raw, self.out_features * self.in_features, 14).reshape(
      self.out_features, self.in_features).cast(dtypes.float16).contiguous())

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

def _install_q4k_primitives(model, gguf:pathlib.Path, meta:dict, generated_policy:dict|None=None,
                            budget:QKPrimitiveBudget|None=None, storage_mode:str="sidecar",
                            route_plan:ModelRoutePlan|None=None, device_facts:object|None=None,
                            debug:bool=False) -> list[Q4KPrimitiveLinear]:
  supported_generated_families = {"q4_k_packed_u32", "q4_k_packed_u32_direct"}
  if storage_mode not in ("sidecar", "q4_ondemand", "shared"):
    raise ValueError(f"unsupported QK primitive storage mode {storage_mode!r}")
  raw_words = Tensor(gguf, dtype=dtypes.uint32)
  installed: list[Q4KPrimitiveLinear] = []
  skipped: collections.Counter[str] = collections.Counter()
  budget = budget or QKPrimitiveBudget()
  eligibility = qk_primitive_eligibility_from_device_facts(device_facts)
  if generated_policy is None and route_plan is None: route_plan = build_model_route_plan(meta)
  for name, dims, typ, off in meta["tensor_infos"]:
    if typ != 12:
      skipped["not_q4_k"] += 1
      continue
    if len(dims) != 2:
      skipped["not_2d"] += 1
      continue
    if not name.endswith(".weight"):
      skipped["not_weight"] += 1
      continue
    rows, cols = tuple(reversed(dims))
    if generated_policy is None:
      if route_plan is None or (route_entry := route_plan.primitive(name)) is None:
        skipped["policy_fallback"] += 1
        continue
      if route_entry.quant_label != "Q4_K" or route_entry.rows != rows or route_entry.cols != cols:
        skipped["policy_unsupported"] += 1
        continue
      parts, opt_specs = route_entry.parts, route_entry.opts
      kernel_mode = route_entry.kernel_mode
      route_role = route_entry.role
    else:
      if (policy_entry := _qk_generated_policy_entry(generated_policy, typ, rows, cols, name)) is None:
        skipped["policy_missing"] += 1
        continue
      if policy_entry["winner"] == "fused_graph":
        skipped["policy_fused"] += 1
        continue
      if policy_entry["family"] not in supported_generated_families:
        skipped["policy_unsupported"] += 1
        continue
      parts, opt_specs = policy_entry["parts"], policy_entry["opts"]
      kernel_mode = "direct_out" if policy_entry["family"] == "q4_k_packed_u32_direct" or policy_entry.get("reduction") == "direct_out" else "partial"
      route_role = ""
      if kernel_mode == "direct_out" and parts != 1:
        skipped["policy_invalid_direct_parts"] += 1
        continue
    byte_start = meta["data_start"] + off
    if byte_start % 4 != 0:
      skipped["misaligned"] += 1
      continue
    module_path = name[:-len(".weight")] if generated_policy is not None else route_entry.module_path
    try: module = _module_at(model, module_path)
    except (AttributeError, IndexError, ValueError):
      skipped["missing_module"] += 1
      continue
    if not hasattr(module, "weight"):
      skipped["missing_weight"] += 1
      continue
    if getattr(module, "bias", None) is not None:
      skipped["bias"] += 1
      continue
    q4_bytes = prod(dims) // 256 * 144
    persistent_bytes = q4_bytes if storage_mode == "sidecar" else 0
    if not budget.reserve(name, persistent_bytes, "Q4_K"):
      skipped["runtime_storage_cap"] += 1
      continue
    source = raw_words[byte_start//4:byte_start//4+q4_bytes//4]
    if storage_mode == "shared":
      words, shared_bytes, nonpersistent_bytes = _shared_packed_view(meta, byte_start, q4_bytes, dtypes.uint32), q4_bytes, 0
    else:
      words = source.contiguous() if storage_mode == "q4_ondemand" else _model_parameter_materialization(module.weight, source.to(None).contiguous())
      shared_bytes, nonpersistent_bytes = 0, q4_bytes if storage_mode == "q4_ondemand" else 0
    q4k_linear = Q4KPrimitiveLinear(module.weight, module.bias, words, rows, cols, parts, tuple(qk_ops.q4k_parse_opt(x) for x in opt_specs), name,
                                    q4_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes,
                                    kernel_mode=kernel_mode, route_role=route_role, eligibility=eligibility)
    _set_module_at(model, module_path, q4k_linear)
    installed.append(q4k_linear)
  if debug:
    skipped_s = " ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
    installed_s = " ".join(f"{x.name}:mode={x.kernel_mode}:parts={x.parts}:opts={[str(o) for o in x.opts]}" for x in installed[:8])
    more_s = f" ...+{len(installed)-8}" if len(installed) > 8 else ""
    summary = _qk_storage_summary(installed)
    cap = -1 if budget.cap_bytes is None else budget.cap_bytes
    print(f"Q4K_PRIMITIVE_DEBUG installed={len(installed)} skipped_total={sum(skipped.values())} {skipped_s} "
          f"source_bytes={summary['source_bytes']} storage_bytes={summary['persistent_bytes']} "
          f"shared_bytes={summary['shared_bytes']} nonpersistent_bytes={summary['nonpersistent_bytes']} "
          f"runtime_cap_bytes={cap} runtime_cap_used_bytes={budget.used_bytes} storage_mode={storage_mode}")
    if installed: print(f"Q4K_PRIMITIVE_DEBUG installed_linears {installed_s}{more_s}")
  return installed

def _install_q6k_primitives(model, gguf:pathlib.Path, meta:dict, generated_policy:dict|None=None,
                            budget:QKPrimitiveBudget|None=None, storage_mode:str="sidecar",
                            route_plan:ModelRoutePlan|None=None, device_facts:object|None=None,
                            debug:bool=False) -> list[Q6KPrimitiveLinear]:
  if storage_mode not in ("sidecar", "shared"):
    raise ValueError(f"unsupported Q6_K primitive storage mode {storage_mode!r}")
  raw_halfs = Tensor(gguf, dtype=dtypes.uint16)
  installed: list[Q6KPrimitiveLinear] = []
  skipped: collections.Counter[str] = collections.Counter()
  budget = budget or QKPrimitiveBudget()
  eligibility = qk_primitive_eligibility_from_device_facts(device_facts)
  if generated_policy is None and route_plan is None: route_plan = build_model_route_plan(meta)
  for name, dims, typ, off in meta["tensor_infos"]:
    if typ != 14:
      skipped["not_q6_k"] += 1
      continue
    if len(dims) != 2:
      skipped["not_2d"] += 1
      continue
    if not name.endswith(".weight"):
      skipped["not_weight"] += 1
      continue
    rows, cols = tuple(reversed(dims))
    if generated_policy is None:
      if route_plan is None or (route_entry := route_plan.primitive(name)) is None:
        skipped["policy_fallback"] += 1
        continue
      if route_entry.quant_label != "Q6_K" or route_entry.rows != rows or route_entry.cols != cols:
        skipped["policy_unsupported"] += 1
        continue
      parts, opt_specs = route_entry.parts, route_entry.opts
      route_role = route_entry.role
    else:
      if (policy_entry := _qk_generated_policy_entry(generated_policy, typ, rows, cols, name)) is None:
        skipped["policy_missing"] += 1
        continue
      if policy_entry["winner"] == "fused_graph":
        skipped["policy_fused"] += 1
        continue
      if policy_entry["family"] != "q6_k_packed_u16":
        skipped["policy_unsupported"] += 1
        continue
      parts, opt_specs = policy_entry["parts"], policy_entry["opts"]
      route_role = ""
    byte_start = meta["data_start"] + off
    if byte_start % 2 != 0:
      skipped["misaligned"] += 1
      continue
    module_path = name[:-len(".weight")] if generated_policy is not None else route_entry.module_path
    try: module = _module_at(model, module_path)
    except (AttributeError, IndexError, ValueError):
      skipped["missing_module"] += 1
      continue
    if not hasattr(module, "weight"):
      skipped["missing_weight"] += 1
      continue
    if getattr(module, "bias", None) is not None:
      skipped["bias"] += 1
      continue
    q6_bytes = prod(dims) // 256 * 210
    persistent_bytes = 0 if storage_mode == "shared" else q6_bytes
    if not budget.reserve(name, persistent_bytes, "Q6_K"):
      skipped["runtime_storage_cap"] += 1
      continue
    if storage_mode == "shared":
      halfs, shared_bytes = _shared_packed_view(meta, byte_start, q6_bytes, dtypes.uint16), q6_bytes
    else:
      halfs, shared_bytes = _model_parameter_materialization(module.weight,
        raw_halfs[byte_start//2:byte_start//2+q6_bytes//2].to(None).contiguous()), 0
    q6k_linear = Q6KPrimitiveLinear(module.weight, module.bias, halfs, rows, cols, parts, tuple(qk_ops.q6k_parse_opt(x) for x in opt_specs), name,
                                    q6_bytes, persistent_bytes, storage_mode, shared_bytes, 0, route_role=route_role, eligibility=eligibility)
    _set_module_at(model, module_path, q6k_linear)
    installed.append(q6k_linear)
  if debug:
    skipped_s = " ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
    installed_s = " ".join(f"{x.name}:parts={x.parts}:opts={[str(o) for o in x.opts]}" for x in installed[:8])
    more_s = f" ...+{len(installed)-8}" if len(installed) > 8 else ""
    summary = _qk_storage_summary(installed)
    cap = -1 if budget.cap_bytes is None else budget.cap_bytes
    print(f"Q6K_PRIMITIVE_DEBUG installed={len(installed)} skipped_total={sum(skipped.values())} {skipped_s} "
          f"source_bytes={summary['source_bytes']} storage_bytes={summary['persistent_bytes']} "
          f"shared_bytes={summary['shared_bytes']} nonpersistent_bytes={summary['nonpersistent_bytes']} "
          f"runtime_cap_bytes={cap} runtime_cap_used_bytes={budget.used_bytes} storage_mode={storage_mode}")
    if installed: print(f"Q6K_PRIMITIVE_DEBUG installed_linears {installed_s}{more_s}")
  return installed
