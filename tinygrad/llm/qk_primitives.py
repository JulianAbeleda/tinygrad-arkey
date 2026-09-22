from __future__ import annotations
import collections, pathlib
from dataclasses import dataclass
from tinygrad import Tensor, UOp, dtypes
from tinygrad.codegen.opt import parse_opt
from tinygrad.helpers import prod
from tinygrad.llm.gguf import MODEL_PARAMETER_ALLOCATION_OWNER
from tinygrad.llm.memory_semantics import MODEL_PARAMETER, memory_semantic_owner, model_parameter
from tinygrad.llm.physical_memory_ledger import allocation_owner, bind_allocation_owner
from tinygrad.llm.decode_routes import q4k_primitive_linear_call, q6k_primitive_linear_call
from tinygrad.llm.model_route_plan import (ModelRoutePlan, build_model_route_plan,
  decode_epilogue_fusion_promoted, decode_q4k_epilogue_fusion_promoted, decode_q4k_epilogue_resadd_promoted,
  decode_q4k_w1w3_fusion_promoted)
from tinygrad.llm.qk_layout import Q4_K, Q6_K, QuantFormat

def _qk_generated_policy_entry(policy:dict|None, typ:int, rows:int, cols:int, name:str|None=None) -> dict|None:
  if policy is None: return None
  if name is not None and (entry:=policy.get("by_tensor", {}).get((name, typ, rows, cols))) is not None: return entry
  return policy.get("by_shape", {}).get((typ, rows, cols))

def kv_cache_fp16_eligible(device_facts:object|None) -> bool:
  """Capability answer for fp16 K/V cache storage: is fp16 EXPRESSIBLE on the resolved target?

  Read from `DeviceCapabilities.supports_fp16`, which the load-entry device-facts scan copies verbatim from
  the opened renderer's `supported_dtypes()` (tinygrad/llm/device_facts.py) -- never from a backend or
  architecture string. This is the same capability authority the fp16 prefill overlay admission uses
  (`v2_on`, model.py) and it replaces the pre-TG3 AMD-gfx1100 string equality that previously decided the
  cache dtype (removed: see decode-kv-store-chain-fusion-scope-20260803.md revision record). An absent or
  unreported fact (None) must never be treated as True: strict equality against a known fact, matching the
  `QKPrimitiveCapability.satisfied` convention."""
  if device_facts is None: return False
  capabilities = getattr(device_facts, "capabilities", None)
  return getattr(capabilities, "supports_fp16", None) is True

@dataclass(frozen=True)
class QKPrimitiveCapability:
  """TG3 capability authority: can the resolved target's renderer express what the Q4_K/Q6_K decode kernels
  require? Every field is copied verbatim from the TG2 renderer/device-facts scan -- never inferred from a
  target/backend string, and never conflated with promotion (see QKPrimitiveRouteAdmission below).
  `backend`/`architecture` are retained only for census/debug identification, not for the eligibility check."""
  backend: str|None = None
  architecture: str|None = None
  wave_size: int|None = None
  supports_warp_shfl_xor: bool|None = None

  @property
  def satisfied(self) -> bool:
    """decode_kernels.py::_lane_partition_reduce_sum -> _warp_reduce_sum_staged reaches warp_shfl_xor
    (codegen/late/warp_reduce.py, TG1) and assumes a 32-wide lane partition (`WARP`). An unreported wave_size
    (None -- e.g. Metal, scope section 3.3) must NEVER be treated as 32: this is strict equality against a
    known fact, not a truthiness check on an absent one."""
    return self.supports_warp_shfl_xor is True and self.wave_size == 32

def qk_primitive_capability_from_device_facts(device_facts:object|None) -> QKPrimitiveCapability:
  """Copy only the immutable renderer/device capability fields the Q4_K/Q6_K primitives require from the
  load-entry DeviceFacts scan (tinygrad/llm/device_facts.py) -- never a fresh probe, never a parallel facts
  object, never a target-string inference."""
  if device_facts is None: return QKPrimitiveCapability()
  capabilities = getattr(device_facts, "capabilities", None)
  return QKPrimitiveCapability(getattr(device_facts, "backend", None), getattr(device_facts, "architecture", None),
                               getattr(capabilities, "wave_size", None), getattr(capabilities, "supports_warp_shfl_xor", None))

@dataclass(frozen=True)
class QKPrimitiveRouteAdmission:
  """The two independent TG3 answers retained by an installed Q4_K/Q6_K primitive: capability (this file,
  read from renderer/device facts) and promotion (ModelRoutePlan.target_promoted, tinygrad/llm/model_route_plan.py
  -- the BoltBeam-sourced route-policy authority). Each is resolved by its own owner; collapsing them back
  into one hardcoded target-string boolean is exactly the pre-TG3 bug this scope package removes.
  `epilogue_fusion_promoted` is the L1 decode epilogue-fusion answer (closed default,
  model_route_plan.decode_epilogue_fusion_promoted, l1-decode-plumbing-fusion-design-20260802.md section 5):
  it gates the fused route variants only -- the legacy `admitted` route is unchanged by it.
  `q4k_epilogue_fusion_promoted` is the L1 M4 q4k GEMV epilogue answer (closed default,
  model_route_plan.decode_q4k_epilogue_fusion_promoted, m4-q4k-epilogue-measurement-record-20260802.md):
  a SEPARATE record from M2's, so the measured non-landing q4k variants stay off while the Q6K in-kernel
  merge stays promoted.
  `q4k_epilogue_resadd_promoted` is the o-proj residual_add variant answer (closed default,
  model_route_plan.decode_q4k_epilogue_resadd_promoted, m4-resadd-landing-scope-20260806.md):
  a SEPARATE record from the combined M4 answer so the residual_add variant can promote alone while
  the ffn_down prelude and fp16_cast stay off the combined record.
  `q4k_w1w3_fusion_promoted` is the fused w1+w3 (gate/up) decode GEMV answer (closed default,
  model_route_plan.decode_q4k_w1w3_fusion_promoted): the fused kernel needs BOTH linears admitted on the
  target and its own measured record (q4k-w1w3-fused-qv-implementation-record-20260803.md)."""
  capability: QKPrimitiveCapability = QKPrimitiveCapability()
  target_promoted: bool = True
  epilogue_fusion_promoted: bool = False
  q4k_epilogue_fusion_promoted: bool = False
  q4k_epilogue_resadd_promoted: bool = False
  q4k_w1w3_fusion_promoted: bool = False

  @property
  def admitted(self) -> bool: return self.capability.satisfied and self.target_promoted

  @property
  def fusion_admitted(self) -> bool: return self.admitted and self.epilogue_fusion_promoted

  @property
  def q4k_epilogue_fusion_admitted(self) -> bool: return self.admitted and self.q4k_epilogue_fusion_promoted

  @property
  def q4k_epilogue_resadd_admitted(self) -> bool: return self.admitted and self.q4k_epilogue_resadd_promoted

  @property
  def w1w3_fusion_admitted(self) -> bool: return self.admitted and self.q4k_w1w3_fusion_promoted

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
  format: QuantFormat = Q4_K
  __slots__ = ()
  @property
  def words(self) -> Tensor: return self.packed
  @words.setter
  def words(self, value:Tensor) -> None: self.packed = value

class Q6KPrimitiveStorage(_QKPrimitiveStorage):
  format: QuantFormat = Q6_K
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
                   name:str, storage, route_role:str, route_admission:QKPrimitiveRouteAdmission|None):
    self.weight, self.bias = weight, bias
    _model_parameter_alias(weight, packed)
    setattr(self, self._storage_attr, storage)
    self.out_features, self.in_features, self.parts, self.opts, self.name = out_features, in_features, parts, opts, name
    self.route_role = route_role
    # TG3 split: `route_admission.admitted` is `capability.satisfied AND target_promoted`, resolved by two
    # separate authorities (see QKPrimitiveRouteAdmission) -- this is the ONLY per-instance gate consulted at
    # call time, and defaults to not-admitted when neither was supplied (isolated construction, tests, etc.).
    self.route_admission = route_admission or QKPrimitiveRouteAdmission()
    self.decode_enabled = False

  def _fallback(self, x:Tensor) -> Tensor:
    return x.linear(self.weight.transpose(), self.bias)

  def _call_with_program_facts(self, call):
    if (binding_hook := getattr(self, "call_metadata_binding", None)) is not None: binding_hook(self)
    return call()

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
               route_admission:QKPrimitiveRouteAdmission|None=None):
    if kernel_mode not in ("partial", "direct_out"): raise ValueError(f"unsupported Q4_K primitive kernel mode {kernel_mode!r}")
    if kernel_mode == "direct_out" and parts != 1: raise ValueError("Q4_K direct_out primitive requires parts=1")
    self._init_common(weight, bias, words, out_features, in_features, parts, opts, name,
      Q4KPrimitiveStorage(words, source_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes), route_role, route_admission)
    self.kernel_mode = kernel_mode

  def __call__(self, x:Tensor, **epilogue_inputs) -> Tensor:
    """Decode GEMV call. epilogue_inputs are threaded only when the fusion gate
    (decode_q4k_epilogue_fusion_promoted, closed default) is open and this linear is eligible; otherwise
    they are silently ignored and the legacy route is used -- the gate-off fallback
    stays live."""
    return self._call_with_program_facts(lambda: q4k_primitive_linear_call(self, x, self._fallback, self.route_admission.admitted, epilogue_inputs=epilogue_inputs))

class Q6KPrimitiveLinear(_QKPrimitiveLinear):
  _storage_attr, _prefill_attr, _ggml_type = "q6k_storage", "_prefill_q6k_halfs", 14
  def __init__(self, weight:Tensor, bias:Tensor|None, halfs:Tensor, out_features:int, in_features:int, parts:int, opts:tuple,
               name:str, source_bytes:int, persistent_bytes:int, storage_mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0, route_role:str="",
               route_admission:QKPrimitiveRouteAdmission|None=None):
    self._init_common(weight, bias, halfs, out_features, in_features, parts, opts, name,
      Q6KPrimitiveStorage(halfs, source_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes), route_role, route_admission)

  def __call__(self, x:Tensor, **epilogue_inputs) -> Tensor:
    """Decode GEMV call. epilogue_inputs are threaded to the fused variant only
    when the fusion gate is open; the legacy route ignores them."""
    return self._call_with_program_facts(lambda: q6k_primitive_linear_call(self, x, self._fallback, self.route_admission.admitted,
                                                                           epilogue_inputs=epilogue_inputs))

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
  ggml_type: int; label: QuantFormat; not_kind_counter: str; dtype: object; block_bytes: int
  generated_policy: object; route_choice: object; parse_opt: object; make_linear: object; installed_detail: object
  allowed_storage_modes: tuple[str, ...]

def _qk_target_promoted(route_plan:object|None, target:tuple[str|None, str|None]) -> bool:
  """TG3 policy check, tolerant of route_plan doubles that only implement `.primitive` (e.g. the
  generated-policy test double that must never be consulted at all -- see
  test_generated_policy_override_still_wins_over_route_plan). Absence of the method reads the same as
  `route_plan is None`: undecided-by-target, not denied (see ModelRoutePlan.target_promoted)."""
  check = getattr(route_plan, "target_promoted", None)
  return True if check is None else check(target)

def _route_choice(label:QuantFormat, route_plan:ModelRoutePlan|None, name:str, rows:int, cols:int, skipped:collections.Counter[str],
                  capability_ok:bool, target_promoted:bool, q4:bool=False) -> _QKInstallChoice|None:
  if route_plan is None or (entry := route_plan.primitive(name)) is None: skipped["policy_fallback"] += 1; return None
  if entry.quant_label != label.name or entry.rows != rows or entry.cols != cols: skipped["policy_unsupported"] += 1; return None
  # TG3: the two admission questions are independently decidable and reported with distinct census reasons --
  # QKPrimitiveCapability (renderer/device facts, tinygrad/llm/device_facts.py) answers (a); ModelRoutePlan
  # .target_promoted (the BoltBeam-sourced route-policy authority) answers (b). Neither is inferred here from
  # a target string.
  if not capability_ok: skipped["capability_missing"] += 1; return None
  if not target_promoted: skipped["policy_target_not_promoted"] += 1; return None
  return _QKInstallChoice(entry.module_path, entry.parts, entry.opts, entry.role, entry.kernel_mode if q4 else "partial")

def _q4_route_choice(route_plan, name, rows, cols, skipped, capability_ok, target_promoted):
  return _route_choice(Q4_K, route_plan, name, rows, cols, skipped, capability_ok, target_promoted, True)
def _q6_route_choice(route_plan, name, rows, cols, skipped, capability_ok, target_promoted):
  return _route_choice(Q6_K, route_plan, name, rows, cols, skipped, capability_ok, target_promoted)

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

def _make_q4(module, packed, rows, cols, choice, opts, name, sizes, storage_mode, route_admission):
  return Q4KPrimitiveLinear(module.weight, module.bias, packed, rows, cols, choice.parts, opts, name, sizes[0], sizes[1], storage_mode, sizes[2], sizes[3], kernel_mode=choice.kernel_mode, route_role=choice.route_role, route_admission=route_admission)

def _make_q6(module, packed, rows, cols, choice, opts, name, sizes, storage_mode, route_admission):
  return Q6KPrimitiveLinear(module.weight, module.bias, packed, rows, cols, choice.parts, opts, name, sizes[0], sizes[1], storage_mode, sizes[2], sizes[3], route_role=choice.route_role, route_admission=route_admission)

def _q4_detail(x): return f"{x.name}:mode={x.kernel_mode}:parts={x.parts}:opts={[str(o) for o in x.opts]}"
def _q6_detail(x): return f"{x.name}:parts={x.parts}:opts={[str(o) for o in x.opts]}"

_Q4_INSTALL_SPEC = _QKInstallSpec(12, Q4_K, "not_q4_k", dtypes.uint32, 144, _q4_generated_choice, _q4_route_choice,
  parse_opt, _make_q4, _q4_detail, ("sidecar", "q4_ondemand", "shared"))
_Q6_INSTALL_SPEC = _QKInstallSpec(14, Q6_K, "not_q6_k", dtypes.uint16, 210, _q6_generated_choice, _q6_route_choice,
  parse_opt, _make_q6, _q6_detail, ("sidecar", "shared"))

def _install_qk_primitives(model, gguf:pathlib.Path, meta:dict, spec:_QKInstallSpec, generated_policy:dict|None, budget:QKPrimitiveBudget|None,
                           storage_mode:str, route_plan:ModelRoutePlan|None, device_facts:object|None, debug:bool):
  if storage_mode not in spec.allowed_storage_modes: raise ValueError(f"unsupported {spec.label.name} primitive storage mode {storage_mode!r}")
  raw, installed, skipped = Tensor(gguf, dtype=spec.dtype), [], collections.Counter()
  budget = budget or QKPrimitiveBudget()
  if generated_policy is None and route_plan is None: route_plan = build_model_route_plan(meta)
  # TG3: capability (renderer/device facts) and promotion (route-plan policy) are properties of the resolved
  # target/plan, not of any individual tensor -- resolve each ONCE per install call, then apply uniformly so
  # every tensor's admission (or rejection, with its own census reason below) stays observable. This replaces
  # the pre-TG3 gate that decided, from a single hardcoded target-string match, whether this function even ran
  # at all (see model.py's use_q4k_primitive/use_q6k_primitive) -- it now always runs, and a target that cannot
  # be admitted is recorded in `skipped`, never silently dropped.
  capability = qk_primitive_capability_from_device_facts(device_facts)
  target_promoted = _qk_target_promoted(route_plan, (capability.backend, capability.architecture))
  route_admission = QKPrimitiveRouteAdmission(capability, target_promoted,
                                              decode_epilogue_fusion_promoted((capability.backend, capability.architecture)),
                                              decode_q4k_epilogue_fusion_promoted((capability.backend, capability.architecture)),
                                              decode_q4k_epilogue_resadd_promoted((capability.backend, capability.architecture)),
                                              decode_q4k_w1w3_fusion_promoted((capability.backend, capability.architecture)))
  for name, dims, typ, off in meta["tensor_infos"]:
    if typ != spec.ggml_type: skipped[spec.not_kind_counter] += 1; continue
    if len(dims) != 2: skipped["not_2d"] += 1; continue
    if not name.endswith(".weight"): skipped["not_weight"] += 1; continue
    rows, cols = tuple(reversed(dims))
    choice = (spec.route_choice(route_plan, name, rows, cols, skipped, capability.satisfied, target_promoted) if generated_policy is None else
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
    if not budget.reserve(name, persistent_bytes, spec.label.name): skipped["runtime_storage_cap"] += 1; continue
    if storage_mode == "shared": packed, shared_bytes = _shared_packed_view(meta, byte_start, source_bytes, spec.dtype), source_bytes
    else:
      source = raw[byte_start//spec.dtype.itemsize:byte_start//spec.dtype.itemsize+source_bytes//spec.dtype.itemsize]
      packed = source.contiguous() if storage_mode == "q4_ondemand" else _model_parameter_materialization(module.weight, source.to(None).contiguous())
      shared_bytes = 0
    sizes = (source_bytes, persistent_bytes, shared_bytes, source_bytes if storage_mode == "q4_ondemand" else 0)
    linear = spec.make_linear(module, packed, rows, cols, choice, tuple(spec.parse_opt(x) for x in choice.opt_specs), name, sizes, storage_mode, route_admission)
    _set_module_at(model, choice.module_path, linear)
    installed.append(linear)
  if debug:
    skipped_s = " ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
    summary, cap = _qk_storage_summary(installed), -1 if budget.cap_bytes is None else budget.cap_bytes
    print(f"{spec.label.name.replace('_', '')}_PRIMITIVE_DEBUG installed={len(installed)} skipped_total={sum(skipped.values())} {skipped_s} "
          f"source_bytes={summary['source_bytes']} storage_bytes={summary['persistent_bytes']} shared_bytes={summary['shared_bytes']} "
          f"nonpersistent_bytes={summary['nonpersistent_bytes']} runtime_cap_bytes={cap} runtime_cap_used_bytes={budget.used_bytes} storage_mode={storage_mode}")
    if installed: print(f"{spec.label.name.replace('_', '')}_PRIMITIVE_DEBUG installed_linears {' '.join(spec.installed_detail(x) for x in installed[:8])}"
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
