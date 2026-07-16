from __future__ import annotations
import contextlib, contextvars, functools, hashlib, itertools, json, pathlib
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, dtypes, function, Device, role_metadata
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import prod
from tinygrad.llm.admission import (
  AUTO_MAX_CONTEXT, AdmissionInputs, ExactSelectedModelPlan, plan_exact_selected_model_load,
  plan_selected_model_memory,
)
from tinygrad.llm.device_facts import scan_device_facts
from tinygrad.llm.gguf import MODEL_PARAMETER_ALLOCATION_OWNER, gguf_load, gguf_load_metadata, gguf_load_with_metadata
from tinygrad.llm.gguf_memory_scan import RuntimeGeometry, selected_gguf_backing_bytes
from tinygrad.llm import route_ops as qk_ops
from tinygrad.llm.decode_routes import FLASH_DECODE_CANDIDATE, flash_decode_attention_route
from tinygrad.llm.prefill_policy import (
  immutable_prefill_policy, prefill_concrete_kv_auto_decision, prefill_policy_strategy,
  prefill_policy_uses_overlay, prefill_v2_validate_ubatch, select_prefill_runtime_policy,
)
from tinygrad.llm.prefill_routes import is_direct_packed_prefill_linear, route_prefill_linear
from tinygrad.llm.prefill_route_census import PrefillRouteAttachment, prefill_forward_scope, record_prefill_route
from tinygrad.llm.qk_primitives import (
  QKConfig, QKPrimitiveBudget, Q4KPrimitiveLinear, Q4KPrimitiveRegistry, Q6KPrimitiveLinear,
  _demote_q6k_to_q4, _install_q4k_fusions, _install_q4k_primitives, _install_q6k_primitives, _qk_storage_summary,
  qk_primitive_eligibility_from_device_facts,
)
from tinygrad.llm.model_facts import model_facts_from_gguf_metadata
from tinygrad.llm.memory_adaptive_authority import resolve_memory_adaptive_policy
from tinygrad.llm.memory_semantics import (KV_CACHE, MODEL_PARAMETER, PREFILL_OUTPUT, RUNTIME_INPUT, RUNTIME_OUTPUT,
                                           RUNTIME_PERSISTENT, bind_memory_semantic_owner, kv_cache, materialize_runtime_input,
                                           runtime_input_materialization,
                                           memory_semantic_owner, model_parameter,
                                           prefill_activation, prefill_output, prefill_scratch, runtime_activation,
                                           runtime_input, runtime_output,
                                           runtime_persistent, runtime_scratch)
from tinygrad.llm.model_route_plan import build_model_route_plan
from tinygrad.llm.route_policy import should_use_flash_decode as _route_should_use_flash_decode
from tinygrad.llm.physical_memory_ledger import AllocationOwner, bind_allocation_owner
from tinygrad.uop.ops import Ops, resolve

_MEMORY_ADAPTIVE_MEASUREMENT_AUTHORITY = contextvars.ContextVar("_memory_adaptive_measurement_authority", default=None)

_GGUF_TENSOR_OWNER = MODEL_PARAMETER_ALLOCATION_OWNER
_KV_CACHE_OWNER = AllocationOwner("kv_cache", "model")
_RUNTIME_PERSISTENT_OWNER = AllocationOwner("runtime_persistent", "model")

def _bind_tensor_owner(tensor:Tensor, owner:AllocationOwner) -> Tensor:
  """Attach semantic ownership to a lazy Tensor's eventual physical base."""
  bind_allocation_owner(tensor.uop.buffer, owner)
  return tensor

def _mark_physical_semantic(tensor:Tensor, mark) -> Tensor:
  """Bind a persistent value's owner without changing its executable graph."""
  owner = {kv_cache: KV_CACHE, model_parameter: MODEL_PARAMETER, runtime_input: RUNTIME_INPUT,
           runtime_persistent: RUNTIME_PERSISTENT}.get(mark)
  if owner is None: raise ValueError("physical semantic marker must name a persistent allocation class")
  bind_memory_semantic_owner(tensor.uop.buf_uop, owner)
  return tensor

def _runtime_input_boundary(tensor:Tensor) -> Tensor:
  """Give feedback decode a fresh input buffer without reclassifying its prefill output allocation."""
  physical_owners = {memory_semantic_owner(uop) for uop in tensor.uop.toposort() if uop.op is Ops.BUFFER}
  output_owners = {PREFILL_OUTPUT, RUNTIME_OUTPUT}
  if memory_semantic_owner(tensor) in output_owners or physical_owners & output_owners:
    if physical_owners - {None, *output_owners}:
      raise ValueError(f"feedback input aliases conflicting physical semantic owners: {physical_owners!r}")
    return runtime_input(tensor.clone())
  # CLONE gives the runtime input a fresh lazy allocation boundary while the
  # ownership carrier remains outside symbolic view/index lowering.
  return _mark_physical_semantic(tensor.clone(), runtime_input)

def _bind_state_dict_owners(state_dict:dict[str, Tensor]) -> None:
  # State values can be lazy views/dequantizations. Bind their physical source leaves, using one model-level owner
  # independent of the names which reference a tied/shared base.
  for tensor in state_dict.values():
    for uop in tensor.uop.toposort():
      if uop.op is Ops.BUFFER:
        bind_allocation_owner(uop.buffer, _GGUF_TENSOR_OWNER)
    model_parameter(tensor)

@contextlib.contextmanager
def _memory_adaptive_measurement_authority(*, device_facts, inventory:dict, workload:dict, collector):
  """Private authority used only by the isolated whole-model measurement seam.

  The authority binds a collector to the exact selected inventory, workload,
  and immutable load-entry DeviceFacts object. It is intentionally not a load
  argument and cannot be supplied by a production model caller.
  """
  if not callable(collector): raise TypeError("measurement collector must be callable")
  authority = (device_facts, inventory, workload, collector)
  token = _MEMORY_ADAPTIVE_MEASUREMENT_AUTHORITY.set(authority)
  try: yield
  finally: _MEMORY_ADAPTIVE_MEASUREMENT_AUTHORITY.reset(token)

def derive_selected_gguf_prefill_inventory(kv:dict, meta:dict, ubatch:int=512) -> dict:
  """Derive runtime route identity only from the explicitly opened GGUF's tensor metadata."""
  facts = model_facts_from_gguf_metadata(kv, meta)
  rows = []
  for tensor in facts.tensors:
    if tensor.role is None: continue
    candidate_controlled = tensor.quant_label in ("Q4_K", "Q6_K")
    # The lazy lm-head projection is physically M=1 after final-token pruning. Other selected prefill linears
    # execute at the concrete ubatch. Fixed rows remain census obligations, but are outside candidate geometry.
    physical_m = 1 if tensor.role == "lm_head" and not candidate_controlled else ubatch
    semantic = {"tensor_identity": tensor.name, "role": tensor.role, "quant_format": tensor.quant_label,
                "candidate_controlled": candidate_controlled,
                "shape": {"m": physical_m, "n": tensor.rows, "k": tensor.cols}}
    if not candidate_controlled: semantic["fixed_route_id"] = "fixed-ggml-linear"
    invocation_id = "invocation:sha256:" + hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rows.append({**semantic, "invocation_id": invocation_id})
  # Some GGUFs tie the runtime output projection to token_embd.weight and omit output.weight. The loaded runtime
  # still executes model.output, so materialize that fixed invocation from the selected source tensor instead of
  # leaving a real prefill linear outside the census.
  if not any(row["role"] == "lm_head" for row in rows):
    tied = next((tensor for tensor in facts.tensors if tensor.name == "token_embd.weight"), None)
    if tied is not None:
      semantic = {"tensor_identity": "output.weight", "source_tensor_identity": tied.name, "role": "lm_head",
                  "quant_format": tied.quant_label, "candidate_controlled": False, "fixed_route_id": "fixed-ggml-linear",
                  "shape": {"m": 1, "n": tied.rows, "k": tied.cols}}
      invocation_id = "invocation:sha256:" + hashlib.sha256(json.dumps(
        semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
      rows.append({**semantic, "invocation_id": invocation_id})
  rows.sort(key=lambda x: x["invocation_id"])
  identity = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  return {"schema": "tinygrad.model_runtime_prefill_inventory.v2", "inventory_identity": identity, "rows": rows}

def _selected_inventory_routes(inventory:dict, candidate_route_id:str) -> dict[str, str]:
  return {row["invocation_id"]: (candidate_route_id if row.get("candidate_controlled") is True else row["fixed_route_id"])
          for row in inventory.get("rows", ())}

def select_memory_adaptive_runtime_policy(*, kv:dict, meta:dict, device_facts, ubatch:int=512,
                                          selected_model_source:str|None=None):
  """Consume an exact measured/cache result, or truthfully select the direct packed baseline.

  Normal loads always select the baseline. The only non-baseline authority is
  the private isolated-measurement context above.
  """
  inventory = derive_selected_gguf_prefill_inventory(kv, meta, ubatch)
  invocation_ids = tuple(row["invocation_id"] for row in inventory["rows"])
  request = {"schema": "tinygrad.model_memory_adaptive_request.v1", "inventory": inventory,
             "device_facts": device_facts.planning_snapshot(), "workload": {"prefill_ubatch": ubatch}}
  authority = _MEMORY_ADAPTIVE_MEASUREMENT_AUTHORITY.get()
  selected = None
  if authority is not None:
    authority_facts, expected_inventory, expected_workload, collector = authority
    if authority_facts is not device_facts:
      raise ValueError("measurement authority does not own the immutable load-entry DeviceFacts snapshot")
    if expected_inventory != inventory or expected_workload != request["workload"] or \
       authority_facts.planning_snapshot() != request["device_facts"]:
      raise ValueError("measurement authority does not exactly match selected inventory, workload, and scanned facts")
    selected = collector(request)
  elif selected_model_source is not None:
    # The model path is the only user-selected authority. The controller owns
    # its own live scan, candidate enumeration, guarded runs, and exact cache;
    # this load accepts the result only after matching it back to this exact
    # opened inventory, workload, and immutable load-entry DeviceFacts scan.
    source = resolve_memory_adaptive_policy(selected_model_source)
    if source is not None:
      from extra.qk.memory_adaptive_runtime_collector import collect_runtime_policy
      selected = collect_runtime_policy(request, source)
  if selected is None:
    return immutable_prefill_policy({"strategy": "DIRECT_PACKED_FALLBACK", "candidate_id": "direct-packed-baseline",
      "routes": _selected_inventory_routes(inventory, "direct-packed-baseline"),
      "provenance": "no production policy collector available",
      "measured": False, "inventory_identity": inventory["inventory_identity"]})
  if not isinstance(selected, dict): raise TypeError("memory-adaptive policy collector must return a mapping or None")
  if selected.get("validated_request") != request:
    raise ValueError("memory-adaptive policy does not exactly match selected GGUF inventory, workload, and scanned device facts")
  validation = selected.get("validation")
  trial = validation == "measurement_trial" and authority is not None
  if selected.get("decision") != "SELECTED" or (validation not in ("exact_cache", "measured") and not trial):
    raise ValueError("memory-adaptive policy must be a cached/validated SELECTED result")
  policy = immutable_prefill_policy(selected["policy"])
  if prefill_policy_strategy(policy) == "BOUNDED_PACKED_TILES":
    raise ValueError("BOUNDED_PACKED_TILES has no production runtime binder; collector may not promote it yet")
  if tuple(sorted(policy["routes"])) != tuple(sorted(invocation_ids)):
    raise ValueError("memory-adaptive policy route coverage does not exactly match selected GGUF inventory")
  fixed_routes = {row["invocation_id"]: row["fixed_route_id"] for row in inventory["rows"]
                  if row.get("candidate_controlled") is False}
  if any(policy["routes"].get(invocation_id) != route_id for invocation_id, route_id in fixed_routes.items()):
    raise ValueError("memory-adaptive policy changed a fixed selected-inventory route")
  memory_facts = policy.get("memory_facts")
  bundle = None
  if memory_facts is not None or policy.get("memory_fact_evidence") is not None:
    from extra.qk.memory_adaptive_allocation_observer import validate_memory_facts
    bundle = validate_memory_facts(policy.get("memory_fact_evidence"), candidate_id=policy["candidate_id"])
    if bundle is None or memory_facts != bundle["facts"]:
      raise ValueError("memory-adaptive policy memory_facts are not bound to complete measured evidence")
  if prefill_policy_strategy(policy) != "DIRECT_PACKED_FALLBACK" and bundle is None and not trial:
    raise ValueError("accelerated memory-adaptive policy requires complete measured memory_facts")
  return policy

_EXACT_ROUTE_MEMORY_KEYS = ("resident_copies", "candidate_workspace_bytes", "batch_size", "kv_element_bytes",
                            "runtime_persistent_bytes", "peak_prefill_activation_bytes", "peak_prefill_output_bytes",
                            "peak_prefill_scratch_bytes")

def _attach_selected_prefill_inventory(model, inventory:dict, policy, scanned_target_facts) -> None:
  """Attach metadata-selected rows to their exact runtime owning linears."""
  routes = dict(policy.get("routes", {})) if policy is not None else {}
  rows = inventory.get("rows", ())
  ids = [row.get("invocation_id") for row in rows]
  if len(ids) != len(set(ids)): raise ValueError("duplicate selected prefill inventory rows")
  if set(routes) != set(ids): raise ValueError("selected policy and inventory attachments differ")
  attached = set()
  for row in rows:
    tensor_identity = row.get("tensor_identity")
    if not isinstance(tensor_identity, str) or not tensor_identity.endswith(".weight"):
      raise ValueError(f"selected prefill tensor identity is not an exact runtime weight path: {tensor_identity!r}")
    obj = model
    try:
      for component in tensor_identity[:-7].split("."):
        obj = obj[int(component)] if component.isdigit() and isinstance(obj, (list, tuple)) else getattr(obj, component)
    except (AttributeError, IndexError, TypeError) as exc:
      raise ValueError(f"selected prefill tensor {tensor_identity!r} has no exact runtime linear") from exc
    if hasattr(obj, "_prefill_route_attachment"): raise ValueError(f"duplicate runtime attachment for {tensor_identity!r}")
    invocation_id = row["invocation_id"]
    setattr(obj, "_prefill_route_attachment", PrefillRouteAttachment(invocation_id, routes[invocation_id], tensor_identity,
                                                                       policy, scanned_target_facts))
    attached.add(invocation_id)
  if attached != set(routes): raise ValueError("selected prefill inventory did not attach exactly once")

def _graph_gemm_registry(policy):
  graph = policy.get("graph_gemm") if policy is not None else None
  candidate_set = graph.get("candidate_set") if isinstance(graph, dict) else None
  if not isinstance(candidate_set, dict): return None
  try:
    from extra.qk.runtime_specs import FullKernelCandidateSet, admit_full_kernel_candidate_set
    return admit_full_kernel_candidate_set(FullKernelCandidateSet.from_json(candidate_set))
  except (KeyError, TypeError, ValueError): return None

def _graph_gemm_binding(policy, registry, role:str, shape:tuple[int, int, int], device_facts):
  """Build only an exact selected-row attachment; absent/ambiguous policy remains ordinary GEMM fallback."""
  graph = policy.get("graph_gemm") if policy is not None else None
  rows = graph.get("policy_rows") if isinstance(graph, dict) else None
  if registry is None or not isinstance(rows, list): return None
  expected_shape = dict(zip(("m", "n", "k"), shape))
  matches = [row for row in rows if isinstance(row, dict) and row.get("role") == role and row.get("shape") == expected_shape]
  if len(matches) != 1: return None
  row = matches[0]
  inventory_identity, candidate_set_identity = row.get("inventory_identity"), row.get("candidate_set_identity")
  if inventory_identity != policy.get("inventory_identity") or candidate_set_identity != graph.get("candidate_set_identity"): return None
  snapshot = device_facts.planning_snapshot() if hasattr(device_facts, "planning_snapshot") else device_facts
  if not isinstance(snapshot, dict): return None
  capabilities = snapshot.get("capabilities", {})
  target = {"backend": snapshot.get("backend"), "arch": snapshot.get("architecture"), "wave_size": capabilities.get("wave_size")}
  if row.get("target") != target: return None
  compile_artifacts = graph.get("compile_artifacts", {})
  compile_artifact = row.get("compile_artifact")
  if compile_artifact is None and isinstance(compile_artifacts, dict):
    compile_artifact = compile_artifacts.get(row.get("candidate_identity"))
  return {"candidate_registry": registry, "inventory_identity": inventory_identity,
          "candidate_set_identity": candidate_set_identity, "scanned_target_facts": {"target": target},
          "selected_policy": row, "compile_artifact": compile_artifact}

# Prefill v2 (opt-in, default off; decode 100% untouched when off). Concrete-ubatch fp16 prefill that lets
# tensor cores apply + the loop-found TC schedule warm-start in. See docs/amd-decode-prefill-v2-gate-20260616.md.
# Costs ~fp16-covered-weight-size extra VRAM (it coexists with the Q4_K decode storage; ~+14GB for 8B) -> OOMs
# small cards. `PREFILL_V2=auto` resolves on/off from detected VRAM in from_gguf (see prefill_v2_auto_decision +
# docs/prefill-default-policy-evaluation-result-20260620.md). Explicit 0/1 always wins.
PREFILL_V2 = False
PREFILL_UBATCH = 512
# Phase-3 routing fix (default ON under PREFILL_V2): route a sub-UBATCH prompt remainder through ONE shifted
# prefill-v2 chunk instead of the slow 32-token symbolic fallback. PREFILL_REMAINDER_FIX=0 reverts. See
# docs/prefill-route-schedule-result-20260620.md.
PREFILL_REMAINDER_FIX = True
# Generated graph GEMM (within PREFILL_V2): the manifest-promoted gfx1100 pp512 candidate set is default-on for its
# exact four dense roles. The manifest owns promotion and supplies the complete runtime environment; this module only
# checks target applicability from the load-entry DeviceFacts scan. Explicit 0/1 remains an override.
PREFILL_GRAPH_GEMM = False

# Concrete-KV prefill (opt-in, default off): pass a CONCRETE start_pos per prefill chunk so KV=start_pos+T is
# concrete -> the attention's reduce tiles/TC fires (symbolic KV blocks it). ~1.24x e2e, byte-identical. Cost: a
# separate concrete prefill jit per distinct start_pos (0,512,...), precompiled at load -> best WARM/server prefill
# but a load-time precompile tax that loses for cold one-shot short prompts. `PREFILL_CONCRETE_KV=auto` enables it
# only under the server profile (see prefill_concrete_kv_auto_decision). See
# docs/prefill-default-policy-evaluation-result-20260620.md, docs/prefill-concrete-kv-policy-result-20260620.md.
# PREFILL_SERVER_PROFILE=1 is a convenience: serve >1 generation / long prompts -> implies PREFILL_V2=auto (if V2
# unset) + concrete-KV on (when V2 ends up on). One-shot short prompts must NOT set it.
# Legacy environment spelling retained for compatibility. Internally this is only a workload-reuse intent.
PREFILL_WORKLOAD_REUSE = False
PREFILL_CONCRETE_KV = False
# P2: explicit TC attention (Q@Kᵀ TC + fp16 scores + softmax + P@V TC, GQA broadcast) for prefill on CONCRETE KV
# (the only regime where the concrete-shape tensor core fires; symbolic KV blocked it -> 0.79x in-model). Needs
# PREFILL_CONCRETE_KV. Research, dNLL-gated. See docs/prefill-concrete-kv-build-scope-20260619.md.
# DEFAULT-ON on its validated target, selected from the load-entry facts before JIT capture. Only active under
# PREFILL_V2; the route's isinstance(start_pos,int) guard keeps
# it to CONCRETE chunks (start_pos=0 by default), the only validated regime. Set PREFILL_TC_ATTN=0/1 to override.
# Concrete first chunk: cuts attention ~18%->~5%, reproducible ~1.16x whole-forward, BYTE-IDENTICAL (rel_RMSE 0.0,
# dNLL 0.0, greedy-exact, 3 sessions). A FUSION win, NOT tensor cores (WMMA does not fire; no warmstart TC-opt for
# attention shapes, no BEAM). See docs/prefill-branch-b-tc-attention-result-20260620.md.
PREFILL_TC_ATTN = False
# HISTORY: the earlier env `PREFILL_TC_ATTENTION` probe reported ~0.8x "REFUTED in-model" -- that was a BROKEN
# harness: it set the typo'd env `PREFILL_TC_ATTENTION` (model reads PREFILL_TC_ATTN) so both arms ran SDPA, AND
# it bound a symbolic start_pos that fails the concrete-int guard so the path never fired. Overturned 2026-06-20
# (correct concrete-int, same-process interleaved synced A/B). See docs/prefill-branch-b-tc-attention-result-20260620.md.
# The loop-found per-shape TC schedule (gate-validated; NO BEAM -- BEAM hangs gfx1100). Forced onto the
# prefill-v2 fp16 matmuls via _WARMSTART_OPTS by shape key. 4x4 is permanently excluded on gfx1100 because the
# generated path hits the VGPR wall; retired experiments do not remain selectable production modes.
def _prefill_v2_without_parked_4x4(opts:tuple) -> tuple:
  up = {o.axis: o.arg for o in opts if o.op is OptOps.UPCAST}
  if up.get(0) == 4 and up.get(1) == 4:
    return tuple(Opt(o.op, o.axis, 2) if o.op is OptOps.UPCAST and o.axis == 1 else o for o in opts)
  return opts

def _prefill_v2_opts(out_f:int, in_f:int) -> tuple:
  # UNROLL(reduce,8): unrolling the K loop makes each thread's global->LDS copy loads contiguous, so they fold
  # from per-element global_load_d16 (+ ~8 v_mov register-init/WMMA) to wide global_load_b128 (~2 v_mov/WMMA).
  # +3.7% pp512, no VGPR spill (UNROLL,4 spills 362), dNLL -0.00013. See docs/prefill-cgw3-copy-unroll-result-20260619.md.
  u0 = 4 if in_f > out_f else 2
  u1 = 4
  return _prefill_v2_without_parked_4x4((Opt(OptOps.TC, 0, (-1, 2, 1)), Opt(OptOps.UPCAST, 0, u0), Opt(OptOps.UPCAST, 1, u1),
                                         Opt(OptOps.UNROLL, 0, 8)))

def _pf16(lin, x:Tensor) -> Tensor:
  return route_prefill_linear(lin, x, prefill_graph_gemm=bool(getattr(lin, "_prefill_graph_gemm", False)))

def _prefill_semantic(enabled:bool, mark, value:Tensor) -> Tensor:
  """Attach the same logical value role in either prefill or runtime/decode."""
  runtime_mark = {prefill_activation: runtime_activation, prefill_output: runtime_output,
                  prefill_scratch: runtime_scratch}.get(mark)
  if runtime_mark is None: raise ValueError("execution semantic requires a prefill role marker")
  return (mark if enabled else runtime_mark)(value)

def _generation_input_slice(tokens:Tensor, start_pos:int|UOp, token_extent:UOp, bound_extent:int) -> Tensor:
  """Retain the lazy symbolic slice used by decode and chunked prefill JITs."""
  return tokens[:, start_pos:start_pos + token_extent]

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  with role_metadata("rope"):
    cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  full_attention_interval: int = 0
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  admit: dict|None = None   # auto-scan admission report (free/budget/weights/kv/prefill terms); None if not resolved
  prefill_memory_plan: str|None = None  # immutable canonical JSON; sole byte/strategy authority for this load
  prefill_policy: object|None = None     # immutable measured policy, or truthful direct-packed baseline
  prefill_device_facts: object|None = None  # the single immutable load-entry DeviceFacts scan
  exact_memory_plan: ExactSelectedModelPlan|None = None  # selected-GGUF ledger and admission decision
  prefill_graph_gemm: bool = False       # selected once from the immutable candidate policy
  prefill_tc_attn: bool = False          # selected once from the immutable candidate policy
  prefill_v2: bool = False               # concrete candidate binding, never a module-global switch
  prefill_ubatch: int = 512              # candidate-local physical M
  prefill_concrete_kv: bool = False      # workload/candidate policy decision
  prefill_remainder_fix: bool = True     # declared candidate tail capability
  prefill_workload_reuse: bool = False
  flash_decode: bool = False             # exact target/shape candidate binding
  lm_head_route: str = "lazy"           # explicit workload semantics; generate consumes only the final token
  kv_quant: bool = False    # KV-quant long-ctx tier: store KV as int8 + fp16 per-(K|V,head,token) scale (halves resident KV)
  ring: bool = False        # StreamingLLM streaming tier (lossy): unbounded logical ctx in the N-token buffer via eviction

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config
    self._prefill_graph_gemm_registry = _graph_gemm_registry(config.prefill_policy)

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = nn.Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = nn.Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = nn.Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    _prefill = getattr(self, "_is_prefill", False)
    if getattr(self, '_prefill_v2', False) and not hasattr(self, 'ffn_gate_exps') and not hasattr(self, 'ffn_gateup'):
      # prefill v2 (dense): fp16 + .contiguous()-isolated matmuls so each is a clean, warmstart-matchable TC
      # kernel (mirrors the gated chained-FFN prefill authority shape). MoE/fused fall through.
      fused = qk_ops.route_prefill_q4k_gate_up(self.ffn_gate, self.ffn_up, x)
      if fused is None:
        g = _prefill_semantic(_prefill, prefill_activation, _pf16(self.ffn_gate, x).contiguous())
        u = _prefill_semantic(_prefill, prefill_activation, _pf16(self.ffn_up, x).contiguous())
      else:
        g, u = (_prefill_semantic(_prefill, prefill_activation, v.contiguous()) for v in fused)
      h = _prefill_semantic(_prefill, prefill_activation, (g.silu() * u).contiguous())
      return _prefill_semantic(_prefill, prefill_activation, _pf16(self.ffn_down, h).contiguous())
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      if hasattr(self, 'exp_probs_b'):
        probs = logits.sigmoid()
        _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
        probs = probs.gather(-1, sel)
        if self.config.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      else:
        vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
        probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      expert_h = _prefill_semantic(_prefill, prefill_activation,
        (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())
      x_down = self.ffn_down_exps(sel, expert_h)  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp_h = _prefill_semantic(_prefill, prefill_activation, self.ffn_gate_shexp(x).silu().contiguous())
        shexp = self.ffn_down_shexp(shexp_h * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    if hasattr(self, "ffn_gateup"):  # B1 fused gate/up
      gate, up = self.ffn_gateup(x)
      return self.ffn_down(_prefill_semantic(_prefill, prefill_activation, gate.silu().contiguous()) * up)
    gated = _prefill_semantic(_prefill, prefill_activation, self.ffn_gate(x).silu().contiguous())
    return self.ffn_down(gated * self.ffn_up(x))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, ring_freqs=None) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # StreamingLLM ring: pass the per-step freqs as an ARGUMENT into _run (a nested @function) so it is a true graph
    # input -- reading it off self inside _run would bake it at _run's compile time (attributes don't rebind).
    _rf = getattr(self, "_ring_freqs", None)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, ring_freqs):
      _prefill = getattr(self, "_is_prefill", False)
      with role_metadata("rms_norm"): normed_x = _prefill_semantic(_prefill, prefill_scratch, self.attn_norm(x))
      attn_out = self._attention(normed_x, start_pos, ring_freqs)
      with role_metadata("residual"): h = _prefill_semantic(_prefill, prefill_activation, x + attn_out)
      with role_metadata("rms_norm"): normed_h = _prefill_semantic(_prefill, prefill_scratch, self.ffn_norm(h))
      ffn_out = self._feed_forward(normed_h)
      with role_metadata("residual"):
        return _prefill_semantic(_prefill, prefill_activation, (h + ffn_out).contiguous())
    # @function wraps the traced return in a call/gettuple node. Mark that concrete block-output boundary as well as
    # its residual creation site so callification cannot hide the allocation identity from the manifest.
    return _prefill_semantic(getattr(self, "_is_prefill", False), prefill_activation, _run(x, start_pos, _rf).contiguous())

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = nn.Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = nn.Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = nn.Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = nn.Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp, ring_freqs=None) -> Tensor:
    _prefill = getattr(self, "_is_prefill", False)
    if getattr(self, '_prefill_v2', False) and not hasattr(self, "attn_qkv"):  # prefill v2: fp16 isolated q/k/v
      q, k, v = (_prefill_semantic(_prefill, prefill_scratch, _pf16(lin, x).contiguous())
                 for lin in (self.attn_q, self.attn_k, self.attn_v))
    elif hasattr(self, "attn_qkv"): q, k, v = self.attn_qkv(x)  # B1 fused q/k/v
    else: q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    q, k, v = (_prefill_semantic(_prefill, prefill_scratch, value) for value in (q, k, v))
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim:
      with role_metadata("rms_norm"):
        q, k = (_prefill_semantic(_prefill, prefill_scratch, norm(value))
                for norm, value in ((self.attn_q_norm, q), (self.attn_k_norm, k)))

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim:
      with role_metadata("rms_norm"):
        q, k = (_prefill_semantic(_prefill, prefill_scratch, norm(value))
                for norm, value in ((self.attn_q_norm, q), (self.attn_k_norm, k)))

    # rope-at-read (DECODE_ROPE_AT_READ, opt-in; requires full-head rope): store UN-roped K and rotate at read -- the
    # prerequisite for the StreamingLLM ring's position re-basing. Q is never cached, so it is always roped here.
    # The ring supplies a per-step PRE-GATHERED freqs table via the JIT-input attribute _ring_freqs (slot-relative
    # positions); when unset the baked self.freqs_cis is used (absolute positions). _ring_freqs implies rope-at-read.
    _ring_freqs = ring_freqs
    # rope-at-read active if: env flag, OR a ring-decode step (freqs supplied), OR the ring is active this generation
    # (covers PREFILL, which must ALSO store un-roped K so the ring decode reads it consistently).
    _rope_read = (_ring_freqs is not None or getattr(self, "_ring_active", False)) \
                 and self.config.rope_dim == self.config.head_dim
    _fr = _ring_freqs if _ring_freqs is not None else self.freqs_cis
    # full-ring (ctx>=N): the buffer is full and the write slot wraps, so the live read length is the WHOLE buffer N
    # (all slots valid), not start_pos+T (start_pos is the wrapped write slot, not a length). Selects [0:N] reads + Tc=N.
    _ring_full = getattr(self, "_ring_full", False)
    _rl = self.config.max_context if _ring_full else (start_pos + T)
    # Q is roped via _fr (the gathered ring table when ring, else freqs_cis) indexed by start_pos: in the full ring
    # start_pos is the write slot wp, and _fr[wp] = freqs[pos_of(wp)] = the query's (newest) position -> consistent
    # with the K positions. In fill / non-ring, _fr == freqs_cis and start_pos is the absolute position (unchanged).
    q = _prefill_semantic(_prefill, prefill_scratch,
                          apply_rope(q[..., :self.config.rope_dim], _fr[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1))
    if not _rope_read:
      k = _prefill_semantic(_prefill, prefill_scratch,
                            apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1))

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    if self.config.kv_quant and _rope_read:
      raise NotImplementedError("KV-quant + rope-at-read not yet composed (Q8 stores roped K; the ring's un-roped K "
                                "path is validated fp16 first). Disable one of DECODE_KV_QUANT / DECODE_ROPE_AT_READ.")
    if self.config.kv_quant:
      # KV-quant write: symmetric per-(K|V, head, token) int8 (absmax over head_dim). k is already roped, so we store
      # roped-then-quantized K (Q8 is orthogonal to RoPE). Store int8 KV + fp16 scale; dequant the re-slice to fp16 for
      # the non-flash (SDPA/prefill) consumers (per-layer transient). The flash route reads int8+scale natively.
      _Hkv = self.config.n_kv_heads
      _kv = _prefill_semantic(_prefill, prefill_scratch, Tensor.stack(k, v))       # [2,B,Hkv,T,Hd] fp16
      _sc = _prefill_semantic(_prefill, prefill_scratch, (_kv.abs().max(axis=-1, keepdim=True) / 127.0).maximum(1e-8))
      _kvq = _prefill_semantic(_prefill, prefill_scratch, (_kv / _sc).round().cast(dtypes.int8))
      _sch = _prefill_semantic(_prefill, prefill_scratch, _sc.reshape(2, B, _Hkv, T).cast(dtypes.float16))
      _st_kv = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(_kvq.uop)
      _st_sc = self.cache_kv_scale[:, :, :, start_pos:start_pos+T].uop.store(_sch.uop)
      assigned_kv = Tensor(self.cache_kv.uop.after(_st_kv))
      assigned_scale = Tensor(self.cache_kv_scale.uop.after(_st_sc))
      _ksc = assigned_scale[0, :, :, 0:start_pos+T].reshape(B, _Hkv, start_pos+T, 1)
      _vsc = assigned_scale[1, :, :, 0:start_pos+T].reshape(B, _Hkv, start_pos+T, 1)
      k = _prefill_semantic(_prefill, prefill_scratch, assigned_kv[0, :, :, 0:start_pos+T, :].cast(dtypes.float16) * _ksc)
      v = _prefill_semantic(_prefill, prefill_scratch, assigned_kv[1, :, :, 0:start_pos+T, :].cast(dtypes.float16) * _vsc)
    else:
      assigned_scale = None
      assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
      _kfull = assigned_kv[0]
      if _rope_read:
        # rope-at-read for the NON-flash (SDPA/prefill) consumers: K is stored un-roped. Rotate the FULL concrete-MAXC
        # cache (positions 0..MAXC-1) THEN slice -- roping before the slice avoids indexing freqs by a SYMBOLIC bound
        # (start_pos+T's vmax can exceed MAXC). The full-MAXC rope is DCE'd on the flash decode path (k unused there;
        # the kernel ropes in-register from `freqs`); it materializes only for prefill/SDPA. Unwritten slots (>ctx) are
        # roped garbage but sliced away below. Explicit (not apply_rope) so the concrete last dim needs no -1 inference.
        _rd = self.config.rope_dim; _hh = _rd // 2; _mc = self.config.max_context
        _cos = _fr[:, :_hh].reshape(1, 1, _mc, _hh); _sin = _fr[:, _hh:].reshape(1, 1, _mc, _hh)
        _k1, _k2 = _kfull[..., :_hh], _kfull[..., _hh:_rd]
        _kfull = (_k1 * _cos - _k2 * _sin).cat(_k2 * _cos + _k1 * _sin, _kfull[..., _rd:], dim=-1)
      k = _kfull[:, :, 0:_rl, :]
      v = assigned_kv[1, :, :, 0:_rl, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    # The model owns two separately captured decode graphs. The immutable
    # candidate binding selects the flash graph upstream; ring decode always
    # uses it because its wrapped write slot is not a logical context length.
    if _ring_freqs is not None or getattr(self, "_use_flash", False):
      Hq, Hkv, Hd = self.config.n_heads, self.config.n_kv_heads, self.config.head_dim
      out = flash_decode_attention_route(q, assigned_kv, start_pos, T, B, Hq, Hkv, Hd, self.config.max_context,
                                         kv_scale=assigned_scale, freqs=(_fr if _rope_read else None),
                                         ring_full=_ring_full)
      attn = out.reshape(B, Hq, T, Hd).cast(q.dtype)
    elif self.config.prefill_tc_attn and getattr(self, '_prefill_v2', False) and isinstance(start_pos, int) and resolve(T != 1):
      # P2: Option-B explicit TC attention on CONCRETE KV. Q@Kᵀ (TC) -> fp16 scores -> softmax -> P@V (TC),
      # GQA via broadcast (K/V per kv-head expanded over the G group dim). Concrete KV=start_pos+T -> TC fires.
      Hkv, Hd, KV = self.config.n_kv_heads, self.config.head_dim, start_pos + T
      G = self.config.n_heads // Hkv; scale = Hd ** -0.5
      qg = q.reshape(B, Hkv, G, T, Hd).cast(dtypes.float16)
      kg = k.reshape(B, Hkv, 1, KV, Hd).cast(dtypes.float16)
      vg = v.reshape(B, Hkv, 1, KV, Hd).cast(dtypes.float16)
      with role_metadata("attn_score"): scores = _prefill_semantic(_prefill, prefill_scratch, (qg @ kg.transpose(-1, -2)).float() * scale)
      with role_metadata("attn_mask"): scores = _prefill_semantic(_prefill, prefill_scratch, scores + mask.reshape(1, 1, 1, T, KV))
      with role_metadata("softmax"): s = _prefill_semantic(_prefill, prefill_scratch, scores.softmax(-1))
      with role_metadata("attn_av"):
        attn = (s.cast(dtypes.float16) @ vg).reshape(B, self.config.n_heads, T, Hd).cast(q.dtype)  # (B,H,T,Hd)
    else:
      attn = _prefill_semantic(_prefill, prefill_scratch,
                               q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True))  # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    out_in = attn if not self.config.attn_output_gate else (attn * gate.sigmoid())
    if getattr(self, '_prefill_v2', False):
      return _prefill_semantic(_prefill, prefill_activation, _pf16(self.attn_output, out_in).contiguous())
    return _prefill_semantic(_prefill, prefill_activation, self.attn_output(out_in))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # The promoted generated decode-attention shape was validated with fp16 K/V cache storage (TG-P14 KV_BOTH parity
      # and roofline closeout). Keep that fact-defined shape on fp16 so the generated tile reads the same cache dtype
      # the promotion measured; other shapes keep the default dtype.
      _generated_decode_shape_supported = qk_primitive_eligibility_from_device_facts( \
        getattr(self.config, "prefill_device_facts", None)).eligible and x.shape[0] == 1 and self.config.n_heads == 32 \
        and self.config.n_kv_heads == 8 and self.config.head_dim == 128
      _kv_dtype = dtypes.float16 if _generated_decode_shape_supported else None
      # Admission guardrail (B5): assert the ACTUAL cache_kv bytes (with the real dtype) fit the admitted VRAM budget
      # before allocating -- converts a silent late OOM into a clear, actionable failure. O(1), no re-probe: the
      # admission report carries free/budget/weights so we re-derive the KV allowance and check this model's total KV.
      _admit = getattr(self.config, "admit", None)
      if _admit and _admit.get("mode", "").split("+")[0] in ("auto", "explicit") and "budget_gb" in _admit:
        _elem = 1 if self.config.kv_quant else (2 if _kv_dtype is dtypes.float16 else dtypes.default_float.itemsize)
        _scale_b = (2 * x.shape[0] * self.config.n_kv_heads * self.config.max_context * 2) if self.config.kv_quant else 0
        _block_kv = 2 * x.shape[0] * self.config.n_kv_heads * self.config.max_context * self.config.head_dim * _elem + _scale_b
        _total_kv = _block_kv * self.config.num_blocks
        _allow = (_admit["budget_gb"] - _admit["weights_gb"] - _admit.get("flash_scratch_gb", 0.0)) * 1e9 \
                 - _admit.get("prefill_gb_per_1k", 0.0) * self.config.max_context / 1000 * 1e9
        if _total_kv > _allow:
          raise RuntimeError(
            f"KV cache admission guard: max_context={self.config.max_context} needs {_total_kv/1e9:.1f}GB of KV "
            f"(dtype {'fp16' if _elem==2 else dtypes.default_float.name}) but only {_allow/1e9:.1f}GB is admissible "
            f"(budget {_admit['budget_gb']:.1f}GB after scanned reserve minus weights {_admit['weights_gb']:.1f}GB + prefill peak). "
            f"Reduce --max_context or use auto. This is the guardrail that prevents a silent OOM.")
      if self.config.kv_quant:
        # KV-quant tier: resident KV is int8 (half the bytes) + a per-(K|V, head, token) fp16 scale buffer. The decode
        # flash route dequants in-register (int8*scale); non-flash consumers dequant their re-sliced K/V to fp16 (per-
        # layer transient). Model-agnostic -- keyed off config.kv_quant, no model-name check.
        self.cache_kv = _mark_physical_semantic(_bind_tensor_owner(Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context,
          self.config.head_dim, dtype=dtypes.int8, device=x.device), _KV_CACHE_OWNER), kv_cache)
        self.cache_kv_scale = _mark_physical_semantic(_bind_tensor_owner(Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context,
          dtype=dtypes.float16, device=x.device), _KV_CACHE_OWNER), kv_cache)
      else:
        self.cache_kv = _mark_physical_semantic(_bind_tensor_owner(Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context,
          self.config.head_dim, dtype=_kv_dtype, device=x.device), _KV_CACHE_OWNER), kv_cache)
      self.freqs_cis = _mark_physical_semantic(_bind_tensor_owner(precompute_freqs_cis(self.config.rope_dim, self.config.max_context,
        self.config.rope_theta, device=x.device), _RUNTIME_PERSISTENT_OWNER), runtime_persistent)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = nn.Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = nn.Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = nn.Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = nn.Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, ring_freqs=None) -> Tensor:
    _prefill = getattr(self, "_is_prefill", False)
    mark_scratch = lambda value: _prefill_semantic(_prefill, prefill_scratch, value)
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    if self.config.q_lora_rank > 0:
      q_a = mark_scratch(self.attn_q_a(x))
      with role_metadata("rms_norm"): q_a_normed = self.attn_q_a_norm(q_a)
      q_proj = mark_scratch(self.attn_q_b(q_a_normed))
    else: q_proj = mark_scratch(self.attn_q(x))
    q = mark_scratch(q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2))
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T]), dim=-1)

    kv_a = mark_scratch(self.attn_kv_a_mqa(x))
    with role_metadata("rms_norm"): c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = apply_rope(
      kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2),
      self.freqs_cis[start_pos:start_pos+T])

    k_store = mark_scratch(c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1))
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    with role_metadata("attn_score"): attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None:
      with role_metadata("attn_mask"): attn = attn + mask
    with role_metadata("softmax"): attn = attn.softmax(-1)
    with role_metadata("attn_av"): attn = attn @ v
    attn = (attn @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return _prefill_semantic(_prefill, prefill_activation, self.attn_output(attn))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = _mark_physical_semantic(_bind_tensor_owner(Tensor.empty(x.shape[0], 1, self.config.max_context,
        self.config.kv_lora_rank + self.config.rope_dim, device=x.device), _KV_CACHE_OWNER), kv_cache)
      self.freqs_cis = _mark_physical_semantic(_bind_tensor_owner(precompute_freqs_cis(self.config.rope_dim, self.config.max_context,
        self.config.rope_theta, device=x.device), _RUNTIME_PERSISTENT_OWNER), runtime_persistent)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv, self.attn_gate = nn.Linear(config.dim, self.conv_channels, bias=False), nn.Linear(config.dim, ssm.inner_size, bias=False)
    self.ssm_alpha, self.ssm_beta = nn.Linear(config.dim, self.num_v_heads, bias=False), nn.Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), nn.Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, ring_freqs=None) -> Tensor:
    _prefill = getattr(self, "_is_prefill", False)
    mark_scratch = lambda value: _prefill_semantic(_prefill, prefill_scratch, value)
    B, T, _ = x.shape
    assert T == 1, "GatedDeltaNetBlock currently only supports T=1"

    # input processing
    x = mark_scratch(x.half())
    out_gate = mark_scratch(self.attn_gate(x).reshape(B, 1, self.num_v_heads, self.head_v_dim))
    beta = mark_scratch(self.ssm_beta(x).sigmoid().reshape(B, self.num_v_heads, 1, 1))
    alpha = mark_scratch(((self.ssm_alpha(x).float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, self.num_v_heads, 1, 1).exp())

    # qkv conv
    conv_window = self.conv_state.cat(self.attn_qkv(x), dim=1)
    conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = q.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    k = k.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    v = v.reshape(B, self.num_v_heads, self.head_v_dim)
    q, k, v = q.mul(self.head_k_dim**-0.5).unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)

    # recurrent
    recurrent_state = self.recurrent_state * alpha
    recurrent_state = recurrent_state + ((v - recurrent_state@k) * beta)@k.transpose(-1, -2)

    # store the updated state
    conv_state_store = self.conv_state.uop.store(conv_window[:, 1:, :].cast(self.conv_state.dtype).uop)
    recurrent_state_store = self.recurrent_state.uop.store(recurrent_state.cast(self.recurrent_state.dtype).uop)
    recurrent_state = Tensor(self.recurrent_state.uop.after(recurrent_state_store, conv_state_store))

    # output
    core_attn_out = self.ssm_norm((recurrent_state@q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim))
    return _prefill_semantic(_prefill, prefill_activation, self.ssm_out((core_attn_out * out_gate.silu()).reshape(B, 1, -1).cast(x.dtype)))

  # recurrent state can't be partially reused after divergence, force a full rebuild
  def _state_reset_ops(self):
    return [self.conv_state.assign(self.conv_state.const_like(0)),
            self.recurrent_state.assign(self.recurrent_state.const_like(0))] if hasattr(self, "conv_state") else []
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = _mark_physical_semantic(Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone(), runtime_persistent)
      self.recurrent_state = _mark_physical_semantic(Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_v_dim, device=x.device).clone(), runtime_persistent)

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(config, config.ssm) if config.ssm and (i+1) % config.full_attention_interval != 0 else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.config = config
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    self._q4k_linears = Q4KPrimitiveRegistry()
    # we specialize the JIT for prefill and rollout; rollout_jit_flash is the context-aware flash-decode
    # decode graph (captured lazily when ctx crosses FLASH_DECODE_THRESHOLD), see __call__/generate.
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    self.rollout_jit_flash = TinyJit(self.forward)
    self.rollout_jit_ring = TinyJit(self.forward_ring)        # ring FILL phase (ctx<N): read [0:start_pos+T], identity freqs
    self.rollout_jit_ring_full = TinyJit(self.forward_ring)   # ring FULL phase (ctx>=N): read [0:N], wrapped write slot + gathered freqs
    # The selected prefill candidate gets a separate concrete-M capture.
    self.prefill_v2_jit = TinyJit(self.forward)
    self.prefill_v2_jits: dict = {}   # concrete-KV: one prefill jit per concrete start_pos (PREFILL_CONCRETE_KV)
    # prefill v2 warmstart table is built here but installed into the global codegen knob ONLY for the duration
    # of the prefill-v2 forward (see __call__), to contain that ambient power rather than leave it set process-wide.
    self._pf16_warmstart:dict|None = None
    if config.prefill_v2:
      prefill_v2_validate_ubatch(config.prefill_ubatch)
      self._pf16_warmstart = self._build_prefill_v2_warmstart()

  @property
  def prefill_memory_plan(self) -> str|None:
    return self.config.prefill_memory_plan

  @property
  def prefill_policy(self): return self.config.prefill_policy

  # the dense FFN + attn projection linears prefill-v2 accelerates (per block)
  _PREFILL_V2_LINEARS = ("ffn_gate", "ffn_up", "ffn_down", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
                         "attn_q", "attn_k", "attn_v", "attn_output")
  def _prefill_v2_role_for_name(self, name:str) -> str:
    if name in ("ffn_gate", "ffn_up", "ffn_gate_shexp", "ffn_up_shexp"): return "ffn_gate_up"
    if name in ("ffn_down", "ffn_down_shexp"): return "ffn_down"
    if name in ("attn_q", "attn_output"): return "attn_qo"
    if name in ("attn_k", "attn_v"): return "attn_kv"
    return ""

  def _prefill_v2_dims(self, lin):
    out_f = getattr(lin, "out_features", None) or lin.weight.shape[0]
    in_f  = getattr(lin, "in_features", None)  or lin.weight.shape[1]
    return (out_f, in_f) if isinstance(out_f, int) and isinstance(in_f, int) else (None, None)

  def _prefill_v2_covered(self):
    # (linear, out_f, in_f) for each covered linear with a known concrete shape -- single source for the
    # warmstart table, the VRAM estimate, and the realization, so they can't drift apart.
    for block in self.blk:
      for n in self._PREFILL_V2_LINEARS:
        lin = getattr(block, n, None)
        if lin is None or getattr(lin, "weight", None) is None: continue
        setattr(lin, "_prefill_selected_strategy", prefill_policy_strategy(self.config.prefill_policy))
        setattr(lin, "_prefill_graph_gemm", self.config.prefill_graph_gemm)
        device_facts = getattr(self.config, "prefill_device_facts", None)
        setattr(lin, "_prefill_device_facts", device_facts)
        if not getattr(lin, "name", ""): setattr(lin, "name", n)
        role = self._prefill_v2_role_for_name(n)
        setattr(lin, "_prefill_graph_role", role)
        out_f, in_f = self._prefill_v2_dims(lin)
        if out_f is not None:
          binding = _graph_gemm_binding(self.config.prefill_policy, getattr(self, "_prefill_graph_gemm_registry", None), role,
                                        (getattr(self.config, "prefill_ubatch", 512), out_f, in_f), device_facts)
          if binding is not None: setattr(lin, "_prefill_graph_gemm_binding", binding)
          yield lin, out_f, in_f
    # LM head is lazy by default: inference consumes only the final token's logits, so preserving self.output(x)
    # lets the downstream [:, -1, :] prune a 512-token vocabulary projection to one token. Full-sequence LM-head
    # materialization is an explicit workload policy, not a prefill default. Only resident_fp16 needs inclusion in
    # the warmstart/VRAM/realize set; direct_packed intentionally keeps _pf16_w absent.
    lin = getattr(self, "output", None)
    if lin is not None and is_direct_packed_prefill_linear(lin) and self.config.lm_head_route == "resident_fp16":
      setattr(lin, "_prefill_device_facts", getattr(getattr(self, "config", None), "prefill_device_facts", None))
      setattr(lin, "_prefill_graph_role", "lm_head")
      out_f, in_f = self._prefill_v2_dims(lin)
      if out_f is not None: yield lin, out_f, in_f

  def _build_prefill_v2_warmstart(self) -> dict:
    # The loop-found per-shape TC schedule for the prefill-v2 fp16 FFN/attn matmuls, keyed by the in-model
    # kernel signature (verified): (frozenset({out_features, PREFILL_UBATCH}), in_features). Shapes are
    # config-fixed so this is computable at init from the (pre-primitive) nn.Linears. A kernel whose key is
    # absent (e.g. a silu-fused gate) keeps the heuristic; a key that applies-then-errors falls back too
    # (postrange.py). NOT set into the global here -- installed only around the prefill-v2 forward (__call__).
    def _opts(out_f, in_f): return _prefill_v2_without_parked_4x4(_prefill_v2_opts(out_f, in_f))
    return {(frozenset({out_f, self.config.prefill_ubatch}), in_f): _opts(out_f, in_f)
            for _, out_f, in_f in self._prefill_v2_covered()}

  def realize_prefill_v2_weights(self) -> int:
    # Realize a clean fp16 weight per covered linear (cached as `_pf16_w`, read by _pf16). The primitives' lazy
    # Q4_K/Q6_K->fp16 dequant graph, used raw, fuses into the matmul -> ~3% peak (no TC win); a realized fp16
    # buffer makes the prefill-v2 matmul a real TC GEMM (~13x prefill on 8B). COST: ~fp16-model-size extra VRAM
    # (it coexists with the Q4_K decode storage). Gated/opt-in; called at the end of from_gguf when PREFILL_V2.
    # Environment limits are diagnostics only. The immutable load plan is the safety authority and cannot be bypassed.
    policy = getattr(self.config, "prefill_policy", None)
    if policy is None:
      # Compatibility for manually constructed/test models predating the runtime policy field.
      if json.loads(getattr(self.config, "prefill_memory_plan", None) or "{}").get("decision") != "FULL_RESIDENT_OVERLAY": return 0
    elif not prefill_policy_uses_overlay(policy): return 0
    covered = list(self._prefill_v2_covered())
    n = 0
    for lin, _, _ in covered:
      lin._pf16_w = _mark_physical_semantic(lin.weight.cast(dtypes.float16).contiguous().realize(), model_parameter); n += 1
    return n

  def precompile_concrete_prefill_jits(self) -> int:
    # Increment 0 ship: with PREFILL_CONCRETE_KV, every prefill chunk runs through a per-start_pos CONCRETE jit
    # (-> the fusion attention path, 1.7-4.4x/chunk faster than the symbolic chunk). Those jits compile on first
    # use (~5s each), so a COLD long prompt pays the tax inline. Precompiling them ONCE at load (here) moves the
    # tax to load time, so every generation -- including the first -- is warm. Bounded: ceil(max_context/UBATCH)
    # jits. Safe to leave the dummy KV behind: a fresh model's first generation starts at start_pos=0 and
    # overwrites the cache in chunk order before any position is read. gfx1100/PREFILL_V2/PREFILL_CONCRETE_KV only.
    if not (self.config.prefill_v2 and self.config.prefill_concrete_kv): return 0
    ubatch = self.config.prefill_ubatch
    temp = materialize_runtime_input(Tensor([0.0]).contiguous())
    dummy = materialize_runtime_input(Tensor.zeros(1, ubatch, dtype="int32").contiguous())
    n = 0
    for sp in range(0, self.max_context - ubatch + 1, ubatch):
      self(dummy, sp, temp, use_flash=False).realize(); n += 1   # populates self.prefill_v2_jits[sp]
    return n

  # Full-sequence LM-head routing is explicit. The default lazy path is both faster for inference (the final-token
  # slice prunes M=512 to M=1) and lower-memory; resident/direct M=512 routes remain available for full-logit jobs.
  def _lm_head_wants_pf16(self) -> bool:
    return (self.config.lm_head_route != "lazy" and bool(self.blk) and
            getattr(self.blk[0], '_prefill_v2', False) and is_direct_packed_prefill_linear(self.output))

  def logits(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    _prefill = resolve(tokens.shape[1] != 1)
    x = _prefill_semantic(_prefill, prefill_activation, self.token_embd(tokens).float())  # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    with role_metadata("rms_norm"): x = _prefill_semantic(_prefill, prefill_scratch, self.output_norm(x))
    if self._lm_head_wants_pf16(): return _prefill_semantic(_prefill, prefill_output, _pf16(self.output, x).contiguous())
    # The lazy LM head is still the selected output tensor's actual runtime invocation.  Record it before Tensor's
    # downstream final-token pruning; the prefill-forward context prevents the same call during decode from counting.
    record_prefill_route(self.output)
    return _prefill_semantic(_prefill, prefill_output, self.output(x))

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    # This runs inside the TinyJit body: feedback's lazy clone/store is captured in the decode schedule instead of
    # being realized by JIT input preparation as a separate per-token copy and synchronization.
    tokens = _runtime_input_boundary(tokens)
    logits = self.logits(tokens, start_pos)[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    sampled = (logits / temperature.maximum(1e-12) -
               (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)
    return _prefill_semantic(resolve(tokens.shape[1] != 1), prefill_output, sampled)

  def forward_ring(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, freqs:Tensor) -> Tensor:
    # StreamingLLM ring decode: `freqs` is a per-step JIT INPUT (the slot-relative pre-gathered cos|sin table). Set it
    # on each block from INSIDE the traced fn so it binds to the graph input (a baked attribute would capture once).
    for block in self.blk: block._ring_freqs = freqs
    return self.forward(tokens, start_pos, temperature)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, use_flash:bool=False,
               ring_freqs:Tensor|None=None, ring_full:bool=False) -> Tensor:
    is_prefill = resolve(tokens.shape[1] != 1)
    # prefill v2: only when opt-in AND this is a CONCRETE-batch prefill chunk. Normal prefill passes a symbolic
    # v_toks (tokens.shape[1] is a UOp -> not int), so the two paths never collide; decode is T==1.
    is_prefill_v2 = self.config.prefill_v2 and is_prefill and isinstance(tokens.shape[1], int)
    for q4k_linear in self._q4k_linears.linears:
      q4k_linear.decode_enabled = not is_prefill
    # context-aware flash: each block reads _use_flash at trace time; rollout_jit (SDPA) and
    # rollout_jit_flash bake distinct attention -- each is only ever called with its own use_flash, so
    # capture is consistent. The decode-only T==1 guard in _attention ignores it during prefill.
    for block in self.blk:
      block._use_flash, block._prefill_v2, block._is_prefill, block._ring_freqs, block._ring_full = \
        use_flash, is_prefill_v2, is_prefill, None, ring_full
    # StreamingLLM ring decode: distinct captured graphs with `freqs` as a per-step JIT input (rebound each token). The
    # FULL-phase graph (ring_full, ctx>=N) reads the whole [0:N] cache and writes at the wrapped slot; the FILL-phase
    # graph reads [0:start_pos+T] like normal decode. block._ring_full (baked bool) selects the read mode in _attention.
    if ring_freqs is not None and not is_prefill:
      _rjit = self.rollout_jit_ring_full if ring_full else self.rollout_jit_ring
      return _rjit(tokens, start_pos, temperature, ring_freqs)
    # concrete-KV: a CONCRETE int start_pos (KV concrete -> attention TC fires) gets a per-start_pos jit.
    if is_prefill_v2 and isinstance(start_pos, int):
      jit = self.prefill_v2_jits.setdefault(start_pos, TinyJit(self.forward))
    else:
      jit = (self.prefill_v2_jit if is_prefill_v2 else self.prefill_jit) if is_prefill else \
            (self.rollout_jit_flash if use_flash else self.rollout_jit)
    if not is_prefill_v2:
      with prefill_forward_scope(is_prefill): return jit(tokens, start_pos, temperature)
    # contain the ambient codegen power: install the warmstart table ONLY around the prefill-v2 forward (it's
    # consulted at kernel-compile time, i.e. this jit's first call), then restore -- decode/other paths never
    # see a populated _WARMSTART_OPTS even within this process.
    import tinygrad.codegen.opt.postrange as pr
    with prefill_forward_scope(True), pr.warmstart_candidate_state(self._pf16_warmstart):
      return jit(tokens, start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:"int|str|None"=None,
                realize:bool=False, stream:str="auto") -> tuple[Transformer, dict]:
    # Probe free VRAM at ENTRY, before gguf_load makes the weight storage resident -- so `free` is the baseline
    # available for weights+KV (the admission budget then subtracts weights itself; probing after gguf_load would
    # double-count weights already in `used`). Total is stable regardless.
    _measurement_authority = _MEMORY_ADAPTIVE_MEASUREMENT_AUTHORITY.get()
    _device_facts = scan_device_facts() if _measurement_authority is None else _measurement_authority[0]
    # Decode primitives are installed only when the selected GGUF and the one
    # live target scan satisfy their structural candidate contract.
    _qk_eligible = qk_primitive_eligibility_from_device_facts(_device_facts).eligible
    use_q4k_primitive = use_q6k_primitive = not isinstance(gguf, Tensor) and _qk_eligible
    _authority_workload = _measurement_authority[2] if _measurement_authority is not None else {}
    _prefill_ubatch = int(_authority_workload.get("prefill_ubatch", PREFILL_UBATCH))
    if _prefill_ubatch <= 0: raise ValueError("selected prefill candidate requires a positive physical M")
    _requested_max_context, _admit_resolved, _ring_admitted = max_context, False, False
    _kv_quant = False
    _overlay_request = None
    _workload_reuse = False
    _runtime_policy = immutable_prefill_policy({"strategy": "DIRECT_PACKED_FALLBACK", "candidate_id": "direct-packed-baseline",
      "routes": {}, "provenance": "preloaded tensors have no selected-GGUF inventory", "measured": False})
    _cov = tuple(f"{n}.weight" for n in Transformer._PREFILL_V2_LINEARS)
    def _print_admission(plan, kv_tag:str, cap_text:str):
      admit = plan.report
      print(f"max_context={admit['mode']} -> {plan.max_context} "
            f"(free {admit.get('free_gb', float('nan')):.1f}GB, budget {admit.get('budget_gb', float('nan')):.1f}GB "
            f"with scanned reserve {admit.get('reserve_gb', float('nan')):.1f}GB, weights {admit.get('weights_gb', plan.weights/1e9):.1f}GB, "
            f"KV{kv_tag} {admit.get('kv_gb_per_1k', plan.kv_per_tok*1000/1e9):.2f}GB/1k, "
            f"prefill-peak {admit.get('prefill_gb_per_1k', plan.prefill_per_tok*1000/1e9):.2f}GB/1k, {cap_text})")
      if admit.get("banner"): print(admit["banner"])
    _runtime_inventory = None
    _exact_memory_plan = None
    _route_memory = {}
    if not isinstance(gguf, Tensor):
      _admit_kv, _admit_meta = gguf_load_metadata(gguf)
      _runtime_inventory = derive_selected_gguf_prefill_inventory(_admit_kv, _admit_meta, _prefill_ubatch)
      _runtime_policy = select_memory_adaptive_runtime_policy(kv=_admit_kv, meta=_admit_meta,
                                                               device_facts=_device_facts, ubatch=_prefill_ubatch,
                                                               selected_model_source=str(pathlib.Path(gguf).expanduser().resolve()))
      _overlay_request = prefill_policy_uses_overlay(_runtime_policy)
      _admit_arch = _admit_kv["general.architecture"]
      _admit_n_heads, _admit_n_kv_heads = _admit_kv[f"{_admit_arch}.attention.head_count"], _admit_kv[f"{_admit_arch}.attention.head_count_kv"]
      _admit_head_dim = _admit_kv.get(f"{_admit_arch}.attention.key_length_mla",
                                      _admit_kv.get(f"{_admit_arch}.attention.key_length",
                                                    _admit_kv[f"{_admit_arch}.embedding_length"] // _admit_n_heads))
      num_blocks = _admit_kv[f"{_admit_arch}.block_count"] - _admit_kv.get(f"{_admit_arch}.nextn_predict_layers", 0)
      trained_ctx = _admit_kv[f"{_admit_arch}.context_length"]
      _provided_route_memory = _runtime_policy.get("memory_facts")
      if _provided_route_memory is not None and not isinstance(_provided_route_memory, dict):
        raise ValueError("prefill policy memory_facts must be a mapping when present")
      _route_memory = dict(_provided_route_memory or {})
      if _route_memory and any(_route_memory.get(key) is None for key in _EXACT_ROUTE_MEMORY_KEYS):
        raise ValueError("prefill policy contains partial memory_facts; exact allocation evidence must be complete")
      if (prefill_policy_strategy(_runtime_policy) != "DIRECT_PACKED_FALLBACK" and not _route_memory and
          _MEMORY_ADAPTIVE_MEASUREMENT_AUTHORITY.get() is None):
        raise ValueError("accelerated prefill policy cannot bypass exact memory admission")
      if isinstance(_runtime_policy.get("memory_fact_evidence"), dict):
        _route_memory["provenance"] = json.dumps(_runtime_policy["memory_fact_evidence"]["provenance"],
                                                   sort_keys=True, separators=(",", ":"))
      _route_memory["candidate_id"] = _runtime_policy["candidate_id"]
      _q4_bytes = selected_gguf_backing_bytes(gguf, _device_facts.capabilities.global_allocation_granularity)
      if _q4_bytes is None:
        raise RuntimeError(f"{_admit_arch}: selected-GGUF backing allocation is unknown from the selected path and scanned allocation granularity")
      _est_fp16 = sum(prod(dims) * 2 for name, dims, _, _ in _admit_meta["tensor_infos"] if any(name.endswith(s) for s in _cov))
      _admit_rope_dim = _admit_kv.get(f"{_admit_arch}.rope.dimension_count", _admit_head_dim)
      _stream = str(stream)
      _plan, _memory_plan, _effective_strategy = plan_selected_model_memory(AdmissionInputs(
        _requested_max_context, trained_ctx, _device_facts.free_vram_bytes, _q4_bytes, _est_fp16, num_blocks, _admit_n_heads,
        _admit_n_kv_heads, _admit_head_dim, _prefill_ubatch, _overlay_request is not False, False,
        f"{_admit_arch} selected GGUF", stream=_stream, rope_dim=_admit_rope_dim, kv_quant_supported=True,
        kv_quant_disabled=False, live_split_s=FLASH_DECODE_CANDIDATE.split_size),
        _device_facts, direct_packed_supported=True, overlay_requested=_overlay_request)
      _v2_on = prefill_policy_strategy(_runtime_policy) in ("FULL_RESIDENT_OVERLAY", "BOUNDED_PACKED_TILES", "DIRECT_PACKED_FALLBACK")
      max_context, _kv_quant = _plan.max_context, _plan.kv_quant
      _admit = dict(_plan.report)
      # A completed machine-search record may carry a fully attributed allocation ledger. Apply it only after context
      # and KV representation are resolved. Ordinary direct-packed loading needs no caller-injected hardware facts.
      if all(_route_memory.get(key) is not None for key in _EXACT_ROUTE_MEMORY_KEYS):
        _geometry = RuntimeGeometry(num_blocks, _admit_n_kv_heads, _admit_head_dim, max_context, _prefill_ubatch,
          batch_size=_route_memory["batch_size"], kv_element_bytes=_route_memory["kv_element_bytes"],
          kv_scale_element_bytes=_route_memory.get("kv_scale_element_bytes"),
          kv_scales_per_token=_route_memory.get("kv_scales_per_token", 0),
          runtime_persistent_bytes=_route_memory["runtime_persistent_bytes"],
          peak_prefill_activation_bytes=_route_memory["peak_prefill_activation_bytes"],
          peak_prefill_output_bytes=_route_memory["peak_prefill_output_bytes"],
          peak_prefill_scratch_bytes=_route_memory["peak_prefill_scratch_bytes"])
        _exact_memory_plan = plan_exact_selected_model_load(gguf, metadata=(_admit_kv, _admit_meta), geometry=_geometry,
                                                             route_memory_facts=_route_memory, facts=_device_facts)
        if not _exact_memory_plan.decision.admitted:
          raise RuntimeError(f"{_admit_arch}: exact selected-GGUF memory admission refused: "
                             f"{'; '.join(_exact_memory_plan.decision.reasons)}")
        _admit["exact_memory_decision"] = json.dumps(_exact_memory_plan.decision.to_dict(), sort_keys=True, separators=(",", ":"))
      _ring_admitted = _admit.get("ring", False)
      _print_admission(_plan, "(int8)" if _kv_quant else "", f"trained {trained_ctx}, fp16-cap {_admit.get('mc_fp16', '-')}, q8-cap {_admit.get('mc_q8', '-')}")
      _admit_resolved = True
    if use_q4k_primitive or use_q6k_primitive:
      kv, state_dict, q4k_meta = gguf_load_with_metadata(gguf)
    else:
      kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
      q4k_meta = None
    _bind_state_dict_owners(state_dict)
    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    if not _admit_resolved:
      num_blocks = kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
      trained_ctx = kv[f'{arch}.context_length']
      _q4_bytes = pathlib.Path(gguf).stat().st_size if not isinstance(gguf, Tensor) else 0
      _est_fp16 = sum(t.numel() * 2 for k, t in state_dict.items() if any(k.endswith(s) for s in _cov))
      _plan, _memory_plan, _effective_strategy = plan_selected_model_memory(AdmissionInputs(
        _requested_max_context, trained_ctx, _device_facts.free_vram_bytes, _q4_bytes, _est_fp16, num_blocks, n_heads, n_kv_heads, head_dim,
        _prefill_ubatch, _overlay_request is not False, False, f"{arch} selected model",
        stream=str(stream), rope_dim=rope_dim, live_split_s=FLASH_DECODE_CANDIDATE.split_size),
        _device_facts, direct_packed_supported=True, overlay_requested=_overlay_request)
      _v2_on = prefill_policy_strategy(_runtime_policy) in ("FULL_RESIDENT_OVERLAY", "BOUNDED_PACKED_TILES", "DIRECT_PACKED_FALLBACK")
      max_context, _kv_quant, _admit = _plan.max_context, _plan.kv_quant, _plan.report
      _ring_admitted = _admit.get("ring", False)
      _print_admission(_plan, "", f"trained {trained_ctx}, mem-cap {_admit.get('mc_mem', '-')}")

    _runtime_policy = select_prefill_runtime_policy(_runtime_policy, scanned_device_facts=_device_facts,
      workload_reuse=_workload_reuse)
    if _runtime_policy["prefill_graph_gemm"]:
      # Candidate artifacts are validated only after the scanned target contract selects their applicability.
      qk_ops.qk_route_manifest_attr("promoted_prefill_candidate_policy")()
    _workload_reuse = bool(_runtime_policy.get("workload_reuse", False))
    _concrete_kv, _ = prefill_concrete_kv_auto_decision(_workload_reuse, _v2_on)
    _flash_decode = FLASH_DECODE_CANDIDATE.bind(1, n_heads, n_kv_heads, head_dim,
                                                _device_facts.selected_device) is not None

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe')),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      full_attention_interval=kv.get(f'{arch}.full_attention_interval', 0),
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      admit=_admit, prefill_memory_plan=_plan.prefill_memory_plan, prefill_policy=_runtime_policy,
      prefill_device_facts=_device_facts, exact_memory_plan=_exact_memory_plan,
      prefill_graph_gemm=bool(_runtime_policy["prefill_graph_gemm"]), prefill_tc_attn=bool(_runtime_policy["prefill_tc_attn"]),
      prefill_v2=_v2_on, prefill_ubatch=_prefill_ubatch, prefill_concrete_kv=_concrete_kv,
      prefill_remainder_fix=True, prefill_workload_reuse=_workload_reuse, flash_decode=_flash_decode, lm_head_route="lazy",
      kv_quant=_kv_quant, ring=_ring_admitted)
    # FAST_EMPTY_INIT: every weight is REPLACED by load_state_dict below, so building the ~254 random init graphs
    # (nn.Linear Tensor.uniform / nn.Embedding glorot_uniform) is wasted work (~2.3s of the load, per profiling).
    # Init EMPTY during construction instead -- correct because nothing reads the random values before they're replaced.
    _saved_init = (Tensor.__dict__.get("uniform"), Tensor.__dict__.get("glorot_uniform"))
    _fe = lambda *shape, **kw: Tensor.empty(*shape)
    Tensor.uniform = Tensor.glorot_uniform = _fe
    try:
      model = Transformer(config)
    finally:
      for _n, _v in zip(("uniform", "glorot_uniform"), _saved_init):
        delattr(Tensor, _n) if _v is None else setattr(Tensor, _n, _v)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    if q4k_meta is not None:
      model_facts = model_facts_from_gguf_metadata(kv, q4k_meta)
      route_plan = build_model_route_plan(q4k_meta, model_facts)
      # The production candidate views the selected GGUF backing directly;
      # it never creates an unchecked model-sized sidecar.
      qk_cfg = QKConfig(False, None, "shared", "shared", False, False, False, (), False)
      primitive_linears = []
      primitive_budget = QKPrimitiveBudget(qk_cfg.max_storage_bytes, qk_cfg.generated_policy_strict)
      q4_storage_mode, q6_storage_mode = qk_cfg.storage_mode, qk_cfg.q6_storage_mode
      if use_q4k_primitive: primitive_linears += _install_q4k_primitives(model, pathlib.Path(gguf), q4k_meta, None,
        primitive_budget, q4_storage_mode, route_plan, device_facts=_device_facts)
      if use_q6k_primitive: primitive_linears += _install_q6k_primitives(model, pathlib.Path(gguf), q4k_meta, None,
        primitive_budget, q6_storage_mode, route_plan, device_facts=_device_facts)
      if qk_cfg.storage_debug:
        summary = _qk_storage_summary(primitive_linears)
        cap = -1 if primitive_budget.cap_bytes is None else primitive_budget.cap_bytes
        by_kind_s = ",".join(f"{k}:{v}" for k, v in summary["by_kind"].items()) or "none"
        by_mode_s = ",".join(f"{k}:{v}" for k, v in summary["by_mode"].items()) or "none"
        print(f"QK_PRIMITIVE_STORAGE_DEBUG installed={len(primitive_linears)} source_bytes={summary['source_bytes']} "
              f"storage_bytes={summary['persistent_bytes']} shared_bytes={summary['shared_bytes']} "
              f"nonpersistent_bytes={summary['nonpersistent_bytes']} runtime_cap_bytes={cap} "
              f"runtime_cap_used_bytes={primitive_budget.used_bytes} by_kind={by_kind_s} by_mode={by_mode_s} "
              f"requested_storage_mode={q4_storage_mode} q4_effective_storage_mode={q4_storage_mode} "
              f"q6_effective_storage_mode={q6_storage_mode}")
      if primitive_linears and qk_cfg.demote_targets:  # B3: requant over-provisioned Q6 tensors -> Q4 (searched set)
        primitive_linears = _demote_q6k_to_q4(model, primitive_linears, qk_cfg.demote_targets)
      if primitive_linears: model._q4k_linears = Q4KPrimitiveRegistry(primitive_linears)
      if primitive_linears and qk_cfg.fuse_q4k:  # B1 horizontal-fusion probe -- REFUTED, experimental only
        import sys as _sys
        print("WARNING: Q4K_FUSE is REFUTED (qk-gemv-final-mile-20260617.md): ~-18% decode + prefill fallback "
              "broken on T>32. Enabled only for experiments; do NOT use for shipped benchmarks.", file=_sys.stderr)
        _install_q4k_fusions(model)
    if _runtime_inventory is not None:
      _attach_selected_prefill_inventory(model, _runtime_inventory, _runtime_policy, _device_facts)
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    # prefill v2 (opt-in): realize fp16 weights now that primitives are installed (shapes/dequant graphs ready)
    if config.prefill_v2: model.realize_prefill_v2_weights()
    # Increment 0 ship: with PREFILL_CONCRETE_KV, precompile the per-start_pos concrete prefill jits at load so the
    # ~5s/jit compile tax is paid once here, not inline on a cold prompt -> every generation is warm.
    if config.prefill_v2 and config.prefill_concrete_kv: model.precompile_concrete_prefill_jits()
    return model, kv

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def reset_generation_state(self) -> None:
    """Forget request-local generation state before an independent prompt.

    Dense attention caches are overwritten by the next prompt from position
    zero. Recurrent blocks additionally own state which must be explicitly
    cleared; keep that reset here so callers and benchmarks do not need to
    know which architecture they loaded.
    """
    resets = [reset for block in self.blk for reset in block._state_reset_ops()]
    if resets: Tensor.realize(*resets)
    self._cached_tokens = []

  def _ring_slot(self, logical_pos:int, N:int, sinks:int) -> int:
    # StreamingLLM ring write slot: FILL in order (the first `sinks` tokens land in slots 0..sinks-1); once FULL,
    # round-robin the WINDOW (slots sinks..N-1) with modulus N-sinks so the sink slots are never overwritten.
    if logical_pos < N: return logical_pos
    return sinks + ((logical_pos - sinks) % (N - sinks))

  def _ring_gather_freqs(self, freqs_full:Tensor, logical_pos:int, N:int, sinks:int) -> Tensor:
    # Pre-gather the rotary table so row==slot -> freqs_full[pos_of(slot)]. FILL: identity (row==slot==abs position, so
    # ctx<N is token-identical). FULL: sinks keep positions 0..sinks-1; window slot s gets recency position
    # N-1 - ((wp - s) mod (N-sinks)) so the newest slot (wp) is N-1 and the oldest window slot is `sinks` -> all rotary
    # phases stay in [0,N) <= trained ctx. Rebuilt each step (shared across layers) and fed as the ring JIT freqs input.
    if logical_pos < N: return freqs_full
    W = N - sinks
    wp = sinks + ((logical_pos - sinks) % W)
    posmap = list(range(sinks)) + [(N - 1) - ((wp - s) % W) for s in range(sinks, N)]
    return freqs_full[Tensor(posmap, dtype=dtypes.int32, device=freqs_full.device)].contiguous()

  def warmup_flash_decode(self):
    # Pre-capture the structurally selected flash candidate. Cache contents are
    # irrelevant for graph capture.
    if self.has_recurrent_block or not self.config.flash_decode: return
    ctx = min(FLASH_DECODE_CANDIDATE.split_size, self.max_context - 1)
    if ctx < 1: return
    v_sp = UOp.variable("start_pos", 0, self.max_context - 1)
    dummy = materialize_runtime_input(Tensor([[0]], dtype="int32").contiguous())
    temp = materialize_runtime_input(Tensor([0.0]).contiguous())
    for _ in range(3):
      try: self(dummy, v_sp.bind(ctx), temp, use_flash=True).realize()
      except Exception: return

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    if self.has_recurrent_block: chunk_size = 1
    _ring = self.config.ring and self.config.rope_dim == self.config.head_dim
    if _ring and len(tokens) > self.max_context:
      # StreamingLLM evicts DURING generation, not prefill: a prompt larger than the physical window N can't be held.
      raise RuntimeError(f"prompt is {len(tokens)} tokens but the streaming window is N={self.max_context}: streaming "
                         f"evicts during generation, not prefill. Shorten the prompt to <={self.max_context} tokens, or "
                         f"use a model/quant that admits a larger window.")
    for _b in self.blk: _b._ring_active = _ring   # make prefill ALSO store un-roped K when the ring is on
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    # Keep request inputs lazy until TinyJit prepares them. Eagerly realizing
    # these tensors changes the first prompt token from the feedback graph's
    # view contract into a bare buffer and makes independent generate calls
    # reuse a graph with the wrong physical input binding count.
    temp = runtime_input_materialization(Tensor([temperature]))
    # assign all input tokens once, then slice from start_pos for the model call
    t = runtime_input_materialization(Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32")
                                      .reshape(1, self.max_context))
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if start_pos < len(self._cached_tokens) and (resets := [r for b in self.blk for r in b._state_reset_ops()]): Tensor.realize(*resets)
    # flash-decode selection is centralized in should_use_flash_decode (default FLASH_DECODE=auto, threshold
    # 512): generate passes no use_flash override and lets that single authority decide per captured graph.
    out, prompt_len = None, len(tokens)
    while _ring or len(tokens) < self.max_context:   # ring: unbounded logical context (caller controls when to stop)
      ubatch = self.config.prefill_ubatch
      if self.config.prefill_v2 and (prompt_len - start_pos) >= ubatch:
        # prefill v2: a CONCRETE-T chunk of all-real prompt tokens (start_pos still symbolic; only the token
        # dim must be concrete for tensor cores). remaining>=UBATCH => start_pos<prompt_len so we slice from t.
        # concrete start_pos -> KV=start_pos+T concrete -> attention TC fires (the validated 1.24x, byte-identical).
        # Default ON for the FIRST chunk (start_pos==0): one cached concrete jit, no multi-chunk compile cost.
        # PREFILL_CONCRETE_KV=1 forces it for ALL chunks (K jits, pays off only when cached / for prompt<=512).
        use_concrete = (start_pos == 0) or self.config.prefill_concrete_kv
        sp, ntv = (start_pos if use_concrete else v_start_pos.bind(start_pos)), ubatch
        out = self(t[:, sp:sp+ubatch], sp, temp, use_flash=False).realize()
      elif self.config.prefill_remainder_fix and self.config.prefill_v2 and start_pos < prompt_len and prompt_len >= ubatch:
        # Phase-3 fix: a sub-UBATCH PROMPT remainder would otherwise fall to many slow 32-token symbolic calls
        # (the fallback trap). Instead process the LAST PREFILL_UBATCH tokens as ONE prefill-v2 chunk by shifting
        # the window back so it ENDS exactly at prompt_len -> all-real tokens (no padding), last position is
        # prompt_len-1 so out.item() is the next token. Re-processes the small overlap with the prior chunk (same
        # tokens -> same KV) -> correct. Symbolic start_pos reuses the one prefill_v2_jit (no per-remainder compile).
        sp = v_start_pos.bind(prompt_len - ubatch)   # symbolic offset -> matches the prefill_v2_jit signature
        out = self(t[:, sp:sp+ubatch], sp, temp, use_flash=False).realize()
        ntv = prompt_len - start_pos                      # advance straight to end of prompt
      elif _ring and start_pos >= prompt_len and out is not None:
        # StreamingLLM ring decode (T=1, past the prompt). Bind the WRAPPED write slot (always in [0,N-1] -> never trips
        # Variable.bind's vmax assert even as the logical position grows unboundedly); feed the per-step pre-gathered
        # freqs (identity while filling -> token-identical; slot-relative once full); ring_full switches to the [0:N]
        # read graph at the wrap. Two graphs total (fill + full), captured once each -- no per-step recompile.
        _N, _sinks = self.max_context, 4
        sp = v_start_pos.bind(self._ring_slot(start_pos, _N, _sinks)); ntv = 1
        _rf = self._ring_gather_freqs(next(b.freqs_cis for b in self.blk if hasattr(b, "freqs_cis")),
                                      start_pos, _N, _sinks)
        out = self(out, sp, temp, use_flash=True, ring_freqs=_rf, ring_full=(start_pos >= _N)).realize()
      else:
        sp, nt = v_start_pos.bind(start_pos), v_toks.bind(min(chunk_size, len(tokens) - start_pos))
        ntv = nt.val
        # Select the flash-decode graph (rollout_jit_flash) vs SDPA graph (rollout_jit) per-token by context, so a
        # generation that STARTS short still crosses over to flash once ctx reaches the threshold. Without this the
        # decode graph is baked SDPA at the start ctx and never switches -> short-prompt decode SDPA-degrades the
        # whole way (e.g. 85->54 tok/s by ctx512). should_use_flash_decode returns False for ntv!=1 (prefill chunks).
        _uf = self.config.flash_decode and _route_should_use_flash_decode(sp, ntv)
        out = self(_generation_input_slice(t, sp, nt, ntv) if start_pos < prompt_len or out is None else out, sp, temp,
                   use_flash=_uf).realize()
      start_pos += ntv
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
