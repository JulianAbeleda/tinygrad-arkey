from __future__ import annotations
import contextlib, contextvars, functools, hashlib, itertools, json, pathlib
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, dtypes, function, Device, role_metadata
from tinygrad.helpers import prod, getenv
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import warmstart_key as _warmstart_key
from tinygrad.llm.admission import (
  AUTO_MAX_CONTEXT, AdmissionInputs, ExactSelectedModelPlan, plan_exact_selected_model_load,
  plan_selected_model_memory, immutable_prefill_policy, prefill_concrete_kv_auto_decision,
  prefill_policy_strategy, prefill_policy_uses_overlay, prefill_v2_validate_ubatch,
  select_prefill_runtime_policy, bounded_packed_projection_proven_eligible,
)
from tinygrad.llm.device_facts import scan_device_facts
from tinygrad.llm.gguf import MODEL_PARAMETER_ALLOCATION_OWNER, gguf_load, gguf_load_metadata, gguf_load_with_metadata
from tinygrad.llm.gguf_memory_scan import RuntimeGeometry, selected_gguf_backing_bytes
from tinygrad.llm.decode_routes import (FLASH_DECODE_CANDIDATE, FLASH_DECODE_G5_CANDIDATE, _kv_store_parts_view,
                                        _q4k_single_projection_load_style,
                                        decode_kv_store_route, flash_decode_attention_route,
                                        q4k_gate_up_primitive_linear_call, q4k_gate_up_rms_affine_qualification_call,
                                        q6k_vocab_top1_call,
                                        should_use_flash_decode as _route_should_use_flash_decode)
from tinygrad.llm.decode_kernels import DecodeRMSNormSpec, emit_decode_rmsnorm_kernel
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program
from tinygrad.llm.shared_q8_attention import (SharedQ8AttentionAdmission, q4k_q8_fine_o_call, q4k_q8_o_call,
                                               shared_q8_attention_call)
from tinygrad.llm.packed_argmax import (make_native_argmax_host_mirror, native_argmax_finite_fp32,
                                        native_argmax_finite_fp32_host_mirror, packed_argmax_finite_fp32,
                                        read_native_argmax_host_mirror)
from tinygrad.llm.producer_kv_cache_sink import ProducerKVCacheSinkAdmission, producer_kv_cache_sink_call
from tinygrad.llm.q4k_kv_pair import q4k_kv_pair_call, q4k_qkv_call
from tinygrad.llm.prefill_routes import direct_packed_prefill_policy, is_direct_packed_prefill_linear, route_prefill_linear, validate_prefill_route_mode
from tinygrad.llm.prefill_memory_plan import Strategy
from tinygrad.llm.prefill_attachments import attach_selected_prefill_inventory
from tinygrad.llm.prefill_route_observer import prefill_route_scope, notify_prefill_route
from tinygrad.llm.qk_primitives import (
  QKConfig, QKPrimitiveBudget, Q4KPrimitiveLinear, Q4KPrimitiveRegistry, Q6KPrimitiveLinear,
  _install_q4k_primitives, _install_q6k_primitives, _qk_storage_summary,
  qk_primitive_capability_from_device_facts, kv_cache_fp16_eligible, _module_at,
)
from tinygrad.llm.model_facts import (
  PREFILL_OVERLAY_LINEAR_NAMES, attach_program_identity_metadata, bind_gguf_program_tensor_facts,
  estimate_prefill_overlay_bytes, is_prefill_overlay_role, model_facts_from_gguf_metadata, normalize_route_role,
)
from tinygrad.llm.qk_layout import QUANT_FORMATS
from tinygrad.llm.memory_adaptive_authority import (adapt_cached_memory_policy, memory_adaptive_adapters_active,
                                                     resolve_memory_adaptive_policy, validate_memory_evidence)
from tinygrad.llm.memory_semantics import (KV_CACHE, MODEL_PARAMETER, PREFILL_OUTPUT, RUNTIME_INPUT, RUNTIME_OUTPUT,
                                           RUNTIME_PERSISTENT, bind_memory_semantic_owner, kv_cache, materialize_runtime_input,
                                           runtime_input_materialization,
                                           memory_semantic_owner, model_parameter,
                                           prefill_activation, prefill_output, prefill_scratch, runtime_activation,
                                           runtime_input, runtime_output,
                                           runtime_persistent, runtime_scratch)
from tinygrad.llm.model_route_plan import (build_model_route_plan, decode_norm_fusion_promoted,
  decode_q4k_epilogue_fusion_promoted, decode_q4k_epilogue_resadd_promoted, decode_q4k_w1w3_fusion_promoted,
  decode_q4k_w1w3_fp16_store_promoted, decode_q4k_gate_up_four_warp_vector_promoted,
  decode_ffn_down_resadd_promoted, decode_kv_store_fusion_promoted,
  decode_rmsnorm_native_lowering_promoted, decode_rmsnorm_native_lowering_site_promoted,
  decode_reduce_output_rmsnorm_promoted, decode_qk_norm_rope_promoted,
  decode_producer_kv_cache_sink_promoted, decode_q4k_kv_pair_promoted,
  decode_native_argmax_threads,
  decode_shared_q8_attention_promoted,
  decode_q6_direct_shared_q8_attention_promoted, decode_q4_direct_shared_q8_attention_promoted,
  decode_shared_q8_q4kv_pair_promoted, decode_shared_q8_q4q6_kv_pair_promoted,
  decode_shared_q8_q4q4_qkv_full_promoted, decode_q4k_q4q4_qkv_full_promoted,
  decode_flash_llama_vec_wide_promoted,
  decode_q4k_ffn_down_fp16_geometry_promoted,
  decode_q6k_ffn_down_fp16_geometry_promoted, decode_q6k_ffn_down_packed_lanemap_promoted,
  decode_q6k_ffn_down_unroll_promoted,
  decode_q6k_v_four_warp_fp16_geometry_promoted)
from tinygrad.llm.prefill_candidate_runtime import decode_prefill_graph_candidate_set, automatic_promoted_prefill_graph_policy
from tinygrad.llm.physical_memory_ledger import AllocationOwner, bind_allocation_owner
from tinygrad.uop.ops import Ops, resolve

_MEMORY_ADAPTIVE_MEASUREMENT_AUTHORITY = contextvars.ContextVar("_memory_adaptive_measurement_authority", default=None)
_GENERIC_LLM_CONTROL = contextvars.ContextVar("_generic_llm_control", default=False)
_PREFILL_ATTENTION_PROFILE_BY_HEADS = {32: "qwen3_8b_q4k_m_gfx1100", 40: "qwen3_14b_q4k_m_gfx1100"}

_GGUF_TENSOR_OWNER = MODEL_PARAMETER_ALLOCATION_OWNER
_KV_CACHE_OWNER = AllocationOwner("kv_cache", "model")
_RUNTIME_PERSISTENT_OWNER = AllocationOwner("runtime_persistent", "model")

@contextlib.contextmanager
def generic_llm_control():
  """Force ordinary Tensor lowering for one model call, without mutating model policy."""
  token = _GENERIC_LLM_CONTROL.set(True)
  try: yield
  finally: _GENERIC_LLM_CONTROL.reset(token)

def _should_use_flash_attention(ring_freqs:Tensor|None, start_pos:int|UOp, T:int|UOp, use_flash:bool) -> bool:
  return ring_freqs is not None or _route_should_use_flash_decode(start_pos, T, use_flash)

def _adaptive_flash_split_count(enabled:bool,start_pos:int,max_context:int)->int|None:
  """Measured G4 context band: S48 wins through Tc=768; S64 wins for Tc=769..1024."""
  return 64 if enabled and max_context <= 1024 and 768 <= start_pos < max_context else None

def _active_horizon_flash_split_count(enabled:bool,start_pos:int,max_context:int)->int|None:
  """Closed-lease wide-Flash selector: S6 through Tc=768, then the installed S8 graph.

  ``start_pos`` is the zero-based KV position written by this decode call, so
  its active context is Tc=start_pos+1. The lease is deliberately exact to
  the measured 1024-token dense-decode workload and never changes the default
  graph when disabled.
  """
  return 6 if enabled and max_context == 1024 and 512 <= start_pos < 768 else None

def _flash_decode_geometry_for_split(base:dict, split_count:int|None) -> dict:
  """Translate a graph-identity split into the geometry lease it was qualified for."""
  geometry = dict(base)
  if split_count == 6:
    geometry.update(split_count=6, llama_vec_wide=True, token_bound=768)
  elif split_count is not None:
    geometry["split_count"] = split_count
  return geometry

def _flash_jit_variant(split_count:int|None, default, s6, s64):
  return s6 if split_count == 6 else s64 if split_count == 64 else default

def _flash_block_geometry(model, index:int, base:dict) -> dict:
  """Merge research block-local Flash geometry after the model-wide lease.

  The model entry points refresh block state before every capture/replay.  A
  separate override map keeps dose-limited experiments from being silently
  erased by that refresh while leaving the production path unchanged.
  """
  override=getattr(model,"_flash_decode_block_geometry_overrides",{}).get(index,{})
  return {**base,**override}

def _request_static_flash_split_count(prompt_len:int, expected_output_tokens:int|None, max_context:int)->int|None:
  """Select the measured single-graph S64 crossing route from an explicit request horizon."""
  if expected_output_tokens is None: return None
  if expected_output_tokens < 0: raise ValueError("expected_output_tokens must be non-negative")
  # Qualified at prompt 704: 65 pre-cliff tokens lose ~20.3 us each and are
  # repaid after about ten post-cliff tokens. Keep admission on that measured
  # near-boundary band until a broader horizon bracket exists.
  return 64 if max_context <= 1024 and 704 <= prompt_len < max_context and prompt_len + expected_output_tokens >= 779 else None

def prefill_v2_target_admitted(device_facts:object|None) -> bool:
  """Whether the concrete fp16 prefill-v2 route is admissible on this load target.

  Candidate contexts are admitted only after their target-specific schedule
  is independently qualified.  NV sm_120 now uses the candidate-owned tile
  with a TC-only warmstart; all four projection shapes match the generic
  reference bit-for-bit.
  """
  return True

# TG8 (docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md): the pre-TG8 gate
# ANDed a shape allowlist with a hardcoded `backend == "AMD" and arch == "gfx1100"` target-string equality.
# The shape half (ADMITTED_GRIDS) stays a shape gate -- it belongs in decode_routes.py-style admission, not
# here. The target half is split out into its OWN, separately named and separately testable policy predicate
# below, `_custom_kernel_prefill_attn_promoted`.
#
# This is deliberately NOT the TG3 quant-gate pattern (capability read from renderer facts + a promoted_targets
# policy that defaults OPEN when no record is loaded, tinygrad/llm/model_route_plan.py) -- but the reason is
# NOT the retired "captured program" claim: the fused kernel is no longer an opaque AMD gfx1100 machine-code
# injection. FlashPrefillAttentionSpec (schedule/wmma/flash_prefill.py) owns the topology as DATA and builds
# the kernel as ordinary UOps through spec.emit(); the renderer lowers them like any other program, and
# injection is via Tensor.uop_program. What still makes admission target-dependent is the FRAGMENT MATH:
# until the per-target decomposition is numerically proven, running it on an unproven target yields a compile
# failure or wrong numbers, not a crash. So the closed default stands, sourced from a BoltBeam route-policy
# record in the exact shape of `load_qk_target_promotion` (model_route_plan.py): JSON,
# `schema == "boltbeam.route_policy.v1"`, `promoted_targets` list of {backend, architecture}. No record ->
# CLOSED (this route's deliberate deviation from TG3's "no record -> open" default); record loaded -> the
# enforced set. The checked-in record is the promotion gate: `git diff` on that file is the promotion review
# artifact, and widening promotion never touches the shape gate or this call site.
_CUSTOM_KERNEL_PREFILL_ATTN_PROMOTION_RECORD = pathlib.Path(__file__).with_name("generated") / "custom-kernel-prefill-attention-route-policy.json"

def _load_custom_kernel_prefill_attn_promotion() -> frozenset[tuple[str|None, str|None]]:
  """Read the fused-prefill-attention promotion record (boltbeam.route_policy.v1, see the comment above)."""
  data = json.loads(_CUSTOM_KERNEL_PREFILL_ATTN_PROMOTION_RECORD.read_text())
  if data.get("schema") != "boltbeam.route_policy.v1":
    raise ValueError(f"{_CUSTOM_KERNEL_PREFILL_ATTN_PROMOTION_RECORD} is not a boltbeam.route_policy.v1 route policy")
  targets = data.get("promoted_targets")
  if targets is None: return frozenset()  # no promotion record -> closed (see the module comment above)
  return frozenset((t.get("backend"), t.get("architecture")) for t in targets)

_CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS: frozenset[tuple[str|None, str|None]] = _load_custom_kernel_prefill_attn_promotion()

def _custom_kernel_prefill_attn_promoted(backend:str|None, arch:str|None) -> bool:
  """TG8 policy authority: is (backend, architecture) promoted for the fused prefill attention route?
  See the module-level comment above `_CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS` for why this must
  default closed rather than reusing ModelRoutePlan.target_promoted's "no record -> open" default."""
  return (backend, arch) in _CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS

def _should_use_custom_kernel_prefill_attn(n_heads:int, n_kv_heads:int, backend:str|None, arch:str|None) -> bool:
  """Independent eligibility boundary for the proven custom-kernel-injection prefill attention route
  (tinygrad/llm/fused_attention.py:custom_kernel_attention -> tinygrad/schedule/wmma/flash_prefill.py
  FlashPrefillAttentionSpec), decoupled from the legacy composite-reduce path's prefill_tc_attn /
  shared_attention_proven_eligible proof (that proof is unrelated evidence for the OFF-critical-path
  class-2-risk `shared_prefill_attention` route -- see fused_attention.py's module docstring; P5b,
  docs/flash-prefill-pure-search-lift-scope-20260724.md). True only for the PROVEN admitted 8B/14B
  shapes (ADMITTED_GRIDS -- extra/llm_research/prefill/prefill_hd_sweep_numerics.py Hd=64/128 lower+numerically
  correct, extra/llm_research/prefill/prefill_flash_e2e_parity.py real-model 8B/14B token parity) on a
  promoted target (_custom_kernel_prefill_attn_promoted, TG8); every other shape/backend/arch safely falls
  back to SDPA (the existing default for all of them today)."""
  from tinygrad.llm.fused_attention import ADMITTED_GRIDS
  return (n_heads, n_kv_heads, 512) in ADMITTED_GRIDS and _custom_kernel_prefill_attn_promoted(backend, arch)

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

def derive_selected_gguf_prefill_inventory(kv:dict, meta:dict, ubatch:int=512, *, lm_head_resident_fp16:bool=False) -> dict:
  """Derive runtime route identity only from the explicitly opened GGUF's tensor metadata."""
  facts = model_facts_from_gguf_metadata(kv, meta)
  rows = []
  for tensor in facts.tensors:
    if tensor.role is None: continue
    # Generation consumes only the final token's output projection.  The
    # pp512 dense candidate set therefore cannot own LM-head merely because
    # its source tensor is quantized; keep it on the explicit fixed lazy route.
    candidate_controlled = tensor.quant_label in QUANT_FORMATS and tensor.role != "lm_head"
    # The lazy lm-head projection is physically M=1 after final-token pruning. Other selected prefill linears
    # execute at the concrete ubatch. Fixed rows remain census obligations, but are outside candidate geometry.
    physical_m = 1 if tensor.role == "lm_head" else ubatch
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
  # fp16 bytes of the covered overlay weights, computed once at derivation from the same role+shape rows the
  # route identity is built from. lm_head joins only under the resident-fp16 workload policy, matching the
  # _prefill_v2_covered() boundary. The inventory_identity intentionally excludes this key: it is a byte
  # estimate, not route identity.
  overlay_bytes = sum(row["shape"]["n"] * row["shape"]["k"] * 2 for row in rows
                      if is_prefill_overlay_role(row["role"]) or
                      (lm_head_resident_fp16 and normalize_route_role(row["role"]) == "lm_head"))
  return {"schema": "tinygrad.model_runtime_prefill_inventory.v2", "inventory_identity": identity, "rows": rows,
          "overlay_bytes": overlay_bytes}

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
    if source is not None and memory_adaptive_adapters_active():
      selected = adapt_cached_memory_policy(request, source)
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
  if prefill_policy_strategy(policy) == "BOUNDED_PACKED_TILES" and not bounded_packed_projection_proven_eligible(policy, device_facts):
    raise ValueError("BOUNDED_PACKED_TILES requires a complete bound packed-projection proof")
  if tuple(sorted(policy["routes"])) != tuple(sorted(invocation_ids)):
    raise ValueError("memory-adaptive policy route coverage does not exactly match selected GGUF inventory")
  fixed_routes = {row["invocation_id"]: row["fixed_route_id"] for row in inventory["rows"]
                  if row.get("candidate_controlled") is False}
  if any(policy["routes"].get(invocation_id) != route_id for invocation_id, route_id in fixed_routes.items()):
    raise ValueError("memory-adaptive policy changed a fixed selected-inventory route")
  memory_facts = policy.get("memory_facts")
  bundle = None
  if memory_facts is not None or policy.get("memory_fact_evidence") is not None:
    bundle = validate_memory_evidence(policy.get("memory_fact_evidence"), candidate_id=policy["candidate_id"])
    if bundle is None or memory_facts != bundle["facts"]:
      raise ValueError("memory-adaptive policy memory_facts are not bound to complete measured evidence")
  if prefill_policy_strategy(policy) != "DIRECT_PACKED_FALLBACK" and bundle is None and not trial:
    raise ValueError("accelerated memory-adaptive policy requires complete measured memory_facts")
  return policy

_EXACT_ROUTE_MEMORY_KEYS = ("resident_copies", "candidate_workspace_bytes", "batch_size", "kv_element_bytes",
                            "runtime_persistent_bytes", "peak_prefill_activation_bytes", "peak_prefill_output_bytes",
                            "peak_prefill_scratch_bytes")

def _graph_gemm_registry(policy):
  graph = policy.get("graph_gemm") if policy is not None else None
  candidate_set = graph.get("candidate_set") if isinstance(graph, dict) else None
  if not isinstance(candidate_set, dict): return None
  try: return decode_prefill_graph_candidate_set(candidate_set)
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

PREFILL_UBATCH = 512
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
  return route_prefill_linear(lin, x)

def _prefill_semantic(enabled:bool, mark, value:Tensor) -> Tensor:
  """Attach the same logical value role in either prefill or runtime/decode."""
  runtime_mark = {prefill_activation: runtime_activation, prefill_output: runtime_output,
                  prefill_scratch: runtime_scratch}.get(mark)
  if runtime_mark is None: raise ValueError("execution semantic requires a prefill role marker")
  return (mark if enabled else runtime_mark)(value)

def _decode_rmsnorm(norm, x:Tensor, promoted:bool, out_dtype=dtypes.float32) -> Tensor|None:
  """L1 M3 fused decode RMSNorm (l1-decode-plumbing-fusion-design-20260802.md section 6 norm
  family): one kernel per norm replaces the generic mean/var reduce + epilogue pair. Returns
  None to keep the legacy graph when the route is not promoted or the shape/strides contract
  does not hold. Decode-only by construction: the call sites gate on `not _prefill`."""
  if not promoted or x.dtype not in (dtypes.float32, dtypes.float16): return None
  dim = x.shape[-1]
  if dim < 32 or dim % 32: return None
  numel = prod(x.shape)
  rows = numel // dim
  if rows < 1 or rows * dim != numel: return None
  # The model stores norm weights packed; the lazy fp16 view is a per-token dequant when fed
  # through the opaque kernel boundary. The fp16 weights are materialized ONCE at load (see
  # Transformer.from_gguf) because the traced call sites run inside a Function dispatch where
  # ALLOW_DEVICE_USAGE is 0 and realizing here would raise. No prep means no fused route.
  w = getattr(norm, "_decode_fused_weight", None)
  if w is None: return None
  # Pass the exact flat activation view. The raw producer uop (x.uop.base) can
  # carry axes the opaque boundary must not materialize (the embedding gather's
  # vocab-block loop shows up as an extra rank and the scheduler stages it at
  # the wrong shape, collapsing the kernel's sumsq to a scalar). The custom-kernel
  # transport contiguous()s this view into the exact (numel,) buffer the emitter
  # indexes, so rows stay row-major. That per-call materialization is the measured
  # reason this route is closed-default non-landing (see the norm-fusion record).
  x_in = x.reshape(numel)
  x_rank = 1
  # 256 elements per warp is the measured occupancy sweet spot for the single-row 4096 shape
  # (512 threads, 8 elems/lane); smaller norms stay at one warp per row.
  warps = max(1, min(16, dim // 256))
  spec = DecodeRMSNormSpec(rows=rows, dim=dim, eps=norm.eps, warps_per_row=warps,
                           x_dtype=x.dtype, weight_dtype=dtypes.float16, out_dtype=out_dtype, x_rank=x_rank)
  program = KernelProgram("decode_norm", spec.kernel_name, KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
                          emit_decode_rmsnorm_kernel(spec), output_spec=OutputSpec((numel,), out_dtype))
  out = execute_promoted_program(None, x_in, w, program=program)
  return out.reshape(x.shape)

def _decode_reduce_output_rmsnorm(norm, x:Tensor, promoted:bool) -> Tensor:
  """Attach the ordinary-UOp cooperative RMSNorm marker at an explicit decode call site.

  Unlike a flag on ``nn.RMSNorm``, this helper cannot leak the experimental marker into
  prefill: every model call site passes a ``not _prefill``-gated promotion decision.  The
  complete ordinary RMSNorm remains source zero and therefore remains the fallback whenever
  late concrete-view admission declines the one-program lowering.
  """
  out = norm(x)
  if not promoted or norm.weight is None: return out
  # Bind the load-time fp16 identity buffer for every row count.  The ordinary
  # fp32 epilogue already rounds through fp16 in-kernel, so the pre-rounded
  # buffer carries the same half values (r2 logits gate passed bitwise), and
  # binding it keeps the fused body from materializing one fresh weight cast
  # kernel per q/k body per token (the measured 0fd8e427 overhead).
  weight = getattr(norm, "_decode_reduce_output_weight", None)
  if weight is None: weight = norm.weight
  return out._semantic_reduce_output_rmsnorm(x, out, weight, norm.eps)

def _decode_reduce_output_rmsnorm_rope(norm, x:Tensor, ordinary_rope:Tensor, freqs:Tensor, promoted:bool) -> Tensor:
  """Attach the closed-default full-head RoPE epilogue to REDUCE_OUTPUT.

  ``ordinary_rope`` is the complete fallback.  The semantic marker retains
  the pre-norm input and the persistent frequency table, allowing rangeify to
  emit one cooperative body without an opaque KernelProgram materialization.
  """
  if not promoted or norm.weight is None: return ordinary_rope
  weight = getattr(norm, "_decode_reduce_output_weight", None)
  if weight is None: weight = norm.weight
  return ordinary_rope._semantic_reduce_output_rmsnorm(x, ordinary_rope, weight, norm.eps, freqs=freqs)

def _decode_reduce_output_rmsnorm_fp16_consumer(norm, x:Tensor, promoted:bool) -> Tensor:
  """Closed-default typed RMSNorm boundary for Q4 decode consumers only.

  Block attention/FFN Q4 consumers already use this fp16 cast as their input
  ABI.  Mark that exact fallback value, rather than piercing its cast later.
  Q/K and output norms retain fp32-output semantics and never use this helper.
  """
  out = norm(x)
  if not promoted or norm.weight is None: return out
  weight = getattr(norm, "_decode_reduce_output_weight", None)
  if weight is None: weight = norm.weight
  typed_out = out.cast(dtypes.float16)
  return typed_out._semantic_reduce_output_rmsnorm(x, typed_out, weight, norm.eps)

def _decode_reduce_output_norm_flags(block, prefill:bool) -> tuple[bool,bool]:
  """Return (attention, FFN) REDUCE_OUTPUT decisions for one block trace."""
  if prefill: return False,False
  global_route=bool(getattr(block,"_decode_reduce_output_rmsnorm_promoted",False))
  shared_lease=isinstance(getattr(block,"_shared_q8_attention_admission",None),SharedQ8AttentionAdmission)
  fused_attn_lease=shared_lease and bool(getattr(block,"_decode_reduce_output_attn_rmsnorm_promoted",False))
  native_attn=bool(getattr(getattr(block,"attn_norm",None),"_rmsnorm_native_promoted",False)) and \
    not getenv("TINYGRAD_NATIVE_ATTN_NORM_COMPLETION_DISABLE",0)
  # The FFN-norm site is independently gateable so a census can close it while
  # the fp32 q/k site stays promoted (the live-split flash route depends on
  # that site).  Absent the knob it follows the global route, so production
  # behavior is unchanged.
  ffn_route=bool(getattr(block,"_decode_reduce_output_ffn_rmsnorm_promoted",global_route))
  return (global_route and not native_attn) or fused_attn_lease,ffn_route


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
  prefill_tc_attn: bool = False          # selected once from the immutable candidate policy
  prefill_custom_kernel_attn: bool = False  # route fused prefill attention through the custom-kernel injection (llm/fused_attention.py) instead of the composite-reduce path
  prefill_v2: bool = False               # concrete candidate binding, never a module-global switch
  prefill_ubatch: int = 512              # candidate-local physical M
  prefill_concrete_kv: bool = False      # workload/candidate policy decision
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

  def _feed_forward(self, x:Tensor, residual:Tensor|None=None) -> Tensor:
    """Dense decode callers may thread the block residual (h) so the ffn_down
    GEMV absorbs the h+ffn_out add in-kernel (M2b, nv-epilogue-absorption-
    route-scope-20260810.md). Prefill/MoE paths ignore the kwarg."""
    _prefill = getattr(self, "_is_prefill", False)
    if getattr(self, '_prefill_v2', False) and not hasattr(self, 'ffn_gate_exps') and not hasattr(self, 'ffn_gateup'):
      # Default-off research overlay for the qualified Qwen3-8B pp512 packed
      # gate/up lifecycle. The finalized native programs participate in the
      # ordinary TinyJit graph; every miss preserves the corrected fp16
      # fallback unchanged.
      if getenv("NV_Q4_IMMA_PP512", 0) and isinstance(getattr(self, "ffn_gate", None), Q4KPrimitiveLinear) \
          and isinstance(getattr(self, "ffn_up", None), Q4KPrimitiveLinear) and x.device == "NV" \
          and x.numel() == 512*4096:
        _binding, _flat = self._nv_q4_imma_pp512_binding, x.reshape(512,4096)
        g = _prefill_semantic(_prefill, prefill_activation,
          _binding.project(_flat, self.ffn_gate.prefill_packed_weight(), model_family="qwen3_8b", role="ffn_gate"))
        u = _prefill_semantic(_prefill, prefill_activation,
          _binding.project(_flat, self.ffn_up.prefill_packed_weight(), model_family="qwen3_8b", role="ffn_up"))
        h = _prefill_semantic(_prefill, prefill_activation, (g.silu() * u).contiguous())
        return _prefill_semantic(_prefill, prefill_activation, _pf16(self.ffn_down, h.reshape(x.shape[:-1]+(12288,))).contiguous())
      # prefill v2 (dense): fp16 + .contiguous()-isolated matmuls so each is a clean, warmstart-matchable TC
      # kernel (mirrors the gated chained-FFN prefill authority shape). MoE/fused fall through.
      g = _prefill_semantic(_prefill, prefill_activation, _pf16(self.ffn_gate, x).contiguous())
      u = _prefill_semantic(_prefill, prefill_activation, _pf16(self.ffn_up, x).contiguous())
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
    # Research-only scalar-packet Q8 fold: one packed W1/W3 producer plus the
    # four-warp Q4/Q8 DP4A resadd consumer replaces the fused16 producer and the
    # installed Q4-down resadd kernel. The admission lives on ffn_down; every
    # gate/up/down shape miss returns None and falls through unchanged.
    _scalar_q8_admission = getattr(self.ffn_down, "_q4k_ffn_down_mmvq_admission", None)
    if (not _prefill and getattr(_scalar_q8_admission, "scalar_q8_packet", False)
        and isinstance(getattr(self, "ffn_gate", None), Q4KPrimitiveLinear)
        and isinstance(getattr(self, "ffn_up", None), Q4KPrimitiveLinear)
        and isinstance(getattr(self, "ffn_down", None), Q4KPrimitiveLinear)):
      from tinygrad.llm.q4k_ffn_down_mmvq import q4k_ffn_down_mmvq_scalar_packet_call
      folded = q4k_ffn_down_mmvq_scalar_packet_call(self.ffn_gate, self.ffn_up, self.ffn_down, x, residual, _scalar_q8_admission)
      if folded is not None: return folded
    if not _prefill and getattr(self, "_decode_q4k_w1w3_fusion_promoted", False):
      _fg, _fu = getattr(self, "ffn_gate", None), getattr(self, "ffn_up", None)
      if isinstance(_fg, Q4KPrimitiveLinear) and isinstance(_fu, Q4KPrimitiveLinear):
        _four_warp_admission = getattr(_fg, "_q4k_gate_up_four_warp_admission", None)
        if _four_warp_admission is not None:
          from tinygrad.llm.q4k_gate_up_four_warp_mmvq import q4k_gate_up_four_warp_call
          _four_warp_z = q4k_gate_up_four_warp_call(_four_warp_admission, _fg, _fu, x)
          if _four_warp_z is not None:
            return self.ffn_down(_four_warp_z) if residual is None else self.ffn_down(_four_warp_z, normed_h=residual)
        # Fused w1+w3 decode GEMV (q4k-w1w3-fused-qv-implementation-record-20260803.md): ONE kernel
        # computes silu(gate(x)) * up(x); the fallback lambda reproduces the legacy chain's z exactly,
        # so an off-target/off-shape admission changes nothing about what ffn_down consumes.
        # M2a (nv-epilogue-absorption-m2a-promotion-record-20260812.md): the fused16 spelling stores
        # fp16 directly, folding ffn_down's input cast (the ordinary E_128_32_3 epilogue). Default-on
        # for NV sm_120 through the loader record; the research lease still forces it where the
        # record is closed (AB arms keep the same control/candidate contract).
        _w1w3_fp16_store = not _prefill and (getattr(self, "_q4k_w1w3_fp16_store_lease", False)
                                             or getattr(self, "_decode_q4k_w1w3_fp16_store_promoted", False))
        z = q4k_gate_up_primitive_linear_call(_fg, _fu, x,
          fallback=lambda: _prefill_semantic(_prefill, prefill_activation, _fg(x).silu().contiguous()) * _fu(x),
          store_fp16=_w1w3_fp16_store)
        return self.ffn_down(z) if residual is None else self.ffn_down(z, normed_h=residual)
    gated = _prefill_semantic(_prefill, prefill_activation, self.ffn_gate(x).silu().contiguous())
    z = gated * self.ffn_up(x)
    return self.ffn_down(z) if residual is None else self.ffn_down(z, normed_h=residual)

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, ring_freqs=None, residual_for_output:Tensor|None=None) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # StreamingLLM ring: pass the per-step freqs as an ARGUMENT into _run (a nested @function) so it is a true graph
    # input -- reading it off self inside _run would bake it at _run's compile time (attributes don't rebind).
    _rf = getattr(self, "_ring_freqs", None)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    def _run(x:Tensor, start_pos:int|UOp, ring_freqs):
      _prefill = getattr(self, "_is_prefill", False)
      _fused_norm = not _prefill and getattr(self, "_decode_norm_fusion_promoted", False)
      # A bounded shared-Q8 lease owns an independent REDUCE_OUTPUT marker only
      # for this block's attention norm. The fused provider consumes its
      # explicit (fallback, x, weight) sources; every miss retains the ordinary
      # fallback and the global reduce-output policy remains closed.
      _reduce_output_attn_norm,_reduce_output_norm=_decode_reduce_output_norm_flags(self,_prefill)
      _epi_fused = not _prefill and getattr(self, "_decode_q4k_epilogue_fusion_promoted", False)
      with role_metadata("rms_norm"):
        _nx = _decode_rmsnorm(self.attn_norm, x, _fused_norm, dtypes.float16)
        normed_x = _prefill_semantic(_prefill, prefill_scratch,
          _nx if _nx is not None else _decode_reduce_output_rmsnorm_fp16_consumer(self.attn_norm, x, _reduce_output_attn_norm))
      # L1 M4: o-proj residual-add epilogue absorption. When the gate is open, the attn_output
      # GEMV adds the block input x as an in-kernel epilogue, saving the generic E_32_32_4
      # elementwise add kernel. The residual is handed to _attention only when THIS block's
      # attn_output is an admitted Q4K primitive -- the absorption signal, not the promotion record
      # (MLA/GatedDeltaNet blocks and nn.Linear attn_output cannot absorb; dropping x there corrupts h).
      _attn_linear = getattr(self, "attn_output", None)
      # The residual_add variant has its OWN per-variant record
      # (decode-q4k-epilogue-resadd-route-policy.json, m4-resadd-landing-scope-20260806.md); the
      # combined M4 flag stays closed so the ffn_down prelude and fp16_cast cannot fire.
      _epi_resadd = not _prefill and getattr(self, "_decode_q4k_epilogue_resadd_promoted", False)
      _epi_residual = (isinstance(_attn_linear, Q4KPrimitiveLinear) and (
        (_epi_fused and getattr(getattr(_attn_linear, "route_admission", None), "q4k_epilogue_fusion_admitted", False)) or
        (_epi_resadd and getattr(getattr(_attn_linear, "route_admission", None), "q4k_epilogue_resadd_admitted", False))))
      attn_out = self._attention(normed_x, start_pos, ring_freqs, residual_for_output=(x if _epi_residual else None))
      with role_metadata("residual"):
        h = _prefill_semantic(_prefill, prefill_activation,
          attn_out if _epi_residual else x + attn_out)
      with role_metadata("rms_norm"):
        _nh = _decode_rmsnorm(self.ffn_norm, h, _fused_norm, dtypes.float16)
        normed_h = _prefill_semantic(_prefill, prefill_scratch,
          _nh if _nh is not None else _decode_reduce_output_rmsnorm_fp16_consumer(self.ffn_norm, h, _reduce_output_norm))
      # Research-only ownership split: expose the residual and exact fp16 norm
      # result as distinct precompiled outputs. The native Q8 producer owns
      # normed_h directly; h remains the residual input to the ordinary tail.
      if _prefill and getenv("NV_Q4_IMMA_PP512", 0): return h, normed_h
      # L1 M4: ffn_down silu*mul prelude + residual epilogue absorption. When the gate is open,
      # the down GEMV reads gate_out and up_out directly and computes silu(gate)*up inline (no
      # E_128_32_3 elementwise kernel), then adds h as an in-kernel epilogue (no E_32_32_4
      # residual-add kernel). The h add is absorbed only when THIS block's ffn_down is an admitted
      # Q4K primitive -- again the absorption signal, never the record alone.
      _ffn_absorbed = False
      _ffn_down_linear = getattr(self, "ffn_down", None)
      # M2b ffn_down residual add (nv-epilogue-absorption-route-scope-20260810.md): under the
      # harness-installed _ffn_down_resadd_lease the ffn_down Q4K/Q6K GEMV absorbs the h+ffn_out
      # add in-kernel (total + h[row], fp32 store) and the block returns ffn_out directly. The
      # lease is checked on the model, the block, AND the linear (the route re-checks it
      # fail-closed), and "gate_out" absent keeps the M4 fused-prelude path closed/distinct. The
      # loader record (_decode_ffn_down_resadd_promoted, decode-ffn-down-resadd-route-policy.json)
      # promotes the same spelling for NV sm_120; the lease still forces it where the record is
      # closed (research arms keep the same control/candidate contract).
      _ffn_resadd_lease = (not _prefill
                           and (getattr(self, "_ffn_down_resadd_lease", False) or getattr(self, "_decode_ffn_down_resadd_promoted", False))
                           and isinstance(_ffn_down_linear, (Q4KPrimitiveLinear, Q6KPrimitiveLinear))
                           and (getattr(_ffn_down_linear, "_ffn_down_resadd_lease", False)
                                or getattr(_ffn_down_linear, "_decode_ffn_down_resadd_promoted", False))
                           and getattr(_ffn_down_linear, "route_role", "") == "ffn_down"
                           and not hasattr(self, "ffn_gate_exps"))
      if (_epi_fused and isinstance(_ffn_down_linear, Q4KPrimitiveLinear) and
          getattr(getattr(_ffn_down_linear, "route_admission", None), "q4k_epilogue_fusion_admitted", False)):
        gate_out = self.ffn_gate(normed_h)
        up_out = self.ffn_up(normed_h)
        ffn_out = self.ffn_down(gate_out, gate_out=gate_out, up_out=up_out, normed_h=h)
        _ffn_absorbed = True
      else:
        # M1 norm-epilogue absorption: research-only, explicit lease.  Retain raw h across the
        # FFN norm boundary and apply its scalar RMS scale plus fp16 affine weight at each Q4
        # packed load, so the ffn-norm chain (r_16_256 + E_32_32_4_f14a5cc0) folds away and the
        # fused gate/up GEMV stores the fp16 z directly.  No loader policy creates this
        # attribute.  Checked BEFORE the M2b branch: under the M1 harness the M2b residual add
        # stays live, so the absorbed z feeds ffn_down(z, normed_h=h) instead of the plain
        # ffn_down(z).
        _rms_affine_weight = getattr(self, "_rms_affine_gateup_norm_weight", None)
        if (not _prefill and _rms_affine_weight is not None
            and isinstance(getattr(self, "ffn_gate", None), Q4KPrimitiveLinear)
            and isinstance(getattr(self, "ffn_up", None), Q4KPrimitiveLinear)):
          z = q4k_gate_up_rms_affine_qualification_call(self.ffn_gate, self.ffn_up, h, _rms_affine_weight,
            self.ffn_norm.eps, fallback=lambda: None, store_fp16=True)
          if z is not None:
            ffn_out = self.ffn_down(z, normed_h=h) if _ffn_resadd_lease else self.ffn_down(z)
            _ffn_absorbed = _ffn_resadd_lease
          elif _ffn_resadd_lease:
            ffn_out = self._feed_forward(normed_h, residual=h)
            _ffn_absorbed = True
          else:
            ffn_out = self._feed_forward(normed_h)
        elif _ffn_resadd_lease:
          ffn_out = self._feed_forward(normed_h, residual=h)
          _ffn_absorbed = True
        else:
          ffn_out = self._feed_forward(normed_h)
      with role_metadata("residual"):
        # M2b absorbed block output: the ffn_down GEMV's AFTER is the concrete contiguous fp32
        # block output, so forcing .contiguous() here would materialize an E_32_32_4 copy per
        # block (same in-place pattern as the M4 attn_out residual above). The plain
        # (h + ffn_out) path keeps its transport contiguous exactly as before.
        return _prefill_semantic(_prefill, prefill_activation,
          ffn_out if _ffn_absorbed else (h + ffn_out).contiguous())
    # @function wraps the traced return in a call/gettuple node. Mark that concrete block-output boundary as well as
    # its residual creation site so callification cannot hide the allocation identity from the manifest.
    _runner = function(precompile=True, allow_implicit=True)(_run)
    if getattr(self, "_is_prefill", False) and getenv("NV_Q4_IMMA_PP512", 0):
      h, normed_h = _runner(x, start_pos, _rf)
      return _prefill_semantic(True, prefill_activation, (h + self._feed_forward(normed_h)).contiguous())
    return _prefill_semantic(getattr(self, "_is_prefill", False), prefill_activation, _runner(x, start_pos, _rf).contiguous())

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

  def _attention(self, x:Tensor, start_pos:int|UOp, ring_freqs=None, residual_for_output:Tensor|None=None) -> Tensor:
    _prefill = getattr(self, "_is_prefill", False)
    if getattr(self, '_prefill_v2', False) and not hasattr(self, "attn_qkv"):  # prefill v2: fp16 isolated q/k/v
      q, k, v = (_prefill_semantic(_prefill, prefill_scratch, _pf16(lin, x).contiguous())
                 for lin in (self.attn_q, self.attn_k, self.attn_v))
    elif hasattr(self, "attn_qkv"): q, k, v = self.attn_qkv(x)  # B1 fused q/k/v
    else:
      # Closed research lease for an exact ordinary Q4/Q4 K/V pair. Harnesses
      # install it only on blocks outside the shared-Q8 triple boundary.
      _q4qkv = q4k_qkv_call(getattr(self, "_q4k_qkv_admission", None), self.attn_q, self.attn_k, self.attn_v, x)
      _q4kv_pair = None if _q4qkv is not None else q4k_kv_pair_call(
        getattr(self, "_q4k_kv_pair_admission", None), self.attn_k, self.attn_v, x)
      # P3a qualification hook: this is CLOSED by construction.  The loader
      # never installs ``_shared_q8_attention_admission``; a harness may lease
      # one exact block and the callee revalidates the real Q4/Q4/{Q4,Q6} tuple.
      # A miss preserves the three ordinary primitive calls verbatim.
      _shared_q8 = None if _q4kv_pair is not None else shared_q8_attention_call(
        getattr(self, "_shared_q8_attention_admission", None), self.attn_q, self.attn_k, self.attn_v, x, start_pos,
        getattr(self, "_shared_q8_attention_norm_weight", None))
      q, k, v = (_q4qkv if _q4qkv is not None else (self.attn_q(x), *_q4kv_pair) if _q4kv_pair is not None else
                 _shared_q8 if _shared_q8 is not None else (self.attn_q(x), self.attn_k(x), self.attn_v(x)))
    q, k, v = (_prefill_semantic(_prefill, prefill_scratch, value) for value in (q, k, v))
    # Qualification-only producer-side cache sink captures the exact terminal projection
    # values before reshape/transpose. The admission object is never installed by the
    # loader, so ordinary production traces do not retain these extra Python references.
    _producer_kv_inputs = (k, v) if (not _prefill and getattr(self, "_producer_kv_cache_sink_admission", None) is not None) else None
    # Decode kv-store fusion (decode-kv-store-chain-fusion-scope-20260803.md, Option A): capture the flat
    # k/v GEMV outputs BEFORE reshape/transpose so the fused store kernel can consume them as [kvh*Hd+elem]
    # views. v is never normed, so the pre-transpose capture is final; k is rebound below after the
    # qk_norm==head_dim norm (Qwen3-8B: qk_norm==128==head_dim) so the kernel receives POST-NORM k, exactly
    # what the legacy apply_rope+store chain would have roped and stored.
    _kv_store_flat = [k, v] if (not _prefill and getattr(self, "_decode_kv_store_fusion_promoted", False)) else None
    _kv_vparts = 1
    if _kv_store_flat is not None:
      # Absorb the q4k GEMV's v-parts reduce into the fused store kernel (decode-kv-store-chain-fusion-
      # scope-20260803.md revision): hand the route the raw parts view (Hkv*Hd, VPART) plus its extent so
      # the kernel sums the partials in-register. A graph without the parts reduce keeps (v, 1) and the
      # reduce materializes as before.
      _kv_store_flat[1], _kv_vparts = _kv_store_parts_view(_kv_store_flat[1])
    _fused_norm = not _prefill and getattr(self, "_decode_norm_fusion_promoted", False)
    _reduce_output_norm = not _prefill and getattr(self, "_decode_reduce_output_rmsnorm_promoted", False)
    _qk_norm_rope = not _prefill and getattr(self, "_decode_qk_norm_rope_promoted", False)
    _q_norm_input = _k_norm_input = None
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim:
      with role_metadata("rms_norm"):
        _nq = _decode_rmsnorm(self.attn_q_norm, q, _fused_norm)
        _nk = _decode_rmsnorm(self.attn_k_norm, k, _fused_norm)
        q = _prefill_semantic(_prefill, prefill_scratch,
          _nq if _nq is not None else _decode_reduce_output_rmsnorm(self.attn_q_norm, q, _reduce_output_norm))
        k = _prefill_semantic(_prefill, prefill_scratch,
          _nk if _nk is not None else _decode_reduce_output_rmsnorm(self.attn_k_norm, k, _reduce_output_norm))

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim:
      _q_norm_input, _k_norm_input = q, k
      with role_metadata("rms_norm"):
        _nq = _decode_rmsnorm(self.attn_q_norm, q, _fused_norm)
        _nk = _decode_rmsnorm(self.attn_k_norm, k, _fused_norm)
        q = _prefill_semantic(_prefill, prefill_scratch,
          _nq if _nq is not None else _decode_reduce_output_rmsnorm(self.attn_q_norm, q, _reduce_output_norm))
        k = _prefill_semantic(_prefill, prefill_scratch,
          _nk if _nk is not None else _decode_reduce_output_rmsnorm(self.attn_k_norm, k, _reduce_output_norm))
      if _kv_store_flat is not None:
        # Post-norm k is a fresh contiguous (B,Hkv,T,Hd) buffer, so the flat (B,T,Hkv*Hd) reshape is a pure
        # view (no copy) with the same [kvh*Hd+elem] linear layout as the pre-transpose capture.
        _kv_store_flat[0] = k.reshape(B, T, self.config.n_kv_heads * self.config.head_dim)

    # Position-invariant decode graph (llama.cpp reference): every structural slice (KV
    # store slot, rope read, KV read extent) indexes the cache by the UNBOUND start_pos
    # variable, so the captured graph key cannot depend on the concrete token position.
    # The bound UOp remains the JIT input carrier; exec supplies the value as var_vals.
    # Same twin pattern as flash_decode_attention_route / decode_kv_store_route /
    # shared_q8_attention_call; unbind() yields the generator's DEFINE_VAR node itself.
    _graph_pos = start_pos.src[0] if (isinstance(start_pos, UOp) and start_pos.op is Ops.BIND and
                                      len(start_pos.src) == 2 and start_pos.src[0].op is Ops.DEFINE_VAR) else start_pos

    # rope-at-read (DECODE_ROPE_AT_READ, opt-in; requires full-head rope): store UN-roped K and rotate at read -- the
    # prerequisite for the StreamingLLM ring's position re-basing. Q is never cached, so it is always roped here.
    # The ring supplies a per-step PRE-GATHERED freqs table via the JIT-input attribute _ring_freqs (slot-relative
    # positions); when unset the baked self.freqs_cis is used (absolute positions). _ring_freqs implies rope-at-read.
    _ring_freqs = ring_freqs
    # rope-at-read active if: env flag, OR a ring-decode step (freqs supplied), OR the ring is active this generation
    # (covers PREFILL, which must ALSO store un-roped K so the ring decode reads it consistently).
    _rope_read = (_ring_freqs is not None or getattr(self, "_ring_active", False)) \
                 and self.config.rope_dim == self.config.head_dim
    # Decode kv-store fusion admission (decode-kv-store-chain-fusion-scope-20260803.md section 6): gate is
    # the promotion record AND concrete decode shape (T==1,B==1; the kernel stores exactly one token slot
    # at `start_pos`, so a batched or chunked trace must keep the legacy chain) AND full-head rope AND
    # qk_norm in (0, head_dim) (the kernel consumes post-norm k via _kv_store_flat) AND no rope-at-read
    # (store roped K, which the ring path must NOT do) AND an fp16/fp32 cache (the kernel writes the
    # cache's own dtype -- fp32 on NV, fp16 on AMD -- so the stored bytes match the legacy chain; a
    # quant/other cache keeps legacy).
    _kv_store_fused = _kv_store_flat is not None and isinstance(T, int) and T == 1 and B == 1 and not _rope_read and \
      self.config.rope_dim == self.config.head_dim and self.config.qk_norm in (0, self.config.head_dim) and \
      self.cache_kv.dtype in (dtypes.float16, dtypes.float32)
    _fr = _ring_freqs if _ring_freqs is not None else self.freqs_cis
    # full-ring (ctx>=N): the buffer is full and the write slot wraps, so the live read length is the WHOLE buffer N
    # (all slots valid), not start_pos+T (start_pos is the wrapped write slot, not a length). Selects [0:N] reads + Tc=N.
    _ring_full = getattr(self, "_ring_full", False)
    _rl = self.config.max_context if _ring_full else (_graph_pos + T)
    # Q is roped via _fr (the gathered ring table when ring, else freqs_cis) indexed by start_pos: in the full ring
    # start_pos is the write slot wp, and _fr[wp] = freqs[pos_of(wp)] = the query's (newest) position -> consistent
    # with the K positions. In fill / non-ring, _fr == freqs_cis and start_pos is the absolute position (unchanged).
    _q_rope = apply_rope(q[..., :self.config.rope_dim], _fr[_graph_pos:_graph_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    if _qk_norm_rope and _q_norm_input is not None and self.config.rope_dim == self.config.head_dim and not _rope_read:
      _q_rope = _decode_reduce_output_rmsnorm_rope(self.attn_q_norm, _q_norm_input, _q_rope, _fr, True)
    q = _prefill_semantic(_prefill, prefill_scratch, _q_rope)
    if not _rope_read and not _kv_store_fused:
      _k_rope = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[_graph_pos:_graph_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)
      if _qk_norm_rope and _k_norm_input is not None and self.config.rope_dim == self.config.head_dim:
        _k_rope = _decode_reduce_output_rmsnorm_rope(self.attn_k_norm, _k_norm_input, _k_rope, self.freqs_cis, True)
      k = _prefill_semantic(_prefill, prefill_scratch, _k_rope)

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
      _st_kv = self.cache_kv[:, :, :, _graph_pos:_graph_pos+T, :].uop.store(_kvq.uop)
      _st_sc = self.cache_kv_scale[:, :, :, _graph_pos:_graph_pos+T].uop.store(_sch.uop)
      assigned_kv = Tensor(self.cache_kv.uop.after(_st_kv))
      assigned_scale = Tensor(self.cache_kv_scale.uop.after(_st_sc))
      _ksc = assigned_scale[0, :, :, 0:_graph_pos+T].reshape(B, _Hkv, _graph_pos+T, 1)
      _vsc = assigned_scale[1, :, :, 0:_graph_pos+T].reshape(B, _Hkv, _graph_pos+T, 1)
      k = _prefill_semantic(_prefill, prefill_scratch, assigned_kv[0, :, :, 0:_graph_pos+T, :].cast(dtypes.float16) * _ksc)
      v = _prefill_semantic(_prefill, prefill_scratch, assigned_kv[1, :, :, 0:_graph_pos+T, :].cast(dtypes.float16) * _vsc)
    else:
      assigned_scale = None
      _producer_assigned_kv = None
      if (_producer_kv_inputs is not None and not _rope_read and not _kv_store_fused and isinstance(B, int) and B == 1 and
          isinstance(T, int) and T == 1 and self.config.rope_dim == self.config.head_dim and
          self.config.qk_norm == self.config.head_dim):
        _producer_assigned_kv = producer_kv_cache_sink_call(
          getattr(self, "_producer_kv_cache_sink_admission", None), self.cache_kv,
          _producer_kv_inputs[0], _producer_kv_inputs[1], self.attn_k_norm, self.freqs_cis, self.config.max_context)
      if _producer_assigned_kv is not None:
        # The cache AFTER carries the K-producer call that also consumed final V;
        # flash therefore waits on the same semantic K/V join as the legacy store.
        assigned_kv = _producer_assigned_kv
      elif _kv_store_fused:
        # Option A: ONE kernel ropes k in-kernel (exact apply_rope arithmetic, fp32), casts k/v to the
        # cache's own dtype, and stores both at slot start_pos -- replacing the k-rope + k-cast + v-cast +
        # Tensor.stack(k,v) + cache store chain. `assigned_kv` is the cache AFTER the store, same contract
        # as the legacy chain (flash route reads assigned_kv directly; the SDPA reads below are DCE'd).
        assigned_kv = decode_kv_store_route(self.cache_kv, _kv_store_flat[0], _kv_store_flat[1], self.freqs_cis,
                                            self.config.n_kv_heads, self.config.head_dim, self.config.max_context,
                                            vparts=_kv_vparts)
      else:
        assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, _graph_pos:_graph_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
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
    # Materialize on CONCRETE extents (max T rows, max_context cols) then slice to the symbolic
    # (T, start_pos+T) view: triu over a symbolic extent lowers arange through a cumsum into a
    # quadratic kernel (two full KV loops per output element), which wedges symbolic prefill at
    # deep context (observed 794ms QK^T epilogue at KV=4096). The concrete mask renders as two
    # linear E kernels and QK^T reads the sliced buffer as data3 with no fused loops.
    # The diagonal/extent index the UNBOUND position variable (_graph_pos), so the mask is
    # position data, not graph structure: it matches the unbound KV read extents symbolically
    # and resolves per-position at exec through var_vals.
    _mask_rows = T.simplify().vmax if isinstance(T, UOp) else T
    mask = Tensor.full((1, 1, _mask_rows, self.config.max_context), float("-inf"), dtype=x.dtype, buffer=True) \
      .triu(_graph_pos+1)[:, :, :T, :_graph_pos+T] if resolve(T != 1) else None
    # The model owns two separately captured decode graphs. The immutable
    # candidate binding selects the flash graph upstream; ring decode always
    # uses it because its wrapped write slot is not a logical context length.
    _o_q8 = None
    _o_q8_fine = False
    if _should_use_flash_attention(_ring_freqs, start_pos, T, getattr(self, "_use_flash", False)):
      Hq, Hkv, Hd = self.config.n_heads, self.config.n_kv_heads, self.config.head_dim
      _flash_geom = getattr(self, "_flash_decode_tile_geometry_lease", None)
      _o_q8_fine = bool((_flash_geom or {}).get("o_q8_fine_owned",False))
      _o_prefetch_groups = int((_flash_geom or {}).get("o_successor_prefetch_groups", 0))
      _o_successor_weights = None
      if _o_prefetch_groups:
        if not isinstance(self.attn_output, Q4KPrimitiveLinear) or not hasattr(self.attn_output, "q4k_storage"):
          raise ValueError("Flash O successor prefetch requires a Q4_K attention-output projection")
        _o_successor_weights = self.attn_output.q4k_storage.words.to(x.device)
      # M2d combine-fp16 lease (nv-epilogue-absorption-route-scope-20260810.md): harness-installed
      # on the model and every block; fail-closed absent keeps the closed-default policy answer
      # (fp32 combine, byte-identical legacy graph).
      _flash_result = flash_decode_attention_route(q, assigned_kv, start_pos, T, B, Hq, Hkv, Hd, self.config.max_context,
                                         kv_scale=assigned_scale, freqs=(_fr if _rope_read else None),
                                         ring_full=_ring_full,
                                         combine_fp16=bool(getattr(self, "_flash_combine_fp16_lease", False)),
                                         tile_geometry=_flash_geom, successor_weights=_o_successor_weights)
      out,_o_q8 = _flash_result if isinstance(_flash_result,tuple) else (_flash_result,None)
      attn = out.reshape(B, Hq, T, Hd).cast(q.dtype)
    elif self.config.prefill_custom_kernel_attn and getattr(self, '_prefill_v2', False) and isinstance(start_pos, int) and resolve(T != 1):
      # P5b: the proven custom-kernel-injection route's OWN independent eligibility boundary
      # (_should_use_custom_kernel_prefill_attn -> ADMITTED_GRIDS + AMD/gfx1100), decoupled from the
      # legacy composite-reduce path below (which requires the separate prefill_tc_attn /
      # shared_attention_proven_eligible proof -- unrelated evidence for a different, class-2-risk
      # path). Self-sufficient: builds its own ctx unconditionally (see below) so it cannot silently
      # fall through to SDPA the way a copy-pasted strategy-gate once did (root-caused via
      # p5_default_binding.py -- see the ctx comment). Admission is bounded to the proven model
      # shapes by _should_use_custom_kernel_prefill_attn, so this branch cannot regress anything else.
      from tinygrad.llm.fused_attention import route_prefill_attention
      from tinygrad.uop.ops import SharedAttentionCandidateContext
      # Self-sufficient ctx: custom_kernel_attention/route_prefill_attention never read ctx.profile or
      # ctx.strategy (confirmed by inspection of fused_attention.py -- it only reads ctx.kv_tokens/
      # start_pos/q_tokens/output_block_base/acc_blocks). The legacy branch below gates ctx construction
      # on `_strategy in ("FULL_RESIDENT_OVERLAY","BOUNDED_PACKED_TILES")` because the COMPOSITE
      # shared_prefill_attention path cares about overlay/weight-loading strategy; that gate has nothing
      # to do with this route and was a copy-paste coupling bug (root-caused via p5_default_binding.py:
      # a DEFAULT model load resolves prefill_policy_strategy=='DIRECT_PACKED_FALLBACK', which made this
      # branch build ctx=None and silently fall through to SDPA even though prefill_custom_kernel_attn
      # was correctly True -- a token match hid a non-firing route). Build ctx unconditionally here;
      # _profile is always non-empty when this branch is reached because
      # _should_use_custom_kernel_prefill_attn already restricted admission to n_heads in {32,40}.
      _profile = _PREFILL_ATTENTION_PROFILE_BY_HEADS.get(self.config.n_heads, "custom_kernel_prefill_attn")
      _ctx = SharedAttentionCandidateContext(_profile, prefill_policy_strategy(self.config.prefill_policy), T, start_pos+T,
        start_pos, self.config.n_heads, self.config.n_kv_heads, self.config.head_dim, True)
      with role_metadata("shared_prefill_attention"):
        attn = _prefill_semantic(_prefill, prefill_scratch,
          route_prefill_attention(q.cast(dtypes.float16), k.cast(dtypes.float16), v.cast(dtypes.float16),
            mask=mask, causal=True, ctx=_ctx, use_custom_kernel=True).cast(q.dtype))
    elif self.config.prefill_tc_attn and getattr(self, '_prefill_v2', False) and isinstance(start_pos, int) and resolve(T != 1):
      # Q/K/V have the same fp16 activation contract for resident-overlay and
      # packed-weight projections.  Capture attention once at that boundary;
      # rangeify either lowers the semantic operation or preserves its exact
      # SDPA fallback.  Do not decompose scores here.
      from tinygrad.llm.flash_prefill_attention import shared_prefill_attention
      from tinygrad.uop.ops import SharedAttentionCandidateContext
      _strategy = prefill_policy_strategy(self.config.prefill_policy)
      _profile = _PREFILL_ATTENTION_PROFILE_BY_HEADS.get(self.config.n_heads, "")
      _ctx = SharedAttentionCandidateContext(_profile, _strategy, T, start_pos+T, start_pos, self.config.n_heads,
        self.config.n_kv_heads, self.config.head_dim, True) if _profile and _strategy in ("FULL_RESIDENT_OVERLAY", "BOUNDED_PACKED_TILES") else None
      with role_metadata("shared_prefill_attention"):
        # The shared-attention grid kernel requires a uniform fp16 Q/K/V contract
        # (see lower_attention_semantic grid_shape check). K/V are already fp16 here
        # but Q arrives fp32, which silently rejected the kernel into SDPA. Enforce
        # the contract; result is cast back to Q's original dtype like the SDPA path.
        import os as _os
        if self.config.prefill_custom_kernel_attn or _os.environ.get("FUSED_CUSTOM_KERNEL"):
          # Custom-kernel injection route: bypass the composite-reduce lowering
          # (class-2) by injecting the proven program via Tensor.uop_program, which
          # realizes Q/K/V as opaque buffers. See llm/fused_attention.py.
          from tinygrad.llm.fused_attention import route_prefill_attention
          attn = _prefill_semantic(_prefill, prefill_scratch,
            route_prefill_attention(q.cast(dtypes.float16), k.cast(dtypes.float16), v.cast(dtypes.float16),
              mask=mask, causal=True, ctx=_ctx, use_custom_kernel=True).cast(q.dtype))
        else:
          _qkv_h = (q.cast(dtypes.float16), k.cast(dtypes.float16), v.cast(dtypes.float16))
          attn = _prefill_semantic(_prefill, prefill_scratch,
            shared_prefill_attention(*_qkv_h, mask=mask, candidate_context=_ctx).cast(q.dtype))
    else:
      attn = _prefill_semantic(_prefill, prefill_scratch,
                               q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True))  # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    out_in = attn if not self.config.attn_output_gate else (attn * gate.sigmoid())
    # L1 M4: when residual_for_output is set (q4k epilogue gate open AND this block's attn_output is an
    # admitted Q4K primitive), the attn_output GEMV adds the residual in-kernel instead of as a separate
    # elementwise kernel. The isinstance guard keeps an nn.Linear attn_output from seeing the kwarg.
    _has_residual = residual_for_output is not None and isinstance(self.attn_output, Q4KPrimitiveLinear)
    if getattr(self, '_prefill_v2', False):
      out = _pf16(self.attn_output, out_in)
      return _prefill_semantic(_prefill, prefill_activation, out.contiguous())
    if _has_residual:
      if _o_q8 is not None:
        _q8_o=(q4k_q8_fine_o_call(True,self.attn_output,_o_q8,residual_for_output) if _o_q8_fine else
               q4k_q8_o_call(True,self.attn_output,_o_q8,residual_for_output))
        if _q8_o is None: raise RuntimeError("combine-owned Q8 O route failed after admission")
        return _prefill_semantic(_prefill,prefill_activation,_q8_o)
      return _prefill_semantic(_prefill, prefill_activation,
        self.attn_output(out_in, residual=residual_for_output))
    return _prefill_semantic(_prefill, prefill_activation, self.attn_output(out_in))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # KV cache dtype decision (decode-kv-store-chain-fusion-scope-20260803.md revision): fp16 storage is
      # byte-identical to fp32 for the decode tile because the tile casts the cache to fp16 on read
      # (flash_decode_attention.py::make_kv_element_loader), and it halves KV bytes. Whether fp16 is
      # EXPRESSIBLE is a device capability (DeviceCapabilities.supports_fp16 from the opened renderer's
      # supported_dtypes() -- never a backend/architecture string; the pre-TG3 AMD gfx1100 eligibility was
      # removed). The validated-shape gate (batch 1, 32 heads, 8 kv heads, 128 head dim) keeps the measured
      # promotion scope; other shapes keep the default dtype.
      _generated_decode_shape_supported = kv_cache_fp16_eligible( \
        getattr(self.config, "prefill_device_facts", None)) and x.shape[0] == 1 and self.config.n_heads == 32 \
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

  def _attention(self, x:Tensor, start_pos:int|UOp, ring_freqs=None, residual_for_output:Tensor|None=None) -> Tensor:
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
    # Position-invariant graph structure: index rope reads and the KV store slot by the
    # UNBOUND start_pos variable (see TransformerBlock._attention for the full rationale).
    _graph_pos = start_pos.src[0] if (isinstance(start_pos, UOp) and start_pos.op is Ops.BIND and
                                      len(start_pos.src) == 2 and start_pos.src[0].op is Ops.DEFINE_VAR) else start_pos
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(apply_rope(q_rope, self.freqs_cis[_graph_pos:_graph_pos+T]), dim=-1)

    kv_a = mark_scratch(self.attn_kv_a_mqa(x))
    with role_metadata("rms_norm"): c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = apply_rope(
      kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2),
      self.freqs_cis[_graph_pos:_graph_pos+T])

    k_store = mark_scratch(c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1))
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, _graph_pos:_graph_pos+T, :].uop.store(k_store.uop)))[:, :, 0:_graph_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    _mask_rows = T.simplify().vmax if isinstance(T, UOp) else T
    mask = Tensor.full((1, 1, _mask_rows, self.config.max_context), float("-inf"), dtype=x.dtype, buffer=True) \
      .triu(_graph_pos+1)[:, :, :T, :_graph_pos+T] if resolve(T != 1) else None
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

  def _attention(self, x:Tensor, start_pos:int|UOp, ring_freqs=None, residual_for_output:Tensor|None=None) -> Tensor:
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
    # Transformer owns attachment construction and benchmark census, so the
    # normalized selected registry must live at this level. Per-block copies
    # are not consulted when bindings are attached to covered linears.
    self._prefill_graph_gemm_registry = _graph_gemm_registry(config.prefill_policy)
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    self._q4k_linears = Q4KPrimitiveRegistry()
    # we specialize the JIT for prefill and rollout; rollout_jit_flash is the context-aware flash-decode
    # decode graph (captured lazily when ctx crosses FLASH_DECODE_THRESHOLD), see __call__/generate.
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    self.rollout_jit_flash = TinyJit(self.forward)
    self.rollout_jit_flash_s6 = TinyJit(self.forward)
    self.rollout_jit_flash_s64 = TinyJit(self.forward)
    self.prefill_greedy_jit = TinyJit(self.forward_greedy)
    self.prefill_v2_greedy_jit = TinyJit(self.forward_greedy)
    self.rollout_greedy_jit = TinyJit(self.forward_greedy)
    self.rollout_greedy_jit_flash = TinyJit(self.forward_greedy)
    self.rollout_greedy_jit_flash_s6 = TinyJit(self.forward_greedy)
    self.rollout_greedy_jit_flash_s64 = TinyJit(self.forward_greedy)
    # Closed-default P5 experiment: two captures can form an alias-free
    # device-resident feedback ring.  A harness must explicitly promote the
    # route and provide the alternating slot; ordinary callers retain the
    # single-capture path and CapturedJit's generic written-input firewall.
    self.rollout_greedy_pingpong_jits = tuple(TinyJit(self.forward_greedy) for _ in range(2))
    self.rollout_greedy_pingpong_jits_flash = tuple(TinyJit(self.forward_greedy) for _ in range(2))
    self.rollout_greedy_pingpong_jits_flash_s6 = tuple(TinyJit(self.forward_greedy) for _ in range(2))
    self.rollout_greedy_pingpong_jits_flash_s64 = tuple(TinyJit(self.forward_greedy) for _ in range(2))
    # Diagnostic-only captures return the already-computed decode logits beside
    # the sampled token. They are separate so the production sampled graph's
    # return/lifetime contract remains byte-for-byte unchanged.
    self.rollout_logits_jit = TinyJit(self.forward_with_logits)
    self.rollout_logits_jit_flash = TinyJit(self.forward_with_logits)
    self.rollout_logits_jit_flash_s6 = TinyJit(self.forward_with_logits)
    self.rollout_logits_jit_flash_s64 = TinyJit(self.forward_with_logits)
    self.rollout_greedy_logits_jit = TinyJit(self.forward_greedy_with_logits)
    self.rollout_greedy_logits_jit_flash = TinyJit(self.forward_greedy_with_logits)
    self.rollout_greedy_logits_jit_flash_s6 = TinyJit(self.forward_greedy_with_logits)
    self.rollout_greedy_logits_jit_flash_s64 = TinyJit(self.forward_greedy_with_logits)
    self.rollout_greedy_logits_pingpong_jits = tuple(TinyJit(self.forward_greedy_with_logits) for _ in range(2))
    self.rollout_greedy_logits_pingpong_jits_flash = tuple(TinyJit(self.forward_greedy_with_logits) for _ in range(2))
    self.rollout_greedy_logits_pingpong_jits_flash_s6 = tuple(TinyJit(self.forward_greedy_with_logits) for _ in range(2))
    self.rollout_greedy_logits_pingpong_jits_flash_s64 = tuple(TinyJit(self.forward_greedy_with_logits) for _ in range(2))
    self.rollout_jit_ring = TinyJit(self.forward_ring)        # ring FILL phase (ctx<N): read [0:start_pos+T], identity freqs
    self.rollout_jit_ring_full = TinyJit(self.forward_ring)   # ring FULL phase (ctx>=N): read [0:N], wrapped write slot + gathered freqs
    # The selected prefill candidate gets a separate concrete-M capture.
    self.prefill_v2_jit = TinyJit(self.forward)
    self.prefill_v2_jits: dict = {}   # concrete-KV: one prefill jit per concrete start_pos
    # prefill v2 warmstart table is built here but installed into the global codegen knob ONLY for the duration
    # of the prefill-v2 forward (see __call__), to contain that ambient power rather than leave it set process-wide.
    self._pf16_warmstart:dict|None = None
    # Packed-WMMA prefill warmstart table: built when prefill_routes.packed_wmma_prefill_enabled()
    # is set (default ON). See _build_packed_wmma_warmstart.
    # NOTE: cannot be built here -- Q4_K/Q6_K packed storage (q4k_storage/prefill_packed_weight) is
    # installed onto these same Linear objects by from_gguf AFTER Transformer.__init__ returns (via
    # _install_q4k_primitives/_install_q6k_primitives, once state_dict is loaded). is_direct_packed_prefill_linear
    # is always False at this point, so this table is built later, in from_gguf (see the
    # `config.prefill_v2 and packed_wmma_prefill_enabled()` call right after realize_prefill_v2_weights()).
    self._packed_wmma_warmstart:dict|None = None
    self._packed_wmma_warmstart_contexts:dict|None = None
    if config.prefill_v2:
      prefill_v2_validate_ubatch(config.prefill_ubatch)

  @property
  def prefill_memory_plan(self) -> str|None:
    return self.config.prefill_memory_plan

  @property
  def prefill_policy(self): return self.config.prefill_policy

  def _prefill_v2_role_for_name(self, name:str) -> str:
    from tinygrad.llm.model_facts import QK_ROUTE_ROLES, normalize_route_role
    role = normalize_route_role(name)
    return role if role in QK_ROUTE_ROLES else ""

  def _prefill_v2_dims(self, lin):
    out_f = getattr(lin, "out_features", None) or lin.weight.shape[0]
    in_f  = getattr(lin, "in_features", None)  or lin.weight.shape[1]
    return (out_f, in_f) if isinstance(out_f, int) and isinstance(in_f, int) else (None, None)

  def _prefill_v2_covered(self):
    # (linear, out_f, in_f) for each covered linear with a known concrete shape -- single source for the
    # warmstart table, the VRAM estimate, and the realization, so they can't drift apart.
    for block in self.blk:
      for n in PREFILL_OVERLAY_LINEAR_NAMES:
        lin = getattr(block, n, None)
        if lin is None or getattr(lin, "weight", None) is None: continue
        setattr(lin, "_prefill_selected_strategy", prefill_policy_strategy(self.config.prefill_policy))
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
    # dense fp16 path has no packed-weight PARAM, so packed_dtype=None (matches the key
    # postrange._warmstart_key computes for a kernel with no packed-weight-dtype PARAM).
    return {_warmstart_key({out_f, self.config.prefill_ubatch}, in_f): _opts(out_f, in_f)
            for lin, out_f, in_f in self._prefill_v2_covered() if getattr(lin, "_pf16_w", None) is not None}

  def _build_packed_wmma_warmstart(self) -> tuple[dict, dict]:
    # Only called when prefill_routes.packed_wmma_prefill_enabled() is set (see __init__). Runs the
    # one-time-per-(quant,role,shape) correctness gate for every covered Q4_K/Q6_K linear whose
    # (quant, role) has a frozen packed-wmma geometry (tinygrad/llm/packed_wmma_prefill.py.
    # PACKED_WMMA_GEOM); an ungated/failing combo is simply omitted here, so it plays no part in this
    # table and route_packed_wmma_prefill's own per-call gate check (same cache) declines it too --
    # never dispatched ungated. NOT set into the global here -- installed only around the prefill-v2
    # forward (__call__), merged with self._pf16_warmstart.
    from tinygrad.llm.packed_wmma_prefill import build_packed_wmma_warmstart_tables
    return build_packed_wmma_warmstart_tables(self._prefill_v2_covered(), self.config.prefill_ubatch)

  def realize_prefill_v2_weights(self) -> int:
    # Realize a clean fp16 weight per covered linear (cached as `_pf16_w`, read by _pf16). The primitives' lazy
    # Q4_K/Q6_K->fp16 dequant graph, used raw, fuses into the matmul -> ~3% peak (no TC win); a realized fp16
    # buffer makes the prefill-v2 matmul a real TC GEMM (~13x on the measured smaller profile). COST: ~fp16-model-size extra VRAM
    # (it coexists with the Q4_K decode storage). Called only for the selected full-overlay representation.
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
    # With selected concrete-KV reuse, every prefill chunk runs through a per-start_pos concrete jit
    # (-> the fusion attention path, 1.7-4.4x/chunk faster than the symbolic chunk). Those jits compile on first
    # use (~5s each), so a COLD long prompt pays the tax inline. Precompiling them ONCE at load (here) moves the
    # tax to load time, so every generation -- including the first -- is warm. Bounded: ceil(max_context/UBATCH)
    # jits. Safe to leave the dummy KV behind: a fresh model's first generation starts at start_pos=0 and
    # overwrites the cache in chunk order before any position is read.
    if not (self.config.prefill_v2 and prefill_v2_target_admitted(self.config.prefill_device_facts) and self.config.prefill_concrete_kv): return 0
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

  @staticmethod
  def _bound_position_var_vals(start_pos: int|UOp) -> dict[str, int]:
    """Value of a bound position UOp, keyed by its unbound variable name. Position-invariant
    decode graphs reference the UNBOUND variable (llama.cpp: positions are launch-time data,
    never graph structure), so the BIND node is pruned from the schedule; eager realize needs
    this carrier to resolve the variable. The JIT supplies the same value from its input args."""
    if isinstance(start_pos, UOp) and start_pos.op is Ops.BIND and len(start_pos.src) == 2 and \
       start_pos.src[1].op is Ops.CONST:
      return {start_pos.src[0].expr: start_pos.src[1].arg}
    return {}

  @staticmethod
  def _stamp_position_var_vals(t: Tensor, vv: dict[str, int]) -> Tensor:
    if vv: t._var_vals = vv
    return t

  def logits(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    _prefill = resolve(tokens.shape[1] != 1)
    _vv = Transformer._bound_position_var_vals(start_pos)
    x = _prefill_semantic(_prefill, prefill_activation, self.token_embd(tokens).float())  # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    _reduce_output_norm = not _prefill and getattr(self, "_decode_reduce_output_rmsnorm_promoted", False)
    with role_metadata("rms_norm"):
      x = _prefill_semantic(_prefill, prefill_scratch,
        _decode_reduce_output_rmsnorm(self.output_norm, x, _reduce_output_norm))
    if self._lm_head_wants_pf16():
      return Transformer._stamp_position_var_vals(_prefill_semantic(_prefill, prefill_output, _pf16(self.output, x).contiguous()), _vv)
    # The lazy LM head is still the selected output tensor's actual runtime invocation.  Record it before Tensor's
    # downstream final-token pruning; the prefill-forward context prevents the same call during decode from counting.
    notify_prefill_route(self.output)
    return Transformer._stamp_position_var_vals(_prefill_semantic(_prefill, prefill_output, self.output(x)), _vv)

  def _decode_vocab_top1_sample(self, tokens:Tensor, start_pos:int|UOp) -> Tensor | None:
    """Decode-only research route for the fused vocab top-1 epilogue.

    Mirrors ``logits`` up to the LM head, then replaces the full 151936-row logits materialization
    plus the four-kernel argmax tail with the P1 packed per-tile reduce.  Returns ``None`` for any
    shape/route that cannot be fused so the ordinary greedy path stays the live fallback.
    """
    _prefill = resolve(tokens.shape[1] != 1)
    if _prefill: return None
    x = _prefill_semantic(_prefill, prefill_activation, self.token_embd(tokens).float())
    for block in self.blk: x = block(x, start_pos)
    _reduce_output_norm = not _prefill and getattr(self, "_decode_reduce_output_rmsnorm_promoted", False)
    with role_metadata("rms_norm"):
      x = _prefill_semantic(_prefill, prefill_scratch,
        _decode_reduce_output_rmsnorm(self.output_norm, x, _reduce_output_norm))
    notify_prefill_route(self.output)
    return q6k_vocab_top1_call(self.output, x, self.output.route_admission.admitted)

  def _forward_sampled(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, with_logits:bool) -> Tensor|tuple[Tensor, Tensor]:
    # This runs inside the TinyJit body: feedback's lazy clone/store is captured in the decode schedule instead of
    # being realized by JIT input preparation as a separate per-token copy and synchronization.
    tokens = _runtime_input_boundary(tokens)
    _vv = Transformer._bound_position_var_vals(start_pos)
    logits = self.logits(tokens, start_pos)[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    scores = logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()
    native_threads = getattr(self, "_decode_native_argmax_lease", getattr(self, "_decode_native_argmax_threads", 0))
    sampled = native_argmax_finite_fp32(scores, native_threads) if native_threads and not resolve(tokens.shape[1] != 1) \
      else scores.argmax(-1, keepdim=True)
    sampled = _prefill_semantic(resolve(tokens.shape[1] != 1), prefill_output, sampled)
    # The diagnostic return must own a fresh replay-written allocation.  The
    # sampling path reads ``logits`` in-place, while a view of that internal
    # producer can be reused by the JIT memory plan after argmax has consumed
    # it.  Keep production byte-identical and return a held copy only for the
    # full-logit oracle.
    sampled = Transformer._stamp_position_var_vals(sampled, _vv)
    return (sampled, Transformer._stamp_position_var_vals(logits.clone(), _vv)) if with_logits else sampled

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return self._forward_sampled(tokens, start_pos, temperature, False)  # type: ignore[return-value]

  def forward_greedy(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    tokens = _runtime_input_boundary(tokens)
    _prefill = resolve(tokens.shape[1] != 1)
    if not _prefill and getattr(self, "_decode_vocab_top1_lease", False):
      sampled = self._decode_vocab_top1_sample(tokens, start_pos)
      if sampled is not None:
        return Transformer._stamp_position_var_vals(_prefill_semantic(_prefill, prefill_output, sampled),
                                                    Transformer._bound_position_var_vals(start_pos))
    logits = self.logits(tokens, start_pos)[:, -1, :]
    native_threads = getattr(self, "_decode_native_argmax_lease", getattr(self, "_decode_native_argmax_threads", 0))
    _host_mirror = getattr(self, "_decode_host_argmax_mirror", None)
    sampled = native_argmax_finite_fp32_host_mirror(logits, _host_mirror, native_threads)[0] if native_threads and _host_mirror is not None else \
      native_argmax_finite_fp32(logits, native_threads) if native_threads else \
      packed_argmax_finite_fp32(logits, -1, keepdim=True) if getattr(self, "_decode_packed_argmax_promoted", False) else \
      logits.argmax(-1, keepdim=True)
    return Transformer._stamp_position_var_vals(_prefill_semantic(resolve(tokens.shape[1] != 1), prefill_output, sampled),
                                                Transformer._bound_position_var_vals(start_pos))

  def forward_greedy_with_logits(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, Tensor]:
    tokens = _runtime_input_boundary(tokens)
    logits = self.logits(tokens, start_pos)[:, -1, :]
    native_threads = getattr(self, "_decode_native_argmax_lease", getattr(self, "_decode_native_argmax_threads", 0))
    _host_mirror = getattr(self, "_decode_host_argmax_mirror", None)
    sampled = native_argmax_finite_fp32_host_mirror(logits, _host_mirror, native_threads)[0] if native_threads and _host_mirror is not None else \
      native_argmax_finite_fp32(logits, native_threads) if native_threads else \
      packed_argmax_finite_fp32(logits, -1, keepdim=True) if getattr(self, "_decode_packed_argmax_promoted", False) else \
      logits.argmax(-1, keepdim=True)
    sampled = _prefill_semantic(resolve(tokens.shape[1] != 1), prefill_output, sampled)
    _vv = Transformer._bound_position_var_vals(start_pos)
    return Transformer._stamp_position_var_vals(sampled, _vv), Transformer._stamp_position_var_vals(logits.clone(), _vv)

  def forward_with_logits(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, Tensor]:
    """Diagnostic decode tap.  The logits already feed sampling; this only retains them as a JIT return."""
    return self._forward_sampled(tokens, start_pos, temperature, True)  # type: ignore[return-value]

  def forward_ring(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, freqs:Tensor) -> Tensor:
    # StreamingLLM ring decode: `freqs` is a per-step JIT INPUT (the slot-relative pre-gathered cos|sin table). Set it
    # on each block from INSIDE the traced fn so it binds to the graph input (a baked attribute would capture once).
    for block in self.blk: block._ring_freqs = freqs
    return self.forward(tokens, start_pos, temperature)

  def decode_with_logits(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, use_flash:bool=False,
                         feedback_slot:int|None=None, flash_split_count:int|None=None) -> tuple[Tensor, Tensor]:
    """Closed-surface diagnostic tap for a normal (non-ring) one-token decode.

    It deliberately rejects prefill and ring inputs rather than silently
    capturing a different production route.  Normal ``__call__`` remains the
    sole production entry point.
    """
    if resolve(tokens.shape[1] != 1): raise ValueError("decode_with_logits only supports one-token decode")
    if _GENERIC_LLM_CONTROL.get(): raise ValueError("decode_with_logits does not support generic-control routing")
    for q4k_linear in self._q4k_linears.linears: q4k_linear.decode_enabled = True
    flash_geometry = _flash_decode_geometry_for_split(
      getattr(self, "_flash_decode_tile_geometry_lease", None) or {}, flash_split_count)
    for index,block in enumerate(self.blk):
      block._use_flash, block._prefill_v2, block._is_prefill, block._ring_freqs, block._ring_full = use_flash, False, False, None, False
      block._flash_decode_tile_geometry_lease = _flash_block_geometry(self,index,flash_geometry) or None
    if feedback_slot not in (None, 0, 1): raise ValueError("feedback_slot must be None, 0, or 1")
    pingpong = bool(getattr(self, "_decode_feedback_pingpong_promoted", False)) and feedback_slot is not None
    greedy_flash_pair = _flash_jit_variant(flash_split_count, self.rollout_greedy_logits_pingpong_jits_flash,
      self.rollout_greedy_logits_pingpong_jits_flash_s6, self.rollout_greedy_logits_pingpong_jits_flash_s64)
    greedy_flash = _flash_jit_variant(flash_split_count, self.rollout_greedy_logits_jit_flash,
      self.rollout_greedy_logits_jit_flash_s6, self.rollout_greedy_logits_jit_flash_s64)
    greedy_jit = ((greedy_flash_pair if use_flash else self.rollout_greedy_logits_pingpong_jits)[feedback_slot]
                  if pingpong else (greedy_flash if use_flash else self.rollout_greedy_logits_jit))
    direct_greedy = bool(getattr(self, "_decode_direct_greedy_promoted", False))
    jit = greedy_jit if direct_greedy and float(temperature.item()) == 0.0 else \
          (_flash_jit_variant(flash_split_count, self.rollout_logits_jit_flash, self.rollout_logits_jit_flash_s6,
                              self.rollout_logits_jit_flash_s64) if use_flash else self.rollout_logits_jit)
    with prefill_route_scope(False), self._decode_callify_substrate(), self._decode_flash_load_schedule_substrate(use_flash):
      return jit(tokens, start_pos, temperature)

  @contextlib.contextmanager
  def _decode_callify_substrate(self):
    """M2c callify substrate: the block-output copy fold (and the reduce-output
    route when its policy promotes) is gated on callify Context flags that
    production decode normally leaves closed. When a promoted policy requires
    them, decode graph capture runs under the Context so the booked graph
    renders in production; closed policies keep the legacy closed-graph
    spelling (no Context, byte-identical legacy graph)."""
    if not getattr(self, "_decode_callify_substrate_promoted", False):
      yield
      return
    from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
    from tinygrad.helpers import Context
    with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
      yield

  @contextlib.contextmanager
  def _decode_flash_load_schedule_substrate(self, use_flash:bool):
    """Contain the measured NV Flash compiler and graph-layout contract.

    The score kernel needs the two-argument launch bound to expose its K/V
    request wall.  The corresponding token-wall qualification used an initial
    graph cap of 33 so score/combine do not straddle the first arbitrary
    doubling seam.  Apply both as one admitted capture lease: changing either
    independently reproduces the measured conversion failure.
    """
    if not use_flash or not getattr(self, "_decode_flash_load_schedule_promoted", False):
      yield
      return
    from tinygrad.helpers import Context
    with Context(NV_FLASH_LOAD_SCHEDULE=1, JIT_BATCH_SIZE=33): yield

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, use_flash:bool=False,
               ring_freqs:Tensor|None=None, ring_full:bool=False, greedy:bool=False, feedback_slot:int|None=None,
               flash_split_count:int|None=None) -> Tensor:
    is_prefill = resolve(tokens.shape[1] != 1)
    generic_control = _GENERIC_LLM_CONTROL.get()
    # prefill v2: only when opt-in AND this is a CONCRETE-batch prefill chunk. Normal prefill passes a symbolic
    # v_toks (tokens.shape[1] is a UOp -> not int), so the two paths never collide; decode is T==1.
    is_prefill_v2 = self.config.prefill_v2 and prefill_v2_target_admitted(self.config.prefill_device_facts) and \
      is_prefill and isinstance(tokens.shape[1], int) and not generic_control
    for q4k_linear in self._q4k_linears.linears:
      q4k_linear.decode_enabled = not is_prefill and not generic_control
    # context-aware flash: each block reads _use_flash at trace time; rollout_jit (SDPA) and
    # rollout_jit_flash bake distinct attention -- each is only ever called with its own use_flash, so
    # capture is consistent. The decode-only T==1 guard in _attention ignores it during prefill.
    flash_geometry = _flash_decode_geometry_for_split(
      getattr(self, "_flash_decode_tile_geometry_lease", None) or {}, flash_split_count)
    for index,block in enumerate(self.blk):
      block._use_flash, block._prefill_v2, block._is_prefill, block._ring_freqs, block._ring_full = \
        use_flash, is_prefill_v2, is_prefill, None, ring_full
      block._flash_decode_tile_geometry_lease = _flash_block_geometry(self,index,flash_geometry) or None
    if is_prefill_v2 and getenv("NV_Q4_IMMA_PP512", 0):
      from extra.llm_research.prefill.nv_q4_imma_pp512_binding import binding_for
      _nv_binding = binding_for("NV")
      _nv_binding.prepare_outputs(len(self.blk)*2)
      _nv_binding.begin_trace()
      for block in self.blk: block._nv_q4_imma_pp512_binding = _nv_binding
    # StreamingLLM ring decode: distinct captured graphs with `freqs` as a per-step JIT input (rebound each token). The
    # FULL-phase graph (ring_full, ctx>=N) reads the whole [0:N] cache and writes at the wrapped slot; the FILL-phase
    # graph reads [0:start_pos+T] like normal decode. block._ring_full (baked bool) selects the read mode in _attention.
    if feedback_slot not in (None, 0, 1): raise ValueError("feedback_slot must be None, 0, or 1")
    if ring_freqs is not None and not is_prefill:
      _rjit = self.rollout_jit_ring_full if ring_full else self.rollout_jit_ring
      return _rjit(tokens, start_pos, temperature, ring_freqs)
    # concrete-KV: a CONCRETE int start_pos (KV concrete -> attention TC fires) gets a per-start_pos jit.
    if is_prefill_v2 and isinstance(start_pos, int):
      jit = self.prefill_v2_jits.setdefault((start_pos, greedy), TinyJit(self.forward_greedy if greedy else self.forward))
    else:
      pingpong = bool(getattr(self, "_decode_feedback_pingpong_promoted", False)) and feedback_slot is not None
      greedy_flash_pair = _flash_jit_variant(flash_split_count, self.rollout_greedy_pingpong_jits_flash,
        self.rollout_greedy_pingpong_jits_flash_s6, self.rollout_greedy_pingpong_jits_flash_s64)
      greedy_flash = _flash_jit_variant(flash_split_count, self.rollout_greedy_jit_flash,
        self.rollout_greedy_jit_flash_s6, self.rollout_greedy_jit_flash_s64)
      rollout_greedy = ((greedy_flash_pair if use_flash else self.rollout_greedy_pingpong_jits)[feedback_slot]
                        if pingpong else (greedy_flash if use_flash else self.rollout_greedy_jit))
      jit = ((self.prefill_v2_greedy_jit if greedy else self.prefill_v2_jit) if is_prefill_v2 else (self.prefill_greedy_jit if greedy else self.prefill_jit)) if is_prefill else \
            (rollout_greedy if greedy else
             (_flash_jit_variant(flash_split_count, self.rollout_jit_flash, self.rollout_jit_flash_s6,
                                 self.rollout_jit_flash_s64) if use_flash else self.rollout_jit))
    if not is_prefill_v2:
      if not is_prefill:
        # Decode captures under the M2c callify substrate when a promoted policy
        # requires it; prefill graphs always stay on the closed-graph spelling.
        with prefill_route_scope(is_prefill), self._decode_callify_substrate(), self._decode_flash_load_schedule_substrate(use_flash):
          return jit(tokens, start_pos, temperature)
      with prefill_route_scope(is_prefill): return jit(tokens, start_pos, temperature)
    # contain the ambient codegen power: install the warmstart table ONLY around the prefill-v2 forward (it's
    # consulted at kernel-compile time, i.e. this jit's first call), then restore -- decode/other paths never
    # see a populated _WARMSTART_OPTS even within this process. The packed-wmma table (only non-empty when
    # prefill_routes.packed_wmma_prefill_enabled() was set at model init, see _build_packed_wmma_warmstart) is
    # merged on top of the dense fp16 table.
    import tinygrad.codegen.opt.postrange as pr
    merged_opts = {**(self._pf16_warmstart or {}), **(self._packed_wmma_warmstart or {})}
    with prefill_route_scope(True), pr.warmstart_candidate_state(merged_opts, self._packed_wmma_warmstart_contexts):
      return jit(tokens, start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:"int|str|None"=None,
                realize:bool=False, stream:str="auto") -> tuple[Transformer, dict]:
    # Probe free VRAM at ENTRY, before gguf_load makes the weight storage resident -- so `free` is the baseline
    # available for weights+KV (the admission budget then subtracts weights itself; probing after gguf_load would
    # double-count weights already in `used`). Total is stable regardless.
    _measurement_authority = _MEMORY_ADAPTIVE_MEASUREMENT_AUTHORITY.get()
    _device_facts = scan_device_facts() if _measurement_authority is None else _measurement_authority[0]
    # Whether the Q4_K/Q6_K install functions even RUN no longer depends on a hardcoded target match (TG3,
    # docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md): the only remaining
    # precondition here is having a selected GGUF path to read tensors from. Capability (renderer/device
    # facts) and promotion (route-plan policy) are resolved per-tensor inside _install_q4k_primitives /
    # _install_q6k_primitives (tinygrad/llm/qk_primitives.py), so a target that cannot be admitted is recorded
    # in the route census instead of silently skipping installation altogether.
    use_q4k_primitive = use_q6k_primitive = not isinstance(gguf, Tensor)
    _authority_workload = _measurement_authority[2] if _measurement_authority is not None else {}
    _prefill_ubatch = int(_authority_workload.get("prefill_ubatch", PREFILL_UBATCH))
    if _prefill_ubatch <= 0: raise ValueError("selected prefill candidate requires a positive physical M")
    _requested_max_context, _admit_resolved, _ring_admitted = max_context, False, False
    _kv_quant = False
    _overlay_request = None
    _workload_reuse = False
    _runtime_policy = immutable_prefill_policy({"strategy": "DIRECT_PACKED_FALLBACK", "candidate_id": "direct-packed-baseline",
      "routes": {}, "provenance": "preloaded tensors have no selected-GGUF inventory", "measured": False})
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
      _automatic_overlay_policy = None
      if prefill_policy_strategy(_runtime_policy) == "DIRECT_PACKED_FALLBACK" and _runtime_policy.get("measured") is False:
        _automatic_overlay_policy = automatic_promoted_prefill_graph_policy(
          _runtime_inventory, _device_facts.planning_snapshot())
      _overlay_request = prefill_policy_uses_overlay(_runtime_policy)
      _admit_arch = _admit_kv["general.architecture"]
      _q4_bytes = selected_gguf_backing_bytes(gguf, _device_facts.capabilities.global_allocation_granularity)
      if _q4_bytes is None:
        raise RuntimeError(f"{_admit_arch}: selected-GGUF backing allocation is unknown from the selected path and scanned allocation granularity")
      _est_fp16 = _runtime_inventory["overlay_bytes"]
      _admission_inputs = AdmissionInputs.from_model_metadata(_requested_max_context, _admit_kv,
        free_vram=_device_facts.free_vram_bytes, q4_bytes=_q4_bytes, est_fp16=_est_fp16,
        prefill_ubatch=_prefill_ubatch, v2_on=_device_facts.capabilities.supports_fp16 is True,
        model_label=f"{_admit_arch} selected GGUF", stream=str(stream), kv_quant_supported=True,
        live_split_s=FLASH_DECODE_CANDIDATE.split_size)
      # The promotion lookup is a caller concern; its result is passed in as a preference, never as a
      # single-strategy restriction (R1/R6). A preferred overlay that does not fit degrades to the packed
      # evaluation inside the planner -- no None -> True -> False replan loop.
      _policy = _automatic_overlay_policy if _automatic_overlay_policy is not None else (
        _runtime_policy if _overlay_request else None)
      _plan, _memory_plan, _effective_strategy = plan_selected_model_memory(
        _admission_inputs, _device_facts, direct_packed_supported=True, policy=_policy)
      if _effective_strategy is Strategy.FULL_RESIDENT_OVERLAY:
        if _automatic_overlay_policy is not None:
          selected = dict(_automatic_overlay_policy)
          selected["memory_plan"] = json.loads(_plan.prefill_memory_plan)
          _runtime_policy = immutable_prefill_policy(selected)
      elif prefill_policy_uses_overlay(_runtime_policy):
        # Planned packed: a retained overlay-strategy policy would realize fp16 weights against a packed plan.
        _runtime_policy = immutable_prefill_policy({"strategy": "DIRECT_PACKED_FALLBACK",
          "candidate_id": "direct-packed-baseline", "routes": {},
          "provenance": "overlay preference degraded to packed at admission", "measured": False})

      _provided_route_memory = _runtime_policy.get("memory_facts")
      if _provided_route_memory is not None and not isinstance(_provided_route_memory, dict):
        raise ValueError("prefill policy memory_facts must be a mapping when present")
      _route_memory = dict(_provided_route_memory or {})
      if _route_memory and any(_route_memory.get(key) is None for key in _EXACT_ROUTE_MEMORY_KEYS):
        raise ValueError("prefill policy contains partial memory_facts; exact allocation evidence must be complete")
      if (prefill_policy_strategy(_runtime_policy) != "DIRECT_PACKED_FALLBACK" and not _route_memory and
          _runtime_policy.get("selection_authority") != "promoted_exact_candidate" and
          _MEMORY_ADAPTIVE_MEASUREMENT_AUTHORITY.get() is None):
        raise ValueError("accelerated prefill policy cannot bypass exact memory admission")
      if isinstance(_runtime_policy.get("memory_fact_evidence"), dict):
        _route_memory["provenance"] = json.dumps(_runtime_policy["memory_fact_evidence"]["provenance"],
                                                   sort_keys=True, separators=(",", ":"))
      _route_memory["candidate_id"] = _runtime_policy["candidate_id"]
      max_context, _kv_quant = _plan.max_context, _plan.kv_quant
      _admit = dict(_plan.report)
      # S4 census (R3): the fp16 overlay is expressible on this target but no promoted candidate exists for
      # (backend, arch, wave_size) -- the load continues on packed, loudly labeled in the admission report and
      # the e2e bench row, never a silent drop.
      if _device_facts.capabilities.supports_fp16 is True and _automatic_overlay_policy is None and not _overlay_request:
        _admit["prefill_overlay_promotion"] = "no-promoted-candidate"
      elif _automatic_overlay_policy is not None and _effective_strategy is Strategy.FULL_RESIDENT_OVERLAY:
        _admit["prefill_overlay_promotion"] = _automatic_overlay_policy["graph_gemm"]["candidate_set_identity"]
      # A completed machine-search record may carry a fully attributed allocation ledger. Apply it only after context
      # and KV representation are resolved. Ordinary direct-packed loading needs no caller-injected hardware facts.
      if all(_route_memory.get(key) is not None for key in _EXACT_ROUTE_MEMORY_KEYS):
        _geometry = RuntimeGeometry(_admission_inputs.num_blocks, _admission_inputs.n_kv_heads,
          _admission_inputs.head_dim, max_context, _prefill_ubatch,
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
      _print_admission(_plan, "(int8)" if _kv_quant else "", f"trained {_admission_inputs.trained_ctx}, fp16-cap {_admit.get('mc_fp16', '-')}, q8-cap {_admit.get('mc_q8', '-')}")
      _admit_resolved = True
    # A single-file path keeps its raw tensor table even on generic backends
    # (notably Metal). This is the exact source authority for fused dequant
    # calls; eligibility controls primitive replacement, not observability.
    if not isinstance(gguf, Tensor) and _admit_kv.get('split.count', 1) <= 1:
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
      _q4_bytes = pathlib.Path(gguf).stat().st_size if not isinstance(gguf, Tensor) else 0
      _est_fp16 = estimate_prefill_overlay_bytes((k, t.numel()) for k, t in state_dict.items())
      _admission_inputs = AdmissionInputs.from_model_metadata(_requested_max_context, kv,
        free_vram=_device_facts.free_vram_bytes, q4_bytes=_q4_bytes, est_fp16=_est_fp16,
        prefill_ubatch=_prefill_ubatch, v2_on=_device_facts.capabilities.supports_fp16 is True,
        stream=str(stream), live_split_s=FLASH_DECODE_CANDIDATE.split_size)
      _plan, _memory_plan, _effective_strategy = plan_selected_model_memory(_admission_inputs,
        _device_facts, direct_packed_supported=True)
      max_context, _kv_quant, _admit = _plan.max_context, _plan.kv_quant, _plan.report
      _ring_admitted = _admit.get("ring", False)
      _print_admission(_plan, "", f"trained {_admission_inputs.trained_ctx}, mem-cap {_admit.get('mc_mem', '-')}")

    _runtime_policy = select_prefill_runtime_policy(_runtime_policy, scanned_device_facts=_device_facts,
      workload_reuse=_workload_reuse)
    _workload_reuse = bool(_runtime_policy.get("workload_reuse", False))
    # S6: the runtime prefill-v2 flag is True for every executed strategy (REFUSE raises at admission), so it
    # is folded to True here; the admission-time capability keeps the v2_on name on AdmissionInputs.
    _concrete_kv, _ = prefill_concrete_kv_auto_decision(_workload_reuse, True)
    _flash_decode = any(candidate.bind(1, n_heads, n_kv_heads, head_dim, _device_facts.selected_device) is not None
                        for candidate in (FLASH_DECODE_CANDIDATE, FLASH_DECODE_G5_CANDIDATE))
    # Warm the fused-attention spec-target cache at load time (decode's resolve-once pattern,
    # decode_routes.py:151): custom_kernel_attention runs inside a Tensor Function dispatch where
    # Device[...] is disallowed, so the runtime lookup must be a cache hit. Devices without a renderer
    # stay unwarmed and fail closed to SDPA at the call site.
    try:
      from tinygrad.llm.fused_attention import warm_attention_spec_target
      warm_attention_spec_target(_device_facts.selected_device)
    except ValueError:
      pass

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
      prefill_tc_attn=bool(_runtime_policy["prefill_tc_attn"]),
      prefill_custom_kernel_attn=_should_use_custom_kernel_prefill_attn(n_heads, n_kv_heads, _device_facts.backend, _device_facts.architecture),
      prefill_v2=True, prefill_ubatch=_prefill_ubatch, prefill_concrete_kv=_concrete_kv,
      prefill_workload_reuse=_workload_reuse, flash_decode=_flash_decode, lm_head_route="lazy",
      kv_quant=_kv_quant, ring=_ring_admitted)
    if config.prefill_v2: validate_prefill_route_mode(config.n_heads, config.n_kv_heads)
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
    # L1 M3: the fused decode RMSNorm route resolves ONCE from the load-entry facts (same
    # (backend, arch) the QK primitives use), never from a target-string guess. Blocks read
    # their own copy so the traced norm call sites need no transformer back-reference.
    _norm_cap = qk_primitive_capability_from_device_facts(_device_facts)
    # Dense S8 Flash load-schedule promotion.  The substrate itself is generic
    # (capture-scoped launch bounds + graph cap), while admission remains at
    # the exact topology qualified at token wall.  Quantization is deliberately
    # absent from this predicate: the Flash graph shape, not the weight format,
    # owns the contract.  MoE stays closed until it receives its own lifecycle
    # qualification.
    model._decode_flash_load_schedule_promoted = bool(
      decode_flash_llama_vec_wide_promoted((_norm_cap.backend, _norm_cap.architecture)) and
      not getenv("TINYGRAD_FLASH_LOAD_SCHEDULE_DISABLE", 0) and config.num_experts == 0 and
      config.num_blocks == 36 and config.n_heads == 32 and config.n_kv_heads == 8 and
      config.head_dim == 128 and config.max_context == 1024)
    # Active-horizon selection is inseparable from explicit capture placement:
    # without both S6/S8 ping-pong pairs prewarmed, the Tc=769 lazy capture
    # spike erases the steady-state saving. `generate` therefore activates
    # the selector only after `_prewarm_active_horizon_flash_pairs` succeeds.
    model._flash_decode_active_horizon_lease = bool(
      model._decode_flash_load_schedule_promoted and
      not getenv("TINYGRAD_FLASH_ACTIVE_HORIZON_DISABLE", 0))
    _norm_promoted = decode_norm_fusion_promoted((_norm_cap.backend, _norm_cap.architecture))
    model._decode_norm_fusion_promoted = _norm_promoted
    for _b in model.blk: _b._decode_norm_fusion_promoted = _norm_promoted
    # L1 M4: q4k GEMV epilogue fusion gate. CLOSED default (decode-q4k-epilogue-fusion-route-policy.json,
    # measured non-landing, m4-q4k-epilogue-measurement-record-20260802.md); M2's Q6K in-kernel merge
    # keeps its own separate record (decode-epilogue-fusion-route-policy.json, NV sm_120). Same
    # resolve-once pattern as M3 -- blocks carry their own copy so the traced call sites need no back-ref.
    _q4k_epi_promoted = decode_q4k_epilogue_fusion_promoted((_norm_cap.backend, _norm_cap.architecture))
    model._decode_q4k_epilogue_fusion_promoted = _q4k_epi_promoted
    for _b in model.blk: _b._decode_q4k_epilogue_fusion_promoted = _q4k_epi_promoted
    # L1 M4 o-proj residual_add variant gate. CLOSED default (decode-q4k-epilogue-resadd-route-policy.json,
    # m4-resadd-landing-scope-20260806.md); separate from the combined M4 record so this one measured
    # copy-free variant can promote alone. Same resolve-once pattern as the M4 combined gate.
    _q4k_resadd_promoted = decode_q4k_epilogue_resadd_promoted((_norm_cap.backend, _norm_cap.architecture))
    model._decode_q4k_epilogue_resadd_promoted = _q4k_resadd_promoted
    for _b in model.blk: _b._decode_q4k_epilogue_resadd_promoted = _q4k_resadd_promoted
    # L1 GEMV substrate: decode shared-Q8 attention group. CLOSED default
    # (decode-shared-q8-attention-route-policy.json, empty promoted_targets until the
    # section-6 full gate passes, nv-gemv-substrate-landing-scope-20260808.md). Same
    # resolve-once pattern as the M4 gate; the loader installs the booked max18
    # cooperative lease (blocks 1-12, 14-18, and 25) only when the record promotes NV sm_120.
    _shared_q8_promoted = decode_shared_q8_attention_promoted((_norm_cap.backend, _norm_cap.architecture))
    model._decode_shared_q8_attention_promoted = _shared_q8_promoted
    for _b in model.blk: _b._decode_shared_q8_attention_promoted = _shared_q8_promoted
    # Q6 V direct-output consumer sub-variant gate. CLOSED default
    # (decode-q6-direct-shared-q8-attention-route-policy.json, empty promoted_targets until
    # the all-depth gate passes, nv-q6-direct-shared-q8-promotion-scope-20260814.md). It only
    # changes the Q6_K V consumer inside an already-installed shared-Q8 lease; Q4_K V blocks and
    # the shared Q8_1 provider are untouched, so the flag is inert unless the group lease above
    # is also promoted. Same resolve-once pattern as the group gate.
    _q6_direct_promoted = decode_q6_direct_shared_q8_attention_promoted((_norm_cap.backend, _norm_cap.architecture))
    model._decode_q6_direct_shared_q8_attention_promoted = _q6_direct_promoted
    for _b in model.blk: _b._decode_q6_direct_shared_q8_attention_promoted = _q6_direct_promoted
    # Cooperative Q4/Q8 direct-output sub-variant. It retains the four warp
    # partials and their left-to-right fp32 association, but merges them inside
    # the producer CTA. This removes 43 completion nodes in the max17 group.
    # TINYGRAD_SHARED_Q8_Q4_PARTIAL_OUTPUT=1 restores the prior partial ABI.
    _q4_direct_promoted = _shared_q8_promoted and decode_q4_direct_shared_q8_attention_promoted(
      (_norm_cap.backend, _norm_cap.architecture)) and not getenv("TINYGRAD_SHARED_Q8_Q4_PARTIAL_OUTPUT", 0)
    model._decode_q4_direct_shared_q8_attention_promoted = _q4_direct_promoted
    for _b in model.blk: _b._decode_q4_direct_shared_q8_attention_promoted = _q4_direct_promoted
    # Shared-Q8 Q4/Q4 K/V dual-output producer. This is separately gated from
    # the direct-output consumer and is inert unless both prerequisite shared
    # Q8 and cooperative direct-output records are active.
    _shared_q8_q4kv_pair_promoted = _q4_direct_promoted and decode_shared_q8_q4kv_pair_promoted(
      (_norm_cap.backend,_norm_cap.architecture))
    model._decode_shared_q8_q4kv_pair_promoted = _shared_q8_q4kv_pair_promoted
    for _b in model.blk: _b._decode_shared_q8_q4kv_pair_promoted = _shared_q8_q4kv_pair_promoted
    _shared_q8_q4q6_pair_promoted = _q4_direct_promoted and _q6_direct_promoted and \
      decode_shared_q8_q4q6_kv_pair_promoted((_norm_cap.backend,_norm_cap.architecture))
    model._decode_shared_q8_q4q6_kv_pair_promoted = _shared_q8_q4q6_pair_promoted
    for _b in model.blk: _b._decode_shared_q8_q4q6_kv_pair_promoted = _shared_q8_q4q6_pair_promoted
    _shared_q8_q4q4_full_promoted = _q4_direct_promoted and decode_shared_q8_q4q4_qkv_full_promoted(
      (_norm_cap.backend,_norm_cap.architecture))
    model._decode_shared_q8_q4q4_qkv_full_promoted = _shared_q8_q4q4_full_promoted
    # Fused w1+w3 (gate/up) decode GEMV gate. CLOSED default (decode-q4k-w1w3-fusion-route-policy.json,
    # NV sm_120 promoted, q4k-w1w3-fused-qv-implementation-record-20260803.md). Same resolve-once
    # pattern as the M4 gate; the fused call additionally requires BOTH ffn_gate and ffn_up to be
    # admitted Q4K primitives, so the model flag alone never changes the legacy chain.
    _w1w3_promoted = decode_q4k_w1w3_fusion_promoted((_norm_cap.backend, _norm_cap.architecture))
    model._decode_q4k_w1w3_fusion_promoted = _w1w3_promoted
    for _b in model.blk: _b._decode_q4k_w1w3_fusion_promoted = _w1w3_promoted
    # M2a fp16-store spelling gate (decode-q4k-w1w3-fp16-store-route-policy.json, NV sm_120 promoted,
    # nv-epilogue-absorption-m2a-promotion-record-20260812.md). SEPARATE record from the w1w3 fusion
    # gate above: the fused16 kernel stores fp16 in-kernel, absorbing the E_128_32_3 ffn-activation
    # cast; it only applies when the fused kernel itself is admitted, so the flag is ANDed with the
    # w1w3 admission. The harness lease `_q4k_w1w3_fp16_store_lease` still forces the spelling where
    # the loader record is closed (research arms), so the AB contract is unchanged.
    _w1w3_fp16_promoted = _w1w3_promoted and decode_q4k_w1w3_fp16_store_promoted((_norm_cap.backend, _norm_cap.architecture))
    model._decode_q4k_w1w3_fp16_store_promoted = _w1w3_fp16_promoted
    for _b in model.blk: _b._decode_q4k_w1w3_fp16_store_promoted = _w1w3_fp16_promoted
    _gateup_fourwarp_promoted = _w1w3_fp16_promoted and decode_q4k_gate_up_four_warp_vector_promoted(
      (_norm_cap.backend, _norm_cap.architecture)) and not getenv("TINYGRAD_Q4K_GATE_UP_FOUR_WARP_DISABLE", 0)
    model._decode_q4k_gate_up_four_warp_vector_promoted = _gateup_fourwarp_promoted
    for _b in model.blk: _b._decode_q4k_gate_up_four_warp_vector_promoted = _gateup_fourwarp_promoted
    # M2b+M2c ffn-down residual-add absorption gate (decode-ffn-down-resadd-route-policy.json,
    # NV sm_120 promoted, nv-epilogue-absorption-m2c-ab-20260811.json BOOKED). M2b: the ffn_down
    # Q4K/Q6K GEMV absorbs the h+ffn_out add in-kernel; M2c: the declared epilogue-absorbing block
    # output gets its CALL rebound to the caller output slot so the identity copies fold. The flag
    # is carried on the model and every block; the ffn_down linear copies are installed after the
    # Q4K/Q6K primitive replacement below (the primitives are the objects that carry route_role).
    # The harness lease _ffn_down_resadd_lease still forces the route where the record is closed.
    _ffn_resadd_promoted = decode_ffn_down_resadd_promoted((_norm_cap.backend, _norm_cap.architecture))
    model._decode_ffn_down_resadd_promoted = _ffn_resadd_promoted
    for _b in model.blk: _b._decode_ffn_down_resadd_promoted = _ffn_resadd_promoted
    # Q4_K FFN-down four-warp fp16 geometry route (decode-q4k-ffn-down-fp16-geometry-route-policy.json,
    # NV sm_120 promoted, -100.3 us at d512). It replaces the installed 1-warp/row epi_ffnresadd GEMV
    # with the 4-warp/row fp16-FMA direct consumer; the control topology it swaps against requires the
    # M2b ffn-down residual-add promotion, so the flag is ANDed with _ffn_resadd_promoted. The Q4-only
    # admission is installed on the replaced Q4K primitives after they are built below.
    model._decode_q4k_ffn_down_fp16_geometry_promoted = _ffn_resadd_promoted and \
      decode_q4k_ffn_down_fp16_geometry_promoted((_norm_cap.backend, _norm_cap.architecture))
    # Q6_K FFN-down four-warp fp16 geometry route (decode-q6k-ffn-down-fp16-geometry-route-policy.json,
    # NV sm_120 promoted, device 25.7 us vs 31.0 us control). Same closed-default shape as the Q4
    # analog: it replaces the row_tile-2 coop Q6 consumer with the 4-warp/row fp16 direct consumer,
    # requires the M2b ffn-down residual-add promotion, and is ANDed with it at resolve time.
    model._decode_q6k_ffn_down_fp16_geometry_promoted = _ffn_resadd_promoted and \
      decode_q6k_ffn_down_fp16_geometry_promoted((_norm_cap.backend, _norm_cap.architecture))
    model._decode_q6k_ffn_down_packed_lanemap_promoted = model._decode_q6k_ffn_down_fp16_geometry_promoted and \
      decode_q6k_ffn_down_packed_lanemap_promoted((_norm_cap.backend, _norm_cap.architecture))
    model._decode_q6k_ffn_down_unroll_promoted = model._decode_q6k_ffn_down_packed_lanemap_promoted and \
      decode_q6k_ffn_down_unroll_promoted((_norm_cap.backend, _norm_cap.architecture))
    # Q6_K attention-V four-warp fp16 geometry route (decode-q6k-v-four-warp-fp16-geometry-route-policy.json,
    # NV sm_120 promoted, in-loop 5.12 us vs 17.94 us control, -147.35 us/token wall bracket). It replaces
    # the parts route on the 10 Q6_K attention-V blocks with a 128-thread four-warp fp16 direct consumer;
    # the V output binds to the kv-store route as a flat fp32 view (vparts=1), so no parts reduce is left.
    model._decode_q6k_v_four_warp_promoted = decode_q6k_v_four_warp_fp16_geometry_promoted(
      (_norm_cap.backend, _norm_cap.architecture))
    # Decode kv-store chain fusion gate (decode-kv-store-chain-fusion-scope-20260803.md). CLOSED default
    # (decode-kv-store-fusion-route-policy.json, empty promoted_targets until a same-session A/B record
    # lands). Same resolve-once pattern as the w1w3 gate; blocks carry their own copy so the traced
    # _attention call sites need no transformer back-reference. The _attention gate additionally
    # requires decode shape (T==1,B==1), full-head rope, qk_norm in (0, head_dim), no rope-at-read, and
    # an fp16 cache, so the model flag alone never changes the legacy chain.
    _kv_store_promoted = decode_kv_store_fusion_promoted((_norm_cap.backend, _norm_cap.architecture))
    model._decode_kv_store_fusion_promoted = _kv_store_promoted
    for _b in model.blk: _b._decode_kv_store_fusion_promoted = _kv_store_promoted
    # Path 3 semantic RMSNorm: per-site marker flags resolve once from load-entry facts. The
    # qualified 4096-wide attention/FFN/output sites use native lowering; Q/K keep the fused
    # reduce-output + RoPE/cache route. nn.RMSNorm additionally rejects prefill shapes.
    _norm_target = (_norm_cap.backend, _norm_cap.architecture)
    _rmsnorm_native_promoted = decode_rmsnorm_native_lowering_promoted(_norm_target)
    model._decode_rmsnorm_native_promoted = _rmsnorm_native_promoted
    for _b in model.blk:
      for _name in ("attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm"):
        _norm = getattr(_b, _name, None)
        if _norm is None: continue
        _norm._rmsnorm_native_promoted = decode_rmsnorm_native_lowering_site_promoted(_norm_target, _name)
        if _norm._rmsnorm_native_promoted and _name in ("attn_norm", "ffn_norm"):
          _norm._rmsnorm_native_output_dtype = dtypes.float16
    model.output_norm._rmsnorm_native_promoted = decode_rmsnorm_native_lowering_site_promoted(_norm_target, "output_norm")
    # Cooperative ordinary-CALL reduce/output route.  Promotion lives on the
    # model/block call sites, never nn.RMSNorm: the call sites gate the marker
    # on ``not _prefill`` before late concrete-view admission runs.
    _reduce_output_promoted = decode_reduce_output_rmsnorm_promoted((_norm_cap.backend, _norm_cap.architecture))
    model._decode_reduce_output_rmsnorm_promoted = _reduce_output_promoted
    # The FFN-norm site stays CLOSED by default even when the fp32 q/k route is
    # promoted.  The residual-bind lowers one cooperative 1_4096 body per FFN
    # norm, but on the GPU that body plus its fresh output materialization costs
    # ~7.5 us against ~6 us for the ordinary reduce + epilogue it replaces, so
    # the d512 reverse wall bracket is net-negative (192.36 -> 190.47 tok/s,
    # C6 19 -> 55).  The booked policy keeps C6 at 19; harnesses open this knob
    # explicitly when the FFN site is under measurement.
    model._decode_reduce_output_ffn_rmsnorm_promoted = False
    for _b in model.blk:
      _b._decode_reduce_output_rmsnorm_promoted = _reduce_output_promoted
      _b._decode_reduce_output_ffn_rmsnorm_promoted = False
    # Full-head Q/K norm+RoPE epilogue. This is a semantic REDUCE_OUTPUT
    # lowering, not an opaque KernelProgram boundary; target policy remains
    # separate and is additionally gated by the established q/k marker route.
    _qk_norm_rope_promoted = _reduce_output_promoted and decode_qk_norm_rope_promoted(
      (_norm_cap.backend, _norm_cap.architecture))
    model._decode_qk_norm_rope_promoted = _qk_norm_rope_promoted
    for _b in model.blk: _b._decode_qk_norm_rope_promoted = _qk_norm_rope_promoted
    # Producer-owned K/V cache sink. The terminal 8x128 K RMSNorm+RoPE body
    # writes K and final V directly to the current fp16/fp32 cache slot,
    # deleting the 36 generic cache-store joins. It composes only with the
    # already-promoted exact Q/K norm+RoPE route and remains shape-gated in the
    # call site. The environment switch restores the complete legacy chain.
    _producer_kv_sink_promoted = _qk_norm_rope_promoted and decode_producer_kv_cache_sink_promoted(
      (_norm_cap.backend, _norm_cap.architecture))
    model._decode_producer_kv_cache_sink_promoted = _producer_kv_sink_promoted
    if _producer_kv_sink_promoted:
      for _index, _b in enumerate(model.blk):
        _b._producer_kv_cache_sink_admission = ProducerKVCacheSinkAdmission(_index)
    # Ordinary Q4/Q4 K/V dual-output producer. Admission is installed only
    # after primitive replacement and after the shared-Q8 leases are known;
    # this resolve-once flag carries only the target/rollback policy.
    model._decode_q4k_kv_pair_promoted = decode_q4k_kv_pair_promoted(
      (_norm_cap.backend, _norm_cap.architecture))
    model._decode_q4k_q4q4_qkv_full_promoted = decode_q4k_q4q4_qkv_full_promoted(
      (_norm_cap.backend,_norm_cap.architecture))
    # One-CTA finite-fp32 decode argmax. The native reducer replaces the three
    # scheduler reduction kernels at the sampled-score tail and keeps a held
    # one-element clone for replay-safe feedback. NV sm_120 is promoted after
    # a token-exact reps=9 reverse bracket recovered 56.386 us/token. The
    # environment switch is an explicit load-time rollback/control arm.
    model._decode_native_argmax_threads = decode_native_argmax_threads((_norm_cap.backend, _norm_cap.architecture))
    # M2c callify substrate: the block-output copy fold and the fp32 q/k reduce-output spelling are
    # gated on the callify owned-precompiled-output-redirect / typed-semantic-input-producer Context
    # flags, which production decode normally leaves closed. When a promoted policy requires them
    # (M2b/M2c here; the reduce-output route when it promotes), the decode path runs under the
    # Context so the booked graph renders in production (see _decode_callify_substrate below).
    model._decode_callify_substrate_promoted = bool(_ffn_resadd_promoted or _reduce_output_promoted)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    if _norm_promoted:
      # Materialize the packed norm weights once at load (~600KB total fp16) so the fused decode
      # kernels read plain buffers. Failure on any norm just declines its fused route (legacy graph).
      for _b in model.blk:
        for _name in ("attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm"):
          _norm = getattr(_b, _name, None)
          if _norm is None: continue
          try:
            _norm._decode_fused_weight = Tensor.empty(_norm.weight.shape[-1], dtype=dtypes.float16,
                                                      device=_norm.weight.device).assign(
              _norm.weight.cast(dtypes.float16).contiguous()).realize()
          except Exception:
            _norm._decode_fused_weight = None
    # The reduce-output route needs the same identity contract for its marker
    # weight: the selector admits an identity buffer directly, while a lazy
    # fp16 cast would need one fresh weight materialization per fused body
    # (the measured 8eeb0be1 overhead in the phase-6 bracket).  Materialize
    # once at load, mirroring _decode_fused_weight above.  This is small
    # (~1.4MB fp16 total) and harmless when the route stays closed.
    for _b in model.blk:
      for _name in ("attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm"):
        _norm = getattr(_b, _name, None)
        if _norm is None or getattr(_norm, "weight", None) is None: continue
        try:
          _norm._decode_reduce_output_weight = Tensor.empty(_norm.weight.shape[-1], dtype=dtypes.float16,
                                                            device=_norm.weight.device).assign(
            _norm.weight.cast(dtypes.float16).contiguous()).realize()
        except Exception:
          _norm._decode_reduce_output_weight = None
    _out_norm = getattr(model, "output_norm", None)
    if _out_norm is not None and getattr(_out_norm, "weight", None) is not None:
      try:
        _out_norm._decode_reduce_output_weight = Tensor.empty(_out_norm.weight.shape[-1], dtype=dtypes.float16,
                                                              device=_out_norm.weight.device).assign(
          _out_norm.weight.cast(dtypes.float16).contiguous()).realize()
      except Exception:
        _out_norm._decode_reduce_output_weight = None
    if q4k_meta is not None:
      model_facts = model_facts_from_gguf_metadata(kv, q4k_meta)
      route_plan = build_model_route_plan(q4k_meta, model_facts)
      # The production candidate views the selected GGUF backing directly;
      # it never creates an unchecked model-sized sidecar.
      qk_cfg = QKConfig(False, None, "shared", "shared", False, False)
      primitive_linears = []
      primitive_budget = QKPrimitiveBudget(qk_cfg.max_storage_bytes, qk_cfg.generated_policy_strict)
      q4_storage_mode, q6_storage_mode = qk_cfg.storage_mode, qk_cfg.q6_storage_mode
      if use_q4k_primitive: primitive_linears += _install_q4k_primitives(model, pathlib.Path(gguf), q4k_meta, None,
        primitive_budget, q4_storage_mode, route_plan, device_facts=_device_facts)
      if use_q6k_primitive: primitive_linears += _install_q6k_primitives(model, pathlib.Path(gguf), q4k_meta, None,
        primitive_budget, q6_storage_mode, route_plan, device_facts=_device_facts)
      # M2b/M2c linear copies of the resolve-once flag: the Q4K/Q6K primitive replacement above is
      # what carries route_role on the ffn_down linears, so the promoted flag is installed HERE on
      # the replacement objects (the model/block copies were set in the resolve-once block above).
      if model._decode_ffn_down_resadd_promoted:
        for _b in model.blk:
          _ffn = getattr(_b, "ffn_down", None)
          if _ffn is not None and isinstance(_ffn, (Q4KPrimitiveLinear, Q6KPrimitiveLinear)) and getattr(_ffn, "route_role", "") == "ffn_down":
            _ffn._decode_ffn_down_resadd_promoted = True
      if model._decode_q4k_gate_up_four_warp_vector_promoted:
        from tinygrad.llm.q4k_gate_up_four_warp_mmvq import Q4KGateUpFourWarpAdmission
        for _idx, _b in enumerate(model.blk):
          _fg, _fu = getattr(_b, "ffn_gate", None), getattr(_b, "ffn_up", None)
          if isinstance(_fg, Q4KPrimitiveLinear) and isinstance(_fu, Q4KPrimitiveLinear) and \
             (getattr(_fg, "out_features", None), getattr(_fg, "in_features", None)) == (12288, 4096) and \
             (getattr(_fu, "out_features", None), getattr(_fu, "in_features", None)) == (12288, 4096):
            _fg._q4k_gate_up_four_warp_admission = Q4KGateUpFourWarpAdmission(_idx, vector_loads=True)
      # Q4_K FFN-down four-warp fp16 geometry route: install the admission on the exact Q4_K
      # 4096x12288 ffn_down role only (Q6 down keeps its own route). The vector-load spelling is
      # bit-exact and passed production d512/d128 reverse brackets; TINYGRAD_Q4K_SCALAR_LOAD=1
      # restores the scalar spelling along with the ordinary Q4 single-projection routes.
      if model._decode_q4k_ffn_down_fp16_geometry_promoted:
        from tinygrad.llm.q4k_ffn_down_mmvq import Q4KFFNDownMMVQAdmission
        for _idx, _b in enumerate(model.blk):
          _ffn = getattr(_b, "ffn_down", None)
          if _ffn is not None and isinstance(_ffn, Q4KPrimitiveLinear) and getattr(_ffn, "route_role", "") == "ffn_down" \
             and getattr(_ffn, "out_features", None) == 4096 and getattr(_ffn, "in_features", None) == 12288:
            _ffn._q4k_ffn_down_mmvq_admission = Q4KFFNDownMMVQAdmission(_idx, fp16_fma=True,
              vector_loads=_q4k_single_projection_load_style(_ffn) == "vector")
      # Q6_K FFN-down four-warp fp16 geometry route: install the admission on the exact Q6_K
      # 4096x12288 ffn_down role only (Q4 down keeps its own route). Normal loads carry no
      # admission, so this stays closed on every other target.
      if model._decode_q6k_ffn_down_fp16_geometry_promoted:
        from tinygrad.llm.q6k_ffn_down_mmvq import Q6KFFNDownMMVQAdmission
        for _idx, _b in enumerate(model.blk):
          _ffn = getattr(_b, "ffn_down", None)
          if _ffn is not None and isinstance(_ffn, Q6KPrimitiveLinear) and getattr(_ffn, "route_role", "") == "ffn_down" \
             and getattr(_ffn, "out_features", None) == 4096 and getattr(_ffn, "in_features", None) == 12288:
            _ffn._q6k_ffn_down_mmvq_admission = Q6KFFNDownMMVQAdmission(_idx, fp16_fma=True,
              packed_lanemap=model._decode_q6k_ffn_down_packed_lanemap_promoted,
              unroll_blocks=4 if model._decode_q6k_ffn_down_unroll_promoted else None)
      # Q6_K attention-V four-warp fp16 geometry route: install the admission on the exact Q6_K
      # 1024x4096 attn_kv role only (Q4 V keeps its own shared-Q8 route). Normal loads carry no
      # admission, so this stays closed on every other target.
      if model._decode_q6k_v_four_warp_promoted:
        from tinygrad.llm.q6k_v_mmvq import Q6KVFourWarpAdmission
        for _idx, _b in enumerate(model.blk):
          _v = getattr(_b, "attn_v", None)
          if _v is not None and isinstance(_v, Q6KPrimitiveLinear) and getattr(_v, "route_role", "") == "attn_kv" \
             and getattr(_v, "out_features", None) == 1024 and getattr(_v, "in_features", None) == 4096:
            _v._q6k_v_four_warp_admission = Q6KVFourWarpAdmission(_idx)
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
      if primitive_linears: model._q4k_linears = Q4KPrimitiveRegistry(primitive_linears)
      primitive_ids = {id(linear) for linear in primitive_linears}
      generic_facts = []
      for fact in model_facts.tensors:
        try: module = _module_at(model, fact.module_path)
        except (AttributeError, IndexError, ValueError): continue
        if id(module) not in primitive_ids: generic_facts.append(fact)
      model._program_identity_raw_fact_names = bind_gguf_program_tensor_facts(q4k_meta, tuple(generic_facts))
      # Attach immutable source facts after all replacements: generic and packed
      # modules retain their original paths/state ownership.
      attachments = attach_program_identity_metadata(model, model_facts.tensors, primitive_linears=primitive_linears, module_at=_module_at)
      model._program_identity_linears = [linear for _path, linear in attachments]
      # L1 GEMV substrate lease install: when the record promotes NV sm_120, install the
      # booked max18 cooperative lease (blocks 1-12, 14-18, and 25; block 0 embedding boundary and
      # block 13 precision boundary stay ordinary) exactly like the
      # qualification harness does. The admission is a trace-time opt-in:
      # shared_q8_attention_call revalidates the real Q4/Q4/{Q4,Q6} tuple and REDUCE_OUTPUT
      # norm marker, returning None to the ordinary primitives on any miss.
      if model._decode_shared_q8_attention_promoted:
        _SHARED_Q8_LEASE = tuple(range(1, 13)) + tuple(range(14, 19)) + (25,)
        for _idx in _SHARED_Q8_LEASE:
          if _idx >= len(model.blk): break
          _b = model.blk[_idx]
          _norm = getattr(_b, "attn_norm", None)
          if _norm is None or getattr(_norm, "weight", None) is None: continue
          _b._shared_q8_attention_admission = SharedQ8AttentionAdmission(_idx, cooperative_q4=True,
            q4_direct_output=model._decode_q4_direct_shared_q8_attention_promoted,
            q6_direct_output=model._decode_q6_direct_shared_q8_attention_promoted,
            q4_kv_pair_output=model._decode_shared_q8_q4kv_pair_promoted and not model._decode_shared_q8_q4q4_qkv_full_promoted and
              isinstance(getattr(_b,"attn_k",None),Q4KPrimitiveLinear) and
              isinstance(getattr(_b,"attn_v",None),Q4KPrimitiveLinear),
            q4_q6_kv_pair_output=model._decode_shared_q8_q4q6_kv_pair_promoted and
              isinstance(getattr(_b,"attn_k",None),Q4KPrimitiveLinear) and
              isinstance(getattr(_b,"attn_v",None),Q6KPrimitiveLinear),
            q4_qkv_triple_output=model._decode_shared_q8_q4q4_qkv_full_promoted and
              isinstance(getattr(_b,"attn_k",None),Q4KPrimitiveLinear) and
              isinstance(getattr(_b,"attn_v",None),Q4KPrimitiveLinear))
          if isinstance(getattr(_b,"attn_k",None),Q4KPrimitiveLinear) and isinstance(getattr(_b,"attn_v",None),Q4KPrimitiveLinear):
            _b.attn_q._shared_q8_qkv_words=_b.attn_k.q4k_storage.words.cat(
              _b.attn_v.q4k_storage.words,dim=0).contiguous().realize()
          _b._decode_reduce_output_attn_rmsnorm_promoted = True
          try:
            _b._shared_q8_attention_norm_weight = Tensor.empty(4096, dtype=dtypes.float16,
              device=_norm.weight.device).assign(_norm.weight.cast(dtypes.float16).contiguous()).realize()
          except Exception:
            # A failed norm copy declines the fused provider for that block (legacy graph).
            _b._shared_q8_attention_norm_weight = None
      # Fuse only the nine ordinary Q4/Q4 K/V pairs proved by the composed
      # profile/wall gate. Shared-Q8 blocks have a distinct producer grammar
      # and are intentionally excluded here; mixed Q4/Q6 pairs also miss.
      if model._decode_q4k_kv_pair_promoted or model._decode_q4k_q4q4_qkv_full_promoted:
        from tinygrad.llm.q4k_kv_pair import Q4KKVPairAdmission, Q4KQKVAdmission
        for _idx, _b in enumerate(model.blk):
          if getattr(_b, "_shared_q8_attention_admission", None) is not None: continue
          _k, _v = getattr(_b, "attn_k", None), getattr(_b, "attn_v", None)
          if isinstance(_k, Q4KPrimitiveLinear) and isinstance(_v, Q4KPrimitiveLinear) and \
             all(getattr(x, "route_role", "") == "attn_kv" and getattr(x, "out_features", None) == 1024 and
                 getattr(x, "in_features", None) == 4096 for x in (_k, _v)):
            _b.attn_q._q4k_qkv_words=_k.q4k_storage.words.cat(_v.q4k_storage.words,dim=0).contiguous().realize()
            if model._decode_q4k_q4q4_qkv_full_promoted: _b._q4k_qkv_admission=Q4KQKVAdmission(_idx)
            elif model._decode_q4k_kv_pair_promoted: _b._q4k_kv_pair_admission = Q4KKVPairAdmission(_idx)
    if _runtime_inventory is not None:
      attach_selected_prefill_inventory(model, _runtime_inventory, _runtime_policy, _device_facts,
                                        direct_packed_policy=direct_packed_prefill_policy(config.n_heads, config.n_kv_heads))
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    # packed-wmma prefill warmstart (default ON, see prefill_routes.packed_wmma_prefill_enabled()): built here,
    # not in __init__, because it needs the just-installed Q4_K/Q6_K packed storage on model.blk[*] linears
    # to identify which covered linears are packed-weight candidates at all.
    #
    # ORDER IS LOAD-BEARING: this MUST run BEFORE realize_prefill_v2_weights(). The warmstart runs the
    # correctness canary, which SPAWNS a child process (packed_wmma_correctness_canary.run_canary ->
    # run_isolated_guarded_execution). realize_prefill_v2_weights() materializes a full fp16 overlay
    # (roughly 19GB for the measured larger profile), so running the canary after it leaves the child with almost no VRAM: the child dies
    # on its own admission ("budget 0.0GB, KV admits 0") and, in flat entry scripts, dies before draining
    # the spawn pipe so the parent blocks in Process.start() ahead of run_isolated's timeout loop. That is
    # the "packed-WMMA HW fault / >240s hang" in docs/BOLTBEAM_GPU_HANG_DIAGNOSIS_HANDOFF_20260724.md.
    # The kernel itself is NOT implicated: its code objects are byte-identical to the 6/6-gated state at
    # c35b5ff53 (bisected compile-only, docs/packed-wmma-14b-codegen-transition-bisect-20260724.md), and
    # this path only began executing on default loads when 6ca798568 flipped
    # TINYGRAD_PREFILL_PACKED_WMMA 0 -> 1 -- the gates that justified that promotion were captured in the
    # opt-in configuration, which never realized the overlay first.
    if config.prefill_v2:
      from tinygrad.llm.prefill_routes import packed_wmma_prefill_enabled
      if packed_wmma_prefill_enabled():
        model._packed_wmma_warmstart, model._packed_wmma_warmstart_contexts = model._build_packed_wmma_warmstart()
    # prefill v2 (opt-in): realize fp16 weights now that primitives are installed (shapes/dequant graphs ready)
    if config.prefill_v2:
      model.realize_prefill_v2_weights()
      # Dense schedules belong only to linears that actually received a resident fp16 overlay.
      # Build after memory-plan realization so packed-only models cannot match unrelated kernels
      # through an empty-dtype shape key.
      model._pf16_warmstart = model._build_prefill_v2_warmstart()
    # Concrete-KV is now the default prefill-v2 execution mode (see prefill_concrete_kv_auto_decision),
    # so per-start_pos jits compile LAZILY on first use by default (cached on the model instance
    # thereafter, model.py's `prefill_v2_jits.setdefault` at __call__) -- a cold prompt pays the
    # ~5s/new-chunk-offset tax inline once, not ceil(max_context/ubatch)*~5s at every load. Only callers
    # who explicitly declare workload reuse (prefill_workload_reuse, still off by default -- nothing
    # currently sets it) pay that bounded precompile-at-load cost up front so every generation is warm.
    if config.prefill_v2 and config.prefill_concrete_kv and config.prefill_workload_reuse:
      model.precompile_concrete_prefill_jits()
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

  def _prewarm_active_horizon_flash_pairs(self) -> None:
    """Capture the qualified S6/S8 greedy ping-pong pairs once per model.

    The selector crosses from S6 to S8 at Tc=769. Letting either pair capture
    at that crossing converts compilation into a token-latency spike, so the
    booked throughput policy explicitly moves both captures to request warmup.
    The dummy KV writes are overwritten by the following prompt.
    """
    if getattr(self, "_flash_decode_active_horizon_prewarmed", False): return
    if not getattr(self, "_flash_decode_active_horizon_lease", False) or self.max_context != 1024: return
    if not getattr(self, "_decode_direct_greedy_promoted", False) or \
       not getattr(self, "_decode_feedback_pingpong_promoted", False): return
    v_sp = UOp.variable("start_pos", 0, self.max_context-1)
    dummy = Tensor([[0]], dtype="int32").contiguous()
    temp = Tensor([0.0]).contiguous()
    for split_count, ctx in ((6, 700), (None, 800)):
      for slot in (0, 1):
        for _ in range(3):
          self(dummy, v_sp.bind(ctx), temp, use_flash=True, greedy=True,
               feedback_slot=slot, flash_split_count=split_count).realize()
    self.reset_generation_state()
    self._flash_decode_active_horizon_prewarmed = True

  def _decode_submit_ahead_eligible(self) -> bool:
    """Closed-default launch-hiding gate for the steady-decode pipeline.

    The submit-ahead route reorders the steady decode loop to submit token
    N+1's graph BEFORE ``item()`` on token N, so the host-side JIT prep for
    N+1 overlaps the GPU execution of N.  It is only safe when the promoted
    greedy pingpong pair is already captured AND its alias contract is
    admitted (distinct fixed returns, read-only inputs): the loop then reads
    the previous output tensor after submitting the next graph without any
    risk of the next graph clobbering it.  A cold or unadmitted pair falls
    back to the ordinary single-sync pingpong route, which is byte-identical.
    """
    if not bool(getattr(self, "_decode_submit_ahead_promoted", False)): return False
    if not bool(getattr(self, "_decode_direct_greedy_promoted", False)): return False
    if not bool(getattr(self, "_decode_feedback_pingpong_promoted", False)): return False
    from tinygrad.llm.feedback_pingpong import pingpong_capture_contract
    pair = self.rollout_greedy_pingpong_jits_flash
    return all(getattr(jit, "captured", None) is not None for jit in pair) and \
      pingpong_capture_contract(pair)["admitted"]

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0, diagnostic_full_logits:bool=False,
               expected_output_tokens:int|None=None, delivery_batch:int=1):
    if not isinstance(delivery_batch, int) or delivery_batch < 1:
      raise ValueError(f"delivery_batch must be a positive integer, got {delivery_batch!r}")
    if delivery_batch > 1 and (temperature != 0.0 or diagnostic_full_logits or expected_output_tokens is None or expected_output_tokens < 1):
      raise ValueError("delivery_batch > 1 requires greedy generation and a positive expected_output_tokens")
    if self.has_recurrent_block: chunk_size = 1
    _ring = self.config.ring and self.config.rope_dim == self.config.head_dim
    if _ring and len(tokens) > self.max_context:
      # StreamingLLM evicts DURING generation, not prefill: a prompt larger than the physical window N can't be held.
      raise RuntimeError(f"prompt is {len(tokens)} tokens but the streaming window is N={self.max_context}: streaming "
                         f"evicts during generation, not prefill. Shorten the prompt to <={self.max_context} tokens, or "
                         f"use a model/quant that admits a larger window.")
    if not _ring and not diagnostic_full_logits and temperature == 0.0:
      self._prewarm_active_horizon_flash_pairs()
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
    out, prompt_len, decode_feedback_phase = None, len(tokens), 0
    request_flash_split = _request_static_flash_split_count(prompt_len, expected_output_tokens, self.max_context)
    direct_greedy = temperature == 0.0 and bool(getattr(self, "_decode_direct_greedy_promoted", False))
    host_argmax_mirror = direct_greedy and not diagnostic_full_logits and \
      bool(getattr(self, "_decode_native_argmax_threads", 0)) and not bool(getattr(self, "_decode_vocab_top1_lease", False)) and \
      bool(getenv("NV_ARGMAX_HOST_MIRROR", 0))
    if host_argmax_mirror and getattr(self, "_decode_host_argmax_mirror", None) is None:
      self._decode_host_argmax_mirror = make_native_argmax_host_mirror("NV", getenv("NV_ARGMAX_HOST_MIRROR_MEMORY", "host"))
    batched_delivery = delivery_batch > 1
    # The promoted native argmax returns a held clone specifically so its GPU
    # feedback value survives replay.  Batching retains that same return before
    # the next replay; it does not require the optional two-capture ping-pong
    # route (the exact batch-ring gate qualified the ordinary capture).
    if batched_delivery and (_ring or not bool(getattr(self, "_decode_native_argmax_threads", 0))):
      raise ValueError("delivery_batch > 1 requires non-ring greedy native-argmax decode")
    delivery_ring = None
    delivery_pending:list[int] = []
    delivery_generated = 0
    if batched_delivery:
      if not str(Device.DEFAULT).startswith("NV"): raise ValueError("delivery_batch > 1 is currently NV-only")
      from tinygrad.device import Buffer, BufferSpec
      # Buffer owns and releases the fixed ring when the generator closes.
      delivery_ring_owner = Buffer("NV", delivery_batch, dtypes.int32, options=BufferSpec(nolru=True), preallocate=True)
      delivery_ring = delivery_ring_owner._buf
    # Launch hiding (submit-ahead) only reorders the steady flash-greedy
    # pingpong decode; everything else keeps the exact existing scheduling.
    submit_ahead = (not _ring and not diagnostic_full_logits and temperature == 0.0 and direct_greedy
                    and self._decode_submit_ahead_eligible())
    read_tok = None
    while _ring or len(tokens) < self.max_context:   # ring: unbounded logical context (caller controls when to stop)
      ubatch = self.config.prefill_ubatch
      if self.config.prefill_v2 and prefill_v2_target_admitted(self.config.prefill_device_facts) and (prompt_len - start_pos) >= ubatch:
        # prefill v2: a CONCRETE-T chunk of all-real prompt tokens (start_pos still symbolic; only the token
        # dim must be concrete for tensor cores). remaining>=UBATCH => start_pos<prompt_len so we slice from t.
        # concrete start_pos -> KV=start_pos+T concrete -> attention TC fires (the validated 1.24x, byte-identical).
        # start_pos==0 is always concrete (one jit, unconditionally). prefill_concrete_kv is now the
        # default for prefill-v2 (see prefill_concrete_kv_auto_decision) so CONTINUATION chunks (sp>0)
        # also take the concrete/TC-attn path instead of the slow symbolic SDPA fallback -- each
        # per-start_pos jit compiles lazily on first use and is cached on the model instance thereafter.
        use_concrete = (start_pos == 0) or self.config.prefill_concrete_kv
        sp, ntv = (start_pos if use_concrete else v_start_pos.bind(start_pos)), ubatch
        out = self(t[:, sp:sp+ubatch], sp, temp, use_flash=False, greedy=direct_greedy).realize()
      elif self.config.prefill_v2 and prefill_v2_target_admitted(self.config.prefill_device_facts) and start_pos < prompt_len and prompt_len >= ubatch:
        # Phase-3 fix: a sub-UBATCH PROMPT remainder would otherwise fall to many slow 32-token symbolic calls
        # (the fallback trap). Instead process the LAST PREFILL_UBATCH tokens as ONE prefill-v2 chunk by shifting
        # the window back so it ENDS exactly at prompt_len -> all-real tokens (no padding), last position is
        # prompt_len-1 so out.item() is the next token. Re-processes the small overlap with the prior chunk (same
        # tokens -> same KV) -> correct. Symbolic start_pos reuses the one prefill_v2_jit (no per-remainder compile).
        sp = v_start_pos.bind(prompt_len - ubatch)   # symbolic offset -> matches the prefill_v2_jit signature
        out = self(t[:, sp:sp+ubatch], sp, temp, use_flash=False, greedy=direct_greedy).realize()
        ntv = prompt_len - start_pos                      # advance straight to end of prompt
      elif _ring and start_pos >= prompt_len and out is not None:
        if diagnostic_full_logits: raise ValueError("diagnostic_full_logits does not support ring decode")
        # StreamingLLM ring decode (T=1, past the prompt). Bind the WRAPPED write slot (always in [0,N-1] -> never trips
        # Variable.bind's vmax assert even as the logical position grows unboundedly); feed the per-step pre-gathered
        # freqs (identity while filling -> token-identical; slot-relative once full); ring_full switches to the [0:N]
        # read graph at the wrap. Two graphs total (fill + full), captured once each -- no per-step recompile.
        _N, _sinks = self.max_context, 4
        sp = v_start_pos.bind(self._ring_slot(start_pos, _N, _sinks)); ntv = 1
        _rf = self._ring_gather_freqs(next(b.freqs_cis for b in self.blk if hasattr(b, "freqs_cis")),
                                      start_pos, _N, _sinks)
        out = self(out, sp, temp, use_flash=True, ring_freqs=_rf, ring_full=(start_pos >= _N), greedy=direct_greedy).realize()
      else:
        # Submit-ahead defers the prefill yield to the first steady-decode
        # iteration, so start_pos == len(tokens) there: the next graph is
        # still a one-token decode, never an empty prefill slice.
        toks_remaining = len(tokens) - start_pos
        if submit_ahead and toks_remaining < 1: toks_remaining = 1
        sp, nt = v_start_pos.bind(start_pos), v_toks.bind(min(chunk_size, toks_remaining))
        ntv = nt.val
        # Select the flash-decode graph (rollout_jit_flash) vs SDPA graph (rollout_jit) per-token by context, so a
        # generation that STARTS short still crosses over to flash once ctx reaches the threshold. Without this the
        # decode graph is baked SDPA at the start ctx and never switches -> short-prompt decode SDPA-degrades the
        # whole way (e.g. 85->54 tok/s by ctx512). should_use_flash_decode returns False for ntv!=1 (prefill chunks).
        _uf = self.config.flash_decode and _route_should_use_flash_decode(sp, ntv)
        _flash_split = request_flash_split if _uf else None
        if _flash_split is None:
          _flash_split = _adaptive_flash_split_count(_uf and getattr(self,"_flash_decode_adaptive_s64_lease",False),start_pos,self.max_context)
        if _flash_split is None:
          _flash_split = _active_horizon_flash_split_count(
            _uf and getattr(self,"_flash_decode_active_horizon_lease",False) and
            getattr(self,"_flash_decode_active_horizon_prewarmed",False),start_pos,self.max_context)
        if submit_ahead and ntv == 1 and start_pos >= prompt_len and _uf and out is not None:
          # Submit token start_pos+1's graph now; its input is the previous
          # output (the token we yield this iteration).  The host-side prep of
          # this submission overlaps the GPU execution of that token, then the
          # bottom of the loop reads the PREVIOUS output after submission.
          prev = out
          out = self(prev, sp, temp, use_flash=_uf, greedy=True,
                     feedback_slot=(decode_feedback_phase & 1),flash_split_count=_flash_split).realize()
          decode_feedback_phase += 1
          read_tok = prev
        else:
          decode_input = _generation_input_slice(t, sp, nt, ntv) if start_pos < prompt_len or out is None else \
                         (out[0] if diagnostic_full_logits and isinstance(out, tuple) else out)
          feedback_slot = (decode_feedback_phase & 1) if start_pos >= prompt_len and \
            bool(getattr(self, "_decode_feedback_pingpong_promoted", False)) else None
          out = (self.decode_with_logits(decode_input, sp, temp, use_flash=_uf, feedback_slot=feedback_slot,flash_split_count=_flash_split)
                 if diagnostic_full_logits and ntv == 1 and start_pos >= prompt_len else
                 self(decode_input, sp, temp, use_flash=_uf, greedy=direct_greedy, feedback_slot=feedback_slot,flash_split_count=_flash_split))
          if feedback_slot is not None: decode_feedback_phase += 1
          if isinstance(out, tuple):
            out = (out[0].realize(), out[1].realize())
          else: out = out.realize()
          if feedback_slot is not None:
            pair = self.rollout_greedy_logits_pingpong_jits_flash if diagnostic_full_logits and _uf else \
                   self.rollout_greedy_logits_pingpong_jits if diagnostic_full_logits else \
                   self.rollout_greedy_pingpong_jits_flash if _uf else self.rollout_greedy_pingpong_jits
            if all(getattr(jit, "captured", None) is not None for jit in pair):
              from tinygrad.llm.feedback_pingpong import pingpong_capture_contract
              if not pingpong_capture_contract(pair)["admitted"]:
                # Keep the generic alias-safe path as the automatic fallback;
                # the harness records the exact failed contract separately.
                self._decode_feedback_pingpong_promoted = False
                decode_feedback_phase = 0
      start_pos += ntv
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      if submit_ahead and read_tok is None:
        # The prefill output is the first deferred token: the steady-decode
        # iteration above submits the NEXT graph and yields this output.
        read_tok = out
        continue
      if submit_ahead:
        sampled, logits = read_tok, None
      else:
        sampled, logits = out if diagnostic_full_logits and isinstance(out, tuple) else (out, None)
      if batched_delivery:
        assert delivery_ring is not None and isinstance(sampled.device, str)
        dev = Device[sampled.device]
        src = sampled.uop.buffer.get_buf(sampled.device)
        slot = len(delivery_pending)
        dev.hw_copy_queue_t().wait(dev.timeline_signal, dev.timeline_value-1).copy(delivery_ring.offset(slot*4, 4), src, 4) \
          .signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
        delivery_pending.append(len(tokens)); tokens.append(0); delivery_generated += 1
        flush = len(delivery_pending) == delivery_batch or delivery_generated == expected_output_tokens or \
          (not _ring and len(tokens) >= self.max_context)
        if not flush: continue
        raw = bytearray(len(delivery_pending)*4)
        dev.allocator._copyout(memoryview(raw), delivery_ring.offset(0, len(raw)))
        delivered = [int.from_bytes(raw[i*4:(i+1)*4], "little", signed=True) for i in range(len(delivery_pending))]
        for idx, token in zip(delivery_pending, delivered): tokens[idx] = token
        delivery_pending.clear()
        self._cached_tokens = tokens[:-1]
        for token in delivered: yield token
        continue
      if host_argmax_mirror:
        Device[sampled.device].synchronize()
        tokens.append(read_native_argmax_host_mirror(self._decode_host_argmax_mirror))
      else:
        tokens.append(int(sampled.item()))
      self._cached_tokens = tokens[:-1]
      # Prompt/prefill produces the first sampled token through the normal
      # prefill graph, so its diagnostic logit is explicitly ``None`` rather
      # than a silently different full-sequence route.
      yield (tokens[-1], logits) if diagnostic_full_logits else tokens[-1]
