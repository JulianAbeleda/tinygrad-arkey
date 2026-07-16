from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Mapping
from tinygrad.llm.device_facts import DeviceFacts
from tinygrad.llm.gguf_memory_scan import CandidateWorkspace, GGUFMemoryScan, RuntimeGeometry, scan_selected_gguf_memory
from tinygrad.llm.prefill_memory_plan import (ByteLifetime, ByteTerm, CandidateMemoryCoverage, DeviceMemoryFacts,
                                              PrefillMemoryPlan, Strategy, plan_prefill_memory)
from tinygrad.llm.memory_ledger import (AllocationProvenance, ExactMemoryDecision, ScannedMemoryBudget,
                                        SelectedModelMemoryLedger, exact_memory_decisions)

AUTO_MAX_CONTEXT = "auto"

def scanned_device_memory_budget(facts:DeviceFacts) -> ScannedMemoryBudget:
  """Live free VRAM minus an observed, allocator-aligned disturbance reserve."""
  total, free = facts.total_vram_bytes, facts.free_vram_bytes
  alignment = facts.capabilities.global_allocation_granularity
  reserve = None
  complete = (facts.memory_probe.state == "ok" and isinstance(total, int) and isinstance(free, int) and
              isinstance(alignment, int) and alignment > 0 and 0 <= free <= total)
  if complete:
    occupied = max(0, total-free)
    reserve = ((occupied + alignment - 1) // alignment) * alignment
  return ScannedMemoryBudget(free if complete else None, reserve, AllocationProvenance(facts.memory_probe.source,
    "live free VRAM minus live occupied-byte disturbance reserve, rounded to scanned allocator granularity"))

def admit_exact_selected_model(ledger:SelectedModelMemoryLedger, budget:ScannedMemoryBudget,
                               candidates:tuple[str, ...]|None=None) -> tuple[ExactMemoryDecision, ...]:
  """Audit exact peak allocation for each candidate; unknown or omitted byte classes fail closed."""
  candidate_ids = ledger.candidates() if candidates is None else candidates
  if not candidate_ids: raise ValueError("at least one candidate_id is required")
  return exact_memory_decisions(ledger, budget, candidate_ids)

@dataclass(frozen=True)
class ExactSelectedModelPlan:
  """Immutable allocation authority retained by the loaded model."""
  scan: GGUFMemoryScan
  decision: ExactMemoryDecision

  def to_dict(self) -> dict[str, Any]:
    return {"model_path": str(self.scan.model_path), "tensor_spans": [x.__dict__ for x in self.scan.tensor_spans],
            "decision": self.decision.to_dict()}

def plan_exact_selected_model_load(model_path, *, metadata:tuple[dict, dict], geometry:RuntimeGeometry,
                                   route_memory_facts:Mapping[str, Any], facts:DeviceFacts) -> ExactSelectedModelPlan:
  """Build and decide the one selected route's ledger without reopening the GGUF.

  Every value here is an allocation fact.  Missing alignment, storage/copy, runtime, output, scratch, or workspace
  evidence remains ``None`` in the report and therefore refuses admission.
  """
  candidate_id = route_memory_facts.get("candidate_id")
  if not isinstance(candidate_id, str) or not candidate_id: raise ValueError("route memory facts require candidate_id")
  copies = route_memory_facts.get("resident_copies")
  workspace = route_memory_facts.get("candidate_workspace_bytes")
  detail = route_memory_facts.get("provenance")
  if not isinstance(detail, str) or not detail: detail = "route did not provide memory-fact provenance"
  alignment = facts.capabilities.global_allocation_granularity
  scan = scan_selected_gguf_memory(model_path, geometry, (CandidateWorkspace(candidate_id, workspace, detail),),
    allocation_alignment=alignment, resident_copies=copies, metadata=metadata)
  budget = scanned_device_memory_budget(facts)
  decision = admit_exact_selected_model(scan.ledger, budget, (candidate_id,))[0]
  return ExactSelectedModelPlan(scan, decision)

@dataclass(frozen=True)
class AdmissionInputs:
  requested:int|str|None; trained_ctx:int; free_vram:int|None; q4_bytes:int; est_fp16:int; num_blocks:int
  n_heads:int; n_kv_heads:int; head_dim:int; prefill_ubatch:int; v2_on:bool; resident_fp16_admit:bool
  model_label:str; stream:str="auto"; rope_dim:int|None=None; kv_quant_supported:bool=False
  kv_quant_disabled:bool=False; live_split_s:int=48

  @staticmethod
  def from_model_metadata(requested:int|str|None, kv:Mapping[str, Any], *, free_vram:int|None, q4_bytes:int,
                          est_fp16:int, prefill_ubatch:int, v2_on:bool, resident_fp16_admit:bool,
                          model_label:str|None=None, stream:str="auto", kv_quant_supported:bool=False,
                          kv_quant_disabled:bool=False, live_split_s:int=48) -> "AdmissionInputs":
    """Single owner for translating GGUF model metadata into context-admission geometry."""
    arch = kv["general.architecture"]
    n_heads, n_kv_heads = kv[f"{arch}.attention.head_count"], kv[f"{arch}.attention.head_count_kv"]
    head_dim = kv.get(f"{arch}.attention.key_length_mla",
                      kv.get(f"{arch}.attention.key_length", kv[f"{arch}.embedding_length"] // n_heads))
    return AdmissionInputs(requested, kv[f"{arch}.context_length"], free_vram, q4_bytes, est_fp16,
      kv[f"{arch}.block_count"] - kv.get(f"{arch}.nextn_predict_layers", 0), n_heads, n_kv_heads, head_dim, prefill_ubatch,
      v2_on, resident_fp16_admit, model_label or f"{arch} selected model", stream, kv.get(f"{arch}.rope.dimension_count", head_dim),
      kv_quant_supported, kv_quant_disabled, live_split_s)

@dataclass(frozen=True)
class ContextMemoryTerms:
  weights:int; kv_per_tok:int; prefill_per_tok:int; flash_scratch:int; kv_scale_per_tok:int

  @staticmethod
  def from_inputs(inp:AdmissionInputs, *, resident_fp16:bool) -> "ContextMemoryTerms":
    return ContextMemoryTerms(weights=inp.q4_bytes + (inp.est_fp16 if resident_fp16 else 0),
      kv_per_tok=2 * inp.n_kv_heads * inp.head_dim * 2 * inp.num_blocks, prefill_per_tok=4 * inp.n_heads * inp.prefill_ubatch,
      flash_scratch=inp.n_heads * inp.live_split_s * (inp.head_dim + 2) * 4, kv_scale_per_tok=2 * inp.n_kv_heads * 2 * inp.num_blocks)

@dataclass(frozen=True)
class AdmissionPlan:
  max_context:int; kv_quant:bool; report:dict; weights:int; kv_per_tok:int; prefill_per_tok:int
  prefill_memory_plan:str|None=None

@dataclass(frozen=True)
class _ContextCandidate:
  """Structural context-storage option; labels are audit output, never memory authority."""
  candidate_id:str; kv_storage:str; bytes_per_token:int; exact_context:bool; ring_buffer:bool

  def to_dict(self, capacity:int, supported:bool=True) -> dict[str, Any]:
    return {"candidate_id": self.candidate_id, "kv_storage": self.kv_storage,
            "bytes_per_token": self.bytes_per_token, "exact_context": self.exact_context,
            "ring_buffer": self.ring_buffer, "supported": supported, "capacity_tokens": max(capacity, 0)}

def _plan_context_admission(inp:AdmissionInputs, budget:ScannedMemoryBudget, terms:ContextMemoryTerms) -> AdmissionPlan:
  kv_quant_shape = inp.head_dim == 128 and inp.n_kv_heads == 8 and inp.n_heads % inp.n_kv_heads == 0; kv_quant_supported = inp.kv_quant_supported and kv_quant_shape and not inp.kv_quant_disabled
  ring_supported = kv_quant_shape and (inp.rope_dim if inp.rope_dim is not None else inp.head_dim) == inp.head_dim
  scale_per_tok = terms.kv_scale_per_tok if kv_quant_supported else 0
  max_context, kv_quant, report = _resolve_max_context_admission(
    inp.requested, inp.trained_ctx, budget, terms.weights, terms.kv_per_tok, terms.prefill_per_tok, terms.flash_scratch, inp.model_label,
    kv_quant_supported=kv_quant_supported, scale_per_tok=scale_per_tok, stream=inp.stream, ring_supported=ring_supported,
  )
  return AdmissionPlan(max_context, kv_quant, report, terms.weights, terms.kv_per_tok, terms.prefill_per_tok)

def _resolve_max_context_admission(requested, trained_ctx:int, scanned_budget:ScannedMemoryBudget, weights_bytes:int,
                                  kv_per_tok:int, prefill_per_tok:int, flash_scratch_bytes:int,
                                  model_label:str, kv_quant_supported:bool=False, scale_per_tok:int=0,
                                  stream:str="auto", ring_supported:bool=False) -> tuple[int, bool, dict]:
  """Resolve exact context, or an auto-only lossy ring, from scanned capacity."""
  is_auto = requested is None or requested == AUTO_MAX_CONTEXT
  ring_ok = ring_supported and stream != "off"
  free_bytes, budget = scanned_budget.free_bytes, scanned_budget.admitted_bytes
  if free_bytes is None or budget is None:
    raise RuntimeError(f"{model_label}: context admission requires scanned total/free VRAM and allocator granularity")
  budget_policy = "device-facts scan"
  fixed = weights_bytes + flash_scratch_bytes
  kv_q8_per_tok = kv_per_tok // 2 + scale_per_tok
  def _mc(kv_pt:int) -> int:
    pt = kv_pt + prefill_per_tok
    return int((budget - fixed) // pt) if pt > 0 else trained_ctx

  mc_fp16 = _mc(kv_per_tok)
  mc_q8 = _mc(kv_q8_per_tok) if kv_quant_supported else -1
  ring_N = min(mc_fp16, trained_ctx) if is_auto else min(int(requested), mc_fp16, trained_ctx)  # fp16 window, in-distribution
  fp16_candidate = _ContextCandidate("kv-fp16-exact", "fp16", kv_per_tok, True, False)
  q8_candidate = _ContextCandidate("kv-q8-exact", "q8", kv_q8_per_tok, True, False)
  ring_candidate = _ContextCandidate("kv-fp16-ring", "fp16", kv_per_tok, False, True)
  target = trained_ctx if is_auto else int(requested)
  report = {"free_gb": free_bytes/1e9, "budget_gb": budget/1e9, "reserve_gb": (free_bytes-budget)/1e9,
            "budget_policy": budget_policy, "weights_gb": weights_bytes/1e9,
            "flash_scratch_gb": flash_scratch_bytes/1e9, "kv_gb_per_1k": kv_per_tok*1000/1e9,
            "prefill_gb_per_1k": prefill_per_tok*1000/1e9, "mc_fp16": max(mc_fp16, 0),
            "mc_q8": max(mc_q8, 0), "mc_ring": max(ring_N, 0), "trained_ctx": trained_ctx, "ring": False,
            "context_candidates": [fp16_candidate.to_dict(mc_fp16), q8_candidate.to_dict(mc_q8, kv_quant_supported),
                                   ring_candidate.to_dict(ring_N, ring_supported)]}

  def _ring_banner(N:int) -> str:
    return (f"STREAMING MODE: fp16 KV window N={N} -- generation is unbounded, but content older than the last ~{N} "
            f"tokens (plus 4 sinks) is evicted and forgotten (StreamingLLM). --stream=off to disable.")

  # stream='on': deliberately unbounded generation -- use the ring even when an exact candidate fits.
  if stream == "on":
    if not ring_supported:
      raise RuntimeError(f"{model_label}: --stream=on requires the ring-capable decode route (structural shape class).")
    if ring_N <= 0:
      raise RuntimeError(f"{model_label}: --stream=on has no positive fp16 window (only {max(ring_N,0)} tokens fit).")
    report.update(mode="ring", kv_quant=False, ring=True, max_context=ring_N, mc_mem=max(ring_N, 0), banner=_ring_banner(ring_N))
    return ring_N, False, report

  # Candidate: fp16 exact.
  if mc_fp16 >= target:
    mc = min(target, mc_fp16, trained_ctx)
    report.update(mode=("auto" if is_auto else "explicit"), kv_quant=False, max_context=mc, mc_mem=max(mc_fp16, 0))
    return mc, False, report

  # Candidate: int8 Q8 (near-lossless).
  if kv_quant_supported and mc_q8 >= target:
    mc = min(target, mc_q8, trained_ctx)
    report.update(mode=("auto+q8" if is_auto else "explicit+q8"), kv_quant=True, max_context=mc,
                  kv_gb_per_1k=kv_q8_per_tok*1000/1e9, mc_mem=max(mc_q8, 0))
    return mc, True, report

  q8note = (f" int8 KV-quant admits {max(mc_q8,0)} (still < {target})." if kv_quant_supported
            else " KV-quant unsupported for this shape.")
  # Explicit --max_context is STRICT: the ring never rescues a requested int (N is physical; 8000 fp16 doesn't fit).
  if not is_auto:
    _hint = (f" For unbounded generation with a smaller lossy window, pass --stream (window would be N={max(ring_N,0)})."
             if ring_supported and ring_N > 0 else "")
    raise RuntimeError(
      f"{model_label}: requested --max_context {requested} needs {target} tokens; fp16 KV admits {max(mc_fp16,0)},{q8note} "
      f"(free {free_bytes/1e9:.1f}GB, weights {weights_bytes/1e9:.1f}GB, budget {budget/1e9:.1f}GB; {budget_policy})."
      f" Largest admissible is {max(max(mc_q8,0), max(mc_fp16,0))}. Reduce --max_context or free VRAM.{_hint}")

  # Candidate: RING -- lossy unbounded context, only when no exact candidate is usable.
  if ring_ok and ring_N > 0:
    report.update(mode="ring", kv_quant=False, ring=True, max_context=ring_N, mc_mem=max(ring_N, 0), banner=_ring_banner(ring_N))
    return ring_N, False, report

  # Final refusal: not even the streaming window is usable.
  _ringnote = (f" even the streaming window has no positive capacity." if ring_supported
               else " (streaming unsupported for this shape).")
  raise RuntimeError(
    f"{model_label}: auto-scan needs {target} tokens; fp16 KV admits {max(mc_fp16,0)},{q8note}{_ringnote} "
    f"(free {free_bytes/1e9:.1f}GB, weights {weights_bytes/1e9:.1f}GB, budget {budget/1e9:.1f}GB; {budget_policy}). "
    f"Refusing: free VRAM or use a smaller model. (Q8+ring composition would ~double the streaming window; not yet implemented.)")

def plan_selected_model_memory(inp:AdmissionInputs, facts:DeviceFacts, *, direct_packed_supported:bool,
                               overlay_requested:bool|None=None) -> tuple[AdmissionPlan, PrefillMemoryPlan, Strategy]:
  scanned_budget = scanned_device_memory_budget(facts)
  reserve = scanned_budget.reserve_bytes
  device = DeviceMemoryFacts(facts.total_vram_bytes, facts.free_vram_bytes,
    ByteTerm("runtime_safety_reserve", reserve, facts.memory_probe.source,
             "align_up(total_vram_bytes - free_vram_bytes, scanned_allocator_granularity)",
             ByteLifetime.SAFETY_RESERVE), facts.memory_probe.source)
  explicit_overlay = overlay_requested is True
  terms = ContextMemoryTerms.from_inputs(inp, resident_fp16=explicit_overlay)
  context = _plan_context_admission(replace(inp, free_vram=facts.free_vram_bytes, resident_fp16_admit=explicit_overlay), scanned_budget, terms)
  scale_per_tok = terms.kv_scale_per_tok if context.kv_quant else 0
  kv_bytes = (terms.kv_per_tok // (2 if context.kv_quant else 1) + scale_per_tok) * context.max_context
  base = (
    ByteTerm("packed_weights", inp.q4_bytes, "selected GGUF tensor/file inventory", "sum resident packed allocations", ByteLifetime.PERSISTENT),
    ByteTerm("kv_cache", kv_bytes, "transformer geometry and admitted context",
             "kv_bytes_per_token * admitted_context + scale_bytes", ByteLifetime.PERSISTENT),
    ByteTerm("prefill_activations", context.prefill_per_tok * context.max_context, "prefill workload geometry",
             "prefill_bytes_per_token * admitted_context", ByteLifetime.PREFILL_PEAK),
    ByteTerm("flash_scratch", terms.flash_scratch, "decode/prefill attention geometry",
             "n_heads * live_split_s * (head_dim + 2) * sizeof(float32)", ByteLifetime.PREFILL_PEAK),
  )
  candidates = [CandidateMemoryCoverage("full-resident-overlay", Strategy.FULL_RESIDENT_OVERLAY,
    (ByteTerm("dense_fp16_overlay", inp.est_fp16, "selected GGUF covered tensor inventory",
              "sum covered tensor elements * sizeof(float16)", ByteLifetime.PERSISTENT),),
    required_invocations=("prefill",), covered_invocations=("prefill",), supported=inp.v2_on)]
  candidates.append(CandidateMemoryCoverage("direct-packed-baseline", Strategy.DIRECT_PACKED_FALLBACK,
    required_invocations=("prefill",), covered_invocations=(("prefill",) if direct_packed_supported else ()),
    supported=direct_packed_supported, reasons=(() if direct_packed_supported else ("direct packed coverage is unavailable",))))
  override = Strategy.FULL_RESIDENT_OVERLAY if explicit_overlay else (Strategy.DIRECT_PACKED_FALLBACK if overlay_requested is False else None)
  memory_plan = plan_prefill_memory(device=device, base_terms=base, candidates=candidates, override=override)
  if memory_plan.decision is Strategy.REFUSE:
    raise RuntimeError(f"{inp.model_label}: memory plan refused load: {'; '.join(memory_plan.reasons)}")
  effective = (Strategy.FULL_RESIDENT_OVERLAY if memory_plan.decision is Strategy.FULL_RESIDENT_OVERLAY else
               Strategy.DIRECT_PACKED_FALLBACK)
  if effective not in memory_plan.feasible_strategies:
    effective = memory_plan.feasible_strategies[0]
  report = {**context.report, "prefill_memory_strategy": effective.value,
            "prefill_memory_feasible": [x.value for x in memory_plan.feasible_strategies],
            "prefill_memory_selection_deferred": memory_plan.decision is None}
  return AdmissionPlan(context.max_context, context.kv_quant, report, context.weights, context.kv_per_tok,
                       context.prefill_per_tok, memory_plan.to_json()), memory_plan, effective
