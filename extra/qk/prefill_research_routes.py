"""Explicit benchmark-only prefill routes; never imported by production routing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from tinygrad import Tensor, dtypes
from tinygrad.llm import route_ops as qk_ops
from extra.qk.cooperative_mmq_gate import admit_cooperative_mmq, canonical_candidate_identity
from extra.qk.mmq_exact_role_spec import DEFAULT_INVENTORY, ExactRoleSpec, exact_role_spec
from extra.qk.prefill.frozen_exact_role_runtime import run_frozen_exact_q4k_research
from extra.qk.prefill.six_row_research_selector import (
  GROUPS, RETAINED_POLICY_IDENTITY, TARGET, ExactSixRowResearchSelector, ResearchPolicyBlocked,
  ResearchSelection, ResearchWorkload,
)
from tinygrad.llm.memory_semantics import mark_candidate_workspace
from tinygrad.llm.prefill_routes import (PrefillLinearRouteSpec, _direct_packed_enabled_for, _direct_packed_module_role,
  _direct_packed_quant, _direct_packed_role, _is_q4k_linear, _is_q6k_linear, _run_direct_packed_baseline,
  notify_prefill_route, prefill_activation, prefill_output)
from tinygrad.llm.prefill_route_observer import (
  PrefillRouteAttachment, PrefillRouteExecution, notify_prefill_route_execution,
)
from tinygrad.uop.ops import UOp

Q4K_Q8_CHOICES = ("", "0", "false", "off", "no", "wmma", "wmma_tiled", "packed_ds4", "packed_row_major", "packed_fused")
_MMQ_DS4_LAST_PACKED: tuple[Any, tuple[Tensor, Tensor, Tensor]] | None = None

@dataclass(frozen=True)
class ExactResearchRouteAuthority:
  """All host-side authority required to opt into the immutable six-row route."""
  policy: Mapping[str, Any]
  target: Mapping[str, Any]
  frozen_bundles: Mapping[str, str | Path]
  fallback_program_identities: Mapping[str, str]
  inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY

@dataclass(frozen=True)
class PrefillResearchRouteConfig:
  q4k_q8_mode: str = ""
  q4k_q8_roles: frozenset[str] | None = None
  cooperative_candidate: Mapping[str, Any] | None = None
  cooperative_evidence: Mapping[str, Any] | None = None
  cooperative_enabled: bool = False
  cooperative_runner: Any | None = None
  generated_tile: bool = False
  wmma_n_tile: int = 256
  wmma_max_raw_elems: int = 64 * 1024 * 1024
  wmma_allow_graph_explosion: bool = False
  wmma_tiled_m_tile: int = 16
  wmma_tiled_n_tile: int = 16
  wmma_tiled_group_tile: int = 1
  exact_policy_enabled: bool = False
  exact_authority: ExactResearchRouteAuthority | None = None

@dataclass(frozen=True)
class DirectPackedPrefillRequest:
  quant: str
  role: str
  m: int
  n: int
  k: int
  bias: bool
  ubatch: int

  @property
  def route_facts(self) -> dict[str, Any]:
    return {"quant": self.quant, "role": self.role, "M": self.m, "N": self.n, "K": self.k,
            "bias": self.bias, "ubatch": self.ubatch}

def build_direct_packed_prefill_request(lin, x:Tensor|None=None, *, ubatch:int|None=None) -> DirectPackedPrefillRequest|None:
  quant, n, k = _direct_packed_quant(lin), getattr(lin, "out_features", None), getattr(lin, "in_features", None)
  if not quant or not all(isinstance(v, int) for v in (n, k)): return None
  requested_ubatch, m = 512 if ubatch is None else int(ubatch), 512 if ubatch is None else int(ubatch)
  if x is not None:
    if len(x.shape) != 3 or x.shape[0] != 1: return None
    m, x_k = x.shape[-2], x.shape[-1]
    if not all(isinstance(v, int) for v in (m, x_k)) or x_k != k: return None
  return DirectPackedPrefillRequest(quant, _direct_packed_module_role(lin), m, n, k,
                                    getattr(lin, "bias", None) is not None, requested_ubatch)

def select_direct_packed_prefill_shadow_request(lin, x:Tensor|None=None, *, ubatch:int|None=None) -> DirectPackedPrefillRequest|None:
  req = build_direct_packed_prefill_request(lin, x, ubatch=ubatch)
  return req if req is not None and _direct_packed_enabled_for(lin, req.quant) else None

def prefill_q4k_q8_mode(mode:str="") -> str:
  mode = str(mode).strip().lower()
  if mode not in Q4K_Q8_CHOICES:
    allowed = ", ".join(repr(x) for x in Q4K_Q8_CHOICES if x)
    raise ValueError(f"PREFILL_Q4K_Q8 must be one of {allowed}, got {mode!r}")
  return "" if mode in ("", "0", "false", "off", "no") else mode

def prefill_q4k_q8_role_enabled(role:str, roles:frozenset[str]|None=None) -> bool: return roles is None or role in roles

def _candidate_workspace_if_attached(value:Tensor, lin) -> Tensor:
  attachment = getattr(lin, "_prefill_route_attachment", None)
  policy = getattr(attachment, "selected_policy", None)
  candidate_id = policy.get("candidate_id") if isinstance(policy, Mapping) else None
  if not isinstance(candidate_id, str) or not candidate_id or not (isinstance(value, UOp) or isinstance(getattr(value, "uop", None), UOp)):
    return value
  return mark_candidate_workspace(value, candidate_id)

def _cooperative_target_matches(lin, required:Any) -> bool:
  facts = getattr(lin, "_prefill_device_facts", None)
  if not isinstance(required, dict) or not required or facts is None: return False
  if hasattr(facts, "planning_snapshot"): facts = facts.planning_snapshot()
  if not isinstance(facts, dict) or not isinstance((capabilities := facts.get("capabilities")), dict): return False
  observed = {"backend": facts.get("backend"), "arch": facts.get("architecture", facts.get("arch")),
              "architecture": facts.get("architecture", facts.get("arch")), **capabilities}
  allowed = {"backend", "arch", "architecture", "wave_size", "max_workgroup_threads",
             "max_workgroup_dimensions", "lds_bytes", "lds_allocation_granularity"}
  for name, value in required.items():
    if name == "capabilities":
      if not isinstance(value, dict) or not value or any(k not in allowed or capabilities.get(k) != v for k, v in value.items()): return False
    elif name not in allowed or observed.get(name) != value: return False
  return True

def _cooperative_evidence_matches(lin, spec:PrefillLinearRouteSpec, candidate:dict[str, Any], evidence:dict[str, Any]) -> bool:
  workload, ev_workload = candidate.get("workload", {}), evidence.get("workload", {})
  if not isinstance(workload, dict) or not isinstance(ev_workload, dict): return False
  if evidence.get("candidate_identity") != canonical_candidate_identity(candidate): return False
  route_id, capability, target = workload.get("route_id", candidate.get("route_id")), workload.get("capability"), workload.get("target")
  if not all(isinstance(x, str) and x.strip() for x in (route_id, capability)) or not _cooperative_target_matches(lin, target): return False
  expected = {"phase": "prefill", "role": spec.role, "quant_format": "Q4_K",
              "shape": {"M": spec.m, "N": spec.n, "K": spec.k}, "target": target,
              "capability": capability, "route_id": route_id}
  if any(workload.get(k) != v or ev_workload.get(k, evidence.get(k)) != v for k, v in expected.items()): return False
  fallback = evidence.get("fallback")
  if evidence.get("fallback_used") is not False or evidence.get("fallback_status") not in ("not_used", "none", False): return False
  if isinstance(fallback, dict) and fallback.get("used") is not False or fallback is not None and not isinstance(fallback, dict): return False
  return all(isinstance(evidence.get(k), str) and evidence[k].strip() for k in ("source_identity", "binary_identity"))

def _cooperative_q4k_binding(lin, spec:PrefillLinearRouteSpec, *, candidate:Mapping[str, Any]|None,
                             evidence:Mapping[str, Any]|None, enabled:bool) -> Any|None:
  if not enabled or spec.quant != "q4k" or not isinstance(candidate, Mapping) or not isinstance(evidence, Mapping): return None
  candidate, evidence = dict(candidate), dict(evidence)
  try: decision = admit_cooperative_mmq(candidate=candidate, evidence=evidence, enabled=True)
  except (KeyError, TypeError, ValueError): return None
  if not _cooperative_evidence_matches(lin, spec, candidate, evidence): return None
  return candidate if decision.admitted and evidence.get("emitter_proven") is True else None

def _direct_packed_spec(lin, x:Tensor) -> PrefillLinearRouteSpec|None:
  if getattr(lin, "bias", None) is not None or len(x.shape) != 3 or x.shape[0] != 1: return None
  m, k, n, in_f = x.shape[-2], x.shape[-1], getattr(lin, "out_features", None), getattr(lin, "in_features", None)
  if not all(isinstance(v, int) for v in (m, k, n, in_f)) or k != in_f: return None
  quant = "q4k" if _is_q4k_linear(lin) else "q6k" if _is_q6k_linear(lin) else ""
  return None if not quant else PrefillLinearRouteSpec("direct_packed", quant, "", m, n, k)

@dataclass(frozen=True)
class _ExactResearchDispatch:
  selection: ResearchSelection
  attachment: PrefillRouteAttachment
  frozen_bundle: str | Path | None
  fallback_program_identity: str | None
  role_spec: ExactRoleSpec | None
  inventory: str | Path | Mapping[str, Any]

def _exact_research_dispatch(lin, spec:PrefillLinearRouteSpec,
                             config:PrefillResearchRouteConfig) -> _ExactResearchDispatch | None:
  """Select an exact route using host facts only, before activation Tensor work."""
  if not config.exact_policy_enabled: return None
  authority = config.exact_authority
  if not isinstance(authority, ExactResearchRouteAuthority):
    raise ResearchPolicyBlocked("enabled exact research route requires explicit policy/bundle authority")
  if dict(authority.target) != TARGET:
    raise ResearchPolicyBlocked("exact research target authority differs from retained AMD gfx1100 wave32 target")
  quant = "Q4_K" if spec.quant == "q4k" else "Q6_K" if spec.quant == "q6k" else ""
  workload = ResearchWorkload("prefill", quant, spec.role, spec.m, spec.n, spec.k,
                              TARGET["backend"], TARGET["arch"], TARGET["wave_size"])
  groups = [group for group in GROUPS if group.workload.key == workload.key]
  if len(groups) != 1: raise ResearchPolicyBlocked("unknown exact research workload; no fallback is implied")
  attachment = getattr(lin, "_prefill_route_attachment", None)
  if not isinstance(attachment, PrefillRouteAttachment):
    raise ResearchPolicyBlocked("exact research route requires an exact runtime attachment")
  group = groups[0]
  selection = ExactSixRowResearchSelector(authority.policy, enabled=True).select(
    attachment.invocation_id, workload, expected_binding_identity=group.expected_binding_identity)
  if attachment.route_id != selection.route_id:
    raise ResearchPolicyBlocked("runtime attachment route differs from selected exact research route")
  attached_policy = attachment.selected_policy
  if not isinstance(attached_policy, Mapping) or attached_policy.get("artifact_identity") != RETAINED_POLICY_IDENTITY or \
     attached_policy.get("binding_identity") != selection.binding_identity:
    raise ResearchPolicyBlocked("runtime attachment policy differs from selected exact research binding")
  if selection.binding_kind == "candidate":
    bundle = authority.frozen_bundles.get(selection.binding_identity)
    if not isinstance(bundle, (str, Path)) or not str(bundle):
      raise ResearchPolicyBlocked("selected exact candidate has no frozen bundle authority")
    role_spec = exact_role_spec(spec.role, shape=(spec.m, spec.n, spec.k), inventory=authority.inventory)
    if role_spec.candidate_canonical_identity != selection.binding_identity:
      raise ResearchPolicyBlocked("frozen role identity differs from selected policy candidate")
    return _ExactResearchDispatch(selection, attachment, bundle, None, role_spec, authority.inventory)
  if selection.binding_kind == "fallback":
    program_identity = authority.fallback_program_identities.get(selection.binding_identity)
    if not isinstance(program_identity, str) or not program_identity:
      raise ResearchPolicyBlocked("selected direct_packed fallback has no program identity authority")
    return _ExactResearchDispatch(selection, attachment, None, program_identity, None, authority.inventory)
  raise ResearchPolicyBlocked(f"unsupported exact research binding kind {selection.binding_kind!r}")

def _notify_exact_execution(lin, dispatch:_ExactResearchDispatch, *, program_identity:str,
                            fallback_used:bool, fallback_reason:str|None) -> None:
  if not isinstance(program_identity, str) or not program_identity:
    raise RuntimeError("exact research execution has no program identity")
  notify_prefill_route_execution(lin, PrefillRouteExecution(
    dispatch.attachment.invocation_id, dispatch.selection.route_id, dispatch.selection.binding_identity,
    program_identity, fallback_used, fallback_reason))

def route_direct_packed_prefill_research(lin, x:Tensor, *, config:PrefillResearchRouteConfig) -> Tensor|None:
  if not isinstance(config, PrefillResearchRouteConfig): raise TypeError("research prefill route requires explicit config")
  if config.q4k_q8_mode not in Q4K_Q8_CHOICES: raise ValueError(f"invalid research q4k_q8_mode {config.q4k_q8_mode!r}")
  global _MMQ_DS4_LAST_PACKED
  if (spec := _direct_packed_spec(lin, x)) is None:
    if config.exact_policy_enabled:
      raise ResearchPolicyBlocked("enabled exact research route does not match a bias-free Q4_K/Q6_K prefill linear")
    return None
  spec = PrefillLinearRouteSpec(spec.route, spec.quant, _direct_packed_role(lin, spec), spec.m, spec.n, spec.k)
  exact_dispatch = _exact_research_dispatch(lin, spec, config)
  if exact_dispatch is not None and exact_dispatch.selection.binding_kind == "fallback":
    out = _run_direct_packed_baseline(lin, x, spec)
    if out is None: raise RuntimeError("selected exact direct_packed fallback produced no output")
    _notify_exact_execution(lin, exact_dispatch, program_identity=exact_dispatch.fallback_program_identity,
                            fallback_used=True,
                            fallback_reason="retained exact six-row policy selected direct_packed fallback")
    return out
  x_batch = prefill_activation(x[0].cast(dtypes.float16).contiguous())
  if exact_dispatch is not None:
    run = run_frozen_exact_q4k_research(
      lin, x_batch.reshape(spec.m, spec.k), role_spec=exact_dispatch.role_spec,
      frozen_bundle=exact_dispatch.frozen_bundle, enabled=True,
      inventory=exact_dispatch.inventory)
    if run is None or run.output is None:
      raise RuntimeError("selected exact frozen candidate returned no runtime result")
    if run.binding.candidate_identity != exact_dispatch.selection.binding_identity or \
       run.binding.role_spec != exact_dispatch.role_spec or \
       run.evidence.get("candidate_identity") != exact_dispatch.selection.binding_identity or \
       run.evidence.get("program_key") != run.binding.program_key or \
       run.evidence.get("shape") != list(exact_dispatch.role_spec.shape) or \
       run.evidence.get("program_shape") != list(exact_dispatch.role_spec.program.shape) or \
       not isinstance(run.binding.program_key, str) or not run.binding.program_key:
      raise RuntimeError("selected exact frozen candidate runtime identity drifted")
    notify_prefill_route(lin)
    _notify_exact_execution(lin, exact_dispatch, program_identity=run.binding.program_key,
                            fallback_used=False, fallback_reason=None)
    return prefill_output(run.output)
  if _is_q4k_linear(lin):
    cooperative = _cooperative_q4k_binding(lin, spec, candidate=config.cooperative_candidate,
                                            evidence=config.cooperative_evidence, enabled=config.cooperative_enabled)
    if cooperative is not None and callable(config.cooperative_runner):
      if (fused := config.cooperative_runner(cooperative, lin, x_batch, spec, x)) is not None:
        notify_prefill_route(lin); return fused
    if config.generated_tile: raise RuntimeError("PREFILL_QK_GENERATED_TILE was retired after the generated packed-tile route was refuted")
    if (mode := config.q4k_q8_mode) and (config.q4k_q8_roles is None or spec.role in config.q4k_q8_roles):
      words = lin.prefill_packed_weight().to(x.device)
      if mode == "wmma":
        xq, xscales = qk_ops.q8_1_quantize(x_batch.cast(dtypes.float32))
        desc = qk_ops.describe_q4k_int8_wmma_prefill(spec.n, spec.k, spec.m, role=spec.role, n_tile=max(16, config.wmma_n_tile))
        raw_elems = desc.groups * desc.m * desc.n
        if raw_elems > config.wmma_max_raw_elems and not config.wmma_allow_graph_explosion:
          raise RuntimeError(f"PREFILL_Q4K_Q8=wmma Tensor-substrate blocked: RAW groups*m*n={raw_elems} > limit={config.wmma_max_raw_elems}")
        out = qk_ops.emit_q4k_int8_wmma_prefill_tensor(words, xq, xscales, desc)
      elif mode == "wmma_tiled":
        xq, xscales = qk_ops.q8_1_quantize(x_batch.cast(dtypes.float32))
        desc = qk_ops.describe_q4k_int8_wmma_tiled_prefill(spec.n, spec.k, spec.m, role=spec.role,
          m_tile=max(16, config.wmma_tiled_m_tile), n_tile=max(16, config.wmma_tiled_n_tile), group_tile=max(1, config.wmma_tiled_group_tile))
        try: out = qk_ops.emit_q4k_int8_wmma_tiled_prefill_tensor(words, xq, xscales, desc)
        except NotImplementedError: out = qk_ops.emit_q4k_int8_wmma_tiled_scheduler_tensor(words, xq, xscales, desc)
      elif mode in ("packed_ds4", "packed_row_major", "packed_fused"):
        factory = qk_ops.packed_fused_candidate if mode == "packed_fused" else qk_ops.packed_row_major_candidate if mode == "packed_row_major" else qk_ops.packed_ds4_candidate
        desc, source = factory(spec.m, spec.n, spec.k, role=spec.role), x_batch.reshape(spec.m, spec.k)
        key = (getattr(x, "uop", x), spec.m, spec.k, str(x.device))
        if _MMQ_DS4_LAST_PACKED is not None and _MMQ_DS4_LAST_PACKED[0] == key: values, scales, sums = _MMQ_DS4_LAST_PACKED[1]
        else:
          packer = qk_ops.pack_q8_1_mmq_fused if mode == "packed_fused" else qk_ops.pack_q8_1_mmq_ds4
          values, scales, sums = (_candidate_workspace_if_attached(v, lin) for v in packer(source, desc))
          _MMQ_DS4_LAST_PACKED = (key, (values, scales, sums))
        out = qk_ops.emit_q4k_q8_mmq_ds4(words, values, scales, sums, desc)
      else: raise RuntimeError(f"PREFILL_Q4K_Q8={mode!r} matched no generated route")
      notify_prefill_route(lin); return prefill_output(out.reshape(1, spec.m, spec.n))
  return _run_direct_packed_baseline(lin, x, spec)

def route_prefill_q4k_gate_up_research(gate, up, x:Tensor, *, config:PrefillResearchRouteConfig) -> tuple[Tensor, Tensor]|None:
  if not isinstance(config, PrefillResearchRouteConfig): raise TypeError("research prefill route requires explicit config")
  if config.q4k_q8_mode != "packed_fused" or config.q4k_q8_roles is not None and "ffn_gate_up" not in config.q4k_q8_roles: return None
  gate_spec, up_spec = _direct_packed_spec(gate, x), _direct_packed_spec(up, x)
  if gate_spec is None or up_spec is None or (gate_spec.m, gate_spec.n, gate_spec.k) != (up_spec.m, up_spec.n, up_spec.k): return None
  m, n, k = gate_spec.m, gate_spec.n, gate_spec.k
  x_batch = prefill_activation(x[0].cast(dtypes.float16).contiguous())
  if not hasattr(gate, "_prefill_fused_gate_up_words"):
    gate._prefill_fused_gate_up_words = gate.prefill_packed_weight().to(x.device).cat(up.prefill_packed_weight().to(x.device), dim=0).contiguous()
  desc, key = qk_ops.packed_fused_candidate(m, n * 2, k, role="ffn_gate_up"), (getattr(x, "uop", x), m, k, str(x.device), "gate_up")
  global _MMQ_DS4_LAST_PACKED
  if _MMQ_DS4_LAST_PACKED is not None and _MMQ_DS4_LAST_PACKED[0] == key: values, scales, sums = _MMQ_DS4_LAST_PACKED[1]
  else:
    values, scales, sums = (_candidate_workspace_if_attached(v, gate) for v in qk_ops.pack_q8_1_mmq_fused(x_batch.reshape(m, k), desc))
    _MMQ_DS4_LAST_PACKED = (key, (values, scales, sums))
  out = prefill_output(qk_ops.emit_q4k_q8_mmq_ds4(gate._prefill_fused_gate_up_words, values, scales, sums, desc).reshape(1, m, n * 2))
  notify_prefill_route(gate); notify_prefill_route(up)
  return out[:, :, :n], out[:, :, n:]

__all__ = ["ExactResearchRouteAuthority", "PrefillResearchRouteConfig", "DirectPackedPrefillRequest", "build_direct_packed_prefill_request",
           "select_direct_packed_prefill_shadow_request", "prefill_q4k_q8_mode", "prefill_q4k_q8_role_enabled",
           "route_direct_packed_prefill_research", "route_prefill_q4k_gate_up_research"]
