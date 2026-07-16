from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from tinygrad import Tensor, dtypes
from tinygrad.llm import route_ops as qk_ops
from tinygrad.llm.memory_semantics import (mark_candidate_workspace, prefill_activation as _prefill_activation,
  prefill_output as _prefill_output, prefill_scratch as _prefill_scratch)
from tinygrad.llm.prefill_route_census import PrefillRouteAttachment, record_prefill_route
from tinygrad.uop.ops import UOp

activation_spec = qk_ops.qk_quant_specs_attr("activation_spec")
quant_spec = qk_ops.qk_quant_specs_attr("quant_spec")
RuntimeOpSpec = qk_ops.qk_runtime_specs_attr("RuntimeOpSpec")

PREFILL_ROUTE_CHOICES = ("auto", "fp16", "direct_packed")
LM_HEAD_PREFILL_ROUTE_CHOICES = ("lazy", "resident_fp16", "direct_packed")
# Handwritten sdot4/MMQ/Q8_1-GEMM prefill research modes deleted 2026-07-06 (no backups; dead end ~237 tok/s).
# Only the generated int8-WMMA parity substrates remain selectable; off-values fall to the direct-packed default.
Q4K_Q8_CHOICES = ("", "0", "false", "off", "no", "wmma", "wmma_tiled", "packed_ds4", "packed_row_major", "packed_fused")
_MMQ_DS4_LAST_PACKED: tuple[Any, tuple[Tensor, Tensor, Tensor]] | None = None


@dataclass(frozen=True)
class PrefillResearchRouteConfig:
  """Explicit test/benchmark configuration; never constructed by production routing."""
  q4k_q8_mode: str = ""
  q4k_q8_roles: frozenset[str] | None = None
  cooperative_candidate: Mapping[str, Any] | None = None
  cooperative_evidence: Mapping[str, Any] | None = None
  cooperative_enabled: bool = False
  generated_tile: bool = False
  wmma_n_tile: int = 256
  wmma_max_raw_elems: int = 64 * 1024 * 1024
  wmma_allow_graph_explosion: bool = False
  wmma_tiled_m_tile: int = 16
  wmma_tiled_n_tile: int = 16
  wmma_tiled_group_tile: int = 1


def _mark_tensor_semantic(value, marker):
  # Route unit tests use graphless structural Tensor stubs. Runtime Tensor/UOp
  # results always take the explicit marking path.
  return marker(value) if isinstance(value, UOp) or isinstance(getattr(value, "uop", None), UOp) else value


def prefill_activation(value): return _mark_tensor_semantic(value, _prefill_activation)
def prefill_output(value): return _mark_tensor_semantic(value, _prefill_output)
def prefill_scratch(value): return _mark_tensor_semantic(value, _prefill_scratch)


def _attached_candidate_id(lin) -> str | None:
  """Read identity only from the selected structural route attachment."""
  attachment = getattr(lin, "_prefill_route_attachment", None)
  if not isinstance(attachment, PrefillRouteAttachment): return None
  policy = attachment.selected_policy
  if not isinstance(policy, Mapping): return None
  candidate_id = policy.get("candidate_id")
  return candidate_id if isinstance(candidate_id, str) and candidate_id else None


def _attached_production_route(lin, x: Tensor) -> str | None:
  """Return the only production route authorized by the immutable attachment.

  Environment variables are deliberately not consulted here.  A route is
  usable only when the attachment binds one exact candidate id to the exact
  route id and the attached structural facts agree with this invocation.
  Unknown/research candidates fail closed to the ordinary fp16 path.
  """
  attachment = getattr(lin, "_prefill_route_attachment", None)
  if not isinstance(attachment, PrefillRouteAttachment): return None
  policy = attachment.selected_policy
  if not isinstance(policy, Mapping): return None
  candidate_id = policy.get("candidate_id")
  if not isinstance(candidate_id, str) or not candidate_id or attachment.route_id != candidate_id: return None
  facts = attachment.scanned_target_facts
  if facts is None: return None
  quant = _direct_packed_quant(lin)
  n, k = getattr(lin, "out_features", None), getattr(lin, "in_features", None)
  if quant == "" or not isinstance(n, int) or not isinstance(k, int) or len(x.shape) != 3 or x.shape[0] != 1:
    return None
  if not isinstance(x.shape[-2], int) or not isinstance(x.shape[-1], int) or x.shape[-1] != k: return None
  # These are the stable production candidate ids.  Research/MMQ ids are not
  # guessed or promoted merely because an environment knob names them.
  baseline_ids = {"direct_packed", "direct-packed-baseline", f"prefill_{quant.lower()}_direct_packed",
                  f"prefill_{quant.lower()}_direct_packed_load_direct_out"}
  if attachment.route_id in baseline_ids:
    return "direct_packed"
  if policy.get("strategy") == "FULL_RESIDENT_OVERLAY" and getattr(lin, "_pf16_w", None) is not None:
    return "fp16"
  return None


def _candidate_workspace_if_attached(value: Tensor, lin) -> Tensor:
  candidate_id = _attached_candidate_id(lin)
  if candidate_id is None or not (isinstance(value, UOp) or isinstance(getattr(value, "uop", None), UOp)): return value
  return mark_candidate_workspace(value, candidate_id)


def _cooperative_target_matches(lin, required: Any) -> bool:
  """Match candidate target requirements against facts scanned by the model/policy.

  Attribute contract: model/policy code must attach ``_prefill_device_facts``
  to each routed linear.  Its value may be a ``DeviceFacts`` instance or its
  mapping-shaped planning snapshot.  Only target/capability facts participate;
  model, profile, dimensions, and VRAM facts are deliberately not selectors.
  """
  facts = getattr(lin, "_prefill_device_facts", None)
  if not isinstance(required, dict) or not required or facts is None: return False
  if hasattr(facts, "planning_snapshot"):
    facts = facts.planning_snapshot()
  if not isinstance(facts, dict): return False
  capabilities = facts.get("capabilities")
  if not isinstance(capabilities, dict): return False
  observed = {"backend": facts.get("backend"), "arch": facts.get("architecture", facts.get("arch")),
              "architecture": facts.get("architecture", facts.get("arch")), **capabilities}
  allowed = {"backend", "arch", "architecture", "wave_size", "max_workgroup_threads",
             "max_workgroup_dimensions", "lds_bytes", "lds_allocation_granularity"}
  for name, value in required.items():
    if name == "capabilities":
      if not isinstance(value, dict) or not value: return False
      if any(key not in allowed or capabilities.get(key) is None or capabilities.get(key) != expected
             for key, expected in value.items()): return False
    elif name not in allowed or observed.get(name) is None or observed.get(name) != value:
      return False
  return True


def _cooperative_evidence_matches(lin, spec: "PrefillLinearRouteSpec", candidate: dict[str, Any],
                                   evidence: dict[str, Any]) -> bool:
  """Require generated evidence to identify this exact route workload.

  Profile and model-path fields are optional provenance. Compatibility is
  determined only by independently repeated structural workload facts.
  """
  workload = candidate.get("workload", {})
  ev_workload = evidence.get("workload", {})
  if not isinstance(workload, dict) or not isinstance(ev_workload, dict): return False
  from tinygrad.llm.cooperative_mmq_gate import canonical_candidate_identity
  if evidence.get("candidate_identity") != canonical_candidate_identity(candidate): return False
  expected_shape = {"M": spec.m, "N": spec.n, "K": spec.k}
  route_id = workload.get("route_id", candidate.get("route_id"))
  if not isinstance(route_id, str) or not route_id.strip(): return False
  capability = workload.get("capability")
  if not isinstance(capability, str) or not capability.strip(): return False
  target = workload.get("target")
  if not _cooperative_target_matches(lin, target): return False
  expected = {"phase": "prefill", "role": spec.role, "quant_format": "Q4_K", "shape": expected_shape,
              "target": target, "capability": capability, "route_id": route_id}
  # Candidate facts are part of the identity; evidence must independently
  # repeat them so a containment report cannot be relabeled at admission.
  for key, value in expected.items():
    if workload.get(key) != value or ev_workload.get(key, evidence.get(key)) != value: return False
  # Accept only an explicit no-fallback report.  Some producers use the flat
  # fields while compile gates use ``fallback.used``; checking only one lets a
  # dynamic-loop candidate claim generated provenance after silently rolling
  # back to the ordinary route.
  fallback = evidence.get("fallback")
  if evidence.get("fallback_used") is not False: return False
  if evidence.get("fallback_status") not in ("not_used", "none", False): return False
  if isinstance(fallback, dict) and fallback.get("used") is not False: return False
  if fallback is not None and not isinstance(fallback, dict): return False
  if not isinstance(evidence.get("source_identity"), str) or not evidence["source_identity"].strip(): return False
  if not isinstance(evidence.get("binary_identity"), str) or not evidence["binary_identity"].strip(): return False
  return True


def _cooperative_q4k_binding(lin, spec: "PrefillLinearRouteSpec", *, candidate: Mapping[str, Any] | None,
                             evidence: Mapping[str, Any] | None, enabled: bool) -> Any | None:
  """Return an admitted cooperative candidate, never a contract-only candidate.

  The payload/evidence are intentionally supplied out-of-band by the search
  runner.  The route remains blocked until a real emitter is exposed through
  route_ops and the evidence is bound to this exact runtime shape.
  """
  if not enabled or spec.quant != "q4k" or not isinstance(candidate, Mapping) or not isinstance(evidence, Mapping): return None
  candidate, evidence = dict(candidate), dict(evidence)
  try:
    from tinygrad.llm.cooperative_mmq_gate import admit_cooperative_mmq
    decision = admit_cooperative_mmq(candidate=candidate, evidence=evidence, enabled=True)
  except (KeyError, TypeError, ValueError):
    return None
  # JSON being parseable is not evidence.  Keep the runtime boundary
  # fail-closed for null, list, and scalar payloads before reading fields.
  if not _cooperative_evidence_matches(lin, spec, candidate, evidence): return None
  # Contracts and evidence from a probe are not an emitter.  Keep this
  # binding path dormant until the generated emitter is actually callable.
  if not decision.admitted or evidence.get("emitter_proven") is not True:
    return None
  return candidate


def _run_cooperative_q4k(candidate: dict[str, Any], lin, x_batch: Tensor,
                         spec: "PrefillLinearRouteSpec", x: Tensor) -> Tensor | None:
  """Run an admitted fused-Q4 candidate; callers retain direct-packed rollback."""
  workload = candidate.get("workload", {})
  descriptor = candidate.get("descriptor", {})
  if workload.get("shape") != {"M": spec.m, "N": spec.n, "K": spec.k}: return None
  from extra.qk.mmq_ds4_logical_emitter import packed_fused_candidate
  # Descriptor geometry is identity evidence, not an unchecked constructor
  # override; the logical emitter owns its validated tile contract.
  if {descriptor.get(k) for k in ("m_tile", "n_tile", "k_tile")} - {None, 16, 256}: return None
  fused = packed_fused_candidate(spec.m, spec.n, spec.k, role=spec.role)
  words = lin.prefill_packed_weight().to(x.device)
  values, scales, sums = qk_ops.pack_q8_1_mmq_fused(x_batch.reshape(spec.m, spec.k), fused)
  out = qk_ops.emit_q4k_q8_mmq_ds4(words, values, scales, sums, fused)
  return prefill_output(out.reshape(1, spec.m, spec.n))


def prefill_route_policy(route:str="auto", *, direct_packed:bool=False) -> str:
  route = str(route).strip().lower()
  if route == "direct": route = "direct_packed"
  if direct_packed and route == "auto": route = "direct_packed"
  if route not in PREFILL_ROUTE_CHOICES:
    raise ValueError(f"PREFILL_ROUTE must be one of {', '.join(PREFILL_ROUTE_CHOICES)}, got {route!r}")
  return route


def prefill_route_strict(strict:bool=False) -> bool:
  return bool(strict)


def prefill_lm_head_route_policy(route:str="lazy") -> str:
  """Select how a full-sequence LM head is evaluated during prefill.

  ``lazy`` preserves the ordinary output-linear graph so a downstream
  ``[:, -1, :]`` can prune the projection to one token. The two explicit
  full-sequence modes are retained for explicitly described workloads that
  consume every token's logits.
  """
  route = str(route).strip().lower()
  if route not in LM_HEAD_PREFILL_ROUTE_CHOICES:
    raise ValueError(f"PREFILL_LM_HEAD_ROUTE must be one of {', '.join(LM_HEAD_PREFILL_ROUTE_CHOICES)}, got {route!r}")
  return route


def prefill_q4k_q8_mode(mode:str="") -> str:
  mode = str(mode).strip().lower()
  if mode not in Q4K_Q8_CHOICES:
    allowed = ", ".join(repr(x) for x in Q4K_Q8_CHOICES if x)
    raise ValueError(f"PREFILL_Q4K_Q8 must be one of {allowed}, got {mode!r}")
  if mode in ("", "0", "false", "off", "no"): return ""
  return mode


def prefill_q4k_q8_role_enabled(role: str, roles:frozenset[str]|None=None) -> bool:
  """Optionally scope an experimental Q4/Q8 lowering to named logical roles."""
  return roles is None or role in roles


def _is_q4k_linear(lin) -> bool: return hasattr(lin, "q4k_storage") and hasattr(lin, "prefill_packed_weight")
def _is_q6k_linear(lin) -> bool: return hasattr(lin, "q6k_storage") and hasattr(lin, "prefill_packed_weight")
def is_direct_packed_prefill_linear(lin) -> bool: return _is_q4k_linear(lin) or _is_q6k_linear(lin)


def _direct_packed_enabled_for(lin, quant:str) -> bool:
  return quant.upper() in ("Q4_K", "Q6_K") and is_direct_packed_prefill_linear(lin)


def _direct_packed_b_upcast(m:int) -> int:
  # 14B pp512 direct-packed: 4 beats the former 16-token unroll; lower values lose occupancy/reuse.
  return min(m, 16, 4)


def _direct_packed_role(lin, spec:"PrefillLinearRouteSpec") -> str:
  role = spec.role
  if role: return role
  for attr in ("route_role", "role"):
    role = str(getattr(lin, attr, ""))
    if role: return role
  name = str(getattr(lin, "name", ""))
  if any(x in name for x in ("ffn_gate", "ffn_up")): return "ffn_gate_up"
  if "ffn_down" in name: return "ffn_down"
  if any(x in name for x in ("attn_q", "attn_output")): return "attn_qo"
  if any(x in name for x in ("attn_k", "attn_v")): return "attn_kv"
  if name == "output.weight" or name.rsplit(".", 1)[-1] == "output": return "lm_head"
  return ""


def _direct_packed_parts(lin, spec:"PrefillLinearRouteSpec") -> int:
  base = int(getattr(lin, "parts", 1))
  role = _direct_packed_role(lin, spec)
  if role == "ffn_down" and spec.quant == "q4k":
    return 1
  if spec.quant == "q6k":
    return 1
  return max(1, base)


def _direct_packed_opts(lin, spec:"PrefillLinearRouteSpec"):
  if spec.quant == "q4k":
    parse = qk_ops.q4k_parse_opt
    # The promoted Q4 baseline owns this measured tile4x4 schedule. Ambient
    # tuning variables cannot relabel its candidate descriptor at runtime.
    return tuple(parse(x) for x in ("LOCAL:0:16", "LOCAL:1:16", "UPCAST:0:4", "UPCAST:1:4"))
  else:
    parse = qk_ops.q6k_parse_opt
  return tuple(getattr(lin, "opts", ())) + (parse(f"UPCAST:1:{_direct_packed_b_upcast(spec.m)}"),)


def prefill_route_wants_resident_fp16(*, est_gb:float, budget_gb:float, has_direct_packed:bool, route:str="auto") -> bool:
  route = prefill_route_policy(route)
  if route == "fp16": return True
  if route == "direct_packed": return False
  return not has_direct_packed or est_gb <= budget_gb


@dataclass(frozen=True)
class PrefillLinearRouteSpec:
  route: str
  quant: str
  role: str
  m: int
  n: int
  k: int

  @property
  def kernel_prefix(self) -> str:
    return f"prefill_{self.quant.lower()}_{self.route}_gemm"

  @property
  def q4k_kernel_prefix(self) -> str:
    return f"prefill_{self.quant.lower()}_direct_packed_load_gemm"

  @property
  def q6k_kernel_prefix(self) -> str:
    return f"prefill_{self.quant.lower()}_direct_packed_load_gemm"

  def runtime_op_spec(self, *, activation_format:str="fp16", lowering_strategy:str="packed_dequant_dot",
                      device:str="unknown") -> RuntimeOpSpec:
    qfmt = "Q4_K" if self.quant == "q4k" else "Q6_K" if self.quant == "q6k" else "unknown"
    role = self.role if self.role else "unknown"
    return RuntimeOpSpec("QuantizedLinear", "prefill", role, {"M": self.m, "N": self.n, "K": self.k},
                         quant_spec(qfmt).tensor_spec(), activation_spec(activation_format).activation_spec(),
                         lowering_strategy=lowering_strategy, device=device,
                         route_id=f"prefill_{self.quant}_{self.route}")


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


@dataclass(frozen=True)
class DirectPackedPrefillCandidate:
  quant: str

  def matches(self, lin, spec:PrefillLinearRouteSpec) -> bool:
    return spec.quant == self.quant

  def run(self, lin, x:Tensor, x_batch:Tensor, spec:PrefillLinearRouteSpec) -> Tensor | None:
    raise NotImplementedError


@dataclass(frozen=True)
class Q4KDirectPackedPrefillCandidate(DirectPackedPrefillCandidate):
  quant: str = "q4k"

  def run(self, lin, x:Tensor, x_batch:Tensor, spec:PrefillLinearRouteSpec) -> Tensor | None:
    words = lin.prefill_packed_weight().to(x.device)
    parts = _direct_packed_parts(lin, spec)
    partials = prefill_scratch(Tensor.empty(spec.n, spec.m, parts, dtype=dtypes.float32, device=x.device))
    opts = _direct_packed_opts(lin, spec)
    output_layout = "direct_out" if parts == 1 else "partials"
    q4_spec = qk_ops.describe_q4k_packed_prefill_generated(spec.n, spec.k, spec.m,
                                                           role=_direct_packed_role(lin, spec), parts=parts,
                                                           output_layout=output_layout, opts=opts)
    if output_layout == "direct_out":
      out = prefill_output(Tensor.empty(spec.m, spec.n, dtype=dtypes.float32, device=x.device).custom_kernel(
        words, x_batch.reshape(spec.m * spec.k), fxn=qk_ops.emit_q4k_packed_prefill_kernel(q4_spec))[0])
      return prefill_output(out.reshape(1, spec.m, spec.n))
    out = prefill_scratch(partials.custom_kernel(words, x_batch.reshape(spec.m * spec.k),
      fxn=qk_ops.emit_q4k_packed_prefill_kernel(q4_spec))[0])
    return prefill_output(out.sum(axis=2).transpose(0, 1).reshape(1, spec.m, spec.n))


@dataclass(frozen=True)
class Q6KDirectPackedPrefillCandidate(DirectPackedPrefillCandidate):
  quant: str = "q6k"

  def run(self, lin, x:Tensor, x_batch:Tensor, spec:PrefillLinearRouteSpec) -> Tensor | None:
    halfs = lin.prefill_packed_weight().to(x.device)
    parts = _direct_packed_parts(lin, spec)
    partials = prefill_scratch(Tensor.empty(spec.n, spec.m, parts, dtype=dtypes.float32, device=x.device))
    opts = _direct_packed_opts(lin, spec)
    output_layout = "direct_out" if parts == 1 else "partials"
    q6_spec = qk_ops.describe_q6k_packed_prefill(spec.n, spec.k, spec.m, role=_direct_packed_role(lin, spec),
                                                 parts=parts, output_layout=output_layout, opts=opts)
    if output_layout == "direct_out":
      out = prefill_output(Tensor.empty(spec.m, spec.n, dtype=dtypes.float32, device=x.device).custom_kernel(
        halfs, x_batch.reshape(spec.m * spec.k), fxn=qk_ops.emit_q6k_packed_prefill_kernel(q6_spec))[0])
      return prefill_output(out.reshape(1, spec.m, spec.n))
    out = prefill_scratch(partials.custom_kernel(halfs, x_batch.reshape(spec.m * spec.k),
      fxn=qk_ops.emit_q6k_packed_prefill_kernel(q6_spec))[0])
    return prefill_output(out.sum(axis=2).transpose(0, 1).reshape(1, spec.m, spec.n))


DIRECT_PACKED_PREFILL_CANDIDATES: tuple[DirectPackedPrefillCandidate, ...] = (
  Q4KDirectPackedPrefillCandidate(),
  Q6KDirectPackedPrefillCandidate(),
)


def select_direct_packed_prefill_candidate(lin, spec:PrefillLinearRouteSpec) -> DirectPackedPrefillCandidate | None:
  for candidate in DIRECT_PACKED_PREFILL_CANDIDATES:
    if candidate.matches(lin, spec): return candidate
  return None


def _direct_packed_quant(lin) -> str:
  if _is_q4k_linear(lin): return "Q4_K"
  if _is_q6k_linear(lin): return "Q6_K"
  return ""


def _direct_packed_module_role(lin) -> str:
  role = str(getattr(lin, "_prefill_graph_role", ""))
  if role: return role
  for attr in ("route_role", "role"):
    role = str(getattr(lin, attr, ""))
    if role: return role
  name = str(getattr(lin, "name", ""))
  if any(x in name for x in ("ffn_gate", "ffn_up")): return "ffn_gate_up"
  if "ffn_down" in name: return "ffn_down"
  if any(x in name for x in ("attn_q", "attn_output")): return "attn_qo"
  if any(x in name for x in ("attn_k", "attn_v")): return "attn_kv"
  if name == "output.weight" or name.rsplit(".", 1)[-1] == "output": return "lm_head"
  return ""


def build_direct_packed_prefill_request(lin, x:Tensor | None=None, *, ubatch:int | None=None) -> DirectPackedPrefillRequest | None:
  quant = _direct_packed_quant(lin)
  n, k = getattr(lin, "out_features", None), getattr(lin, "in_features", None)
  if quant == "" or not all(isinstance(v, int) for v in (n, k)): return None
  requested_ubatch = 512 if ubatch is None else int(ubatch)
  m = requested_ubatch
  if x is not None:
    if len(x.shape) != 3 or x.shape[0] != 1: return None
    m, x_k = x.shape[-2], x.shape[-1]
    if not all(isinstance(v, int) for v in (m, x_k)) or x_k != k: return None
  return DirectPackedPrefillRequest(quant, _direct_packed_module_role(lin), m, n, k,
                                    getattr(lin, "bias", None) is not None, requested_ubatch)


def select_direct_packed_prefill_shadow_request(lin, x:Tensor | None=None, *, ubatch:int | None=None) -> DirectPackedPrefillRequest | None:
  req = build_direct_packed_prefill_request(lin, x, ubatch=ubatch)
  if req is None: return None
  if not _direct_packed_enabled_for(lin, req.quant): return None
  return req


def _direct_packed_spec(lin, x:Tensor) -> PrefillLinearRouteSpec | None:
  if getattr(lin, "bias", None) is not None or len(x.shape) != 3 or x.shape[0] != 1: return None
  m, k = x.shape[-2], x.shape[-1]
  n, in_f = getattr(lin, "out_features", None), getattr(lin, "in_features", None)
  if not all(isinstance(v, int) for v in (m, k, n, in_f)) or k != in_f: return None
  quant = "q4k" if _is_q4k_linear(lin) else "q6k" if _is_q6k_linear(lin) else ""
  if quant == "": return None
  return PrefillLinearRouteSpec("direct_packed", quant, _direct_packed_module_role(lin), m, n, k)


def _attached_direct_packed_spec(lin, x:Tensor) -> PrefillLinearRouteSpec | None:
  """Build the production baseline spec from attachment and structural facts only."""
  if _attached_production_route(lin, x) != "direct_packed": return None
  if getattr(lin, "bias", None) is not None or len(x.shape) != 3 or x.shape[0] != 1: return None
  m, k = x.shape[-2], x.shape[-1]
  n, in_f = getattr(lin, "out_features", None), getattr(lin, "in_features", None)
  if not all(isinstance(v, int) for v in (m, k, n, in_f)) or k != in_f: return None
  quant = "q4k" if _is_q4k_linear(lin) else "q6k" if _is_q6k_linear(lin) else ""
  if quant == "": return None
  return PrefillLinearRouteSpec("direct_packed", quant, _direct_packed_module_role(lin), m, n, k)


def _run_direct_packed_baseline(lin, x:Tensor, spec:PrefillLinearRouteSpec) -> Tensor | None:
  x_batch = prefill_activation(x[0].cast(dtypes.float16).contiguous())
  candidate = select_direct_packed_prefill_candidate(lin, spec)
  if candidate is None: return None
  out = candidate.run(lin, x, x_batch, spec)
  record_prefill_route(lin)
  return out


def route_direct_packed_prefill(lin, x:Tensor) -> Tensor | None:
  """Production direct-packed baseline; attachment is the sole selector."""
  spec = _attached_direct_packed_spec(lin, x)
  return None if spec is None else _run_direct_packed_baseline(lin, x, spec)


def route_direct_packed_prefill_research(lin, x:Tensor, *, config:PrefillResearchRouteConfig) -> Tensor | None:
  """Test/benchmark-only explicit dispatch for historical prefill research routes."""
  if not isinstance(config, PrefillResearchRouteConfig): raise TypeError("research prefill route requires explicit config")
  if config.q4k_q8_mode not in Q4K_Q8_CHOICES: raise ValueError(f"invalid research q4k_q8_mode {config.q4k_q8_mode!r}")
  global _MMQ_DS4_LAST_PACKED
  spec = _direct_packed_spec(lin, x)
  if spec is None: return None
  x_batch = prefill_activation(x[0].cast(dtypes.float16).contiguous())
  if _is_q4k_linear(lin):
    role = _direct_packed_role(lin, spec)
    # Probe the promoted binding at the generated-route boundary.  It is
    # deliberately a no-op today: no callable cooperative emitter is proven.
    # The ordinary generated/direct route below remains the rollback.
    cooperative = _cooperative_q4k_binding(lin, spec, candidate=config.cooperative_candidate,
                                            evidence=config.cooperative_evidence, enabled=config.cooperative_enabled)
    if cooperative is not None:
      fused = _run_cooperative_q4k(cooperative, lin, x_batch, spec, x)
      if fused is not None: record_prefill_route(lin); return fused
    if config.generated_tile:
      raise RuntimeError("PREFILL_QK_GENERATED_TILE was retired after the generated packed-tile route was refuted; "
                         "use the Q4KPrefillRouteSpec direct-packed default or PREFILL_Q4K_Q8=wmma_tiled research.")
    q8_mode = config.q4k_q8_mode
    role_enabled = config.q4k_q8_roles is None or role in config.q4k_q8_roles
    if q8_mode and role_enabled:
      words = lin.prefill_packed_weight().to(x.device)
      if q8_mode == "wmma":
        xq, xscales = qk_ops.q8_1_quantize(x_batch.cast(dtypes.float32))
        wmma_spec = qk_ops.describe_q4k_int8_wmma_prefill(spec.n, spec.k, spec.m, role=role,
                                                          n_tile=max(16, config.wmma_n_tile))
        raw_elems = wmma_spec.groups * wmma_spec.m * wmma_spec.n
        raw_limit = config.wmma_max_raw_elems
        if raw_elems > raw_limit and not config.wmma_allow_graph_explosion:
          raise RuntimeError(f"PREFILL_Q4K_Q8=wmma Tensor-substrate blocked for full-model shape "
                             f"role={role or '?'} m={spec.m} n={spec.n} k={spec.k}: RAW groups*m*n={raw_elems} "
                             f"> limit={raw_limit}. This parity/codegen substrate is correct, but 14B authority "
                             f"needs the next fused/tiled generated emitter, not many Tensor matmul graph fragments. "
                             f"Set PREFILL_Q4K_WMMA_ALLOW_GRAPH_EXPLOSION=1 only for debugging.")
        out = qk_ops.emit_q4k_int8_wmma_prefill_tensor(words, xq, xscales, wmma_spec)
        record_prefill_route(lin); return prefill_output(out.reshape(1, spec.m, spec.n))
      if q8_mode == "wmma_tiled":
        xq, xscales = qk_ops.q8_1_quantize(x_batch.cast(dtypes.float32))
        tiled_spec = qk_ops.describe_q4k_int8_wmma_tiled_prefill(
          spec.n, spec.k, spec.m, role=role,
          m_tile=max(16, config.wmma_tiled_m_tile), n_tile=max(16, config.wmma_tiled_n_tile),
          group_tile=max(1, config.wmma_tiled_group_tile))
        try:
          out = qk_ops.emit_q4k_int8_wmma_tiled_prefill_tensor(words, xq, xscales, tiled_spec)
        except NotImplementedError:
          out = qk_ops.emit_q4k_int8_wmma_tiled_scheduler_tensor(words, xq, xscales, tiled_spec)
        record_prefill_route(lin); return prefill_output(out.reshape(1, spec.m, spec.n))
      if q8_mode in ("packed_ds4", "packed_row_major", "packed_fused"):
        candidate_factory = (qk_ops.packed_fused_candidate if q8_mode == "packed_fused" else
                             qk_ops.packed_row_major_candidate if q8_mode == "packed_row_major" else qk_ops.packed_ds4_candidate)
        candidate = candidate_factory(spec.m, spec.n, spec.k, role=role)
        source = x_batch.reshape(spec.m, spec.k)
        cache_key = (getattr(x, "uop", x), spec.m, spec.k, str(x.device))
        if _MMQ_DS4_LAST_PACKED is not None and _MMQ_DS4_LAST_PACKED[0] == cache_key:
          values, scales, sums = _MMQ_DS4_LAST_PACKED[1]
        else:
          packer = qk_ops.pack_q8_1_mmq_fused if q8_mode == "packed_fused" else qk_ops.pack_q8_1_mmq_ds4
          values, scales, sums = (_candidate_workspace_if_attached(value, lin) for value in packer(source, candidate))
          _MMQ_DS4_LAST_PACKED = (cache_key, (values, scales, sums))
        out = qk_ops.emit_q4k_q8_mmq_ds4(words, values, scales, sums, candidate)
        record_prefill_route(lin); return prefill_output(out.reshape(1, spec.m, spec.n))
      raise RuntimeError(f"PREFILL_Q4K_Q8={q8_mode!r} matched no generated route; the handwritten sdot4/MMQ/Q8_1-GEMM "
                         f"modes were deleted 2026-07-06. Only generated modes or off-values are valid.")
  return _run_direct_packed_baseline(lin, x, spec)


def route_prefill_q4k_gate_up(gate, up, x: Tensor) -> tuple[Tensor, Tensor] | None:
  """Production hook: no fused gate/up candidate is currently attachment-promoted."""
  return None


def route_prefill_q4k_gate_up_research(gate, up, x: Tensor, *,
                                       config:PrefillResearchRouteConfig) -> tuple[Tensor, Tensor] | None:
  """Test/benchmark-only explicit dispatch for horizontal Q4/Q8 gate/up research."""
  if not isinstance(config, PrefillResearchRouteConfig): raise TypeError("research prefill route requires explicit config")
  if config.q4k_q8_mode != "packed_fused": return None
  if config.q4k_q8_roles is not None and "ffn_gate_up" not in config.q4k_q8_roles: return None
  gate_spec, up_spec = _direct_packed_spec(gate, x), _direct_packed_spec(up, x)
  if gate_spec is None or up_spec is None or (gate_spec.m, gate_spec.k) != (up_spec.m, up_spec.k): return None
  if gate_spec.n != up_spec.n: return None
  m, n, k = gate_spec.m, gate_spec.n, gate_spec.k
  x_batch = prefill_activation(x[0].cast(dtypes.float16).contiguous())
  words_attr = "_prefill_fused_gate_up_words"
  if not hasattr(gate, words_attr):
    fused_words = gate.prefill_packed_weight().to(x.device).cat(up.prefill_packed_weight().to(x.device), dim=0).contiguous()
    setattr(gate, words_attr, fused_words)
  words = getattr(gate, words_attr)
  candidate = qk_ops.packed_fused_candidate(m, n * 2, k, role="ffn_gate_up")
  cache_key = (getattr(x, "uop", x), m, k, str(x.device), "gate_up")
  global _MMQ_DS4_LAST_PACKED
  if _MMQ_DS4_LAST_PACKED is not None and _MMQ_DS4_LAST_PACKED[0] == cache_key:
    values, scales, sums = _MMQ_DS4_LAST_PACKED[1]
  else:
    values, scales, sums = (_candidate_workspace_if_attached(value, gate)
                            for value in qk_ops.pack_q8_1_mmq_fused(x_batch.reshape(m, k), candidate))
    _MMQ_DS4_LAST_PACKED = (cache_key, (values, scales, sums))
  out = prefill_output(qk_ops.emit_q4k_q8_mmq_ds4(words, values, scales, sums, candidate).reshape(1, m, n * 2))
  record_prefill_route(gate); record_prefill_route(up)
  return out[:, :, :n], out[:, :, n:]


def route_prefill_linear(lin, x:Tensor, *, prefill_graph_gemm:bool) -> Tensor:
  route = _attached_production_route(lin, x)
  w = getattr(lin, "_pf16_w", None)

  if route == "direct_packed":
    routed = route_direct_packed_prefill(lin, x)
    if routed is not None: return routed

  if route == "fp16" and prefill_graph_gemm and w is not None:
    routed = qk_ops.route_pf16_graph_gemm(lin, x)
    if routed is not None: record_prefill_route(lin); return routed
  if w is None: w = lin.weight.cast(dtypes.float16)
  b = getattr(lin, "bias", None)
  out = x.cast(dtypes.float16).linear(w.transpose(), b.cast(dtypes.float16) if b is not None else None)
  record_prefill_route(lin)
  return out
