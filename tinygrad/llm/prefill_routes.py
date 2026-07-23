from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping
from tinygrad import Tensor, dtypes
from tinygrad.llm import route_ops as qk_ops
from tinygrad.llm.memory_semantics import (prefill_activation as _prefill_activation,
  prefill_output as _prefill_output, prefill_scratch as _prefill_scratch)
from tinygrad.llm.prefill_route_observer import PrefillDirectPackedBinding, PrefillRouteAttachment, _ACTIVE, notify_prefill_route
from tinygrad.uop.ops import UOp

_PREFILL_ROUTE_OVERRIDE: ContextVar[Callable[[object, Tensor], Tensor | None] | None] = \
  ContextVar("tinygrad_prefill_route_override", default=None)


@contextmanager
def prefill_route_override(route: Callable[[object, Tensor], Tensor | None]) -> Iterator[None]:
  """Install one explicit, context-local route for benchmark/research calls.

  Ordinary model execution never installs this scope and therefore retains the
  immutable production attachment path below. The override is deliberately a
  callable rather than an environment switch: the caller must already hold all
  policy, artifact, and fallback authority, and any exception fails the call.
  """
  if not callable(route): raise TypeError("prefill route override must be callable")
  token = _PREFILL_ROUTE_OVERRIDE.set(route)
  try: yield
  finally: _PREFILL_ROUTE_OVERRIDE.reset(token)


def _mark_tensor_semantic(value, marker):
  # Route unit tests use graphless structural Tensor stubs. Runtime Tensor/UOp
  # results always take the explicit marking path.
  return marker(value) if isinstance(value, UOp) or isinstance(getattr(value, "uop", None), UOp) else value


def prefill_activation(value): return _mark_tensor_semantic(value, _prefill_activation)
def prefill_output(value): return _mark_tensor_semantic(value, _prefill_output)
def prefill_scratch(value): return _mark_tensor_semantic(value, _prefill_scratch)


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
  if policy.get("strategy") == "BOUNDED_PACKED_TILES":
    from tinygrad.llm.prefill_policy import bounded_packed_projection_proven_eligible
    proof = policy.get("bounded_packed_projection_proof", {})
    if (bounded_packed_projection_proven_eligible(policy, facts) and attachment.allocation_owner_identity ==
        proof.get("allocation_owner_identity")):
      return "bounded_packed"
  if policy.get("strategy") == "FULL_RESIDENT_OVERLAY" and getattr(lin, "_pf16_w", None) is not None:
    return "fp16"
  return None


def _is_q4k_linear(lin) -> bool: return hasattr(lin, "q4k_storage") and hasattr(lin, "prefill_packed_weight")
def _is_q6k_linear(lin) -> bool: return hasattr(lin, "q6k_storage") and hasattr(lin, "prefill_packed_weight")
def is_direct_packed_prefill_linear(lin) -> bool: return _is_q4k_linear(lin) or _is_q6k_linear(lin)


def _direct_packed_enabled_for(lin, quant:str) -> bool:
  return quant.upper() in ("Q4_K", "Q6_K") and is_direct_packed_prefill_linear(lin)


def _direct_packed_b_upcast(m:int) -> int:
  # 14B pp512 direct-packed: 4 beats the former 16-token unroll; lower values lose occupancy/reuse.
  return min(m, 16, 4)


def _direct_packed_role(lin, spec:"PrefillLinearRouteSpec") -> str:
  return spec.role or _direct_packed_module_role(lin)


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
    # The full-vocabulary LM head is the one output shape whose four-way
    # upcast joins non-contiguous vocabulary rows into a constructed float32
    # value on HIP. Keep its established local tile but leave the output
    # scalar-addressable; normal prefill roles retain the measured 4x4 path.
    # Some model-forward attachments name this projection `output` rather
    # than `lm_head`; the full-vocabulary width is the structural ownership
    # fact at the primitive boundary. No admitted dense prefill role reaches
    # this extent.
    if spec.role == "lm_head" or spec.n >= 131072:
      return tuple(parse(x) for x in ("LOCAL:0:16", "LOCAL:1:16"))
    return tuple(parse(x) for x in ("LOCAL:0:16", "LOCAL:1:16", "UPCAST:0:4", "UPCAST:1:4"))
  else:
    parse = qk_ops.q6k_parse_opt
  return tuple(getattr(lin, "opts", ())) + (parse(f"UPCAST:1:{_direct_packed_b_upcast(spec.m)}"),)


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

@dataclass(frozen=True)
class DirectPackedPrefillFormat:
  quant: str
  describe_op: str
  emit_op: str

  def describe(self, lin, spec:PrefillLinearRouteSpec, *, parts:int, output_layout:str, opts):
    return getattr(qk_ops, self.describe_op)(spec.n, spec.k, spec.m, role=_direct_packed_role(lin, spec),
      parts=parts, output_layout=output_layout, opts=opts)

  def emit(self, route_spec):
    return getattr(qk_ops, self.emit_op)(route_spec)


@dataclass(frozen=True)
class DirectPackedPrefillCandidate:
  format: DirectPackedPrefillFormat

  @property
  def quant(self) -> str: return self.format.quant

  def matches(self, lin, spec:PrefillLinearRouteSpec) -> bool:
    return spec.quant == self.quant

  def run(self, lin, x:Tensor, x_batch:Tensor, spec:PrefillLinearRouteSpec) -> Tensor | None:
    return _execute_direct_packed_prefill(self.format, lin, x, x_batch, spec)

class Q4KDirectPackedPrefillCandidate(DirectPackedPrefillCandidate):
  def __init__(self): super().__init__(DirectPackedPrefillFormat(
    "q4k", "describe_q4k_packed_prefill_generated", "emit_q4k_packed_prefill_kernel"))

class Q6KDirectPackedPrefillCandidate(DirectPackedPrefillCandidate):
  def __init__(self): super().__init__(DirectPackedPrefillFormat(
    "q6k", "describe_q6k_packed_prefill", "emit_q6k_packed_prefill_kernel"))


def _execute_direct_packed_prefill(format:DirectPackedPrefillFormat, lin, x:Tensor, x_batch:Tensor,
                                    spec:PrefillLinearRouteSpec) -> Tensor:
  packed_weight = lin.prefill_packed_weight().to(x.device)
  parts = _direct_packed_parts(lin, spec)
  output_layout = "direct_out" if parts == 1 else "partials"
  route_spec = format.describe(lin, spec, parts=parts, output_layout=output_layout, opts=_direct_packed_opts(lin, spec))
  kernel = format.emit(route_spec)
  activation = x_batch.reshape(spec.m * spec.k)
  if output_layout == "direct_out":
    out = prefill_output(Tensor.empty(spec.m, spec.n, dtype=dtypes.float32, device=x.device).custom_kernel(
      packed_weight, activation, fxn=kernel)[0])
    return prefill_output(out.reshape(1, spec.m, spec.n))
  partials = prefill_scratch(Tensor.empty(spec.n, spec.m, parts, dtype=dtypes.float32, device=x.device))
  out = prefill_scratch(partials.custom_kernel(packed_weight, activation, fxn=kernel)[0])
  return prefill_output(out.sum(axis=2).transpose(0, 1).reshape(1, spec.m, spec.n))


DIRECT_PACKED_PREFILL_CANDIDATES: tuple[DirectPackedPrefillCandidate, ...] = (
  Q4KDirectPackedPrefillCandidate(), Q6KDirectPackedPrefillCandidate(),
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


def _attached_direct_packed_spec(lin, x:Tensor) -> PrefillLinearRouteSpec | None:
  """Build the production baseline spec from attachment and structural facts only."""
  if _attached_production_route(lin, x) not in ("direct_packed", "bounded_packed"): return None
  binding = getattr(lin, "_prefill_direct_packed_binding", None)
  if not isinstance(binding, PrefillDirectPackedBinding) or binding.phase != "prefill" or not _ACTIVE.get(): return None
  if getattr(lin, "bias", None) is not None or len(x.shape) != 3 or x.shape[0] != 1: return None
  m, k = x.shape[-2], x.shape[-1]
  n, in_f = getattr(lin, "out_features", None), getattr(lin, "in_features", None)
  if not all(isinstance(v, int) for v in (m, k, n, in_f)) or k != in_f: return None
  if binding.shape != (m, n, k) or binding.role != _direct_packed_module_role(lin): return None
  attachment = getattr(lin, "_prefill_route_attachment", None)
  if not isinstance(attachment, PrefillRouteAttachment) or attachment.invocation_id != binding.invocation_id: return None
  quant = "q4k" if _is_q4k_linear(lin) else "q6k" if _is_q6k_linear(lin) else ""
  if quant == "": return None
  route = "bounded_packed" if _attached_production_route(lin, x) == "bounded_packed" else "direct_packed"
  return PrefillLinearRouteSpec(route, quant, _direct_packed_module_role(lin), m, n, k)


def _run_direct_packed_baseline(lin, x:Tensor, spec:PrefillLinearRouteSpec) -> Tensor | None:
  x_batch = prefill_activation(x[0].cast(dtypes.float16).contiguous())
  candidate = select_direct_packed_prefill_candidate(lin, spec)
  if candidate is None: return None
  out = candidate.run(lin, x, x_batch, spec)
  notify_prefill_route(lin)
  return out


def route_direct_packed_prefill(lin, x:Tensor) -> Tensor | None:
  """Production direct-packed baseline; attachment is the sole selector."""
  spec = _attached_direct_packed_spec(lin, x)
  return None if spec is None else _run_direct_packed_baseline(lin, x, spec)


def packed_wmma_prefill_enabled() -> bool:
  """Gate for the packed-WMMA prefill candidates (Q4KPackedWmmaPrefillCandidate /
  Q6KPackedWmmaPrefillCandidate, extra/qk/prefill/packed_wmma_prefill_candidates.py).

  Default is ON: the packed-WMMA route is correctness-gated (6/6 combos, max_abs 0.0)
  and fails closed for anything ungated or unknown-shaped. Set TINYGRAD_PREFILL_PACKED_WMMA=0
  to revert to the direct-packed baseline only.
  """
  from tinygrad.helpers import getenv
  return bool(getenv("TINYGRAD_PREFILL_PACKED_WMMA", 1))


def route_packed_wmma_prefill(lin, x:Tensor) -> Tensor | None:
  """Production packed-WMMA route: an accelerated implementation of the direct-packed
  strategy for Q4_K/Q6_K prefill linears whose (quant, role) has a gated, frozen packed-WMMA
  geometry (see PACKED_WMMA_GEOM). Only reachable when packed_wmma_prefill_enabled(); declines
  (returns None) for anything ungated, unknown-shaped, or outside the frozen geometry table --
  the caller falls through to route_direct_packed_prefill.
  """
  if not packed_wmma_prefill_enabled(): return None
  spec = _attached_direct_packed_spec(lin, x)
  if spec is None: return None
  from extra.qk.prefill.packed_wmma_prefill_candidates import select_packed_wmma_prefill_candidate
  candidate = select_packed_wmma_prefill_candidate(lin, spec)
  if candidate is None: return None
  x_batch = prefill_activation(x[0].cast(dtypes.float16).contiguous())
  out = candidate.run(lin, x, x_batch, spec)
  if out is None: return None
  notify_prefill_route(lin)
  return prefill_output(out)


def route_prefill_linear(lin, x:Tensor) -> Tensor:
  if (override := _PREFILL_ROUTE_OVERRIDE.get()) is not None:
    routed = override(lin, x)
    if routed is not None: return routed
  route = _attached_production_route(lin, x)
  w = getattr(lin, "_pf16_w", None)

  if route in ("direct_packed", "bounded_packed"):
    routed = route_packed_wmma_prefill(lin, x)
    if routed is not None: return routed
    routed = route_direct_packed_prefill(lin, x)
    if routed is not None: return routed

  # Exact binding presence is the only Graph-GEMM execution authority.
  if route == "fp16" and getattr(lin, "_prefill_graph_gemm_binding", None) is not None and w is not None:
    routed = qk_ops.route_pf16_graph_gemm(lin, x)
    if routed is not None: notify_prefill_route(lin); return routed
  if w is None: w = lin.weight.cast(dtypes.float16)
  b = getattr(lin, "bias", None)
  out = x.cast(dtypes.float16).linear(w.transpose(), b.cast(dtypes.float16) if b is not None else None)
  notify_prefill_route(lin)
  return out
