from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping
from tinygrad import Tensor, dtypes
from tinygrad.llm.packed_wmma_prefill import select_packed_wmma_prefill_candidate
from tinygrad.llm.model_facts import packed_linear_quant, route_role_for_linear
from tinygrad.llm.prefill_graph_gemm import route_pf16_graph_gemm
from tinygrad.llm.memory_semantics import (prefill_activation as _prefill_activation,
  prefill_output as _prefill_output, prefill_scratch as _prefill_scratch)
from tinygrad.llm.prefill_route_observer import PrefillDirectPackedBinding, PrefillRouteAttachment, _ACTIVE, notify_prefill_route
from tinygrad.llm.route_selection import RouteCandidatePolicy, RouteLifecycle, parse_route_mode
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
  quant = packed_linear_quant(lin)
  n, k = getattr(lin, "out_features", None), getattr(lin, "in_features", None)
  if quant == "" or not isinstance(n, int) or not isinstance(k, int) or len(x.shape) != 3 or x.shape[0] != 1:
    return None
  if not isinstance(x.shape[-2], int) or not isinstance(x.shape[-1], int) or x.shape[-1] != k: return None
  # These are the stable production candidate ids.  Research/MMQ ids are not
  # guessed or promoted merely because an environment knob names them.
  # Legacy attachments named the packed-storage strategy after its old
  # handwritten fallback. They now authorize only the exact searched
  # packed-WMMA rows; a selector decline falls through to ordinary tinygrad.
  baseline_ids = {"direct_packed", "direct-packed-baseline", f"prefill_{quant.lower()}_direct_packed",
                  f"prefill_{quant.lower()}_direct_packed_load_direct_out"}
  if attachment.route_id in baseline_ids:
    return "packed_wmma"
  if policy.get("strategy") == "BOUNDED_PACKED_TILES":
    from tinygrad.llm.prefill_policy import bounded_packed_projection_proven_eligible
    proof = policy.get("bounded_packed_projection_proof", {})
    if (bounded_packed_projection_proven_eligible(policy, facts) and attachment.allocation_owner_identity ==
        proof.get("allocation_owner_identity")):
      return "packed_wmma"
  if policy.get("strategy") == "FULL_RESIDENT_OVERLAY" and getattr(lin, "_pf16_w", None) is not None:
    return "fp16"
  return None


def is_direct_packed_prefill_linear(lin) -> bool: return bool(packed_linear_quant(lin))


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

def _direct_packed_quant(lin) -> str:
  return packed_linear_quant(lin)


def _direct_packed_module_role(lin) -> str:
  return route_role_for_linear(lin)


def _attached_packed_wmma_spec(lin, x:Tensor) -> PrefillLinearRouteSpec | None:
  """Build a spec for the exact searched packed-WMMA selector.

  Packed storage and a legacy attachment are necessary but not sufficient: the
  selector still has to match one of its six promoted rows. All other shapes use
  the normal tinygrad graph.
  """
  if _attached_production_route(lin, x) != "packed_wmma": return None
  binding = getattr(lin, "_prefill_direct_packed_binding", None)
  if not isinstance(binding, PrefillDirectPackedBinding) or binding.phase != "prefill": return None
  if not _ACTIVE.get(): return None
  if getattr(lin, "bias", None) is not None or len(x.shape) != 3 or x.shape[0] != 1: return None
  m, k = x.shape[-2], x.shape[-1]
  n, in_f = getattr(lin, "out_features", None), getattr(lin, "in_features", None)
  if not all(isinstance(v, int) for v in (m, k, n, in_f)) or k != in_f: return None
  if binding.shape != (m, n, k) or binding.role != _direct_packed_module_role(lin): return None
  attachment = getattr(lin, "_prefill_route_attachment", None)
  if not isinstance(attachment, PrefillRouteAttachment) or attachment.invocation_id != binding.invocation_id: return None
  quant = {"Q4_K": "q4k", "Q6_K": "q6k"}.get(packed_linear_quant(lin), "")
  if quant == "": return None
  return PrefillLinearRouteSpec("packed_wmma", quant, _direct_packed_module_role(lin), m, n, k)


PREFILL_PACKED_WMMA_POLICY = RouteCandidatePolicy("packed-wmma-prefill", RouteLifecycle.PROMOTED)
PREFILL_FP16_POLICY = RouteCandidatePolicy("fp16-prefill", RouteLifecycle.FALLBACK)
_GENERIC_PREFILL_POLICY = RouteCandidatePolicy("generic-tinygrad-prefill", RouteLifecycle.FALLBACK)


def direct_packed_prefill_policy(n_heads:int, n_kv_heads:int) -> RouteCandidatePolicy:
  """Compatibility API: the former direct-packed fallback is now generic tinygrad."""
  del n_heads, n_kv_heads
  return _GENERIC_PREFILL_POLICY


def prefill_route_mode(getenv_fn=None) -> str:
  """Canonical prefill selector with compatibility for the former boolean gate."""
  if getenv_fn is None:
    from tinygrad.helpers import getenv
    getenv_fn = getenv
  canonical = str(getenv_fn("TINYGRAD_PREFILL_ROUTE", "")).strip()
  if canonical:
    selected = parse_route_mode("TINYGRAD_PREFILL_ROUTE", allowed=("auto", "packed_wmma", "direct_packed", "fp16"),
                                aliases={"packed-wmma": "packed_wmma", "direct-packed": "direct_packed"}, getenv_fn=getenv_fn)
    return "fp16" if selected == "direct_packed" else selected
  selected = parse_route_mode("TINYGRAD_PREFILL_PACKED_WMMA", allowed=("auto", "direct_packed"), default="1",
                          aliases={"1": "auto", "true": "auto", "on": "auto", "yes": "auto",
                                   "0": "direct_packed", "false": "direct_packed", "off": "direct_packed", "no": "direct_packed"},
                          getenv_fn=getenv_fn)
  return "fp16" if selected == "direct_packed" else selected


def packed_wmma_prefill_enabled() -> bool:
  """Gate for the packed-WMMA prefill candidates (Q4KPackedWmmaPrefillCandidate /
  Q6KPackedWmmaPrefillCandidate, tinygrad/llm/packed_wmma_prefill.py).

  Default is ON: the packed-WMMA route is correctness-gated (6/6 combos, max_abs 0.0)
  and fails closed for anything ungated or unknown-shaped. Set TINYGRAD_PREFILL_PACKED_WMMA=0
  to use the ordinary tinygrad graph fallback.
  """
  return prefill_route_mode() in ("auto", "packed_wmma")


def validate_prefill_route_mode(n_heads:int, n_kv_heads:int) -> None:
  del n_heads, n_kv_heads
  prefill_route_mode()  # parse once so invalid explicit modes still fail loudly


def validate_packed_wmma_prefill_mode(n_heads:int, n_kv_heads:int) -> None:
  """Compatibility alias for external callers."""
  validate_prefill_route_mode(n_heads, n_kv_heads)


def route_packed_wmma_prefill(lin, x:Tensor) -> Tensor | None:
  """Production packed-WMMA route: an accelerated implementation of the direct-packed
  strategy for Q4_K/Q6_K prefill linears whose (quant, role) has a gated, frozen packed-WMMA
  geometry. Only reachable when packed_wmma_prefill_enabled(); declines (returns None)
  for anything ungated, unknown-shaped, or outside the frozen geometry table --
  the caller falls through to the ordinary tinygrad graph.
  """
  if not packed_wmma_prefill_enabled(): return None
  spec = _attached_packed_wmma_spec(lin, x)
  if spec is None: return None
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
  mode = prefill_route_mode()

  if route == "packed_wmma" and mode in ("auto", "packed_wmma"):
    routed = route_packed_wmma_prefill(lin, x)
    if routed is not None: return routed

  # Exact binding presence is the only Graph-GEMM execution authority.
  if route == "fp16" and getattr(lin, "_prefill_graph_gemm_binding", None) is not None and w is not None:
    routed = route_pf16_graph_gemm(lin, x)
    if routed is not None: notify_prefill_route(lin); return routed
  if w is None: w = lin.weight.cast(dtypes.float16)
  b = getattr(lin, "bias", None)
  out = x.cast(dtypes.float16).linear(w.transpose(), b.cast(dtypes.float16) if b is not None else None)
  notify_prefill_route(lin)
  return out
