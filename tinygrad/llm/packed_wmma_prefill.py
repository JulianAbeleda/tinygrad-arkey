"""Selected packed-WMMA prefill runtime for the supported 14B linear shapes.

This module contains only the production selection and execution contract.  Candidate
generation, tuning, qualification canaries, and rejected MMQ experiments remain outside
the tinygrad runtime.  The six rows below are frozen promotion results: changing a shape,
geometry, or identity requires a new qualification campaign.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt.packed_weight import PackedWeightTransform
from tinygrad.llm.model_facts import packed_linear_quant, route_role_for_linear
from tinygrad.llm.qk_layout import Q4_K, Q6_K, QUANT_FORMATS, QuantFormat


@dataclass(frozen=True)
class PackedWmmaRoute:
  quant: QuantFormat
  role: str
  shape: tuple[int, int, int]
  geometry: tuple[int, int, int, int, int, int]
  canonical_identity: str
  canary_max_abs_error: float = 0.0

  def __post_init__(self):
    if isinstance(self.quant, str):
      try: quant = QUANT_FORMATS[self.quant]
      except KeyError as exc: raise ValueError(f"packed-WMMA quant must be Q4_K or Q6_K, got {self.quant!r}") from exc
      object.__setattr__(self, "quant", quant)
    elif self.quant is not Q4_K and self.quant is not Q6_K:
      raise ValueError(f"packed-WMMA quant must be Q4_K or Q6_K, got {self.quant!r}")

  @property
  def geom(self) -> dict[str, int]:
    return dict(zip(("tm", "tn", "tk", "wm", "wn", "bc"), self.geometry))


# Exact current production rows.  Identities are the semantic SHA-256 values produced by
# the promoted buffer-2 candidate templates after geometry mutation and packed-weight
# binding; profile labels are intentionally absent from identity.
PACKED_WMMA_ROUTES: tuple[PackedWmmaRoute, ...] = (
  PackedWmmaRoute(Q4_K, "attn_qo", (512, 5120, 5120), (128, 32, 32, 4, 1, 1),
                  "3506a4e53c3375aabdda6ca3fe451a7730a332d4ff757709794969aa36f5baae"),
  PackedWmmaRoute(Q4_K, "attn_kv", (512, 1024, 5120), (64, 32, 32, 2, 1, 1),
                  "32e1d4aeef93c04b5fbbead31199de2ce71593751341e524d156f365de57b360"),
  PackedWmmaRoute(Q4_K, "ffn_gate_up", (512, 17408, 5120), (256, 64, 32, 8, 1, 1),
                  "eb1bef353afeec27d2aa569d0e6df03894be63fd133614fff87b8a29a0e7c677"),
  PackedWmmaRoute(Q4_K, "ffn_down", (512, 5120, 17408), (256, 128, 32, 8, 2, 2),
                  "1b543534011aa9060bd7aea1e13c7c61891d6b45b3184885609344e539cfa1c3"),
  PackedWmmaRoute(Q6_K, "attn_kv", (512, 1024, 5120), (64, 32, 32, 2, 1, 1),
                  "f80fd2595f3c3f25a9256867f34dbee13457f8fcd47840b58784660661b64081"),
  PackedWmmaRoute(Q6_K, "ffn_down", (512, 5120, 17408), (256, 64, 32, 8, 1, 1),
                  "ac1184a78db8be3ca22379a37531af54415666e1a5260195a0adddd4b8fcdf15"),
)
PACKED_WMMA_ROUTE_BY_KEY = {(row.quant, row.role, row.shape): row for row in PACKED_WMMA_ROUTES}

# Compatibility geometry view used by callers and qualification tooling. A geometry is
# selected only through the exact route table above; this derived map grants no admission.
PACKED_WMMA_GEOM: dict[tuple[str, str], dict[str, int]] = {
  (row.quant, row.role): row.geom for row in PACKED_WMMA_ROUTES
}


@dataclass(frozen=True)
class _LDSWindow:
  role: str
  base: int
  end: int
  stride_bytes: int


@dataclass(frozen=True)
class _TileGeometry:
  tile: tuple[int, int, int]
  waves: tuple[int, int]
  threads: int
  wave_size: int
  lds_windows: tuple[_LDSWindow, ...]

  @property
  def lds_bytes(self) -> int: return self.lds_windows[-1].end


@dataclass(frozen=True)
class _CandidateContext:
  schema_version: str
  canonical_identity: str
  geometry: _TileGeometry
  pipeline: Any
  packed_weight: PackedWeightTransform

  @property
  def packed_operand_b(self) -> PackedWeightTransform: return self.packed_weight


def _route(quant: QuantFormat, role: str, shape: tuple[int, int, int]) -> PackedWmmaRoute | None:
  return PACKED_WMMA_ROUTE_BY_KEY.get((quant, role, shape))


# Promotion recorded full-output max_abs_error=0.0 for all six exact rows.  Runtime admission
# consumes that immutable result and caches it once per process.  The optional verifier is a
# qualification seam: dev/exp may rerun the isolated GPU canary without making master import
# the qualification harness.
CanaryVerifier = Callable[[PackedWmmaRoute], tuple[bool, float | None]]
_CANARY_VERIFIER: CanaryVerifier | None = None
_GATE_CACHE: dict[tuple[QuantFormat, str, int, int, int], tuple[bool, float | None]] = {}
_ENTRY_CACHE: dict[tuple[QuantFormat, str, int, int, int], dict[str, Any]] = {}


def set_packed_wmma_canary_verifier(verifier: CanaryVerifier | None) -> None:
  """Install a qualification verifier and invalidate cached admissions."""
  global _CANARY_VERIFIER
  _CANARY_VERIFIER = verifier
  _GATE_CACHE.clear()
  _ENTRY_CACHE.clear()


def gate_combo(quant: QuantFormat, role: str, shape: tuple[int, int, int]) -> bool:
  key = (quant, role, *shape)
  if key not in _GATE_CACHE:
    row = _route(quant, role, shape)
    if row is None:
      _GATE_CACHE[key] = (False, None)
    elif _CANARY_VERIFIER is None:
      _GATE_CACHE[key] = (row.canary_max_abs_error == 0.0, row.canary_max_abs_error)
    else:
      try: _GATE_CACHE[key] = _CANARY_VERIFIER(row)
      except Exception: _GATE_CACHE[key] = (False, None)
  return bool(_GATE_CACHE[key][0])


def gate_result(quant: QuantFormat, role: str, shape: tuple[int, int, int]) -> tuple[bool, float | None] | None:
  return _GATE_CACHE.get((quant, role, *shape))


def packed_half_carrier(src: Tensor, transform: PackedWeightTransform, n: int, k: int) -> Tensor:
  """Movement-only packed-to-half carrier consumed by packed-weight postrange lowering."""
  blocks, halfwords = n * k // transform.block_elems, int(transform.block_bytes) // 2
  return src.bitcast(dtypes.uint16).reshape(blocks, halfwords).pad(((0, 0), (0, 128-halfwords))) \
    .reshape(blocks, 128, 1).expand(blocks, 128, 2).reshape(n, k).bitcast(dtypes.float16)


def _candidate_context(row: PackedWmmaRoute) -> tuple[_CandidateContext, PackedWeightTransform]:
  from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan
  g = row.geom
  stride, a_end = 80, g["tm"] * 80
  geometry = _TileGeometry((g["tm"], g["tn"], g["tk"]), (g["wm"], g["wn"]),
                           g["wm"] * g["wn"] * 32, 32,
                           (_LDSWindow("A", 0, a_end, stride),
                            _LDSWindow("B", a_end, a_end + g["tn"] * 80, stride)))
  # Preserve the established single-buffer context identity: buffer-1 carries
  # no explicit pipeline object, while buffer-2 owns a typed stage-1 plan.
  pipeline = KernelStage1PipelinePlan(g["bc"], geometry.lds_bytes, 1) if g["bc"] > 1 else None
  transform = PackedWeightTransform(row.quant, row.shape[1], row.shape[2])
  return _CandidateContext("boltbeam.full_kernel_candidate.v1", row.canonical_identity, geometry, pipeline, transform), transform


def warmstart_entry(quant: QuantFormat, role: str, shape: tuple[int, int, int]) -> dict[str, Any]:
  """Return the exact compiler warmstart entry for an admitted production row."""
  key = (quant, role, *shape)
  if not gate_combo(quant, role, shape): raise ValueError(f"packed-WMMA row {(quant, role, shape)!r} is not admitted")
  if key not in _ENTRY_CACHE:
    from tinygrad.codegen.opt import Opt, OptOps
    from tinygrad.codegen.opt.postrange import warmstart_key
    row = PACKED_WMMA_ROUTE_BY_KEY[(quant, role, shape)]
    context, transform = _candidate_context(row)
    m, n, k = shape
    _ENTRY_CACHE[key] = {"key": warmstart_key({m, n}, k, transform.storage_dtype),
      "opt": (Opt(OptOps.TC, 0, (-1, 2, 1)),), "context": context, "transform": transform,
      "m": m, "n": n, "k": k, "canonical_identity": row.canonical_identity, "one_buffer": False}
  return _ENTRY_CACHE[key]


@dataclass(frozen=True)
class PackedWmmaPrefillCandidate:
  quant: QuantFormat

  def matches(self, lin, spec) -> bool:
    return getattr(spec, "quant", None) == self.quant

  def run(self, lin, x: Tensor, x_batch: Tensor, spec) -> Tensor | None:
    del x
    role = str(getattr(spec, "role", "") or "")
    shape = (getattr(spec, "m", 0), getattr(spec, "n", 0), getattr(spec, "k", 0))
    if _route(self.quant, role, shape) is None or not gate_combo(self.quant, role, shape): return None
    try: entry = warmstart_entry(self.quant, role, shape)
    except Exception: return None
    transform = entry["transform"]
    b = packed_half_carrier(lin.prefill_packed_weight(), transform, shape[1], shape[2])
    out = (x_batch @ b.transpose()).contiguous()
    setattr(lin, "_prefill_full_kernel_candidate_identity", entry["canonical_identity"])
    setattr(lin, "_prefill_full_kernel_candidate_one_buffer", False)
    return out.reshape(1, shape[0], shape[1])


class Q4KPackedWmmaPrefillCandidate(PackedWmmaPrefillCandidate):
  def __init__(self): super().__init__(Q4_K)


class Q6KPackedWmmaPrefillCandidate(PackedWmmaPrefillCandidate):
  def __init__(self): super().__init__(Q6_K)


PACKED_WMMA_PREFILL_CANDIDATES: tuple[PackedWmmaPrefillCandidate, ...] = (
  Q4KPackedWmmaPrefillCandidate(), Q6KPackedWmmaPrefillCandidate())


def select_packed_wmma_prefill_candidate(lin, spec) -> PackedWmmaPrefillCandidate | None:
  del lin
  shape = (getattr(spec, "m", 0), getattr(spec, "n", 0), getattr(spec, "k", 0))
  role = str(getattr(spec, "role", "") or "")
  for candidate in PACKED_WMMA_PREFILL_CANDIDATES:
    if candidate.matches(None, spec) and _route(candidate.quant, role, shape) is not None: return candidate
  return None


def build_packed_wmma_warmstart_tables(covered_linears, ubatch: int) -> tuple[dict, dict]:
  opts, contexts = {}, {}
  for lin, out_f, in_f in covered_linears:
    quant = packed_linear_quant(lin)
    if quant is None: continue
    role = route_role_for_linear(lin)
    shape = (ubatch, out_f, in_f)
    if _route(quant, role, shape) is None or not gate_combo(quant, role, shape): continue
    try: entry = warmstart_entry(quant, role, shape)
    except Exception: continue
    opts[entry["key"]], contexts[entry["key"]] = entry["opt"], entry["context"]
  return opts, contexts


__all__ = ["PACKED_WMMA_GEOM", "PACKED_WMMA_ROUTES", "PackedWmmaPrefillCandidate",
  "Q4KPackedWmmaPrefillCandidate", "Q6KPackedWmmaPrefillCandidate", "build_packed_wmma_warmstart_tables",
  "gate_combo", "gate_result", "packed_half_carrier", "select_packed_wmma_prefill_candidate",
  "set_packed_wmma_canary_verifier", "warmstart_entry"]
