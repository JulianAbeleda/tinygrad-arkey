from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from tinygrad import Tensor, dtypes
from tinygrad.llm import route_ops as qk_ops

activation_spec = qk_ops.qk_quant_specs_attr("activation_spec")
quant_spec = qk_ops.qk_quant_specs_attr("quant_spec")
RuntimeOpSpec = qk_ops.qk_runtime_specs_attr("RuntimeOpSpec")

PREFILL_ROUTE_CHOICES = ("auto", "fp16", "direct_packed")
LM_HEAD_PREFILL_ROUTE_CHOICES = ("lazy", "resident_fp16", "direct_packed")
# Handwritten sdot4/MMQ/Q8_1-GEMM prefill research modes deleted 2026-07-06 (no backups; dead end ~237 tok/s).
# Only the generated int8-WMMA parity substrates remain selectable; off-values fall to the direct-packed default.
Q4K_Q8_CHOICES = ("", "0", "false", "off", "no", "wmma", "wmma_tiled", "packed_ds4")
_MMQ_DS4_LAST_PACKED: tuple[Any, tuple[Tensor, Tensor, Tensor]] | None = None


def _env(key:str, default:Any=0) -> Any:
  return type(default)(os.environ.get(key, default))


def prefill_route_policy() -> str:
  route = str(_env("PREFILL_ROUTE", "auto")).strip().lower()
  if route == "direct": route = "direct_packed"
  if bool(_env("PREFILL_QK_DIRECT", 0)) and route == "auto": route = "direct_packed"
  if route not in PREFILL_ROUTE_CHOICES:
    raise ValueError(f"PREFILL_ROUTE must be one of {', '.join(PREFILL_ROUTE_CHOICES)}, got {route!r}")
  return route


def prefill_route_strict() -> bool:
  return bool(_env("PREFILL_ROUTE_STRICT", _env("QK_GENERATED_POLICY_STRICT", 0)))


def prefill_lm_head_route_policy() -> str:
  """Select how a full-sequence LM head is evaluated during prefill.

  ``lazy`` preserves the ordinary output-linear graph so a downstream
  ``[:, -1, :]`` can prune the projection to one token.  The two explicit
  full-sequence modes are retained for workloads that consume every token's
  logits.  ``PREFILL_LM_HEAD_DIRECT=1`` remains a compatibility alias for the
  direct-packed experiment.
  """
  route = str(os.environ.get("PREFILL_LM_HEAD_ROUTE", "")).strip().lower()
  if not route:
    route = "direct_packed" if bool(_env("PREFILL_LM_HEAD_DIRECT", 0)) else "lazy"
  if route not in LM_HEAD_PREFILL_ROUTE_CHOICES:
    raise ValueError(f"PREFILL_LM_HEAD_ROUTE must be one of {', '.join(LM_HEAD_PREFILL_ROUTE_CHOICES)}, got {route!r}")
  return route


def prefill_q4k_q8_mode() -> str:
  mode = str(os.environ.get("PREFILL_Q4K_Q8", "")).strip().lower()
  if mode not in Q4K_Q8_CHOICES:
    allowed = ", ".join(repr(x) for x in Q4K_Q8_CHOICES if x)
    raise ValueError(f"PREFILL_Q4K_Q8 must be one of {allowed}, got {mode!r}")
  if mode in ("", "0", "false", "off", "no"): return ""
  return mode


def _is_q4k_linear(lin) -> bool: return hasattr(lin, "q4k_storage") and hasattr(lin, "prefill_packed_weight")
def _is_q6k_linear(lin) -> bool: return hasattr(lin, "q6k_storage") and hasattr(lin, "prefill_packed_weight")
def is_direct_packed_prefill_linear(lin) -> bool: return _is_q4k_linear(lin) or _is_q6k_linear(lin)


def _csv_set(key:str, default:str) -> set[str]:
  raw = str(_env(key, default)).replace(" ", "")
  return {x for x in raw.split(",") if x}


def _direct_packed_enabled_for(lin, quant:str) -> bool:
  quants = _csv_set("PREFILL_DIRECT_QUANTS", "Q4_K,Q6_K")
  if quants and quant.upper() not in quants: return False
  names = _csv_set("PREFILL_DIRECT_TENSORS", "")
  if names and not any(n in str(getattr(lin, "name", "")) for n in names): return False
  skip = _csv_set("PREFILL_DIRECT_SKIP_TENSORS", "")
  if skip and any(n in str(getattr(lin, "name", "")) for n in skip): return False
  return True


def _direct_packed_b_upcast(m:int) -> int:
  # 14B pp512 direct-packed: 4 beats the former 16-token unroll; lower values lose occupancy/reuse.
  return min(m, 16, max(1, int(_env("PREFILL_DIRECT_B_UPCAST", 4))))


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
  keys = []
  role = _direct_packed_role(lin, spec)
  role_key = "".join(ch.upper() if ch.isalnum() else "_" for ch in role)
  if role_key: keys.append(f"PREFILL_DIRECT_{role_key}_PARTS")
  keys += [f"PREFILL_DIRECT_{spec.quant.upper()}_PARTS", "PREFILL_DIRECT_PARTS"]
  for key in keys:
    raw = os.environ.get(key)
    if raw not in (None, ""):
      return max(1, int(raw))
  if role == "ffn_down" and spec.quant == "q4k":
    return 1
  if spec.quant == "q6k":
    return 1
  return max(1, base)


def _direct_packed_opts(lin, spec:"PrefillLinearRouteSpec"):
  role = _direct_packed_role(lin, spec)
  role_key = "".join(ch.upper() if ch.isalnum() else "_" for ch in role)
  if spec.quant == "q4k":
    parse = qk_ops.q4k_parse_opt
    override_key, extra_key = "PREFILL_Q4K_DIRECT_OPTS", "PREFILL_Q4K_DIRECT_EXTRA_OPTS"
  else:
    parse = qk_ops.q6k_parse_opt
    override_key, extra_key = "PREFILL_Q6K_DIRECT_OPTS", "PREFILL_Q6K_DIRECT_EXTRA_OPTS"
  if role_key:
    role_override = str(os.environ.get(f"PREFILL_DIRECT_{role_key}_OPTS", "")).strip()
    role_extra = str(os.environ.get(f"PREFILL_DIRECT_{role_key}_EXTRA_OPTS", "")).strip()
    if role_override: override_key, extra_key = f"PREFILL_DIRECT_{role_key}_OPTS", f"PREFILL_DIRECT_{role_key}_EXTRA_OPTS"
    elif role_extra: extra_key = f"PREFILL_DIRECT_{role_key}_EXTRA_OPTS"
  override = str(os.environ.get(override_key, "")).strip()
  if override:
    return tuple(parse(x) for x in override.replace(";", ",").split(",") if x.strip())
  extra = str(os.environ.get(extra_key, "")).strip()
  if spec.quant == "q4k" and str(os.environ.get("PREFILL_Q4K_DIRECT_SCHEDULE", "tile4x4")).strip().lower() != "legacy":
    # 14B pp512: this keeps Q4 dequant token-invariant across a 4x4 register tile. Clean whole-prefill: 135.7 -> 172.7 tok/s.
    opts = tuple(parse(x) for x in ("LOCAL:0:16", "LOCAL:1:16", "UPCAST:0:4", "UPCAST:1:4"))
  else:
    opts = tuple(getattr(lin, "opts", ())) + (parse(f"UPCAST:1:{_direct_packed_b_upcast(spec.m)}"),)
  if extra:
    opts += tuple(parse(x) for x in extra.replace(";", ",").split(",") if x.strip())
  return opts


def prefill_route_wants_resident_fp16(*, est_gb:float, budget_gb:float, has_direct_packed:bool) -> bool:
  route = prefill_route_policy()
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
    suffix = "direct_packed_load" if bool(_env("PREFILL_Q4K_PACKED_LOAD", 1)) else self.route
    return f"prefill_{self.quant.lower()}_{suffix}_gemm"

  @property
  def q6k_kernel_prefix(self) -> str:
    suffix = "direct_packed_load" if bool(_env("PREFILL_Q6K_PACKED_LOAD", 1)) else self.route
    return f"prefill_{self.quant.lower()}_{suffix}_gemm"

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
    partials = Tensor.empty(spec.n, spec.m, parts, dtype=dtypes.float32, device=x.device)
    opts = _direct_packed_opts(lin, spec)
    if bool(_env("PREFILL_Q4K_PACKED_LOAD", 1)):
      output_layout = "partials"
      if parts == 1 and bool(_env("PREFILL_DIRECT_OUT", 1)):
        output_layout = "reduce_out" if bool(_env("PREFILL_Q4K_REDUCE_OUT", 0)) else "direct_out"
      q4_spec = qk_ops.describe_q4k_packed_prefill_generated(spec.n, spec.k, spec.m,
                                                             role=_direct_packed_role(lin, spec), parts=parts,
                                                             output_layout=output_layout, opts=opts)
      if output_layout in ("direct_out", "reduce_out"):
        out = Tensor.empty(spec.m, spec.n, dtype=dtypes.float32, device=x.device).custom_kernel(
          words, x_batch.reshape(spec.m * spec.k), fxn=qk_ops.emit_q4k_packed_prefill_kernel(q4_spec))[0]
        return out.reshape(1, spec.m, spec.n)
      out = partials.custom_kernel(words, x_batch.reshape(spec.m * spec.k),
        fxn=qk_ops.emit_q4k_packed_prefill_kernel(q4_spec))[0]
    else:
      kernel = qk_ops.q4k_gemm_kernel
      out = partials.custom_kernel(words, x_batch.reshape(spec.m * spec.k),
        fxn=kernel(spec.n, spec.k, spec.m, parts, "prefill", opts, name=spec.q4k_kernel_prefix))[0]
    return out.sum(axis=2).transpose(0, 1).reshape(1, spec.m, spec.n)


@dataclass(frozen=True)
class Q6KDirectPackedPrefillCandidate(DirectPackedPrefillCandidate):
  quant: str = "q6k"

  def run(self, lin, x:Tensor, x_batch:Tensor, spec:PrefillLinearRouteSpec) -> Tensor | None:
    halfs = lin.prefill_packed_weight().to(x.device)
    parts = _direct_packed_parts(lin, spec)
    partials = Tensor.empty(spec.n, spec.m, parts, dtype=dtypes.float32, device=x.device)
    opts = _direct_packed_opts(lin, spec)
    if bool(_env("PREFILL_Q6K_PACKED_LOAD", 1)):
      output_layout = "direct_out" if parts == 1 and bool(_env("PREFILL_DIRECT_OUT", 1)) else "partials"
      q6_spec = qk_ops.describe_q6k_packed_prefill(spec.n, spec.k, spec.m, role=_direct_packed_role(lin, spec),
                                                   parts=parts, output_layout=output_layout, opts=opts)
      if output_layout == "direct_out":
        out = Tensor.empty(spec.m, spec.n, dtype=dtypes.float32, device=x.device).custom_kernel(
          halfs, x_batch.reshape(spec.m * spec.k), fxn=qk_ops.emit_q6k_packed_prefill_kernel(q6_spec))[0]
        return out.reshape(1, spec.m, spec.n)
      out = partials.custom_kernel(halfs, x_batch.reshape(spec.m * spec.k),
        fxn=qk_ops.emit_q6k_packed_prefill_kernel(q6_spec))[0]
    else:
      kernel = qk_ops.q6k_gemm_kernel
      out = partials.custom_kernel(halfs, x_batch.reshape(spec.m * spec.k),
        fxn=kernel(spec.n, spec.k, spec.m, parts, opts, name=spec.q6k_kernel_prefix))[0]
    return out.sum(axis=2).transpose(0, 1).reshape(1, spec.m, spec.n)


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
  requested_ubatch = int(_env("PREFILL_UBATCH", 512)) if ubatch is None else int(ubatch)
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
  if bool(_env("PREFILL_DIRECT_REQUIRE_UBATCH", 1)) and m != _env("PREFILL_UBATCH", 512): return None
  quant = "q4k" if _is_q4k_linear(lin) else "q6k" if _is_q6k_linear(lin) else ""
  if quant == "": return None
  if not _direct_packed_enabled_for(lin, "Q4_K" if quant == "q4k" else "Q6_K"): return None
  return PrefillLinearRouteSpec("direct_packed", quant, _direct_packed_module_role(lin), m, n, k)


def route_direct_packed_prefill(lin, x:Tensor) -> Tensor | None:
  global _MMQ_DS4_LAST_PACKED
  spec = _direct_packed_spec(lin, x)
  if spec is None: return None
  x_batch = x[0].cast(dtypes.float16).contiguous()
  if _is_q4k_linear(lin):
    role = _direct_packed_role(lin, spec)
    if bool(_env("PREFILL_QK_GENERATED_TILE", 0)):
      raise RuntimeError("PREFILL_QK_GENERATED_TILE was retired after the generated packed-tile route was refuted; "
                         "use the Q4KPrefillRouteSpec direct-packed default or PREFILL_Q4K_Q8=wmma_tiled research.")
    q8_mode = prefill_q4k_q8_mode()
    if q8_mode:
      words = lin.prefill_packed_weight().to(x.device)
      if q8_mode == "wmma":
        xq, xscales = qk_ops.q8_1_quantize(x_batch.cast(dtypes.float32))
        wmma_spec = qk_ops.describe_q4k_int8_wmma_prefill(spec.n, spec.k, spec.m, role=role,
                                                          n_tile=max(16, int(_env("PREFILL_Q4K_WMMA_N_TILE", 256))))
        raw_elems = wmma_spec.groups * wmma_spec.m * wmma_spec.n
        raw_limit = int(_env("PREFILL_Q4K_WMMA_MAX_RAW_ELEMS", 64 * 1024 * 1024))
        if raw_elems > raw_limit and not bool(_env("PREFILL_Q4K_WMMA_ALLOW_GRAPH_EXPLOSION", 0)):
          raise RuntimeError(f"PREFILL_Q4K_Q8=wmma Tensor-substrate blocked for full-model shape "
                             f"role={role or '?'} m={spec.m} n={spec.n} k={spec.k}: RAW groups*m*n={raw_elems} "
                             f"> limit={raw_limit}. This parity/codegen substrate is correct, but 14B authority "
                             f"needs the next fused/tiled generated emitter, not many Tensor matmul graph fragments. "
                             f"Set PREFILL_Q4K_WMMA_ALLOW_GRAPH_EXPLOSION=1 only for debugging.")
        out = qk_ops.emit_q4k_int8_wmma_prefill_tensor(words, xq, xscales, wmma_spec)
        return out.reshape(1, spec.m, spec.n)
      if q8_mode == "wmma_tiled":
        xq, xscales = qk_ops.q8_1_quantize(x_batch.cast(dtypes.float32))
        tiled_spec = qk_ops.describe_q4k_int8_wmma_tiled_prefill(
          spec.n, spec.k, spec.m, role=role,
          m_tile=max(16, int(_env("PREFILL_Q4K_WMMA_TILED_M_TILE", 16))),
          n_tile=max(16, int(_env("PREFILL_Q4K_WMMA_TILED_N_TILE", 16))),
          group_tile=max(1, int(_env("PREFILL_Q4K_WMMA_TILED_GROUP_TILE", 1))))
        try:
          out = qk_ops.emit_q4k_int8_wmma_tiled_prefill_tensor(words, xq, xscales, tiled_spec)
        except NotImplementedError:
          out = qk_ops.emit_q4k_int8_wmma_tiled_scheduler_tensor(words, xq, xscales, tiled_spec)
        return out.reshape(1, spec.m, spec.n)
      if q8_mode == "packed_ds4":
        candidate = qk_ops.packed_ds4_candidate(spec.m, spec.n, spec.k, role=role)
        source = x_batch.reshape(spec.m, spec.k)
        cache_key = (getattr(x, "uop", x), spec.m, spec.k, str(x.device))
        if _MMQ_DS4_LAST_PACKED is not None and _MMQ_DS4_LAST_PACKED[0] == cache_key:
          values, scales, sums = _MMQ_DS4_LAST_PACKED[1]
        else:
          values, scales, sums = qk_ops.pack_q8_1_mmq_ds4(source, candidate)
          _MMQ_DS4_LAST_PACKED = (cache_key, (values, scales, sums))
        out = qk_ops.emit_q4k_q8_mmq_ds4(words, values, scales, sums, candidate)
        return out.reshape(1, spec.m, spec.n)
      raise RuntimeError(f"PREFILL_Q4K_Q8={q8_mode!r} matched no generated route; the handwritten sdot4/MMQ/Q8_1-GEMM "
                         f"modes were deleted 2026-07-06. Only generated modes or off-values are valid.")
  candidate = select_direct_packed_prefill_candidate(lin, spec)
  if candidate is None: return None
  return candidate.run(lin, x, x_batch, spec)


def route_prefill_linear(lin, x:Tensor, *, prefill_graph_gemm:bool) -> Tensor:
  route = prefill_route_policy()
  w = getattr(lin, "_pf16_w", None)

  if route == "direct_packed" or (route == "auto" and w is None and is_direct_packed_prefill_linear(lin)):
    routed = route_direct_packed_prefill(lin, x)
    if routed is not None: return routed
    if route == "direct_packed" and prefill_route_strict():
      raise RuntimeError(f"PREFILL_ROUTE=direct_packed did not bind for {getattr(lin, 'name', type(lin).__name__)}")

  if prefill_graph_gemm and w is not None:
    routed = qk_ops.route_pf16_graph_gemm(lin, x)
    if routed is not None: return routed
  if w is None: w = lin.weight.cast(dtypes.float16)
  b = getattr(lin, "bias", None)
  return x.cast(dtypes.float16).linear(w.transpose(), b.cast(dtypes.float16) if b is not None else None)
