from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from tinygrad import Tensor, dtypes
from tinygrad.llm import route_ops as qk_ops

PREFILL_ROUTE_CHOICES = ("auto", "fp16", "direct_packed", "chunked")


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
  name = str(getattr(lin, "name", ""))
  if any(x in name for x in ("ffn_gate", "ffn_up")): return "ffn_gate_up"
  if "ffn_down" in name: return "ffn_down"
  if any(x in name for x in ("attn_q", "attn_output")): return "attn_qo"
  if any(x in name for x in ("attn_k", "attn_v")): return "attn_kv"
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


def prefill_route_wants_resident_fp16(*, est_gb:float, budget_gb:float, has_direct_packed:bool, prefill_chunked:bool) -> bool:
  route = prefill_route_policy()
  if route == "fp16": return True
  if route in ("direct_packed", "chunked") or prefill_chunked: return False
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


def _direct_packed_spec(lin, x:Tensor) -> PrefillLinearRouteSpec | None:
  if getattr(lin, "bias", None) is not None or len(x.shape) != 3 or x.shape[0] != 1: return None
  m, k = x.shape[-2], x.shape[-1]
  n, in_f = getattr(lin, "out_features", None), getattr(lin, "in_features", None)
  if not all(isinstance(v, int) for v in (m, k, n, in_f)) or k != in_f: return None
  if bool(_env("PREFILL_DIRECT_REQUIRE_UBATCH", 1)) and m != _env("PREFILL_UBATCH", 512): return None
  quant = "q4k" if _is_q4k_linear(lin) else "q6k" if _is_q6k_linear(lin) else ""
  if quant == "": return None
  if not _direct_packed_enabled_for(lin, "Q4_K" if quant == "q4k" else "Q6_K"): return None
  return PrefillLinearRouteSpec("direct_packed", quant, str(getattr(lin, "_prefill_graph_role", "")), m, n, k)


def route_direct_packed_prefill(lin, x:Tensor) -> Tensor | None:
  spec = _direct_packed_spec(lin, x)
  if spec is None: return None
  x_batch = x[0].cast(dtypes.float16).contiguous()
  parts = _direct_packed_parts(lin, spec)
  if _is_q4k_linear(lin):
    words = lin.prefill_packed_weight().to(x.device)
    partials = Tensor.empty(spec.n, spec.m, parts, dtype=dtypes.float32, device=x.device)
    opts = _direct_packed_opts(lin, spec)
    generated_tile_on = bool(_env("PREFILL_QK_GENERATED_TILE", 0))
    generated_tile_roles = _csv_set("PREFILL_QK_GENERATED_TILE_ROLES", "ffn_gate_up")
    role = _direct_packed_role(lin, spec)
    if generated_tile_on and (not generated_tile_roles or role in generated_tile_roles):
      try:
        tile_mode = str(os.environ.get("PREFILL_QK_GENERATED_TILE_MODE", "lane_partials")).strip().lower()
        default_rows, default_tokens = (1, 4) if tile_mode == "direct_warp" else (4, 8)
        tile_spec = qk_ops.describe_q4k_packed_prefill_tile(
          spec.n, spec.k, spec.m, role=role,
          row_tile=max(1, int(_env("PREFILL_QK_GENERATED_TILE_ROWS", default_rows))),
          token_tile=max(1, int(_env("PREFILL_QK_GENERATED_TILE_TOKENS", default_tokens))),
          output_layout=tile_mode)
      except Exception:
        if prefill_route_strict(): raise
      else:
        if tile_spec.output_layout == "direct_warp":
          out = Tensor.empty(spec.m, spec.n, dtype=dtypes.float32, device=x.device).custom_kernel(
            words, x_batch.reshape(spec.m * spec.k), fxn=qk_ops.emit_q4k_packed_prefill_tile(tile_spec))[0]
          return out.reshape(1, spec.m, spec.n)
        else:
          tile_partials = Tensor.empty(spec.n, spec.m, 8, dtype=dtypes.float32, device=x.device)
          out = tile_partials.custom_kernel(words, x_batch.reshape(spec.m * spec.k),
            fxn=qk_ops.emit_q4k_packed_prefill_tile(tile_spec))[0]
          return out.sum(axis=2).transpose(0, 1).reshape(1, spec.m, spec.n)
    q8_mode = str(os.environ.get("PREFILL_Q4K_Q8", "")).strip().lower()
    if q8_mode and q8_mode not in ("0", "false", "off", "no"):
      xq, xscales = qk_ops.q8_1_quantize(x_batch.cast(dtypes.float32))
      if q8_mode == "mmq":
        xq_words = Tensor.empty(xq.numel() // 4, dtype=dtypes.uint32, device=x.device).custom_kernel(
          xq, fxn=qk_ops.q8_signed_pack_u32_kernel(spec.m * spec.k))[0]
        mmq_partials = Tensor.empty(spec.n, spec.m, 8, dtype=dtypes.float32, device=x.device)
        out = mmq_partials.custom_kernel(words, xq_words, xscales,
          fxn=qk_ops.q4k_q8_1_sdot4_coop_gemm_kernel(spec.n, spec.k, spec.m, 1, 1,
                                                     name="prefill_q4k_q8_1_mmq_direct_packed_gemm"))[0]
        return out.sum(axis=2).transpose(0, 1).reshape(1, spec.m, spec.n)
      if q8_mode == "sdot4":
        if parts != 1:
          if prefill_route_strict(): raise RuntimeError("PREFILL_Q4K_Q8=sdot4 requires parts=1")
        else:
          xq_words = Tensor.empty(xq.numel() // 4, dtype=dtypes.uint32, device=x.device).custom_kernel(
            xq, fxn=qk_ops.q8_signed_pack_u32_kernel(spec.m * spec.k))[0]
          out = partials.custom_kernel(words, xq_words, xscales,
            fxn=qk_ops.q4k_q8_1_sdot4_gemm_kernel(spec.n, spec.k, spec.m, parts, "none", (),
                                                  name="prefill_q4k_q8_1_sdot4_direct_packed_gemm"))[0]
          return out.sum(axis=2).transpose(0, 1).reshape(1, spec.m, spec.n)
      out = partials.custom_kernel(words, xq, xscales,
        fxn=qk_ops.q4k_q8_1_gemm_kernel(spec.n, spec.k, spec.m, parts, "prefill", opts,
                                        name="prefill_q4k_q8_1_direct_packed_gemm"))[0]
    else:
      kernel = qk_ops.q4k_gemm_packed_load_kernel if bool(_env("PREFILL_Q4K_PACKED_LOAD", 1)) else qk_ops.q4k_gemm_kernel
      if parts == 1 and bool(_env("PREFILL_DIRECT_OUT", 1)) and bool(_env("PREFILL_Q4K_PACKED_LOAD", 1)):
        out = Tensor.empty(spec.m, spec.n, dtype=dtypes.float32, device=x.device).custom_kernel(
          words, x_batch.reshape(spec.m * spec.k),
          fxn=qk_ops.q4k_gemm_packed_load_direct_out_kernel(spec.n, spec.k, spec.m, "prefill", opts,
                                                            name="prefill_q4k_direct_packed_load_direct_out_gemm"))[0]
        return out.reshape(1, spec.m, spec.n)
      out = partials.custom_kernel(words, x_batch.reshape(spec.m * spec.k),
        fxn=kernel(spec.n, spec.k, spec.m, parts, "prefill", opts, name=spec.q4k_kernel_prefix))[0]
  else:
    halfs = lin.prefill_packed_weight().to(x.device)
    partials = Tensor.empty(spec.n, spec.m, parts, dtype=dtypes.float32, device=x.device)
    opts = _direct_packed_opts(lin, spec)
    kernel = qk_ops.q6k_gemm_packed_load_kernel if bool(_env("PREFILL_Q6K_PACKED_LOAD", 1)) else qk_ops.q6k_gemm_kernel
    if parts == 1 and bool(_env("PREFILL_DIRECT_OUT", 1)) and bool(_env("PREFILL_Q6K_PACKED_LOAD", 1)):
      out = Tensor.empty(spec.m, spec.n, dtype=dtypes.float32, device=x.device).custom_kernel(
        halfs, x_batch.reshape(spec.m * spec.k),
        fxn=qk_ops.q6k_gemm_packed_load_direct_out_kernel(spec.n, spec.k, spec.m, opts,
                                                          name="prefill_q6k_direct_packed_load_direct_out_gemm"))[0]
      return out.reshape(1, spec.m, spec.n)
    out = partials.custom_kernel(halfs, x_batch.reshape(spec.m * spec.k),
      fxn=kernel(spec.n, spec.k, spec.m, parts, opts, name=spec.q6k_kernel_prefix))[0]
  return out.sum(axis=2).transpose(0, 1).reshape(1, spec.m, spec.n)


def route_prefill_linear(lin, x:Tensor, *, prefill_graph_gemm:bool, prefill_chunked:bool) -> Tensor:
  route = prefill_route_policy()
  w = getattr(lin, "_pf16_w", None)

  if route == "direct_packed" or (route == "auto" and w is None and is_direct_packed_prefill_linear(lin)):
    routed = route_direct_packed_prefill(lin, x)
    if routed is not None: return routed
    if route == "direct_packed" and prefill_route_strict():
      raise RuntimeError(f"PREFILL_ROUTE=direct_packed did not bind for {getattr(lin, 'name', type(lin).__name__)}")

  if route == "chunked": prefill_chunked = True
  if prefill_chunked and w is None:
    if is_direct_packed_prefill_linear(lin): w_local = lin.prefill_fp16_weight()
    else: w_local = lin.weight.cast(dtypes.float16).contiguous()
    if prefill_graph_gemm:
      routed = qk_ops.route_pf16_graph_gemm(lin, x, w=w_local)
      if routed is not None: return routed
    b = getattr(lin, "bias", None)
    return x.cast(dtypes.float16).linear(w_local.transpose(), b.cast(dtypes.float16) if b is not None else None)

  if prefill_graph_gemm and w is not None:
    routed = qk_ops.route_pf16_graph_gemm(lin, x)
    if routed is not None: return routed
  if w is None: w = lin.weight.cast(dtypes.float16)
  b = getattr(lin, "bias", None)
  return x.cast(dtypes.float16).linear(w.transpose(), b.cast(dtypes.float16) if b is not None else None)
