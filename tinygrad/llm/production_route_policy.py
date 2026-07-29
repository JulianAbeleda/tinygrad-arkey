"""Master-owned facts for selecting current LLM production route identities.

These are runtime selection facts, deliberately separate from experimental
emitters, search tooling, and handwritten dev oracles.  They preserve the
currently selected configurations but do not imply that a plan was generated
autonomously or that it has an artifact catalog entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


GENERIC_ROUTE_ID = "ordinary_tinygrad_graph"


@dataclass(frozen=True)
class ProductionRouteRequest:
  workload: str
  role: str
  quant: str
  target_backend: str
  target_architecture: str
  shape: Mapping[str, int]


@dataclass(frozen=True)
class ProductionRoute:
  route_id: str
  workload: str
  roles: tuple[str, ...]
  quant_formats: tuple[str, ...]
  target_backend: str
  target_architecture: str
  selected_config: tuple[tuple[str, object], ...]

  def config(self) -> dict[str, object]: return dict(self.selected_config)


def _config(**values: object) -> tuple[tuple[str, object], ...]: return tuple(values.items())


# These values mirror the runtime-selected route identities/configurations,
# not an inferred search population.  Wildcard dense roles remain guarded by
# the runtime integration until those call sites are migrated to this policy.
PRODUCTION_ROUTES: tuple[ProductionRoute, ...] = (
  ProductionRoute("decode_q4k_g3_generated", "decode", ("ffn_gate_up", "ffn_down", "attn_qo"), ("Q4_K",),
                  "AMD", "gfx1100", _config(variant="g3_lanemap", wave_size=32)),
  ProductionRoute("decode_q6k_coop_generated", "decode", ("ffn_down", "lm_head", "attn_v"), ("Q6_K",),
                  "AMD", "gfx1100", _config(variant="coop_or_partial", wave_size=32)),
  ProductionRoute("decode_flash_live_split_g4_kvboth", "decode", ("attention_tile", "attention_combine"), ("fp16",),
                  "AMD", "gfx1100", _config(B=1, Hq=32, Hkv=8, Hd=128, split_size=48, staging="KV_BOTH", stage_width=1)),
  ProductionRoute("decode_flash_live_split_g5_kvboth", "decode", ("attention_tile", "attention_combine"), ("fp16",),
                  "AMD", "gfx1100", _config(B=1, Hq=40, Hkv=8, Hd=128, split_size=32, query_group_size=2, staging="KV_BOTH", stage_width=4)),
  ProductionRoute("prefill_wmma_lds_dbuf_generated", "prefill", ("attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"), ("fp16",),
                  "AMD", "gfx1100", _config(M=512, transport="lds", buffer_count=2)),
  ProductionRoute("prefill_q4k_direct_tile4x4_default", "prefill", ("attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"), ("Q4_K",),
                  "AMD", "gfx1100", _config(M=512, output="direct_packed")),
  ProductionRoute("prefill_q6k_direct_generated", "prefill", ("attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"), ("Q6_K",),
                  "AMD", "gfx1100", _config(M=512, output="direct_packed")),
  ProductionRoute("packed_wmma_prefill_generated", "prefill", ("attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"), ("Q4_K", "Q6_K"),
                  "AMD", "gfx1100", _config(M=512, instruction="wmma", selected_rows=6)),
  ProductionRoute("prefill_flash_attention_generated", "prefill", ("attention_tile",), ("fp16",),
                  "AMD", "gfx1100", _config(B=1, Hkv=8, Hd=128, q_tokens=512, causal=True)),
)

ROUTES_BY_ID = {route.route_id: route for route in PRODUCTION_ROUTES}


def _matches_shape(route: ProductionRoute, request: ProductionRouteRequest) -> bool:
  shape = request.shape
  config = route.config()
  if route.route_id == "decode_q4k_g3_generated":
    return (shape.get("K"), shape.get("N")) in {(4096, 12288), (12288, 4096), (4096, 4096)}
  if route.route_id == "decode_flash_live_split_g4_kvboth":
    return all(shape.get(k) == config[k] for k in ("B", "Hq", "Hkv", "Hd")) and shape.get("context", 0) >= 512
  if route.route_id == "decode_flash_live_split_g5_kvboth":
    return all(shape.get(k) == config[k] for k in ("B", "Hq", "Hkv", "Hd")) and shape.get("context", 0) >= 512
  if route.route_id == "prefill_flash_attention_generated":
    return (shape.get("B") == 1 and shape.get("Hq") in (32, 40) and shape.get("Hkv") == 8 and
            shape.get("Hd") == 128 and shape.get("q_tokens") == 512 and shape.get("kv_tokens", 0) >= 512)
  if route.route_id == "packed_wmma_prefill_generated":
    covered = {("Q4_K", "attn_qo", 512, 5120, 5120), ("Q4_K", "attn_kv", 512, 1024, 5120),
               ("Q4_K", "ffn_gate_up", 512, 17408, 5120), ("Q4_K", "ffn_down", 512, 5120, 17408),
               ("Q6_K", "attn_kv", 512, 1024, 5120), ("Q6_K", "ffn_down", 512, 5120, 17408)}
    return (request.quant, request.role, shape.get("M"), shape.get("N"), shape.get("K")) in covered
  return shape.get("M") == 512


def select_current_route(request: ProductionRouteRequest) -> ProductionRoute | None:
  """Return the exact current route metadata, or ``None`` for generic fallback."""
  # Packed-WMMA is the current selected fast path for its exact six
  # quant/role/shape combinations; direct-packed remains the next selected
  # configuration and the fallback for all other M=512 packed linears.
  for route in sorted(PRODUCTION_ROUTES, key=lambda row: row.route_id != "packed_wmma_prefill_generated"):
    if (route.workload == request.workload and request.role in route.roles and request.quant in route.quant_formats and
        route.target_backend == request.target_backend and route.target_architecture == request.target_architecture and
        _matches_shape(route, request)):
      return route
  return None


__all__ = ["GENERIC_ROUTE_ID", "PRODUCTION_ROUTES", "ROUTES_BY_ID", "ProductionRoute", "ProductionRouteRequest", "select_current_route"]
