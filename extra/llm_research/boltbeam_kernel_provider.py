"""Fail-closed tinygrad emitter registry for canonical Boltbeam kernel candidates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from typing import Any, Callable

KERNEL_CANDIDATE_SCHEMA = "boltbeam.kernel_candidate.v1"
KERNEL_ROUTE_SCHEMA = "boltbeam.kernel_route.v1"


def _canonical(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def candidate_hash(candidate: dict[str, Any]) -> str:
  return hashlib.sha256(_canonical(candidate).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class GeneratedKernel:
  candidate_hash: str
  kernel_id: str
  family: str
  kind: str
  artifact: Any
  launch: dict[str, Any]


Emitter = Callable[[dict[str, Any], str], GeneratedKernel]
_EMITTERS: dict[str, Emitter] = {}


def register_emitter(family: str):
  def decorator(fn: Emitter) -> Emitter:
    if family in _EMITTERS: raise RuntimeError(f"duplicate Boltbeam emitter family {family!r}")
    _EMITTERS[family] = fn
    return fn
  return decorator


def _require_exact(candidate: dict[str, Any], expected: dict[str, Any]) -> None:
  actual = candidate.get("parameters")
  if actual != expected:
    missing = sorted(set(expected)-set(actual or {})) if isinstance(actual, dict) else sorted(expected)
    unknown = sorted(set(actual or {})-set(expected)) if isinstance(actual, dict) else []
    raise ValueError(f"Boltbeam family parameters differ from the admitted contract; missing={missing}, unknown={unknown}")


def _validate_candidate(candidate: Any, claimed_hash: str | None) -> tuple[dict[str, Any], str]:
  if not isinstance(candidate, dict) or candidate.get("schema_version") != KERNEL_CANDIDATE_SCHEMA:
    raise ValueError("unsupported Boltbeam kernel candidate")
  required = {"schema_version", "kernel_id", "family", "phase", "role", "operation", "shape", "abi", "parameters",
              "launch", "resources", "target", "correctness", "provenance"}
  if set(candidate) != required: raise ValueError("Boltbeam kernel candidate fields do not match the v1 contract")
  identity = candidate_hash(candidate)
  if claimed_hash is not None and claimed_hash != identity: raise ValueError("Boltbeam candidate hash mismatch")
  return json.loads(_canonical(candidate)), identity


@register_emitter("packed_q6_q8_streamk_main.v1")
def _q6_main(candidate: dict[str, Any], identity: str) -> GeneratedKernel:
  expected = {"tile_m": 128, "tile_n": 128, "tile_k": 256, "streamk_owners": 170,
    "streamk_segment": 0, "streamk_segments_in_cta": True, "segment_order": "ascending",
    "prefetch_second_panel": True, "combined_initial_publish": True, "factor_dA": False,
    "oracle_publisher": True, "weight_scale_contract": "trusted_fp16_packed",
    "partial_output_layout": "destination_major"}
  _require_exact(candidate, expected)
  if candidate["shape"] != {"m": 512, "n": 4096, "k": 12288}: raise ValueError("unsupported Q6 main shape")
  if candidate["launch"] != {"grid": [170, 1, 1], "block": [256, 1, 1], "dynamic_shared_memory_bytes": 58368}:
    raise ValueError("unsupported Q6 main launch")
  from extra.llm_research.prefill.nv_q6_oracle_reduction_policy import build_packed_one_body_ast
  ast = build_packed_one_body_ast(segment_order=expected["segment_order"],
                                  partial_output_layout=expected["partial_output_layout"])
  return GeneratedKernel(identity, candidate["kernel_id"], candidate["family"], "uop", ast, candidate["launch"])


@register_emitter("q6_destination_major_fixup.v1")
def _q6_fixup(candidate: dict[str, Any], identity: str) -> GeneratedKernel:
  expected = {"rows": 128, "cols": 128, "tiles_m": 4, "tiles_n": 32, "slices": 4, "threads": 128,
              "outputs_per_thread": 32, "fold_order": "slot0_slot1_slot2", "fadd": "rn"}
  _require_exact(candidate, expected)
  if candidate["shape"] != {"m": 512, "n": 4096, "contributors": 3}: raise ValueError("unsupported Q6 fixup shape")
  if candidate["launch"] != {"grid": [128, 4, 1], "block": [128, 1, 1], "dynamic_shared_memory_bytes": 0}:
    raise ValueError("unsupported Q6 fixup launch")
  from extra.llm_research.prefill.nv_q6_destination_partial import destination_major_fixup_source
  return GeneratedKernel(identity, candidate["kernel_id"], candidate["family"], "source",
                         destination_major_fixup_source(), candidate["launch"])


def generate_kernel(candidate: dict[str, Any], claimed_hash: str | None = None) -> GeneratedKernel:
  candidate, identity = _validate_candidate(candidate, claimed_hash)
  family = candidate["family"]
  emitter = _EMITTERS.get(family)
  if emitter is None: raise ValueError(f"no tinygrad emitter registered for Boltbeam family {family!r}")
  return emitter(candidate, identity)


def generate_route(route: dict[str, Any], claimed_hash: str | None = None) -> tuple[GeneratedKernel, ...]:
  if not isinstance(route, dict) or route.get("schema_version") != KERNEL_ROUTE_SCHEMA: raise ValueError("unsupported Boltbeam route")
  route_identity = hashlib.sha256(_canonical(route).encode("ascii")).hexdigest()
  if claimed_hash is not None and claimed_hash != route_identity: raise ValueError("Boltbeam route hash mismatch")
  components = route.get("components")
  if not isinstance(components, list) or not components: raise ValueError("Boltbeam route has no components")
  seen: set[str] = set(); generated = []
  for row in components:
    if not isinstance(row, dict) or set(row) != {"name", "candidate_hash", "candidate", "depends_on"}:
      raise ValueError("invalid Boltbeam route component")
    if not isinstance(row["depends_on"], list) or any(dep not in seen for dep in row["depends_on"]):
      raise ValueError("Boltbeam route dependency is missing or forward")
    if row["name"] in seen: raise ValueError("duplicate Boltbeam route component")
    generated.append(generate_kernel(row["candidate"], row["candidate_hash"])); seen.add(row["name"])
  outputs = route.get("outputs")
  if not isinstance(outputs, list) or not outputs or any(name not in seen for name in outputs):
    raise ValueError("Boltbeam route outputs are not generated components")
  return tuple(generated)


__all__ = ["GeneratedKernel", "candidate_hash", "generate_kernel", "generate_route", "register_emitter"]
