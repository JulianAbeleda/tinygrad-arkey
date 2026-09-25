"""Promoted tensor-core candidates for dense bf16-weight projections.

Every candidate belongs to the precontract tile family of the NV sm_120 prefill schedule promoted for Qwen3-8B
(``prefill_sm120_lds_dbuf_candidate_set.json``: tile, warps, two LDS buffers).  The artifact holds one compact
candidate set per promoted (geometry) template plus a route table: for each exact (role, rows, N, K) the split-K
factor S and the canonical identity of the candidate that computes one K/S slice.  A split-K route runs the slices
as one batched GEMM over strided views (no copies) and sums the fp32 partials.

A routed projection rounds its activation to bf16, pads rows up to the nearest promoted row count (chunking above
the largest), and accumulates in fp32 on the tensor cores.  Any shape, role, or target without an exact promoted
route declines to the ordinary linear path.  Minted by ``extra/llm_research/mint_typed_candidate_template.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import json, math
from pathlib import Path
from typing import Any

from tinygrad import Tensor, Device, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.llm.prefill_candidate_runtime import CandidateAdmission, candidate_registry, expand_compact_candidate_set

DENSE_BF16_ARTIFACT = Path(__file__).with_name("generated") / "dense_bf16_sm120_candidate_set.json"
ROUTES_SCHEMA = "tinygrad.dense_bf16_candidate_routes.v1"
WEIGHT_ROW_TILE = 128   # weights are padded once to this; every promoted tile_n divides it
# The candidate owns its complete output tile; only the tensor-core opt is legal on top of it
# (docs/task_workflow/output/nv-prefill-corrected-tile-result.md).
_CANDIDATE_OPTS = (Opt(OptOps.TC, 0, (-1, 2, 1)),)
_DTYPES = {"a":"bf16", "b":"bf16", "accumulator":"fp32", "c":"fp32"}


@dataclass(frozen=True)
class DenseRoute:
  role: str
  m: int
  n: int
  k: int
  split_k: int
  admission: CandidateAdmission


def device_target(device: str) -> dict[str, Any]:
  renderer = Device[device].renderer
  threads = {tc.threads for tc in renderer.tensor_cores}
  return {"backend":device.split(":")[0], "arch":renderer.target.arch, "wave_size":threads.pop() if len(threads) == 1 else None}


def load_routes(raw: dict, backend: str, arch: str, wave_size: int) -> dict[tuple[str, int, int, int], DenseRoute] | None:
  if raw.get("schema") != ROUTES_SCHEMA: raise ValueError("dense bf16 route artifact schema drifted")
  if raw.get("target") != {"backend":backend, "arch":arch, "wave_size":wave_size}: return None
  admissions: dict[str, CandidateAdmission] = {}
  for compact in raw["sets"]:
    if compact["target"] != raw["target"] or compact["template"]["dtypes"] != _DTYPES:
      raise ValueError("dense bf16 candidate set target or dtype contract drifted")
    for admission in candidate_registry(expand_compact_candidate_set(compact, backend, arch, wave_size)).admissions:
      admissions[admission.canonical_identity] = admission
  routes = {}
  for row in raw["routes"]:
    role, m, n, k, split = row["role"], row["m"], row["n"], row["k"], row["split_k"]
    admission = admissions.get(row["canonical_identity"])
    if admission is None: raise ValueError(f"route {role} {m}x{n}x{k} names an unknown candidate")
    workload = admission.normalized_payload["workload"]
    if (workload["role"], *(workload["shape"][x] for x in "mnk")) != (role, m, n, k // split) or k % split:
      raise ValueError(f"route {role} {m}x{n}x{k}/{split} does not match its candidate workload")
    if (key := (role, m, n, k)) in routes: raise ValueError(f"duplicate dense bf16 route {key}")
    routes[key] = DenseRoute(role, m, n, k, split, admission)
  return routes


@cache
def dense_bf16_routes(backend: str, arch: str, wave_size: int) -> dict[tuple[str, int, int, int], DenseRoute] | None:
  return load_routes(json.loads(DENSE_BF16_ARTIFACT.read_text()), backend, arch, wave_size)


def _routes_for(device: str):
  target = device_target(device)
  if target["wave_size"] is None: return None
  return dense_bf16_routes(target["backend"], target["arch"], target["wave_size"])


def plan_rows(routes, role: str, n: int, k: int, rows: int) -> list[tuple[int, int, DenseRoute]] | None:
  """(start, rows, route) chunks: each chunk pads up to the nearest promoted row count; above the largest, chunk."""
  counts = sorted(m for (r, m, rn, rk) in routes if (r, rn, rk) == (role, n, k))
  if not counts: return None
  plan, start = [], 0
  while start < rows:
    take = min(rows - start, counts[-1])
    plan.append((start, take, routes[(role, next(m for m in counts if m >= take), n, k)]))
    start += take
  return plan


def _install(admission, out_dims: set[int], reduce: int) -> None:
  import tinygrad.codegen.opt.postrange as pr
  key = pr.warmstart_key(out_dims, reduce)
  existing = (pr._WARMSTART_CANDIDATE_CONTEXTS or {}).get(key)
  if existing is not None and existing != admission.context:
    raise ValueError(f"candidate warmstart key collision for {key!r}")
  pr._WARMSTART_OPTS = {**(pr._WARMSTART_OPTS or {}), key:_CANDIDATE_OPTS}
  pr._WARMSTART_CANDIDATE_CONTEXTS = {**(pr._WARMSTART_CANDIDATE_CONTEXTS or {}), key:admission.context}


def pad_weight(weight: Tensor, device: str | None = None) -> Tensor:
  """Tile-pad a (N, K) bf16 weight's rows once; the first N rows remain a contiguous prefix view."""
  if _routes_for(device or weight.device) is None or weight.dtype != dtypes.bfloat16: return weight
  n = weight.shape[0]
  if n % WEIGHT_ROW_TILE == 0: return weight
  return weight.pad(((0, -(-n // WEIGHT_ROW_TILE) * WEIGHT_ROW_TILE - n), (0, 0))).contiguous().realize()


def _chunk(a: Tensor, weight: Tensor, route: DenseRoute) -> Tensor:
  m, (n, k), s = route.m, weight.shape, route.split_k
  if s == 1:
    _install(route.admission, {m, n}, k)
    return a.dot(weight.T, dtype=dtypes.float).contiguous()
  ks = k // s
  _install(route.admission, {s, m, n}, ks)
  a3, w3 = a.reshape(m, s, ks).permute(1, 0, 2), weight.reshape(n, s, ks).permute(1, 0, 2)
  return a3.dot(w3.transpose(1, 2), dtype=dtypes.float).contiguous().sum(0)


def route_dense_bf16(x: Tensor, weight: Tensor, role: str, n_out: int, *, min_rows: int = 1) -> Tensor | None:
  """x (..., K) @ weight[:n_out].T in fp32 through promoted candidates, or None to decline.

  ``weight`` is the (Np, K) tile-padded bf16 weight from ``pad_weight``."""
  if not isinstance(x.device, str) or weight.dtype != dtypes.bfloat16 or weight.ndim != 2 or x.ndim < 2: return None
  if not all(isinstance(s, int) for s in x.shape): return None
  routes = _routes_for(x.device)
  if routes is None: return None
  n_pad, k = weight.shape
  rows = math.prod(x.shape[:-1])
  if x.shape[-1] != k or n_out > n_pad or rows < min_rows: return None
  if (plan := plan_rows(routes, role, n_pad, k, rows)) is None: return None
  flat = x.reshape(rows, k).cast(dtypes.bfloat16)
  outs = []
  for start, take, route in plan:
    a = flat[start:start + take]
    if route.m != take: a = a.pad(((0, route.m - take), (0, 0)))
    # The padded product is its own buffer: slicing a lazy product would shrink the GEMM to the unpadded
    # (unpromoted) shape and silently fall back to the generic schedule.
    outs.append(_chunk(a.contiguous(), weight, route)[:take, :n_out])
  out = outs[0] if len(outs) == 1 else outs[0].cat(*outs[1:])
  return out.reshape(*x.shape[:-1], n_out)


def route_dense_bf16_hilo(x: Tensor, weight: Tensor, role: str, n_out: int, *, min_rows: int = 1) -> Tensor | None:
  """Near-float32 activations on the bf16 candidates: x = hi + lo + O(2**-17 |x|), both halves stacked as rows of one
  routed GEMM (the weight is read once, each bf16 x bf16 product is exact in fp32), then summed."""
  if not all(isinstance(s, int) for s in x.shape) or x.ndim < 2: return None
  flat = x.reshape(-1, x.shape[-1]).float()
  hi = flat.cast(dtypes.bfloat16)
  lo = (flat - hi.float()).cast(dtypes.bfloat16)
  rows = flat.shape[0]
  out = route_dense_bf16(hi.cat(lo), weight, role, n_out, min_rows=2 * min_rows)
  return None if out is None else (out[:rows] + out[rows:]).reshape(*x.shape[:-1], n_out)


class CandidateLinear:
  """A bias-free bf16 projection that routes through promoted candidates and otherwise calls ``fallback``.

  ``hilo`` keeps near-float32 activations (two bf16 terms); without it the activation is rounded to bf16 once.
  The weight is tile-padded once and ``fallback.weight`` becomes the unpadded prefix view of the same buffer."""
  def __init__(self, weight: Tensor, role: str, *, min_rows: int = 1, hilo: bool = False, fallback=None):
    self.out_features = weight.shape[0]
    self.padded = pad_weight(weight)
    self.weight = self.padded if self.padded is weight else self.padded[:self.out_features]
    if fallback is not None: fallback.weight = self.weight
    self.bias, self.role, self.min_rows, self.hilo, self.fallback = None, role, min_rows, hilo, fallback

  def __call__(self, x: Tensor) -> Tensor:
    route = route_dense_bf16_hilo if self.hilo else route_dense_bf16
    routed = route(x, self.padded, self.role, self.out_features, min_rows=self.min_rows)
    if routed is not None: return routed
    return self.fallback(x) if self.fallback is not None else x.linear(self.weight.transpose())


__all__ = ["DENSE_BF16_ARTIFACT", "NEMOTRON_H_ROLES", "CandidateLinear", "DenseRoute", "bind_candidate_linears", "dense_bf16_routes",
           "device_target", "load_routes", "pad_weight", "plan_rows", "route_dense_bf16", "route_dense_bf16_hilo"]


# Nemotron-H projection attribute -> promoted role.  Every bf16 projection of the 4B release is covered.
NEMOTRON_H_ROLES = {"ssm_in":"ssm_in", "ssm_out":"ssm_out", "attn_q":"attn_q", "attn_k":"attn_kv", "attn_v":"attn_kv",
                    "attn_output":"attn_o", "ffn_up":"ffn_up", "ffn_down":"ffn_down"}


def bind_candidate_linears(model, roles: dict[str, str] = NEMOTRON_H_ROLES, *, output_role: str | None = "output",
                           min_rows: int = 1, hilo: bool = False) -> int:
  """Replace each bias-free bf16 projection on model.blk[*] (and model.output) with a CandidateLinear; returns the count."""
  bound = 0
  owners = [(block, attr, role) for block in getattr(model, "blk", ()) for attr, role in roles.items()]
  if output_role is not None: owners.append((model, "output", output_role))
  for owner, attr, role in owners:
    lin = getattr(owner, attr, None)
    if lin is None or isinstance(lin, CandidateLinear) or getattr(lin, "bias", None) is not None: continue
    if getattr(lin, "weight", None) is None or lin.weight.dtype != dtypes.bfloat16: continue
    setattr(owner, attr, CandidateLinear(lin.weight, role, min_rows=min_rows, hilo=hilo, fallback=lin))
    bound += 1
  return bound
