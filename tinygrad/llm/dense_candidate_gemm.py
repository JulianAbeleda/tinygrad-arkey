"""Promoted tensor-core candidates for dense bf16-weight projections.

The NV sm_120 prefill schedule promoted for Qwen3-8B (128x128x32 tile, 4x2
warps, two LDS buffers; ``prefill_sm120_lds_dbuf_candidate_set.json``) carries
no model facts beyond its exact workloads, so the same schedule is stamped onto
another model's exact bf16 workloads and promoted as its own compact artifact
(``extra/llm_research/mint_typed_candidate_template.py --dense-bf16``).

A routed projection rounds its activation to bf16, pads rows to the tile, and
accumulates in fp32 on the tensor cores.  Rows are processed in chunks whose
padded sizes are exactly the promoted workloads; any shape, role, or target
without an exact promoted row declines to the ordinary linear path.
"""
from __future__ import annotations

from functools import cache
import json, math
from pathlib import Path
from typing import Any

from tinygrad import Tensor, Device, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.llm.prefill_candidate_runtime import CandidateRegistry, candidate_registry, expand_compact_candidate_set

DENSE_BF16_ARTIFACT = Path(__file__).with_name("generated") / "dense_bf16_sm120_candidate_set.json"
# The candidate owns its complete output tile; only the tensor-core opt is legal on top of it
# (docs/task_workflow/output/nv-prefill-corrected-tile-result.md).
_CANDIDATE_OPTS = (Opt(OptOps.TC, 0, (-1, 2, 1)),)


def device_target(device: str) -> dict[str, Any]:
  renderer = Device[device].renderer
  threads = {tc.threads for tc in renderer.tensor_cores}
  return {"backend":device.split(":")[0], "arch":renderer.target.arch, "wave_size":threads.pop() if len(threads) == 1 else None}


@cache
def dense_bf16_registry(backend: str, arch: str, wave_size: int) -> CandidateRegistry | None:
  raw = json.loads(DENSE_BF16_ARTIFACT.read_text())
  if raw.get("target") != {"backend":backend, "arch":arch, "wave_size":wave_size}: return None
  if raw["template"]["dtypes"] != {"a":"bf16", "b":"bf16", "accumulator":"fp32", "c":"fp32"}:
    raise ValueError("dense bf16 candidate artifact dtype contract drifted")
  return candidate_registry(expand_compact_candidate_set(raw, backend, arch, wave_size))


def _registry_for(device: str) -> tuple[CandidateRegistry | None, dict[str, Any]]:
  target = device_target(device)
  if target["wave_size"] is None: return None, target
  return dense_bf16_registry(target["backend"], target["arch"], target["wave_size"]), target


def _tile(registry: CandidateRegistry) -> tuple[int, int]:
  tiles = {admission.geometry.tile[:2] for admission in registry.admissions}
  if len(tiles) != 1: raise ValueError("dense bf16 candidate set mixes tile geometries")
  return tiles.pop()


def promoted_row_chunks(registry: CandidateRegistry) -> tuple[int, ...]:
  return tuple(sorted({admission.normalized_payload["workload"]["shape"]["m"] for admission in registry.admissions}))


def _install(admission, m: int, n: int, k: int) -> None:
  import tinygrad.codegen.opt.postrange as pr
  key = pr.warmstart_key({m, n}, k)
  existing = (pr._WARMSTART_CANDIDATE_CONTEXTS or {}).get(key)
  if existing is not None and existing != admission.context:
    raise ValueError(f"candidate warmstart key collision for {key!r}")
  pr._WARMSTART_OPTS = {**(pr._WARMSTART_OPTS or {}), key:_CANDIDATE_OPTS}
  pr._WARMSTART_CANDIDATE_CONTEXTS = {**(pr._WARMSTART_CANDIDATE_CONTEXTS or {}), key:admission.context}


def pad_weight(weight: Tensor, device: str | None = None) -> Tensor:
  """Tile-pad a (N, K) bf16 weight's rows once; the first N rows remain a contiguous prefix view."""
  registry, _ = _registry_for(device or weight.device)
  if registry is None or weight.dtype != dtypes.bfloat16: return weight
  tile_n = _tile(registry)[1]
  n = weight.shape[0]
  if n % tile_n == 0: return weight
  return weight.pad(((0, -(-n // tile_n) * tile_n - n), (0, 0))).contiguous().realize()


def route_dense_bf16(x: Tensor, weight: Tensor, role: str, n_out: int, *, min_rows: int = 1) -> Tensor | None:
  """x (..., K) @ weight[:n_out].T in fp32 through promoted candidates, or None to decline.

  ``weight`` is the (Np, K) tile-padded bf16 weight from ``pad_weight``."""
  if not isinstance(x.device, str) or weight.dtype != dtypes.bfloat16 or weight.ndim != 2 or x.ndim < 2: return None
  if not all(isinstance(s, int) for s in x.shape): return None
  registry, target = _registry_for(x.device)
  if registry is None: return None
  n_pad, k = weight.shape
  rows = math.prod(x.shape[:-1])
  if x.shape[-1] != k or n_out > n_pad or rows < min_rows: return None
  tile_m = _tile(registry)[0]
  chunk = max(promoted_row_chunks(registry))
  plan = []
  for start in range(0, rows, chunk):
    m = min(rows, start + chunk) - start
    m_pad = -(-m // tile_m) * tile_m
    admission = registry.get(role, (m_pad, n_pad, k), target)
    if admission is None: return None
    plan.append((start, m, m_pad, admission))
  flat = x.reshape(rows, k).cast(dtypes.bfloat16)
  outs = []
  for start, m, m_pad, admission in plan:
    _install(admission, m_pad, n_pad, k)
    a = flat[start:start + m]
    if m_pad != m: a = a.pad(((0, m_pad - m), (0, 0)))
    # The padded product is its own buffer: slicing a lazy product would shrink the GEMM to the unpadded
    # (unpromoted) shape and silently fall back to the generic schedule.
    outs.append(a.contiguous().dot(weight.T, dtype=dtypes.float).contiguous()[:m, :n_out])
  out = outs[0] if len(outs) == 1 else outs[0].cat(*outs[1:])
  return out.reshape(*x.shape[:-1], n_out)


class CandidateLinear:
  """A bias-free bf16 projection that routes through promoted candidates and otherwise stays plain."""
  def __init__(self, weight: Tensor, role: str, *, min_rows: int = 1):
    self.out_features = weight.shape[0]
    self.padded = pad_weight(weight)
    self.weight = self.padded if self.padded is weight else self.padded[:self.out_features]
    self.bias, self.role, self.min_rows = None, role, min_rows

  def __call__(self, x: Tensor) -> Tensor:
    routed = route_dense_bf16(x, self.padded, self.role, self.out_features, min_rows=self.min_rows)
    return routed if routed is not None else x.linear(self.weight.transpose())


__all__ = ["DENSE_BF16_ARTIFACT", "NEMOTRON_H_ROLES", "CandidateLinear", "bind_candidate_linears", "dense_bf16_registry", "device_target", "pad_weight",
           "promoted_row_chunks", "route_dense_bf16"]


# Nemotron-H projection attribute -> promoted role.  Every bf16 projection of the 4B release is covered.
NEMOTRON_H_ROLES = {"ssm_in":"ssm_in", "ssm_out":"ssm_out", "attn_q":"attn_q", "attn_k":"attn_kv", "attn_v":"attn_kv",
                    "attn_output":"attn_o", "ffn_up":"ffn_up", "ffn_down":"ffn_down"}


def bind_candidate_linears(model, roles: dict[str, str] = NEMOTRON_H_ROLES, *, output_role: str | None = "output",
                           min_rows: int = 1) -> int:
  """Replace each bias-free bf16 projection on model.blk[*] (and model.output) with a CandidateLinear; returns the count."""
  bound = 0
  owners = [(block, attr, role) for block in getattr(model, "blk", ()) for attr, role in roles.items()]
  if output_role is not None: owners.append((model, "output", output_role))
  for owner, attr, role in owners:
    lin = getattr(owner, attr, None)
    if lin is None or isinstance(lin, CandidateLinear) or getattr(lin, "bias", None) is not None: continue
    if getattr(lin, "weight", None) is None or lin.weight.dtype != dtypes.bfloat16: continue
    setattr(owner, attr, CandidateLinear(lin.weight, role, min_rows=min_rows))
    bound += 1
  return bound
