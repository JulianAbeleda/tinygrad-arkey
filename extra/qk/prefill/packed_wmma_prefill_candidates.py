"""Production packed-WMMA prefill candidates.

Wraps the PROVEN scratch packed-WMMA route (real Qwen3-14B pp512 ~1851 tok/s, 99.4%+ of
llama.cpp, see scratch bench e2e_packed_wmma_bench_q6_nopad.py) as candidate classes
selectable through the normal prefill dispatcher (tinygrad/llm/prefill_routes.py ->
route_prefill_linear / route_packed_wmma_prefill), instead of the scratch
`prefill_route_override` research hook.

Default is ON: tinygrad.llm.prefill_routes.packed_wmma_prefill_enabled() gates every
entry point. Any ungated or unknown (quant, role, shape) combo silently declines, and
the caller falls through to the direct-packed baseline. Set TINYGRAD_PREFILL_PACKED_WMMA=0
to disable and use direct-packed only.

Geometry table: FROZEN. Each (quant, role) entry was independently correctness-gated and
throughput-measured in the scratch bench; changing a tile here requires re-verifying both
before promotion. See e2e_packed_wmma_bench_q6_nopad.py GEOM (lines ~56-75) for the original
measurements this table was copied from.

Correctness gate: each (quant, role, shape) combo is gated EXACTLY ONCE per process (cached
in _GATE_CACHE) via the shared isolated correctness canary
(extra/qk/prefill/packed_wmma_correctness_canary.build_artifact/run_canary) before its
warmstart schedule entry is ever installed. A combo that fails, errors, or is not yet gated
declines (returns None from `run` / is omitted from the warmstart table) -- the caller then
falls through to the production direct-packed route. Nothing here is EVER dispatched ungated.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from tinygrad import Tensor, dtypes

# Verified-correct geometries, keyed by (quant, role). Copied verbatim from the scratch bench's
# gated/measured GEOM table (e2e_packed_wmma_bench_q6_nopad.py:56-75). DO NOT tune live.
PACKED_WMMA_GEOM: dict[tuple[str, str], dict[str, int]] = {
  ("Q4_K", "attn_qo"):     dict(tm=128, tn=32,  tk=32, wm=4, wn=1, bc=1),
  ("Q4_K", "attn_kv"):     dict(tm=64,  tn=32,  tk=32, wm=2, wn=1, bc=1),
  ("Q4_K", "ffn_gate_up"): dict(tm=256, tn=64,  tk=32, wm=8, wn=1, bc=1),
  ("Q4_K", "ffn_down"):    dict(tm=256, tn=128, tk=32, wm=8, wn=2, bc=2),
  ("Q6_K", "ffn_down"):    dict(tm=256, tn=64,  tk=32, wm=8, wn=1, bc=1),
  ("Q6_K", "attn_kv"):     dict(tm=64,  tn=32,  tk=32, wm=2, wn=1, bc=1),
}

# Schedule-template source only (candidate_set lookup keys off role, not this profile string) --
# NOT a shape constraint. Gating/dispatch always use the live (m, n, k) of the actual invocation.
PACKED_WMMA_TEMPLATE_PROFILE = "qwen3_14b_q4k_m_gfx1100"

_GATE_CACHE: dict[tuple[str, str, int, int, int], tuple[bool, float | None]] = {}
_ENTRY_CACHE: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}


def _mutate_payload(base_payload: dict, g: dict[str, int], stride: int = 80) -> dict:
  p = copy.deepcopy(base_payload)
  sched = p["schedule"]
  sched["tile"] = {"m": g["tm"], "n": g["tn"], "k": g["tk"]}
  sched["waves"] = {"m": g["wm"], "n": g["wn"]}
  sched["threads"] = g["wm"] * g["wn"] * 32
  a_end = g["tm"] * stride
  b_end = a_end + g["tn"] * stride
  sched["lds"]["windows"] = {"a": [0, a_end], "b": [a_end, b_end]}
  sched["lds"]["strides"] = {"a": stride, "b": stride}
  sched["pipeline"]["buffer_count"] = g["bc"]
  return p


def _template_payload_for_role(role: str) -> dict:
  import json
  from pathlib import Path
  from extra.qk.route_manifest import promoted_prefill_candidate_policy
  path = promoted_prefill_candidate_policy()["candidate_set_path"]
  candidate_set = json.loads(Path(path).read_text())
  payloads = [row["payload"] for row in candidate_set["entries"]]
  template = next((p for p in payloads if p["workload"]["role"] == role), None)
  if template is None: raise ValueError(f"candidate set has no schedule template for role {role!r}")
  return template


def _payload_for_shape(role: str, shape: tuple[int, int, int]) -> dict:
  from extra.qk.runtime_specs import rebind_full_kernel_workload
  template = _template_payload_for_role(role)
  return rebind_full_kernel_workload(template, profile=PACKED_WMMA_TEMPLATE_PROFILE, role=role,
                                      shape=shape).to_json()["payload"]


def _run_gate(quant: str, role: str, shape: tuple[int, int, int], geom: dict[str, int]) -> tuple[bool, float | None]:
  import os
  import tempfile
  from extra.qk.prefill.packed_wmma_correctness_canary import build_artifact, run_canary
  base = _payload_for_shape(role, shape)
  mutated = _mutate_payload(base, geom)
  fd, artifact_path = tempfile.mkstemp(prefix=f"packed_wmma_canary_{quant}_{role}_", suffix=".npz")
  os.close(fd)
  try:
    build_artifact(quant, artifact_path, shape)
    outcome = run_canary(quant, artifact_path, timeout_seconds=90.0, base_payload=mutated)
  finally:
    try: os.remove(artifact_path)
    except OSError: pass
  passed = bool(outcome.get("passed"))
  max_abs = outcome.get("guarded", {}).get("max_abs_error") if isinstance(outcome.get("guarded"), dict) else None
  return passed, max_abs


def gate_combo(quant: str, role: str, shape: tuple[int, int, int]) -> bool:
  """Correctness-gate (quant, role, shape) exactly once (cached). Any failure/exception/unknown
  geometry is treated as a decline -- callers must never dispatch an ungated combo."""
  key = (quant, role, *shape)
  if key not in _GATE_CACHE:
    geom = PACKED_WMMA_GEOM.get((quant, role))
    if geom is None:
      _GATE_CACHE[key] = (False, None)
    else:
      try:
        _GATE_CACHE[key] = _run_gate(quant, role, shape, geom)
      except Exception:
        _GATE_CACHE[key] = (False, None)
  return _GATE_CACHE[key][0]


def gate_result(quant: str, role: str, shape: tuple[int, int, int]) -> tuple[bool, float | None] | None:
  return _GATE_CACHE.get((quant, role, *shape))


def warmstart_entry(quant: str, role: str, shape: tuple[int, int, int]) -> dict[str, Any]:
  """Build (and cache) the postrange warmstart opts/context/transform for a GATED (quant, role,
  shape) combo. Caller is responsible for gating first via gate_combo()."""
  key = (quant, role, *shape)
  if key not in _ENTRY_CACHE:
    from extra.qk.runtime_specs import derive_packed_weight_candidate, full_kernel_workload
    from extra.qk.prefill.current_prefill_execution_adapter import admit_current_prefill
    from tinygrad.codegen.opt import Opt, OptOps
    import tinygrad.codegen.opt.postrange as pr

    g = PACKED_WMMA_GEOM[(quant, role)]
    base_payload = _payload_for_shape(role, shape)
    mutated = _mutate_payload(base_payload, g)
    entry = derive_packed_weight_candidate(mutated, quant)
    payload = entry.to_json()["payload"]
    admission = admit_current_prefill(payload, entry.canonical_identity)
    m, n, k = full_kernel_workload(admission.normalized_payload).shape
    transform = admission.context.packed_weight
    if transform is None: raise ValueError(f"packed-wmma combo {(quant, role)} is not a packed-weight candidate")
    warm_key = pr.warmstart_key({m, n}, k, transform.storage_dtype)
    opt = (Opt(OptOps.TC, 0, (-1, 2, 1)),)
    _ENTRY_CACHE[key] = {"key": warm_key, "opt": opt, "context": admission.context, "transform": transform,
                          "m": m, "n": n, "k": k}
  return _ENTRY_CACHE[key]


# PrefillLinearRouteSpec.quant follows the prefill_routes.py convention ("q4k"/"q6k", no
# underscore); PACKED_WMMA_GEOM (and the packed_weight_candidate/canary machinery) key off
# the GGUF quant-format spelling ("Q4_K"/"Q6_K"). This is the single translation point.
_SPEC_QUANT_TO_FORMAT = {"q4k": "Q4_K", "q6k": "Q6_K"}


@dataclass(frozen=True)
class PackedWmmaPrefillCandidate:
  """Shared matches()/run() implementation for the Q4_K/Q6_K packed-WMMA candidates. Declines
  (returns None) for anything outside the frozen, gated (quant, role, shape=pp512) surface --
  the caller (route_packed_wmma_prefill) then falls through to the direct-packed baseline."""
  quant: str

  def matches(self, lin, spec) -> bool:
    return _SPEC_QUANT_TO_FORMAT.get(str(getattr(spec, "quant", ""))) == self.quant

  def run(self, lin, x: Tensor, x_batch: Tensor, spec) -> Tensor | None:
    role = getattr(spec, "role", "") or ""
    if not role: return None
    combo = (self.quant, role)
    if combo not in PACKED_WMMA_GEOM: return None
    m, n, k = spec.m, spec.n, spec.k
    shape = (m, n, k)
    if not gate_combo(self.quant, role, shape): return None
    try:
      e = warmstart_entry(self.quant, role, shape)
    except Exception:
      return None
    if (e["m"], e["n"], e["k"]) != shape: return None

    transform = e["transform"]
    packed_weight = lin.prefill_packed_weight()
    blocks, halfwords = n * k // transform.block_elems, transform.block_bytes // 2
    b = packed_weight.bitcast(dtypes.uint16).reshape(blocks, halfwords).pad(((0, 0), (0, 128 - halfwords))) \
      .reshape(blocks, 128, 1).expand(blocks, 128, 2).reshape(n, k).bitcast(dtypes.half)
    out = x_batch @ b.transpose()
    return out.reshape(1, m, n)


class Q4KPackedWmmaPrefillCandidate(PackedWmmaPrefillCandidate):
  def __init__(self): super().__init__("Q4_K")


class Q6KPackedWmmaPrefillCandidate(PackedWmmaPrefillCandidate):
  def __init__(self): super().__init__("Q6_K")


PACKED_WMMA_PREFILL_CANDIDATES: tuple[PackedWmmaPrefillCandidate, ...] = (
  Q4KPackedWmmaPrefillCandidate(), Q6KPackedWmmaPrefillCandidate(),
)


def select_packed_wmma_prefill_candidate(lin, spec) -> PackedWmmaPrefillCandidate | None:
  for candidate in PACKED_WMMA_PREFILL_CANDIDATES:
    if candidate.matches(lin, spec): return candidate
  return None


def build_packed_wmma_warmstart_tables(covered_linears, ubatch: int) -> tuple[dict, dict]:
  """Build the combined {warmstart_key: opts} / {warmstart_key: candidate_context} tables for
  every gated (quant, role) combo found among `covered_linears` (iterable of (lin, out_f, in_f)
  triples, e.g. Transformer._prefill_v2_covered()) at the given physical prefill M (`ubatch`).
  Ungated/unknown combos are silently omitted (they simply never get a packed-wmma warmstart
  entry and fall through per-call in `run`)."""
  from tinygrad.llm.prefill_routes import _direct_packed_module_role, _is_q4k_linear, _is_q6k_linear
  opts: dict = {}
  contexts: dict = {}
  for lin, out_f, in_f in covered_linears:
    if _is_q4k_linear(lin): quant = "Q4_K"
    elif _is_q6k_linear(lin): quant = "Q6_K"
    else: continue
    role = str(getattr(lin, "_prefill_graph_role", "")) or _direct_packed_module_role(lin)
    if (quant, role) not in PACKED_WMMA_GEOM: continue
    shape = (ubatch, out_f, in_f)
    if not gate_combo(quant, role, shape): continue
    try:
      e = warmstart_entry(quant, role, shape)
    except Exception:
      continue
    opts[e["key"]] = e["opt"]
    contexts[e["key"]] = e["context"]
  return opts, contexts
