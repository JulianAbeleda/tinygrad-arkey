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
import json
from pathlib import Path
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

# This is deliberately a single exact replacement, not a geometry policy.  The
# two-buffer candidate set remains the default and every other payload continues
# through its existing admission path.  The one-buffer payload is selected only
# when all independently captured evidence still names the same artifacts and
# identities.  Missing, malformed, or drifted evidence is a decline to buffer2.
_SINGLE_BUFFER_ATTN_QO = {
  "profile": "qwen3_8b_q4k_m_gfx1100", "quant": "Q4_K", "role": "attn_qo", "shape": (512, 4096, 4096),
  "compile_identity": "555ff0474238c6408dfe79ff14e26febf588072bd7aefaac312e521748b55ece",
  "packed_identity": "83dcb21acfed699cb0f27101422c6bbd6243236d6daf0f4431ce6c0efa6e550a",
  "compile": "bench/prefill-lds-single-buffer-probe-20260723/attn-qo-compile-only.json",
  "numeric": "bench/prefill-lds-single-buffer-probe-20260723/attn-qo-q4k-runtime-ab.json",
  "timing": "bench/prefill-lds-single-buffer-probe-20260723/attn-qo-q4k-pinned-track.json",
}


def _single_buffer_attn_qo_admitted(payload: dict[str, Any], quant: str, role: str,
                                    shape: tuple[int, int, int]) -> bool:
  """Validate the evidence join for the sole registered one-buffer payload."""
  spec = _SINGLE_BUFFER_ATTN_QO
  if (quant, role, shape) != (spec["quant"], spec["role"], spec["shape"]): return False
  try:
    from extra.qk.runtime_specs import _canonical_full_kernel_identity
    probe = one_buffer_payload(payload)
    if _canonical_full_kernel_identity(probe) != spec["compile_identity"]: return False
    root = Path(__file__).resolve().parents[3]
    compile_row, numeric_row, timing_row = (json.loads((root / spec[key]).read_text()) for key in ("compile", "numeric", "timing"))
    compiled = compile_row["probe"]
    resources = compiled["resources"]
    if (compile_row.get("schema") != "tinygrad.prefill_lds_single_buffer_compile_probe.v1" or
        compile_row.get("profile") != spec["profile"] or compile_row.get("role") != role or
        compiled.get("canonical_identity") != spec["compile_identity"] or
        resources.get("lds_bytes") != 20480 or any(resources.get(k) != 0 for k in ("scratch_bytes", "vgpr_spills", "sgpr_spills"))): return False
    matching = [r for r in numeric_row.get("runs", ()) if r.get("label") == "one_buffer_probe" and
                r.get("phase") == "warmed_sample"]
    guarded = matching[0]["outcome"]["guarded"] if len(matching) == 1 else {}
    # Keep the evidence join decomposed into the actual forward-path guards:
    # exact geometry, warmed phase, guarded producer/body execution, and the
    # complete timing tail. A missing field declines to the two-buffer route.
    phase_ok = len(matching) == 1 and matching[0].get("phase") == "warmed_sample"
    producer_ok = guarded.get("identity", {}).get("canonical_identity") == spec["packed_identity"]
    body_ok = all(guarded.get(k) is True for k in ("passed", "numerics_passed", "full_output_compared", "guards_intact", "finite_output")) and \
      guarded.get("max_abs_error") == 0.0
    if (numeric_row.get("schema") != "tinygrad.prefill_lds_single_buffer_runtime_probe.v1" or
        numeric_row.get("profile") != spec["profile"] or numeric_row.get("role") != role or
        not (phase_ok and producer_ok and body_ok)): return False
    variants = [v for v in timing_row.get("variants", ()) if v.get("label") == "one_buffer_probe"]
    samples = variants[0].get("samples", ()) if len(variants) == 1 else ()
    if (timing_row.get("schema") != "tinygrad.prefill_lds_single_buffer_peak_track.v1" or
        timing_row.get("profile") != spec["profile"] or timing_row.get("role") != role or
        timing_row.get("sample_count") != 10 or timing_row.get("warmup_count") != 3 or len(samples) != 10 or
        not all(s.get("canonical_identity") == spec["packed_identity"] and s.get("max_abs_error") == 0.0 and
                s.get("gpu_state", {}).get("power_dpm_force_performance_level", "").strip() == "manual" and
                "1249Mhz *" in s.get("gpu_state", {}).get("pp_dpm_mclk", "") and
                "2304Mhz" in s.get("gpu_state", {}).get("pp_dpm_sclk", "") for s in samples)): return False
    tail_ok = timing_row.get("sample_count") == 10 and timing_row.get("warmup_count") == 3 and len(samples) == 10 and \
      all(s.get("canonical_identity") == spec["packed_identity"] and s.get("max_abs_error") == 0.0 and
          s.get("gpu_state", {}).get("power_dpm_force_performance_level", "").strip() == "manual" and
          "1249Mhz *" in s.get("gpu_state", {}).get("pp_dpm_mclk", "") and
          "2304Mhz" in s.get("gpu_state", {}).get("pp_dpm_sclk", "") for s in samples) and \
      variants[0].get("summary", {}).get("median_ms") == 0.36741999999999997
    return tail_ok
  except (KeyError, OSError, TypeError, ValueError):
    return False


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


def one_buffer_payload(payload: dict[str, Any]) -> dict[str, Any]:
  probe = copy.deepcopy(payload)
  pipeline = probe["schedule"]["pipeline"]
  if (pipeline.get("buffer_count"), pipeline.get("stage_count")) != (2, 1):
    raise ValueError("one-buffer admission requires the exact two-buffer stage-1 baseline")
  pipeline["buffer_count"] = 1
  return probe


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
    one_buffer = _single_buffer_attn_qo_admitted(mutated, quant, role, shape)
    if one_buffer: mutated = one_buffer_payload(mutated)
    entry = derive_packed_weight_candidate(mutated, quant)
    payload = entry.to_json()["payload"]
    admission = admit_current_prefill(payload, entry.canonical_identity)
    m, n, k = full_kernel_workload(admission.normalized_payload).shape
    transform = admission.context.packed_weight
    if transform is None: raise ValueError(f"packed-wmma combo {(quant, role)} is not a packed-weight candidate")
    warm_key = pr.warmstart_key({m, n}, k, transform.storage_dtype)
    opt = (Opt(OptOps.TC, 0, (-1, 2, 1)),)
    # The canonical identity is bound at the model-forward call site below;
    # retaining it here prevents a successful offline warmstart from being
    # misattributed as a selected one-buffer execution.
    _ENTRY_CACHE[key] = {"key": warm_key, "opt": opt, "context": admission.context, "transform": transform,
                          "m": m, "n": n, "k": k, "canonical_identity": admission.canonical_identity,
                          "one_buffer": one_buffer}
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
    if e.get("one_buffer") and (self.quant, role, shape) != ("Q4_K", "attn_qo", (512, 4096, 4096)): return None

    transform = e["transform"]
    packed_weight = lin.prefill_packed_weight()
    blocks, halfwords = n * k // transform.block_elems, transform.block_bytes // 2
    b = packed_weight.bitcast(dtypes.uint16).reshape(blocks, halfwords).pad(((0, 0), (0, 128 - halfwords))) \
      .reshape(blocks, 128, 1).expand(blocks, 128, 2).reshape(n, k).bitcast(dtypes.half)
    # Keep the primitive's vectorized accumulator store rank-2. Letting the
    # final (1,M,N) view fuse into this producer can make HIP assign a vector
    # expression to a constructed float4 value instead of an addressable
    # scalar output lane. The concrete rank-2 boundary is the same output ABI
    # used by the direct-packed primitive; the model-facing batch view stays
    # metadata-only after this scalarized store.
    out = (x_batch @ b.transpose()).contiguous()
    # This is the real model-forward binding, rather than a warmstart-table
    # annotation. The exact identity is observable by the whole-model route
    # census/provenance hooks only after this guarded body has been selected.
    setattr(lin, "_prefill_full_kernel_candidate_identity", e["canonical_identity"])
    setattr(lin, "_prefill_full_kernel_candidate_one_buffer", bool(e.get("one_buffer")))
    if e.get("one_buffer"):
      from extra.qk.prefill_graph_gemm_route import record_model_forward_candidate
      record_model_forward_candidate(role=role, shape=shape, canonical_identity=e["canonical_identity"], one_buffer=True)
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
