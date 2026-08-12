#!/usr/bin/env python3
"""NV wall-bracket A/B for M1: ffn-norm epilogue absorption into the fused w1+w3 gate/up GEMV.

Sibling of ``extra/llm_research/decode/nv_epilogue_absorption_ab.py`` (the
M2b/M2c/M2d residual-family campaign, LANDED: residual/cast/contiguous row
exhausted at 193.36 tok/s).  M1 attacks the next in-row item of the norms row:
per decode token the 36 ffn-norm chains render ``r_16_256`` (scale reduce,
~3.84 us) + ``E_32_32_4_f14a5cc0`` (fp16 epilogue, ~2.27 us), then the fused
``q4k_g3_lanemap_gemv_w1w3fused16_*`` gate/up GEMV consumes the fp16 normed
hidden state.  The candidate arm installs ``_rms_affine_gateup_norm_weight`` on
every block (an owned fp16 buffer of ``ffn_norm.weight``, the exact buffer the
control epilogue reads), so ``q4k_gate_up_rms_affine_qualification_call`` fuses
the norm INTO the gate/up GEMV: each packed-Q4 load applies
``(half)((h*s)*w)`` -- the control epilogue's single fp16 RNE round at the very
end of the fp32 multiply chain, weight upcast fp16->fp32 -- and the scale is the
bitwise-exact control reduce ``(h.float().square().mean(-1,keepdim=True)+eps)
.rsqrt()`` rendered on the RAW fp32 h, so the route renders the SAME
``r_16_256`` program as control.  The fused z is stored fp16 under its own
``q4k_g3_lanemap_w1w3_rms_affine16_*`` name (mirroring the landed fused16
store), so the graph's fp32->fp16 ffn-activation cast stays folded and the
ffn_down consumer sees the same fp16 ABI as control.  The M2b ffn_down residual
add stays live: the block threads ``ffn_down(z, normed_h=h)``.

The control arm is the LANDED M2d candidate conditions (callify flags +
``_decode_reduce_output_rmsnorm_promoted`` + ``_decode_direct_greedy_promoted``
+ M2a fp16-store lease + M2b ffn_down resadd lease + M2d flash-combine fp16
lease) WITHOUT the M1 lease.  Both arms run as fresh processes under
``timeout ... flock -w`` on the shared GPU bench lock so JIT capture and
allocator state cannot leak across arms.  Unlike M2d, the M1 fp16 round is REAL
in both the eager JIT=0 baseline and the captured decode graph (control renders
``E_32_32_4_f14a5cc0`` in the eager pass too), so no ``_without_*`` lease-strip
hack is needed: the exact-logits gate validates the whole construction and the
eager cache is arm-invariant because the fused values are bitwise-identical.

Gate order is fixed: NV render smoke (Xid 31 class) with the rms_affine16
kernels in the compiled set, then the exact full-logit gate, then the M1 census,
then the serialized reverse control/candidate/control wall bracket.  The census
gate FAILS CLOSED if the ffn-norm epilogues remain, if the fused16 kernels stay,
if the rms_affine16 bodies do not swap 1:1, if the scale reduce count shifts, or
if any unrelated program count shifts; the expected drops are derived from the
freshly measured control arm, never a stale constant.  Any gate failure writes a
NO-GO record with the exact evidence; M1 books only when every gate passes and
the bracket promotes at +50 us/token against BOTH bracketing controls.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, pathlib, statistics, sys, time
import numpy as np

from extra.llm_research.decode.nv_fusion_population_ledger import POP_NORMS, classify as _ledger_classify
from extra.llm_research.decode.nv_fusion_cost_model import predict_wall_delta, reconcile_cost_prediction
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _load, _prompt
from extra.llm_research.decode.nv_reduce_output_primitive_ab import validate_logits_gate as _c6_validate_logits_gate
from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import (
  PROMOTION_US, TM_RE, _child_root, _digest, _require_candidate_callify_flags,
  _run_child, _settled_continuous_windows, _timing_hash_authority, _validate_run_extent,
  _write_record, tok_per_s,
)

SCHEMA = "tinygrad.nv_epilogue_absorption_m1_ab.v1"
SMOKE_SCHEMA = "tinygrad.nv_epilogue_absorption_m1_ab.smoke.v1"
LOGITS_SCHEMA = "tinygrad.nv_epilogue_absorption_m1_ab.logits.v1"
CENSUS_SCHEMA = "tinygrad.nv_epilogue_absorption_m1_ab.census.v1"
TIMING_SCHEMA = "tinygrad.nv_epilogue_absorption_m1_ab.timing.v1"

LEASE = "_q4k_w1w3_fp16_store_lease"
LEASE2 = "_ffn_down_resadd_lease"
LEASE3 = "_flash_combine_fp16_lease"
LEASE4 = "_rms_affine_gateup_norm_weight"
FUSED16_PREFIX = "q4k_g3_lanemap_gemv_w1w3fused16_"
RMS_AFFINE16_PREFIX = "q4k_g3_lanemap_w1w3_rms_affine16_"
RMS_AFFINE_PREFIX = "q4k_g3_lanemap_w1w3_rms_affine_"
# M1 families: the ordinary ffn-norm chain (scale reduce r_16_256 + fp16
# epilogue E_32_32_4_f14a5cc0) folds into the fused gate/up GEMV.  The
# r_16_256 count must stay IDENTICAL between arms: the route's bitwise-exact
# scale renders the same reduce program 1:1 with the control chain's.
REDUCE_PREFIX = "r_16_256"
NORM_EPILOGUE_PREFIX = "E_32_32_4_f14a5cc0"
# M2d families that must stay byte-identical between arms.
RESADD_PREFIX = "E_32_32_4_02a9738c"
BLOCK_OUTPUT_COPY_PREFIX = "E_32_32_4_fab82d40"
ATTENTION_CAST_PREFIX = "E_32_32_4_0a5eb0ac"
COMBINE_F16_PREFIX = "flash_fused_gmax_combine_f16_"
COMBINE_F32_PREFIX = "flash_fused_gmax_combine_"
COMBINE_COPY_PREFIX = "E_32_32_4_3b0fcfbc"
FFNRESADD_PREFIX = "q4k_g3_lanemap_gemv_epi_ffnresadd_"
FFNRESADD_SUFFIX = "_epi_ffnresadd"

CONSTRUCTION = {
  "route": "M1 ffn-norm epilogue absorption into the fused w1+w3 gate/up GEMV (r_16_256 + E_32_32_4_f14a5cc0 per block fold away)",
  "population": "norms (remaining ffn-norm chains)",
  "mechanism": "the fused gate/up GEMV applies the ordinary ffn-norm epilogue per packed-Q4 load: (half)((h*s)*w) with ONE fp16 RNE round at the end of the fp32 multiply chain (weight upcast fp16->fp32) and the bitwise-exact control scale (h.float().square().mean(-1,keepdim=True)+eps).rsqrt() rendered on the raw fp32 h (the SAME r_16_256 reduce program as control). The fp16 norm weight buffer is the loader's _decode_reduce_output_weight (or a fresh cast), i.e. the exact buffer the control epilogue reads. The fused z stores fp16 under q4k_g3_lanemap_w1w3_rms_affine16_* (mirroring the landed fused16 store), so the ffn-activation cast stays folded and the ffn_down consumer sees the same fp16 ABI as control. The M2b ffn_down residual add stays live: the block threads ffn_down(z, normed_h=h).",
  "codegen_path": "both arms run the LANDED M2d candidate conditions (callify flags + _decode_reduce_output_rmsnorm_promoted + _decode_direct_greedy_promoted + _q4k_w1w3_fp16_store_lease + _ffn_down_resadd_lease + _flash_combine_fp16_lease on the model and every block); the candidate additionally installs _rms_affine_gateup_norm_weight on every block (an owned fp16 buffer of ffn_norm.weight), which opens q4k_gate_up_rms_affine_qualification_call -> q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(store_fp16=True); the model checks the M1 lease BEFORE the M2b branch and threads ffn_down(z, normed_h=h) so the M2b residual add is preserved",
  "census_target": "E_32_32_4_f14a5cc0 37 -> 1 (the 36 ffn epilogues fold; the one non-FFN chain stays); q4k_g3_lanemap_gemv_w1w3fused16_12288_4096 36 -> 0 swapped 1:1 with q4k_g3_lanemap_w1w3_rms_affine16_12288_4096 0 -> 36; r_16_256 stays 37 == 37 (36 route scale reduces + 1 remaining chain); all other program counts byte-identical to control; honest net program delta -36 (594 -> 558)",
  "correctness_contract": {
    "full_logit_fp32_sha256": "bitwise identical to control over the stacked rows",
    "token_stream": "identical to control",
    "per_row_argmax": "equals the sampled token",
    "promotion": "+50 us/token vs both bracketing controls (control / candidate / control)",
    "census": "E_32_32_4_f14a5cc0 dropped by exactly the control fused16 count (the 36 ffn epilogues), rms_affine16 bodies present 1:1 with the control fused16 count, fused16 at zero, r_16_256 counts identical, no other program-count shift; FAIL CLOSED on any unrelated delta",
  },
  "question": "Does the M1 in-kernel ffn-norm absorption survive NV render (Xid 31 class), preserve exact full logits, remove exactly the 36 ffn-norm epilogues with the fused16 -> rms_affine16 1:1 swap and no r_16_256 shift, and book the remaining ffn-norm share of the norms row under the reverse wall bracket?",
}

# Predicted-wall-delta contract (nv_fusion_cost_model.py): the prediction is
# derived from the llama reference shape (norm arithmetic NEVER enters the
# matmul inner loop; llama keeps the norm as one fused rms_norm_f32 kernel) plus
# the per-element arithmetic of this candidate (the epilogue re-executes once
# per matrix dot, R=2; x streams fp32 instead of fp16; r_16_256 must stay).
# The bracket reconciles the measured delta against this range: CONFIRMED /
# EXPLAINED pass with evidence; CONTRADICTED fails the campaign closed.
COST_PREDICTION = {
  "contract": "before implementing, derive the predicted wall delta from the llama reference shape plus per-element instruction/traffic arithmetic; the wall bracket then either confirms it or explains the gap",
  "llama_reference": "rms_norm_f32 = ONE fused reduce+affine kernel, fp32 out; norm never enters the matmul; matmul consumes quantize_q8_1 activation (llama_tinygrad_role_manifest.py)",
  "arithmetic": {
    "redundancy": 2,
    "redundancy_note": "the norm epilogue re-executes once per matrix dot (gate and up each recompute (half)((h*s)*w))",
    "per_element_extra_ops": "2 FMUL + fp16 RNE cast + upcast per element per matrix",
    "x_traffic": "fp32 16KB vs fp16 8KB per token per block",
    "scale_reduce_retained": "r_16_256 must stay (bitwise contract; llama keeps n_f32)",
  },
  "formula": "blocks x [ (R - 1) x M_removed - R x launch_us ]; positive = candidate slower",
  "tolerance_us": 20.0,
  "assumptions": {
    "launch_us": "1.5 (range 1.0-2.0), E_32_32_4 class floor, m4-resadd-landing-scope-20260806.md",
    "M_removed": "control census median of the folded epilogue family, measured per run before the bracket",
  },
  "unmodeled": ["in-kernel critical path (occupancy/dependency chain)", "activation traffic"],
}


def _configure(model, arm: str) -> None:
  """Both arms set the LANDED M2d candidate conditions (M2a fp16-store lease +
  M2b ffn-down residual-add lease + M2d flash-combine fp16 lease on the model
  and every block); the candidate additionally installs the M1 norm-epilogue
  lease (_rms_affine_gateup_norm_weight, an owned fp16 buffer of the ffn-norm
  weight) on every block.  No loader policy creates any lease."""
  model._decode_direct_greedy_promoted = True
  _require_candidate_callify_flags()
  model._decode_reduce_output_rmsnorm_promoted = True
  for block in model.blk:
    block._decode_reduce_output_rmsnorm_promoted = True
  setattr(model, LEASE, True)
  for block in model.blk: setattr(block, LEASE, True)
  setattr(model, LEASE2, True)
  for block in model.blk:
    setattr(block, LEASE2, True)
    ffn_down = getattr(block, "ffn_down", None)
    if ffn_down is not None: setattr(ffn_down, LEASE2, True)
  setattr(model, LEASE3, True)
  for block in model.blk: setattr(block, LEASE3, True)
  if arm == "candidate":
    from tinygrad import dtypes
    for block in model.blk:
      ffn_norm = getattr(block, "ffn_norm", None)
      if ffn_norm is None or getattr(ffn_norm, "weight", None) is None:
        raise RuntimeError(f"M1 candidate requires ffn_norm.weight on block {block}")
      weight = getattr(ffn_norm, "_decode_reduce_output_weight", None)
      if weight is None:
        weight = ffn_norm.weight.cast(dtypes.float16).contiguous().realize()
      setattr(block, LEASE4, weight)
  elif arm != "control":
    raise ValueError(f"unknown arm {arm!r}")


def _gates(model) -> dict:
  from tinygrad import dtypes
  m1_weights = []
  for block in getattr(model, "blk", None) or []:
    weight = getattr(block, LEASE4, None)
    m1_weights.append(None if weight is None else {
      "dtype": str(weight.dtype), "is_fp16": bool(weight.dtype == dtypes.float16), "shape": list(weight.shape)})
  return {
    "decode_direct_greedy_promoted": bool(getattr(model, "_decode_direct_greedy_promoted", False)),
    "reduce_output_rmsnorm_promoted": bool(getattr(model, "_decode_reduce_output_rmsnorm_promoted", False)),
    "w1w3_fp16_store_lease": bool(getattr(model, LEASE, False)),
    "block_w1w3_fp16_store_lease": [bool(getattr(block, LEASE, False)) for block in getattr(model, "blk", None) or []],
    "ffn_down_resadd_lease": bool(getattr(model, LEASE2, False)),
    "block_ffn_down_resadd_lease": [bool(getattr(block, LEASE2, False)) for block in getattr(model, "blk", None) or []],
    "flash_combine_fp16_lease": bool(getattr(model, LEASE3, False)),
    "block_flash_combine_fp16_lease": [bool(getattr(block, LEASE3, False)) for block in getattr(model, "blk", None) or []],
    "rms_affine_gateup_lease_model": bool(getattr(model, LEASE4, False)),
    "rms_affine_gateup_lease_blocks": m1_weights,
  }


def _assert_control_closed(gates: dict) -> None:
  leased = []
  if not gates.get("w1w3_fp16_store_lease"): leased.append(f"model.{LEASE}")
  for index, value in enumerate(gates.get("block_w1w3_fp16_store_lease") or []):
    if not value: leased.append(f"block[{index}].{LEASE}")
  if not gates.get("ffn_down_resadd_lease"): leased.append(f"model.{LEASE2}")
  for index, value in enumerate(gates.get("block_ffn_down_resadd_lease") or []):
    if not value: leased.append(f"block[{index}].{LEASE2}")
  if not gates.get("flash_combine_fp16_lease"): leased.append(f"model.{LEASE3}")
  for index, value in enumerate(gates.get("block_flash_combine_fp16_lease") or []):
    if not value: leased.append(f"block[{index}].{LEASE3}")
  if gates.get("rms_affine_gateup_lease_model"): leased.append(f"model.{LEASE4}")
  for index, value in enumerate(gates.get("rms_affine_gateup_lease_blocks") or []):
    if value is not None: leased.append(f"block[{index}].{LEASE4}")
  if leased:
    raise RuntimeError(f"control arm requires the LANDED M2d candidate (M2a+M2b+M2d leases) WITHOUT {LEASE4}, observed: {leased}")


def _assert_candidate_configured(gates: dict) -> None:
  missing = []
  for lease, key in ((LEASE, "w1w3_fp16_store_lease"), (LEASE2, "ffn_down_resadd_lease"), (LEASE3, "flash_combine_fp16_lease")):
    if not gates.get(key): missing.append(f"model.{lease}")
  for index, value in enumerate(gates.get("block_w1w3_fp16_store_lease") or []):
    if not value: missing.append(f"block[{index}].{LEASE}")
  for index, value in enumerate(gates.get("block_ffn_down_resadd_lease") or []):
    if not value: missing.append(f"block[{index}].{LEASE2}")
  for index, value in enumerate(gates.get("block_flash_combine_fp16_lease") or []):
    if not value: missing.append(f"block[{index}].{LEASE3}")
  if gates.get("rms_affine_gateup_lease_model"): missing.append(f"model.{LEASE4} (unexpected; install per-block only)")
  for index, value in enumerate(gates.get("rms_affine_gateup_lease_blocks") or []):
    if value is None or value.get("is_fp16") is not True or value.get("shape") != [4096]:
      missing.append(f"block[{index}].{LEASE4} (expected fp16 (4096,), got {value})")
  if missing:
    raise RuntimeError(f"candidate arm requires M2a+M2b+M2d leases everywhere and {LEASE4} fp16 (4096,) on every block: {missing}")


def validate_logits_gate(control: dict, candidate: dict) -> dict:
  """Exact-output gate with the M1 logits schema (mirror of the fp32 q/k gate)."""
  for label, row in (("control", control), ("candidate", candidate)):
    if row.get("schema") != LOGITS_SCHEMA:
      raise ValueError(f"{label} logits row requires schema {LOGITS_SCHEMA!r}, got {row.get('schema')!r}")
  return _c6_validate_logits_gate(control, candidate)


def _model(arm: str, model_path: str, max_context: int):
  _require_candidate_callify_flags()
  model = _load(model_path, max_context)
  _configure(model, arm)
  gates = _gates(model)
  if arm == "control": _assert_control_closed(gates)
  else: _assert_candidate_configured(gates)
  return model, gates


@contextlib.contextmanager
def _m1_arm_context(arm: str):
  """Both M1 arms run as the LANDED M2d candidate: the callify Context flags are
  live for control AND candidate (the M1 control is the M2d LANDED candidate
  WITHOUT the M1 lease), so the only census delta between the arms is the
  ffn-norm absorption."""
  if arm not in ("control", "candidate"):
    raise ValueError(f"unknown arm {arm!r}")
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    yield


def smoke(arm: str, model_path: str, depth: int, max_context: int) -> dict:
  """Phase 0 NV render smoke: compile and run one decode token under the
  candidate conditions.  Success is survival (no Xid 31 MMU fault) with the
  rms_affine16 kernels in the compiled program set, no fused16 kernel, and the
  ffn-norm epilogue count reduced to the non-FFN chain."""
  if arm != "candidate":
    raise ValueError("smoke requires the candidate arm; the rms_affine16 kernels only exist under the candidate conditions")
  from tinygrad import Device
  from tinygrad.engine.jit import GraphAdmissionCensus, observe_graph_admissions
  from tinygrad.helpers import Context
  with _m1_arm_context(arm):
    model, gates = _model(arm, model_path, max_context)
    model.reset_generation_state()
    gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
    admission = GraphAdmissionCensus()
    try:
      prelude = int(next(gen))
      token = None
      for index in range(3):
        if index == 1:
          with Context(TRACEMETA=1), observe_graph_admissions(admission):
            token = int(next(gen))
        else:
          next(gen)
      if token is None: raise RuntimeError("smoke observation window did not run")
      Device[Device.DEFAULT].synchronize()
      programs = [record.program_name for record in admission.records if record.program_name]
      counts: dict[str, int] = {}
      for name in programs: counts[name] = counts.get(name, 0) + 1
      affine16 = sum(count for name, count in counts.items() if name.startswith(RMS_AFFINE16_PREFIX))
      affine32 = sum(count for name, count in counts.items() if name.startswith(RMS_AFFINE_PREFIX))
      fused16 = sum(count for name, count in counts.items() if name.startswith(FUSED16_PREFIX))
      epilogue = sum(count for name, count in counts.items() if name.startswith(NORM_EPILOGUE_PREFIX))
      reduce = sum(count for name, count in counts.items() if name.startswith(REDUCE_PREFIX))
      resadd = sum(count for name, count in counts.items() if name.startswith(RESADD_PREFIX))
      copy = sum(count for name, count in counts.items() if name.startswith(BLOCK_OUTPUT_COPY_PREFIX))
      attn_cast = sum(count for name, count in counts.items() if name.startswith(ATTENTION_CAST_PREFIX))
      combine_f16 = sum(count for name, count in counts.items() if name.startswith(COMBINE_F16_PREFIX))
      combine_f32 = sum(count for name, count in counts.items()
                        if name.startswith(COMBINE_F32_PREFIX) and not name.startswith(COMBINE_F16_PREFIX))
      combine_copy = sum(count for name, count in counts.items() if name.startswith(COMBINE_COPY_PREFIX))
      ffnresadd = sum(count for name, count in counts.items()
                      if name.startswith(FFNRESADD_PREFIX) or name.endswith(FFNRESADD_SUFFIX))
      return {"schema": SMOKE_SCHEMA, "arm": arm, "mode": "smoke", "gates": gates,
              "device": str(Device[Device.DEFAULT]), "survive": True,
              "prelude_token": prelude, "token": token,
              "decode_observation": "second decode token after the prelude (index 1 of 3), mirroring capture_decode_graph",
              "rms_affine16_count": affine16, "rms_affine32_count": affine32,
              "fused16_count": fused16, "norm_epilogue_count": epilogue, "scale_reduce_count": reduce,
              "residual_add_count": resadd, "block_output_copy_count": copy,
              "attention_cast_count": attn_cast, "combine_f16_count": combine_f16,
              "combine_f32_count": combine_f32, "combine_copy_count": combine_copy,
              "ffn_down_resadd_count": ffnresadd,
              "program_count": len(programs), "program_names": programs}
    finally:
      gen.close()


def logits(arm: str, model_path: str, depth: int, count: int, max_context: int) -> tuple[dict, np.ndarray]:
  from tinygrad import Tensor, UOp
  from tinygrad.helpers import Context
  with _m1_arm_context(arm):
    model, gates = _model(arm, model_path, max_context)
    gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
    try: prelude = int(next(gen))
    finally: gen.close()
    token, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
    start_pos = UOp.variable("start_pos", 0, max_context - 1)
    # Eager baseline is arm-invariant WITHOUT any lease strip: the M1 fp16 round
    # is real in both the eager JIT=0 pass (control renders E_32_32_4_f14a5cc0
    # there too) and the captured decode graph, and the fused values are
    # bitwise-identical, so the cache written at `depth` matches between arms.
    with Context(JIT=0):
      _, eager_logits = model.forward_with_logits(token, start_pos.bind(depth), temp)
    if not np.isfinite(eager_logits.numpy()).all(): raise RuntimeError("non-finite eager logits")
    samples, rows = [], []
    for idx in range(count):
      sample, full = model.decode_with_logits(token, start_pos.bind(depth + 1 + idx), temp)
      row, sid = full.numpy(), int(sample.item())
      finite = bool(np.isfinite(row).all())
      argmax = int(row.argmax(axis=-1).item())
      if not finite or sid != argmax:
        nonfinite = int((~np.isfinite(row)).sum()) if not finite else 0
        raise RuntimeError(
          f"invalid diagnostic output at row {idx}: finite={finite} non_finite_count={nonfinite} "
          f"sid={sid} argmax={argmax} sid_equals_argmax={sid == argmax} "
          f"row_min={float(row.min())} row_max={float(row.max())} "
          f"row_sha256={_digest(row)[:16]}")
      samples.append(sid); rows.append(row)
    stacked = np.stack(rows)
    return {"schema": LOGITS_SCHEMA, "arm": arm, "mode": "logits", "gates": gates,
            "prelude_token": prelude, "tokens": samples, "shape": list(stacked.shape),
            "dtype": str(stacked.dtype), "logits_sha256": _digest(stacked)}, stacked


def census(arm: str, model_path: str, depth: int, max_context: int) -> dict:
  """DEBUG kernel census of one decode token, classified through the population
  ledger; the M1/M2 families counted by exact name."""
  from tinygrad.helpers import Context
  with _m1_arm_context(arm):
    model, gates = _model(arm, model_path, max_context)
    gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
    with Context(DEBUG=0): next(gen)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
      with Context(DEBUG=2): token = int(next(gen))
    gen.close()
    rows = []
    for line in capture.getvalue().splitlines():
      if (match := TM_RE.match(line)):
        us = float(match.group(2)) * (1000.0 if match.group(3) == "ms" else 1.0)
        rows.append((match.group(1), us))
    hist: dict[str, list[float]] = {}
    for name, us in rows: hist.setdefault(name, []).append(us)
    population_counts: dict[str, int] = {}
    norms_roles: dict[str, int] = {}
    program_counts: dict[str, int] = {}
    for name, _ in rows:
      pop, role, _ = _ledger_classify(name)
      population_counts[pop] = population_counts.get(pop, 0) + 1
      if pop == POP_NORMS: norms_roles[role] = norms_roles.get(role, 0) + 1
      program_counts[name] = program_counts.get(name, 0) + 1
    def family(prefix: str) -> int:
      return sum(count for name, count in program_counts.items() if name.startswith(prefix))
    fused16_count = family(FUSED16_PREFIX)
    rms_affine16_count = family(RMS_AFFINE16_PREFIX)
    rms_affine32_count = family(RMS_AFFINE_PREFIX)
    norm_epilogue_count = family(NORM_EPILOGUE_PREFIX)
    scale_reduce_count = family(REDUCE_PREFIX)
    return {"schema": CENSUS_SCHEMA, "arm": arm, "mode": "census", "gates": gates, "token": token,
            "kernels": len(rows), "kernel_us": round(sum(us for _, us in rows), 3),
            "norms_kernels": sum(1 for name, _ in rows if _ledger_classify(name)[0] == POP_NORMS),
            "norms_roles": norms_roles, "population_counts": population_counts,
            "program_counts": program_counts,
            "w1w3_fused16_count": fused16_count, "w1w3_rms_affine16_count": rms_affine16_count,
            "w1w3_rms_affine32_count": rms_affine32_count,
            "norm_epilogue_count": norm_epilogue_count, "scale_reduce_count": scale_reduce_count,
            "ffn_residual_add_count": family(RESADD_PREFIX),
            "block_output_copy_count": family(BLOCK_OUTPUT_COPY_PREFIX),
            "attention_cast_count": family(ATTENTION_CAST_PREFIX),
            "flash_combine_f16_count": family(COMBINE_F16_PREFIX),
            "flash_combine_f32_count": family(COMBINE_F32_PREFIX) - family(COMBINE_F16_PREFIX),
            "combine_copy_count": family(COMBINE_COPY_PREFIX),
            "ffn_down_resadd_count": sum(count for name, count in program_counts.items()
                                         if name.startswith(FFNRESADD_PREFIX) or name.endswith(FFNRESADD_SUFFIX)),
            "norm_epilogue_us": round(sum(us for name, us in rows if name.startswith(NORM_EPILOGUE_PREFIX)), 3),
            "scale_reduce_us": round(sum(us for name, us in rows if name.startswith(REDUCE_PREFIX)), 3),
            "rms_affine16_us": round(sum(us for name, us in rows if name.startswith(RMS_AFFINE16_PREFIX)), 3),
            "fused16_us": round(sum(us for name, us in rows if name.startswith(FUSED16_PREFIX)), 3),
            "histogram": sorted(((name, len(vals), statistics.median(vals)) for name, vals in hist.items()),
                                key=lambda row: (-row[1], -row[2]))}


def timing_child(arm: str, model_path: str, depth: int, count: int, max_context: int,
                 reps: int, settled_continuous: bool) -> dict:
  from tinygrad import Device
  with _m1_arm_context(arm):
    model, gates = _model(arm, model_path, max_context)
    prompt, dev = _prompt(model_path, depth), Device[Device.DEFAULT]
    if settled_continuous:
      gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
      try: settled = _settled_continuous_windows(gen, dev, count, reps)
      finally: gen.close()
      return {"schema": TIMING_SCHEMA, "arm": arm, "gates": gates, "settled_continuous": True,
              "warmup_decode_calls": 6, "reps": reps, "tokens_per_rep": count, **settled}
    samples, hashes = [], []
    warm = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
    try:
      for _ in range(3): next(warm)
    finally: warm.close()
    for _ in range(reps):
      model.reset_generation_state()
      gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
      out = []
      try:
        next(gen); dev.synchronize(); started = time.perf_counter_ns()
        for _ in range(count): out.append(int(next(gen)))
        dev.synchronize(); samples.append((time.perf_counter_ns() - started) / count / 1e6)
      finally: gen.close()
      hashes.append(hashlib.sha256(",".join(map(str, out)).encode()).hexdigest())
    return {"schema": TIMING_SCHEMA, "arm": arm, "gates": gates, "settled_continuous": False,
            "reps": reps, "tokens_per_rep": count, "samples_ms_per_token": samples,
            "median_ms_per_token": statistics.median(samples), "token_hashes": hashes,
            "tokens_identical_within_arm": len(set(hashes)) == 1}


def validate_census(control: dict, candidate: dict) -> dict:
  """M1 census gate.  The expected drops are derived from the freshly measured
  control arm: the ffn-norm epilogue count must drop by exactly the control
  fused16 count (36 ffn chains), the fused16 kernels must vanish 1:1 with the
  rms_affine16 bodies appearing, the r_16_256 scale-reduce count must stay
  IDENTICAL (the route renders the same reduce program), the net program delta
  must equal the fused16 drop, and no OTHER program count may shift (all M2
  families byte-identical).  FAIL CLOSED with the exact evidence on any
  violation."""
  for label, row in (("control", control), ("candidate", candidate)):
    if row.get("schema") != CENSUS_SCHEMA:
      raise ValueError(f"{label} census row requires schema {CENSUS_SCHEMA!r}, got {row.get('schema')!r}")
  control_counts = control.get("program_counts") or {}
  candidate_counts = candidate.get("program_counts") or {}
  fused16_control = int(control.get("w1w3_fused16_count", 0))
  fused16_candidate = int(candidate.get("w1w3_fused16_count", 0))
  affine16_control = int(control.get("w1w3_rms_affine16_count", 0))
  affine16_candidate = int(candidate.get("w1w3_rms_affine16_count", 0))
  affine32_candidate = int(candidate.get("w1w3_rms_affine32_count", 0))
  epilogue_control = int(control.get("norm_epilogue_count", 0))
  epilogue_candidate = int(candidate.get("norm_epilogue_count", 0))
  reduce_control = int(control.get("scale_reduce_count", 0))
  reduce_candidate = int(candidate.get("scale_reduce_count", 0))
  net_delta = int(candidate.get("kernels", 0)) - int(control.get("kernels", 0))
  resadd_candidate = int(candidate.get("ffn_residual_add_count", 0))
  resadd_control = int(control.get("ffn_residual_add_count", 0))
  copy_candidate = int(candidate.get("block_output_copy_count", 0))
  copy_control = int(control.get("block_output_copy_count", 0))
  attn_cast_candidate = int(candidate.get("attention_cast_count", 0))
  attn_cast_control = int(control.get("attention_cast_count", 0))
  combine_f16_control = int(control.get("flash_combine_f16_count", 0))
  combine_f16_candidate = int(candidate.get("flash_combine_f16_count", 0))
  combine_f32_control = int(control.get("flash_combine_f32_count", 0))
  combine_f32_candidate = int(candidate.get("flash_combine_f32_count", 0))
  combine_copy_control = int(control.get("combine_copy_count", 0))
  combine_copy_candidate = int(candidate.get("combine_copy_count", 0))
  ffnresadd_control = int(control.get("ffn_down_resadd_count", 0))
  ffnresadd_candidate = int(candidate.get("ffn_down_resadd_count", 0))
  side_effects = {name: int(candidate_counts.get(name, 0)) - int(control_counts.get(name, 0))
                  for name in sorted(set(control_counts) | set(candidate_counts))
                  if int(candidate_counts.get(name, 0)) != int(control_counts.get(name, 0))}
  allowed_side_effects = {
    name for name in side_effects
    if name.startswith(FUSED16_PREFIX) or name.startswith(RMS_AFFINE16_PREFIX)
    or name.startswith(NORM_EPILOGUE_PREFIX)}
  unrelated_deltas = {name: delta for name, delta in side_effects.items() if name not in allowed_side_effects}
  control_pops = control.get("population_counts") or {}
  candidate_pops = candidate.get("population_counts") or {}
  population_deltas = {pop: int(candidate_pops.get(pop, 0)) - int(control_pops.get(pop, 0))
                       for pop in sorted(set(control_pops) | set(candidate_pops)) if pop != POP_NORMS
                       if int(candidate_pops.get(pop, 0)) != int(control_pops.get(pop, 0))}
  conditions = {
    "control_is_landed_m2d_candidate": (resadd_control == 0 and copy_control == 0 and attn_cast_control == 0
                                        and combine_f16_control > 0 and combine_f32_control == 0
                                        and combine_copy_control == 0 and ffnresadd_control > 0 and fused16_control > 0),
    "m1_ffn_epilogues_folded": epilogue_candidate == epilogue_control - fused16_control,
    "m1_fused16_gone": fused16_candidate == 0,
    "m1_affine16_swap_backed": affine16_candidate == fused16_control and affine16_control == 0 and affine32_candidate == 0,
    "m1_scale_reduce_identical": reduce_candidate == reduce_control,
    "m2_families_identical": (resadd_candidate == 0 and copy_candidate == 0 and attn_cast_candidate == 0
                              and combine_f16_candidate == combine_f16_control and combine_f32_candidate == 0
                              and combine_copy_candidate == 0 and ffnresadd_candidate == ffnresadd_control),
    "net_delta_matches_drop": net_delta == -fused16_control,
    "no_unrelated_program_shift": not unrelated_deltas,
  }
  fail_closed = []
  if not conditions["control_is_landed_m2d_candidate"]:
    fail_closed.append(f"FAIL CLOSED: control is not the LANDED M2d candidate (resadd {resadd_control}, copies {copy_control}, attn cast {attn_cast_control}, f16/f32 combine {combine_f16_control}/{combine_f32_control}, opaque {combine_copy_control}, ffnresadd {ffnresadd_control}, fused16 {fused16_control})")
  if not conditions["m1_ffn_epilogues_folded"]:
    fail_closed.append(f"FAIL CLOSED: ffn-norm epilogues must drop {epilogue_control} -> {epilogue_control - fused16_control}, got {epilogue_candidate}")
  if not conditions["m1_fused16_gone"]:
    fail_closed.append(f"FAIL CLOSED: fused16 kernels remain ({fused16_candidate})")
  if not conditions["m1_affine16_swap_backed"]:
    fail_closed.append(f"FAIL CLOSED: rms_affine16 {affine16_control} -> {affine16_candidate} must equal the control fused16 {fused16_control} (fp32 variant {affine32_candidate})")
  if not conditions["m1_scale_reduce_identical"]:
    fail_closed.append(f"FAIL CLOSED: r_16_256 scale-reduce count shifted {reduce_control} -> {reduce_candidate}; the route scale must render the same reduce program")
  if not conditions["m2_families_identical"]:
    fail_closed.append(f"FAIL CLOSED: M2 families shifted (resadd {resadd_candidate}, copies {copy_candidate}, attn cast {attn_cast_candidate}, f16/f32 combine {combine_f16_candidate}/{combine_f32_candidate}, opaque {combine_copy_candidate}, ffnresadd {ffnresadd_candidate})")
  if not conditions["net_delta_matches_drop"]:
    fail_closed.append(f"FAIL CLOSED: net program delta {net_delta} != -{fused16_control} (fused16 drop)")
  if not conditions["no_unrelated_program_shift"]:
    fail_closed.append(f"FAIL CLOSED: unrelated program-count shifts: {unrelated_deltas}")
  return {"fused16_control": fused16_control, "fused16_candidate": fused16_candidate,
          "rms_affine16_control": affine16_control, "rms_affine16_candidate": affine16_candidate,
          "rms_affine32_candidate": affine32_candidate,
          "norm_epilogue_control": epilogue_control, "norm_epilogue_candidate": epilogue_candidate,
          "scale_reduce_control": reduce_control, "scale_reduce_candidate": reduce_candidate,
          "ffn_residual_add_control": resadd_control, "ffn_residual_add_candidate": resadd_candidate,
          "block_output_copy_control": copy_control, "block_output_copy_candidate": copy_candidate,
          "attention_cast_control": attn_cast_control, "attention_cast_candidate": attn_cast_candidate,
          "flash_combine_f16_control": combine_f16_control, "flash_combine_f16_candidate": combine_f16_candidate,
          "flash_combine_f32_control": combine_f32_control, "flash_combine_f32_candidate": combine_f32_candidate,
          "combine_copy_control": combine_copy_control, "combine_copy_candidate": combine_copy_candidate,
          "ffn_down_resadd_control": ffnresadd_control, "ffn_down_resadd_candidate": ffnresadd_candidate,
          "honest_net_program_delta": net_delta,
          "program_side_effects": side_effects, "unrelated_program_deltas": unrelated_deltas,
          "non_norms_population_deltas": population_deltas,
          "conditions": conditions, "fail_closed": fail_closed,
          "note": "expected drops derived from the measured control arm; the rms_affine16 body applies (half)((h*s)*w) per packed-Q4 load with the bitwise-exact control scale rendered by the same r_16_256 reduce program; the fused16 -> rms_affine16 swap is 1:1; the fp16 store keeps the ffn-activation cast folded and the ffn_down fp16 ABI intact",
          "gate_pass": bool(all(conditions.values())) and not fail_closed}


def validate_timing_bracket(rows: list[dict], settled_continuous: bool = True) -> dict:
  """Reverse control/candidate/control bracket; promotion requires +50 us/token
  vs both bracketing controls with an identical token stream."""
  if len(rows) != 3: raise ValueError("timing bracket needs exactly control/candidate/control rows")
  for index, row in enumerate(rows):
    if row.get("schema") != TIMING_SCHEMA:
      raise ValueError(f"timing bracket row {index} requires schema {TIMING_SCHEMA!r}, got {row.get('schema')!r}")
  hashes = _timing_hash_authority(rows, settled_continuous)
  controls = (rows[0]["median_ms_per_token"], rows[2]["median_ms_per_token"])
  candidate = rows[1]["median_ms_per_token"]
  deltas_us = [(control - candidate) * 1000.0 for control in controls]
  promoted = len(hashes) == 1 and all(delta >= PROMOTION_US for delta in deltas_us)
  return {"schema": SCHEMA, "mode": "wall-bracket", "arms": rows,
          "all_token_hashes_equal": len(hashes) == 1, "settled_continuous": settled_continuous,
          "control_a_ms": controls[0], "control_b_ms": controls[1],
          "control_bracket_median_ms": statistics.median(controls), "candidate_ms": candidate,
          "candidate_minus_control_a_us": deltas_us[0], "candidate_minus_control_b_us": deltas_us[1],
          "candidate_minus_control_bracket_us": (statistics.median(controls) - candidate) * 1000.0,
          "promotion_us": PROMOTION_US, "promoted": bool(promoted),
          "note": "wall evidence only; booking requires the exact-logits gate and the M1 census gate"}


def validate_cost_prediction(bracket: dict, control_census: dict, candidate_census: dict) -> dict:
  """M1 predicted-wall-delta gate (nv_fusion_cost_model.py).

  The COST_PREDICTION table is derived from the llama reference shape plus the
  per-element arithmetic of the candidate; the measured bracket delta must then
  confirm it or explain the gap.  A measured delta outside the predicted range
  on the opposite side of zero is a CONTRADICTION and FAILS CLOSED: the premise
  was unbacked by the llama-shaped arithmetic, so the campaign cannot book even
  if the raw bracket numbers promoted."""
  hist = control_census.get("histogram") or []
  removed_medians = {name: med for name, _, med in hist if name.startswith(NORM_EPILOGUE_PREFIX)}
  if not removed_medians:
    return {"run": True, "result": "NO-GO", "reason": "control census has no norm-epilogue family to model"}
  # All 37 control epilogues are the same kernel body, so the family median is the
  # per-block M_removed; only the 36 ffn chains fold, but the per-block term uses the
  # measured median of the identical body either way.
  prediction = predict_wall_delta(36, {NORM_EPILOGUE_PREFIX: statistics.median(removed_medians.values())},
                                  {NORM_EPILOGUE_PREFIX: COST_PREDICTION["arithmetic"]["redundancy"]})
  # Repo-wide bracket convention: candidate_minus_control_bracket_us is computed
  # as (control - candidate), so POSITIVE means the candidate is FASTER.  The cost
  # model reconciles candidate-minus-control (positive = candidate SLOWER), so the
  # field must be negated before reconciliation or a measured loss reads as a win.
  measured = -bracket["candidate_minus_control_bracket_us"]
  reconciliation = reconcile_cost_prediction(measured, prediction,
                                             tolerance_us=COST_PREDICTION["tolerance_us"])
  return {"run": True, "result": "PASS" if reconciliation["result"] != "CONTRADICTED" else "FAIL",
          "contract": COST_PREDICTION, "prediction": prediction, "reconciliation": reconciliation,
          "measured_delta_us": measured, "bracket_field_us": bracket["candidate_minus_control_bracket_us"],
          "note": reconciliation["note"]}


HARD_STOP_NOTES = [
  "Every GPU arm runs as a fresh process under `timeout ... flock -w 90 /tmp/gpu-bench.lock`; no arm holds the lock across a wall-bracket step.",
  "Phase 0 (NV render smoke) must survive on sm_120 (no Xid 31 MMU fault) with the rms_affine16 kernels (q4k_g3_lanemap_w1w3_rms_affine16_*) in the compiled program set, no fused16 kernel, the ffn-norm epilogue reduced to the non-FFN chain, no E_32_32_4_02a9738c residual add, no E_32_32_4_fab82d40 copy, no E_32_32_4_0a5eb0ac attention cast, no legacy fp32 combine, and no E_32_32_4_3b0fcfbc opaque copy.",
  "The exact full-logit gate (fp32 SHA-256 over the stacked rows, token stream, shape, per-row argmax == sampled token) must pass before any census or bracket arm runs.",
  "The census gate FAILS CLOSED if the ffn-norm epilogues do not drop by exactly the control fused16 count, if the fused16 kernels remain, if the rms_affine16 bodies do not swap 1:1, if the r_16_256 scale-reduce count shifts, if any M2 family shifts, or if any unrelated program count shifts; expected counts derive from the measured control arm.",
  "The wall bracket requires identical token-stream hashes and a candidate median at least +50 us/token faster than BOTH bracketing controls.",
  "The predicted-wall-delta gate (COST_PREDICTION + validate_cost_prediction) runs after the bracket: the measured delta must CONFIRM the llama-shaped prediction or EXPLAIN the gap with named residual causes; a CONTRADICTION (measured outside the predicted range on the opposite side of zero) FAILS CLOSED and the campaign cannot book.",
  "No policy promotion: no route-policy record changes; the lease attribute is harness-installed only.",
]

ISOLATION_NOTES = [
  "The M1 control arm IS the LANDED M2d candidate (same callify flags, reduce-output promotion, the M2a fp16-store lease, the M2b ffn_down residual-add lease, and the M2d flash-combine fp16-store lease), so the only inter-arm delta is the M1 norm-epilogue lease.",
  "The exact-logits gate runs the eager JIT=0 finite check inside the child before comparing stacked-row SHAs, and any non-finite row fails closed.",
  "Unlike M2d, no eager lease-strip is needed: the M1 fp16 round is real in BOTH the eager JIT=0 pass and the captured decode graph (control renders E_32_32_4_f14a5cc0 in the eager pass too), and the fused values are bitwise-identical to control, so the cache written at `depth` is arm-invariant.",
]

CITATIONS = [
  "docs/task_workflow/input/nv-epilogue-absorption-route-scope-20260810.md",
  "extra/llm_research/decode/nv_epilogue_absorption_ab.py",
  "tinygrad/llm/decode_kernels.py",
  "tinygrad/llm/decode_routes.py",
  "tinygrad/llm/model.py",
]


def no_go_record(model: str = DEFAULT_MODEL, depth: int = 512) -> dict:
  """Base NO-GO record; the orchestrator overwrites each phase with the exact
  evidence as the campaign advances."""
  return {
    "schema": SCHEMA, "mode": "ab", "date": "2026-08-12",
    "target": {"model": model, "depth": depth, "device": "NV sm_120", "gpu": "RTX 5090"},
    "question": CONSTRUCTION["question"], "construction": CONSTRUCTION,
    "verdict": "NO-GO",
    "smoke": {"run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the NV render smoke arm"},
    "logits_gate": {
      "run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the exact full-logit arm",
      "contract": "full fp32 logits SHA-256 over the stacked rows identical to control; identical token stream; identical shape; per-row argmax equals the sampled token",
    },
    "census": {
      "run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the M1 census arm",
      "contract": "E_32_32_4_f14a5cc0 dropped by exactly the control fused16 count; fused16 at zero; rms_affine16 bodies 1:1 with the control fused16 count; r_16_256 counts identical; M2 families byte-identical; net program delta equals the fused16 drop; no unrelated program-count shift; FAIL CLOSED on any violation",
    },
    "wall_bracket": {
      "run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the reverse control/candidate/control wall bracket",
      "promotion_us": PROMOTION_US,
      "contract": "all three token-stream hashes identical; candidate median at least +50 us/token faster than both bracketing controls",
    },
    "cost_prediction": {
      "run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the predicted-wall-delta reconciliation",
      "contract": "the measured bracket delta must CONFIRM the llama-shaped prediction or EXPLAIN the gap with named residual causes; CONTRADICTED fails closed",
    },
    "hard_stop_notes": list(HARD_STOP_NOTES),
    "isolation_notes": list(ISOLATION_NOTES),
    "citations": CITATIONS,
  }


def _child_command(args, mode: str, arm: str, out: pathlib.Path, include_reps: bool = True) -> list[str]:
  cmd = ["timeout", f"{args.timeout}s", "flock", "-w", str(args.lock_wait), args.lock,
         sys.executable, str(pathlib.Path(__file__).resolve()), "--mode", mode, "--arm", arm,
         "--model", args.model, "--depth", str(args.depth), "--count", str(args.count),
         "--max-context", str(args.max_context), "--out", str(out)]
  if include_reps: cmd += ["--reps", str(args.reps)]
  if args.settled_continuous and mode == "timing-child": cmd.append("--settled-continuous")
  return cmd


def _guarded_child(record: dict, phase: str, cmd: list[str], out: pathlib.Path, fail_reason: str) -> dict | None:
  """Run one campaign child; on failure record a NO-GO phase with the raw
  child stderr instead of letting the orchestrator crash without evidence."""
  try:
    return _run_child(cmd, out)
  except RuntimeError as exc:
    record[phase] = {"run": True, "result": "NO-GO", "reason": fail_reason,
                     "stderr": getattr(exc, "stderr", None) or str(exc)[-4000:]}
    record["hard_stop_notes"] = HARD_STOP_NOTES + [f"HARD STOP at {phase}: {fail_reason}"]
    return None


def wall_bracket(args) -> dict:
  """Serialized reverse control/candidate/control timing, each child a fresh
  process under ``timeout ... flock -w`` on the shared GPU bench lock."""
  root = _child_root(pathlib.Path(args.out), ".timing")
  root.mkdir(parents=True, exist_ok=True)
  rows = []
  for sequence, arm in enumerate(("control", "candidate", "control")):
    out = root / f"{arm}-{sequence}.json"
    rows.append(_run_child(_child_command(args, "timing-child", arm, out), out))
  return validate_timing_bracket(rows, args.settled_continuous)


def ab(args) -> dict:
  """Campaign orchestrator: smoke -> exact logits -> census -> wall bracket.
  HARD STOP with a NO-GO record at the first failed gate; BOOKED only when
  every gate passes and the bracket promotes."""
  root = _child_root(pathlib.Path(args.out), ".children")
  root.mkdir(parents=True, exist_ok=True)
  record = no_go_record(args.model, args.depth)
  smoke_out = root / "smoke-candidate.json"
  smoke_result = _guarded_child(record, "smoke",
                                _child_command(args, "smoke", "candidate", smoke_out, include_reps=False),
                                smoke_out,
                                "NV render smoke child failed (Xid 31 class); raw child stderr captured below")
  if smoke_result is None:
    return _write_record(record, pathlib.Path(args.out))
  smoke_gate = (smoke_result.get("survive") is True
                and int(smoke_result.get("rms_affine16_count", 0)) > 0
                and int(smoke_result.get("fused16_count", 0)) == 0
                and int(smoke_result.get("rms_affine32_count", 0)) == 0
                and int(smoke_result.get("norm_epilogue_count", 0)) <= 1
                and int(smoke_result.get("scale_reduce_count", 0)) > 0
                and int(smoke_result.get("residual_add_count", 0)) == 0
                and int(smoke_result.get("block_output_copy_count", 0)) == 0
                and int(smoke_result.get("attention_cast_count", 0)) == 0
                and int(smoke_result.get("combine_f16_count", 0)) > 0
                and int(smoke_result.get("combine_f32_count", 0)) == 0
                and int(smoke_result.get("combine_copy_count", 0)) == 0
                and int(smoke_result.get("ffn_down_resadd_count", 0)) > 0)
  record["smoke"] = {"run": True, "result": "PASS" if smoke_gate else "NO-GO", "evidence": smoke_result}
  if not smoke_gate:
    record["hard_stop_notes"] = HARD_STOP_NOTES + [
      "HARD STOP at Phase 0: smoke did not survive, the rms_affine16 bodies were absent, fused16 kernels remained, the ffn-norm epilogue count was not reduced, or an M2 family regressed."]
    return _write_record(record, pathlib.Path(args.out))
  control_logits = _guarded_child(record, "logits_gate",
                                  _child_command(args, "logits", "control", root / "control-logits.json", include_reps=False),
                                  root / "control-logits.json",
                                  "the control exact-logits child failed; raw child stderr captured below")
  if control_logits is None:
    return _write_record(record, pathlib.Path(args.out))
  candidate_logits = _guarded_child(record, "logits_gate",
                                    _child_command(args, "logits", "candidate", root / "candidate-logits.json", include_reps=False),
                                    root / "candidate-logits.json",
                                    "the candidate exact-logits child failed; raw child stderr captured below")
  if candidate_logits is None:
    return _write_record(record, pathlib.Path(args.out))
  logits_gate = validate_logits_gate(control_logits, candidate_logits)
  record["logits_gate"] = {"run": True, "result": "PASS" if logits_gate["gate_pass"] else "FAIL",
                           "control_evidence": control_logits, "candidate_evidence": candidate_logits, **logits_gate}
  if not logits_gate["gate_pass"]:
    record["hard_stop_notes"] = HARD_STOP_NOTES + [
      "HARD STOP at Phase 1: exact full-logit gate FAIL (bitwise logits differ or token stream diverged)."]
    return _write_record(record, pathlib.Path(args.out))
  control_census = _guarded_child(record, "census",
                                  _child_command(args, "census", "control", root / "control-census.json", include_reps=False),
                                  root / "control-census.json",
                                  "the control census child failed; raw child stderr captured below")
  if control_census is None:
    return _write_record(record, pathlib.Path(args.out))
  candidate_census = _guarded_child(record, "census",
                                    _child_command(args, "census", "candidate", root / "candidate-census.json", include_reps=False),
                                    root / "candidate-census.json",
                                    "the candidate census child failed; raw child stderr captured below")
  if candidate_census is None:
    return _write_record(record, pathlib.Path(args.out))
  census_gate = validate_census(control_census, candidate_census)
  record["census"] = {"run": True, "result": "PASS" if census_gate["gate_pass"] else "FAIL",
                      "control_evidence": control_census, "candidate_evidence": candidate_census, **census_gate}
  if not census_gate["gate_pass"]:
    record["hard_stop_notes"] = HARD_STOP_NOTES + [
      "HARD STOP at Phase 2: M1 census gate FAIL (ffn epilogues not folded, fused16 remained, rms_affine16 swap not 1:1, r_16_256 shifted, M2 family shifted, or unrelated program shift)."]
    return _write_record(record, pathlib.Path(args.out))
  bracket = wall_bracket(args)
  record["wall_bracket"] = {"run": True, "result": "PROMOTED" if bracket["promoted"] else "NO-GO", **bracket}
  cost_gate = validate_cost_prediction(bracket, control_census, candidate_census)
  record["cost_prediction"] = cost_gate
  if cost_gate["result"] == "FAIL":
    record["verdict"] = "NO-GO"
    record["hard_stop_notes"] = HARD_STOP_NOTES + [
      "HARD STOP at Phase 4: predicted-wall-delta CONTRADICTION (measured delta outside the llama-shaped prediction range on the opposite side of zero); the premise was unbacked and cannot book even if the raw bracket numbers promoted."]
    return _write_record(record, pathlib.Path(args.out))
  if bracket["promoted"]:
    record["verdict"] = "BOOKED"
    record["hard_stop_notes"] = HARD_STOP_NOTES
  else:
    record["hard_stop_notes"] = HARD_STOP_NOTES + [
      "HARD STOP at Phase 3: reverse wall bracket NO-GO (candidate not +50 us/token faster than both controls or token streams differ)."]
  return _write_record(record, pathlib.Path(args.out))


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--mode", choices=("smoke", "logits", "census", "timing-child", "ab"), required=True)
  ap.add_argument("--arm", choices=("control", "candidate"))
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=32)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--out", required=True)
  ap.add_argument("--timeout", type=int, default=600)
  ap.add_argument("--lock-wait", type=int, default=90)
  ap.add_argument("--lock", default="/tmp/gpu-bench.lock")
  ap.add_argument("--settled-continuous", action="store_true")
  args = ap.parse_args()
  _validate_run_extent(args.depth, args.count, args.max_context, args.reps, args.settled_continuous)
  if args.mode in ("smoke", "logits", "census", "timing-child") and args.arm is None:
    ap.error(f"--arm is required for --mode {args.mode}")
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  if args.mode == "smoke":
    if args.arm != "candidate":
      ap.error("smoke requires --arm candidate: the rms_affine16 bodies only exist under the candidate conditions")
    result = smoke(args.arm, args.model, args.depth, args.max_context)
  elif args.mode == "logits":
    result, array = logits(args.arm, args.model, args.depth, args.count, args.max_context)
    np.savez_compressed(out.with_suffix(".npz"), logits=array)
  elif args.mode == "census":
    result = census(args.arm, args.model, args.depth, args.max_context)
  elif args.mode == "timing-child":
    result = timing_child(args.arm, args.model, args.depth, args.count, args.max_context,
                          args.reps, args.settled_continuous)
  else:
    ab(args)
    return 0
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
