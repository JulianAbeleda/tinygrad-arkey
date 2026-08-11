#!/usr/bin/env python3
"""NV wall-bracket A/B for residual-family epilogue absorption (M2b: ffn_down in-kernel residual add).

Sibling of ``extra/llm_research/decode/nv_reduce_output_fp32_qk_ab.py`` (the
booked fp32 q/k norms route).  M2 attacks the unbooked +240.106 us
residual/cast/contiguous row (the 194 tok/s lever): per decode token the
ffn_down GEMVs (``q4k_g3_lanemap_gemv_4096_12288`` x18 +
``q6k_gen_coop_4096_12288_inkernel`` x18) store fp32 and the graph renders 36
ordinary ``E_32_32_4_02a9738c`` fp32 residual adds (``h + ffn_out``, ~61.7 us).
The candidate arm leases ``_ffn_down_resadd_lease`` on the model, every block,
and every ffn_down linear, so the ffn_down GEMVs render their own
``*_epi_ffnresadd`` variants that add the hidden-state residual h in-kernel
(``total + h[row]``, fp32 store -- the in-kernel add is the same fp32
expression the separate add kernel lowers, so the bytes are bitwise-identical)
and the standalone add folds away.  The fp16-store spelling of the original
M2b premise is NOT bitwise-safe: the next block's attention residual consumes
the fp32 block output, so storing fp16 would round the residual and break the
exact-logits gate (see the scope doc, section 3 M2b).

The control arm is the BOOKED M2a candidate conditions (callify flags +
``_decode_reduce_output_rmsnorm_promoted`` + ``_decode_direct_greedy_promoted``
+ the M2a ``_q4k_w1w3_fp16_store_lease``), WITHOUT the M2b lease.  Both arms
run as fresh processes under ``timeout ... flock -w`` on the shared GPU bench
lock so JIT capture and allocator state cannot leak across arms.

Gate order is fixed: NV render smoke (Xid 31 class) with the fused16 kernels
in the compiled set, then the exact full-logit gate, then the M2 census, then
the serialized reverse control/candidate/control wall bracket.  The census
gate FAILS CLOSED if the residual add remains (E_32_32_4_02a9738c still
rendered), if the ``*_epi_ffnresadd`` variants are absent, or if any unrelated
program count shifts; the expected drop is derived from the freshly measured
control arm, never a stale constant.  Any gate failure writes a NO-GO record
with the exact evidence; the residual row books only when every gate passes
and the bracket promotes at +50 us/token against BOTH bracketing controls.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, pathlib, statistics, sys, time
import numpy as np

from extra.llm_research.decode.nv_fusion_population_ledger import POP_NORMS, classify as _ledger_classify
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _load, _prompt
from extra.llm_research.decode.nv_reduce_output_primitive_ab import validate_logits_gate as _c6_validate_logits_gate
from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import (
  PROMOTION_US, TM_RE, _child_root, _digest, _require_candidate_callify_flags,
  _run_child, _settled_continuous_windows, _timing_hash_authority, _validate_run_extent,
  _write_record, tok_per_s,
)

SCHEMA = "tinygrad.nv_epilogue_absorption_ab.v1"
SMOKE_SCHEMA = "tinygrad.nv_epilogue_absorption_ab.smoke.v1"
LOGITS_SCHEMA = "tinygrad.nv_epilogue_absorption_ab.logits.v1"
CENSUS_SCHEMA = "tinygrad.nv_epilogue_absorption_ab.census.v1"
TIMING_SCHEMA = "tinygrad.nv_epilogue_absorption_ab.timing.v1"

LEASE = "_q4k_w1w3_fp16_store_lease"
LEASE2 = "_ffn_down_resadd_lease"
FUSED16_PREFIX = "q4k_g3_lanemap_gemv_w1w3fused16_"
FUSED_PREFIX = "q4k_g3_lanemap_gemv_w1w3fused_"
CAST_PREFIX = "E_128_32_3"
RESADD_PREFIX = "E_32_32_4_02a9738c"
FFNRESADD_PREFIX = "q4k_g3_lanemap_gemv_epi_ffnresadd_"
FFNRESADD_SUFFIX = "_epi_ffnresadd"

CONSTRUCTION = {
  "route": "residual-family epilogue absorption M2b (ffn_down in-kernel residual add)",
  "population": "residual_cast_contiguous",
  "mechanism": "the ffn_down Q4K/Q6K decode GEMVs add the hidden-state residual h in-kernel (total + h[row], fp32 store) under their own *_epi_ffnresadd names; the standalone fp32 h+ffn_out add (E_32_32_4_02a9738c) folds away; the in-kernel add is the same fp32 expression the separate add kernel lowers, so stored bytes are bitwise-identical; the fp16-store spelling is not bitwise-safe (next-block attention residual consumes the fp32 block output)",
  "codegen_path": "candidate sets _ffn_down_resadd_lease=True on the model, every block, and every ffn_down linear on top of the booked M2a candidate conditions (callify flags + _decode_reduce_output_rmsnorm_promoted + _decode_direct_greedy_promoted + _q4k_w1w3_fp16_store_lease); the block threads normed_h=h into ffn_down and skips its own h+ffn_out add; decode_routes picks the ffn_down_resadd epilogue (Q4K) or the q6k coop epilogue variant",
  "census_target": "E_32_32_4_02a9738c 36 -> 0; q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288 x18 + q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd x18; all other program counts byte-identical to control; honest net program delta -36",
  "correctness_contract": {
    "full_logit_fp32_sha256": "bitwise identical to control over the stacked rows",
    "token_stream": "identical to control",
    "per_row_argmax": "equals the sampled token",
    "promotion": "+50 us/token vs both bracketing controls (control / candidate / control)",
    "census": "E_32_32_4_02a9738c gone, *_epi_ffnresadd bodies present 1:1 with the control add count, no other program-count shift; FAIL CLOSED on any unrelated delta",
  },
  "question": "Does adding h+ffn_out in-kernel in the ffn_down GEMVs survive NV render (Xid 31 class), preserve exact full logits, remove exactly the E_32_32_4_02a9738c add programs with no other census shift, and book the residual-family share of the +240.106 us row under the reverse wall bracket?",
}


def _configure(model, arm: str) -> None:
  """Both arms set the booked M2a candidate conditions (the M2b control IS the
  booked M2a candidate); the candidate additionally installs the M2b
  ffn-down residual-add lease on the model, every block, and every ffn_down
  linear.  No loader policy creates either lease."""
  model._decode_direct_greedy_promoted = True
  _require_candidate_callify_flags()
  model._decode_reduce_output_rmsnorm_promoted = True
  for block in model.blk:
    block._decode_reduce_output_rmsnorm_promoted = True
  # M2a fp16-store lease is the M2b CONTROL baseline (present in both arms).
  setattr(model, LEASE, True)
  for block in model.blk: setattr(block, LEASE, True)
  if arm == "candidate":
    setattr(model, LEASE2, True)
    for block in model.blk:
      setattr(block, LEASE2, True)
      ffn_down = getattr(block, "ffn_down", None)
      if ffn_down is not None: setattr(ffn_down, LEASE2, True)
  elif arm != "control":
    raise ValueError(f"unknown arm {arm!r}")


def _gates(model) -> dict:
  return {
    "decode_direct_greedy_promoted": bool(getattr(model, "_decode_direct_greedy_promoted", False)),
    "reduce_output_rmsnorm_promoted": bool(getattr(model, "_decode_reduce_output_rmsnorm_promoted", False)),
    "block_reduce_output_rmsnorm_promoted": [
      bool(getattr(block, "_decode_reduce_output_rmsnorm_promoted", False)) for block in model.blk
    ] if getattr(model, "blk", None) else None,
    "w1w3_fp16_store_lease": bool(getattr(model, LEASE, False)),
    "block_w1w3_fp16_store_lease": [
      bool(getattr(block, LEASE, False)) for block in model.blk
    ] if getattr(model, "blk", None) else None,
    "ffn_down_resadd_lease": bool(getattr(model, LEASE2, False)),
    "block_ffn_down_resadd_lease": [
      bool(getattr(block, LEASE2, False)) for block in model.blk
    ] if getattr(model, "blk", None) else None,
    "ffn_down_linear_resadd_lease": [
      bool(getattr(getattr(block, "ffn_down", None), LEASE2, False)) for block in model.blk
    ] if getattr(model, "blk", None) else None,
  }


def _assert_control_closed(gates: dict) -> None:
  leased = []
  if gates.get("ffn_down_resadd_lease"):
    leased.append(f"model.{LEASE2}")
  for index, value in enumerate(gates.get("block_ffn_down_resadd_lease") or []):
    if value: leased.append(f"block[{index}].{LEASE2}")
  for index, value in enumerate(gates.get("ffn_down_linear_resadd_lease") or []):
    if value: leased.append(f"block[{index}].ffn_down.{LEASE2}")
  if leased:
    raise RuntimeError(f"control arm requires the booked M2a candidate WITHOUT the M2b lease, observed: {leased}")


def _assert_candidate_configured(gates: dict) -> None:
  missing = []
  if not gates.get("ffn_down_resadd_lease"):
    missing.append(f"model.{LEASE2}")
  for index, value in enumerate(gates.get("block_ffn_down_resadd_lease") or []):
    if not value: missing.append(f"block[{index}].{LEASE2}")
  for index, value in enumerate(gates.get("ffn_down_linear_resadd_lease") or []):
    if not value: missing.append(f"block[{index}].ffn_down.{LEASE2}")
  if missing:
    raise RuntimeError(f"candidate arm requires {LEASE2} on the model, every block, and every ffn_down linear: {missing}")


def validate_logits_gate(control: dict, candidate: dict) -> dict:
  """Exact-output gate with the M2 logits schema (mirror of the fp32 q/k gate:
  the shared C6 child contract plus this campaign's own schema name)."""
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
def _m2_arm_context(arm: str):
  """Both M2 arms run as the BOOKED fp32 q/k candidate: the callify Context
  flags are live for control AND candidate (the M2 control is the booked
  candidate WITHOUT the fp16-store lease), so the only census delta between
  the arms is the cast drop.  Do not reuse the closed-graph control context
  from the fp32 q/k campaign."""
  if arm not in ("control", "candidate"):
    raise ValueError(f"unknown arm {arm!r}")
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    yield


def smoke(arm: str, model_path: str, depth: int, max_context: int) -> dict:
  """Phase 0 NV render smoke: compile and run one decode token under the
  candidate conditions.  Success is survival (no Xid 31 MMU fault) with the
  ``*_epi_ffnresadd`` kernels in the compiled program set and no
  ``E_32_32_4_02a9738c`` residual-add program."""
  if arm != "candidate":
    raise ValueError("smoke requires the candidate arm; the ffn_down_resadd kernels only exist under the candidate conditions")
  from tinygrad import Device
  from tinygrad.engine.jit import GraphAdmissionCensus, observe_graph_admissions
  from tinygrad.helpers import Context
  with _m2_arm_context(arm):
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
      return {"schema": SMOKE_SCHEMA, "arm": arm, "mode": "smoke", "gates": gates,
              "device": str(Device[Device.DEFAULT]), "survive": True,
              "prelude_token": prelude, "token": token,
              "decode_observation": "second decode token after the prelude (index 1 of 3), mirroring capture_decode_graph",
              "fused16_body_present": bool(any(name.startswith(FUSED16_PREFIX) for name in programs)),
              "fused_body_present": bool(any(name.startswith(FUSED_PREFIX) for name in programs)),
              "cast_present": bool(any(name.startswith(CAST_PREFIX) for name in programs)),
              "ffn_resadd_body_present": bool(any(name.startswith(FFNRESADD_PREFIX) or name.endswith(FFNRESADD_SUFFIX) for name in programs)),
              "residual_add_present": bool(any(name.startswith(RESADD_PREFIX) for name in programs)),
              "program_count": len(programs), "program_names": programs}
    finally:
      gen.close()


def logits(arm: str, model_path: str, depth: int, count: int, max_context: int) -> tuple[dict, np.ndarray]:
  from tinygrad import Tensor, UOp
  from tinygrad.helpers import Context
  with _m2_arm_context(arm):
    model, gates = _model(arm, model_path, max_context)
    gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
    try: prelude = int(next(gen))
    finally: gen.close()
    token, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
    start_pos = UOp.variable("start_pos", 0, max_context - 1)
    with Context(JIT=0): _, eager_logits = model.forward_with_logits(token, start_pos.bind(depth), temp)
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
  ledger; the M2 families (fused16 / fused / E_128_32_3 / E_32_32_4_02a9738c /
  *_epi_ffnresadd) counted by exact name."""
  from tinygrad.helpers import Context
  with _m2_arm_context(arm):
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
    cast_count = sum(count for name, count in program_counts.items() if name.startswith(CAST_PREFIX))
    fused16_count = sum(count for name, count in program_counts.items() if name.startswith(FUSED16_PREFIX))
    fused_count = sum(count for name, count in program_counts.items() if name.startswith(FUSED_PREFIX))
    residual_add_count = sum(count for name, count in program_counts.items() if name.startswith(RESADD_PREFIX))
    ffn_down_resadd_count = sum(count for name, count in program_counts.items()
                                if name.startswith(FFNRESADD_PREFIX) or name.endswith(FFNRESADD_SUFFIX))
    return {"schema": CENSUS_SCHEMA, "arm": arm, "mode": "census", "gates": gates, "token": token,
            "kernels": len(rows), "kernel_us": round(sum(us for _, us in rows), 3),
            "norms_kernels": sum(1 for name, _ in rows if _ledger_classify(name)[0] == POP_NORMS),
            "norms_roles": norms_roles, "population_counts": population_counts,
            "program_counts": program_counts,
            "ffn_activation_cast_count": cast_count, "w1w3_fused16_count": fused16_count,
            "w1w3_fused_count": fused_count, "cast_us": round(sum(us for name, us in rows if name.startswith(CAST_PREFIX)), 3),
            "fused16_us": round(sum(us for name, us in rows if name.startswith(FUSED16_PREFIX)), 3),
            "fused_us": round(sum(us for name, us in rows if name.startswith(FUSED_PREFIX)), 3),
            "ffn_residual_add_count": residual_add_count,
            "ffn_down_resadd_count": ffn_down_resadd_count,
            "ffn_residual_add_us": round(sum(us for name, us in rows if name.startswith(RESADD_PREFIX)), 3),
            "ffn_down_resadd_us": round(sum(us for name, us in rows
                                            if name.startswith(FFNRESADD_PREFIX) or name.endswith(FFNRESADD_SUFFIX)), 3),
            "histogram": sorted(((name, len(vals), statistics.median(vals)) for name, vals in hist.items()),
                                key=lambda row: (-row[1], -row[2]))}


def timing_child(arm: str, model_path: str, depth: int, count: int, max_context: int,
                 reps: int, settled_continuous: bool) -> dict:
  from tinygrad import Device
  with _m2_arm_context(arm):
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
  """M2b census gate.  The expected add drop is derived from the freshly
  measured control arm (never a stale constant): every E_32_32_4_02a9738c
  residual-add program must vanish, exactly that many ``*_epi_ffnresadd``
  bodies must appear in its place, the net program delta must equal the drop,
  and no OTHER program count may shift (the M2a fused16/cast families must be
  byte-identical between arms since both run the booked M2a candidate).
  FAIL CLOSED with the exact evidence on any violation."""
  for label, row in (("control", control), ("candidate", candidate)):
    if row.get("schema") != CENSUS_SCHEMA:
      raise ValueError(f"{label} census row requires schema {CENSUS_SCHEMA!r}, got {row.get('schema')!r}")
  control_counts = control.get("program_counts") or {}
  candidate_counts = candidate.get("program_counts") or {}
  cast_control = int(control.get("ffn_activation_cast_count", 0))
  cast_candidate = int(candidate.get("ffn_activation_cast_count", 0))
  fused16_control = int(control.get("w1w3_fused16_count", 0))
  fused16_candidate = int(candidate.get("w1w3_fused16_count", 0))
  fused_control = int(control.get("w1w3_fused_count", 0))
  fused_candidate = int(candidate.get("w1w3_fused_count", 0))
  resadd_control = int(control.get("ffn_residual_add_count", 0))
  resadd_candidate = int(candidate.get("ffn_residual_add_count", 0))
  ffnresadd_control = int(control.get("ffn_down_resadd_count", 0))
  ffnresadd_candidate = int(candidate.get("ffn_down_resadd_count", 0))
  net_delta = int(candidate.get("kernels", 0)) - int(control.get("kernels", 0))
  side_effects = {name: int(candidate_counts.get(name, 0)) - int(control_counts.get(name, 0))
                  for name in sorted(set(control_counts) | set(candidate_counts))
                  if int(candidate_counts.get(name, 0)) != int(control_counts.get(name, 0))}
  allowed_side_effects = {
    name for name in side_effects
    if name.startswith(CAST_PREFIX) or name.startswith(FUSED16_PREFIX) or name.startswith(FUSED_PREFIX)
    or name.startswith(RESADD_PREFIX) or name.startswith(FFNRESADD_PREFIX) or name.endswith(FFNRESADD_SUFFIX)}
  # M2b swap families: the plain ffn_down GEMV whose epilogue absorbs the add becomes its
  # *_epi_ffnresadd twin 1:1 (q4k_g3_lanemap_gemv_4096_12288 ->
  # q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288; the q6k coop twin appends the suffix). A
  # control-side negative delta is admitted only when the candidate renders the exact twin
  # count, and every *_epi_ffnresadd body must be backed by that twin (fail-closed: no stray
  # bodies without the plain family shrinking).
  twin_counts: dict[str, int] = {}
  for cname, ccount in candidate_counts.items():
    if cname.startswith(FFNRESADD_PREFIX) or cname.endswith(FFNRESADD_SUFFIX):
      twin_counts[cname.replace("_epi_ffnresadd", "")] = twin_counts.get(cname.replace("_epi_ffnresadd", ""), 0) + ccount
  ffnresadd_swaps = {name: -delta for name, delta in side_effects.items()
                     if delta < 0 and twin_counts.get(name, 0) == -delta}
  ffnresadd_bodies = {name: delta for name, delta in side_effects.items()
                      if delta > 0 and (name.startswith(FFNRESADD_PREFIX) or name.endswith(FFNRESADD_SUFFIX))}
  swap_backed = all(ffnresadd_swaps.get(name.replace("_epi_ffnresadd", ""), 0) == delta
                    for name, delta in ffnresadd_bodies.items())
  unrelated_deltas = {name: delta for name, delta in side_effects.items()
                      if name not in allowed_side_effects and name not in ffnresadd_swaps}
  control_pops = control.get("population_counts") or {}
  candidate_pops = candidate.get("population_counts") or {}
  population_deltas = {pop: int(candidate_pops.get(pop, 0)) - int(control_pops.get(pop, 0))
                       for pop in sorted(set(control_pops) | set(candidate_pops)) if pop != POP_NORMS
                       if int(candidate_pops.get(pop, 0)) != int(control_pops.get(pop, 0))}
  conditions = {
    "control_has_residual_adds": resadd_control > 0,
    "candidate_residual_adds_gone": resadd_candidate == 0,
    "ffnresadd_replaces_one_for_one": ffnresadd_candidate == resadd_control,
    "control_has_no_ffnresadd": ffnresadd_control == 0,
    "m2a_fused16_identical": fused16_candidate == fused16_control and fused16_control > 0,
    "m2a_casts_identical": cast_candidate == cast_control == 0,
    "m2a_fp32_fused_identical": fused_candidate == fused_control == 0,
    "net_delta_matches_drop": net_delta == -resadd_control,
    "no_unrelated_program_shift": not unrelated_deltas,
    "ffnresadd_swap_backed": swap_backed,
  }
  fail_closed = []
  if not conditions["control_has_residual_adds"]:
    fail_closed.append("FAIL CLOSED: the control census has no E_32_32_4_02a9738c residual adds to absorb")
  if not conditions["candidate_residual_adds_gone"]:
    fail_closed.append(f"FAIL CLOSED: candidate still renders {resadd_candidate} E_32_32_4_02a9738c add programs (control had {resadd_control})")
  if not conditions["ffnresadd_replaces_one_for_one"]:
    fail_closed.append(f"FAIL CLOSED: *_epi_ffnresadd bodies {ffnresadd_candidate} do not replace the {resadd_control} control adds 1:1")
  if not conditions["control_has_no_ffnresadd"]:
    fail_closed.append(f"FAIL CLOSED: control renders {ffnresadd_control} *_epi_ffnresadd bodies (expected 0)")
  if not conditions["m2a_fused16_identical"]:
    fail_closed.append(f"FAIL CLOSED: M2a fused16 counts differ between arms ({fused16_control} -> {fused16_candidate})")
  if not conditions["m2a_casts_identical"]:
    fail_closed.append(f"FAIL CLOSED: M2a cast counts differ ({cast_control} -> {cast_candidate})")
  if not conditions["m2a_fp32_fused_identical"]:
    fail_closed.append(f"FAIL CLOSED: M2a fp32 fused counts differ ({fused_control} -> {fused_candidate})")
  if not conditions["net_delta_matches_drop"]:
    fail_closed.append(f"FAIL CLOSED: net program delta {net_delta} != -{resadd_control}")
  if not conditions["no_unrelated_program_shift"]:
    fail_closed.append(f"FAIL CLOSED: unrelated program-count shifts: {unrelated_deltas}")
  if not conditions["ffnresadd_swap_backed"]:
    fail_closed.append(f"FAIL CLOSED: *_epi_ffnresadd bodies without their plain ffn_down GEMV twin: {ffnresadd_bodies}")
  return {"cast_control": cast_control, "cast_candidate": cast_candidate,
          "fused16_control": fused16_control, "fused16_candidate": fused16_candidate,
          "fused_control": fused_control, "fused_candidate": fused_candidate,
          "ffn_residual_add_control": resadd_control, "ffn_residual_add_candidate": resadd_candidate,
          "ffn_down_resadd_control": ffnresadd_control, "ffn_down_resadd_candidate": ffnresadd_candidate,
          "honest_net_program_delta": net_delta,
          "program_side_effects": side_effects, "unrelated_program_deltas": unrelated_deltas,
          "ffnresadd_swaps": ffnresadd_swaps, "ffnresadd_bodies": ffnresadd_bodies,
          "non_norms_population_deltas": population_deltas,
          "conditions": conditions, "fail_closed": fail_closed,
          "note": "expected drop derived from the measured control arm; the *_epi_ffnresadd body is the same fp32 add expression fused into the GEMV epilogue (bitwise-identical bytes)",
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
          "note": "wall evidence only; booking requires the exact-logits gate and the M2 census gate"}


HARD_STOP_NOTES = [
  "Every GPU arm runs as a fresh process under `timeout ... flock -w 90 /tmp/gpu-bench.lock`; no arm holds the lock across a wall-bracket step.",
  "Phase 0 (NV render smoke) must survive on sm_120 (no Xid 31 MMU fault) with the *_epi_ffnresadd kernels (q4k_g3_lanemap_gemv_epi_ffnresadd_* and q6k_gen_coop_*_epi_ffnresadd) in the compiled program set and no E_32_32_4_02a9738c residual add.",
  "The exact full-logit gate (fp32 SHA-256 over the stacked rows, token stream, shape, per-row argmax == sampled token) must pass before any census or bracket arm runs.",
  "The census gate FAILS CLOSED if the residual add remains, if the *_epi_ffnresadd bodies do not appear 1:1 with the control add count, if the M2a fused16/cast families shift between arms, or if any unrelated program count shifts; expected counts derive from the measured control arm.",
  "The wall bracket requires identical token-stream hashes and a candidate median at least +50 us/token faster than BOTH bracketing controls.",
  "No policy promotion: no route-policy record changes; the lease attribute is harness-installed only.",
]

ISOLATION_NOTES = [
  "The M2b control arm is the BOOKED M2a candidate (same callify flags, reduce-output promotion, and the M2a fp16-store lease), so the only inter-arm delta is the ffn_down residual-add lease.",
  "The exact-logits gate runs the eager JIT=0 finite check inside the child before comparing stacked-row SHAs, and any non-finite row fails closed.",
]

CITATIONS = [
  "docs/task_workflow/input/nv-epilogue-absorption-route-scope-20260810.md",
  "docs/task_workflow/input/nv-reduce-output-fp32-qk-route-scope-20260810.md",
  "extra/llm_research/decode/nv_reduce_output_fp32_qk_ab.py",
  "tinygrad/llm/decode_kernels.py",
  "tinygrad/llm/decode_routes.py",
]


def no_go_record(model: str = DEFAULT_MODEL, depth: int = 512) -> dict:
  """Base NO-GO record; the orchestrator overwrites each phase with the exact
  evidence as the campaign advances."""
  return {
    "schema": SCHEMA, "mode": "ab", "date": "2026-08-10",
    "target": {"model": model, "depth": depth, "device": "NV sm_120", "gpu": "RTX 5090"},
    "question": CONSTRUCTION["question"], "construction": CONSTRUCTION,
    "verdict": "NO-GO",
    "smoke": {"run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the NV render smoke arm"},
    "logits_gate": {
      "run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the exact full-logit arm",
      "contract": "full fp32 logits SHA-256 over the stacked rows identical to control; identical token stream; identical shape; per-row argmax equals the sampled token",
    },
    "census": {
      "run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the M2 census arm",
      "contract": "E_32_32_4_02a9738c gone; *_epi_ffnresadd bodies 1:1 with the control add count; M2a fused16/cast families identical between arms; net program delta equals the add drop; no unrelated program-count shift; FAIL CLOSED on any violation",
    },
    "wall_bracket": {
      "run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the reverse control/candidate/control wall bracket",
      "promotion_us": PROMOTION_US,
      "contract": "all three token-stream hashes identical; candidate median at least +50 us/token faster than both bracketing controls",
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
                and bool(smoke_result.get("fused16_body_present"))
                and not bool(smoke_result.get("cast_present"))
                and bool(smoke_result.get("ffn_resadd_body_present"))
                and not bool(smoke_result.get("residual_add_present")))
  record["smoke"] = {"run": True, "result": "PASS" if smoke_gate else "NO-GO", "evidence": smoke_result}
  if not smoke_gate:
    record["hard_stop_notes"] = HARD_STOP_NOTES + [
      "HARD STOP at Phase 0: smoke did not survive, the fused16 bodies were absent, E_128_32_3 casts remained, the *_epi_ffnresadd bodies were absent, or E_32_32_4_02a9738c residual adds remained."]
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
      "HARD STOP at Phase 2: M2b census gate FAIL (residual add remained, *_epi_ffnresadd absent/mismatched, M2a families shifted, or unrelated program shift)."]
    return _write_record(record, pathlib.Path(args.out))
  bracket = wall_bracket(args)
  record["wall_bracket"] = {"run": True, "result": "PROMOTED" if bracket["promoted"] else "NO-GO", **bracket}
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
      ap.error("smoke requires --arm candidate: the fused16 bodies only exist under the candidate conditions")
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
