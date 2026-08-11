#!/usr/bin/env python3
"""NV wall-bracket A/B for residual-family epilogue absorption (M2a: fp16 store).

Sibling of ``extra/llm_research/decode/nv_reduce_output_fp32_qk_ab.py`` (the
booked fp32 q/k norms route).  M2 attacks the unbooked +240.106 us
residual/cast/contiguous row (the 194 tok/s lever): per decode token the fused
w1+w3 GEMV stores its result in fp32 and the graph renders 36 ordinary
``E_128_32_3`` ffn-activation casts (fp32 -> fp16, ~57.6 us) before the
ffn_down GEMV consumes the value.  The candidate arm leases
``_q4k_w1w3_fp16_store_lease`` on the model and every block, so the fused
kernel renders its own ``q4k_g3_lanemap_gemv_w1w3fused16_*`` variant that
stores the same value already cast to fp16 (the in-kernel cast is the same
round-to-nearest-even conversion the separate cast kernel lowers, so the
bytes are bitwise-identical) and the consumer cast folds away.

The control arm is the BOOKED fp32 q/k candidate conditions (callify flags +
``_decode_reduce_output_rmsnorm_promoted`` on the model and every block plus
``_decode_direct_greedy_promoted``), WITHOUT the M2 lease.  Both arms run as
fresh processes under ``timeout ... flock -w`` on the shared GPU bench lock so
JIT capture and allocator state cannot leak across arms.

Gate order is fixed: NV render smoke (Xid 31 class) with the fused16 kernels
in the compiled set, then the exact full-logit gate, then the M2 census, then
the serialized reverse control/candidate/control wall bracket.  The census
gate FAILS CLOSED if the cast remains (E_128_32_3 still rendered), if the
fused16 variant is absent, or if any unrelated program count shifts; the
expected drop is derived from the freshly measured control arm, never a stale
constant.  Any gate failure writes a NO-GO record with the exact evidence;
the residual row books only when every gate passes and the bracket promotes
at +50 us/token against BOTH bracketing controls.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, pathlib, statistics, sys, time
import numpy as np

from extra.llm_research.decode.nv_fusion_population_ledger import POP_NORMS, classify as _ledger_classify
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _load, _prompt
from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import (
  PROMOTION_US, TM_RE, _arm_context, _child_root, _digest, _require_candidate_callify_flags,
  _run_child, _settled_continuous_windows, _timing_hash_authority, _validate_run_extent,
  _write_record, tok_per_s, validate_logits_gate,
)

SCHEMA = "tinygrad.nv_epilogue_absorption_ab.v1"
SMOKE_SCHEMA = "tinygrad.nv_epilogue_absorption_ab.smoke.v1"
LOGITS_SCHEMA = "tinygrad.nv_epilogue_absorption_ab.logits.v1"
CENSUS_SCHEMA = "tinygrad.nv_epilogue_absorption_ab.census.v1"
TIMING_SCHEMA = "tinygrad.nv_epilogue_absorption_ab.timing.v1"

LEASE = "_q4k_w1w3_fp16_store_lease"
FUSED16_PREFIX = "q4k_g3_lanemap_gemv_w1w3fused16_"
FUSED_PREFIX = "q4k_g3_lanemap_gemv_w1w3fused_"
CAST_PREFIX = "E_128_32_3"

CONSTRUCTION = {
  "route": "residual-family epilogue absorption M2a (fp16 store)",
  "population": "residual_cast_contiguous",
  "mechanism": "the fused w1+w3 decode GEMV stores its result already cast to fp16 under its own q4k_g3_lanemap_gemv_w1w3fused16_* name; ffn_down's input cast (E_128_32_3, fp32->fp16) folds away; the in-kernel cast is the same round-to-nearest-even conversion the separate cast kernel lowers, so stored bytes are bitwise-identical",
  "codegen_path": "candidate sets _q4k_w1w3_fp16_store_lease=True on the model and every block on top of the booked fp32 q/k candidate conditions (callify flags + _decode_reduce_output_rmsnorm_promoted + _decode_direct_greedy_promoted); the route call q4k_gate_up_primitive_linear_call(..., store_fp16=True) picks the fused16 kernel and an fp16 OutputSpec",
  "census_target": "E_128_32_3 36 -> 0; q4k_g3_lanemap_gemv_w1w3fused16_12288_4096 x36; all other program counts byte-identical to control; honest net program delta -36",
  "correctness_contract": {
    "full_logit_fp32_sha256": "bitwise identical to control over the stacked rows",
    "token_stream": "identical to control",
    "per_row_argmax": "equals the sampled token",
    "promotion": "+50 us/token vs both bracketing controls (control / candidate / control)",
    "census": "E_128_32_3 gone, fused16 bodies present 1:1 with the control cast count, no other program-count shift; FAIL CLOSED on any unrelated delta",
  },
  "question": "Does storing the fused w1+w3 result as fp16 in-kernel survive NV render (Xid 31 class), preserve exact full logits, remove exactly the E_128_32_3 cast programs with no other census shift, and book the residual-family share of the +240.106 us row under the reverse wall bracket?",
}


def _configure(model, arm: str) -> None:
  """Both arms set the booked fp32 q/k candidate conditions (the M2 control is
  the booked candidate); the candidate additionally installs the fp16-store
  lease on the model and every block.  No loader policy creates the lease."""
  model._decode_direct_greedy_promoted = True
  _require_candidate_callify_flags()
  model._decode_reduce_output_rmsnorm_promoted = True
  for block in model.blk:
    block._decode_reduce_output_rmsnorm_promoted = True
  if arm == "candidate":
    setattr(model, LEASE, True)
    for block in model.blk: setattr(block, LEASE, True)
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
  }


def _assert_control_closed(gates: dict) -> None:
  leased = []
  if gates.get("w1w3_fp16_store_lease"):
    leased.append(f"model.{LEASE}")
  for index, value in enumerate(gates.get("block_w1w3_fp16_store_lease") or []):
    if value: leased.append(f"block[{index}].{LEASE}")
  if leased:
    raise RuntimeError(f"control arm requires the closed production graph, observed fp16-store leases: {leased}")


def _assert_candidate_configured(gates: dict) -> None:
  missing = []
  if not gates.get("w1w3_fp16_store_lease"):
    missing.append(f"model.{LEASE}")
  for index, value in enumerate(gates.get("block_w1w3_fp16_store_lease") or []):
    if not value: missing.append(f"block[{index}].{LEASE}")
  if missing:
    raise RuntimeError(f"candidate arm requires {LEASE} on the model and every block: {missing}")


def _model(arm: str, model_path: str, max_context: int):
  _require_candidate_callify_flags()
  model = _load(model_path, max_context)
  _configure(model, arm)
  gates = _gates(model)
  if arm == "control": _assert_control_closed(gates)
  else: _assert_candidate_configured(gates)
  return model, gates


def smoke(arm: str, model_path: str, depth: int, max_context: int) -> dict:
  """Phase 0 NV render smoke: compile and run one decode token under the
  candidate conditions.  Success is survival (no Xid 31 MMU fault) with the
  fused16 kernels (``q4k_g3_lanemap_gemv_w1w3fused16_*``) in the compiled
  program set and no ``E_128_32_3`` cast program."""
  if arm != "candidate":
    raise ValueError("smoke requires the candidate arm; the fused16 kernels only exist under the candidate conditions")
  from tinygrad import Device
  from tinygrad.engine.jit import GraphAdmissionCensus, observe_graph_admissions
  from tinygrad.helpers import Context
  with _arm_context(arm):
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
              "program_count": len(programs), "program_names": programs}
    finally:
      gen.close()


def logits(arm: str, model_path: str, depth: int, count: int, max_context: int) -> tuple[dict, np.ndarray]:
  from tinygrad import Tensor, UOp
  from tinygrad.helpers import Context
  with _arm_context(arm):
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
  ledger; the M2 families (fused16 / fused / E_128_32_3) counted by exact name."""
  from tinygrad.helpers import Context
  with _arm_context(arm):
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
    return {"schema": CENSUS_SCHEMA, "arm": arm, "mode": "census", "gates": gates, "token": token,
            "kernels": len(rows), "kernel_us": round(sum(us for _, us in rows), 3),
            "norms_kernels": sum(1 for name, _ in rows if _ledger_classify(name)[0] == POP_NORMS),
            "norms_roles": norms_roles, "population_counts": population_counts,
            "program_counts": program_counts,
            "ffn_activation_cast_count": cast_count, "w1w3_fused16_count": fused16_count,
            "w1w3_fused_count": fused_count, "cast_us": round(sum(us for name, us in rows if name.startswith(CAST_PREFIX)), 3),
            "fused16_us": round(sum(us for name, us in rows if name.startswith(FUSED16_PREFIX)), 3),
            "fused_us": round(sum(us for name, us in rows if name.startswith(FUSED_PREFIX)), 3),
            "histogram": sorted(((name, len(vals), statistics.median(vals)) for name, vals in hist.items()),
                                key=lambda row: (-row[1], -row[2]))}


def timing_child(arm: str, model_path: str, depth: int, count: int, max_context: int,
                 reps: int, settled_continuous: bool) -> dict:
  from tinygrad import Device
  with _arm_context(arm):
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
  """M2 census gate.  The expected cast drop is derived from the freshly
  measured control arm (never a stale constant): every E_128_32_3 cast must
  vanish, exactly that many fused16 bodies must appear in its place, no fused
  (fp32) w1w3 body may remain, the net program delta must equal the drop, and
  no OTHER program count may shift.  FAIL CLOSED with the exact evidence on
  any violation."""
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
  net_delta = int(candidate.get("kernels", 0)) - int(control.get("kernels", 0))
  side_effects = {name: int(candidate_counts.get(name, 0)) - int(control_counts.get(name, 0))
                  for name in sorted(set(control_counts) | set(candidate_counts))
                  if int(candidate_counts.get(name, 0)) != int(control_counts.get(name, 0))}
  allowed_side_effects = {
    name for name in side_effects
    if name.startswith(CAST_PREFIX) or name.startswith(FUSED16_PREFIX) or name.startswith(FUSED_PREFIX)}
  unrelated_deltas = {name: delta for name, delta in side_effects.items() if name not in allowed_side_effects}
  control_pops = control.get("population_counts") or {}
  candidate_pops = candidate.get("population_counts") or {}
  population_deltas = {pop: int(candidate_pops.get(pop, 0)) - int(control_pops.get(pop, 0))
                       for pop in sorted(set(control_pops) | set(candidate_pops)) if pop != POP_NORMS
                       if int(candidate_pops.get(pop, 0)) != int(control_pops.get(pop, 0))}
  conditions = {
    "control_has_casts": cast_control > 0,
    "candidate_casts_gone": cast_candidate == 0,
    "fused16_replaces_one_for_one": fused16_candidate == cast_control,
    "control_has_no_fused16": fused16_control == 0,
    "candidate_keeps_no_fp32_fused": fused_candidate == 0,
    "control_fused_matches_casts": fused_control == cast_control,
    "net_delta_matches_drop": net_delta == -cast_control,
    "no_unrelated_program_shift": not unrelated_deltas,
  }
  fail_closed = []
  if not conditions["control_has_casts"]:
    fail_closed.append("FAIL CLOSED: the control census has no E_128_32_3 casts to absorb")
  if not conditions["candidate_casts_gone"]:
    fail_closed.append(f"FAIL CLOSED: candidate still renders {cast_candidate} E_128_32_3 cast programs (control had {cast_control})")
  if not conditions["fused16_replaces_one_for_one"]:
    fail_closed.append(f"FAIL CLOSED: fused16 bodies {fused16_candidate} do not replace the {cast_control} control casts 1:1")
  if not conditions["candidate_keeps_no_fp32_fused"]:
    fail_closed.append(f"FAIL CLOSED: candidate keeps {fused_candidate} fp32 w1w3 fused bodies (expected 0)")
  if not conditions["net_delta_matches_drop"]:
    fail_closed.append(f"FAIL CLOSED: net program delta {net_delta} != -{cast_control}")
  if not conditions["no_unrelated_program_shift"]:
    fail_closed.append(f"FAIL CLOSED: unrelated program-count shifts: {unrelated_deltas}")
  return {"cast_control": cast_control, "cast_candidate": cast_candidate,
          "fused16_control": fused16_control, "fused16_candidate": fused16_candidate,
          "fused_control": fused_control, "fused_candidate": fused_candidate,
          "honest_net_program_delta": net_delta,
          "program_side_effects": side_effects, "unrelated_program_deltas": unrelated_deltas,
          "non_norms_population_deltas": population_deltas,
          "conditions": conditions, "fail_closed": fail_closed,
          "note": "expected drop derived from the measured control arm; the fused16 body is the same store expression wrapped in one half cast (bitwise-identical bytes)",
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
  "Phase 0 (NV render smoke) must survive on sm_120 (no Xid 31 MMU fault) with the fused16 kernels (q4k_g3_lanemap_gemv_w1w3fused16_*) in the compiled program set and no E_128_32_3 cast.",
  "The exact full-logit gate (fp32 SHA-256 over the stacked rows, token stream, shape, per-row argmax == sampled token) must pass before any census or bracket arm runs.",
  "The census gate FAILS CLOSED if the cast remains, if the fused16 bodies do not appear 1:1 with the control cast count, if any fp32 fused body remains, or if any unrelated program count shifts; expected counts derive from the measured control arm.",
  "The wall bracket requires identical token-stream hashes and a candidate median at least +50 us/token faster than BOTH bracketing controls.",
  "No policy promotion: no route-policy record changes; the lease attribute is harness-installed only.",
]

ISOLATION_NOTES = [
  "The M2 control arm is the BOOKED fp32 q/k candidate (same callify flags and reduce-output promotion), so the only inter-arm delta is the fp16-store lease.",
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
      "contract": "E_128_32_3 gone; fused16 bodies 1:1 with the control cast count; no fp32 fused body remains; net program delta equals the cast drop; no unrelated program-count shift; FAIL CLOSED on any violation",
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
                and not bool(smoke_result.get("cast_present")))
  record["smoke"] = {"run": True, "result": "PASS" if smoke_gate else "NO-GO", "evidence": smoke_result}
  if not smoke_gate:
    record["hard_stop_notes"] = HARD_STOP_NOTES + [
      "HARD STOP at Phase 0: smoke did not survive, the fused16 bodies were absent, or E_128_32_3 casts remained."]
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
      "HARD STOP at Phase 2: M2 census gate FAIL (cast remained, fused16 absent/mismatched, or unrelated program shift)."]
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
