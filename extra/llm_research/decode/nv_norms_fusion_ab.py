#!/usr/bin/env python3
"""Exact-output fusion A/B for the NORMS anchor-child population.

Campaign arm for the norms row of the NV decode fusion/dataflow workstream
(d512 Qwen3-8B-Q4_K_M on RTX 5090).  The construction under test is the
ffn/next RMSNorm *epilogue* absorbed as an ordinary in-core per-load affine
epilogue of the consuming quant kernel, with the norm reduce nodes staying
separate.  The admissible construction rule is the boundary-free ordinary-UOp
gate: ordinary UOps in-core only, no CUSTOM boundary, no adapters, no
CONTIGUOUS materialization, no lazy-view stripping.

The A/B order is fixed: boundary-free gate first, then the exact full-logit
gate, then the norms-confined census, then the serialized reverse
control/candidate/control wall bracket under the shared GPU bench lock.  Every
GPU arm runs as a fresh process so JIT capture and allocator state cannot leak
across arms.  A construction that fails the boundary-free gate or the
exact-logits gate stops the campaign: the record is written as NO-GO with the
exact evidence and the correctness contract is never weakened.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, os, pathlib, re, statistics, subprocess, sys, time
import numpy as np

from extra.llm_research.decode.nv_boundary_free_ordinary_uop_gate import run_v3 as _phase0_gate_run
from extra.llm_research.decode.nv_fusion_cost_model import predict_wall_delta, reconcile_cost_prediction
from extra.llm_research.decode.nv_fusion_population_ledger import POP_NORMS, classify as _ledger_classify
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _load, _prompt
from extra.llm_research.decode.nv_shared_q8_progressive_qualification import (
  _settled_continuous_windows, _timing_hash_authority, _validate_run_extent,
)

SCHEMA = "tinygrad.nv_norms_fusion_ab.v1"
GATE_SCHEMA = "tinygrad.nv_norms_fusion_ab.gate.v1"
LOGITS_SCHEMA = "tinygrad.nv_norms_fusion_ab.logits.v1"
CENSUS_SCHEMA = "tinygrad.nv_norms_fusion_ab.census.v1"
TIMING_SCHEMA = "tinygrad.nv_norms_fusion_ab.timing_child.v1"

DIM, EPS = 4096, 1e-6
PROMOTION_US = 50.0
# P2b redirect-on full-logit control authority (8 decoded rows, float32).
CONTROL_LOGITS_REFERENCE = "71c0a2b092cbc2e40c22b42cd4f6f3c84fe56fd40f2bfd008efc5b76be0ae0f0"
# Ledger reference on the redirect-on authority DAG (nv-fusion-exhaustive-scope-20260805.md).
CENSUS_REFERENCE = {
  "population": "norms", "node_count": 362, "total_us": 650.752,
  "fusion_candidate_count": 288, "fusion_candidate_us": 544.128,
  "anchor_child_epilogue_count": 215, "anchor_child_epilogue_us": 306.368,
  "reduction_count": 145,
}

CONSTRUCTION = {
  "route": "decode_norm_fusion",
  "population": "norms",
  "mechanism": "ffn/next RMSNorm epilogue absorbed as an ordinary in-core per-load affine epilogue of the consuming quant kernel; norm reduce nodes stay",
  "codegen_path": "y = (x.float() * s).cast(float16) * w with s the bitwise-shared fp32 reduce scalar rsqrt(mean(x^2)+eps), fused into the consuming q4k GEMV body as ordinary UOps",
  "boundary_rule": "ordinary UOps in-core only; no CUSTOM kernel boundary, no adapters, no CONTIGUOUS materialization, no lazy-view stripping",
  "census_target": "norms epilogue nodes removed (215 anchor-child epilogues / 306.368 us on the authority census); reduce nodes unchanged",
  "correctness_contract": {
    "full_logit_fp32_sha256_32_rows": "bitwise identical to control",
    "token_stream": "identical to control",
    "per_row_argmax": "equals the sampled token",
    "promotion": "+50 us/token vs both bracketing controls (control / candidate / control)",
    "census": "changes only the norms population node census",
  },
  "question": "Can the ffn/next RMSNorm epilogue be absorbed as an ordinary in-core epilogue of the consuming quant kernel under the boundary-free gate, and does the exact-output A/B book the norms attribution row?",
}


# Predicted-wall-delta contract (nv_fusion_cost_model.py): the prediction is
# derived from the llama reference shape (the norm is ONE fused rms_norm_f32
# kernel and its arithmetic never enters the matmul) plus the per-element
# arithmetic of this candidate (the ffn/next norm epilogue re-executes once per
# matrix dot, R=2; x streams fp32 instead of fp16; the r_16_256 scale reduce
# must stay).  The bracket reconciles the measured delta against this range:
# CONFIRMED / EXPLAINED pass with evidence; CONTRADICTED fails closed.
COST_PREDICTION = {
  "contract": "before implementing, derive the predicted wall delta from the llama reference shape plus per-element instruction/traffic arithmetic; the wall bracket then either confirms it or explains the gap",
  "llama_reference": "rms_norm_f32 = ONE fused reduce+affine kernel, fp32 out; norm never enters the matmul; matmul consumes a compact quantized activation (llama_tinygrad_role_manifest.py)",
  "arithmetic": {
    "redundancy": 2,
    "redundancy_note": "the ffn/next norm epilogue re-executes once per matrix dot (gate and up each recompute the fused affine)",
    "per_element_extra_ops": "2 FMUL + fp16 RNE cast + upcast per element per matrix",
    "x_traffic": "fp32 vs fp16 activation bytes double across the absorbed rows",
    "scale_reduce_retained": "the r_16_256 scale reduce must stay (bitwise contract; llama keeps n_f32)",
  },
  "formula": "blocks x [ (R - 1) x M_removed - R x launch_us ]; positive = candidate slower",
  "tolerance_us": 20.0,
  "assumptions": {
    "launch_us": "1.5 (range 1.0-2.0), E_32_32_4 class floor, m4-resadd-landing-scope-20260806.md",
    "M_removed": "control census median of the folded norms-epilogue family, measured per run before the bracket",
    "blocks": 36,
  },
  "unmodeled": ["in-kernel critical path (occupancy/dependency chain)", "activation traffic"],
}


class ConstructionGapError(RuntimeError):
  """The norms in-core epilogue construction is not expressible boundary-free."""


def _digest(a: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def candidate_topology_probe(dim: int = DIM, eps: float = EPS) -> dict:
  """CPU evidence for the fused epilogue construction (numerics + lowering).

  The reference is the ordinary two-program RMSNorm pair.  The candidate is
  the per-load affine ``y = (x.float()*s).cast(fp16)*w`` with ``s`` the
  bitwise-shared fp32 reduce scalar; the reduce node itself stays a separate
  program.  Everything here runs on the CPU scheduler, so it is hermetic.
  """
  from tinygrad import Tensor, dtypes, nn
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import Ops
  with Context(DEV="CPU"):
    x = Tensor.randn(1, dim, dtype=dtypes.float16).realize()
    w = Tensor.randn(dim, dtype=dtypes.float16).realize()
    norm = nn.RMSNorm(dim, eps=eps)
    norm.weight = w
    norm._rmsnorm_native_promoted = False
    reference = norm(x)
    scale = (x.float().square().mean(-1, keepdim=True) + eps).rsqrt()
    fused = (x.float() * scale).cast(dtypes.float16) * w
    ref_np, fused_np = reference.numpy(), fused.numpy()
    bitwise_equal = bool(np.array_equal(ref_np.view(np.uint16), fused_np.view(np.uint16)))
    max_abs = float(np.max(np.abs(ref_np.astype(np.float32) - fused_np.astype(np.float32))))
    contains_custom = any(u.op is Ops.CUSTOM for u in fused.uop.toposort())
    contains_contiguous = any(u.op is Ops.CONTIGUOUS for u in fused.uop.toposort())
    fused_pair_linear, _ = fused.linear_with_vars()
    fused_pair_program_count = len(fused_pair_linear.src)
    # Epilogue-only lowering with the reduce scalar pre-realized: exactly one
    # ordinary elementwise program, i.e. the epilogue half is a 1-program
    # affine when the reduce result is shared as a scalar input.
    scale_r = scale.realize()
    affine = (x.float() * scale_r).cast(dtypes.float16) * w
    affine_linear, _ = affine.linear_with_vars()
    affine_program_count = len(affine_linear.src)
  return {
    "fused_epilogue_bitwise_equal": bitwise_equal,
    "fused_epilogue_max_abs": max_abs,
    "fused_epilogue_contains_custom": contains_custom,
    "fused_epilogue_contains_contiguous": contains_contiguous,
    "fused_pair_program_count": fused_pair_program_count,
    "affine_epilogue_program_count": affine_program_count,
    "reduce_scalar_dtype": "float32",
    "consumer_is_ordinary": False,
    "consumer_is_ordinary_reason": ("the decode RMSNorm consumer is the opaque q4k GEMV custom "
      "program (q4k_g3_lanemap_gemv_*); ordinary UOps cannot fuse an epilogue into a custom-program "
      "body without a CUSTOM boundary"),
    "reduce_absorbable": False,
    "reduce_absorbable_reason": ("no generic cross-thread reduction-to-output scheduler primitive "
      "exists; the phase-0 gate records the ordinary pair as one reduction program plus one "
      "dependent epilogue program"),
    "candidate_removes_programs_in_graph": False,
    "candidate_removes_programs_in_graph_reason": ("absorbing the epilogue would require a custom "
      "boundary inside the opaque consumer; no replayable ordinary program is removed from the "
      "decode graph"),
  }


def boundary_free_gate() -> dict:
  """Run the phase-0 gate and the candidate probe; verdict PASS only if all
  boundary-free admission conditions hold.  Deterministically CONSTRUCTION_GAP
  today: the phase-0 baseline is a gap and the opaque consumer / reduction
  absorbability conditions are unmet."""
  from tinygrad.helpers import Context
  with Context(DEV="CPU"):
    phase0 = _phase0_gate_run(POP_NORMS)["populations"][POP_NORMS]
  probe = candidate_topology_probe()
  conditions = {
    "phase0_verdict_pass": phase0["verdict"] == "PASS",
    "consumer_is_ordinary": probe["consumer_is_ordinary"],
    "reduce_absorbable": probe["reduce_absorbable"],
    "candidate_removes_programs_in_graph": probe["candidate_removes_programs_in_graph"],
  }
  passed = all(conditions.values())
  reason = ("boundary-free ordinary-UOp construction is expressible" if passed else
    "ordinary RMSNorm remains a reduction program plus a dependent epilogue program; the q4k "
    "consumer is an opaque custom program and no generic cross-thread reduction-to-output "
    "scheduler primitive exists, so the norms epilogue cannot be absorbed in-core without a "
    "custom boundary")
  return {
    "schema": GATE_SCHEMA,
    "verdict": "PASS" if passed else "CONSTRUCTION_GAP",
    "conditions": conditions,
    "reason": reason,
    "contract": {
      "candidate_must_be_ordinary_uop": True,
      "candidate_must_not_materialize_lazy_input": True,
      "candidate_must_be_replay_profile_visible": True,
      "no_custom_boundary": True,
      "no_adapters": True,
    },
    "phase0_baseline": phase0,
    "candidate_probe": probe,
  }


def _configure(arm: str) -> None:
  if arm == "candidate":
    raise ConstructionGapError(
      "the norms in-core epilogue construction cannot be built under the boundary-free "
      "ordinary-UOp contract; see nv_boundary_free_ordinary_uop_gate CONSTRUCTION_GAP")
  if arm != "control":
    raise ValueError(f"unknown arm {arm!r}")


def _gates(model) -> dict:
  """Report the loaded model's norm-route promotion state (all closed for control)."""
  return {
    "norm_fusion_promoted": bool(getattr(model, "_decode_norm_fusion_promoted", False)),
    "reduce_output_rmsnorm_promoted": bool(getattr(model, "_decode_reduce_output_rmsnorm_promoted", False)),
    "q4k_epilogue_fusion_promoted": bool(getattr(model, "_decode_q4k_epilogue_fusion_promoted", False)),
    "block_norm_fusion_promoted": bool(getattr(model.blk[0], "_decode_norm_fusion_promoted", False))
      if getattr(model, "blk", None) else None,
  }


def _model(arm: str, model_path: str, max_context: int):
  _configure(arm)
  model = _load(model_path, max_context)
  gates = _gates(model)
  if arm == "control" and any(gates.get(key) for key in ("norm_fusion_promoted",
      "reduce_output_rmsnorm_promoted", "q4k_epilogue_fusion_promoted", "block_norm_fusion_promoted")):
    raise RuntimeError(f"control arm requires the closed model graph, observed promoted routes: {gates}")
  return model, gates


def logits(arm: str, model_path: str, depth: int, count: int, max_context: int) -> tuple[dict, np.ndarray]:
  from tinygrad import Tensor, UOp
  from tinygrad.helpers import Context
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
    if not np.isfinite(row).all() or sid != int(row.argmax(axis=-1).item()):
      raise RuntimeError(f"invalid diagnostic output at row {idx}")
    samples.append(sid); rows.append(row)
  stacked = np.stack(rows)
  return {"schema": LOGITS_SCHEMA, "arm": arm, "mode": "logits", "gates": gates,
          "prelude_token": prelude, "tokens": samples, "shape": list(stacked.shape),
          "dtype": str(stacked.dtype), "logits_sha256": _digest(stacked)}, stacked


TM_RE = re.compile(r"^\*\*\* NV\s+\d+\s+(\S+)\s+arg\s+\d+.*?tm\s+([\d.]+)(us|ms)/")


def census(arm: str, model_path: str, depth: int, max_context: int) -> dict:
  from tinygrad.helpers import Context
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
  for name, _ in rows:
    pop, role, _ = _ledger_classify(name)
    population_counts[pop] = population_counts.get(pop, 0) + 1
    if pop == POP_NORMS: norms_roles[role] = norms_roles.get(role, 0) + 1
  return {"schema": CENSUS_SCHEMA, "arm": arm, "mode": "census", "gates": gates, "token": token,
          "kernels": len(rows), "kernel_us": round(sum(us for _, us in rows), 3),
          "norms_kernels": sum(1 for name, _ in rows if _ledger_classify(name)[0] == POP_NORMS),
          "norms_roles": norms_roles, "population_counts": population_counts,
          "histogram": sorted(((name, len(vals), statistics.median(vals)) for name, vals in hist.items()),
                              key=lambda row: (-row[1], -row[2]))}


def timing_child(arm: str, model_path: str, depth: int, count: int, max_context: int,
                 reps: int, settled_continuous: bool) -> dict:
  from tinygrad import Device
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


def validate_logits_gate(control: dict, candidate: dict) -> dict:
  """Exact-output gate: full fp32 logits SHA-256, token stream, shape identical."""
  tokens_equal = control.get("tokens") == candidate.get("tokens")
  sha_equal = control.get("logits_sha256") == candidate.get("logits_sha256")
  shape_equal = control.get("shape") == candidate.get("shape")
  return {"tokens_equal": bool(tokens_equal), "logits_sha256_equal": bool(sha_equal),
          "shape_equal": bool(shape_equal), "control_sha256": control.get("logits_sha256"),
          "candidate_sha256": candidate.get("logits_sha256"),
          "gate_pass": bool(tokens_equal and sha_equal and shape_equal)}


def validate_census(control: dict, candidate: dict) -> dict:
  """Census confinement: only norms-population kernel counts may change, the
  norm reduce roles stay identical, and norms epilogues are removed."""
  control_pops = control.get("population_counts") or {}
  candidate_pops = candidate.get("population_counts") or {}
  non_norms_changed = sorted(
    p for p in set(control_pops) | set(candidate_pops)
    if p != POP_NORMS and control_pops.get(p) != candidate_pops.get(p))
  control_roles = control.get("norms_roles") or {}
  candidate_roles = candidate.get("norms_roles") or {}
  reduce_roles = [r for r in sorted(set(control_roles) | set(candidate_roles)) if "reduce" in r]
  reduce_unchanged = all(control_roles.get(r, 0) == candidate_roles.get(r, 0) for r in reduce_roles)
  norms_before = int(control.get("norms_kernels", 0))
  norms_after = int(candidate.get("norms_kernels", 0))
  epilogues_removed = norms_before - norms_after
  confined = not non_norms_changed
  return {"confined_to_norms": confined, "non_norms_changed": non_norms_changed,
          "norms_kernels_control": norms_before, "norms_kernels_candidate": norms_after,
          "norms_epilogues_removed": epilogues_removed, "norms_reduce_unchanged": bool(reduce_unchanged),
          "gate_pass": bool(confined and reduce_unchanged and epilogues_removed > 0)}


def validate_timing_bracket(rows: list[dict], settled_continuous: bool = True) -> dict:
  """Reverse control/candidate/control bracket; promotion requires +50 us/token
  vs both bracketing controls with an identical token stream."""
  if len(rows) != 3: raise ValueError("timing bracket needs exactly control/candidate/control rows")
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
          "note": "wall evidence only; booking requires the exact-logits gate and norms-confined census"}


def validate_cost_prediction(bracket: dict, control_census: dict, candidate_census: dict) -> dict:
  """Norms predicted-wall-delta gate (nv_fusion_cost_model.py).

  The COST_PREDICTION table is derived from the llama reference shape plus the
  per-element arithmetic of the candidate; the measured bracket delta must then
  confirm it or explain the gap.  A measured delta outside the predicted range
  on the opposite side of zero is a CONTRADICTION and FAILS CLOSED."""
  hist = control_census.get("histogram") or []
  removed_medians = {name: med for name, _, med in hist
                     if _ledger_classify(name)[0] == POP_NORMS and _ledger_classify(name)[1].endswith("epilogue")}
  if not removed_medians:
    return {"run": True, "result": "NO-GO", "reason": "control census has no norms-epilogue family to model"}
  prediction = predict_wall_delta(COST_PREDICTION["assumptions"]["blocks"],
                                  {"norms_epilogue": statistics.median(removed_medians.values())},
                                  {"norms_epilogue": COST_PREDICTION["arithmetic"]["redundancy"]})
  measured = -bracket["candidate_minus_control_bracket_us"]
  reconciliation = reconcile_cost_prediction(measured, prediction,
                                             tolerance_us=COST_PREDICTION["tolerance_us"])
  return {"run": True, "result": "PASS" if reconciliation["result"] != "CONTRADICTED" else "FAIL",
          "contract": COST_PREDICTION, "prediction": prediction, "reconciliation": reconciliation,
          "measured_delta_us": measured, "bracket_field_us": bracket["candidate_minus_control_bracket_us"],
          "note": reconciliation["note"]}


def ledger_census(dag_path: str | None) -> dict:
  """Static norms census from the authority DAG, with a fail-closed reference
  fallback when the capture file is unavailable."""
  if dag_path and pathlib.Path(dag_path).is_file():
    from extra.llm_research.decode.nv_fusion_population_ledger import analyze, load
    result = analyze(load(dag_path))
    norms = result["populations"][POP_NORMS]
    return {"run": True, "authority_dag": dag_path, "capture": result["capture"],
            "norms": {key: norms[key] for key in (
              "node_count", "total_us", "mean_us", "max_us", "reduction_count", "epilogue_count",
              "fusion_candidate_count", "fusion_candidate_us", "fusion_candidate_epilogue_count",
              "fusion_candidate_epilogue_us", "boundary_free_eligible")},
            "all_populations": {p: {key: r[key] for key in (
              "node_count", "total_us", "fusion_candidate_count", "fusion_candidate_us")}
              for p, r in result["populations"].items()},
            "note": "ledger census on the redirect-on authority DAG; the candidate census arm must stay norms-confined"}
  return {"run": False, "authority_dag": dag_path, "reference": CENSUS_REFERENCE,
          "note": "authority DAG unavailable; reference census from nv-fusion-exhaustive-scope-20260805.md"}


HARD_STOP_NOTES = [
  "Boundary-free construction gate returned CONSTRUCTION_GAP (phase-0 baseline plus the candidate probe); "
  "nv_boundary_free_ordinary_uop_gate.py must pass before any norms GPU arm.",
  "The exact-output full-logit gate (fp32 SHA-256 over 32 rows, token stream, per-row argmax) was not "
  "authorized because the construction gate failed first.",
  "No norms census arm and no reverse control/candidate/control wall bracket ran; the +50 us promotion "
  "gate is not evaluated.",
  "Closed constructions are not reopened: nv_rmsnorm_native_microgate.py, nv_rmsnorm_scale_gateup_microgate.py, "
  "nv_reduce_output_rmsnorm_microgate.py, nv_rmsnorm_scale_gateup_one_layer_qualification.py; the audit closed "
  "the REDUCE_OUTPUT wrapper and the typed-CALL producer (0 reducers / 875 programs).",
]

CITATIONS = [
  "docs/task_workflow/input/nv-fusion-exhaustive-scope-20260805.md",
  "docs/task_workflow/input/nv-decode-exhaustive-forward-scope-20260805.md",
  "docs/task_workflow/input/nv-decode-final-accounting-audit-20260805.md",
  "extra/llm_research/decode/nv_fusion_population_ledger.py",
  "extra/llm_research/decode/nv_boundary_free_ordinary_uop_gate.py",
  "extra/llm_research/decode/nv_projection_epilogue_qualification.py",
  "extra/llm_research/decode/nv_shared_q8_progressive_qualification.py",
  "extra/llm_research/decode/nv_predispatch_full_logits_qualification.py",
  "extra/llm_research/decode/nv_rmsnorm_native_microgate.py",
  "extra/llm_research/decode/nv_rmsnorm_scale_gateup_microgate.py",
  "extra/llm_research/decode/nv_reduce_output_rmsnorm_microgate.py",
  "extra/llm_research/decode/nv_rmsnorm_scale_gateup_one_layer_qualification.py",
  "docs/task_workflow/input/nv-reduce-output-rmsnorm-microgate-record-20260805.md",
  "docs/task_workflow/input/nv-reduce-output-typed-call-input-reopen-record-20260805.md",
]


def no_go_record(gate: dict, census_evidence: dict | None = None,
                 model: str = DEFAULT_MODEL, depth: int = 512) -> dict:
  census_evidence = census_evidence or ledger_census(None)
  return {
    "schema": SCHEMA, "mode": "ab", "date": "2026-08-06",
    "target": {"model": model, "depth": depth, "device": "NV sm_120", "gpu": "RTX 5090"},
    "question": CONSTRUCTION["question"], "construction": CONSTRUCTION,
    "verdict": "NO-GO",
    "boundary_free_gate": gate,
    "logits_gate": {
      "run": False, "result": "NOT_AUTHORIZED",
      "reason": "HARD STOP: the boundary-free construction gate returned CONSTRUCTION_GAP; no exact-full-logit GPU arm was authorized",
      "contract": "full fp32 logits SHA-256 over 32 rows identical to control; identical token stream; per-row argmax equals the sampled token",
      "control_arm_reference_sha256": CONTROL_LOGITS_REFERENCE,
      "control_arm_reference_note": "P2b redirect-on full-logit authority, 8 rows (nv_p2b_redirect_on_logits.json)",
    },
    "census": census_evidence,
    "wall_bracket": {
      "run": False, "result": "NOT_AUTHORIZED",
      "reason": "HARD STOP: the reverse control/candidate/control wall bracket requires the boundary-free gate, the exact-logits gate and norms-confined census to pass first",
      "promotion_us": PROMOTION_US,
    },
    "hard_stop_notes": HARD_STOP_NOTES,
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


def _run_child(cmd: list[str], out: pathlib.Path) -> dict:
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if run.returncode:
    raise RuntimeError(f"child failed rc={run.returncode}: {run.stderr[-4000:]}")
  return json.loads(pathlib.Path(out).read_text())


def _run_full_ab(args) -> dict:
  """Authorized full path: exact logits gate, then census, then wall bracket."""
  root = pathlib.Path(args.out).with_suffix("") + ".children"
  root.mkdir(parents=True, exist_ok=True)
  control_logits = _run_child(_child_command(args, "logits", "control", root / "control-logits.json", False),
                              root / "control-logits.json")
  candidate_logits = _run_child(_child_command(args, "logits", "candidate", root / "candidate-logits.json", False),
                                root / "candidate-logits.json")
  logits_gate = validate_logits_gate(control_logits, candidate_logits)
  if not logits_gate["gate_pass"]:
    return no_go_record(boundary_free_gate(), ledger_census(args.census_dag), args.model, args.depth) | {
      "logits_gate": logits_gate, "census": {"run": False, "result": "NOT_AUTHORIZED",
      "reason": "exact-logits gate failed; HARD STOP"}, "wall_bracket": {"run": False, "result": "NOT_AUTHORIZED",
      "reason": "exact-logits gate failed; HARD STOP"}}
  control_census = _run_child(_child_command(args, "census", "control", root / "control-census.json", False),
                              root / "control-census.json")
  candidate_census = _run_child(_child_command(args, "census", "candidate", root / "candidate-census.json", False),
                                root / "candidate-census.json")
  census_gate = validate_census(control_census, candidate_census)
  if not census_gate["gate_pass"]:
    return no_go_record(boundary_free_gate(), ledger_census(args.census_dag), args.model, args.depth) | {
      "logits_gate": logits_gate, "census": census_gate, "wall_bracket": {"run": False,
      "result": "NOT_AUTHORIZED", "reason": "census confinement gate failed; HARD STOP"}}
  wall = wall_bracket(args)
  booked = logits_gate["gate_pass"] and census_gate["gate_pass"] and wall["promoted"]
  record = no_go_record(boundary_free_gate(), ledger_census(args.census_dag), args.model, args.depth)
  cost_gate = validate_cost_prediction(wall, control_census, candidate_census)
  record["cost_prediction"] = cost_gate
  if cost_gate["result"] == "FAIL": booked = False
  if booked:
    record["verdict"] = "BOOKED"
    record["census"] = census_gate
    record["wall_bracket"] = wall
    record["logits_gate"] = logits_gate
  else:
    record["census"] = census_gate
    record["wall_bracket"] = wall
    record["logits_gate"] = logits_gate
  return record


def wall_bracket(args) -> dict:
  """Serialized reverse control/candidate/control timing, each child a fresh
  process under ``timeout ... flock -w`` on the shared GPU bench lock."""
  root = pathlib.Path(args.out).with_suffix("") + ".timing"
  root.mkdir(parents=True, exist_ok=True)
  rows = []
  for sequence, arm in enumerate(("control", "candidate", "control")):
    out = root / f"{arm}-{sequence}.json"
    rows.append(_run_child(_child_command(args, "timing-child", arm, out), out))
  return validate_timing_bracket(rows, args.settled_continuous)


def ab(args) -> dict:
  """Campaign orchestrator: gate first; on CONSTRUCTION_GAP write the NO-GO
  record and stop before any GPU arm."""
  gate = boundary_free_gate()
  census_evidence = ledger_census(args.census_dag)
  record = no_go_record(gate, census_evidence, args.model, args.depth) if gate["verdict"] != "PASS" \
    else _run_full_ab(args)
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
  print(json.dumps(record, sort_keys=True))
  return record


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--mode", choices=("gate", "logits", "census", "timing-child", "ab"), required=True)
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
  ap.add_argument("--census-dag", default="/tmp/nv_p4_redirect_on_dag_20260805.json")
  args = ap.parse_args()
  _validate_run_extent(args.depth, args.count, args.max_context, args.reps, args.settled_continuous)
  if args.mode in ("logits", "census", "timing-child") and args.arm is None:
    ap.error(f"--arm is required for --mode {args.mode}")
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  if args.mode == "gate":
    result = boundary_free_gate()
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1
  if args.mode == "logits":
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
