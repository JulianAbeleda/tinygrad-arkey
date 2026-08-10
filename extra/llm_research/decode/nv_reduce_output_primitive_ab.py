#!/usr/bin/env python3
"""NV wall-bracket A/B for the generic REDUCE_OUTPUT primitive (norms row).

Campaign arm for the 08-07 capability audit's missing construct C1: the
generic cooperative reduction-to-output primitive.  The candidate arm
reproduces the production census conditions exactly
(``_decode_reduce_output_rmsnorm_promoted = True`` on the model and every
block, plus ``CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT = 1`` and
``CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER = 1``); the control arm is the closed
production graph (no promotion flags, no callify Context flags) and fails
closed if any promoted route is observed.

There is no construction-gate phase in this campaign: the CPU capability gate
already proved construction (docs/task_workflow/input/nv-generic-reduce-output-primitive-record-20260809.md).
The order is fixed: NV render smoke (Xid 31 class) first, then the exact
full-logit gate, then the norms-confined census, then the serialized reverse
control/candidate/control wall bracket under the shared GPU bench lock.  Every
GPU arm runs as a fresh process under ``timeout ... flock -w`` so JIT capture
and allocator state cannot leak across arms.  Any gate failure writes a NO-GO
record with the exact evidence; the norms row books only when every gate passes
and the bracket promotes at +50 us/token against BOTH bracketing controls.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, pathlib, re, statistics, subprocess, sys, time
import numpy as np

from extra.llm_research.decode.nv_fusion_population_ledger import POP_NORMS, classify as _ledger_classify
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _load, _prompt
from extra.llm_research.decode.nv_shared_q8_progressive_qualification import (
  _settled_continuous_windows, _timing_hash_authority, _validate_run_extent,
)

SCHEMA = "tinygrad.nv_reduce_output_primitive_ab.v1"
SMOKE_SCHEMA = "tinygrad.nv_reduce_output_primitive_ab.smoke.v1"
LOGITS_SCHEMA = "tinygrad.nv_reduce_output_primitive_ab.logits.v1"
CENSUS_SCHEMA = "tinygrad.nv_reduce_output_primitive_ab.census.v1"
TIMING_SCHEMA = "tinygrad.nv_reduce_output_primitive_ab.timing.v1"

PROMOTION_US = 50.0
# Committed census artifact expectations
# (docs/task_workflow/output/nv-generic-reduce-output-census-20260809.json).
CENSUS_REFERENCE = {
  "selector_admissions": 108,
  "fused_bodies": 54,
  "ordinary_reductions_in_baseline": 145,
  "net_call_delta_vs_ordinary": 1,
  "net_call_delta_vs_typed": 72,
  "ordinary_baseline_total_calls": 1008,
  "candidate_total_calls": 1009,
  "program_delta_vs_ordinary_baseline": {
    "E_32_32_4_02a9738c0547f555": -36,
    "E_32_32_4_81c96a8e654e707f": 36,
    "E_32_32_4_8eeb0be1271d29e7": 54,
    "E_32_32_4_f14a5cc0d0ed4c90": -18,
    "E_32_32_4_fab82d40f922cf5f": -71,
    "r_16_256_ed256c4ae79e0e20": -18,
    "reduce_output_rmsnorm_1_4096": 54,
  },
  "artifact": "docs/task_workflow/output/nv-generic-reduce-output-census-20260809.json",
}

CONSTRUCTION = {
  "route": "decode_reduce_output_rmsnorm",
  "population": "norms",
  "mechanism": "generic cooperative reduction-to-output primitive (REDUCE_OUTPUT): one ordinary CALL body reduce_output_rmsnorm_1_4096 replaces the r_16_256 reduce plus its E_32_32_4 rmsnorm epilogue on the 16x32x8 (dim 4096) association",
  "codegen_path": "candidate sets _decode_reduce_output_rmsnorm_promoted=True on the model and every block and decodes under CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1 plus CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1; the C6 CALL-input spelling is admitted through the M4-style typed-view proof",
  "census_target": "108 selector admissions / 54 fused reduce_output_rmsnorm_1_4096 bodies in the captured decode graph (both baselines: 0); net +1 vs the ordinary baseline (1008 -> 1009 calls), +72 vs typed",
  "correctness_contract": {
    "full_logit_fp32_sha256": "bitwise identical to control over the stacked rows",
    "token_stream": "identical to control",
    "per_row_argmax": "equals the sampled token",
    "promotion": "+50 us/token vs both bracketing controls (control / candidate / control)",
    "census": "fused bodies replace norms reduce/epilogue programs; q/k norm reduces untouched; callify-redirect shifts on non-norms families reported with exact program names",
  },
  "question": "Does the generic cooperative reduction-to-output primitive survive NV render (Xid 31 class), preserve exact full logits, show the expected fused-body census with an honest net program delta, and book the +495.330 us norms row under the reverse wall bracket?",
}


def _digest(a: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def tok_per_s(median_ms: float) -> float:
  """tok/s conversion helper: 1000 / median ms per token."""
  return 1000.0 / median_ms


@contextlib.contextmanager
def _arm_context(arm: str):
  """Candidate arms decode with both callify Context flags live; control arms
  decode on the closed production graph with no flags."""
  if arm == "control":
    yield
    return
  if arm != "candidate":
    raise ValueError(f"unknown arm {arm!r}")
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    yield


def _require_candidate_callify_flags() -> None:
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  missing = [name for name, var in (
    ("CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT", CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT),
    ("CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER", CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER)) if not var.value]
  if missing:
    raise RuntimeError(f"candidate arm requires callify Context flags: {missing}")


def _configure(model, arm: str) -> None:
  """Set the candidate promotion conditions on a fresh model; the control arm
  stays the closed production graph."""
  if arm == "candidate":
    _require_candidate_callify_flags()
    model._decode_reduce_output_rmsnorm_promoted = True
    for block in model.blk:
      block._decode_reduce_output_rmsnorm_promoted = True
  elif arm != "control":
    raise ValueError(f"unknown arm {arm!r}")


def _gates(model) -> dict:
  """Report the loaded model's reduce-output promotion state."""
  return {
    "reduce_output_rmsnorm_promoted": bool(getattr(model, "_decode_reduce_output_rmsnorm_promoted", False)),
    "block_reduce_output_rmsnorm_promoted": [
      bool(getattr(block, "_decode_reduce_output_rmsnorm_promoted", False)) for block in model.blk
    ] if getattr(model, "blk", None) else None,
  }


def _assert_control_closed(gates: dict) -> None:
  """Fail closed if any promoted reduce-output route is observed."""
  promoted = []
  if gates.get("reduce_output_rmsnorm_promoted"):
    promoted.append("model._decode_reduce_output_rmsnorm_promoted")
  for index, value in enumerate(gates.get("block_reduce_output_rmsnorm_promoted") or []):
    if value: promoted.append(f"block[{index}]._decode_reduce_output_rmsnorm_promoted")
  if promoted:
    raise RuntimeError(f"control arm requires the closed model graph, observed promoted routes: {promoted}")


def _assert_candidate_configured(gates: dict) -> None:
  """Fail closed if the promotion flag is missing on the model or any block."""
  missing = []
  if not gates.get("reduce_output_rmsnorm_promoted"):
    missing.append("model._decode_reduce_output_rmsnorm_promoted")
  for index, value in enumerate(gates.get("block_reduce_output_rmsnorm_promoted") or []):
    if not value: missing.append(f"block[{index}]._decode_reduce_output_rmsnorm_promoted")
  if missing:
    raise RuntimeError(f"candidate arm requires _decode_reduce_output_rmsnorm_promoted on the model and every block: {missing}")


def _model(arm: str, model_path: str, max_context: int):
  if arm == "candidate": _require_candidate_callify_flags()
  model = _load(model_path, max_context)
  _configure(model, arm)
  gates = _gates(model)
  if arm == "control": _assert_control_closed(gates)
  else: _assert_candidate_configured(gates)
  return model, gates


def smoke(arm: str, model_path: str, depth: int, max_context: int) -> dict:
  """Phase 0 NV render smoke: compile and run one decode token under the
  candidate conditions.  Success is survival (no Xid 31 MMU fault) with the
  fused ``reduce_output_rmsnorm_1_4096`` body in the compiled program set."""
  if arm != "candidate":
    raise ValueError("smoke requires the candidate arm; the fused body does not exist under the closed control graph")
  from tinygrad import Device
  from tinygrad.engine.jit import GraphAdmissionCensus, observe_graph_admissions
  with _arm_context(arm):
    model, gates = _model(arm, model_path, max_context)
    gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
    admission = GraphAdmissionCensus()
    try:
      with observe_graph_admissions(admission):
        token = int(next(gen))
      Device[Device.DEFAULT].synchronize()
      programs = [record.program_name for record in admission.records if record.program_name]
      return {"schema": SMOKE_SCHEMA, "arm": arm, "mode": "smoke", "gates": gates,
              "device": str(Device[Device.DEFAULT]), "survive": True, "token": token,
              "fused_body_present": bool(any("reduce_output_rmsnorm" in name for name in programs)),
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
      if not np.isfinite(row).all() or sid != int(row.argmax(axis=-1).item()):
        raise RuntimeError(f"invalid diagnostic output at row {idx}")
      samples.append(sid); rows.append(row)
    stacked = np.stack(rows)
    return {"schema": LOGITS_SCHEMA, "arm": arm, "mode": "logits", "gates": gates,
            "prelude_token": prelude, "tokens": samples, "shape": list(stacked.shape),
            "dtype": str(stacked.dtype), "logits_sha256": _digest(stacked)}, stacked


TM_RE = re.compile(r"^\*\*\* NV\s+\d+\s+(\S+)\s+arg\s+\d+.*?tm\s+([\d.]+)(us|ms)/")


def _fused_body_count(record: dict) -> int:
  """Count fused reduce-output bodies by program-name prefix.  The ledger has
  no rule for ``reduce_output_rmsnorm_1_4096`` yet (it classifies as
  other/unclassified), so ledger classification must NOT be used for it."""
  counts = record.get("program_counts")
  if counts is not None:
    return int(sum(count for name, count in counts.items() if "reduce_output_rmsnorm" in name))
  return int(record.get("fused_bodies", 0))


def census(arm: str, model_path: str, depth: int, max_context: int) -> dict:
  """DEBUG kernel census of one decode token, classified through the population
  ledger; fused reduce-output bodies counted by name prefix."""
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
    return {"schema": CENSUS_SCHEMA, "arm": arm, "mode": "census", "gates": gates, "token": token,
            "kernels": len(rows), "kernel_us": round(sum(us for _, us in rows), 3),
            "norms_kernels": sum(1 for name, _ in rows if _ledger_classify(name)[0] == POP_NORMS),
            "norms_roles": norms_roles, "population_counts": population_counts,
            "program_counts": program_counts, "fused_bodies": _fused_body_count({"program_counts": program_counts}),
            "fused_body_names": sorted({name for name in program_counts if "reduce_output_rmsnorm" in name}),
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
  """Reduce-output census gate.  The callify Context flags legitimately shift
  non-norms program families (E_32_32_4 residual/contiguous families moved
  -36/+36/+54/-71 in the committed census artifact), so this gate does NOT
  require blanket non-norms population equality.  It requires fused bodies to
  appear, the rmsnorm_reduce (r_16_256) count to drop consistently with the
  fused-body count, norms epilogues to be removed, and the untouched q/k
  reduce roles to stay identical.  The honest net program delta and every
  non-norms shift are recorded with the exact program families."""
  control_roles = control.get("norms_roles") or {}
  candidate_roles = candidate.get("norms_roles") or {}
  control_fused = _fused_body_count(control)
  candidate_fused = _fused_body_count(candidate)
  rmsnorm_reduce_control = int(control_roles.get("rmsnorm_reduce", 0))
  rmsnorm_reduce_candidate = int(candidate_roles.get("rmsnorm_reduce", 0))
  rmsnorm_reduce_drop = rmsnorm_reduce_control - rmsnorm_reduce_candidate
  other_reduce_roles = sorted({role for role in set(control_roles) | set(candidate_roles)
                               if "reduce" in role and role != "rmsnorm_reduce"})
  other_reduce_unchanged = all(int(control_roles.get(role, 0)) == int(candidate_roles.get(role, 0))
                               for role in other_reduce_roles)
  epilogue_roles = sorted({role for role in set(control_roles) | set(candidate_roles) if "epilogue" in role})
  epilogues_removed = sum(int(control_roles.get(role, 0)) - int(candidate_roles.get(role, 0)) for role in epilogue_roles)
  honest_net_program_delta = int(candidate.get("kernels", 0)) - int(control.get("kernels", 0))
  control_programs = control.get("program_counts") or {}
  candidate_programs = candidate.get("program_counts") or {}
  side_effects = {name: int(candidate_programs.get(name, 0)) - int(control_programs.get(name, 0))
                  for name in sorted(set(control_programs) | set(candidate_programs))
                  if int(candidate_programs.get(name, 0)) != int(control_programs.get(name, 0))}
  control_pops = control.get("population_counts") or {}
  candidate_pops = candidate.get("population_counts") or {}
  population_deltas = {pop: int(candidate_pops.get(pop, 0)) - int(control_pops.get(pop, 0))
                       for pop in sorted(set(control_pops) | set(candidate_pops)) if pop != POP_NORMS
                       if int(candidate_pops.get(pop, 0)) != int(control_pops.get(pop, 0))}
  conditions = {
    "fused_bodies_present": candidate_fused > 0,
    "rmsnorm_reduce_drop_consistent": rmsnorm_reduce_drop > 0 and candidate_fused >= rmsnorm_reduce_drop,
    "other_reduce_roles_unchanged": bool(other_reduce_unchanged),
    "epilogues_removed_positive": epilogues_removed > 0,
  }
  return {"fused_bodies_control": control_fused, "fused_bodies_candidate": candidate_fused,
          "rmsnorm_reduce_control": rmsnorm_reduce_control, "rmsnorm_reduce_candidate": rmsnorm_reduce_candidate,
          "rmsnorm_reduce_drop": rmsnorm_reduce_drop, "epilogues_removed": epilogues_removed,
          "honest_net_program_delta": honest_net_program_delta, "conditions": conditions,
          "callify_redirect_side_effects": side_effects, "non_norms_population_deltas": population_deltas,
          "note": "fused bodies replace norms reduce/epilogue programs; callify-redirect shifts on non-norms families are reported, not hidden",
          "gate_pass": bool(all(conditions.values()))}


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
          "note": "wall evidence only; booking requires the exact-logits gate and the reduce-output census gate"}


HARD_STOP_NOTES = [
  "Every GPU arm runs as a fresh process under `timeout ... flock -w 90 /tmp/gpu-bench.lock`; no arm holds the lock across a wall-bracket step.",
  "Phase 0 (NV render smoke) must survive on sm_120 (no Xid 31 MMU fault) with the fused reduce_output_rmsnorm_1_4096 body in the compiled program set.",
  "The exact full-logit gate (fp32 SHA-256 over the stacked rows, token stream, shape) must pass before any census or bracket arm runs.",
  "The census gate requires fused bodies > 0, a consistent rmsnorm_reduce drop, removed norms epilogues, and untouched q/k reduce roles; callify-redirect shifts on non-norms families are reported, not hidden.",
  "The wall bracket requires identical token-stream hashes and a candidate median at least +50 us/token faster than BOTH bracketing controls.",
  "No policy promotion: decode-reduce-output-rmsnorm-route-policy.json stays promoted_targets: []; no model wiring change; no default flip.",
]

CITATIONS = [
  "docs/task_workflow/input/nv-reduce-output-wall-bracket-scope-20260809.md",
  "docs/task_workflow/input/nv-generic-reduce-output-primitive-record-20260809.md",
  "docs/task_workflow/output/nv-generic-reduce-output-census-20260809.json",
  "extra/llm_research/decode/nv_fusion_population_ledger.py",
  "extra/llm_research/decode/nv_shared_q8_progressive_qualification.py",
  "extra/llm_research/decode/nv_predispatch_full_logits_qualification.py",
  "extra/llm_research/decode/nv_norms_fusion_ab.py",
  "scratchpad/nv_reduce_output_rmsnorm_census.py",
]


def no_go_record(model: str = DEFAULT_MODEL, depth: int = 512) -> dict:
  """Base NO-GO record; the orchestrator overwrites each phase with the exact
  evidence as the campaign advances."""
  return {
    "schema": SCHEMA, "mode": "ab", "date": "2026-08-09",
    "target": {"model": model, "depth": depth, "device": "NV sm_120", "gpu": "RTX 5090"},
    "question": CONSTRUCTION["question"], "construction": CONSTRUCTION,
    "census_reference": CENSUS_REFERENCE,
    "verdict": "NO-GO",
    "smoke": {"run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the NV render smoke arm"},
    "logits_gate": {
      "run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the exact full-logit arm",
      "contract": "full fp32 logits SHA-256 over the stacked rows identical to control; identical token stream; identical shape; per-row argmax equals the sampled token",
    },
    "census": {
      "run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the norms-confined census arm",
      "contract": "fused reduce_output_rmsnorm_1_4096 bodies > 0; rmsnorm_reduce drop consistent with body count; norms epilogues removed; q/k reduce roles untouched; honest net program delta and callify-redirect side effects reported",
    },
    "wall_bracket": {
      "run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the reverse control/candidate/control wall bracket",
      "promotion_us": PROMOTION_US,
      "contract": "all three token-stream hashes identical; candidate median at least +50 us/token faster than both bracketing controls",
    },
    "hard_stop_notes": list(HARD_STOP_NOTES),
    "citations": CITATIONS,
  }


class ChildFailure(RuntimeError):
  def __init__(self, message: str, stderr: str):
    super().__init__(message)
    self.stderr = stderr


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
    raise ChildFailure(f"child failed rc={run.returncode}: {run.stderr[-4000:]}", run.stderr[-4000:])
  return json.loads(pathlib.Path(out).read_text())


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


def _write_record(record: dict, out: pathlib.Path) -> dict:
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
  print(json.dumps(record, sort_keys=True))
  return record


def ab(args) -> dict:
  """Campaign orchestrator: smoke -> exact logits -> census -> wall bracket.
  HARD STOP with a NO-GO record at the first failed gate; BOOKED only when
  every gate passes and the bracket promotes."""
  root = pathlib.Path(args.out).with_suffix("") + ".children"
  root.mkdir(parents=True, exist_ok=True)
  record = no_go_record(args.model, args.depth)
  smoke_out = root / "smoke-candidate.json"
  try:
    smoke_result = _run_child(_child_command(args, "smoke", "candidate", smoke_out, include_reps=False), smoke_out)
  except RuntimeError as exc:
    record["smoke"] = {"run": True, "result": "NO-GO",
                       "reason": "NV render smoke child failed (Xid 31 class); raw child stderr captured below",
                       "stderr": getattr(exc, "stderr", None) or str(exc)[-4000:]}
    record["hard_stop_notes"] = HARD_STOP_NOTES + ["HARD STOP at Phase 0: the NV render smoke child failed; no logits/census/bracket arm ran."]
    return _write_record(record, pathlib.Path(args.out))
  smoke_gate = smoke_result.get("survive") is True and bool(smoke_result.get("fused_body_present"))
  record["smoke"] = {"run": True, "result": "PASS" if smoke_gate else "NO-GO", "evidence": smoke_result}
  if not smoke_gate:
    record["hard_stop_notes"] = HARD_STOP_NOTES + ["HARD STOP at Phase 0: smoke did not survive or the fused body was absent from the compiled program set."]
    return _write_record(record, pathlib.Path(args.out))
  control_logits = _run_child(_child_command(args, "logits", "control", root / "control-logits.json", include_reps=False),
                              root / "control-logits.json")
  candidate_logits = _run_child(_child_command(args, "logits", "candidate", root / "candidate-logits.json", include_reps=False),
                                root / "candidate-logits.json")
  logits_gate = validate_logits_gate(control_logits, candidate_logits)
  record["logits_gate"] = {"run": True, "result": "PASS" if logits_gate["gate_pass"] else "FAIL",
                           "control_evidence": control_logits, "candidate_evidence": candidate_logits, **logits_gate}
  if not logits_gate["gate_pass"]:
    record["hard_stop_notes"] = HARD_STOP_NOTES + ["HARD STOP at Phase 1: the exact full-logit gate failed; no census and no bracket arm ran."]
    return _write_record(record, pathlib.Path(args.out))
  control_census = _run_child(_child_command(args, "census", "control", root / "control-census.json", include_reps=False),
                              root / "control-census.json")
  candidate_census = _run_child(_child_command(args, "census", "candidate", root / "candidate-census.json", include_reps=False),
                                root / "candidate-census.json")
  census_gate = validate_census(control_census, candidate_census)
  record["census"] = {"run": True, "result": "PASS" if census_gate["gate_pass"] else "FAIL",
                      "control_evidence": control_census, "candidate_evidence": candidate_census, **census_gate}
  if not census_gate["gate_pass"]:
    record["hard_stop_notes"] = HARD_STOP_NOTES + ["HARD STOP at Phase 2: the census gate failed; no wall bracket ran."]
    return _write_record(record, pathlib.Path(args.out))
  wall = wall_bracket(args)
  record["wall_bracket"] = {"run": True, "result": "PROMOTED" if wall["promoted"] else "NOT_PROMOTED", **wall}
  record["tok_per_s"] = {
    "conversion": "tok/s = 1000 / median_ms_per_token",
    "control_bracket_median_ms": wall["control_bracket_median_ms"],
    "candidate_median_ms": wall["candidate_ms"],
    "control_tok_per_s": tok_per_s(wall["control_bracket_median_ms"]),
    "candidate_tok_per_s": tok_per_s(wall["candidate_ms"]),
    "gain_tok_per_s": tok_per_s(wall["candidate_ms"]) - tok_per_s(wall["control_bracket_median_ms"]),
  }
  booked = logits_gate["gate_pass"] and census_gate["gate_pass"] and wall["promoted"]
  record["verdict"] = "BOOKED" if booked else "NO-GO"
  if not booked:
    record["hard_stop_notes"] = HARD_STOP_NOTES + [
      "Wall bracket completed but did not promote; the measured deltas are recorded above. "
      "An unpromoted bracket due to the net call-count overhead (+1 vs ordinary, +72 vs typed) authorizes the Phase 6 route-efficiency follow-up."]
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
      ap.error("smoke requires --arm candidate: the fused body only exists under the candidate conditions")
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
