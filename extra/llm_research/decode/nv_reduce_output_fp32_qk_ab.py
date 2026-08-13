#!/usr/bin/env python3
"""NV wall-bracket A/B for the fp32 q/k reduce-output route (norms row).

Sibling of ``extra/llm_research/decode/nv_reduce_output_primitive_ab.py`` (the
C6 campaign).  The fp32 route attacks the unbooked q/k mass of the +495.330 us
norms row: per decode token the ordinary graph renders 36 q norms (rows=32 x
128) and 36 k norms (rows=8 x 128) as single-kernel reduces
(``r_2_8_4_4_16`` / ``r_8_16_8``) plus elementwise epilogues, and the C6
marker route cannot reach them (marker bails on rows != 1 and the consumer is
an ordinary elementwise through a PERMUTE view, not a C6 CALL argument).

The candidate arm reproduces the C6 campaign conditions exactly
(``_decode_reduce_output_rmsnorm_promoted = True`` on the model and every
block, plus ``CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT = 1`` and
``CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER = 1``) so the 4096-dim C6 route stays
booked at 18 bodies; the fp32 route adds one cooperative body per q norm
(``reduce_output_rmsnorm_32_128``) and one per k norm
(``reduce_output_rmsnorm_8_128``) for 90 fused bodies total.  The control arm
is otherwise the closed production graph and fails closed if any promoted
reduce-output route is observed.

Gate order is fixed: NV render smoke (Xid 31 class) with the fused q/k bodies
in the compiled set, then the exact full-logit gate, then the fp32 census, then
the serialized reverse control/candidate/control wall bracket under the shared
GPU bench lock.  Every GPU arm runs as a fresh process under
``timeout ... flock -w`` so JIT capture and allocator state cannot leak across
arms.  The census gate FAILS CLOSED if the q/k bodies do not appear (selector
still rejects the fp32 route); any gate failure writes a NO-GO record with the
exact evidence, and the q/k share of the norms row books only when every gate
passes and the bracket promotes at +50 us/token against BOTH bracketing
controls.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, pathlib, statistics, sys, time
import numpy as np

from extra.llm_research.decode.nv_fusion_cost_model import reconcile_cost_prediction
from extra.llm_research.decode.nv_fusion_population_ledger import POP_NORMS, classify as _ledger_classify
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _prompt
from extra.llm_research.decode.nv_reduce_output_primitive_ab import (
  PROMOTION_US, TM_RE, _arm_context, _assert_candidate_configured,
  _assert_control_closed, _child_root, _configure, _digest, _gates,
  _model, _require_candidate_callify_flags, _run_child, _settled_continuous_windows,
  _timing_hash_authority, _validate_run_extent, _write_record, tok_per_s,
  validate_logits_gate as _c6_validate_logits_gate,
)

SCHEMA = "tinygrad.nv_reduce_output_fp32_qk_ab.v1"
SMOKE_SCHEMA = "tinygrad.nv_reduce_output_fp32_qk_ab.smoke.v1"
LOGITS_SCHEMA = "tinygrad.nv_reduce_output_fp32_qk_ab.logits.v1"
CENSUS_SCHEMA = "tinygrad.nv_reduce_output_fp32_qk_ab.census.v1"
TIMING_SCHEMA = "tinygrad.nv_reduce_output_fp32_qk_ab.timing.v1"

# Committed census expectations for the fp32 route, updated to the
# production-observed contract (logits-verified at 420c4afc2 + fixes, ffn
# residual-bind at 9ed605f46): every q/k marker admits (36 q + 36 k) and
# lowers one fused body per marker.  The warp-coop carriers (17 q + 17 k)
# keep their exact REDUCE as a materializing kernel, now rendered as a real
# reduce under its own ``r_32_32_4_4_*``/``r_8_32_4_4_*`` name (the pre-fix
# silent ``E_*`` copies are gone), so the ledger's q/k reduce roles (which
# count only the ordinary ``r_2_8_4_4_16``/``r_8_16_8`` names) drop 36 -> 0
# and the 17+17 materialized reduces are reported as exact side effects.  The
# q/k epilogue roles drop to 0.  With the ffn-norm residual bind, the C6
# route fuses 55 bodies (36 ffn + 18 attn + final norm); only the output norm
# stays ordinary, so rmsnorm reduce 56 -> 1, epilogue 55 -> 1, and the final
# epilogue role 1 -> 0.  Each ordinary q/k norm renders TWO epilogue kernels,
# so the q/k epilogue drop is 2 x the fused body count.
# (docs/task_workflow/input/nv-reduce-output-fp32-qk-route-scope-20260810.md).
CENSUS_REFERENCE = {
  "fused_bodies_total": 127,
  "fused_bodies_c6": 55,   # reduce_output_rmsnorm_1_4096 (36 ffn + 18 attn + final norm)
  "fused_bodies_q": 36,    # reduce_output_rmsnorm_32_128 (every q marker)
  "fused_bodies_k": 36,    # reduce_output_rmsnorm_8_128 (every k marker)
  "q_norm_reduce": [36, 0],   # ledger roles; the 17 warp-coop materialized reduces render as new r_32_32_4_4_* kernels (side effect)
  "k_norm_reduce": [36, 0],   # ledger roles; the 17 warp-coop materialized reduces render as new r_8_32_4_4_* kernels (side effect)
  "q_norm_epilogue": [72, 0],   # two epilogue kernels per ordinary q norm, all fused away
  "k_norm_epilogue": [72, 0],   # two epilogue kernels per ordinary k norm, all fused away
  "rmsnorm_reduce": [56, 1],    # C6 bodies (only the output norm stays ordinary)
  "rmsnorm_epilogue": [55, 1],  # only the output norm epilogue stays ordinary
  "final_rmsnorm_epilogue": [1, 0],  # final norm fuses into a C6 body
  "net_norms_kernels_delta": -326,  # 328 -> 2 (127 fused bodies counted separately by prefix)
  "artifact": "docs/task_workflow/input/nv-reduce-output-fp32-qk-route-scope-20260810.md",
}

CONSTRUCTION = {
  "route": "decode_reduce_output_rmsnorm fp32 q/k",
  "population": "norms",
  "mechanism": "fp32 cooperative reduce-output bodies for the q/k norms: one reduce_output_rmsnorm_32_128 body per q norm and one reduce_output_rmsnorm_8_128 body per k norm replace the ordinary single-kernel q/k reduces (r_2_8_4_4_16 / r_8_16_8) plus their elementwise epilogues; the warp-coop family keeps its exact partials REDUCE as a materializing kernel (bitwise-identical); the ffn residual bind lifts the C6 4096-dim route to 55 reduce_output_rmsnorm_1_4096 bodies",
  "codegen_path": "candidate sets _decode_reduce_output_rmsnorm_promoted=True on the model and every block and decodes under CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1 plus CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1; the fp32 marker rows (8/32 x 128) admit through the PERMUTE-view carrier",
  "census_target": "127 fused bodies (55 C6 incl. final norm + 36 q + 36 k); q/k reduce ledger roles 36 -> 0 each, with the 17 q + 17 k warp-coop materialized reduces persisting as real kernels under their own r_32_32_4_4_*/r_8_32_4_4_* names (reported as exact side effects; bitwise-identical values); q/k epilogue roles 72 -> 0 (2 epilogue kernels per ordinary norm); C6 rmsnorm roles reduce 56 -> 1, epilogue 55 -> 1, final epilogue 1 -> 0; net norms kernels -326 (328 -> 2)",
  "correctness_contract": {
    "full_logit_fp32_sha256": "bitwise identical to control over the stacked rows",
    "token_stream": "identical to control",
    "per_row_argmax": "equals the sampled token",
    "promotion": "+50 us/token vs both bracketing controls (control / candidate / control)",
    "census": "fused bodies counted by the reduce_output_rmsnorm name prefix; q/k reduce and epilogue drops consistent with body counts; FAIL CLOSED if the q/k bodies do not appear; non-norms family shifts reported with exact program names",
  },
  "question": "Does the fp32 q/k reduce-output route survive NV render (Xid 31 class), preserve exact full logits, show the expected fused-body census with an honest net program delta, and book the q/k share of the +495.330 us norms row under the reverse wall bracket?",
}

# Predicted-wall-delta contract (nv_fusion_cost_model.py): the prediction is
# derived from llama's own d512 shape census (nv-llama-d512-node-ledger-20260812.json
# and the stage-3 scope table): the q/k norms render in llama as
# `rms_norm_f32<256>` grid [rows,1,1] at 1.30 us/launch (72 launches = 93.3 us)
# and the 4096-dim block norms at 2.88 us/launch (73 launches = 210.2 us).
# The reconcile term models the removed control-census kernel mass against the
# llama floor for the same rows: if the fused bodies land at llama's per-launch
# cost, the wall delta is the llama body mass minus the removed kernel mass;
# the envelope covers bodies at zero cost (save all removed mass) up to bodies
# at twice the llama floor.  A measured delta beyond the envelope on the
# opposite side of zero CONTRADICTS the llama-shaped premise and fails the
# campaign closed.
COST_PREDICTION = {
  "contract": "before implementing, derive the predicted wall delta from the llama reference shape plus per-launch arithmetic; the wall bracket then either confirms it or explains the gap",
  "llama_reference": "rms_norm_f32<256> grid [rows,1,1] at 1.30 us/launch (72 launches = 93.3 us) for q/k; rms_norm_f32<1024> grid [1,1,1] at 2.88 us/launch (73 launches = 210.2 us) for 4096 (stage-3 scope table, nv-reduce-output-stage3-geometry-scope-20260813.md)",
  "arithmetic": {
    "formula": "point = added_body_mass - removed_control_mass (positive = candidate slower); lo = llama_floor_body_mass - removed_control_mass (best case: bodies reach llama's per-launch floor); hi = point + llama_floor_body_mass",
    "added_body_mass": "candidate-census medians of the fused reduce_output_rmsnorm bodies x the added count",
    "llama_floor_body_mass": "fused body counts x llama floor (reduce_output_rmsnorm_32_128 / _8_128: 1.30 us/launch; reduce_output_rmsnorm_1_4096: 2.88 us/launch)",
    "removed_control_mass": "control-census medians of the replaced q/k reduce + epilogue families and the r_16_256 / E_32_32_4_f14a5cc0 families x the count drops",
    "envelope": "best case bodies at llama's floor (save the most mass) to pessimistic bodies at 2x their measured cost",
  },
  "tolerance_us": 20.0,
  "unmodeled": ["launch overlap (removed kernels partially hidden behind the matmul stream)", "in-kernel critical path / occupancy", "non-norms callify redirects"],
}

LLAMA_FLOOR_US = {"reduce_output_rmsnorm_32_128": 1.30, "reduce_output_rmsnorm_8_128": 1.30,
                  "reduce_output_rmsnorm_1_4096": 2.88}
REPLACED_PREFIXES = ("r_2_8_4_4_16", "r_8_16_8", "E_2_8_16_4", "E_8_2_16_4", "E_4_2_8_16_4",
                     "r_16_256", "E_32_32_4_f14a5cc0")


def _fused_body_families(record: dict) -> dict:
  """Count fused reduce-output bodies by program-name prefix, split into the
  C6 family (``reduce_output_rmsnorm_1_4096``) and the fp32 q/k families
  (``reduce_output_rmsnorm_32_128`` / ``reduce_output_rmsnorm_8_128``)."""
  counts = record.get("program_counts") or {}
  c6 = sum(count for name, count in counts.items() if name.startswith("reduce_output_rmsnorm_1_4096"))
  q = sum(count for name, count in counts.items() if name.startswith("reduce_output_rmsnorm_32_128"))
  k = sum(count for name, count in counts.items() if name.startswith("reduce_output_rmsnorm_8_128"))
  total = sum(count for name, count in counts.items() if "reduce_output_rmsnorm" in name)
  return {"c6": int(c6), "q": int(q), "k": int(k), "total": int(total)}


def smoke(arm: str, model_path: str, depth: int, max_context: int) -> dict:
  """Phase 0 NV render smoke: compile and run one decode token under the
  candidate conditions.  Success is survival (no Xid 31 MMU fault) with the
  fused q/k bodies (``reduce_output_rmsnorm_32_128`` / ``reduce_output_rmsnorm_8_128``)
  in the compiled program set."""
  if arm != "candidate":
    raise ValueError("smoke requires the candidate arm; the fused q/k bodies do not exist under the closed control graph")
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
              "fused_body_present": bool(any("reduce_output_rmsnorm" in name for name in programs)),
              "fused_c6_body_present": bool(any("reduce_output_rmsnorm_1_4096" in name for name in programs)),
              "fused_q_body_present": bool(any("reduce_output_rmsnorm_32_128" in name for name in programs)),
              "fused_k_body_present": bool(any("reduce_output_rmsnorm_8_128" in name for name in programs)),
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
  ledger; fused reduce-output bodies counted by name prefix with the fp32 q/k
  families broken out."""
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
    bodies = _fused_body_families({"program_counts": program_counts})
    return {"schema": CENSUS_SCHEMA, "arm": arm, "mode": "census", "gates": gates, "token": token,
            "kernels": len(rows), "kernel_us": round(sum(us for _, us in rows), 3),
            "norms_kernels": sum(1 for name, _ in rows if _ledger_classify(name)[0] == POP_NORMS),
            "norms_roles": norms_roles, "population_counts": population_counts,
            "program_counts": program_counts,
            "fused_bodies": bodies["total"], "fused_bodies_c6": bodies["c6"],
            "fused_bodies_q": bodies["q"], "fused_bodies_k": bodies["k"],
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
  """Exact-output gate: fp32 SHA-256 over the stacked rows identical to control,
  token streams identical, shape identical (C6 logits child pattern, including
  the eager JIT=0 finite check that runs inside the child before this gate)."""
  for label, row in (("control", control), ("candidate", candidate)):
    if row.get("schema") != LOGITS_SCHEMA:
      raise ValueError(f"{label} logits row requires schema {LOGITS_SCHEMA!r}, got {row.get('schema')!r}")
  return _c6_validate_logits_gate(control, candidate)


def validate_census(control: dict, candidate: dict) -> dict:
  """fp32 q/k census gate.  Fused bodies are counted by the
  ``reduce_output_rmsnorm`` name prefix.  Every q/k marker admits, so the
  q/k epilogue drop must equal 2 x the q/k body counts; the q/k reduce drop
  must equal the body counts (the ledger's q/k reduce roles count the
  ordinary ``r_2_8_4_4_16``/``r_8_16_8`` names, all of which fuse; the 17+17
  warp-coop materialized reduces persist as real kernels under their own
  ``r_32_32_4_4_*``/``r_8_32_4_4_*`` names and are reported as exact side
  effects).  The C6 route fuses the ffn/attn block norms plus the final
  norm (55 bodies), so the
  rmsnorm reduce drop must match the C6 body count, the rmsnorm epilogue
  drop must be C6 bodies - 1 (the final norm epilogue is the separate role
  and fuses to 0).  Observed body counts must match the committed reference
  (a drift in admission flips the gate and forces a reference update).  If
  the q/k bodies do not appear (selector still rejects the fp32 route) the
  gate FAILS CLOSED with a clear message and never silently passes.
  Non-norms family shifts are reported with the exact program names, not
  hidden."""
  for label, row in (("control", control), ("candidate", candidate)):
    if row.get("schema") != CENSUS_SCHEMA:
      raise ValueError(f"{label} census row requires schema {CENSUS_SCHEMA!r}, got {row.get('schema')!r}")
  control_roles = control.get("norms_roles") or {}
  candidate_roles = candidate.get("norms_roles") or {}
  control_bodies = _fused_body_families(control)
  candidate_bodies = _fused_body_families(candidate)
  q_reduce_drop = int(control_roles.get("q_norm_reduce", 0)) - int(candidate_roles.get("q_norm_reduce", 0))
  k_reduce_drop = int(control_roles.get("k_norm_reduce", 0)) - int(candidate_roles.get("k_norm_reduce", 0))
  q_epilogue_drop = int(control_roles.get("q_norm_epilogue", 0)) - int(candidate_roles.get("q_norm_epilogue", 0))
  k_epilogue_drop = int(control_roles.get("k_norm_epilogue", 0)) - int(candidate_roles.get("k_norm_epilogue", 0))
  rmsnorm_reduce_drop = int(control_roles.get("rmsnorm_reduce", 0)) - int(candidate_roles.get("rmsnorm_reduce", 0))
  rmsnorm_epilogue_drop = int(control_roles.get("rmsnorm_epilogue", 0)) - int(candidate_roles.get("rmsnorm_epilogue", 0))
  final_control = int(control_roles.get("final_rmsnorm_epilogue", 0))
  final_candidate = int(candidate_roles.get("final_rmsnorm_epilogue", 0))
  q_bodies, k_bodies, c6_bodies = candidate_bodies["q"], candidate_bodies["k"], candidate_bodies["c6"]
  q_remaining = int(candidate_roles.get("q_norm_reduce", 0))
  k_remaining = int(candidate_roles.get("k_norm_reduce", 0))
  conditions = {
    "control_has_no_bodies": control_bodies["total"] == 0,
    "qk_bodies_present": q_bodies > 0 and k_bodies > 0,
    "q_reduce_drop_consistent": q_reduce_drop > 0 and q_reduce_drop == q_bodies - q_remaining,
    "k_reduce_drop_consistent": k_reduce_drop > 0 and k_reduce_drop == k_bodies - k_remaining,
    "q_reduce_remaining_matches_reference": q_remaining == CENSUS_REFERENCE["q_norm_reduce"][1],
    "k_reduce_remaining_matches_reference": k_remaining == CENSUS_REFERENCE["k_norm_reduce"][1],
    "q_epilogue_drop_consistent": q_epilogue_drop > 0 and q_epilogue_drop == 2 * q_bodies,
    "k_epilogue_drop_consistent": k_epilogue_drop > 0 and k_epilogue_drop == 2 * k_bodies,
    "c6_bodies_present": c6_bodies > 0,
    "rmsnorm_reduce_drop_consistent": rmsnorm_reduce_drop > 0 and rmsnorm_reduce_drop == c6_bodies,
    "rmsnorm_epilogue_drop_consistent": rmsnorm_epilogue_drop > 0 and rmsnorm_epilogue_drop == c6_bodies - 1,
    "final_epilogue_fused_consistent": final_control == 1 and final_candidate == 0,
    "body_counts_match_reference": (q_bodies, k_bodies, c6_bodies) == (
      CENSUS_REFERENCE["fused_bodies_q"], CENSUS_REFERENCE["fused_bodies_k"], CENSUS_REFERENCE["fused_bodies_c6"]),
  }
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
  norms_role_deltas = {role: int(candidate_roles.get(role, 0)) - int(control_roles.get(role, 0))
                       for role in sorted(set(control_roles) | set(candidate_roles))
                       if int(candidate_roles.get(role, 0)) != int(control_roles.get(role, 0))}
  unexpected_roles = sorted({role for role in set(control_roles) | set(candidate_roles) if role not in (
    "q_norm_reduce", "k_norm_reduce", "q_norm_epilogue", "k_norm_epilogue",
    "rmsnorm_reduce", "rmsnorm_epilogue", "final_rmsnorm_epilogue")})
  reference = {
    "expected_total": CENSUS_REFERENCE["fused_bodies_total"], "observed_total": candidate_bodies["total"],
    "expected_c6": CENSUS_REFERENCE["fused_bodies_c6"], "observed_c6": c6_bodies,
    "expected_q": CENSUS_REFERENCE["fused_bodies_q"], "observed_q": q_bodies,
    "expected_k": CENSUS_REFERENCE["fused_bodies_k"], "observed_k": k_bodies,
    "expected_net_norms_kernels_delta": CENSUS_REFERENCE["net_norms_kernels_delta"],
  }
  fail_closed = []
  if not conditions["qk_bodies_present"]:
    fail_closed.append(
      "FAIL CLOSED: the fp32 q/k fused bodies did not appear (selector still rejects the fp32 route): "
      f"q bodies={q_bodies}, k bodies={k_bodies}, q_norm_reduce drop={q_reduce_drop}, "
      f"k_norm_reduce drop={k_reduce_drop}")
  if not conditions["control_has_no_bodies"]:
    fail_closed.append("FAIL CLOSED: the control census unexpectedly contains fused reduce-output bodies")
  return {"fused_bodies_control": control_bodies["total"], "fused_bodies_candidate": candidate_bodies["total"],
          "fused_bodies_c6_candidate": c6_bodies, "fused_bodies_q_candidate": q_bodies,
          "fused_bodies_k_candidate": k_bodies,
          "q_norm_reduce_control": int(control_roles.get("q_norm_reduce", 0)),
          "q_norm_reduce_candidate": int(candidate_roles.get("q_norm_reduce", 0)),
          "q_norm_reduce_drop": q_reduce_drop,
          "k_norm_reduce_control": int(control_roles.get("k_norm_reduce", 0)),
          "k_norm_reduce_candidate": int(candidate_roles.get("k_norm_reduce", 0)),
          "k_norm_reduce_drop": k_reduce_drop,
          "q_norm_epilogue_control": int(control_roles.get("q_norm_epilogue", 0)),
          "q_norm_epilogue_candidate": int(candidate_roles.get("q_norm_epilogue", 0)),
          "q_norm_epilogue_drop": q_epilogue_drop,
          "k_norm_epilogue_control": int(control_roles.get("k_norm_epilogue", 0)),
          "k_norm_epilogue_candidate": int(candidate_roles.get("k_norm_epilogue", 0)),
          "k_norm_epilogue_drop": k_epilogue_drop,
          "rmsnorm_reduce_control": int(control_roles.get("rmsnorm_reduce", 0)),
          "rmsnorm_reduce_candidate": int(candidate_roles.get("rmsnorm_reduce", 0)),
          "rmsnorm_reduce_drop": rmsnorm_reduce_drop,
          "rmsnorm_epilogue_control": int(control_roles.get("rmsnorm_epilogue", 0)),
          "rmsnorm_epilogue_candidate": int(candidate_roles.get("rmsnorm_epilogue", 0)),
          "rmsnorm_epilogue_drop": rmsnorm_epilogue_drop,
          "final_rmsnorm_epilogue_control": final_control, "final_rmsnorm_epilogue_candidate": final_candidate,
          "honest_net_program_delta": honest_net_program_delta,
          "norms_role_deltas": norms_role_deltas, "unexpected_norms_roles": unexpected_roles,
          "conditions": conditions, "fail_closed": fail_closed, "reference": reference,
          "callify_redirect_side_effects": side_effects, "non_norms_population_deltas": population_deltas,
          "note": "fused bodies counted by reduce_output_rmsnorm name prefix; q/k reduce drops equal body counts and epilogue drops equal 2x body counts; C6 55 bodies (36 ffn + 18 attn + final norm, final epilogue role fuses); non-norms shifts reported, not hidden",
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
          "note": "wall evidence only; booking requires the exact-logits gate and the fp32 q/k census gate"}


def validate_cost_prediction(bracket: dict, control_census: dict, candidate_census: dict) -> dict:
  """Predicted-wall-delta gate (nv_fusion_cost_model.py).  The COST_PREDICTION
  table is derived from llama's rms_norm_f32 per-launch floors; the measured
  bracket delta must confirm the llama-shaped arithmetic or explain the gap.
  A measured delta beyond the envelope on the opposite side of zero is a
  CONTRADICTION and FAILS CLOSED: the premise was unbacked, so the campaign
  cannot book even if the raw bracket numbers promoted."""
  control_hist = {name: (count, med) for name, count, med in (control_census.get("histogram") or [])}
  candidate_hist = {name: (count, med) for name, count, med in (candidate_census.get("histogram") or [])}
  removed_terms = {}
  for name, (count, med) in control_hist.items():
    if not name.startswith(REPLACED_PREFIXES):
      continue
    drop = count - candidate_hist.get(name, (0, 0))[0]
    if drop > 0:
      removed_terms[name] = {"dropped_count": drop, "control_median_us": med, "mass_us": round(drop * med, 3)}
  removed_mass = sum(term["mass_us"] for term in removed_terms.values())
  added_terms = {}
  for name, (count, med) in candidate_hist.items():
    if not name.startswith("reduce_output_rmsnorm"):
      continue
    add = count - control_hist.get(name, (0, 0))[0]
    if add > 0:
      added_terms[name] = {"added_count": add, "candidate_median_us": med, "mass_us": round(add * med, 3),
                           "llama_floor_us": LLAMA_FLOOR_US.get(name)}
  added_mass = sum(term["mass_us"] for term in added_terms.values())
  llama_body_mass = sum(term["added_count"] * (term["llama_floor_us"] or 0.0) for term in added_terms.values())
  point = round(added_mass - removed_mass, 3)
  lo = round(llama_body_mass - removed_mass, 3)
  hi = round(point + max(1.0, llama_body_mass), 3)
  measured = -bracket["candidate_minus_control_bracket_us"]
  reconciliation = reconcile_cost_prediction(measured, {"predicted_delta_us": point, "range_us": [lo, hi]},
                                             tolerance_us=COST_PREDICTION["tolerance_us"])
  return {"run": True, "result": "PASS" if reconciliation["result"] != "CONTRADICTED" else "FAIL",
          "contract": COST_PREDICTION, "prediction": {"predicted_delta_us": point, "range_us": [lo, hi],
                                                      "llama_floor_body_mass_us": round(llama_body_mass, 3),
                                                      "removed_control_mass_us": round(removed_mass, 3)},
          "removed_terms": removed_terms, "added_terms": added_terms,
          "reconciliation": reconciliation, "measured_delta_us": measured,
          "bracket_field_us": bracket["candidate_minus_control_bracket_us"],
          "note": "positive measured delta = candidate slower; the llama floor envelope is bodies at zero cost .. bodies at twice the llama floor"}


HARD_STOP_NOTES = [
  "Every GPU arm runs as a fresh process under `timeout ... flock -w 90 /tmp/gpu-bench.lock`; no arm holds the lock across a wall-bracket step.",
  "Phase 0 (NV render smoke) must survive on sm_120 (no Xid 31 MMU fault) with the fused q/k bodies (reduce_output_rmsnorm_32_128 / reduce_output_rmsnorm_8_128) in the compiled program set.",
  "The exact full-logit gate (fp32 SHA-256 over the stacked rows, token stream, shape, per-row argmax == sampled token) must pass before any census or bracket arm runs.",
  "The census gate FAILS CLOSED if the q/k fused bodies do not appear (selector still rejects the fp32 route); q/k reduce drops equal body counts, q/k epilogue drops equal 2x body counts (two epilogue kernels per ordinary norm), C6 is 55 bodies (36 ffn + 18 attn + final norm) with the final epilogue role fusing to 0, and observed body counts must match the committed reference; non-norms family shifts are reported with exact program names.",
  "The wall bracket requires identical token-stream hashes and a candidate median at least +50 us/token faster than BOTH bracketing controls.",
  "The predicted-wall-delta gate (COST_PREDICTION + validate_cost_prediction) runs after the bracket: the measured delta must CONFIRM the llama-shaped prediction or EXPLAIN the gap with named residual causes; a CONTRADICTION (measured outside the envelope on the opposite side of zero) FAILS CLOSED and the campaign cannot book.",
  "The reduce-output route policy is production-promoted on NV sm_120 (P2 site-absorption booking, 882ce66a5); the control arm constructs the closed route-less graph explicitly (forced False flags on the model and every block), so the bracket still measures the route package vs the route-less graph. The stage-3 geometry commits change no production default; promotion lands only when this bracket passes.",
]

ISOLATION_NOTES = [
  "Isolation matrix inherited from the C6 sibling campaign (nv_reduce_output_primitive_ab.py): control (no flags) and callify-only and promo-only all return finite logits with real sampled tokens; callify-only and promo-only logits are byte-identical.",
  "The C6 candidate's JIT-captured decode logits tap returned all-NaN while the eager JIT=0 path was finite and correct; the exact-logits gate therefore runs the eager JIT=0 finite check inside the child before comparing stacked-row SHAs, and any non-finite row fails closed.",
  "The fp32 q/k route reuses the same candidate context (callify flags + reduce-output promotion) and the same census/logits children; its NV render behavior is measured fresh by the orchestrator's smoke arm under the shared GPU bench lock.",
]

CITATIONS = [
  "docs/task_workflow/input/nv-reduce-output-fp32-qk-route-scope-20260810.md",
  "docs/task_workflow/input/nv-reduce-output-phase6-route-efficiency-scope-20260810.md",
  "extra/llm_research/decode/nv_reduce_output_primitive_ab.py",
  "extra/llm_research/decode/nv_fusion_population_ledger.py",
  "extra/llm_research/decode/nv_shared_q8_progressive_qualification.py",
  "extra/llm_research/decode/nv_predispatch_full_logits_qualification.py",
]


def no_go_record(model: str = DEFAULT_MODEL, depth: int = 512) -> dict:
  """Base NO-GO record; the orchestrator overwrites each phase with the exact
  evidence as the campaign advances."""
  return {
    "schema": SCHEMA, "mode": "ab", "date": "2026-08-10",
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
      "run": False, "result": "NOT_AUTHORIZED", "reason": "campaign stopped before the fp32 census arm",
      "contract": "fused reduce-output bodies counted by name prefix; q/k reduce roles 36 -> 0 each and epilogue roles 72 -> 0; C6 55 bodies (36 ffn + 18 attn + final norm, final epilogue role 1 -> 0); FAIL CLOSED if the q/k bodies do not appear; body counts must match the committed reference; honest net program delta and non-norms side effects reported",
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
                and bool(smoke_result.get("fused_q_body_present"))
                and bool(smoke_result.get("fused_k_body_present")))
  record["smoke"] = {"run": True, "result": "PASS" if smoke_gate else "NO-GO", "evidence": smoke_result}
  if not smoke_gate:
    record["hard_stop_notes"] = HARD_STOP_NOTES + [
      "HARD STOP at Phase 0: smoke did not survive or the fused q/k bodies were absent from the compiled program set."]
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
      "HARD STOP at Phase 1: the exact full-logit gate failed; no census and no bracket arm ran."]
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
    record["hard_stop_notes"] = HARD_STOP_NOTES + ["HARD STOP at Phase 2: the census gate failed; no wall bracket ran."]
    return _write_record(record, pathlib.Path(args.out))
  try:
    wall = wall_bracket(args)
  except RuntimeError as exc:
    record["wall_bracket"] = {"run": True, "result": "NO-GO",
                              "reason": "a wall-bracket timing child failed; raw child stderr captured below",
                              "stderr": getattr(exc, "stderr", None) or str(exc)[-4000:]}
    record["hard_stop_notes"] = HARD_STOP_NOTES + [
      "HARD STOP at Phase 3: a wall-bracket timing child failed; no bracket result."]
    return _write_record(record, pathlib.Path(args.out))
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
  cost_gate = validate_cost_prediction(wall, control_census, candidate_census)
  record["cost_prediction"] = cost_gate
  if cost_gate["result"] == "FAIL":
    record["verdict"] = "NO-GO"
    record["hard_stop_notes"] = HARD_STOP_NOTES + [
      "HARD STOP at Phase 3: predicted-wall-delta CONTRADICTION (measured delta outside the llama-shaped envelope on the opposite side of zero); the premise was unbacked and cannot book even if the raw bracket numbers promoted."]
    return _write_record(record, pathlib.Path(args.out))
  record["verdict"] = "BOOKED" if booked else "NO-GO"
  if not booked:
    record["hard_stop_notes"] = HARD_STOP_NOTES + [
      "Wall bracket completed but did not promote; the measured deltas are recorded above. "
      "An unpromoted bracket keeps the q/k share of the norms row unbooked."]
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
      ap.error("smoke requires --arm candidate: the fused q/k bodies only exist under the candidate conditions")
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
