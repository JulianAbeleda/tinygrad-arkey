#!/usr/bin/env python3
"""Authority gate for flipping PREFILL_SOFTMAX_REDUCE_FUSE's DEFAULT from off to on.

Sibling of extra/qk/prefill/prefill_causal_tile_skip_promotion_gate.py and deliberately the same shape: reads ONLY
pre-collected, transcribed evidence (docs/prefill-softmax-reduce-fuse-evidence-20260724.json), embeds no
measurements of its own, never invents a number, and FAILS CLOSED on any missing/malformed/null field
rather than passing by omission. It does not flip any default; it reports PASS/FAIL to inform a human.

This is NOT a new-route promotion. `PREFILL_SOFTMAX_REDUCE_FUSE` (commit 23b8e05fc) changes how the HIP
renderer NAMES intermediates inside the ALREADY-promoted `prefill_flash_attention_generated` row. Route id,
emitter, route_attribution chain and provenance (machine_authored_generated) are identical either way, so
this gate gatekeeps a default-value change inside one existing row's flag -- it does not restructure
extra/qk/route_manifest.py's ROUTES table.

WHERE THIS GATE IS STRICTER THAN ITS SIBLING, and why:

  The causal-tile-skip flag lived in the attention EMITTER. This one lives in tinygrad/renderer/cstyle.py,
  the renderer EVERY AMD kernel goes through. Its predicate fires for any Ops.CUSTOMI with dtype float and
  child_count > 1, and the decode route builds float CUSTOMI of exactly that family (flash_kernels.py's
  __builtin_amdgcn_fdot2; schedule/wmma/softmax.py's "bpermute" row-state broadcast). So this gate ALSO
  requires cross-route decode protection, which tinygrad/llm/prefill_policy.py:14
  _SHARED_ATTENTION_PROOF_FIELDS already demands by name (decode_nonregression_8b AND
  decode_nonregression_14b): "Enabling one shared compiler path changes both supported model routes. A
  synthetic or one-model proof is therefore not enough." Byte-identical decode code objects satisfy that
  more strongly than any timing comparison could; anything short of identity forces a MEASURED decode
  throughput+correctness comparison, and this gate says so instead of quietly accepting identity-by-assumption.

  It also requires whole-unit-suite failure-set EQUALITY (not zero -- there are pre-existing failures), for
  the same reason: a shared-renderer change can move a failure anywhere in the suite, so a -k subset is not
  a proof.

ON THROUGHPUT EVIDENCE: every manifest-admitted shape must show a real, above-noise speed win on the
changed kernel. Whole-model paired A/B is preferred. A shape may instead be carried by ATTENTION-LOCAL
paired A/B *only* via the narrow UNDERPOWERED_BY_INSTRUMENT hatch in _check_underpowered_whole_model, which
requires the whole-model run to have actually happened, the mechanism's predicted delta to be below what
this instrument can resolve at MIN_SIGNAL_TO_NOISE, the measurement to CORROBORATE that prediction, and no
regression at any length. Both branches require the measured signal to clear its own control noise by
MIN_SIGNAL_TO_NOISE. Absence of a whole-model number with no such justification is a FAIL.

Run: PYTHONPATH=. python3 extra/qk/prefill/prefill_softmax_reduce_fuse_promotion_gate.py
"""
from __future__ import annotations

import json
import pathlib
import sys

from extra.qk.route_manifest import ROUTES
from extra.qk.prefill.promotion_gate_common import load_evidence, fail, build_result, main as _common_main

GATE = "prefill_softmax_reduce_fuse_promotion"
ROUTE_ID = "prefill_flash_attention_generated"
FLAG = "PREFILL_SOFTMAX_REDUCE_FUSE"
SCHEMA = "prefill-softmax-reduce-fuse-promotion-evidence.v1"
EVIDENCE_PATH = pathlib.Path(__file__).resolve().parents[3] / "docs" / "prefill-softmax-reduce-fuse-evidence-20260724.json"

# Same methodology thresholds as the sibling gate (docs/prefill-needle-theories-20260724.md "Measurement
# methodology"): a delta only counts as signal if it clears the measured same-config noise by a real margin.
MIN_PAIRS = 3                     # independent same-session paired A/B runs for a whole-model claim
MIN_MEAN_DELTA_PCT = 1.0
MIN_SIGNAL_TO_NOISE = 2.0
MAX_NOISE_FLOOR_PCT = 1.0
MIN_CLEARING_LENGTHS = 2          # >=2 of the four whole-prefill lengths must clear both bars (see _check_whole_model_ab)
MAX_PREDICTION_MISMATCH_FRAC = 0.5  # measured vs mechanism-predicted whole-model delta must corroborate within 50%
MIN_ATTENTION_LOCAL_DELTA_PCT = 10.0   # attention-local timing isolates the changed kernel; demand a big win
MIN_ATTENTION_LOCAL_REPS = 2

# The decode geometries a decode-identity claim has to cover, keyed the same way as the prefill shapes.
DECODE_SHAPES = ("8B", "14B")


def _required_shape_keys() -> list[str]:
  """Every grid this route's manifest row claims eligibility for -- derived from the manifest, not hardcoded."""
  keys = []
  for g in ROUTES[ROUTE_ID]["shape_guards"]:
    hq = g.get("Hq")
    if hq == 32: keys.append("8B")
    elif hq == 40: keys.append("14B")
    else: keys.append(f"Hq={hq}")
  return keys


def _check_shared_renderer(ev: dict) -> list[str]:
  """The leg that has no analogue in the sibling gate: this flag is in the SHARED renderer."""
  fails = []
  sr = ev.get("shared_renderer_risk")
  if not isinstance(sr, dict):
    return ["shared_renderer_risk: missing entirely -- a cstyle.py change cannot be promoted without it"]
  ident = sr.get("decode_codegen_identity")
  if not isinstance(ident, dict) or ident.get("status") != "PASS":
    fails.append(f"shared_renderer_risk.decode_codegen_identity missing or not PASS ({ident!r}); "
                 f"if decode codegen DIFFERS, promotion needs MEASURED decode throughput + correctness in both arms")
  else:
    # Two ways this leg can be a lie: nothing was compiled, or the flag never took effect.
    for field in ("harness", "method", "proof_it_compiled", "result",
                  "non_vacuity_control", "flag_actually_took_effect_control"):
      if not ident.get(field):
        fails.append(f"decode_codegen_identity.{field} missing -- an identity claim with no proof that both "
                     f"arms actually compiled, and that the flag actually changed something somewhere, is not evidence")
  # prefill_policy.py:_SHARED_ATTENTION_PROOF_FIELDS names these two by name.
  for shape in DECODE_SHAPES:
    key = f"decode_nonregression_{shape.lower()}"
    if sr.get(key) is not True:
      fails.append(f"shared_renderer_risk.{key} is not True ({sr.get(key)!r}); "
                   f"tinygrad/llm/prefill_policy.py:_SHARED_ATTENTION_PROOF_FIELDS requires both model routes")
  if not sr.get("decode_nonregression_basis"):
    fails.append("shared_renderer_risk.decode_nonregression_basis missing -- say WHAT proves non-regression")
  return fails


def _check_suite_and_guards(ev: dict) -> list[str]:
  fails = []
  suite = ev.get("full_unit_suite_failure_set")
  if not isinstance(suite, dict) or suite.get("status") != "PASS":
    fails.append(f"full_unit_suite_failure_set missing or not PASS ({suite!r})")
  else:
    if suite.get("failure_set_diff") not in ("EMPTY", "EMPTY -- byte-identical sorted failure sets"):
      fails.append(f"full unit suite failure sets are NOT equal: {suite.get('failure_set_diff')!r}")
    off, on = suite.get("off"), suite.get("on")
    if not isinstance(off, dict) or not isinstance(on, dict):
      fails.append("full_unit_suite_failure_set needs both an off and an on arm summary")
    elif (off.get("failed"), off.get("passed")) != (on.get("failed"), on.get("passed")):
      fails.append(f"unit suite counts differ: off={off.get('failed')}F/{off.get('passed')}P "
                   f"on={on.get('failed')}F/{on.get('passed')}P")
  guards = ev.get("guards")
  if not isinstance(guards, dict):
    fails.append("guards block missing")
  else:
    for name in ("assert_pure_machine_search", "validate_manifest"):
      g = guards.get(name)
      if not isinstance(g, dict) or g.get("status") != "PASS":
        fails.append(f"guards.{name} missing or not PASS ({g!r})")
    if guards.get("validate_manifest", {}).get("violations") not in ([], None) and \
       guards.get("validate_manifest", {}).get("violations"):
      fails.append(f"validate_manifest reported violations: {guards['validate_manifest']['violations']!r}")
    if guards.get("provenance", "").split(" ")[0] != "machine_authored_generated":
      fails.append(f"guards.provenance={guards.get('provenance')!r}, expected machine_authored_generated")
  return fails


def _check_whole_model_ab(name: str, ab: dict) -> list[str]:
  """Whole-model paired A/B, scored against what the MECHANISM predicts rather than uniformly.

  DELIBERATE DIVERGENCE FROM THE SIBLING GATE, recorded here so it cannot be mistaken for a relaxation of
  convenience. prefill_causal_tile_skip_promotion_gate.py requires BOTH pp512 and pp4096 to clear 2x the
  noise floor, because tile-skipping removes trip count at every depth. This flag removes ~292 instructions
  from the KV-loop BODY, so its whole-model effect is (KV-loop share of chunk time) x (kernel speedup) --
  intrinsically smallest at pp512, where the KV loop is the smallest share of the chunk, and largest at
  pp4096. Demanding equal signal/noise at pp512 would be demanding the effect be LARGER than its own theory
  predicts, and would reject a change for behaving exactly as understood.

  So instead: the deepest length (where the mechanism predicts the most) must clear the bar; at least
  MIN_CLEARING_LENGTHS of the measured lengths must clear it; and NO measured length may regress beyond the
  noise floor. Plus attention_local_paired_ab is mandatory for every shape (see _check_shape), which times
  the changed kernel directly instead of inferring it through a whole-model number. Net: stricter than the
  sibling on regression coverage and on direct-kernel evidence, looser only where the theory says to be.
  """
  fails = []
  pairs = ab.get("pairs") or []
  if len(pairs) < MIN_PAIRS:
    fails.append(f"shape {name}: only {len(pairs)} paired A/B runs recorded, need >= {MIN_PAIRS}")
  noise = ab.get("back_to_back_same_config_noise_pct")
  if noise is None or noise > MAX_NOISE_FLOOR_PCT:
    fails.append(f"shape {name}: same-config noise floor missing or too high ({noise!r} > {MAX_NOISE_FLOOR_PCT})")
    return fails
  deltas = {m: ab.get(f"pp{m}_mean_delta_pct") for m in (512, 1024, 2048, 4096)}
  if any(v is None for v in deltas.values()):
    fails.append(f"shape {name}: whole-model deltas missing for {[k for k,v in deltas.items() if v is None]} "
                 f"-- all four whole-prefill lengths must be reported so a regression cannot hide in an unreported one")
  measured = {k: v for k, v in deltas.items() if v is not None}
  # 1. No length may regress past the noise floor.
  for k, v in measured.items():
    if v < -noise:
      fails.append(f"shape {name}: pp{k} REGRESSED {v}% (worse than -{noise}% noise floor)")
  # 2. The deepest measured length must clear the bar -- that is where the mechanism predicts the most.
  if measured:
    deep = max(measured)
    dv = measured[deep]
    if dv < MIN_MEAN_DELTA_PCT:
      fails.append(f"shape {name}: deepest length pp{deep} delta {dv}% below required {MIN_MEAN_DELTA_PCT}%")
    elif (dv / noise) < MIN_SIGNAL_TO_NOISE:
      fails.append(f"shape {name}: deepest length pp{deep} signal/noise {dv / noise:.2f}x below "
                   f"required {MIN_SIGNAL_TO_NOISE}x")
  # 3. Enough lengths must clear the bar that this is not one lucky point.
  clearing = [k for k, v in measured.items() if v >= MIN_MEAN_DELTA_PCT and (v / noise) >= MIN_SIGNAL_TO_NOISE]
  if len(clearing) < MIN_CLEARING_LENGTHS:
    fails.append(f"shape {name}: only {len(clearing)} whole-prefill length(s) {clearing} clear "
                 f"{MIN_MEAN_DELTA_PCT}% and {MIN_SIGNAL_TO_NOISE}x noise, need >= {MIN_CLEARING_LENGTHS}")
  # 4. A loop-BODY win must grow with depth. If it does not, the attribution is wrong even if the number is up.
  if len(measured) >= 2:
    ks = sorted(measured)
    if measured[ks[-1]] <= measured[ks[0]]:
      fails.append(f"shape {name}: delta does not grow with context (pp{ks[0]}={measured[ks[0]]}% -> "
                   f"pp{ks[-1]}={measured[ks[-1]]}%); a KV-loop-body win must scale with KV depth, so this "
                   f"number is not attributable to the change")
  return fails


def _check_attention_local(name: str, al: dict) -> list[str]:
  """Attention-local device-synced timing of the CHANGED kernel, with the unchanged SDPA baseline as control."""
  fails = []
  if al.get("all_configs_numeric_ok") is not True:
    fails.append(f"shape {name}: attention_local_paired_ab.all_configs_numeric_ok is not True -- a speed "
                 f"number from an unverified result is not evidence")
  fused = al.get("fused_ms")
  control = al.get("control_sdpa_ms_spread_pct")
  if not isinstance(fused, dict) or not fused:
    return fails + [f"shape {name}: attention_local_paired_ab.fused_ms missing"]
  if not isinstance(control, dict) or not control:
    return fails + [f"shape {name}: attention_local_paired_ab.control_sdpa_ms_spread_pct missing -- without an "
                    f"unchanged-code control there is no noise floor to clear"]
  for kv, row in fused.items():
    off, on = row.get("off") or [], row.get("on") or []
    if len(off) < MIN_ATTENTION_LOCAL_REPS or len(on) < MIN_ATTENTION_LOCAL_REPS:
      fails.append(f"shape {name} {kv}: needs >= {MIN_ATTENTION_LOCAL_REPS} interleaved reps per arm "
                   f"(got {len(off)}/{len(on)})")
    delta = row.get("mean_delta_pct")
    if delta is None or -delta < MIN_ATTENTION_LOCAL_DELTA_PCT:
      fails.append(f"shape {name} {kv}: mean_delta_pct={delta!r} is not a speedup of at least "
                   f"{MIN_ATTENTION_LOCAL_DELTA_PCT}% (negative == faster)")
      continue
    noise = control.get(kv)
    if noise is None:
      fails.append(f"shape {name} {kv}: no control noise recorded for this config")
    elif noise and (-delta / noise) < MIN_SIGNAL_TO_NOISE:
      fails.append(f"shape {name} {kv}: signal/noise {-delta / noise:.2f}x below required {MIN_SIGNAL_TO_NOISE}x")
  return fails


def _check_shape(name: str, entry: dict | None) -> list[str]:
  if entry is None:
    return [f"shape {name}: no evidence entry at all"]
  fails = []
  numerics = entry.get("numerics")
  if not numerics or numerics.get("status") != "PASS":
    fails.append(f"shape {name}: numerics missing or not PASS ({numerics!r})")
  else:
    # A rendering-only change must be numerically INERT, not merely within tolerance. Sweeps that report a
    # scalar error only (prefill_hd_sweep_numerics prints max_abs_err, not the tensor) cannot establish
    # bit-identity, so they are not required to claim it -- but at least one sweep must establish it, and no
    # sweep may contradict it.
    claims = [sub.get("on_off_bit_identical") for sub in numerics.values()
              if isinstance(sub, dict) and "on_off_bit_identical" in sub]
    if any(x is not True for x in claims):
      fails.append(f"shape {name}: a numerics sweep reports ON/OFF output NOT bit-identical ({claims!r}); "
                   f"a renderer-naming change that moves a bit is not the change it claims to be")
    if not claims:
      fails.append(f"shape {name}: no numerics sweep establishes ON/OFF bit-identity of the output tensor; "
                   f"matching max_abs_err scalars are weaker and do not substitute")
    for key, sub in numerics.items():
      if isinstance(sub, dict) and not sub.get("harness"):
        fails.append(f"shape {name}: numerics.{key} does not name the harness that produced it")
  parity = entry.get("token_parity")
  if not parity or parity.get("status") != "PASS" or parity.get("match_in_both_arms") is not True:
    fails.append(f"shape {name}: token_parity missing, not PASS, or not matched in BOTH arms ({parity!r})")

  ab = entry.get("whole_model_paired_ab")
  al = entry.get("attention_local_paired_ab")
  if isinstance(ab, dict) and ab.get("status") == "PASS":
    fails += _check_whole_model_ab(name, ab)
    if isinstance(al, dict) and al.get("status") == "PASS": fails += _check_attention_local(name, al)
  elif isinstance(ab, dict) and ab.get("status") == "UNDERPOWERED_BY_INSTRUMENT":
    fails += _check_underpowered_whole_model(name, ab)
    if not isinstance(al, dict) or al.get("status") != "PASS":
      fails.append(f"shape {name}: whole-model arm is under-powered, so attention_local_paired_ab MUST carry the "
                   f"shape, and it is missing or not PASS ({al!r})")
    else:
      fails += _check_attention_local(name, al)
  else:
    fails.append(f"shape {name}: whole_model_paired_ab missing or not PASS/UNDERPOWERED_BY_INSTRUMENT ({ab!r})")
  return fails


def _check_underpowered_whole_model(name: str, ab: dict) -> list[str]:
  """Escape hatch for a shape whose whole-model instrument cannot credibly resolve the predicted effect.

  Deliberately narrow, and NOT satisfiable by "we didn't measure it". The shape must still have been
  measured; the mechanism must predict a delta this instrument cannot resolve at MIN_SIGNAL_TO_NOISE; the
  measurement must AGREE with that prediction (two independent instruments corroborating each other is the
  actual evidence here, and it is only available because both were run); and nothing may regress.

  Why this exists: 14B must run with TINYGRAD_PREFILL_PACKED_WMMA=0 to avoid a known GPU fault, which
  disables its packed-WMMA prefill fast path and leaves the chunk ~94% GEMM-bound. The attention kernel this
  flag changes is then a few percent of chunk time, so even a 25-30% kernel win lands near the noise floor
  at the whole-model level. That is a property of the only 14B-safe configuration on this box, not a property
  of the change.
  """
  fails = []
  if not ab.get("why"):
    fails.append(f"shape {name}: whole_model_paired_ab=UNDERPOWERED_BY_INSTRUMENT with no 'why'")
  pairs = ab.get("measured_pairs") or []
  if len(pairs) < 1:
    fails.append(f"shape {name}: UNDERPOWERED_BY_INSTRUMENT still requires the measurement to have been RUN; "
                 f"no measured_pairs recorded. 'Under-powered' is not a synonym for 'not attempted'")
  noise = ab.get("noise_floor_pct")
  pred = ab.get("predicted_whole_model_delta_pct")
  meas = ab.get("measured_mean_delta_pct")
  if noise is None or pred is None or meas is None:
    fails.append(f"shape {name}: UNDERPOWERED_BY_INSTRUMENT needs noise_floor_pct, "
                 f"predicted_whole_model_delta_pct and measured_mean_delta_pct (got {noise!r}, {pred!r}, {meas!r})")
    return fails
  # 1. The instrument must genuinely be unable to resolve what the mechanism predicts.
  if noise and (pred / noise) >= MIN_SIGNAL_TO_NOISE:
    fails.append(f"shape {name}: predicted whole-model delta {pred}% is {pred / noise:.2f}x the {noise}% noise "
                 f"floor, i.e. this instrument CAN resolve it at the {MIN_SIGNAL_TO_NOISE}x bar -- so measure it "
                 f"properly and use the PASS branch instead of this hatch")
  # 2. Prediction and measurement must corroborate. This is the leg that makes the hatch evidence.
  if pred and abs(meas - pred) / abs(pred) > MAX_PREDICTION_MISMATCH_FRAC:
    fails.append(f"shape {name}: measured whole-model delta {meas}% disagrees with the {pred}% predicted from the "
                 f"attention-local win and attention's share of chunk time by more than "
                 f"{MAX_PREDICTION_MISMATCH_FRAC:.0%} -- the two instruments do not corroborate, so neither is trusted")
  # 3. Nothing may regress, at any length.
  for k, v in (ab.get("per_length_mean_delta_pct") or {}).items():
    if v < -noise:
      fails.append(f"shape {name}: pp{k} REGRESSED {v}% (worse than -{noise}% noise floor)")
  if not ab.get("per_length_mean_delta_pct"):
    fails.append(f"shape {name}: UNDERPOWERED_BY_INSTRUMENT must report per_length_mean_delta_pct so a regression "
                 f"cannot hide behind a favourable mean")
  return fails


def evaluate() -> dict:
  route_row = ROUTES.get(ROUTE_ID)
  if route_row is None:
    return fail(GATE, f"manifest has no route {ROUTE_ID!r}")
  if route_row.get("provenance") != "machine_authored_generated":
    return fail(GATE, f"{ROUTE_ID} provenance={route_row.get('provenance')!r}, expected machine_authored_generated "
                       f"(a renderer naming change must not move this row out of the allowed-default set)")

  required = _required_shape_keys()
  evidence = load_evidence(EVIDENCE_PATH, SCHEMA, ROUTE_ID, FLAG)
  if evidence is None:
    return fail(GATE, f"evidence artifact missing or invalid: {EVIDENCE_PATH}", required_shapes=required)

  shapes = evidence.get("shapes", {})
  failures: dict[str, list[str]] = {}
  for name in required:
    if fails := _check_shape(name, shapes.get(name)): failures[name] = fails
  if fails := _check_shared_renderer(evidence): failures["_shared_renderer"] = fails
  if fails := _check_suite_and_guards(evidence): failures["_suite_and_guards"] = fails

  verdict = "PASS" if not failures else "FAIL"
  return build_result(
    gate=GATE, route_id=ROUTE_ID, flag=FLAG,
    provenance=route_row.get("provenance"),
    required_shapes=required,
    evidence_path=EVIDENCE_PATH,
    failures=failures,
    verdict=verdict,
    extra={"required_decode_shapes": list(DECODE_SHAPES)},
    note=("every manifest-admitted prefill shape has complete passing evidence, decode codegen is proven "
          "byte-identical on both decode-admitted geometries, the whole unit-test failure set is unchanged, "
          "and the purity/manifest guards pass -- the default may be flipped ON" if verdict == "PASS" else
          "at least one leg is incomplete; leave the default OFF"),
  )


def main() -> int:
  return _common_main(evaluate)


if __name__ == "__main__":
  sys.exit(main())
