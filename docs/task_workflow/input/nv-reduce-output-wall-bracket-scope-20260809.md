# NV generic reduce-output primitive wall bracket scope (exact-logits + norms row)

Date: 2026-08-09
Branch: `nvidia-bringup-20260731`, HEAD `026396f78` (post CPU capability gate
PASS for the generic cooperative reduction-to-output primitive).
Status: **measurement scope. Authorizes the NV exact-logits gate plus the
reverse control/candidate/control wall bracket for the norms row under the
shared GPU bench lock, and (gated on that result) the route-efficiency
follow-up. No policy promotion, no model wiring change, no correctness
contract weakening.**

## 1. Why this scope exists

The 08-07 capability audit attributed 658.359 us of the 662.128 us
fusion/dataflow ledger bucket to ONE missing construct, C1: a generic
cooperative reduction-to-output primitive. The norms row (+495.330 us
attribution, +19.27 tok/s ceiling) and the flash row (+163.029 us, +5.93
tok/s ceiling) both sit behind it.

The CPU capability gate (`nv-generic-reduce-output-primitive-record-20260809.md`)
just proved the construct is reachable in the production ordinary DAG for the
first time: the spec/emitter is fully spec-driven, the C6 CALL-input spelling
is admitted through the M4-style typed-view proof, the hermetic gate is 10/10
on `DEV=CPU`, and the production decode census shows **108 selector
admissions / 54 fused `reduce_output_rmsnorm_1_4096` bodies** (both baselines:
0), all on the `16x32x8` (`r_16_256`, dim 4096) association.

That gate deliberately claimed no recovered wall: the NV render path (the
Xid 31 class) was not re-tested, the fused body is bitwise proven only in the
hermetic form, and the census is an admission census, not a timing bracket.
The 495.330 us / 163.029 us rows remain ledger attribution behind C1.

This scope is the missing measurement arm: exact full-logit equality first
(correctness is never weakened), then a norms-confined census, then the
serialized reverse control/candidate/control wall bracket under lock. It
books the norms row only when the bracket promotes; otherwise it records the
honest evidence and (gated) scopes the route-efficiency fix that the census
surfaced.

## 2. Current state and the known caveat

The census artifact
`docs/task_workflow/output/nv-generic-reduce-output-census-20260809.json`
records the honest caveat verbatim:

> The CALL-input route emits one fused body per consuming call argument plus
> one weight materialization each, so the captured decode call count is not
> reduced (net +72 vs the typed baseline, +1 vs the ordinary baseline). No
> recovered wall is claimed. The hermetic single-consumer STORE/CALL form
> still fuses 2->1 in isolation.

Consequences for the bracket design:

- The candidate arm MUST reproduce the census conditions exactly:
  `_decode_reduce_output_rmsnorm_promoted = True` on the model and every
  block, plus `CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT = 1` and
  `CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER = 1`. The census that produced 108/54
  ran with those two callify Context flags; a candidate arm without them
  measures a different graph (0 admissions).
- The control arm MUST be the closed production graph: no promotion flags, no
  callify Context flags, `promoted_targets: []` unchanged.
- Because net call count is +1 vs the ordinary baseline, a wall win is not
  guaranteed: 54 fused bodies replace the ordinary two-program pairs they
  consume, but the route also adds per-argument bodies and weight
  materializations. The bracket measures the truth; the record reports it
  either way. If the bracket does not promote, Phase 6 (route efficiency) is
  authorized to close the +72.

## 3. Campaign design (all GPU arms under `/tmp/gpu-bench.lock`)

The harness is a new module
`extra/llm_research/decode/nv_reduce_output_primitive_ab.py` modeled on the
existing `nv_norms_fusion_ab.py` conventions: fresh process per arm,
`timeout ... flock -w ...` children, settled-continuous timing windows, and
the same gate ordering. The construction differs from the norms epilogue
campaign: the boundary-free construction gate is NOT part of this campaign
(the CPU gate already proved construction); the campaign starts at exact
logits.

### Phase 0: NV render smoke (Xid 31 class re-test)

Before any logits/timing arm, a fresh child under lock compiles and runs the
fused `reduce_output_rmsnorm_1_4096` body on NV sm_120 with the candidate
conditions. Success: the child process survives (no Xid 31 MMU fault), the
decode completes at least one token, and the fused body appears in the
compiled program set. Failure: HARD STOP, NO-GO record with the raw child
stderr; no further arms.

### Phase 1: exact full-logit gate

Two fresh children under lock, `control` and `candidate`, each serializing
full fp32 logits over `count` decoded rows at fixed depth (d512 authority
corpus, same `_load`/`_prompt` conventions as
`nv_predispatch_full_logits_qualification.py`). Gate contract (identical to
the established norms A/B):

1. `logits_sha256` identical (bitwise fp32 over the stacked rows).
2. Token stream identical.
3. Shape identical; per-row argmax equals the sampled token.

Failure of any clause: HARD STOP, NO-GO record. The correctness contract is
never weakened.

### Phase 2: norms-confined census

Two fresh children under lock (`control`, `candidate`) capturing the DEBUG
kernel census of one decode token. Gate contract, modeled on
`validate_census` from `nv_norms_fusion_ab.py` but with the reduce-output
construction's expectations:

- Non-norms populations unchanged between control and candidate.
- Norms reduce roles unchanged in count.
- Candidate shows the expected `reduce_output_rmsnorm_1_4096` bodies (the
  norms reduce/epilogue programs they replace disappear from the norms
  population); `epilogues_removed > 0` and `fused_bodies > 0`.
- The honest net program delta (+72 vs typed baseline, +1 vs ordinary) is
  recorded in the census artifact, not hidden.

### Phase 3: reverse control/candidate/control wall bracket

Three fresh children under lock in fixed order control / candidate / control,
settled-continuous windows (`_settled_continuous_windows`), each returning
median ms/token plus the token-stream hash. Promotion requires (same as the
established `validate_timing_bracket`):

- All three token-stream hashes identical.
- Candidate median at least +50 us/token faster than BOTH bracketing controls.

The +50 us threshold is the established promotion bar. The norms row
attribution is +495.330 us; the bracket books what the 54 bodies actually
recover, and the record converts the measured delta to tok/s (ceiling
207.09 tok/s if the full 495.330 us were recovered; any positive booked delta
is smaller).

### Phase 4: record and booking

- BOOKED: exact-logits PASS + census PASS + bracket promoted. Record writes
  the booked delta, tok/s conversion, and the exact per-arm evidence.
- NO-GO with evidence: any gate failure or unpromoted bracket. Record writes
  the exact evidence and the reason. An unpromoted bracket due to the net
  call-count overhead authorizes Phase 6.

## 5. Success criteria

1. Hermetic harness tests green on CPU (`test/unit/test_nv_reduce_output_primitive_ab.py`):
   validators, gate ordering, child command construction, census expectations,
   and the candidate-arm conditions (promotion flag + both callify Context
   flags) fail closed if missing.
2. Existing tripwire green: `test_generic_reduce_output.py`,
   `test_reduce_output_rmsnorm.py`, `test_shared_q8_attention_landing.py`,
   `test_decode_graph_position_invariance.py`, M4/M5 sets, `python3 sz.py`
   within budget.
3. Phase 0 (NV render smoke) PASS under lock, or a documented HARD STOP.
4. Exact-logits gate PASS, or NO-GO with evidence.
5. Census PASS with the honest net program delta recorded, or NO-GO.
6. Wall bracket run to completion; verdict BOOKED or NO-GO with the exact
   deltas; no recovered-wall claim beyond the bracket.
7. Record doc + artifacts committed and pushed; policy JSON untouched.

## 6. HARD STOP

- No GPU arm outside `flock -w 600 /tmp/gpu-bench.lock`; every arm is a fresh
  process; no arm may hold the lock across a wall bracket step.
- No policy promotion: `decode-reduce-output-rmsnorm-route-policy.json` stays
  `promoted_targets: []`; no model wiring change; no default flip.
- No M4/M5/Path-3/M3 record changes; no changes to `decode_routes.py`,
  `qk_primitives.py`, or the shared-Q8 promotion record.
- No `--no-verify`; code changes require test changes in the same commit;
  `[nv]` prefix for code, `[docs]` for records/artifacts.
- Scratch in `/tmp` only (disk ~99% full); no large artifacts in the repo;
  the census/timing artifacts committed to
  `docs/task_workflow/output/` are small JSON only.
- The 08-05 body digest pin stays (hermetic test already locks it).
- If exact-logits fails, no census and no bracket arm runs; if census fails,
  no bracket arm runs.

## 7. Deliverables

1. Harness `extra/llm_research/decode/nv_reduce_output_primitive_ab.py` plus
   hermetic unit tests `test/unit/test_nv_reduce_output_primitive_ab.py` in
   one `[nv]` commit.
2. GPU campaign artifacts under lock (smoke, logits control/candidate,
   census control/candidate, timing control/candidate/control) committed as
   small JSON to `docs/task_workflow/output/`.
3. Record `docs/task_workflow/input/nv-reduce-output-wall-bracket-record-20260809.md`
   with verdict (BOOKED / NO-GO), exact evidence, tok/s conversion, and the
   reason this books (or does not yet book) the 495.330 us norms row; flash
   row (+163.029 us) explicitly out of scope here (requires the flash
   construct, separately scoped).
4. Gated Phase 6 follow-up scope (route efficiency) only if the bracket is
   NO-GO for the net-call-count reason.
5. Commits on `nvidia-bringup-20260731`, pushed.

## 8. Why this is not a wall claim

Booking requires the exact-logits gate, the census, and the promoted
bracket, in that order, each under lock in a fresh process. The record
converts only the measured median delta into tok/s. The ceilings from the
08-05 audit (207.09 tok/s at full norms booking, +19.27) are reference math,
not predictions.
