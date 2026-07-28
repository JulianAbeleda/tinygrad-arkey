# attn_qo staged re-certification at 9119a7462 — STOPPED at C8 transition (fix incomplete)

Date: 2026-07-20. Repo root: `/home/ubuntu/tinygrad-arkey`, clean HEAD `9119a7462d046b62c6efafb9a6329e9de53c26c4`
(the §5.4 C8 harness preconstruction fix). Hardware: AMD RX 7900 XTX (gfx1100). Single GPU lane, sequential.

## Outcome

**C6 correctness and C7 memory admission both reproduce cleanly at the new commit (PASS, zero mismatches,
both queues).** The frozen `attn_qo` staged family/bundle/HSACO were reused byte-for-byte, unchanged, as
required.

**C8 does NOT clear.** The `direct_packed -> staged_candidate` transition **still faults** (identical SQ
type-2 signature, sh0+sh1 wave 3, gfxhub page fault, MES-unresponsive GPU reset) even with the §5.4
preconstruction fix from commit `9119a7462` active. Worse: the exact `[staged_candidate, direct_packed,
staged_candidate]` control sequence that **passed** in the manual pre-fix exoneration experiment
(`attn-qo-c8-transition-lifecycle-exoneration-20260720.md`, in the `951d3615c` artifact directory) **now
faults at position 2** (the second candidate invocation, a reuse) when run through the *fixed* harness. Only
`[staged_candidate]` alone still passes.

This is the STOP condition in the task brief: "the transition STILL faults after preconstruction (means the
fix is incomplete)." Per method §12.9's amendment, the candidate remains **not disqualified** (the harness is
still implicated, not proven-clean), but `BLOCKED_AT_C8` stands and C8 timing was **not** attempted. GPU
health was verified clean before and after every run; both faults recovered via automatic GPU reset with no
lingering wedge (confirmed by rocm-smi and a live tinygrad AMD compute round-trip immediately after run 3).

## What was regenerated at 9119a7462 (all PASS)

Ordered commands actually run (from repo root, `DEV=AMD`, one GPU lane, PM4 with `AMD_AQL=0`, AQL with
`AMD_AQL=1`). `BUNDLE`/`FAM` below are the retained `951d3615c` bundle and the reused staged-family manifest
(the r1-family evidence file already has the exact `frozen_staged_family.v1` schema/shape and loads cleanly
against HEAD's bundle bytes — no regeneration needed, confirming byte-for-byte family reuse):

```
BUNDLE=docs/artifacts/qwen3-14b-prefill-attn-qo-staged-951d3615c-20260719/bundle
FAM=docs/artifacts/qwen3-14b-prefill-attn-qo-staged-951d3615c-20260719/evidence/qk-attn-qo-staged-951d3615c-final-r1-20260719-family.json
OUT=<this artifact>/evidence

# C4 runtime-preconstruction canary, both queues
AMD_AQL=0 python -m extra.qk.mmq_frozen_staged_family_execution c4 --role attn_qo \
  --frozen-bundle $BUNDLE --staged-family-manifest $FAM --queue-mode PM4 --output $OUT/c4-pm4.json
AMD_AQL=1 python -m extra.qk.mmq_frozen_staged_family_execution c4 --role attn_qo \
  --frozen-bundle $BUNDLE --staged-family-manifest $FAM --queue-mode AQL --output $OUT/c4-aql.json

# C5 phase-isolated prefix 1 -> 3, C6 full 20, both queues
for Q in PM4:0 AQL:1; do queue=${Q%%:*}; aql=${Q##*:}; for pe in 1 3 20; do
  AMD_AQL=$aql python -m extra.qk.mmq_frozen_staged_family_execution prefix --role attn_qo \
    --frozen-bundle $BUNDLE --staged-family-manifest $FAM --queue-mode $queue \
    --prefix-epochs $pe --runtime-canary $OUT/c4-${queue,,}.json \
    --output $OUT/<c5-${queue,,}-prefix$pe|c6-${queue,,}-full20>.json
done; done

# C7 authority snapshot (pins live commit/tree)
python -m extra.qk.mmq_staged_c7_authority collect --selected-device AMD \
  --output $OUT/c7-authority-snapshot.json

# C7 memory requirements + guarded capture + ledger build
python -m extra.qk.mmq_frozen_staged_c7_census requirements --role attn_qo \
  --frozen-bundle $BUNDLE --staged-family-manifest $FAM --output $OUT/c7-requirements.json
AMD_AQL=0 python -m extra.qk.mmq_frozen_staged_c7_census capture --role attn_qo \
  --frozen-bundle $BUNDLE --staged-family-manifest $FAM --queue-mode PM4 \
  --runtime-canary-isolation $OUT/c4-pm4.json --authority-snapshot $OUT/c7-authority-snapshot.json \
  --output $OUT/c7-capture-pm4.json
AMD_AQL=1 python -m extra.qk.mmq_frozen_staged_c7_census capture --role attn_qo \
  --frozen-bundle $BUNDLE --staged-family-manifest $FAM --queue-mode AQL \
  --runtime-canary-isolation $OUT/c4-aql.json --authority-snapshot $OUT/c7-authority-snapshot.json \
  --output $OUT/c7-capture-aql.json
python -m extra.qk.mmq_frozen_staged_c7_census build --role attn_qo \
  --frozen-bundle $BUNDLE --staged-family-manifest $FAM \
  --pm4-observation $OUT/c7-capture-pm4.json --aql-observation $OUT/c7-capture-aql.json \
  --authority-snapshot $OUT/c7-authority-snapshot.json --output $OUT/c7-ledger.json

# C6 composition (no CLI in extra/qk/mmq_attn_qo_c6_binding.py; called via compose_c6.py, CPU-only,
# no Device import) -> $OUT/c6-composition.json, PASS, evidence_identity
# sha256:894b9c18d8b0194a545ae41ea67a50440e0a489616df3b055359610602b6bd65

# Direct-packed queue qualifications (untimed), both queues
AMD_AQL=0 python -m extra.qk.mmq_attn_qo_c8_runtime --role attn_qo \
  --frozen-bundle $BUNDLE --staged-family-manifest $FAM --composition $OUT/c6-composition.json \
  --authority-snapshot $OUT/c7-authority-snapshot.json --queue-mode PM4 \
  --output $OUT/direct-packed-pm4-qualification.json
AMD_AQL=1 python -m extra.qk.mmq_attn_qo_c8_runtime --role attn_qo \
  --frozen-bundle $BUNDLE --staged-family-manifest $FAM --composition $OUT/c6-composition.json \
  --authority-snapshot $OUT/c7-authority-snapshot.json --queue-mode AQL \
  --output $OUT/direct-packed-aql-qualification.json

# C8 guarded route-sequence transition diagnostics (no CLI for run_guarded_persistent_c8_route_sequence;
# called via run_transition.py, reusing attn_qo_c8_runner_factory + the config above)
python run_transition.py --queue-mode PM4 --sequence staged_candidate --label run1
python run_transition.py --queue-mode PM4 --sequence direct_packed,staged_candidate --label run2
python run_transition.py --queue-mode PM4 --sequence staged_candidate,direct_packed,staged_candidate --label run3
```

## Per-stage results

| Gate | Queue | Status | Key facts |
|---|---|---|---|
| C4 canary | PM4 | PASS | clean timeline, zero-target dispatch |
| C4 canary | AQL | PASS | clean timeline, zero-target dispatch |
| C5 prefix 1 | PM4 | PASS | |
| C5 prefix 3 | PM4 | PASS | |
| C6 full 20 | PM4 | PASS | 2,621,440 values, `mismatch_count=0`, `max_abs_error=0.00341796875` (within `rtol=atol=0.003`) |
| C5 prefix 1 | AQL | PASS | |
| C5 prefix 3 | AQL | PASS | |
| C6 full 20 | AQL | PASS | 2,621,440 values, `mismatch_count=0`, same max error |
| C7 authority | — | PASS | `software_identity=sha256:c75c6266c6ceeb782ceb1849243ad1bbcb836314a2dc045c47a6adb1e7d06a1d`; `repository.commit=9119a7462d046b62c6efafb9a6329e9de53c26c4`, `clean=true` |
| C7 capture | PM4 | PASS | |
| C7 capture | AQL | PASS | |
| C7 ledger | — | PASS | `dense_fp16_weight_materialization=false`; `admitted_bytes=25248309248` |
| C6 composition | — | PASS | `evidence_identity=sha256:894b9c18d8b0194a545ae41ea67a50440e0a489616df3b055359610602b6bd65` |
| direct_packed qualification | PM4 | PASS | untimed, `qualification_only=true` |
| direct_packed qualification | AQL | PASS | untimed, `qualification_only=true` |
| C8 transition `[staged_candidate]` | PM4 | **PASS** | preconstruct fired; single clean invocation |
| C8 transition `[direct_packed, staged_candidate]` | PM4 | **FAULT** | preconstruct fired *before* the loop (confirmed: candidate construction happened before the direct route), yet the candidate's real first dispatch (position 1) still faults — SQ type-2, sh0+sh1 wave 3, gfxhub page fault, MES unresponsive, GPU reset. Same signature as the original `f0d7a09ce` disqualification. |
| C8 transition `[staged_candidate, direct_packed, staged_candidate]` | PM4 | **FAULT** | position 0 (first real candidate dispatch, `initialization_count=1`) PASSES cleanly (33.9 ms); position 1 `direct_packed` PASSES (647.7 ms); position 2 (candidate **reuse**, `invocation_index=1`) **FAULTS** — the same signature. This exact sequence PASSED in the pre-fix manual exoneration experiment; it does not pass through the fixed harness. |
| C8 timing | — | **NOT ATTEMPTED** | blocked behind the transition gate per method §7 (C8 exit evidence requires either a winner or an explicit measured fallback at the *actual* route boundary; a faulting transition forbids substituting route-isolated timing) |
| AQL transition diagnostics | — | **NOT RUN** | stopped after the PM4 finding per the single-GPU-lane STOP condition, to avoid burning further resets on a known-blocked path before triage |

## Interpretation

The §5.4 preconstruction fix in `9119a7462` calls `runners.candidate.preconstruct(queue_mode=...)` — which
performs `get_runtime` + code-upload + runtime-cache identity checks, explicitly **without any target
dispatch** — before the route loop starts. That is sufficient to make runtime/code-object *construction*
happen on a clean device. It is **not** sufficient to prevent the fault: the candidate's first *real*
dispatch, whenever it occurs after a `direct_packed` route has touched the device, still faults with the
identical signature, whether or not `preconstruct` ran first.

This contradicts the narrower hypothesis in the exoneration doc, which inferred from the manual 3-run
experiment that *construction-after-direct* was the failure mode and *dispatch-after-direct* (given prior
clean construction) was safe. Run 3 here is a direct counter-example: `preconstruct` gave the candidate a
clean *construction* at position 0 before either route ran, and its *first real dispatch* also happened at
position 0 (clean device, PASS) — matching the manual run 3 setup exactly — yet the *second* dispatch
(reuse, position 2, after an intervening `direct_packed` route) now faults, where the manual run 3 reuse at
the equivalent point passed.

The most likely explanations, in order of plausibility, none yet confirmed:
1. `preconstruct()` leaves `persistent_session_state` in a form that is subtly different from what natural
   lazy first-invocation construction left in the manual experiment (e.g., different buffer allocation
   order/timing relative to the runtime cache, or an extra HCQ queue/signal object created and discarded),
   so the two "successful position-0 construction" states are not actually equivalent despite identical
   logged `runtime_identity`/`buffer_ranges` shapes.
2. The fault is nondeterministic / VRAM-fragmentation-sensitive rather than strictly lifecycle-order-sensitive,
   and the manual 3-run experiment's PASS at position 2 was not fully deterministic.
3. Some other environment/state delta between the exoneration's detached-worktree run (at `f0d7a09ce`) and
   this run (at `9119a7462`, in-tree) matters even though both pin their own commit via `software_identity`.

None of these is established by the evidence collected here — establishing which requires further guarded
GPU diagnosis (e.g., re-run the exact manual pre-fix pattern again at HEAD without going through
`preconstruct` at all, to check hypothesis 2; diff `persistent_session_state` between a `preconstruct`-first
run and a natural-lazy-construction run to check hypothesis 1) before the harness fix can be trusted or a
disqualification can be considered under §12.9.

## Retained evidence (uncommitted)

All files under `evidence/` in this directory: `c4-{pm4,aql}.json`, `c5-{pm4,aql}-prefix{1,3}.json`,
`c6-{pm4,aql}-full20.json`, `c7-authority-snapshot.json`, `c7-requirements.json`,
`c7-capture-{pm4,aql}.json`, `c7-ledger.json`, `c6-composition.json`, `c6-correctness-evidence.json`,
`direct-packed-{pm4,aql}-qualification.json`, `runner-config.json`, `compose_c6.py` (composition driver),
`run_transition.py` (transition driver), `transition-run{1,2,3}-pm4.json` (the three guarded route-sequence
results, including full kernel-fault-evidence journal excerpts for runs 2 and 3).

Repo state: clean, no edits to any tracked file. All new files are under `docs/artifacts/` and untracked; the
`/tmp/qk-attn-qo-9119a7462-20260720` working copy is also left in place. GPU confirmed healthy and idle after
the last run (`rocm-smi` clean, live `Tensor.ones(4)+1` round-trip on `Device["AMD"]` succeeded).

## Recommendation

Do not lift `BLOCKED_AT_C8`. Do not attempt C8 timing. Before any further GPU work on this role:
1. Treat the `9119a7462` preconstruction fix as **not proven** to close the transition fault — despite
   passing its own unit tests (construction-only lifecycle correctness), it does not prevent the real-dispatch
   fault this task exists to clear.
2. Diagnose why `preconstruct`-mediated construction differs from natural lazy construction at the
   `persistent_session_state` / HCQ level — this is now a harness/runtime investigation, not an `attn_qo`
   kernel investigation (the kernel still passes standalone and in isolated C6 full-20 runs on both queues).
3. Re-run this exact PM4 diagnostic sequence only after a new fix lands; do not run the AQL side of this
   diagnostic (queue-mode-symmetric fault is very likely but unconfirmed) until the PM4 path is resolved, to
   conserve GPU-reset budget.
