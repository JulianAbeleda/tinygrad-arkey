# Decode Standalone-Retention Staged Attack Scope

Date: 2026-06-19

## Purpose

Attempt, in a disciplined staged way, to preserve more of tinygrad's standalone decode MMVQ efficiency inside the real
model.

Authority numbers:

| Metric | Value |
|---|---:|
| tinygrad standalone GEMV/MMVQ surface | `~76%` HBM peak |
| tinygrad in-model weight-GEMV bucket | `~44%` HBM peak |
| llama.cpp in-model weight-GEMV bucket | `~54%` HBM peak |
| target if we match llama retention | `~1.19x` decode |
| theoretical full standalone transfer | `~1.56x` decode |

The goal is not to promise `76%` retention. The goal is to audit and attack every prerequisite that would be required
for it, stopping whenever a stage fails its movement gate.

## Current State

Already known:

- Decode is HBM weight-streaming-bound, not VALU/codegen-bound.
- Standalone tinygrad GEMV/MMVQ can be strong.
- Q4_K `blk.0.attn_output` is not a runtime/cache identity bug:
  - in-model ATT role join launches `q4k_coop_partial_4096_4096`;
  - plus `r_32_32_4_8` reduce/glue and `E_32_32_4n1` glue;
  - ATT body attribution passes.
- Existing env knobs (`Q4K_COOP_RT`, `Q6K_COOP_RT`, coop on/off) did not move a high-share role enough.
- Prior B2 runtime/cache identity doc says broad program identity closes, but ATT has only role-joined Q4_K
  `attn_output` so far.

Open:

- Q6_K `ffn_down` and `lm_head` need the same ATT role-join evidence.
- The aggregate reduce/glue/stage-2 tax across roles is not fully priced against W==D decode.
- We have not attempted a direct-output/reduce-fusion proof after ATT made the role lifecycle visible.
- q8 activation reuse exists as a research route, but native producer work is still codegen/lifecycle-walled and reuse
  count is capped for Q4_K gate/up.

## Non-Goals

- Do not change defaults.
- Do not start another standalone kernel benchmark.
- Do not tune old env knobs again.
- Do not claim quality for lossy q8 routes without dNLL.
- Do not reopen spec-decode here; spec remains orthogonal and may be the pragmatic route if native retention stalls.

## Stage 0 - Source-of-Truth Setup

Goal: prevent stale results from steering the attack.

Work:

- Use `inference-perf-measured-map-20260619.md` plus this scope as authority.
- Treat `76%` as an oracle ceiling, not an expected outcome.
- Track every stage with one of:
  - `bounded_fix`;
  - `project_level`;
  - `closed`;
  - `research_only`.

Deliverable:

- this scope and a README pointer.

## Stage 1 - Q6_K Role-Joined ATT

Goal: determine whether high-share Q6_K roles are using the intended in-model program/lifecycle.

Targets, in order:

| Target | Why |
|---|---|
| `blk.0.ffn_down` | high-share Q6_K role; FFN dominates decode |
| `lm_head` | huge Q6_K output projection; often a decode tail bottleneck |

Work:

1. Extend `extra/qk_att_inmodel_role_join.py` to target arbitrary roles.
2. Capture the real activation for each role:
   - `ffn_down`: run block-0 attention + gate/up/SwiGLU input path and isolate the `ffn_down` call;
   - `lm_head`: capture final hidden state or use the smallest equivalent in-model activation with the installed
     Q6_K primitive.
3. Open ATT interval around just the role call.
4. Capture HCQ program identities in that interval:
   - program name;
   - launch shape;
   - code hash;
   - reduce/glue kernels;
   - body packet counts and wave packet counts.

Artifacts:

- `bench/qk-att-inmodel-role-join/q6_ffn_down.json`
- `bench/qk-att-inmodel-role-join/q6_lm_head.json`

Pass:

- ATT body packets > 0;
- Q6_K native coop/primitive program present;
- no dense fallback;
- reduce/glue kernels identified.

Decision:

| Finding | Next |
|---|---|
| wrong/fallback program | bounded runtime/route fix |
| intended Q6 program + large reduce/glue | Stage 2 |
| intended Q6 program + little glue | Stage 3 scheduler/resource |
| ATT cannot attribute Q6 role | close ATT for Q6 and use HCQ program identity + timing only |

## Stage 2 - Reduce/Glue Tax Ledger

Goal: price the stage-2 reduce/glue lifecycle across all high-share roles.

Roles:

- Q4_K `attn_q/o`;
- Q4_K `ffn_gate/up`;
- Q6_K `ffn_down`;
- Q6_K `lm_head`;
- Q6_K `attn_k/v` only if share justifies it.

Work:

1. For each role, count kernels in the role interval:
   - main MMVQ;
   - reduce;
   - elementwise/glue;
   - layout/reshape.
2. Use existing timing/PMU authority where reliable; otherwise use A/B isolation:
   - current route;
   - main-kernel-only approximation if available;
   - reduce/glue disabled or replaced only if correctness can be preserved.
3. Produce an Amdahl ledger:
   - local role tax;
   - weight-GEMV bucket share;
   - projected W==D movement.

Artifact:

- `bench/qk-decode-standalone-retention/reduce_glue_ledger.json`

Build gate:

- a single role or shared reduce primitive must project `>=5%` W==D movement, or local role movement `>=10%` on a
  high-share role.

Kill:

- if reduce/glue sums to low single-digit decode movement, do not build reduce fusion.

## Stage 3 - One Direct-Output / Reduce-Fusion Proof

Run only if Stage 2 clears the build gate.

Goal: prove the lifecycle tax can be removed for one role without breaking graph capture.

Candidates:

| Candidate | Surface | Notes |
|---|---|---|
| Q4_K direct output | `attn_output` or `attn_q` | existing `kernel_mode=direct_out` precedent, but coop currently writes partials |
| Q6_K direct output | `ffn_down` or `lm_head` | likely higher Amdahl if reduce/glue is material |
| in-kernel wave reduce | Q4/Q6 coop | harder; must avoid VGPR/spill and preserve memory coalescing |

Work:

- implement behind a research flag only;
- one role first;
- compare role-local correctness and timing;
- then W==D ctx sweep if role movement clears threshold.

Gate:

- role-local `>=10%` improvement;
- projected W==D `>=5%`;
- graph capture survives;
- byte-identical or fp-reassociation-tolerant.

Kill:

- if direct output lowers memory coalescing/occupancy and loses the role-local gate.

## Stage 4 - Activation Lifecycle / q8 Reuse

Run after Q6 identity and reduce/glue are understood.

Goal: decide whether q8 activation reuse is still worth native work.

Known facts:

- reuse is best for `ffn_gate` + `ffn_up` (`2` consumers);
- prior q8 research route moved gate/up lifecycle but projected only `~3-6%` decode;
- native fused producer is blocked by multi-granularity reduction + multi-output stores.

Work:

- refresh q8 gate/up route only if Stage 1-3 leave `ffn_gate/up` as a top remaining gap;
- require W==D A/B and dNLL;
- do not fund native producer unless artifact/research route clears `>=5%` sustained decode.

Gate:

- sustained W==D `>=3%`, strong `>=5%`;
- dNLL within gate.

Decision:

| Finding | Decision |
|---|---|
| `>=5%` clean | scope native producer/compiler support |
| `3-5%` clean | keep research flag, do not default |
| `<3%` | close q8 as insufficient for base decode |

## Stage 5 - Scheduler / Resource Preservation

Run if:

- Q4/Q6 role identity is correct;
- reduce/glue does not clear Amdahl;
- q8 does not clear Amdahl;
- in-model bucket still sits near `44%`.

Goal: decide whether to fund a project-level AMD MMVQ scheduler/resource effort.

Evidence required:

- ATT/HCQ role joins show intended programs;
- role traces show lower wave coverage/resource behavior than standalone or llama-like contract;
- no bounded wiring/reduce/q8 route remains.

Project shape:

- renderer/scheduler changes for low-VGPR, high-grid MMVQ contracts;
- possibly imported mature MMVQ artifact family for Q4/Q6 as oracle;
- graph/runtime scheduling audit if traces show gaps around role intervals.

Gate to start:

- modeled `44% -> 54%` bucket recovery remains plausible;
- no smaller stage can deliver `>=5%` W==D;
- implementation owner accepts project-level scope.

## Stage 6 - Combined Retention Experiment

Only after one or more previous stages pass.

Work:

- combine passing byte-identical pieces behind one research flag;
- W==D ctx sweep: `128, 512, 1024, 4096`;
- PMU/ATT spot-check to confirm mechanism moved;
- dNLL only if q8/lossy route included.

Success:

- minimum: sustained decode `>=5%`;
- strong: `>=10%`;
- bucket target: in-model weight-GEMV `>=54%` HBM.

## Expected Outcomes

| Outcome | Meaning |
|---|---|
| bounded route found | one of Q6 route identity, reduce fusion, or q8 reuse moves W==D |
| all bounded routes fail | `76%` retention requires project-level MMVQ scheduler/lifecycle work |
| project declined | prioritize spec-decode as the practical route to beat llama |

## Immediate Next Step

Execute Stage 1 for `blk.0.ffn_down`.

That is the highest-information next measurement: if high-share Q6_K has a wiring/identity issue, we get a bounded
fix. If it does not, the path toward `76%` shifts away from wiring and toward reduce/glue accounting or project-level
scheduler/resource work.

