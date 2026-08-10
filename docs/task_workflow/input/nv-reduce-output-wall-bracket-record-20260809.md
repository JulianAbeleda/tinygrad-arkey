# NV generic REDUCE_OUTPUT wall bracket record (NO-GO at the exact-logits gate)

Status: **NO-GO. The generic cooperative reduction-to-output primitive survives
NV render (no Xid 31 class fault) and reproduces the census fused-body shape
(54 `reduce_output_rmsnorm_1_4096` bodies in one decode token), but the
candidate arm's JIT-captured decode logits tap returns all-NaN, so the exact
full-logit gate fails closed. No census arm, no wall bracket, no tok/s claim,
no policy promotion.**

Scope: `docs/task_workflow/input/nv-reduce-output-wall-bracket-scope-20260809.md`.
Branch `nvidia-bringup-20260731`. Campaign harness
`extra/llm_research/decode/nv_reduce_output_primitive_ab.py`, run 2026-08-10 on
the RTX 5090 (sm_120, Qwen3-8B-Q4_K_M at fixed depth 512) under the shared GPU
bench lock with a fresh process per arm.

## Verdict

1. Phase 0 (NV render smoke) PASS: the candidate graph survives and the
   observed decode window contains exactly **54 fused
   `reduce_output_rmsnorm_1_4096` CALLs** (program_count 937), matching the
   committed census reference (108 admissions / 54 fused bodies). The prelude
   token is real (13876); the decode scalar is the known NV sentinel (151936),
   unchanged from the closed graph.
2. Phase 1 (exact full-logit gate) FAIL: the candidate child aborts at row 0
   with `finite=False non_finite_count=151936 sid=151936 argmax=0
   row_sha256=a6d89127557209cf`. Every one of the 151936 fp32 logits is NaN
   and the sampled-token scalar reads the sentinel. The control arm passes
   with finite logits and real sampled tokens. HARD STOP at Phase 1; census,
   bracket, and tok/s phases did not run.
3. Isolation matrix (fresh process per configuration): control (no flags),
   callify-only, and promo-only all return finite logits with real sampled
   tokens, and callify-only vs promo-only are byte-identical (same row SHA).
   The all-NaN appears only when the fused body is actually admitted into the
   captured graph (callify flags + reduce-output promotion together). The
   eager `JIT=0` path under the same candidate conditions is finite and
   correct.

## What this means

The fused primitive's numerics are exact: the CPU hermetic gate
(`docs/task_workflow/input/nv-generic-reduce-output-primitive-record-20260809.md`)
proved bitwise equality of the body in tiny shapes, and the eager path is
finite and correct on NV. The failure is in **NV JIT capture of the callify
precompiled-output redirect**: when the fused body is in the captured graph,
the decode logits output binding reads uninitialized memory (all-NaN) and the
scalar reads the sentinel. The wall bracket is therefore not authorized, and
the +495.330 us norms row is NOT booked.

## Evidence

- Campaign record (full JSON, smoke program list included):
  `docs/task_workflow/output/nv-reduce-output-wall-bracket-20260809.json`.
- Child artifacts: `/tmp/ro-ab-record.children/smoke-candidate.json`,
  `control-logits.json` (finite, tokens `[64461, 4710, ...]`).
- Isolation runs: `/tmp/diag_isolate.py` cases `callify-only`, `promo-only`,
  `candidate-sdpa`, `candidate-flash` (both JIT taps all-NaN under the
  candidate).

## Follow-up

The next phase is the callify precompiled-output redirect capture fix on NV:
make the captured decode graph bind the fused body's output to the logits
consumer (and the sampler scalar) instead of an uninitialized buffer, then
re-run this exact campaign. No policy promotion; the route policy stays
`promoted_targets: []`.
