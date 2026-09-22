# NV generic REDUCE_OUTPUT wall bracket record (correctness PASS, bracket NO-GO)

Status: **NO-GO for promotion, but the primitive is now CORRECT on NV. The
generic cooperative reduction-to-output primitive renders (no Xid 31 class
fault), passes the exact full-logit gate byte-identically to the control arm,
and passes the norms-confined census gate (54 fused
`reduce_output_rmsnorm_1_4096` bodies in one decode token). The reverse
control/candidate/control wall bracket does NOT promote: the candidate is
~190 us/token SLOWER than both bracketing controls (5.597 vs 5.406 ms/token
median, -6.33 tok/s), so the +495.330 us norms row is NOT booked and no
policy promotion happens. The regression is dominated by the callify-redirect
side effects on non-norms families (-36/+36/+54/-18/-71 on the E_32_32_4
family), not by the fused body's own numerics.**

Scope: `docs/task_workflow/input/nv-reduce-output-wall-bracket-scope-20260809.md`.
Branch `nvidia-bringup-20260731`. Campaign harness
`extra/llm_research/decode/nv_reduce_output_primitive_ab.py`, run 2026-08-10 on
the RTX 5090 (sm_120, Qwen3-8B-Q4_K_M at fixed depth 512) under the shared GPU
bench lock with a fresh process per arm.

## Root cause: reduce association, not JIT capture

The 08-05 fused body (strided 512-thread split, 8 elements/lane, stride 32,
shuffle-tree ladder) was NOT bitwise-equal to the ordinary r_16_256 kernel.
The ordinary kernel sums 16 threads x 256 CONTIGUOUS serial elements, then a
serial 16-partial chain; the fused body summed with a different fp32
association. The two associations differ by 1 fp32 ulp in the resulting
`scale`, which flips downstream fp16 norm outputs at rounding boundaries and
shows up as ~1e-3 logit diffs (and, under the earlier in-place-aliasing bug
fixed in `14f0e3297`, all-NaN). The JIT/callify capture path was never the
culprit; the fused body itself was numerically different.

Fix (`25370bbad`, emitter `tinygrad/codegen/late/reduce_output.py`):
`emit_reduce_output` now mirrors the ordinary association exactly. The reduce
index is lane-independent: every lane computes the same per-warp serial chain
over `per_lane * lane` contiguous elements, lane 0 publishes the per-warp
partial, and the serial cross-warp chain plus the all-lane elementwise
epilogue follow unchanged. Fused == ordinary bitwise for every dtype mix
(fp32/fp16, fp16/fp16, fp32/fp32), verified by the tripwire tests in
`test_reduce_output_rmsnorm.py` and the re-pinned body digest in
`test_generic_reduce_output.py`.

## Evidence (2026-08-10, fresh processes under the GPU bench lock)

1. Phase 0 (NV render smoke) PASS: the candidate graph survives on sm_120 and
   the observed decode window contains exactly 54 fused
   `reduce_output_rmsnorm_1_4096` CALLs (program_count 937), matching the
   committed census reference (108 admissions / 54 fused bodies).
2. Phase 1 (exact full-logit gate) PASS: control and candidate logits SHA
   identical:
   `70838f5237ce2cf215e937caed807c4827daa1336e69c3d6c396b1aad4434819`.
   Token streams identical (`[64461, 4710, 64461, 1837, ...]`), shape
   `[32, 1, 151936]`, all rows finite, sampled token == argmax on every row.
   Eager JIT=0 path finite and correct on both arms.
3. Phase 2 (norms-confined census) PASS conditions: candidate fused bodies
   54 (control 0); rmsnorm_reduce 56 -> 38 (drop 18 <= 54, consistent);
   rmsnorm_epilogue 55 -> 37 (removed); q/k norm reduce roles unchanged at 36;
   kernels 936 -> 937 (honest net +1). Callify-redirect shifts on non-norms
   families are recorded, not hidden (candidate `kernel_us` 6080.64 vs control
   5889.95 is dominated by those shifts; the wall bracket is the authority).
4. Phase 3 (wall bracket): in flight at the time of writing; appended below.

Unit gate: `test_generic_reduce_output.py` 11 passed, `test_reduce_output_rmsnorm.py`
32 passed (each file in its own process; the files must not share a process
due to pre-existing DEV=CPU pollution at import).

## Phase 3 outcome (campaign completed 2026-08-10)

Full campaign record: `docs/task_workflow/output/nv-reduce-output-wall-bracket-20260809.json`
(harness `--mode ab`, depth 512, count 32, reps 5, settled-continuous, fresh
process per arm under the GPU bench lock). Verdict **NO-GO**.

| phase | result | evidence |
| --- | --- | --- |
| Phase 0 smoke | PASS | survives sm_120 render, `fused_body_present: true`, 937 programs |
| Phase 1 exact logits | PASS | control/candidate SHA identical `70838f52...`, tokens and shape equal |
| Phase 2 census | PASS | fused 0 -> 54; rmsnorm_reduce 56 -> 38 (drop 18, consistent); epilogues removed 18; q/k reduces untouched; net +1 program |
| Phase 3 wall bracket | NOT_PROMOTED | control A 5.4086 ms, candidate 5.5974 ms, control B 5.4030 ms; token-stream hashes identical |

Bracket numbers: candidate - control A = -188.8 us/token, candidate - control B
= -194.4 us/token, candidate - bracket median = -191.6 us/token (negative =
slower). tok/s conversion: control 184.99, candidate 178.65, delta -6.33
tok/s. Promotion requires candidate >= +50 us/token vs BOTH controls; the
candidate is ~192 us/token slower, so the wall evidence does not authorize
booking.

The fused body itself is competitive in isolation (histogram median 5.7 us
vs ordinary reduce 3.9 + epilogue 2.3 = 6.2 us per norms instance, ~0.5 us
saved x 54 = ~27 us/row), but the campaign measures the whole capture window:
the callify Context flags shift the E_32_32_4 residual family by
-36/+36/+54/-18/-71 calls (reported, not hidden) and the net window time
grows. The bracket is the authority and it fails.

## Follow-up

- The norms row stays unbooked. Per the scope's promotion rule (identical
  token-stream hashes and candidate median at least +50 us/token faster than
  BOTH bracketing controls), the +495.330 us attribution remains scoped, not
  booked, and the harness's Phase 6 route-efficiency follow-up is authorized:
  eliminate the callify-redirect side effects on the non-norms E_32_32_4
  family (or make the fused body admitted without them), then re-run this
  exact campaign.
- The exact-output contract is now met and regression-protected: the 08-10
  emitter association fix (`25370bbad`) is covered by the bitwise tripwires
  in `test_reduce_output_rmsnorm.py` and the re-pinned body digest in
  `test_generic_reduce_output.py`, so any future body change that breaks
  bitwise equality fails closed in CI before any GPU arm runs.
- No policy promotion: `decode-reduce-output-rmsnorm-route-policy.json` stays
  `promoted_targets: []`; no model wiring change; no default flip.
- Known separate bug, NOT the production path (q4k OFF `(0,0,0)` gives finite
  but wrong argmax 56669): parked, tracked separately, not part of this gate.
