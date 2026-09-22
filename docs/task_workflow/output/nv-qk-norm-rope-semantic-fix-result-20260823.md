# Q/K norm+RoPE semantic fusion review and fix (2026-08-23)

## Findings first

1. **[MEASURED] The opaque-admission `NO_GO_WALL` did not refute the fusion.**
   It measured `+36.131 us/token`, but its candidate graph added 73 launches.
   The report's causal attribution was not closed with per-row durations,
   device union, overlap, and host gap; launch counts alone support an
   inference, not a measured wall partition.
2. **[MEASURED] The original evidence package was incomplete.** It had no SHA
   manifest, did not retain GPU clock state in the wall arms, and its two
   census files were raw name-count maps rather than the self-describing
   schema emitted by the cited census script.
3. **[MEASURED] The semantic fix removes the opaque materializations.** The
   candidate uses the existing `REDUCE_OUTPUT` ownership/lowering path, binds
   the persistent `[MAXC,128]` frequency table directly, and indexes it with
   the existing `start_pos` graph variable. No frequency-slice copy is added.
4. **[MEASURED] The corrected graph has 560 nodes versus 596 control.** It
   replaces 36 Q norms and 36 K norms with 72 fused bodies and removes 36
   standalone Q-rope kernels. The opaque route's `E_4_2_8_16_4`,
   `E_8_32_4`, and additional `E_32_32_4` materializations are absent.
5. **[MEASURED] Device profile:** node sum changes `4618.896 -> 4506.368 us`
   (`-112.528 us`) and union changes `4614.625 -> 4502.000 us`
   (`-112.625 us`). Overlap changes `4.553 -> 4.147 us`.
6. **[MEASURED] Three independent unprofiled reverse wall brackets pass with
   identical frozen token SHA.** Deltas are `-46.568`, `-44.898`, and
   `-40.557 us/token`; the conservative confirmed recovery is
   **40.557 us/token**.
7. **[MEASURED] The candidate is below both controls in all three brackets.**
   The final authority retained P0 and observed the same 2790 MHz SM / 14001
   MHz memory clocks in every arm. Token
   stream SHA is `f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`.
8. **[INFERRED]** The old `96-101 us` wall ceiling was overstated because the
   installed graph contains only one standalone rope launch per layer. The
   semantic candidate removes 36 launches, not 72.
9. **[MEASURED] The principal explicitly accepted the result below the normal
   `+50 us/token` promotion bar.** The shipped policy now promotes only
   `NV/sm_120`; all other targets remain closed.
10. **[MEASURED] The production-default route is installed.** A fresh process
    loaded the model without the research override, observed the production
    policy on all 36 blocks, reproduced the frozen 32-token SHA, and measured
    `4708.321 us/token` at P0 / 2797 / 14001 MHz. This single arm verifies the
    installed route; it is not a fourth recovery measurement.
11. **[MEASURED] The production-default profile is the intended semantic
    graph.** It has 560 nodes, including 36
    `reduce_output_rmsnorm_rope_32_128` and 36
    `reduce_output_rmsnorm_rope_8_128` bodies. Its node sum is `4508.064 us`,
    union is `4505.125 us`, and overlap is `3.664 us`.
12. **[MEASURED] A subsequent composed reverse bracket supersedes the endpoint
    projection.** Both routes together recover `120.481 us/token`; the fresh
    production candidate is `4697.289 us/token` (`212.889 tok/s`) and leaves
    `530.622 us/token` to 240. The former `4677.993 us/token` value remains
    recorded only as a cross-session ledger projection.

## Implementation

The closed-default route extends `ReduceOutputSpec` with an `identity|rope`
epilogue, emits the bit-exact paired half rotation from the ordinary semantic
fallback, and admits only rows 8/32 by dimension 128 with a persistent fp32
frequency table. Unsupported shapes and ownership spellings fail closed.

The research wall harness toggles the semantic route instead of installing
the opaque `KernelProgram` monkey patch. Production is target-policy promoted
for `NV/sm_120`; the policy is closed for every other target.

## Validation

* `68 passed` across the route-policy, reduce-output, generic reduce-output,
  and decode graph-position-invariance tests.
* `py_compile` passed for every modified Python module and harness.
* `git diff --check` passed.
* Raw wall arms, graph profiles, production-default verification, summaries,
  and hashes are retained under
  `docs/task_workflow/evidence/nv-qk-norm-rope-semantic-fix-20260823/`.

## Verdict

`SEMANTIC_FUSION_ACCEPTED_PROMOTED_BOOKED`

The semantic fusion is enabled for NV sm_120 and its conservative measured
floor of `40.557 us/token` is booked by explicit principal acceptance. This
does not change the overall `240_UNMEASURED` verdict.
