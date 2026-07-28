# attn_qo C8 transition fault: lifecycle root cause + kernel exoneration (2026-07-20)

## Summary

The `attn_qo` staged candidate's `BLOCKED_AT_C8` / `DISQUALIFIED` classification (handoff §1.6) is a
**misattribution**. A guarded GPU experiment proves the C8 transition fault is a **harness lifecycle bug** — lazy
first-time construction of the candidate runtime + five buffers on the shared `Device["AMD"]` singleton *after* a
`direct_packed` route — and **not a property of the `attn_qo` kernel**. The fix is the pattern method §5.4 / commit
`8269edefe` already established for `attn_kv`: preconstruct the candidate as a separate lifecycle phase before any
route runs.

## Experiment

Ran the existing C8 transition harness (`run_guarded_persistent_c8_route_sequence` /
`run_persistent_c8_route_sequence_worker` in `extra/qk/mmq_frozen_staged_c8_sessions.py`) with the retained
`attn_qo` staged family + composition (this artifact directory), reusing only existing wiring
(`attn_qo_c8_runner_factory`, `build_attn_qo_direct_packed_objects`). No repo files were edited. To satisfy the
`validate_live_software` commit pin (`f0d7a09ce`), the driver ran from a detached worktree at that exact commit
(reproducing the original decision's software state); stale direct-packed qualification JSONs were regenerated with
the existing `qualify_one_queue` entrypoint (pure toolchain drift, unrelated to `attn_qo`).

| Run | Route sequence | Result | Health before/after | Kernel faults |
|---|---|---|---|---|
| 1 | `[staged_candidate]` | **PASS** | true / true | none |
| 2 | `[direct_packed, staged_candidate]` | **FAULT** | true / true | SQ type-2 (sh0+sh1, wave 3), gfxhub page fault, MES unresponsive → GPU reset → recovered |
| 3 | `[staged_candidate, direct_packed, staged_candidate]` | **PASS** | true / true | none |

Run 2 reproduces the original documented signature exactly (candidate faults on invocation 0, position 1).

**Run 3 reuse verification (the crux):** position 0 and position 2 breadcrumbs show `initialization_count: 1` at
both, `invocation_count` incrementing 1→2 (not reset), and identical `runtime_identity` (`object_id`,
`program_key`, `library_va`, `entry_va`) and `buffer_ranges`. The candidate runtime/buffers were constructed once at
position 0 (clean device) and **reused unmodified** at position 2 — dispatched immediately after a `direct_packed`
invocation — and passed.

## Conclusion

- **Construction-after-direct** faults (run 2). **Dispatch-after-direct** is safe when the candidate was
  preconstructed clean (run 3). The only differing variable is whether the candidate's first-time
  runtime/buffer construction happens on a clean device or one already loaded by the direct route.
- The `attn_qo` kernel is exonerated: it passes standalone (run 1) and across the route boundary when constructed
  correctly (run 3). Every §1.6 static exoneration (identical code bytes, ISA def/use clean, C3 addresses in-bounds)
  already showed the kernel is fine; this experiment closes the one variable those checks did not cover.
- Root cause is the anti-pattern method §5.4 forbids. Fix = preconstruct the candidate runtime/buffers before any
  route in the C8 session harness.

## Implications

1. Lift the `attn_qo` candidate `DISQUALIFIED` disposition; re-run C8 with the preconstruction fix to obtain a real
   timing result (`CERTIFIED_WIN` / `CERTIFIED_FALLBACK`). Note: re-running through the fixed harness requires
   re-certifying C6+C7 at the new commit, because `software_identity` pins the clean commit
   (`mmq_staged_c7_authority.py:251-256`).
2. The same harness bug would mis-`DISQUALIFY` any role at C8 (including `ffn_gate_up` later); the fix clears it for
   the whole ladder. This is distinct from `ffn_gate_up`'s C5 fault, which reproduces standalone.
3. The transition harness is wired per-role (`build_direct_packed_objects` allowlists `attn_qo`/`ffn_gate_up`,
   `mmq_attn_qo_c8_runtime.py:75`), which is why no known-good cross-role control was run before disqualifying.
   §12.9 is amended to require ruling out this lifecycle pattern (or a preconstruction-correct harness) before a
   cross-route fault may disqualify a candidate.
