# Decode Native Renderer DNR-3C9 New Information Ledger - 2026-06-20

## Verdict

`SCOPE_DNR3C9_NEW_INFORMATION_EXHAUSTED_NATIVE_PARKED`

DNR-3C9 exhausts what new information is required before reopening the native DNR-3C decode schedule path.
The current decision remains: park local native schedule rewrites until one reopen gate passes.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c9_new_info_ledger.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c9_new_info_ledger_result.json
```

## Closed Facts

These are no longer open blockers:

| fact | decision |
|---|---|
| native q8 decode correctness path is real | do not reopen Q4_K addressing, q8 addressing, scale/min extraction, dot4 selection, or gate/up correctness |
| static count matching is insufficient | do not add branch, wait, marker, load, or LDS-count patches without new attribution |
| descriptors do not show a simple scratch/private/LDS explanation | do not expect a one-line descriptor flag to explain the oracle gap |
| PMC capture is usable directionally | use PMC to confirm candidates, not as timing authority |
| first issue-order experiment is only a small local win | park local DNR-3C schedule rewrites unless new information names a larger lever |

## Missing Information

| id | missing information | tool needed | priority |
|---|---|---|---|
| NINFO-1 | oracle VGPR/SGPR/resource envelope | oracle code-object metadata extraction | P0 |
| NINFO-2 | oracle ISA and semantic schedule map | oracle disassembly annotated into semantic stages | P0 |
| NINFO-3 | SQTT/body timeline mapped to q8 PCs | repaired tinygrad SQTT or ROCprofiler/AQLprofile ATT PC join | P0 |
| NINFO-4 | true live-range pressure and allocator model | live-interval builder plus oracle metadata/disassembly | P1 |
| NINFO-5 | counter-to-time calibration | repeated PMC/timing calibration matrix | P1 |
| NINFO-6 | oracle runtime/launch integration differences | one-clock interleaved oracle/native/C7C harness with descriptor dump | P1 |
| NINFO-7 | new decode primitive route | route-level decode promotion audit | P2 |

## P0 Details

### NINFO-1: Oracle Resource Envelope

Question: does the hipcc/LLD oracle use a different VGPR/SGPR allocation, occupancy, or launch resource envelope?

We currently have only partial oracle metadata: local size, group segment, private segment, and kernarg size. Missing:

- oracle VGPR count;
- oracle SGPR count;
- oracle occupancy;
- oracle full kernel descriptor resource fields;
- oracle per-phase live range.

Minimum evidence:

- oracle allocated VGPR/workitem and SGPR count;
- oracle private/scratch bytes;
- oracle LDS/group segment bytes;
- oracle occupancy estimate;
- same fields for native, best static, and C7C-best.

Reopen native only if this names an implementable resource-envelope difference with credible `>=30us` upside.

### NINFO-2: Oracle ISA And Semantic Schedule

Question: what exact instruction order does the oracle use for load, unpack/select, dot4, scale/min, cross-wave
reduction, waits, and stores?

Missing:

- ordered oracle disassembly tied to semantic stages;
- register operand map for q4, q8, scales, and accumulators;
- waitcnt placement by dependency reason;
- branch/exec predicate purpose;
- instruction-level stage overlap model.

Minimum evidence:

- stage-labeled oracle ISA table;
- native/C7C equivalent stage table;
- one unimplemented mechanism that is not just count matching;
- dependency-safe construction path in native.

### NINFO-3: SQTT/Body Timeline

Question: where does native spend cycles inside the q8 kernel body, and does the oracle avoid those stalls?

Missing:

- PC-level native body timeline;
- PC-level oracle body timeline;
- stall reason per PC or stage;
- mapping from PCs to disassembled instructions;
- same-run alignment between timing and trace capture.

Minimum evidence:

- nonzero body packets for the q8 kernel;
- PC-to-ISA join for native and oracle, or native and C7C-best;
- stage-level stall histogram;
- one stall class with plausible `>=30us` movement.

## P1/P2 Details

NINFO-4 is needed if the resource story points to live ranges rather than descriptor fields. It must produce
per-instruction live intervals, peak VGPR/SGPR by semantic stage, and occupancy estimates for native, best static,
C7C-best, and oracle.

NINFO-5 is needed because PMC moved in the expected direction while timing did not clear gates. It must establish
whether SQ wait/busy deltas predict wall-time movement or are only directional noise.

NINFO-6 is needed to ensure the oracle target is fair: native, best static, C7C-best, and oracle must run in one
clock-controlled interleaved harness with descriptor and dispatch metadata.

NINFO-7 is the route-level escape hatch. If local native q8 MMVQ is the wrong primitive, the next evidence should be
route-level: q8 artifact reuse, fused decode primitive, quality policy, fallback behavior, and promotion threshold.

## Reopen Gates

| gate | requires | pass condition |
|---|---|---|
| resource reopen | NINFO-1, NINFO-4 | resource/live-range delta names an implementable native change with credible `>=30us` upside |
| timeline reopen | NINFO-2, NINFO-3 | PC/stage timeline names a dominant stall and a dependency-safe schedule transform |
| counter reopen | NINFO-5 | PMC family correlates with wall time and predicts a material candidate before implementation |
| fair-oracle reopen | NINFO-6 | same-run oracle comparison changes the target or exposes an implementable launch/code-object difference |
| route reopen | NINFO-7 | route policy accepts a small native win or requires native as part of a broader decode route |

## Do Not Do

- do not add more load-count-only, branch-count-only, wait-count-only, marker-count-only, or LDS-count-only patches;
- do not start BEAM/search from static shape similarity;
- do not promote DNR-3C native schedule from the current C7D result;
- do not treat PMC direction as timing authority without calibration;
- do not reopen Q4_K/q8 address/correctness as if they are still the blocker.

## Decision

Native DNR-3C is parked. Minimum to resume: one reopen gate must pass. Otherwise continue with route-level decode
work or oracle/tooling extraction, not another local native schedule rewrite.

No renderer defaults changed.
