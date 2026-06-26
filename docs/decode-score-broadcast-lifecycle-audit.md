# Decode score-broadcast lifecycle audit (2026-06-26)

Candidate: `decode_attention_physical_tile_score_broadcast_lifecycle`
Activation: `DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE=1 DECODE_ATTN_SCORE_BROADCAST_CHUNKS=4 V_DOT2_LOWERING=1`
Repo state at audit: HEAD `65150e89f`, dirty tree.

Goal of the audit: decide whether the diagnosis, the lifecycle fixes, and the next decision for the
score-broadcast decode-attention route are technically sound, and whether the candidate is promotable.

Method: six-dimension audit (diagnosis causality, JIT-change safety, scratch/kernel safety, route-gate
rigor, W==D / harness contract, corpus + search alignment), each read from primary sources and then
adversarially re-checked. Key numbers re-derived directly from the on-disk artifacts.

## Verdict table

| Area | Verdict | Basis |
|---|---|---|
| Route correctness | PASS (shallow) | `score_broadcast_route_latest.json` @14:03 `SCORE_BROADCAST_ROUTE_CLEAN__WD_NEXT`: tokens `== [315,24231,6009,979,220,576]` baseline, `has_owned=0`, state + 4 PV chunks + combine present, fail-closed (`route_gate.py:65-71,92`). Only 6 greedy tokens; no full-model dNLL yet. |
| TinyJit capture/replay | PASS (liveness-only) | `score_broadcast_jit_phase_latest.json` all 8 modes pass. But these gates assert *no MMU crash*, not numeric correctness (`jit_phase_gate.py:144-145`), and they do not verify the graph-batch barrier actually took effect. |
| Materialization safety | CLAIM OVERSTATED | Route is **not** materialization-free: `E_98304_32_3` (= 2×49152, combined K+V full-MAXC copy) fires every step, present only in the score-broadcast arm (39 programs vs 34 baseline), inserted right before the state kernel. The gate's purity check is a hardcoded `"49152"` literal and is blind to it. |
| W==D performance | FAIL — decisively non-promotable | `runtime-overhead/result.json`: 82.5 / 5.8 / 3.1 / 0.7 tok/s @ ctx 128/512/1024/4096 vs baseline 82.3 / 103.2 / 101.3 / 94.0 → ~1.0× / 18× / 33× / **134× slower**, monotonic in ctx, GPU-bound (median host-sync 1.4%). Structural, not tuning. |
| Promotion eligibility | DO NOT PROMOTE | Failed shipping gate; correctly default-off / ineligible. |
| Search / purity implication | `SEARCH_SPACE_INCOMPLETE` | The winning structure (`FusedScorePVLifecycle`) is "Not implemented" (mmu-scope primitive map). Not a true wall; **not** `SEARCH_BLOCKED_BY_RUNTIME` (runtime is solved and the route is GPU-bound). |

## What the handoff got right

- The route gate genuinely passes after the chunks-default fix. The earlier eval run
  `20260626T133814 … FAIL_CORRECTNESS / SCORE_BROADCAST_ROUTE_FAIL` was the **pre-fix** run (the gate
  cleared `DECODE_ATTN_SCORE_BROADCAST_CHUNKS` to 0); it is stale.
- TinyJit capture/replay is clean across all 8 phases.
- The reduced-chunk path cannot be gamed into a pass: the route raises `RuntimeError` for `chunks<4`
  unless `DECODE_ATTN_SCORE_BROADCAST_DIAGNOSTIC_CHUNKS` is set, and the gate requires
  `not diagnostic_chunks` (route purity is triple-guarded).

## What the handoff got wrong or overstated

1. **"Hidden materialization E_49152 — absent" is misleading.** `E_49152` specifically is absent, but a
   different full-cache copy `E_98304_32_3` fires every token and the gate cannot see it
   (`qk_decode_search_gate.py:36` matches only `"4915"/"49152"`). The route is correct + owned-free, not
   materialization-clean. (Identity of `E_98304` as the K+V copy is inferred from size = 2×49152, its
   position before the state kernel, and the route's `cache_kv.reshape(2*Hkv*MAXC*Hd)`; confirm by live
   capture. The rigor gap — the detector cannot see a non-49152 materialization — is certain regardless.)

2. **"Graph batching was the MMU root cause and the barrier is the decisive fix" is unproven.** See
   diagnosis assessment below. mmu-scope's "the decisive fix is the local graph-batch barrier" should be
   downgraded to "plausible but uncontrolled".

3. **W==D was still in flight, and even the completed artifact is refutation-grade, not
   promotion-grade.** `result.json` is single-arm, stale (11:06, pre-fix), env/git/candidate-id
   unstamped; it has no baseline arm, no repro band, no dNLL. Adequate to reject a 134× regression;
   insufficient to promote anything. The contract-complete evaluator `wd` rung never ran
   (`wd.checked=false`; route-gate FAIL broke out before it).

## Diagnosis assessment (is the MMU diagnosis sound?)

Partially. The phase isolation is good: it correctly excludes eager route math (`eager_x2` passes
everywhere), excludes "needs many blocks" (a *single* block faults at depth 1), and localizes the fault
to the TinyJit **capture** phase.

The specific attribution — "MMU fault = HCQ graph batching, fixed by the `JIT_NO_GRAPH_KERNEL_PREFIXES`
barrier" — is **confounded and unproven**:

- Every FAIL artifact ran `chunks=1`; the only PASS ran `chunks=4`. There is no barrier-off @ `chunks=4`
  control, so the flip cannot be attributed to the barrier alone.
- The PASS run added the barrier **and** persistent scratch together; no artifact separates them.
- The MMU fault is async (surfaces in `synchronize()/_free`, addr `0x7FCA00000000`, `NotPresent=1
  ReadOnly=1`) and does not name a faulting kernel, so a latent out-of-bounds write that the barrier
  *masks* via changed buffer lifetime is equally consistent with the evidence.
- The block-depth "FAIL_BOUNDARY_FOUND" overstates a bisection that did not happen: depth-0 has zero
  route kernels (vacuous pass) and `build()` breaks on first fail, so depths > 1 never ran.

Net: the route is crash-free and (shallowly) numerically correct under full-model capture — that
*outcome* is artifact-backed. The *mechanism* is not. This does not affect the promotion verdict (the
route is dead on perf regardless), but it matters for the reusable asset: do not bank "barrier = MMU
fix" until the controlled experiment confirms it, especially given hazard H1.

## Code hazards (file-path grounded)

**H1 — the graph-batch barrier can be a silent no-op (process-order-dependent).**
`tinygrad/engine/jit.py:32` reads `JIT_NO_GRAPH_KERNEL_PREFIXES` through tinygrad's `@functools.cache`
`getenv` (`tinygrad/helpers.py:165-166`). `extra/qk_flash_decode.py` installs the prefixes by mutating
*raw* `os.environ` at route-build time (`_install_score_broadcast_no_graph_prefixes`). If any kernel
reaches `graph_split_rewrite` (and thus the cached `getenv("JIT_NO_GRAPH_KERNEL_PREFIXES","")`) before
the install runs — e.g. a prefill TinyJit captured first — `getenv` memoizes `""` and the barrier never
applies for decode. No gate verifies the barrier took effect (they count PROGRAM ops, which appear in
toposort whether graphed or not). The Python caching fact is reproduced; the prefill-first ordering in
the real harness is inferred, not yet observed.

**H2 — two UNGATED core changes ride along to the default path.**
`tinygrad/schedule/__init__.py:133-139` switches BIND collection from `big_sink.src[1:]` to
`big_sink.toposort()` (a global superset scan that can newly trip the bind-mismatch `RuntimeError`);
`tinygrad/codegen/__init__.py:146` runs an unconditional `pm_unbind` graph_rewrite on every kernel
sink. Both look defensive (guards present, no-op when no BIND survives) but, unlike the env-gated
barrier, they alter behavior for **every model in the process**, not just this opt-in route.

**H3 — route-gate materialization check is too narrow.**
`extra/qk_decode_search_gate.py:36` is `copies = [n for n in names if "4915" in n or "49152" in n]`.
`full_maxc_copy_kernels` / `buffer_identity_inputs` are captured into the artifact but never read by the
pass boolean (`route_gate.py:68-71`), and there is no upper-bound/count check on the generated-program
set. So `E_98304_32_3` and any future non-49152-sized materialization slip through. The "clean" verdict
means *token-equal + owned-free + E_49152-absent*, not *materialization-free*.

**H4 — scratch is safe only by an unenforced invariant.**
`_SCORE_BROADCAST_SCRATCH` is keyed only by shape `(device,Hq,Hd,Hkv,MAXC,L,Smax)` — no
layer/stream/graph identity. All transformer layers share one set of 6 buffers; cross-layer clobber is
prevented *only* by the incidental residual-stream data dependency, with no batch-1 assertion. The
combine's correctness also depends on the state kernel's per-split `m` and each PV chunk's internal `m`
being bit-identical (`kernels:57-58` vs `:109-110`); any future lowering divergence yields a silent
`exp(m_state − m_pv)` factor that does not cancel. Safe for the current single-stream decode; fragile.

**H5 — env install never torn down.** `_install_score_broadcast_no_graph_prefixes` mutates `os.environ`
process-wide and never unsets it, plus a second global `os.environ.setdefault("V_DOT2_LOWERING","1")`
inside the decode helper. Behaviorally contained today (prefixes are route-specific kernel names), but
latent cross-test leakage.

## Why it is non-promotable (W==D)

The slowdown is structural, not tuning:

- The route is `DECODE_ATTN_GENERATED_WHOLECACHE` (non-flash): each of its 6 kernels scans the whole KV
  cache, O(ctx) per token, versus the shipped flash split-KV route (online softmax) at 2 attention
  kernels.
- It recomputes q.k once per PV chunk (4 chunks) plus the state pass — ~5 q.k passes vs 1.
- It additionally materializes the combined K+V cache each step (`E_98304`).
- `D ≈ W` at every ctx (host-sync ≤ 4.8%, median 1.4%) and `debug2_unbatched_gpu_ms ≈ W`, so the cost is
  genuine GPU work, not host/launch overhead.
- The underlying A2 whole-cache skeleton already regresses to ~74/73/69/63% of owned at ctx
  512/1024/2048/4096 *before* the broadcast decomposition adds the 4× q.k recompute.

No kernel tuning closes a 134× algorithmic deficit. A post-fix evaluator run would classify `REST`
(no rung cleared a promotion gate), never `PASS_PROMOTE`.

## Search / purity label

`SEARCH_SPACE_INCOMPLETE` (equivalently `SEARCH_BLOCKED_BY_CODEGEN`). The runtime block (TinyJit capture
MMU fault) was real and is now addressed; the route is GPU-bound, so `SEARCH_BLOCKED_BY_RUNTIME` would
mislabel it. The structure that would make a generated whole-cache attention competitive —
`FusedScorePVLifecycle`: a single q.k pass feeding all 128 PV columns inside a flash/online-softmax tile
with an efficient split-KV combine — is not yet emittable. The 4-chunk broadcast route is hand-authored
in `extra/qk_flash_decode.py` (fixed `flash_pall_score_*` prefixes), not a point sampled from the
exposed primitive vocabulary, so it is a manual experiment, not a search result. Its positive
contribution to the search program is the lifecycle assets, not the kernel route.

## Refutation (record as an asset)

> Unfused whole-cache score-broadcast decode route is correctness/capture-clean but W==D-refuted:
> GPU-bound, 6 attention kernels/token, recomputes q.k per PV chunk, plus an undetected `E_98304` full
> K+V cache materialization; regresses monotonically ~1.0×@128 → ~0.0074×@4096 vs owned flash.
> Abandoning flash for whole-cache + per-chunk recompute violates IO-awareness and split-KV economics.

Do **not** reopen:
- reduced/diagnostic chunk counts 1/2/3 as correctness or W==D candidates (duplication-only);
- `NO_MEMORY_PLANNER=1` / global `JIT=2` as fixes (diagnostic-only, refuted);
- the incremental metadata-fusion micro-step lane (handoff already concluded simple incremental fusion
  loses parallelism/memory shape);
- the stale `FAIL_CORRECTNESS` eval (route-gate chunks=0 bug, now fixed) as authority.

Reusable assets that survive the refutation: the `JIT_NO_GRAPH_KERNEL_PREFIXES` graph-batch barrier
(generic, env-driven, PROGRAM-op gated) — pending the H1 fix and the controlled barrier experiment — and
the `CapturePhaseGate` (`qk_decode_score_broadcast_jit_phase_gate.py`).

Resolution plan: see `docs/decode-score-broadcast-lifecycle-resolution-plan.md`.
