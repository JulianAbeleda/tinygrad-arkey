# Resolution plan — score-broadcast decode lifecycle (handoff to next session)

To: the next Claude working this lane.
From: 2026-06-26 audit (see `docs/decode-score-broadcast-lifecycle-audit.md`).

## TL;DR

The score-broadcast candidate `decode_attention_physical_tile_score_broadcast_lifecycle` is
**non-promotable** — correct + capture-clean, but 18×–134× slower than the owned flash route, structural
(whole-cache, 6 unfused kernels, q.k recomputed per PV chunk, + a full K+V cache copy). **Do not promote
it and do not keep tuning it.** Three classes of work remain:

1. **Close it out cleanly** — record the refutation, mark the candidate refuted.
2. **Resolve the latent hazards that ship regardless** — the diagnosis confound, and the code hazards
   (H1 barrier no-op, H2 ungated core changes, H3 gate blindness) that live in the working tree and
   would ride along with any merge.
3. **Move to the real lever** — build `FusedScorePVLifecycle`, the missing primitive.

Do these in order. Steps 1–2 are cheap and protect correctness/honesty of the assets; step 3 is the
research.

## Why you should not reopen the route itself

W==D is decisive and over-determined (see audit). `ctx128≈parity` is irrelevant — decode authority is
ctx≥512, where it is 18×–134× slower and worsening linearly with ctx. The A2 whole-cache base already
loses (~74/73/69/63% of owned) before the broadcast decomposition adds the 4× q.k recompute. No tuning
closes this. Treat the route as a parked, correctness-clean diagnostic only.

## Step-by-step

| # | Action | Gate / done-when | Stop condition |
|---|---|---|---|
| 1 | Record the refutation in the candidate registry: set `decode_attention_physical_tile_score_broadcast_lifecycle` to refuted / non-promotable with `verdict=REST`, link this plan and the audit doc. Add the refutation text (audit doc) to wherever refutations are tracked. | `candidates.json` entry shows refuted + reason; the stale `FAIL_CORRECTNESS` run is annotated as superseded-by-perf, not authority. | — |
| 2 | **Confirm the `E_98304` identity** (H3). One-shot live capture of the score-broadcast route; inspect `E_98304_32_3`'s buffers/AST. Confirm it is the `cache_kv.reshape(2*Hkv*MAXC*Hd)` K+V materialization (vs a benign compute kernel). | A note in the audit doc: confirmed/!confirmed, with the kernel's input buffers. | If it is *not* a materialization, downgrade the H3 "not materialization-clean" claim accordingly (the detector-blindness gap still stands). |
| 3 | **Fix H3 (route-gate rigor).** Broaden `qk_decode_search_gate.py` materialization detection beyond the `"49152"` literal: flag any elementwise/copy kernel that is (a) present only in the candidate arm and (b) sized ≥ a full-MAXC tensor; and add an upper-bound check that `generated_attention_programs == 6` with no extra `flash_*`/large-`E_*` kernels. Wire `full_maxc_copy_kernels`/`buffer_identity_inputs` into the pass boolean. | Re-run route gate: it now FAILs on the current score-broadcast route (because `E_98304` is present), or PASSes only if the materialization is provably removed. | — |
| 4 | **Fix H1 (barrier no-op).** Make the barrier read fresh, not the cached `getenv`. Options, simplest first: (a) in `tinygrad/engine/jit.py` read `os.environ.get("JIT_NO_GRAPH_KERNEL_PREFIXES","")` directly instead of `getenv(...)`; or (b) clear `getenv`'s cache after install; or (c) set the env before first tinygrad import in the route's entrypoints. | Add a gate asserting the barrier actually took effect under a prefill→decode order (e.g. assert the `flash_pall_*` kernels are emitted as standalone PROGRAMs, not wrapped in a graphed `CUSTOM_FUNCTION(LINEAR(...))`). The gate passes only when the barrier is observed active. | If the barrier cannot be observed active even after the fix, the MMU "fix" is something else (see step 5) — record that. |
| 5 | **Resolve the diagnosis confound** (protect the asset before banking it). Run the missing controlled cell: barrier **off** @ `chunks=4` (with persistent scratch on). Then, if needed, scratch-off @ `chunks=4` with barrier on. | A 2×2 (barrier × chunks at fixed scratch) that isolates which intervention actually removes the MMU fault. Update mmu-scope: replace "the decisive fix is the local graph-batch barrier" with the controlled result. | If barrier-off @ chunks=4 also passes, the barrier is **not** the fix — the asset is `chunks=4` addressing or persistent scratch; relabel and keep only what is load-bearing. |
| 6 | **Decide H2 (ungated core changes).** Determine whether `schedule/__init__.py:133-139` (toposort BIND scan) and `codegen/__init__.py:146` (unconditional `pm_unbind`) are needed at all once the candidate is parked. Run the baseline default decode + a prefill regression with and without them. | Either: gate both behind the route (so the default path is byte-for-byte unchanged), or keep them only if a regression run on the shipped owned route + baseline W==D (82.3/103.2/101.3/94.0) is unchanged and the bind-mismatch `RuntimeError` cannot trigger on default workloads. | If baseline W==D or any default gate regresses, revert/gate them — they must not ship for a parked candidate. |
| 7 | **(Optional, low priority) Harden scratch (H4)** only if the route is kept runnable: assert batch-1/single-stream at route build, and assert (or structurally guarantee) the state-kernel and PV-kernel `m` recurrences stay bit-identical. | A guard that raises rather than silently producing wrong attention if the invariant is violated. | Skip if the route is fully parked and never run outside the gate. |
| 8 | **Forward lever: build `FusedScorePVLifecycle`.** A single q.k pass feeding all 128 PV columns inside a flash/online-softmax tile with an efficient split-KV combine. Restore the flash IO-aware structure (not whole-cache); preserve T=1 KV-split parallelism; report split-KV economics (tile_us / combine_us / combine_fraction / occupancy) **before** W==D. | A generated candidate that is flash-structured, route-clean (with the step-3 stricter gate), and clears W==D vs baseline. | Per corpus: do not start an unfused/whole-cache variant; do not promote on route-gate PASS alone — W==D is authority. |

## Sequencing notes

- Steps 1–4 are independent and can be done in any order; do them first — they are the honesty/hygiene
  fixes and they make the route gate trustworthy for the next candidate.
- Step 5 gates whether "graph-batch barrier" goes into the corpus as a proven asset. Cheap; do it before
  any doc claims the barrier is the fix.
- Step 6 is the only thing that touches the **shipped default path** — treat as the highest-risk item to
  leave unresolved if this branch is ever merged.
- Step 8 is the actual research and should reuse the (now-trustworthy) `CapturePhaseGate` and the
  (now-fixed) barrier.

## Acceptance for "issue resolved"

- Candidate registry records the refutation; no path treats route-gate PASS as a promotion signal.
- Route gate fails-closed on the `E_98304` materialization (or it is proven removed).
- The barrier is either observably active (H1 fixed + asserted) or honestly relabeled as not the fix.
- The default decode path is byte-for-byte unchanged unless a baseline-W==D regression run proves the
  core changes inert.
- The next candidate in this lane is flash-structured (`FusedScorePVLifecycle`), not whole-cache.
