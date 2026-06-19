# CG-W3 patch plan — vectorize the prefill fp16 global→LDS copy (tinygrad-core, dependency-free)

Concrete plan for the last dependency-free prefill lever, for a dedicated session. Target: the warmstarted in-model
WMMA matmul's hot loop (`r_8_64…768`) spends **127 `v_mov` + 16 per-element `global_load_d16`** marshaling the fp16
global→LDS copy (CG-W1.5); every kernel-level fix is refuted (CG-W2/W2b: no UOp-structure change vectorizes it).
Goal: make the copy's **global read** vectorize to `b128`, killing the d16 + `v_mov` overhead → toward Tensile's 66
TFLOPS, no dependency, all-AMD-matmul general.

## Mechanism (grounded in the codegen)
Vectorized loads in tinygrad require: **UPCAST (expand) a unit-stride axis** (`codegen/opt/heuristic.py`,
`Opt(OptOps.UPCAST, axis, n)`) → that expands the access into a group of adjacent `Ops.INDEX`/`LOAD` →
**`fold_expanded_index` (`codegen/late/devectorizer.py:81`)** merges the contiguous group into one
`buf.ptrdtype...vec(len(grp))` load → cstyle renders a `b128`. No UPCAST on the contiguous axis ⇒ scalar `half`
loads ⇒ clang `d16_b16`/`d16_hi_b16` + per-register `v_mov` init.

**Why the copy isn't vectorized:** the matmul UPCASTs the **WMMA/compute** axes (M/N tile). The global→LDS copy's
read is **strided relative to those** (it gathers `a[k_tile]` then transposes into LDS), so the copy's
**global-contiguous axis is not the UPCAST'd one** → its loads stay scalar `d16`. Tensile instead reads global
**contiguous-wide (b128)** and does the transpose in the **LDS write** (strided `ds_store offset:`, cheap).

## Two routes (the patch)
### Route A — copy-stage UPCAST heuristic (direct, hand-coded)
In `hand_coded_optimizations` (heuristic.py), when a kernel has a global→LDS *transposing copy* feeding a WMMA, add
an opt that **UPCASTs the global-read-contiguous axis of the copy load** (not just the compute axes), so the read
expands→folds to `b128`; accept the strided LDS store (LDS offset immediates are cheap). Concretely: detect the copy
load's unit-stride global axis and `apply_opt(Opt(OptOps.UPCAST, that_axis, 8))` for the copy sub-graph.
- Hard part: the copy and compute **share the kernel's axis structure** — UPCASTing the copy axis must not break the
  WMMA tiling. May require a **two-stage copy** (a separate wide global→reg load stage + a transposed reg→LDS store
  stage) so each stage's contiguous axis vectorizes independently. That is a scheduler/lowering change, not a
  one-line opt.

### Route B — REFUTED (2026-06-19): BEAM is not in the prefill path and doesn't help
CORRECTION: the prefill warmstart is **NOT BEAM-found** — it is a **hand-coded, loop-found fixed 3-opt tuple**
(`model.py:27`: `(Opt(TC,0,(-1,2,1)), Opt(UPCAST,0,4|2), Opt(UPCAST,1,4))`) applied per shape-key. **We do not run
BEAM for prefill, or anywhere at runtime.** The Route-B spike (`beam-hang-premise-audit-20260619.md`) found (a) BEAM
does *not* hang on these shapes (the "BEAM-hang wall" was a never-verified, false assumption), and (b) BEAM
*underperforms* — 14-17 TFLOPS vs the hand-coded warmstart's ~48 (and Tensile's 66). So "fix BEAM to find the
schedule" is moot: BEAM is worse than the tuple we already hand-pick. **Route B is closed.**

Implication for the copy-vectorization lever: it is about **extending/altering the hand-coded opt tuple** (can an
added copy-axis vectorizing opt make the global read fold to b128?) OR the **two-stage copy** (Route A) — NOT about
BEAM. The cheap first test: add a copy-axis `UPCAST` to `_prefill_v2_opts` and check the ISA. Caveat (CG-W1.5): the
existing TC+2×UPCAST tuple already applies and still emits 127 `v_mov` + d16 — so a contiguous copy axis to UPCAST
may not exist (the read is strided by the transpose), pushing this to the two-stage-copy scheduler change.

## Files / surfaces
- `codegen/opt/heuristic.py` — `hand_coded_optimizations` (UPCAST axis selection; Route A).
- `codegen/late/devectorizer.py` — `fold_expanded_index` (the load-merge; verify it fires once the copy axis is
  UPCAST'd; it merges contiguous groups, so the read must be contiguous post-UPCAST).
- the warmstart table generator (`extra/qk_prefill_gate.py` / `_WARMSTART_OPTS`) — re-derive per-shape opts with the
  copy axis vectorized (Route A or B).
- `renderer/cstyle.py` — only if the b128 load needs a render tweak (likely already handled via `float4`).
- the BEAM search + the gfx1100 hang (Route B) — `codegen/opt/search.py` + the timeout/compile path.

## Test matrix (gates, in order)
1. **ISA (noise-free, FIRST):** the copy load is `global_load_b128` (not `d16`); loop `v_mov` ≪ 120; mse < 1e-6.
2. **isolated matmul TFLOPS — FAIR back-to-back** vs the strided baseline (NEVER single-run; the CG-W2 clock-ramp
   near-miss: 42/65/67 for the same kernel). Gate ≥62 TFLOPS on ffn_gate/up; then ffn_down + attn_q/o.
3. **in-model** (no flag — it's a general codegen improvement): warm pp512 + pp1024 vs PREFILL_V2 and vs llama
   (3394), dNLL ≤ 0.01.
4. **NO decode regression:** decode W==D ctx-sweep unchanged (shared-codegen change — the true ship constraint).
5. **full test suite:** `test/test_ops.py`, `test/test_linearizer*.py`, `test/test_schedule.py`,
   `test/test_uops.py` green.
6. **no dependency, no BEAM at runtime** (Route A); Route B re-derives the warmstart offline.

## Risks (honest)
- **Broad blast radius:** the UPCAST heuristic / devectorizer feeds *all* AMD codegen — a copy-vectorization change
  can regress other kernels (incl. decode). Gate 4 is the constraint.
- **Warp-uncoalescing trade:** reading global contiguous-per-thread (for vectorization) is uncoalesced across the
  warp; CG-W2 showed the naive version was *slower*. The win requires wide loads to beat the coalescing loss — only
  validated by gate 2 (fair). Plausible the trade is net-zero on gfx1100 (IC-served), in which case **KILL**.
- **Route A scheduler complexity:** the two-stage copy is a real scheduler/lowering change, not a heuristic tweak.
- **Route B:** the gfx1100 BEAM hang may be deep (driver/compile), not a quick fix.

## Recommendation / sequencing
1. **First, a 1-day spike on Route B's prerequisite:** diagnose *why* BEAM hangs on gfx1100 (compile timeout? a
   specific opt? driver?). If it's a bounded fix, Route B is the highest-leverage, most-general path (unlocks search
   for this and other schedules).
2. **If BEAM is intractable, Route A:** prototype the two-stage copy in the `amd_copy` kernel (wide global→reg →
   transposed reg→LDS), gate on ISA (b128) + fair TFLOPS ≥62 *before* touching the shared heuristic. If the
   isolated kernel can't beat the strided baseline fairly (the uncoalescing trade), **KILL** — the pure-tinygrad
   prefill lever is closed and the route rests at PREFILL_V2 / the Tensile dependency.

## Against the principles
- *audit before build*: gate the shared-codegen change behind an isolated `amd_copy` two-stage-copy proof (ISA + fair
  TFLOPS) — do not touch `heuristic.py`/`devectorizer.py` until the isolated kernel clears ≥62.
- *measurement confounds*: ISA first, fair back-to-back TFLOPS only (the 42/65/67 lesson).
- *contain dangerous power*: shared codegen → full suite + decode-W==D gate (4,5) are mandatory.
- *label state*: OPEN/project-level; the last pure-tinygrad prefill lever; KILL-able if the uncoalescing trade is net-zero.

## Deliverables (for the dedicated session)
Route-B spike: BEAM-hang root-cause note + (if fixable) a re-derived warmstart with vectorized copy. Route-A:
`extra/qk_wmma_twostage_copy.py` (isolated proof, ISA-led), then the gated `heuristic.py`/`devectorizer.py` patch +
the full test matrix. Result doc `prefill-cgw3-result-…md` with the KILL-or-ship verdict.
