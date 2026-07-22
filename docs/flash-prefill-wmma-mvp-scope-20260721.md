# Scope (for deepseek): Fused-Flash-Prefill-with-WMMA — MVP proof-of-theory

**Author:** Claude (Opus 4.8) · **Date:** 2026-07-21 · **Executor:** deepseek · **Reviewer after MVP:** Claude
**Parent doc:** `docs/flash-prefill-scope-20260721.md` (the reuse-decode-kernel path is NO-GO; this is the follow-on it points to)
**Framing doc:** `docs/8b-vs-14b-prefill-regression-20260721.md` (roofline framing, two-ceiling model)

---

## 0. Read this first — what this task is and is NOT

This is an **MVP proof-of-theory**, not a build. You are proving *one* claim cheaply, then **stopping at a GO/NO-GO gate** so Claude can review before any real build is funded.

**The claim to prove (in roofline terms, NOT "beat llama"):**
> A fused online-softmax attention kernel whose `QKᵀ` (score) and `P·V` (context) matmuls run on **WMMA tensor cores**, keeping the `T×KV` score tile in **LDS/registers** (never spilled to HBM), moves prefill attention **off the memory roofline and onto the compute roofline** — and this improvement is **weight-format-independent** (helps both the 8B fp16-overlay path and the 14B packed-WMMA path, because attention operates on fp16 activations in both).

**Why we believe it (from the parent doc):** materialized SDPA already hits WMMA but spills the score matrix to HBM (~215 GB of score traffic at 14B/pp4096); the decode-kernel-reuse flash path fuses but scores on scalar `fdot2` (never WMMA) → capped ~1.5–2.5× slower than SDPA. Neither is fused-**and**-WMMA. This MVP builds the minimal kernel that is both.

**Do NOT:** build routing integration, a geometry sweep, GQA variants, quant-KV, decode support, or the autotuner. Those are the *full build*, scoped **only after** this MVP passes review. Scope creep here is failure.

## ⭐⭐ MEASURED GATE SIGNAL (2026-07-21) — GO to fund the B build; real WMMA kernels, not a projection

Bracketed the fused-flash win with **real measured kernels** at the MVP config (`T=KV=2048, H=8, Hd=128`, gfx1100, warmed clocks, DEBUG=2 `tm`):

| stage | SDPA (measured) | fused floor (measured WMMA pieces) |
|---|---|---|
| `QKᵀ` | 304µs @ 28.4 TFLOP/s (WMMA) | 296µs (same WMMA) |
| softmax | 362µs, ~550 GFLOP/s (HBM-bound, 2 passes) | ~0 — fused in-register, no HBM |
| `P·V` | **1187µs @ 9 TFLOP/s** (reads fp32 probs from HBM) | 460µs @ 18.7 TFLOP/s (resident fp16 probs, WMMA) |
| **total** | **~1853µs** | **~756µs → 2.45×** |

**Why this GO is credible where deepseek's was not:** deepseek compared SDPA to an *idealized matmul-only floor* (13.9ms, softmax/staging deleted — unbuildable). This compares SDPA to **two real achievable WMMA kernels** (a `QKᵀ` and a *resident-probs* `P·V`) that run today. The spill cost is concretely located: softmax's 362µs of HBM passes vanish under fusion, and PV drops 1187→460µs because it stops reading freshly-materialized fp32 probs from HBM. **The occupancy failure that killed the decode-reuse path (M=1, no parallelism, scalar `fdot2`) does NOT apply** — prefill is M=2048 rows (full occupancy) with genuine WMMA on both matmuls.

**The one thing this bracket still does NOT measure** (and only the built kernel will): whether fusing `QKᵀ`+softmax+`P·V` into a *single* kernel raises register/LDS pressure enough to cut occupancy vs running them as three kernels. That is the residual build risk — but the 2.45× headroom is a large cushion, and even a substantial fusion tax likely still lands below SDPA. **Verdict: fund the B build.** The bracket is enough to commit; the built kernel converts 2.45×-with-a-caveat into a hard number.

---

## ⭐ PIVOT (2026-07-21) — hand jig abandoned; MVP is now the minimal B slice (scheduler fusion)

Hands-on finding (Claude, before spending the jig's build cost): WMMA on gfx1100 is **declarative** (`tc.py:amd_rdna3` swizzle) and the scheduler applies it by reshaping matmul operands + emitting the `Ops.WMMA` UOp inside the TC opt (`postrange.py:_apply_tc_opt`) — verified by compiling a real fp16 matmul (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, `float8 = wmma(half16, half16, float8)`, wave32). **Consequence:** a hand-authored fused flash kernel would have to *reproduce that swizzle + fragment loads by hand* — i.e. reimplement the TC opt inline — which is near-B complexity done by hand and then thrown away. That is exactly why the decode kernel uses scalar `fdot2` and why prior attempts stalled here.

The cost calculus inverts: the jig's expensive core (hand WMMA swizzle) is throwaway, while **Option B reuses the scheduler's WMMA for free** and its only new work is the online-softmax *blocking* rewrite. So the MVP proof is now a **minimal B slice**: land the smallest scheduler-side blocked-online-softmax fusion for ONE config (14B, `T=KV=2048`, causal, fp16, resident K/V), reusing the scheduler's existing WMMA, and measure it against SDPA at the §5 gate. Keepable (first brick of B), not disposable. The disposable-hand-jig sections below are superseded; kept as history.

---

**The (superseded) hand jig was DISPOSABLE SCAFFOLDING — it would have been `rm`'d at the end.** The two files you create (`flash_prefill_wmma_kernel.py`, `flash_prefill_wmma_mvp_gate.py`) exist *only* to produce the gate number. At the end of the task — **whether GO or NO-GO** — you `rm` them and commit the removal. They are **never shipped, never wired into routing, never kept.** This is deliberate: the *shipped* path is scheduler-native (see §7, Option B), so a hand-authored UOp kernel cannot be the deliverable — it would be exactly the hand-kernel we are choosing *not* to ship. The MVP hand-builds only because that is the cheapest way to get a trustworthy number; the number is the deliverable, the kernel is a throwaway jig.

**Also note (kickback correction, 2026-07-21):** a *roofline arithmetic projection* — measuring SDPA and comparing it to an idealized matmul-only floor — is **M0, not M3, and NOT a gate pass.** The reuse-path NO-GO already proved projections and reality diverge precisely at the fusion/softmax/barrier/occupancy overhead an idealized floor deletes. M3 requires the **actual built, correctness-validated kernel** measured in absolute `tm`, plus a `C_peak` measurement that is self-consistent (a kernel reporting >100% of `C_peak` means `C_peak` was mis-measured — fix the denominator before comparing fractions). No kernel built = automatic NO-GO by §5's hard prerequisite.

**Success is a number, not a vibe:** the MVP passes iff the fused-WMMA kernel's attention op is measurably closer to the **compute roofline** than materialized SDPA is, at one large-context config, with correctness held. Exact gate in §5.

---

## 1. The two-ceiling roofline model (the yardstick — replaces llama)

gfx1100 has two rooflines. **Measure both empirically at M0 — do not hardcode spec sheets** (card variant ambiguity: XTX vs GRE differ in bandwidth):

- **Compute ceiling** `C_peak` (fp16 WMMA TFLOP/s): measure via a large square fp16×fp16 GEMM (e.g. 4096³) already lowering to WMMA, `achieved = 2·M·N·K / tm_seconds`. Record it.
- **Memory ceiling** `B_peak` (HBM GB/s): measure via a large device-to-device copy / streaming-read microbench, `achieved = bytes / tm_seconds`. Record it.

For any kernel, report **both fractions**: `compute_frac = achieved_TFLOPs / C_peak` and `mem_frac = achieved_GBs / B_peak`. An op is "on" the ceiling it saturates. **This table — not llama — is the yardstick for every claim in this task.**

Attention today: materialized SDPA is pinned near `B_peak` (HBM-bound by score spill). The MVP's job is to produce a kernel pinned near `C_peak` instead. The delta between those is the honest ROI of the full build.

---

## 2. Reuse map — what already exists (DO NOT re-implement)

**Anti-duplication is a hard requirement.** Before writing any new file, confirm you are not duplicating these. Take the *structure/primitive*, add the new piece only.

| Need | Reuse this (file:symbol) | What to take | What to change |
|---|---|---|---|
| Online-softmax merge (running max/sum, correction, d-sharded PV), LDS K/V staging, WAR-barrier pattern | `extra/qk/flash_kernels.py:flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` | The whole fused-loop skeleton + the two-pass split-score merge + the `mxu0→barrier→end` WAR-barrier fix (lines ~114–146) | **Replace the scalar-`fdot2` `_dot_reduce` (line 91–98) with a WMMA-tiled `QKᵀ`.** Generalize M=1 (one query row/workgroup) to an **M-query tile**. |
| WMMA tile emission for fp16×fp16 | The **scheduler's existing WMMA lowering** — the same path materialized SDPA (`model.py:591`, `qg @ kg.transpose`) already uses to hit tensor cores | Emit the score/PV matmuls as fp16×fp16 ops the scheduler tensor-cores natively (TC opt). **Do not write a new WMMA emitter.** | Nothing — the point is scores are plain fp16 matmuls; the scheduler already knows WMMA. The novelty is *fusing* them, not lowering them. |
| Candidate + geometry-table + gate + warmstart structure | `extra/qk/prefill/packed_wmma_prefill_candidates.py` (`PACKED_WMMA_GEOM`, `gate_combo`, `warmstart_entry`, `PackedWmmaPrefillCandidate`) | The *pattern*: a frozen geom table keyed by (config) → tile dims, a gate, a warmstart entry builder | New table `FLASH_PREFILL_GEOM` for MVP's single config only (one row). Mirror the shape; do not fork the file's logic. |
| Candidate scoring / ranking (for the full-build sweep later, NOT the MVP) | `extra/qk/bubblebeam_futuresight.py:score_candidate/rank_candidates` | Note it exists; the full build's geometry search plugs in here | **MVP hardcodes one geometry.** Do not build search yet. |
| Correctness reference | `model.py:583–598` — the concrete TC-attn path (`scores = qg@kgᵀ·scale + mask; s = scores.softmax(-1); out = s@vg`) | Golden output to diff against (fp16 tolerance) | This IS the reference; the MVP must match it numerically. |
| Measurement harness | `extra/qk/prefill_whole_synced.py` (canonical prefill harness) | DEBUG=2 `tm` GPU kernel time (never wall-clock), warmup to boost clocks (≥200 dispatch) | Add per-op time extraction for the attention kernel only. |

**Files you will CREATE (all under `extra/qk/`, which sz.py leaves unbudgeted — do not add core budget) — both DISPOSABLE, `rm`'d at task end (§0):**
- `extra/qk/flash_prefill_wmma_kernel.py` — the throwaway fused-WMMA proof kernel (hand-authored UOps, reused from `flash_kernels.py`). Exists to produce the gate number, then deleted.
- `extra/qk/flash_prefill_wmma_mvp_gate.py` — throwaway correctness+roofline harness (imports the kernel, the reference, the two-ceiling measurement). Not wired into routing. Deleted at task end.

**Files you will NOT touch in the MVP:** `model.py`, `prefill_routes.py`, `prefill_policy.py`, `postrange.py`, `kernel_lds.py`. No routing, no defaults, no warmstart tables. The MVP runs from its own gate script.

---

## 3. The MVP kernel — minimal spec

**One config only** (pick the config where the score-spill penalty is largest so the signal is clean):
- Model geometry: **14B** — `Hq=40, Hkv=8, G=5, Hd=128`.
- Context: **one large size**, `T=KV=2048` (big enough that attention is memory-bound and the score tile is real; small enough to iterate fast). Causal mask.
- Precision: fp16 activations (Q/K/V), fp32 softmax accumulators — same as the reference.
- KV source: a plain resident fp16 K/V tensor (NOT the concrete-KV cache machinery, NOT quant-KV — those are full-build).

**Structure (the fusion + WMMA that is the whole point):**
1. Workgroup owns an **M-query tile** (block of query rows) for one GQA query head (G=5 query heads share each KV head — MVP may pin one head; full head-sharding is full-build).
2. Loop over KV in blocks: stage a K block into LDS (reuse `flash_kernels.py` staging).
3. **Score = `Q_tile · Kᵀ_block` on WMMA** → the `M×blockKV` score tile lives in **LDS/registers**. (This is the line the reuse path did on scalar `fdot2`; here it is WMMA.)
4. Online-softmax merge over the score tile (reuse the merge; running max/sum/correction).
5. **Context += `P · V_block` on WMMA**, d-sharded PV (reuse).
6. Causal block-skip (reuse) for blocks fully above the diagonal.
7. Never write the `M×KV` score matrix to HBM — that traffic deletion is the roofline win.

**Correctness:** diff the kernel output against the `model.py:583–598` reference on the same random Q/K/V. fp16 tolerance (rel err ~1e-2). Correctness is a hard gate — a fast wrong kernel is a NO-GO.

---

## 4. Milestones (stop at the gate)

- **M0 — Ceilings + reference.** Measure `C_peak`, `B_peak` (§1). Stand up the reference output + the SDPA baseline's own compute/mem fractions at the MVP config. Deliverable: the two-ceiling table + "SDPA attention sits at X% mem-ceiling, Y% compute-ceiling."
- **M1 — Fused kernel, correct, scalar-scored (bridge).** Get the M-query-tile fused loop correct first with the *reuse* scalar path, to isolate fusion from WMMA. Confirms the harness + correctness diff work. (This reproduces the parent doc's ~1.5× regime — expected.)
- **M2 — Swap score/PV to WMMA.** The real step: `QKᵀ` and `PV` on tensor cores, scores staying in LDS. Correctness held.
- **M3 — Roofline measurement + GATE.** Report the kernel's `compute_frac`/`mem_frac` vs SDPA's. Apply the §5 gate. **STOP. Hand to Claude.**

Each milestone: commit on master (`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`), push origin/master, one-line result in the gate script's output log. No branches.

---

## 5. GO / NO-GO gate (roofline, not llama)

At M3, with **correctness held** (hard prerequisite — fail here = NO-GO regardless of speed):

**GO** if the fused-WMMA kernel is **both**:
1. **On the compute ceiling, not the memory ceiling:** `compute_frac` materially higher and `mem_frac` materially lower than SDPA's — i.e. the score-spill HBM traffic is gone and the op is now compute-bound. (Quant: the kernel deletes ≥~80% of SDPA's score-matrix HBM bytes — computable directly from shapes — AND `compute_frac ≥ ~1.5× SDPA's compute_frac`.)
2. **Faster in absolute GPU `tm`** than materialized SDPA at the MVP config (any margin > noise; DEBUG=2, warmed clocks). This is the sanity floor — beating the reuse path's 1.5–2.5× *deficit* means landing < SDPA.

**NO-GO** if: correctness fails, OR the kernel is still slower than SDPA (means WMMA-in-fused-flash on this scheduler can't clear materialization overhead → the full build is not worth funding), OR it's faster but still memory-bound (means the win came from something other than the theorized traffic deletion — investigate before trusting it).

**Report to Claude at the gate:** the two-ceiling table (SDPA vs MVP kernel), the deleted-HBM-bytes number, absolute `tm`, correctness diff, and a one-paragraph GO/NO-GO recommendation. Do not proceed past M3.

---

## 6. Guardrails (non-negotiable)

- **Single GPU lane.** One run at a time. Before running: `pkill` stray python, confirm VRAM free (`rocm-smi`). Do **not** background benches + report "waiting" — it causes MMU faults + VRAM contention. Run it, wait, read the result.
- **`tm`, never wall-clock.** DEBUG=2 GPU kernel time. Warm to boost clocks (≥200 dispatches) or you'll measure cold-clock garbage (the prior "wash" artifact).
- **Python:** `/home/ubuntu/tinygrad-arkey/.venv/bin/python`. Temp files in `/home/ubuntu/.claude/jobs/6db6b205/tmp/`.
- **No BEAM.** Upstream BEAM hangs gfx1100 (`model.py:261`). The MVP hardcodes one geometry; the search tool (for the full build) is BubbleBeam+FutureSight, static-first.
- **Commit on master, never branches.** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Push origin/master.
- **No duplicates.** If you find yourself re-implementing online-softmax, LDS staging, WMMA emission, or a candidate/gate structure, STOP — reuse §2.
- **Measure, don't narrate.** A plausible story from a fragment is not evidence. Run the cheap check.

---

## 7. Exit → review → full build

At M3 GO/NO-GO, **`rm` the two disposable MVP files and commit the removal** (§0), then **hand back to Claude** with the §5 report (the report is text — it survives the file deletion). Then:
- **If NO-GO:** bank the roofline result in this doc's parent chain; the flash-fusion lever is closed with a *measured* ceiling reason. Done.
- **If GO:** Claude reviews the numbers, then writes a **second, exhaustive scope** for the full build. **The full build is Option B — teach the scheduler to fuse attention as a scheduler-native primitive. It is NOT hand-building/shipping the MVP kernel (that is why the MVP kernel is thrown away).**

### The full build is B (scheduler-native flash-fusion), not A (hand kernel) — decided

The shipped path must stay **machine-generated and scheduler-native**, consistent with how the GEMM win was made (packed-WMMA is a view-chain the scheduler lowers to WMMA — no hand kernel). Flash is the one place where "match llama" and "scheduler-native" pull apart, because llama's advantage *is* a hand-fused kernel and tinygrad's scheduler does not fuse attention today (it materializes the score tensor → the HBM spill). The two options were:

- **A — hand-build the fused kernel** (llama-style, `flash_kernels.py`-style). Fast, proven, but off-philosophy and hard to feed BubbleBeam (you'd be autotuning hand-written UOp geometry, not scheduler opts). **Rejected.**
- **B — teach the scheduler a flash-fusion primitive** (CHOSEN): a schedule pattern where online-softmax keeps the `T×KV` score tile in LDS/registers (never materialized to HBM) and lowers `QKᵀ`/`PV` to WMMA via the existing TC opt. Bigger — a genuine compiler project — but it stays scheduler-native, generalizes past this one kernel, and plugs into BubbleBeam+FutureSight (and the eventual static→dynamic autotuner) the same way packed-WMMA's geometry search does.

So the MVP kernel proves the *physics* (the fused-WMMA roofline is reachable); the full build then makes the scheduler produce that schedule itself — no shipped hand kernel. The MVP jig is deleted precisely so no one mistakes it for the deliverable.

The full-build scope (written only after a GO) covers: the scheduler flash-fusion primitive (the core B work), GQA head-sharding, the geometry sweep via BubbleBeam+FutureSight (`FLASH_PREFILL_GEOM` populated by search), multi-KV-size coverage, concrete-KV/quant-KV integration, routing (`prefill_routes.py`/`prefill_policy.py` new strategy), the fp16-path (8B) validation sweep, and the static→dynamic autotuner flip. Out of this task's bounds by design.

**The MVP's only job: cheaply turn "we think fused-WMMA-flash moves attention to the compute roofline" into a measured yes/no — using a throwaway hand kernel — so the weeks-scale *scheduler-native* build is funded on evidence, not hope.**

---

# FULL BUILD SCOPE (for deepseek) — scheduler-native flash-prefill fusion

**Status:** MVP gate = **GO** (measured 2.45× bracket, see top of doc). This is the funded build. Executor: deepseek. Reviewer: Claude. Commit on master, no branches, `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`, push origin/master.

## B.0 — Objective (one sentence)

Teach the **rangeify scheduler** to rewrite the attention subgraph `(Q@Kᵀ·scale + mask).softmax(-1) @ V` into a **single kernel** that tiles the KV axis, carries the online-softmax running state `(m, l, acc)` across blocks, keeps each block's `M×blockKV` score **resident (LDS/registers, never the full `T×KV` in HBM)**, and lowers both matmuls to WMMA via the **existing** TC opt — converting the measured 2.45× bracket into a hard, correctness-validated number.

## B.1 — The scheduler map (verified entry points — trace these, don't guess)

This is NOT upstream tinygrad. It is a **rangeify-based** scheduler. The real sites:

| Concern | File / symbol | Role |
|---|---|---|
| Rewrite machinery | `tinygrad/uop/ops.py` — `graph_rewrite`, `PatternMatcher`, `UPat` | how every rewrite is written (see `function.py:pm_ctx`, `realize.py:pm_flatten_linear` for examples to model on) |
| **Where kernel boundaries form** | `tinygrad/schedule/rangeify.py` — `rangeify_codegen`, `pm_add_buffers_local`, `pm_store_ranges` | ranges/buffers get assigned here; a reduce that writes a buffer another reduce reads = a kernel boundary = the HBM spill. **The fusion must restructure the graph BEFORE this so the blocked-attention lands as one range nest.** |
| Codegen chain | `tinygrad/codegen/__init__.py` | the ordered `graph_rewrite` pipeline; find where to insert the attention rewrite (before rangeify buffer insertion) |
| WMMA (REUSE, do not touch) | `tinygrad/codegen/opt/postrange.py:_apply_tc_opt` + `tinygrad/codegen/opt/tc.py:amd_rdna3` | already lowers an fp16 matmul in a kernel to `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`. Once `QKᵀ`/`PV` are inside one kernel, TC opt tensor-cores them for free. |
| Pattern origin | `tinygrad/tensor.py:1175` `scaled_dot_product_attention` → `qk.cast(...).softmax(-1) @ value` | the exact tensor-level expression that produces the subgraph to match |

**First deliverable (B-M0, do before writing any rewrite): a trace.** Dump the UOp/range graph for the MVP config attention (reuse the `sdpa_bench.py` shape: `T=KV=2048, H=8, Hd=128`) and identify (a) the exact UOp pattern for `softmax(QKᵀ)@V`, (b) the precise rangeify site where the `scores`/`probs` buffers get inserted (the boundary to eliminate), (c) where in the `codegen/__init__.py` chain to insert the rewrite. Write this map to `docs/flash-prefill-fusion-trace-<date>.md`. **Do not start the rewrite until this trace is confirmed with Claude** — a wrong insertion point is the most expensive mistake here.

## B.2 — The rewrite design

The online-softmax math is a **known result** (FlashAttention, Dao 2022) and is already implemented as a UOp recurrence in `extra/qk/flash_kernels.py` (the running `m`/`l`/`acc` + correction `corr = exp(m_old - m_new)`, lines ~114–146). **Reuse that math as the reference for the recurrence — do NOT reuse the coupled decode kernel body** (its split/combine/cache-indexing are what made extraction fail; see the pivot note). The rewrite emits the recurrence into the scheduler graph, not a hand kernel.

Transform, per query-tile × KV-block:
1. `S_block = Q_tile @ Kᵀ_block · scale` — a matmul reduce over `Hd`, kept as a small `M×blockKV` tile (fits LDS/regs). TC opt → WMMA.
2. `S_block += additive_mask` — **−∞ additive mask, never a bool `maximum`** (a raw bool-vector max hit a `MAX→CMPLT` lowering bug in an earlier attempt; the additive form sidesteps it, and matches `flash_kernels.py`'s "sc=-inf for OOB").
3. `m_new = max(m, rowmax(S_block))`; `p = exp(S_block − m_new)`; `corr = exp(m − m_new)`.
4. `l = l·corr + rowsum(p)`; `acc = acc·corr + p @ V_block` — second matmul reduce over `blockKV`. TC opt → WMMA.
5. After the KV loop: `out = acc / l`. Causal: skip blocks fully above the diagonal.

The key scheduler property to achieve: steps 1–4 for a given query-tile must live in **one range nest** so `S_block`/`p` are registers/LDS, not buffers. That is the entire compiler task — the rangeify graph must not insert a buffer between the `QKᵀ` reduce and the `PV` reduce.

## B.3 — Milestones (gate on B-M4)

- **B-M0 — Trace** (§B.1). Confirm the pattern, the boundary site, the insertion point. Review with Claude. **Hard stop before B-M1.**
- **B-M1 — Correct fused rewrite, one config, WMMA-off acceptable.** Land the rewrite so `softmax(QKᵀ)@V` for `T=KV=2048,H=8` becomes one kernel with the score tile resident. Correctness first: diff vs the `model.py:583–598` concrete-TC-attn reference (fp16 tol ~1e-2). Speed irrelevant here — prove the *single-kernel, no-spill* structure exists.
- **B-M2 — WMMA on.** Confirm TC opt lowers both matmuls in the fused kernel (inspect generated `__builtin_amdgcn_wmma`). Correctness held.
- **B-M3 — Occupancy/pressure tune.** This is the residual risk the bracket couldn't measure: does single-kernel register/LDS pressure cut occupancy? Tune the query-tile M and KV-block sizes (this is where BubbleBeam+FutureSight geometry search plugs in — mirror `PACKED_WMMA_GEOM`/`FLASH_PREFILL_GEOM`; static table first, per the static-for-troubleshootability decision).
- **B-M4 — Gate re-run + report.** Measure the *built fused kernel* vs SDPA at the MVP config (DEBUG=2 `tm`, warm clocks). Apply §5 gate (compute_frac up, mem_frac down, ≥80% score-HBM-bytes deleted, faster `tm`, correctness held). Report to Claude with the two-ceiling table. This converts the 2.45× bracket into a hard number.

## B.4 — After the core lands (breadth — separate milestones, do NOT block B-M4 on these)

GQA head-sharding (`Hq=40,Hkv=8,G=5`); multi-KV-size coverage (512→4096); the `FLASH_PREFILL_GEOM` geometry sweep via BubbleBeam+FutureSight (`extra/qk/bubblebeam_futuresight.py:score_candidate/rank_candidates`); concrete-KV / quant-KV integration (the `prefill_concrete_kv` path); routing (`prefill_routes.py`/`prefill_policy.py` — new strategy alongside `BOUNDED_PACKED_TILES`); the **fp16-path (8B) validation** (the fusion is weight-format-independent — confirm the same win on the overlay path); the static→dynamic autotuner flip once end-to-end is proven.

## B.5 — Guardrails + anti-duplication (hard)

- **Reuse, don't reinvent:** online-softmax math ← `flash_kernels.py` (math only, not the kernel); WMMA ← TC opt (`postrange.py`/`tc.py`), never a hand emitter; rewrite machinery ← `graph_rewrite`/`PatternMatcher`; geometry search ← `bubblebeam_futuresight.py`; correctness ref ← `model.py:583–598`; measurement ← `prefill_whole_synced.py` / DEBUG=2 `tm`. If you're re-implementing any of these, stop.
- **Single GPU lane.** `pkill` strays + confirm VRAM free (`rocm-smi`) before each run. Never background benches + report "waiting" (MMU faults / VRAM contention). Run, wait, read.
- **`tm` not wall-clock**, warm ≥200 dispatches. `/home/ubuntu/tinygrad-arkey/.venv/bin/python`. Temp in `/home/ubuntu/.claude/jobs/6db6b205/tmp/`.
- **No BEAM** (hangs gfx1100). **Commit on master, no branches**, the Co-Authored-By trailer, push origin/master.
- **Core budget:** the rewrite lives in `tinygrad/` (it's real core), so watch `sz.py` (BUDGET_DIRS ≤35000, target 30000; currently ~30,294). Keep the fusion pass lean; geometry tables / search stay in `extra/qk/` (unbudgeted).

## B.6 — Honest fallback

If the single-kernel no-spill structure proves not landable in the rangeify scheduler within a reasonable budget (e.g. the range model can't express the cross-block `(m,l,acc)` recurrence without a buffer), that is itself the decision-relevant finding: it means B is more expensive than the 2.45× headroom justifies *right now*, and the fallback is to bank the trace + the blocker precisely (what in rangeify forces the buffer) and stop — not to silently regress to a hand kernel. Report that to Claude rather than grinding indefinitely.

**Bottom line for deepseek: B-M0 trace first (confirm with Claude), then land the fused no-spill rewrite for one config (correctness before speed), then WMMA, then tune, then re-run the gate. The measured 2.45× says the prize is real; your job is to make the scheduler produce it.**
