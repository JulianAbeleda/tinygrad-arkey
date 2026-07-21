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

**Files you will CREATE (all under `extra/qk/`, which sz.py leaves unbudgeted — do not add core budget):**
- `extra/qk/flash_prefill_wmma_kernel.py` — the fused-WMMA kernel (the one real artifact).
- `extra/qk/flash_prefill_wmma_mvp_gate.py` — standalone correctness+roofline harness for the MVP (imports the kernel, the reference, the two-ceiling measurement). Not wired into routing.

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

At M3 GO/NO-GO, **hand back to Claude** with the §5 report. Then:
- **If NO-GO:** bank the roofline result in this doc's parent chain; the flash-fusion lever is closed with a *measured* ceiling reason. Done.
- **If GO:** Claude reviews the MVP kernel + numbers, then writes a **second, exhaustive scope** for the full build — covering: GQA head-sharding, the geometry sweep via BubbleBeam+FutureSight (`FLASH_PREFILL_GEOM` populated by search), multi-KV-size coverage, concrete-KV/quant-KV integration, routing (`prefill_routes.py`/`prefill_policy.py` new strategy), the fp16-path (8B) validation sweep, and the static→dynamic autotuner flip. That build scope is out of this task's bounds by design.

**The MVP's only job: cheaply turn "we think fused-WMMA-flash moves attention to the compute roofline" into a measured yes/no, so the weeks-scale build is funded on evidence, not hope.**
