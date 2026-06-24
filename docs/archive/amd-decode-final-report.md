# AMD Decode Flywheel — Final Report

> **SUPERSEDED (2026-06-16) — current decode state is `amd-decode-banked-20260616.md`.** This Jun-15 report
> predates Phase 2 + the bank; the flywheel verdict is in `amd-decode-flywheel-postmortem.md`. Historical.

Date: 2026-06-15
GPU: AMD RX 7900 XTX (gfx1100, RDNA3). Framework: tinygrad (this fork). Model: Qwen3-8B Q4_K_M (GGUF).
Measured device peaks: HBM **859 GB/s** (warm streaming copy, 89% of datasheet), fp16 compute
**83.64 TFLOPS**.

---

## TL;DR

We set out to prove a specific thesis: **a learned "flywheel" can drive machine search to produce
Q4_K kernels competitive with hand-tuned llama.cpp, without per-kernel hand-tuning** (the
Ansor/AutoTVM model — author a template once, let search tune it). After a long, disciplined
investigation the answer is precise and two-sided:

- **The on-target quantized-decode paths are dead for the loop.** The batch-1 Q4_K GEMV space is
  *flat* (a deterministic lookup ties the learned model); the fused Q4_K→WMMA path is *framework-walled*
  (a custom kernel that manually stages the dequant in LDS — the only way to fuse cheaply — cannot be
  auto-tiled and tops out at ~3–6% of peak, ~10× slower than native matmul).
- **But the loop mechanism itself works**, demonstrated on the one space that is simultaneously rich,
  competitive, and learnable: the **native-matmul opt-schedule space**. A cost model trained on
  accumulated `(shape, config → device_time)` data guides search to **95% of oracle in a median of 1
  measured config, vs ~86 for random** (≈86× fewer trials), robustly beats the deterministic lookup
  (which collapses to 0.05 across diverse shapes), and **improves as the corpus grows** (best-of-5:
  0.67 at 1 shape → 0.98 at 25).

The real deliverable is not a kernel that beats llama.cpp — it is a **precisely-located map of where
learned kernel-search has a home and where it does not**, every boundary grounded in device
measurements, plus a clean existence proof of the loop on the substrate that supports it.

---

## The mission, and how it narrowed

Mission (the user's words): *"machine search as competitive with llama.cpp,"* without per-kernel
hand-tuning. The benchmark is llama.cpp's single-stream Q4_K decode on this GPU (**103.84 tok/s**; our
deterministic generated policy sits at ~50% on 8B, 61.6% on 14B). The distinctive claim is *machine
search* (learned, automated); the benchmark is llama.cpp.

The investigation repeatedly discovered that the mission's three requirements — *machine-search* ∧
*on-target Q4_K* ∧ *competitive-with-llama.cpp* — do not intersect, and narrowed accordingly: from
"beat llama.cpp on decode" to "does the learned loop have **any** home, and where?"

---

## The arc, phase by phase (what happened and why)

**Phase 3F–4.3 — the triage line (learned cost model vs priors).** Built a leak-free XGBoost cost
model over kernel-candidate features; it beat the `mechanism_prior` baseline on a family-split holdout
(macro-F1 0.87 vs 0.48). Every apparent win dissolved under scrutiny: the "model beats prior" margins
were a **floor-collapse artifact** of the safe-skip metric, and a fair **deterministic class-skip**
gate matched the model exactly (48 vs 48 experiments saved at 100% recall). At that feature set the
learned model added **nothing** over a cheap deterministic rule.

**Phase G0 / M — the metric was noise.** The 3F–4.x outcomes were scored on wall-clock `q4_eff`
(~28–35 GB/s, dominated by ~0.27 ms launch overhead). Re-based to the **device** metric vs measured
peak: the same "winning" schedules were *slower* on device; **0 of 7** beat the baseline by >2%
(median −38.6%). Located the real bottleneck: loads already wide (`b128`); the kernel body is
dominated by Q4_K dequant ALU (~3862 vector ops); both bandwidth and ALU under-utilized →
**latency/occupancy-bound** on the dequant dependency chain.

**Primitive analysis + Phase B0 — batching is the lever.** Reduced to load / dequant / reduce: a
batch-1 GEMV has zero weight reuse, the dequant sits on the reduction's critical path, and there is
too little parallelism to hide it. The structural fix is reuse via **batching**: B0 measured a
**13–26× per-token** amortization. The one genuine batch-1 win in the whole program is `packed_load`
at **+6%** (correctly measured on device).

**Phase W1 / W1b' — the fused dequant→WMMA primitive.** Forcing tensor cores on a fused
`dequant→cast(f16)→matmul` emits WMMA but runs **28× slow (23 GFLOPS)** — the rendered ISA confirmed
the dequant is recomputed *inline* feeding every WMMA. **W1b' fixed it**: a hand-authored custom
kernel that dequants the weight tile **once** into an LDS (`DEFINE_LOCAL`) buffer, barriers, then a
matmul reduce the TC opt turns into WMMA reading the staged tile. Correct, reads compressed, dequant
staged once (verified: all dequant pre-barrier, all WMMA post-barrier), and fusing is **~free** vs the
same-structure fp16 ceiling (mean 1.04×). The competitive *primitive* existed.

**Phase W2 — the framework wall.** Grid parallelism (~70×) and split-K K-tiling made the fused kernel
correct at real `K=4096`, but it **plateaus at ~3–6% of peak** while **native fp16 matmul reaches
33–98%**. The fused custom kernel is ~10× slower than native, and 5–6× slower even at small-N
memory-bound decode. Root cause (robust): not the dequant (the manually-staged fp16 *ceiling* also
caps ~3–8%); it is that a custom kernel which **manually stages LDS** applies only the `TC` opt, while
native matmul applies a full `TC+UPCAST×2+LOCAL` schedule — appending those exact opts to the fused
kernel barely helps (3.0→3.7%). **Fundamental tension: the manual LDS staging that makes fusion free
is exactly what blocks the auto-tiling that reaches peak.** A competitive *fused* quantized GEMM is
not expressible via tinygrad `custom_kernel` + opts.

**Pivot — Phase N (loop substrate).** With the on-target spaces dead, we reframed around *the loop*:
it is only worth building if **one** space is simultaneously **rich** (search matters), **competitive**
(a point worth finding), and **learnable** (a model beats the deterministic baseline). Scorecard:

| Space | Rich? | Competitive? | Learnable? | Verdict |
|---|---|---|---|---|
| Q4_K GEMV (decode) | flat | ~50% | **no** (lookup ties model) | dead |
| Fused Q4_K→WMMA | yes | **no** (~6%, walled) | n/a | dead |
| **Native fp16 matmul** | **yes** | **yes** (33–98%) | **untested** | **the only candidate** |

**N0a — matmul_decoded is the competitive batched path.** A cheap dequant pass (compressed→fp16,
8603 GFLOPS) + native matmul beats the W2 fused kernel **4.5–9.6× per-call** at every batch size
(dequant ~112 µs, fully amortized). **N0b — the opt space is favorable:** rugged (111–223× spread
between best/worst config), sharp (2–10 of ~250 within 10% of best), **no universal winner** (0 configs
top-5 across all shapes — a lookup fails), but **structured** (configs cluster by shape-family).

**N1 / N1.1 — it is learnable.** Leave-one-shape-out XGBoost (shape+config → tflops) over a 26-shape,
3878-record dataset: the model's top-1 reaches **0.922 of oracle** (pre-registered 0.90 gate **passes**
— closed by adding small-N coverage, threshold never moved), vs a deterministic lookup that **collapses
to 0.054** (model wins **26/26 folds**). Transfer rises with experience (top-1: 0.46 at 1 train shape
→ 0.92 at 25).

**N2 — the loop works.** Model-guided best-of-K / oracle: **0.92 (K=1) → 0.98 (K=5) → 0.99 (K=20)** vs
random 0.48 → 0.72 → 0.85. **Trials to 95% of oracle: guided median 1.0 vs random 86.3.** Online
flywheel (best-of-5 vs corpus size): **0.67 (1 shape) → 0.98 (25)**. All gates pass.

---

## What is real (findings, ranked)

1. **The loop mechanism works on a real substrate.** On native matmul — rich + competitive + learnable
   — a learned cost model guides search to near-oracle in ~1 trial (≈86× fewer than random), beats the
   deterministic baseline robustly, and improves with accumulated experience. *(The positive.)*
2. **A competitive fused quantized GEMM is not expressible in tinygrad.** Fusion (manual LDS staging)
   and peak tiling (auto-opt) are mutually exclusive in the `custom_kernel` + opt model; the fused
   path caps ~6% vs native 33–98%. *(The decisive negative — and the precise framework limit.)*
3. **The original triage premise was a dead end** at its feature set: a deterministic rule matched the
   learned model, and the metric being optimized was wall-clock noise.
4. **The real, correctly-measured optimizations are structural, not learned-triage:** `packed_load`
   +6% (batch-1), batching 13–26×/token, and matmul_decoded (native matmul, 33–98% peak) for the
   batched regime.

## The meta-conclusion (the generalizable result)

Learned kernel-search has a home **only** when the space is rich + competitive + learnable, and it has
**no** home under two anti-conditions that recurred throughout:

- **Flat / deterministic** — the answer is a lookup (GEMV decode): a model cannot beat a constant.
- **Physics- or framework-bound** — the answer is set by the roofline or by what the framework can
  express (batch-1 reuse structure; fused-vs-tiling), not by a selector.

The native-matmul space is the one place that escapes both: tiling choice genuinely matters (rugged),
a competitive point exists (33–98%), no single config wins everywhere (no lookup), yet the optimum is
predictable from shape features (learnable). That is exactly where the flywheel earns its keep.

---

## Scope boundaries and honest caveats

- **Decoupling.** The positive (N1/N2) is on **native fp16 matmul**, which serves quantized inference
  via **matmul_decoded for the batched regime** — it is a *general learned-autotuning* result. It is
  **not** a llama.cpp single-stream-decode win; that bar's on-target spaces (GEMV, fused-WMMA) are
  dead. We never relabel one as the other.
- **Pilot scale.** 26 shapes, a 277-schedule sample of the opt space, leave-one-shape-out CV — a
  credible pilot, not a paper-scale study.
- **Offline simulation.** N2's "measure the top-K" is a lookup of already-measured true device times;
  it faithfully simulates guided search but is not yet wired into a live BEAM warm-start.
- **No goalpost-moving.** The N1 strict gate (top-1 ≥ 0.90) was reported as a FAIL at 0.89 and only
  declared PASS after **adding data** lifted it to 0.922 — the threshold was never changed.

## Methodology lessons (the part that generalizes)

- **Validate the metric before optimizing anything.** Wall-clock vs device timing was the difference
  between optimizing noise and optimizing the kernel.
- **Measure against the roofline.** "% of measured peak" tells you whether headroom even exists; we
  nearly declared "no headroom" when it was ~5×.
- **Adversarially verify your own wins.** Every positive that we scrutinized shrank or vanished
  (floor-collapse, the 0/7 re-audit, the fused "win" that was a structure-matched ceiling).
- **Pre-register failure modes and freeze predictions.** Deciding in advance what "the lookup wins" or
  "no competitive point" means kept us from rationalizing flattering results.
- **Reduce to primitives when stuck.** Batching, and later the fusion-vs-tiling tension, only became
  obvious after stripping to load/dequant/reduce and stage/tile.
- **Don't move the goalposts.** Close a missed gate with more evidence, not a softer threshold.

## Artifacts & reproducibility

- Plans/reports: `docs/amd-decode-flywheel-proof-plan.md` (phases M–W2), `docs/amd-decode-loop-substrate.md`
  (Phase N), `docs/amd-decode-flywheel-postmortem.md`, this report.
- Kernels/harnesses: `extra/qk_marlin_w1b.py` (fused primitive), `extra/qk_marlin_w2.py` (grid + split-K
  + the W2 verdict), `extra/qk_matmul_decoded.py` (N0a), `extra/qk_beam_log.py` + `extra/qk_loop_dataset*.py`
  (opt-space dataset), `extra/qk_loop_learnability.py` (N1), `extra/qk_loop_search.py` (N2).
- Data/artifacts: `bench/amd-decode-flywheel-proof-20260614/{wmma-w1,wmma-w1b,wmma-w2,native-matmul-N0}/`.
- Tests: `test/external/test_qk_{wmma_w1,marlin_w1b,marlin_w2,matmul_decoded,beam_log,loop_learnability,loop_search}.py`.

## What a follow-up would do

1. **Make the loop live** — wire the cost model into a tinygrad BEAM warm-start and measure real
   wall-clock autotuning speedup on fresh shapes (turn the offline simulation into a tool). **DONE — see
   the Phase L addendum below.**
2. **Scale the substrate study** — more shapes/ops (conv, attention), the full opt space, cross-op
   transfer; test whether the flywheel generalizes beyond matmul.
3. **Close the decode gap separately** — the llama.cpp bar is a GEMV problem; the lever is int8
   activation / DP4A `mmvq`. Now quantified (`docs/amd-decode-consolidated-first-principles.md`): fp emits
   ~4.06 VALU/weight vs a DP4A floor ~1.35 (~3× headroom, all in the dot), but tinygrad emits zero
   `v_dot4` — the gap is a renderer codegen feature, not a search result.

## Addendum (2026-06-15) — Phase L: the loop is LIVE (follow-up #1 resolved), and its boundary

Turned N2's offline simulation (which looked up measured times) into a real autotuner that times
candidates LIVE on device on FRESH held-out shapes. `extra/qk_loop_live.py`,
`extra/qk_loop_beam_warmstart.py`, `bench/.../loop-live-{L0,L1,L2}/`.

- **L0/L1 — PASS (the positive).** On 6 GEMM shapes absent from the 26-shape corpus, the N1 model ranks
  the 277 configs and its top-8 are timed live: **mean 0.977 of the live oracle** (vs random 0.821),
  **95% of oracle reached in a median of 3 live timings** (random ~82), and **~42× wall-clock speedup**
  (time the guided top-8 vs the exhaustive 277 sweep). The offline 0.92/86× result HOLDS on real silicon
  for unseen shapes. Honest weak spot: small-N (N=64) needs more timings (k95=12, guided@8=0.92).
- **L2 — NEGATIVE, and informative (the boundary).** Wiring the model into tinygrad's NATIVE `beam_search`
  as a candidate-pruner (optional, default-OFF hook at `search.py`) does NOT transfer: pruning to the
  model's top-K saves wall-clock (8.5×) but collapses kernel quality to 0.60; relaxing the prune recovers
  quality (0.91 at keep_k=48) but erases the speedup (1.9×) — no operating point wins both. Cause: the
  model trained on COMPLETE 277-config schedules scores native BEAM's PARTIAL schedules out-of-distribution
  and has no features for BEAM's larger action pool (`SWAP`/`GROUP`/`THREAD` — the cold winner uses `SWAP`).

The two-sided Phase-L result sharpens the meta-conclusion at the integration layer: **the learned loop is
a real, live, 42× autotuning tool on the substrate it was trained for, and does not transfer to a
structurally different search substrate without retraining.** Native-BEAM integration would need a dataset
of partial-schedule timings over BEAM's full action space — the "scale the substrate" follow-up.

## Addendum 2 (2026-06-15) — the v_dot4 decode lever REOPENS (D0), and the scale-substrate blockers

Pursuing the two leftover follow-ups (`docs/amd-loop-scale-and-vdot4-plan.md`):

- **The decode "DP4A is dead" verdict was an asm-volatile artifact — REOPENED.** Phase D concluded DP4A
  doesn't help, but it emitted v_dot4 via `asm volatile` (a scheduling barrier, its slowest variant at 35
  GB/s). The renderer targets HIP C++, where the clean path is the compiler builtin `__builtin_amdgcn_udot4`
  (schedulable). gfx1100 accepts the UNSIGNED builtin with `__attribute__((target("dot-insts")))`, and Q4_K
  already uses the unsigned dot + bias correction. **D0** (`dp4a-d0/BUILTIN_VS_ASM_RESULT.md`): the same
  Q4_K GEMV via the builtin hits **169.6 Q4-GB/s ≈ fp's 173** at full occupancy, **2.54× over the asm
  version**, exact-correct, and realizes the consolidated doc's predicted instruction floor (~1.58
  VALU/weight vs fp 4.06). The decode instruction-count lever is REAL and kernel-competitive.
- **D1 — wired e2e, and it's a definitive NULL.** Built the builtin-udot4 GEMV with the occupancy fix
  (64 rows/wg) → **302 Q4-GB/s standalone, 1.77× FASTER than fp** (171), correct. Wired into decode
  (`Q4K_VDOT=1`, default-off; the renderer emits the `_dp4a` helper when referenced). End-to-end
  (`dp4a-d0/D1_E2E_RESULT.md`): **decode tok/s is UNCHANGED (30.2 vs fp 30.3)** despite the kernel being
  1.77× faster and reading HALF the bytes/token. **The kernel win does not cash out e2e** — decode is
  latency/launch-bound at the token level, not GEMV-throughput-bound. So the v_dot4 lever is real in
  isolation and null in practice: the decode gap is structural (per-token latency / many small launches),
  not a single-kernel instruction-count gap. This CLOSES the decode lever hunt — the last candidate
  (DP4A) is overturned-then-nulled, every kernel-level lever exhausted.
- **Scale-the-substrate is blocked on this setup.** Harvesting partial schedules over native BEAM's full
  action space HANGS gfx1100 (HW faults) — only the curated 277-config substrate is stable. Conv ASTs
  build, but the matmul opt-candidate set fails on conv's reduce kernel (different axes) and its baseline
  is tiny (0.1 TF, likely flat). Both need work beyond this hardware's stable envelope.
