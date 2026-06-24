# 8B decode — remaining-gap research scope (2026-06-18)

Goal: close the knowledge gaps on why **llama.cpp ~92–99 tok/s** vs **tinygrad banked 68.3/66.3/60.9 @ctx512/1024/4096**
(~1.45×), and decide whether any remaining path is real, bounded, and worth building. **Not a shipping doc — nothing
routed or defaulted.** Method: principles-strict (full primitive boundary; in-model W==D is final authority for
decode; every result labeled diagnostic/candidate/shipped/refuted/deferred; no win unless required activation/
layout/reduction/runtime work is included). Qwen3-8B-Q4_K_M, RX 7900 XTX / gfx1100.

## TL;DR

- **The remaining decode (batch-1) gap is dominated by `compiler/register-scheduling` on the one role coop can't
  touch (Q4_K ffn_gate/up, 44% of weight traffic).** Measured: tinygrad custom_kernel 57% peak vs handwritten-HIP
  *identical structure* 65% vs llama 70% — the 57→65 recovery isolates it to per-thread code quality. **Surface is
  bounded but tinygrad-internals (high risk), and gated by a format wall** (any int-dot win is eaten by the q8 pack
  → 0.96× + lossy). The only byte-identical lever left is improving the fp-coop kernel's codegen. **Verdict: open
  but unbounded-in-practice (no small proof; internals).**
- **q8 side-channel (Track 1): verdict D** — cost target reachable, but blocked by a multi-output fused custom-norm
  build with no codebase precedent, ~+3–4% EV. Reopen only bundled with a broader fused-norm refactor.
- **Spec decode (Track 3): the measured surprise.** The "slow verify" was a harness wiring bug (verify ran the dense
  `_fallback`), BUT fixing it does **not** save spec decode: the batched-K verify GEMM costs **4.53× one pass for
  K+1=5 (~K, no weight amortization)**, so spec still loses. **The precise missing primitive = a weight-reuse
  (LDS-tiled) batched-K GEMM — the SAME primitive gap that blocks prefill.** Verdict D (deep codegen/runtime).
- **Best explanation of llama's ~100:** llama does ~⅓ the instructions/weight in the GEMV inner loop (q8 activation
  + native signed dot4 + block-amortized affine) compiled into tightly register-scheduled per-thread code, with a
  128-thread/row in-kernel warp-shuffle reduction. tinygrad's byte-identical fp path pays 4.06 VALU/weight and its
  custom_kernel lowering produces lower-quality per-thread code than clang. **tinygrad's path to ~100 is
  MMVQ codegen/work-decomposition + activation lifecycle — both bounded surfaces, neither cheap, the int-dot one
  also lossy.** No single bounded edit closes it; spec-decode (the only orthogonal "beat llama" route) is blocked by
  the prefill-class weight-reuse primitive.

---

## Track 1 — Activation lifecycle / q8 side-channel → **VERDICT D** (feasible but deep)

Primitive scoped: `ffn_norm_fp_with_q8_sidechannel_for_gate_up` — can Q4_K ffn_gate/up use int-dot without a
standalone q8 activation tax, by a fused RMSNorm producer emitting fp + q8-packed + q8-scales at ≤4.8µs effective?

- **Cost target is reachable (so NOT C).** Current pack = 29.7µs/4 kernels, all launch/ramp-bound (~7µs floor each;
  the activation is 16KB ≈ 0.02µs of real transfer). Break-even ≤4.8µs (1.15× coop gate; 0µs → 1.20×). Folding
  quant+pack onto the RMSNorm *apply* pass runs the per-32 max on **already-resident normalized values** (no extra
  global read, no extra launch) → design estimate ~0–5µs effective, plausibly clears break-even.
- **The per-32 max cannot piggyback the RMSNorm reduction** (`tinygrad/nn/__init__.py:300`): RMSNorm reduces
  `mean(x²)` (one scalar/row, sum-of-squares, over pre-norm x); q8 needs `max(|y|)` over 128 per-32 blocks of the
  *post-norm* y. Different operator AND granularity AND input — no algebraic derivation. It can share the *data
  pass* only inside a hand-written kernel.
- **Blocking layer = multi-output fused custom-norm build.** Must emit fp + packed-q8 + scales from one launch:
  multi-store plumbing that "has repeatedly fought custom_kernel," with **zero single→multi-output precedent in the
  codebase** (every `custom_kernel` is single-output: `model.py:143/155/160`). Must also stay correct at T=1 and T>1
  with an fp fallback, and clear an **unmeasured dNLL ≤0.01** gate (path is q8-lossy, rel 0.006).
- **EV structurally capped ~+3–4% decode** (gate+up = 2 of 7 linears; reuse ceiling **2** because k/v are Q6_K).
- Prior audits (`q8-sidechannel-ffn-verdict`, `q4k-ffn-q8-lifecycle-verdict`, `qk-q8-activation-lifecycle-verdict`)
  all converged on C/D. **This scope confirms D.** Reopen only bundled with a broader fused-norm refactor (which
  pays for the multi-output plumbing anyway), or if a model change lifts the reuse-2 cap.

## Track 2 — MMVQ efficiency-gap accounting → **gap table**

Per-role decode breakdown (weights = 4.68 GB/token, all roles batch-1 weight-bandwidth-bound; llama aggregate
~70% HBM peak). %peak figures from the MMVQ closeout + coop result docs + handwritten-mmvq bench.

| role | traffic share | tinygrad path | tg %peak | llama %peak | activation | byte-identical | **remaining-delta class** |
|---|---:|---|---:|---:|---|:--:|---|
| **Q4_K ffn_gate/up** | **44.0%** | base fp GEMV, 1 row/thread; **coop NOT routed** (already 41%) | 41 (base) / 48 (fp-coop) / 57 (sudot4-128, lossy) | 70 | fp16 | ✓ (fp-coop) | **compiler/register-scheduling** |
| Q6_K ffn_down | 15.7% | coop SHIPPED (pos→LOCAL) | 14→**39** | ~70 | fp16 | ✓ | work-decomposition (bounded) |
| Q4_K attn_q/o | 14.5% | coop SHIPPED (lane4→LOCAL) | 19→**29** | ~70 | fp16 | ✓ | work-decomposition (bounded) |
| Q4_K ffn_down | 10.9% | split-K fp GEMV (parts=4) | **35.5**→40 | ~70 | fp16 | ✓ | compiler/register-scheduling |
| Q6_K lm_head | 10.8% | coop SHIPPED (pos→LOCAL) | 10→**51** | ~70 | fp16 | ✓ | work-decomposition (largely settled) |
| Q6_K attn_k/v | 1.3% | split-K fp (coop unrouted) | 8.9→14 | ~70 | fp16 | ✓ | **already-refuted** (Amdahl ~+0.5%) |

**Classification rollup:**
- **compiler/register-scheduling DOMINATES** (Q4_K ffn_gate/up = 44% traffic, the only large role coop can't help —
  it's already coalesced at 41%). The ladder is decisive: tg custom_kernel **57%** vs **handwritten-HIP identical
  structure 65%** vs llama 70%. The +8% recovery on identical structure ⇒ the residual is **per-thread codegen**
  (clang register-alloc/ILP/scheduling vs tinygrad custom_kernel lowering), not the dot and not decomposition.
- **work-decomposition** residual on the coop roles (8–16 threads/row vs llama 128; global-partials + external
  `.sum` vs in-kernel warp-shuffle): **real but bounded** — the 128-thread version was built and **refuted as a
  standalone win** (every *correct* 128-thread variant ≤ the 8-thread coop; reduction/occupancy overhead offsets the
  K-parallelism).
- **already-refuted:** dp4a in isolation (+1% e2e); 128-thread decomposition standalone; horizontal QKV fusion
  (Q4K_FUSE −18%); whole-linear int-dot (q8-pack wall: saves ~11µs, pays ~15µs → 0.96× + lossy).
- **format-mandated-unpack / activation-lifecycle** is the *wall* that blocks converting the ffn_gate/up codegen
  residual into a routed win (int-dot needs the q8 pack → eaten + lossy). Only dodge = Track 1's epilogue (D).

**Bounded?** The surface is named (`tinygrad/renderer/cstyle.py` custom_kernel lowering; AMD compile path; the
dequant→dot inner loop's register-blocking/ILP). The native signed dot4 piece is already shipped (`sudot4`). But the
only **byte-identical** surface left is improving the **fp-coop** kernel's per-thread codegen (48% vs the 70%
READRAW ceiling for this access pattern) — genuinely tinygrad-internals, **high risk, no small proof, no q8 wall but
no instruction headroom in the fp dot either.**

## Track 3 — Spec-decode runtime primitive → **VERDICT D** (deep), optimistic version **REFUTED by probe**

Scope: any research path left after the fused-graph verdict, scoping only the missing primitives.

**New diagnostic (this scope, measured — `extra/qk_spec_decode_lowsync.py --measure-verify`, ctx512 K=4):**

| measurement | ms | × one pass | note |
|---|---:|---:|---|
| single T==1 pass (coop decode) | 13.07 | 1.00 | the production decode kernel |
| T=K+1 verify **fallback** (decode_enabled=False, dense) | 144.06 | **11.02** | what both spec harnesses accidentally ran (`.logits()` bypasses the `decode_enabled` toggle; `qk_spec_decode_lowsync.py:28`, `generate.py:37`) — dense re-dequant of all weights/token |
| T=K+1 verify **batched GEMM** (decode_enabled=True) | 59.21 | **4.53** | the existing `q4k/q6k_gemm_kernel` (`model.py:117-126/214-222`); exact (argmax identical ✓) |

**Findings:**
1. The docs' "verify falls off the fast path → slow prefill kernels" was a **harness wiring bug**: the batched-K
   GEMM exists and is wired for 2≤K≤32, but the harnesses called `.logits()` directly (never toggling
   `decode_enabled`), so verify ran the dense `_fallback` (`model.py:114`→`_fallback`). Routing it correctly is a
   2.43× speedup (144→59ms) and exact.
2. **BUT the corrected batched verify is still 4.53× one pass for K+1=5 — i.e. ~K, NOT ~1.** It does **not** amortize
   the weight read across the K columns (the `UPCAST:1:min(K,16)` dequant-hoist saves dequant ALU, not the weight
   HBM re-read). So spec decode still loses: 59ms verify for ~2–2.8 accepted tokens (acceptance is excellent, proven)
   = ~21–29 ms/accepted-token vs 13 ms/token baseline, before adding draft+sync. **The optimistic "verify ≈ 1 pass"
   ceiling is refuted by measurement.**
3. **The precise missing primitive = a weight-reuse (LDS-tiled) batched-K decode GEMM** that reads each packed weight
   block once and does all K+1 dot products before moving on. This is the **same primitive gap as prefill** (the
   prefill plan: tinygrad matmul emits WMMA but LDS=0, re-reads operands → ~27% peak; llama rocBLAS stages a 128×128
   tile in 25.6KB LDS → ~80%). Batch-1 decode escapes it (one column, nothing to reuse — irreducible 1× weight read);
   batch-K verify and prefill both hit it. BEAM would find the LDS-tiling opt but hangs gfx1100.

**Feasibility matrix (path | correctness | required primitive | ceiling | blocking layer | verdict):**

| path | correctness | required primitive | expected ceiling | blocking layer | verdict |
|---|:--:|---|---|---|:--:|
| (a) fast T=K+1 verify | exact ✓ | route through `decode_enabled` (exists) | **measured 4.53× one pass → spec still loses** | none (wiring) — but ceiling refutes it | **refuted** (probe run) |
| (b) batched-K decode primitive | exact ✓ | **weight-reuse / LDS-tiled batched-K GEMM** (read-once, K-reuse) | verify → ~1–1.5 pass → spec ~1.4–1.6× *if* host pipelined | tinygrad codegen (LDS tiling; BEAM hangs) = prefill-class wall | **deferred (D)** |
| (c) two-model graph composition | exact ✓ | TinyJit scheduling of draft+target+accept w/o per-graph-opt loss | ~115 tok/s if host-free | tinygrad runtime/scheduler (fused form refuted 163ms; 2-graph-pipelined unexplored) | **D** |
| (d) async pipelining | exact ✓ | overlap pass-N accept with pass-N+1 draft | hides ~68/86ms host-sync — but **only after (b)** makes verify cheap | tinygrad sync model | **D (conditional on b)** |

---

## What is CLOSED — do not reopen

- **dp4a / int-dot in isolation** — refuted (+1% e2e; kernel win real but e2e-null at batch-1).
- **Whole-linear int-dot (sudot4) for Q4_K gate/up** — refuted by the q8-pack wall (saves ~11µs, pays ~15µs →
  0.96× fp-coop + lossy; reuse ceiling 2).
- **128-thread/row decomposition as a standalone decode win** — refuted (every correct variant ≤ 8-thread coop).
- **Horizontal QKV/FFN fusion (Q4K_FUSE)** — refuted (−18%).
- **Spec-decode fused one-sync TinyJit** — refuted (163ms; two-model fusion schedules pathologically).
- **Spec-decode path (a) "just route the batched verify"** — refuted by this scope's probe (4.53× one pass).
- **Host/runtime overhead as the decode bottleneck** — refuted (W==D, host-sync 0%; decode is GPU-bound).
- **q8 separate-kernel pack / graph-reuse** — refuted (0.94–0.96× coop).

## What remains technically OPEN (with the precise missing layer)

1. **fp-coop per-thread codegen** for Q4_K ffn_gate/up (and ffn_down) — byte-identical, no q8 wall. Missing: tinygrad
   custom_kernel lowering matching clang register-alloc/ILP (48%→toward 65%). **Internals, high risk, no small proof,
   no instruction headroom in the fp dot.** This is the single biggest *bounded-surface* lever and the most likely
   real contributor if it could be moved.
2. **Weight-reuse / LDS-tiled batched-K GEMM** — the prefill-class primitive. Unblocks **both** prefill (Increment 2+)
   **and** spec-decode verify (path b/d). Missing: LDS cache-blocking codegen (a GROUP/LOCAL-into-LDS opt; BEAM finds
   it but hangs gfx1100) — the documented wall-class. **Highest leverage** (one primitive, two payoffs) but a deep
   `[codegen]`/`[runtime]` arc.
3. **q8-as-RMSNorm-epilogue** (Track 1 D) — only worth bundling with a fused-norm refactor; ~+3–4% EV, lossy.

## What, if anything, EARNS a build

- **Nothing earns a standalone shipping build at current EV/risk.** All bounded *cheap* levers are spent or refuted.
- **If one deep arc is funded, fund the weight-reuse / LDS-tiled batched-K GEMM** (open item 2): it is the unique
  primitive that pays off twice (prefill **and** the only orthogonal beat-llama route, spec decode), and it has a
  proven reference shape in-repo (`extra/gemm/amd_flash_attention.py` LDS tiling, the WR1–3 warp-reduce assets). The
  blocker is the gfx1100 BEAM hang / SHAPED_WMMA-convention wall, not expressibility.
- **Candidate (diagnostic, not earned):** an fp-coop codegen micro-arc on ffn_gate/up — only if a cheap renderer
  lever (register-blocking the dequant→dot chain) can be shown in isolation first; otherwise it's pure internals.

## Best explanation of why llama reaches ~100 tok/s

llama reads the 4.68 GB of weights at ~70% HBM peak because its MMVQ does the **minimum instructions per weight**
in the dequant→dot inner loop — a **q8_1-quantized activation** (packed once, reused, ~3.8% cost) through a **native
`v_dot4_i32_iu8` signed dot4** (~1.35 VALU/weight vs tinygrad fp's 4.06) with **block-amortized fp affine** —
compiled by **hipcc/clang into tightly register-blocked, well-scheduled per-thread code**, driven by a
**128-thread/row decomposition with an in-kernel warp-shuffle+LDS reduction and a single write**. tinygrad's
byte-identical fp-coop path pays a **4.06-VALU/weight fp dequant** (the int→fp convert and scalar MAC llama folds
away), keeps activations **fp16 with no q8 pack** (what keeps it byte-identical and competitive, but forecloses the
int-dot), uses **8–16 threads/row + a global-partials round-trip + external `.sum`**, and — decisively for the
dominant ffn_gate/up role — its **custom_kernel lowering produces lower-quality per-thread code than clang**
(handwritten-HIP identical structure recovers 57%→65%). The dequant ALU itself halves bandwidth (the controlled
READRAW-vs-GEMV experiment: 730→365 GB/s); llama wins that ALU race with ~⅓ the instructions/weight in
better-scheduled code. That is the ~70% vs ~41–51% delta — **not the memory system, and not the dot in isolation.**

## Is tinygrad's path to ~100 activation-lifecycle, MMVQ codegen/work-decomp, runtime/spec, or not bounded?

**MMVQ codegen + work-decomposition is the dominant bounded surface, but it is tinygrad-internals (per-thread code
quality + LDS-tiled reduction), not a single edit, and the int-dot half is gated by the activation-lifecycle (q8
pack) wall + lossiness.** Spec-decode (the orthogonal "beat llama" route) is blocked by the **same weight-reuse /
LDS-tiling primitive as prefill** — the one deep arc worth funding if any. So: **the path to ~100 is "MMVQ
codegen/work-decomposition + the LDS-tiling primitive," both real and bounded in *surface* but deep in *build* and
partly lossy — not closeable by the cheap-knob class this campaign has exhausted.** Under the current architecture
(no LDS-tiling codegen on gfx1100 without the BEAM hang, custom_kernel lowering below clang quality), parity is
**not reachable by a bounded edit**; it requires one of the two deep codegen arcs above. Nothing in this scope is
routed or defaulted.

## Provenance / artifacts
Track docs: `q8-sidechannel-ffn-{producer-audit,design-options,verdict}-20260618.md`,
`q4k-ffn-q8-lifecycle-{scope,verdict}-20260618.md`, `qk-mmvq-int-dot-closeout-20260618.md`,
`qk-mmvq-coop-{q4k-attn,ffn-down}-result-20260617.md`, `qk-mmvq-q6k-lm-head-arc-20260617.md`,
`llama-q4k-mmvq-scheduler-audit-20260618.md`, `spec-decode-{production-verdict,fused-graph-scope,low-sync-verdict}-20260618.md`,
`amd-decode-prefill-plan.md` (the prefill LDS-tiling diagnosis), `gpu-performance-first-principles.md`.
New probe: `extra/qk_spec_decode_lowsync.py --measure-verify` (verify-cost isolation; diagnostic). No defaults changed.
