# Fused-Flash CONCRETE-Shape Decode-Attention Gate — Result

Date: 2026-06-21

The **decisive first gate** of `POST_MATMUL_PV_FULL_FUSED_FLASH` (scope:
`docs/post-matmul-pv-decode-strategic-scope-20260621.md`, Phase 1). One question: **can tinygrad express a fixed-shape,
llama-style, hardware-aware decode-attention primitive that beats `gqa_coop_vec` locally at concrete ctx1024?** If yes →
the deep fused-flash project is funded. If no → `REST_DECODE` + v2 is the evidence-backed pivot.

## Decision: **`FUSED_FLASH_CONCRETE_GATE_FAIL_LOCAL_AB`**

The concrete-shape candidate is **value-correct** (rel_rmse 4.9e-4) but **0.965× @ctx1024** vs the **strict same-shape
concrete** `gqa_coop_vec` (reproduced 0.965 / 0.967 / 0.969×) → **loses the local A/B**. Per discipline, **stopped before
any W==D / model route**. Fixing the shape **did remove the matmul-PV symbolic-split blocker** (S=8 fair splits, no
overread) — yet the candidate only **ties** coop, because tinygrad's "tiled" matmul at the decode shape renders
**register-tiled global-load code (16 wg, 305 GFLOPS, no LDS, no `v_dot2`)**, not llama's one-kernel LDS-staged
`v_dot2` fused tile. **The true single fused LDS-tiled kernel stays inexpressible** (the tiled-GEMM codegen and the
`.set/.after` fusion idiom are mutually exclusive). **Bounded *and* concrete-shape decode levers are now both
exhausted → fall back to `POST_MATMUL_PV_REST_DECODE_V2`.**

> The apparent **1.42×** vs the canonical **symbolic** comparator is a **concreteness artifact**, **not a win** — see
> "The concreteness trap" below. Reporting it as a win would be exactly the proxy-win the research principles forbid.

---

## Phase 0 — literature-grounded design note (written before coding)

### Literature mapped to design constraints (not decoration)

| paper | principle | how this candidate implements it |
|---|---|---|
| **FlashAttention** (2205.14135) | IO-aware tiling; keep online-softmax `(m,l)` on-chip; use SRAM/LDS deliberately; avoid HBM materialization | q·k and PV ride tinygrad's **tiled-GEMM codegen** (the only path that *can* stage to LDS / vectorize); the online-softmax `(m, l)` state is carried in registers within the per-split max/prob/combine reduces; the only HBM intermediates are `scores` and `prob` (unavoidable across tinygrad's separate kernels) |
| **Flash-Decoding** (crfm 2023 / pytorch blog) | T=1 has **no token axis** → manufacture parallelism by **splitting KV** across workers, then **rescale/combine** partials by LSE | `S = Tc/L = 8` KV-splits at ctx1024 → the PV matmul batches over `(Hkv, S) = 64`; `flash_gmax` + lean `den`/`comb` do the **LSE rescale/combine** across the 8 splits |
| **FlashDecoding++** (2311.01282) | synchronized partial-softmax update overhead; **flat-GEMM under-utilization**; static dataflow loses | the partial softmax (`flash_max`/`flash_prob`) is kept cheap and async (exp once/key); **the flat-GEMM under-utilization is the explicit risk this gate measures** — the PV GEMM has M=G=4 (skinny) and only 64 logical batches |
| **FlashInfer** (2024) | decode / prefill / append attention differ; kernel must match phase + lifecycle | this is the **decode (T=1) phase only**, concrete shape, **no model route**, gated by `decode_eval` lifecycle |

### Concrete tensor shapes (all fixed — no symbolic K)
- `q`: `[Hq=32, Hd=128]` fp16; `K,V`: `[Hkv=8, MAXC=4096, Hd=128]` fp16; `G = Hq/Hkv = 4`; `L = 128`; **ctx = 1024 fixed**.
- `S = ctx/L = 8` (concrete); `ctx == S·L` exactly → **no range masking**.
- `scores = (q_g[Hkv,G,Hd] @ K[:,:1024,:]ᵀ)·scale → [Hq, 1024]` (tiled GEMM, K=Hd=128 concrete).
- `prob[Hq,1024]` → `A = prob.reshape(Hkv,G,S,L).permute(0,2,1,3) = [Hkv,S,G,L]`.
- `PV = A @ V[:,:1024,:].reshape(Hkv,S,L,Hd) → [Hkv,S,G,Hd]` (tiled GEMM, **K=L=128 concrete**).
- `l = A.sum(-1) = [Hkv,S,G]` (softmax denom per split); LSE combine → `out[Hq,Hd]`.

### Range / LDS / register / mapping
- **q·k mapping:** plain `Tensor @ Tensor` → tinygrad emits `r_4_16_8_16_4_32_4` (LOCAL 8×16 = 128 threads, UPCAST 4,
  UNROLL 4) — register-tiled.
- **PV mapping:** plain batched `Tensor @ Tensor` → `r_8_2_8_16_4_4_32_4` (grid 2×8 = **16 workgroups**, LOCAL 128
  threads, UPCAST 4×4, UNROLL 4, `float buf0[16]` register accumulators).
- **online softmax update:** per-split `m` = `flash_max`; `prob = exp(score − m_s)` once/key = `flash_prob`; global
  `m` = `flash_gmax`; `den = Σ_s exp(m_s − gm)·l`; `out = Σ_s exp(m_s − gm)·PV / den` — the exact stable per-split
  LSE used by `gqa_coop_vec`.
- **LDS layout / register state (intended vs actual):** *intended* — q·k and PV LDS-staged like llama. *Actual*
  (measured below) — tinygrad emits **no `__shared__`/LDS** for these decode-shape matmuls; they are **register-tiled
  with global loads** (16 accumulators in `buf0`). This is the load-bearing finding.
- **expected workgroups / KV splits:** S=8 KV-splits; PV logical batch `Hkv·S = 64`, rendered as **16 grid
  workgroups × 128 threads**; q·k 128-thread workgroups. (coop's partial uses the same 64 = Hkv·S workgroups — the
  parallelism is *matched*; the difference is per-kernel codegen, not split count.)

### How it avoids Path A's redundant work
- exp is computed **once per key** (`flash_prob`), **never per output-dim lane** — Path A's `0.725×` death (W=129
  lanes recompute exp) is structurally excluded. PV is a real matmul (each `prob` read once), not a per-`d` reduction.

### How it differs from the closed lanes
| closed lane | why it lost | how this differs |
|---|---|---|
| **raw fused flash tile** (`fused_flash_naive_loses_to_optimized_split`, 2.5–3.3× slower) | one hand-C kernel lacks GQA V-reuse + coalescing | keeps coop's coalesced GQA structure; routes the **PV through the tiled-GEMM codegen** instead of a hand reduction |
| **scalar LDS+GQA tile** (0.21×) | per-lane redundancy collapsed workgroups | no per-lane redundancy; preserves Hkv·S parallelism |
| **Path A fused softmax+V tail** (0.725×) | per-`d`-lane exp recompute | exp once/key; PV is a matmul |
| **matmul-PV diagnostic** (`BLOCKED_BY_LAYOUT`, 0.936×@1024) | symbolic `Tc` → forced concrete `Smax=32` → full-MAXC **overread** (4–8× extra) | **fixes the shape** → `S=8` concrete & **fair** (no overread) — removes that exact blocker; this is the legitimate concrete first gate the matmul-PV result could not run |

### What would make it inexpressible
- A **single** fused kernel that is **both** LDS-tiled `v_dot2` (q·k+PV) **and** carries online-softmax state needs the
  tiled-GEMM codegen *and* the `.set/.after` fusion idiom **simultaneously** — and they are **mutually exclusive** in
  tinygrad (tiled-GEMM fires only on standalone `Tensor @ Tensor`; the fusion idiom produces scalar reductions with no
  LDS/`v_dot2`). The candidate sidesteps this by staying **multi-kernel** (matmuls tiled, softmax scalar). A *truly*
  one-kernel llama-class tile is the inexpressible object — confirmed below by the register-tiled, no-LDS rendering.

---

## Phase 1 — concrete µkernel harness (`extra/qk_fused_flash_concrete_gate_ab.py`)

Candidate = concrete-shape flash-decode pipeline (above). Comparators: **(primary, authority)** `gqa_coop_vec` at the
**same fixed shape, concrete**; **(cross-check)** the canonical **symbolic** `gqa_coop_vec`. Throughput, median-of-3,
clock-pinned (perf-state restored to `auto`).

### Correctness (vs numpy reference)

| ctx | rel_rmse | max_abs | gate (≤1e-3) |
|---:|---:|---:|---|
| 512  | 4.7e-4 | 1.7e-4 | PASS |
| **1024** | **4.9e-4** | **1.2e-4** | **PASS** |
| 2048 | 5.1e-4 | 8.2e-5 | PASS |

(≤1e-5 unattainable — the matmul reorders fp accumulation vs the scalar partial; ~5e-4 matches coop's own fp-reassoc
band, so ≤1e-3 is the justified threshold.)

### Local A/B (throughput, clock-pinned, ctx1024 fixed)

| comparator | candidate µs | comparator µs | **speedup** | authority |
|---|---:|---:|---:|---|
| **`gqa_coop_vec` concrete (same fixed shape)** | **60.5** | **58.4** | **0.965×** | **PRIMARY / gate** |
| `gqa_coop_vec` symbolic (canonical) | 60.5 | 85.6 | 1.416× | cross-check — **concreteness artifact, not a win** |

Reproduced: concrete 0.965 / 0.967 / 0.969× (decode_eval re-run). **Gate FAIL** (needs ≥1.05×@ctx1024; got 0.965×).

### The concreteness trap (why 1.42× is not a win — load-bearing)
The candidate is **concrete** (no general ctx); the canonical comparator is **symbolic** (`start_pos` `DEFINE_VAR`).
The symbolic comparator's 85.6 µs carries **~27 µs of var-binding / dynamic-shape JIT overhead** over its own concrete
form (58.4 µs) — pure host/graph tax, not GPU codegen. To be **usable in-model** the candidate **must** become
symbolic-count (general ctx), which **re-introduces the matmul-PV `BLOCKED_BY_LAYOUT` problem** (symbolic `Tc` can't
reshape into a tiled `(S,L)` batched matmul) **and** would pay the same ~27 µs generalization tax. So the honest,
apples-to-apples comparator is **concrete-vs-concrete (0.965×)**; the 1.42× is the symbolic generalization tax the
candidate would also owe. Quoting it would be the exact "proxy win" the research principles forbid.

### Codegen / resource evidence (`--gflops`, DEBUG=2/6)

| kernel | role | rendered shape | GFLOPS | LDS | `v_dot2` |
|---|---|---|---:|---|---|
| `r_4_16_8_16_4_32_4` | q·k matmul | 128-thread wg, UPCAST4/UNROLL4 | 386 | none | none |
| `r_8_2_8_16_4_4_32_4` | **PV matmul** | **16 grid wg** × 128 thr, `float buf0[16]`, UPCAST4×4 | **305** | **none** | **none** |
| `flash_max/prob/gmax`, `ffcg_den/comb` | softmax + LSE | 1-thread-wide reduces | ~5–15 | none | none |

- **11 kernels** total (candidate) vs coop's **7** — two extra layout kernels (`prob`-permute copy, `l`-reduce).
- **The PV matmul renders register-tiled, no `__shared__`/LDS, no `v_dot2`** (HIP-C body: `float buf0[16]`
  accumulators, plain global loads). At decode shape tinygrad's tiled-GEMM codegen applies **register/LOCAL tiling**
  but **not** LDS staging — so the candidate is **not** actually llama-class LDS-tiled.
- **PV = 305 GFLOPS at 16 wg** vs the matmul-PV diagnostic's **1078 GFLOPS at 256 wg** (ctx4096, S=32) — the
  **FlashDecoding++ flat-GEMM under-utilization** materializing: at the fair ctx1024 split count the GEMM has too few
  workgroups (16) to fill the 96-CU GPU, so the "tiled win" evaporates.

### Why it loses even with the blocker removed (named failing layer)
The matmul-PV diagnostic was blocked at the **layout** layer (symbolic split count → overread). Fixing the shape clears
that — and exposes the **next** layer: **(a) instruction-selection / memory-path** — tinygrad emits register-tiled
global-load matmuls (no LDS/`v_dot2`) at decode shape, so the "tiled" PV is only ~305 GFLOPS at 16 wg; **(b) work
decomposition** — FlashDecoding++ flat-GEMM under-utilization (16 wg ≪ 96 CUs); **(c) graph integration** — 2 extra
layout kernels' launch+HBM cost. Sum: the pipeline **ties** coop's scalar-partial path (which keeps a single coalesced
GQA partial and fewer launches). The llama-class win needs **all three** folded into **one** LDS-staged `v_dot2`
kernel — the inexpressible object.

---

## Phase 2 — lifecycle

`decode_eval` candidate `fused_flash_concrete_gate` (family `attention_split`, `ab_script` runner, correctness_req
`byte_exact`) → **`FAIL_LOCAL_AB`** (0.969× < 1.05; match=True) →
`bench/qk-decode-eval/runs/20260621T130044-fused_flash_concrete_gate.json`. Refutation
`fused_flash_concrete_gate_register_tiled_not_lds` added (prunes "fix ctx concrete to unblock matmul-PV for a decode
win" / "claim a win vs the symbolic comparator from a concrete candidate"). **W==D: NOT reached** (local failed —
discipline = stop). No `tinygrad/` change, no model route, no default change.

---

## Expressibility finding (the durable result)

tinygrad has **two** codegen paths that *could* produce LDS/vectorized attention, and they are **mutually exclusive**:
1. **tiled-GEMM codegen** — fires only on a standalone `Tensor @ Tensor` with a concrete reduce dim → separate
   kernels, HBM round-trips between q·k / softmax / PV, and (at decode shape) **register-tiled, no LDS, no `v_dot2`**.
2. **`.set/.after` register-array fusion** — fuses online softmax into one kernel but emits a **scalar** hand reduction
   (0 `v_dot2`, 0 LDS), the `flash_partial` class.

llama's `flash_attn_tile` is **both at once** in **one** kernel (LDS-staged K/V + dense `v_dot2` + register online
softmax). **No tinygrad path expresses that** — it is `FUSED_FLASH_CONCRETE_GATE_NEEDS_AMDGCN_ESCAPE` territory (a
raw AMDGCN/HSACO kernel), not a bounded tinygrad build. The concrete gate's measured `FAIL_LOCAL_AB` is the **closest
expressible realization losing**, and the codegen dump is the **direct evidence** that the inexpressible object is the
real prize. Both bounded **and** concrete-shape decode levers are now exhausted.

---

## Acceptance gates

| gate | result |
|---|---|
| G1 design note maps candidate to FlashAttention / Flash-Decoding / FlashDecoding++ (+FlashInfer) | PASS (Phase 0 table) |
| G2 concrete-shape design documented | PASS (shapes/ranges/mapping/workgroups) |
| G3 µkernel runs or precise expressibility blocker | PASS (runs; register-tiled-not-LDS blocker named + dumped) |
| G4 correctness measured | PASS (rel_rmse 4.9e-4) |
| G5 local A/B vs `gqa_coop_vec` measured (correctness passed) | PASS (0.965× concrete authority; 1.42× symbolic flagged as artifact) |
| G6 candidate through decode_eval/lifecycle | PASS (`FAIL_LOCAL_AB`, match=True; refutation banked) |
| G7 no model/default/W==D route | PASS (`git diff tinygrad/` empty; no model route; no W==D) |
| G8 no closed lane reopened | PASS (the concrete first gate was explicitly authorized & never run; not the symbolic matmul-PV nor the raw fused tile) |
| G9 policy guard passes | PASS (run pre-commit) |
| G10 tree clean after commit / unrelated dirty listed | PASS (commit below; pre-existing unrelated dirty `structure/.../performance-primitive-research-principles.md`) |

## Decision enum
**`FUSED_FLASH_CONCRETE_GATE_FAIL_LOCAL_AB`** (the candidate ran, was value-correct, and lost the strict same-shape
local A/B at 0.965×). The deeper root cause — the true one-kernel LDS-tiled `v_dot2` fused tile is inexpressible
without an AMDGCN escape hatch — is documented as the supporting finding (it would be
`FUSED_FLASH_CONCRETE_GATE_NEEDS_AMDGCN_ESCAPE` if pursued).

## Next action — **`REST_DECODE` + v2 fallback** (`POST_MATMUL_PV_REST_DECODE_V2`)
The cheap, decisive fused-flash first gate has now been **run** and **failed**: even with the matmul-PV symbolic-split
blocker removed by fixing the shape, the closest expressible LDS-tiled-building-block realization only **ties**
`gqa_coop_vec` (0.965×), because tinygrad emits register-tiled (not LDS/`v_dot2`) matmuls at the decode shape and the
flat-GEMM under-utilization + extra layout kernels offset the benefit. Decode is therefore **capped at tinygrad's
current backend ceiling** with a concrete, counter+codegen-level reason. Per the strategic scope, the recommendation
**collapses to `POST_MATMUL_PV_REST_DECODE_V2`**: rest decode, pivot to the v2 / lifecycle-search / tooling
consolidation (Phase 5 sketch in the scope). Reopen decode only if (a) a renderer that emits **LDS-tiled fused
reductions** lands, or (b) an **AMDGCN/HSACO escape hatch** for the single fused tile is funded (default-off,
shape-guarded, same W==D + quality gates), or (c) new timed evidence overturns a closure-table verdict. The llama
oracle stays the validated, non-promotable target.

## Changed files
`extra/qk_fused_flash_concrete_gate_ab.py` (new), `bench/qk-fused-flash-concrete-gate/` (artifacts),
`bench/qk-decode-eval/candidates.json` (+`fused_flash_concrete_gate`), `bench/qk-lifecycle-search/refutations.json`
(+`fused_flash_concrete_gate_register_tiled_not_lds`), this doc, handoff/READMEs.

## Boundary
No `tinygrad/` change, no model route/default, no W==D route, no closed lane reopened, no tuning sweep, no
weak-baseline benchmarking (the symbolic-comparator 1.42× is explicitly flagged as a concreteness artifact, not a win).
Clock-pinned diagnostic; perf-state restored to `auto`.
