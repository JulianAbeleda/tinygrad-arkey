# 8B Decode Attention Primitive Search Arc — RESULT: SHIPPED (flash variant `hoisted`)

> **SUPERSEDED AS DEFAULT (still historically valid).** `hoisted` was the default until 2026-06-17, when
> `FLASH_VARIANT=gqa_coop` (cooperative GQA V-reuse) replaced it as the default — gqa_coop strictly dominates
> hoisted in-model (+3.9…+19.8% across ctx, byte-identical). Current authority:
> `qk-gqa-coop-decode-attention-result-20260617.md`. The hoisted result below remains correct (vs v1) and
> `FLASH_VARIANT=hoisted` is still a valid override.

Qwen3-8B-Q4_K_M, gfx1100, FLASH_DECODE on (ctx ≥ 512). Successor to the flash-threshold ship
(`qk-8b-attention-fusion-result-20260617.md`). This arc turned flash-decode from a single hand-written
kernel into a **searched primitive family** `decode_attention ∈ {v1, hoisted} × L∈{64,128,256,512}` and
shipped the winner as the default.

> **This is not a policy tweak — it is a searched primitive-family win.** The prior flash work tuned a
> *policy* (which kernel to call: SDPA vs flash, and at what ctx threshold). This arc changed the *primitive
> itself*: it defined a legal implementation space for decode attention, machine-searched it, and the search
> found **structural waste inside the existing flash kernel** (a 129× redundant `exp`) that a policy choice
> could never reach. The win is a new, faster member of the primitive family — not a different dispatch of the
> old one.

> **Hardware caveat (resolved 2026-06-17):** ROCm product string misreported the card as GRE; 24GB VRAM
> confirms RX 7900 XTX. Benchmarks are XTX, but compare only within the same harness/config. Provenance:
>
> | check | value | verdict |
> |---|---|---|
> | `rocminfo` Marketing Name | `AMD Radeon RX 7900 XTX` | XTX (from amdgpu driver, authoritative) |
> | VRAM total | 25,753,026,560 B = **23.98 GiB ≈ 24 GB** | **XTX (24 GB); GRE is 16 GB** — decisive |
> | tinygrad device / arch | single AMD device, `gfx1100`, PCI `0000:08:00.0` | — |
> | visibility env | `HIP_/ROCR_/CUDA_VISIBLE_DEVICES` all unset, 1 GPU | no wrong-device selection |
> | `rocm-smi` Card model | `RX 7900 GRE [XFX]` | **MISIDENTIFIED** — XTX/XT/GRE share Navi 31 dev-id `0x744c`; rocm-smi's SKU lookup misread it |
>
> So absolute "% of llama" comparisons on this host are valid. (The default-path tok/s below — e.g. ctx512 ≈ 43 —
> are not "low for an XTX": they are at long context. The campaign's ~64 tok/s figure is short-ctx (~ctx8);
> decode decays ~3.4× by ctx4096, and ctx512 ≈ 38–39 is the established XTX baseline, which matches `v1` here.)

## Verdict: SHIPPED — exact, monotone, grows with KV

Default flipped `v1`→`hoisted` (`model.py`) and `FLASH_L` 256→128. In-model W==D warm device-feed,
byte-identical greedy at every ctx:

| ctx | pre-arc (v1/L256) | shipped (hoisted/L128) | speedup |
| ---: | ---: | ---: | ---: |
| 512  | 39.0 | 43.5 | **+11.5%** |
| 1024 | 33.8 | 39.1 | **+15.7%** |
| 2048 | 27.0 | 32.7 | **+21.1%** |
| 4096 | 19.2 | 24.8 | **+29.2%** |

Clears the arc's success target at every ctx (ctx512 +5–10%, ctx1024 +5%, ctx4096 "preserve or improve").
Gain grows with KV because the optimized kernel's share of GPU time grows with context.

## Narrative correction (this changes the project conclusion)

The prior bank concluded: **"every bounded/local decode lever is shipped, refuted, or necessary — only deep
codegen remains; the 8B gap is GPU-kernel-structural."** That framing was about the *cross-layer program
granularity* gap (≈780 progs/token vs llama ≈260) and it treated the existing kernels as already efficient.

This arc shows that was **premature**. The corrected conclusion:

> **Bounded primitive search can still find structural waste *inside* existing kernels.**

`flash_partial` had been shipped and treated as done, yet it was recomputing a `d`-independent `exp` once per
output lane — 129× redundant work hiding in plain sight inside the dominant attention kernel. No policy knob,
GEMV tweak, or scheduler change could see it; a *primitive-level search that interrogates the kernel's own
decomposition* did. This is directly aligned with the project's first principles: **a performance primitive is
an operation plus its data movement / compute structure** — so the unit of optimization (and of search) must
be the primitive's internal shape, not just which primitive gets dispatched. Before declaring any path
exhausted, audit the dominant kernel's per-lane redundancy.

## What the win is (the anatomy → lever)

**Phase 0 anatomy (JIT/eager, relative proxy).** Of the 5 UOp flash kernels + score matmul, **`flash_partial`
is the entire attention cost** — 22.3% of total GPU @ctx512, ~30% @ctx1024 — while `flash_max/gmax/den/combine`
are ~2.6% combined. SDPA by contrast spends ~59% in `r_*` (QK^T/softmax/PV reduces); flash already beats it
(~9% less GPU @ctx1024), confirming the shipped threshold.

**The lever.** `flash_partial` ranged the output dim `d` over `W=Hd+1=129` as a **GLOBAL** axis, but the
softmax probability `p = exp(score[h,t] − m)` is **independent of `d`** — so the transcendental `exp` was
recomputed **129× redundantly** (once per output lane instead of once per key). `flash_partial` was
exp-throughput-bound.

**The fix (`hoisted`).** Compute `p` once per key in a new elementwise `flash_prob` kernel, then make
`flash_partial_v2` a pure weighted-sum (`Σ p·v_aug`, no exp). This is a **legal decomposition, not a coupled
multi-accumulator reduce** — so it does **not** hit the linearizer wall that forces the 5-kernel softmax split
(the documented kill gate was avoided, not fought). Cost: +1 kernel/layer (`flash_prob`, ~15µs), far
outweighed by `flash_partial` dropping ~43% (395→224µs @ctx1024 isolated).

## Gates

- **Isolated (Phase 2):** flash kernels 1.45–1.56× faster across ctx 512/1024 × L 256/128; max-err identical
  to v1 vs a numpy SDPA reference. Regression guard added to `extra/qk_flash_decode.py` `__main__`
  (v1 and hoisted both exact; `max|v1 − hoisted| = 0`, bit-identical).
- **In-model (Phase 3–4):** W==D device-feed (decode is GPU-bound, W≈D), byte-identical greedy at ctx
  512/1024/2048/4096. ctx<512 stays SDPA (unaffected). Unlike the dp4a standalone-win-that-died, this carries
  because decode is GPU-bound here and `flash_partial` was ~30% of it.

## The searched family + per-KV policy (Track 3)

`extra/qk_flash_variant_search.py` (worker+orchestrator, W==D per cell, exactness gate) over the grid.
Artifacts: `bench/qk-flash-variant-search/{flash-variant-search.json, accepted-flash-variant.json}`.

Full per-cell tok/s (exact gate PASS — every cell greedy-identical to v1):

| ctx | v1/L256 | ho/L256 | ho/L128 | ho/L64 | best |
| ---: | ---: | ---: | ---: | ---: | :-- |
| 512  | 39.0 | 42.0 | 43.5 | 44.0 | ho/L64 |
| 1024 | 33.8 | 37.9 | 39.0 | 39.3 | ho/L64 |
| 2048 | 27.0 | 31.9 | 32.6 | 32.5 | ho/L128 |
| 4096 | 19.3 | 24.6 | 24.8 | 24.4 | ho/L128 |

- **`hoisted` dominates `v1` at every ctx** — the big, monotone lever (chosen as default).
- **L is a marginal per-KV refinement** (~1–4%): `L=64` best for flash-active short ctx (<2048),
  `L=128` for long. `L=128` is the best single default (≥ `L=256` everywhere) → shipped. The full per-KV
  optimum (L64<2048, L128≥2048) is recorded in `accepted-flash-variant.json` for future ctx-dependent L
  selection (deferred: ~1% gain over flat L128, needs per-band JIT graphs).

## Changed

- `tinygrad/llm/model.py`: `FLASH_VARIANT` default `hoisted`; `FLASH_L` default 128.
- `extra/qk_flash_decode.py`: `flash_prob_kernel` + `flash_partial_v2_kernel` + `variant` param; `__main__`
  UOp-variant exactness self-test.
- `extra/qk_search_spec.py`: `SearchSpace.FLASH_VARIANT`.
- `extra/qk_flash_variant_search.py`: Track-3 primitive-family search runner.

## Next lever (deferred / future)

1. **Per-KV L selection in `model.py`** (ctx-dependent L via trace-time ctx, like `should_use_flash_decode`):
   ~1% over flat L128; costs a JIT graph per ctx band. Low priority.
2. **`flash_partial` further — register-blocking REFUTED** (`qk-8b-flash-partial-register-blocking-refuted-20260617.md`):
   `flash_partial_v2` is cache/occupancy-bound, not redundant-work-bound; register-blocking `d` gives only
   1.07–1.08× kernel (BD=2) and regresses at BD≥4 → ~+1.4% decode @ctx1024 (below gate), ~+3.8% @ctx4096
   (marginal). No cheap win remains; the next attention win needs a different primitive shape (high-occupancy
   WMMA + cooperative GQA V-reuse via LDS — a separate `[codegen]` arc), not a bounded variant.
3. **Phase B (decode-block fusion):** RoPE+KV-write+attention — the ~21 progs/layer → llama ~7 gap. Separate,
   harder arc (the linearizer wall is real there).
