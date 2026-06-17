# 8B flash-decode `flash_partial_v2` register-blocking — REFUTED (2026-06-17)

Follow-up to the shipped flash-variant win (`qk-8b-flash-variant-result-20260617.md`). Continuation of the
*same method* — dominant kernel → inspect redundancy → legal variant → isolated search → in-model gate — applied
to the new top flash kernel after the `hoisted` ship. **Outcome: refuted at the isolated/projection gate. Not
shipped. No code change** (the variant lived only in a throwaway prototype).

Hardware: RX 7900 XTX (gfx1100), Qwen3-8B-Q4_K_M. See provenance note in the flash-variant result doc.

## Phase 0 — anatomy after `hoisted`

Re-profiled the shipped `hoisted`/L128 flash-decode (eager, relative proxy). The new top flash kernel is
**`flash_partial_v2` itself**; everything else is small:

| ctx | flash_partial_v2 | other flash kernels (prob/combine/max/den/gmax) |
|---|---|---|
| 1024 | 20.4% of decode GPU | ~0.5–1.3% each |
| 4096 | **47.5%** (single biggest kernel, > GEMV) | ~0.4–2.2% each |

So there is no second cheap redundancy in the combine/prob/den/gmax kernels — the only candidate is
`flash_partial_v2`'s own structure.

## Phase 1–3 — the one legal lever: register-blocking `d`

`flash_partial_v2` is a pure weighted-sum reduce `pout[h,s,d] = Σ_t prob[h,t]·v_aug[t,d]` with the output dim
`d` spread across `W=Hd+1=129` **GLOBAL** lanes (one output per thread, prob loaded once per lane).
Bandwidth math: at ctx4096 it moves ~32MB V (4× GQA-redundant over 8MB) at ~33 GB/s effective — but V (8MB)
fits the XTX's 64MB Infinity Cache and prob (16KB/head) fits L1, so it is **not HBM-bound; it is
cache/occupancy/latency-bound**. The L-sweep was already near-flat (L64≈L128≈L256), so it is not
serial-chain-latency-bound either.

The remaining untested, legal lever (no LDS, no coupled-reduce wall): **register-block `d`** — each thread owns
`BD` output dims (BD independent accumulators = more ILP, prob read once per key and reused across the block).
Isolated device `tm` (concrete-shape prototype, vs the shipped `flash_partial_v2`):

| ctx | v2 (BD=1) | BD=2 | BD=4 | BD=8 |
|---|---|---|---|---|
| 1024 | 154µs | **145µs (1.07×)** | 154µs (1.00×) | 167µs (0.92×) |
| 4096 | 591µs | **546µs (1.08×)** | 566µs (1.04×) | — |

- **BD=2 wins only ~7–8% on the kernel; BD≥4 regresses** — fewer threads hurt this occupancy-sensitive kernel.
  The prob-load redundancy I hypothesized is L1-cached, so removing it barely helps. Confirms cache/occupancy
  binding, not redundant-work binding.
- **Projected decode translation** (kernel share × kernel win): **~+1.4% @ctx1024** (below the ≥3–5% gate),
  **~+3.8% @ctx4096** (marginal, context-specific only).
- **Correctness needs codegen-idiom work:** the multi-output register reduce hits the GLOBAL-range vs
  `c_regs`/`UOp.special` idiom mismatch (NaN from the multi-element `.set` init) — "codegen-convention, not
  kernel-authoring" (same wall class as WR4 / SHAPED_WMMA revival). Timings above are still representative (the
  kernel does the work, just from a bad init), so the perf verdict stands without resolving correctness.

## Verdict — REFUTED (kill gate)

| signal | reading |
|---|---|
| BD=2: 1.07–1.08× kernel-only | small isolated gain |
| BD≥4: regresses | occupancy/cache-bound, not redundant-work-bound |
| projected decode ~1.4% @ctx1024 | below gate |
| projected decode ~3.8% @ctx4096 | marginal, context-specific |
| correctness needs codegen idiom work | complexity exceeds value |
| hoisted-exp already captured the redundancy | remaining path is not a bounded variant |

**`flash_partial_v2` no longer has cheap redundant-work wins.** The exp-hoist ship already removed the structural
redundancy in this path. The next attention win requires a **different primitive shape — high-occupancy
WMMA + cooperative GQA V-reuse via LDS** — not register-blocking. That is a separate, harder `[codegen]` arc
(the documented wall: warp/WMMA flash, `extra/gemm/amd_flash_attention.py`), not a bounded variant of the
current kernel. Not pursued here.

## Lasting takeaways

- The bounded-primitive-search method correctly **refuted fast** when the dominant kernel's binding shifted from
  redundant-work (hoist-able) to occupancy/cache (needs a new primitive shape). That is the kill gate working.
- Diagnostic that generalizes: if smaller-`L` (more splits) and register-blocking both fail to move a reduce
  kernel, it is occupancy/cache-bound, and the lever is a different memory-locality primitive, not a knob.
