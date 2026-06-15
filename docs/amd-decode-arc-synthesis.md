# The AMD decode arc — a synthesis through the lens of primitives

Date: 2026-06-15. Hardware: RX 7900 XTX (gfx1100), HBM peak 859 GB/s. Model: Qwen3-8B Q4_K_M.
Bar: llama.cpp = 105.7 tok/s (57% of peak) on this exact GPU.

This is the capstone for the decode-optimization arc. It is organized around the **primitive** — the unit
that turned out to be the whole story. The headline: tinygrad decode went from **22% → ~48–50% of llama.cpp
with byte-identical output**, and the lever was not a faster kernel but **completing primitive coverage**.

---

## 0. What a "primitive" is here

A *decode primitive* is a hand/machine-specialized GPU kernel for one quantized matmul shape, wired in to
replace tinygrad's generic auto-generated reduce for that op during batch-1 decode. It has three parts:
- a **kernel** (e.g. `q4k_gemv_partial`, `q6k_gemv_partial`) that unpacks the quant format and does the dot;
- a **policy** (`_q4k_policy`, `_q6k_policy`) that decides *which* tensors get a primitive and with what opts
  (LOCAL/parts) — this is the machine-search output, the learned coverage map;
- an **install path** (`_install_q4k_primitives`, `_install_q6k_primitives`) that swaps the module in and
  manages repacked storage (Q4_K is 144 B/block = 36 uint32 words; Q6_K is 210 B/block = 105 uint16 halfs).

Everything below is a story about the gap between "a primitive exists" and "a primitive is *applied to the
op that matters*."

---

## 1. The kernel question — answered, with a win

The mission asked: can machine search produce a decode GEMV kernel competitive with llama.cpp? **Yes — it
exceeds it.** Measured cold (>96 MB Infinity Cache working set), launch-amortized, at forced full memory
clock, the int-dot (`v_dot4`/`udot4`) Q4_K GEMV sustains:

| kernel (standalone, full clock) | % of HBM peak |
|---|---:|
| our int-dot Q4_K GEMV | **76%** |
| our fp-dequant Q4_K GEMV | 56% |
| llama.cpp end-to-end | 57% |

So at the kernel level the primitive is *better* than the reference. This is banked and independent of
everything that follows. (`bench/.../KERNEL_BEATS_LLAMACPP.md`.)

## 2. The measurement reckoning — three confounds that nearly wrote false conclusions

Getting that number trustworthy required killing three confounds, each of which had produced a wrong
conclusion earlier in the arc (`amd-decode-measurement-confounds`):
1. **Infinity Cache (96 MB)** — small working sets read from L3, not HBM → inflated bandwidth.
2. **Launch overhead** — few reps of a small kernel → deflated bandwidth.
3. **Memory-clock ramp (96→1249 MHz over ~4 s)** — short benchmarks run under-clocked; this *faked* a
   "small kernels can't saturate" size-scaling that vanished once the clock was forced high.

The discipline lesson: on this GPU, no bandwidth number is trustworthy unless it is cold, launch-amortized,
**and** at forced full clock. Several intermediate docs (NARROW_RESULT, the old V1) were corrected because
they violated one of these.

## 3. Localizing the e2e gap — it is primitive COVERAGE, not kernel quality

With the kernel proven good, why was end-to-end decode still 22% of llama.cpp? A per-kernel profile of one
clean decode token (`bench/.../BREAKDOWN_RESULT.md`) settled it:
- the token is ~20 ms GPU-busy + ~25 ms host/sync;
- of the GPU work, **one kernel — `r_32_32_4_48` — was 59%**, running at ~38 GB/s (~4% of peak).

That kernel was **not a primitive**. It was the generic fp-dequant reduce for the **Q6_K `ffn_down`** matmul.
The GGUF told the real story: Qwen3-8B Q4_K_M is **mixed-quantization** —

```
attn_q/k/output, ffn_gate/ffn_up : Q4_K (all 36 layers)  -> primitive existed and was applied -> fast
ffn_down : Q6_K x18, Q4_K x18     attn_v : Q6_K x18, Q4_K x18     output(lm_head): Q6_K
```

The Q4_K primitives ran fine (31–54%, near llama). But **every Q6_K matmul fell back to the slow reduce**,
because the install path only installed Q4_K primitives. The dominant decode cost was a *coverage hole*, not
a slow kernel. (The k/v-coverage ablation that returned null was the clue that sharpened this: attn_k is
Q4_K-and-tiny, attn_v is half Q6_K — the cost was Q6_K `ffn_down`, which no primitive touched.)

## 4. The fix — completing primitive coverage

The Q6_K primitive was **already fully built** (`extra/q6_k_gemv_primitive.py`, `Q6KPrimitiveLinear`,
`_install_q6k_primitives`, `_q6k_policy`, plus a self-test) — and gated behind a `Q6K_PRIMITIVE` flag that
nothing set. The fix was to make the primitive path *complete by default*:

- **Q4K_PRIMITIVE now implies Q6K_PRIMITIVE** (opt out with `Q6K_PRIMITIVE=0`). Q6_K dequant is exact, so
  this is pure speed with **byte-identical output** (a 20-token greedy continuation matches the pure-fp
  baseline exactly; unpack self-test max_abs = 0).

| config | tok/s | of llama.cpp |
|---|---:|---:|
| Q4_K primitive only (Q6_K falls back) | 23.1 | 22% |
| **+ Q6_K primitive (now default)** | **50.8** | **48%** |
| + attn_v + lm_head (`Q6K_COVER_MORE`) | 53.1 | 50% |

`r_32_32_4_48` is gone; the Q6_K `ffn_down` is now `q6k_gemv_partial` at 16% (was 59%). Two supporting fixes:
a `_set_module_at` top-level-path bug (blocked the lm_head), and an empirical refutation of the stale
"attn_v/output lose to the fused graph" comment (they now win +5%, gated pending 14B/32B validation).

## 5. The synthesis — what the arc says *about primitives*

1. **Coverage dominates kernel quality at the margin.** A 76%-peak kernel applied to 80% of the matmuls
   beats a 100%-peak kernel applied to 60% of them. The single biggest decode win in this whole arc came
   not from a faster kernel but from *applying an existing kernel to the op that was 59% of the work*. The
   first question for any quantized model should be "what's the quant-type histogram, and is every dominant
   shape covered?" — not "how fast is my best kernel?"
2. **Mixed quantization is a coverage trap.** Q4_K_M is not "Q4_K" — it sprinkles Q6_K on the
   perplexity-sensitive tensors (ffn_down, attn_v, lm_head), which happen to be large. A primitive system
   keyed on one format silently leaves the heavy tensors on the slow path. Coverage policy must be
   format-complete, not format-assumed.
3. **The policy is the machine-search artifact.** `_q4k_policy`/`_q6k_policy` are the learned map of
   (tensor → kernel + opts). The arc's value is less "a fast kernel" and more "a verified coverage map," and
   its failure mode is a stale entry (the "lose to fused graph" comment that was no longer true).
4. **Exactness changes the calculus.** Because Q6_K dequant is lossless, completing coverage carried zero
   accuracy cost — so it could be made the default, not an opt-in experiment. Primitives that are exact
   should be on by default; only lossy ones (int8-activation paths) stay gated.
5. **Measurement honesty is load-bearing.** Every wrong turn in this arc (kernel-is-12%, dequant-caps-at-24%,
   small-kernels-don't-saturate) came from an uncontrolled confound, and every correction came from
   profiling the *real* in-graph kernel rather than a proxy. The primitives only got fixed because the
   profile got honest enough to name `r_32_32_4_48` and trace it to a quant type.

## 6. What remains — and it is no longer a primitive problem

After completing coverage, the decode token is ~no-longer-GEMV-bound:
- **~25 ms/token host/sync** (per-token `.item()` CPU round-trip, argmax over 151,936 vocab, replay launch).
  llama.cpp's tight loop has ~0 here. This is now the largest single lever.
- the Q6_K **lm_head** and the **attention reduces** — smaller, and partly addressed by `Q6K_COVER_MORE`.

These are loop/structure problems (fewer host syncs, on-GPU sampling), not kernel or coverage problems. The
primitive question is, for this model, essentially closed: every dominant matmul now runs on a primitive
that meets or beats llama.cpp's kernel.

## Ledger (corrected claims, for anyone reading the older docs)
- "in-graph Q4_K GEMV runs at 12%" (NARROW_RESULT) — **wrong**; measured a non-decode fallback. Real: 31–54%.
- "weight read is 95% of the token" (old V1) — **wrong** (circular); real weight read ≈ 16% once kernels are
  fast, the rest is host + non-GEMV.
- "small per-layer kernels can't saturate" — **wrong**; a clock-ramp artifact.
- "attn_v/output lose to the fused graph" — **no longer true** on this GPU (+5%).
- Standing and correct: the standalone int-dot kernel beats llama.cpp (76% vs 57%); Q6_K coverage was the
  dominant e2e lever; the decode gap is now host overhead, not the GEMV.

Source docs: `bench/amd-decode-flywheel-proof-20260614/{KERNEL_BEATS_LLAMACPP, prefetch-gemv/{PERLAYER_RESULT,
BREAKDOWN_RESULT, Q6K_FIX_RESULT}}.md`. Memories: `amd-decode-{kernel-beats-llamacpp, measurement-confounds,
real-bottleneck}`.
