# Sub-4-bit decode arc — scope & gates (2026-06-17)

Can demoting Q4/Q6 bulk weight tensors to **Q3_K / Q2_K** beat the banked Q6→Q4 decode frontier without
breaking quality? This is dangerous (new quantizer + packed layout + GEMV kernel + per-tensor policy), so the
arc proves **byte frontier → reconstruction error → dNLL quality** FIRST, and only builds a kernel if a real
candidate passes. `[test]`/`[nn]` with explicit kill gates. **Non-goals:** no model.py edits, no new runtime
flags, no GPU kernel, nothing default-on until Phase 3 accepts a candidate.

## Why the Q6→Q4 frontier is tapped

`bench/qk-demote-search/search.json`: ffn_down Q6→Q4 accepted (+14% tok/s, dNLL ≈ +0.0005); attn_v Q6→Q4
accepted (no throughput gain); lm_head/output Q6→Q4 **rejected** (dNLL +0.051 ≫ 0.01). Decode rests ~64 tok/s
(~63% of llama). The only remaining read-fewer-bytes lever is sub-4-bit demotion of the Q4/Q6 bulk.

## Byte frontier (Phase 1, `bench/qk-sub4-byte-census/`, Qwen3-8B-Q4_K_M)

Total weights 5022 MB; **decode-bandwidth (read fully per token) 4670 MB**. By share of decode bandwidth:

| role | qtype | MB | % decode bw | Q3_K MB | Q2_K MB | notes |
|---|---|---:|---:|---:|---:|---|
| ffn_down | Q6_K | 1253 | 26.8% | 779 | 595 | already Q6→Q4-demotable; big |
| ffn_gate | Q4_K | 1019 | 21.8% | 779 | 595 | bulk |
| ffn_up | Q4_K | 1019 | 21.8% | 779 | 595 | bulk |
| lm_head | Q6_K | 510 | 10.9% | 267 | 204 | **Q6→Q4 already rejected (dNLL)** |
| attn_output | Q4_K | 340 | 7.3% | 260 | 198 | |
| attn_q | Q4_K | 340 | 7.3% | 260 | 198 | |
| attn_v | Q6_K | 104 | 2.2% | 65 | 50 | already demoted |
| attn_k | Q4_K | 85 | 1.8% | 65 | 50 | |
| embedding | Q4_K | 350 | (0%) | — | — | gathered (1 row/token), NOT bw-relevant |

**Top-3 sub4 targets: ffn_down, ffn_gate, ffn_up (~70% of decode bandwidth combined).**

## The ceiling is modest (set expectations)

Ideal (pure HBM-bandwidth ratio): all-bw-roles → Q3K **1.436×**, → Q2K **1.881×**. But decode is ~55% host
overhead + ~45% GPU, and the GPU side runs ~76% HBM peak. Realistic wall ceiling (Amdahl, ideal kernel):
- all→Q3K: GPU bytes ×0.70 → wall ≈ 0.55 + 0.45·0.70 = **~1.16× wall**.
- all→Q2K: GPU bytes ×0.53 → wall ≈ 0.55 + 0.45·0.53 = **~1.27× wall**.
And that assumes (a) quality holds, (b) the sub4 GEMV is as bandwidth-efficient as Q4 (unpack overhead could
erase it). So sub4's best plausible decode payoff is ~1.1–1.25× wall — worth it only if quality passes cleanly.

## Candidate formats

- **Q3_K**: 110 B / 256 = 3.4375 bpw (vs Q4_K 4.5). Prototype proxy: per-32 sub-block asymmetric (scale+min)
  via the generalized qkx2 search (nmax=7), fp super-scale. Real Q3_K is finer (16-elem sub-blocks + 6-bit
  super-scale); the proxy is the right ballpark (if proxy passes dNLL, real Q3_K passes).
- **Q2_K**: 84 B / 256 = 2.625 bpw. Proxy nmax=3.

## Quality gates

dNLL (teacher-forced, fixed calibration) vs the current model: **accept ≤ 0.01, borderline ≤ 0.02 (maybe),
reject > 0.01**. A candidate is worth a kernel only if dNLL ≤ 0.01 AND projected byte saving ≥ 5% of decode bw.

## Kernel risks (deferred to Phase 4+)

New quantizer + packed layout + a Q3/Q2 GEMV that beats Q4's bytes/sec (unpack cost: 3-bit packing crosses byte
boundaries — more ALU per byte than Q4's nibbles). Kill if unpack overhead erases the byte saving or the kernel
is slower than Q4.

## Kill conditions (stop the arc)

- byte census shows only tiny savings — **partly true: ceiling ~1.16–1.27× wall, modest.**
- offline Q3/Q2 reconstruction error obviously huge (Phase 2).
- fake-dequant dNLL rejects all meaningful candidates (Phase 3) → **bank sub4 as quality-refuted.**
- (later) GPU kernel slower than Q4 / e2e win < noise.

## Result & closeout — REFUTED (Phase 3, dNLL fake-dequant)

Q2 was killed at Phase 2 (reconstruction ~0.36 rel). Q3 single-role dNLL (`bench/qk-sub4-nll/search.json`,
fake-dequant into the dense path, teacher-forced):

| role | qtype | dNLL | decode bw | verdict |
|---|---|---:|---:|---|
| ffn_down | Q3 | **+0.0281** (3-win) | 12.8% | reject |
| ffn_gate | Q3 | **+0.0216** | 5.1% | reject |
| ffn_up | Q3 | **+0.0260** | 5.1% | reject |
| attn_output | Q3 | **+0.0402** | 1.7% | reject |
| attn_q | Q3 | +0.0006 | 1.7% | quality-ok but **<5% bw** |

**No candidate passes dNLL ≤ 0.01 AND ≥5% bandwidth saving → sub4 is quality-refuted. Do NOT build a Q3/Q2
GEMV kernel.** Every high-byte role rejects at Q3 (2–4× the 0.01 budget); the only quality-passing role (attn_q,
+0.0006) saves just 1.7% of decode bandwidth — not worth a kernel.

**Measurement note (mattered):** ffn_down Q3 looked like a *pass* at single-window (dNLL −0.0045 — implausibly
good for ~15% rel error), but 3 windows flipped it to +0.0281 — the single-window value was noise. Multi-window
confirmation caught a false positive; the gate is only trustworthy with multiple windows. (Q2 not dNLL-tested —
already refuted at reconstruction.)

**Why this makes sense:** Q3 adds ~2× the reconstruction error of the accepted Q6→Q4 demotion (Phase 2:
~0.15 vs ~0.072 rel), and dNLL is roughly quadratic in weight error, so ~4× the dNLL — the accepted Q6→Q4 was
~+0.0005, so Q3 landing at ~+0.02–0.04 is consistent. Sub-4-bit is simply past this model's quality cliff for
the bulk tensors. The cheap gate (no kernel) earned its keep: it refuted sub4 before any kernel/format work.

## Plan

Phase 1 census ✅ → Phase 0 (this doc) → Phase 2 offline Q3/Q2 quant error by role → **Phase 3 dNLL fake-dequant
gate → STOP & decide.** Phases 4–6 (reference format, GPU GEMV, gated policy) only if Phase 3 accepts a
candidate. Reuses: `gguf_load_with_metadata`, `extra/qk_quantize._make_qkx2` (generalized to N-bit),
`extra/qk_nll_eval`/`qk_prefill_v2_nll_eval` (dNLL), the `QK_DEMOTE_TENSORS`/policy machinery (for later).
