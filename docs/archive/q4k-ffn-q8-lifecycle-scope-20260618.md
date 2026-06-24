# Primitive boundary: `q4k_ffn_mmvq_sudot4_with_q8_lifecycle` (2026-06-18)

Applying the Performance Primitive Research Principles to the MMVQ blocker. The prior `sudot4` work optimized a
*kernel*; the **primitive** is the whole chain, and it's the chain that failed whole-linear.

## The full primitive (not "the kernel")
A performance primitive = math + layout + **activation format** + memory path + work decomposition + lowering +
scheduling + **integration boundary**. For Q4_K FFN int-dot:

```
q4k_ffn_mmvq_sudot4_with_q8_lifecycle =
  fp activation source (RMSNorm output, fp16/fp32)
  → q8 pack: max(abs)/127 scale + quantize int8 + pack 4→uint32     [29.7µs, 4 kernels]
  → q8 scale layout: f32[IN/32], llama-q8_1-compatible
  → q8 reuse: shared across gate+up (auto-commoned by TinyJit)
  → packed Q4_K extract (v>>sh)&0x0F0F0F0F
  → signed dot4 (__builtin_amdgcn_sudot4 → v_dot4_i32_iu8 neg_lo)   [kernel 55µs / 57% peak]
  → qsum/min correction (_sdot4(0x01010101,q8)) + per-group scale + block d/dmin epilogue
  → in-kernel warp-shuffle + LDS reduce, one write
  → quality gate (q8-lossy, rel 0.006 → dNLL ≤ 0.01 required)
  → in-model integration (route gate/up behind a flag; W==D)
```

## Why "kernel-only" is NOT the primitive
The kernel saves ~11µs/linear (66.1→55.0). But the **activation-format conversion** (q8 pack, 29.7µs;
amortized ~14.9µs/linear over gate+up) is *part of the primitive* and costs more than the kernel saves. A faster
matvec that requires a more expensive activation format is not a faster primitive. The fp coop path's structural
advantage is that its activation format is the native fp — **zero conversion**.

## Why `sudot4` alone is closed until q8 is solved
`sudot4` is correct and fast at the kernel boundary (banked: `qk-dot4-isa-audit-20260618.md`). Re-tuning it
cannot change the whole-linear verdict, because the loss is entirely in the activation-format stage. Per the
research principles: **do not optimize the kernel again until the activation lifecycle clears its own gate.**

## Required gates (in order)
1. **Activation economics:** whole-linear (q8 pack included, paired gate+up) ≥1.15× fp coop AND ≥1.05× opaque.
2. **In-model:** W==D, ≥+5% decode ctx512/1024, no ctx4096/prefill regression.
3. **Quality (lossy path):** dNLL ≤ 0.01 (or an agreed budget) vs the byte-identical fp coop. Never default
   without it.

## Status
This scope frames the audit in `q4k-ffn-q8-lifecycle-verdict-20260618.md` (Phases 1-7). Supersedes/extends the
earlier `qk-q8-activation-lifecycle-*-20260618.md` with the graph-reuse + fused-pack prototypes the principles
demand.
