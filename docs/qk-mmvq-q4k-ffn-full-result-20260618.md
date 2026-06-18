# Q4_K ffn_gate/up full-MMVQ (Family A: q8_1 + dp4a) — REFUTED 2026-06-18

The earned Family-A build (q8_1 activations + `v_dot4_u32_u8` dp4a int-dot inside the coalesced coop structure +
fp affine epilogue). Built, correct, measured. **Verdict: REFUTED — the dot was never the bottleneck; the
limiter is the nibble + scale UNPACK ALU, which dp4a does not touch.** No defaults changed; the prototype kernel
was removed (refuted, unused). RX 7900 XTX, Qwen3-8B.

## What was built (Phase 3A/3B)

- Phase 3A: q8_1 activation pack = `q8_1_quantize` + `q8_1_bias_pack_u32_kernel` → 29.6 µs, reusable for gate+up.
- Phase 3B: `q4k_coop_q8_partial_kernel` — lane4→LOCAL coalesced; each lane packs its 4 nibbles → u32, reads the
  bias-packed q8 word, calls `v_dot4_u32_u8` (real dp4a), applies the per-group Q4_K affine to its partials
  (affine is linear in dot/qsum → per-lane partials sum correctly via stage-2 `.sum`). **Correct** (rel_err
  0.0062 = q8-quant level, cos 0.999985).

## Phase 3C — whole-linear gate: FAIL

| variant | µs | GB/s | % peak |
|---|---|---|---|
| base fp default | 78.0 | 363 | 40% |
| fp coop (prior) | 65.7 | 431 | 48% |
| READRAW (no dequant) | 44.8 | 632 | 70% |
| **dp4a-coop kernel (Family A)** | **80.0** | **354** | **39%** |
| q8 pack (amortized gate+up) | 29.6 | — | — |

- dp4a kernel = **39% peak — no faster than fp (40%), WORSE than fp coop (48%).**
- Whole-linear (q8 pack + kernel): **single 0.71×, pair (1 pack, gate+up) 0.82×** vs base — **slower.**
- Gate (≥1.3× whole-linear, q8 cost counted): **FAIL by a wide margin.**

## Why (the decisive correction to the Phase-1 audit)

READRAW (70%) proved the *memory schedule* can hit 70% — but the 40→70 gap is **NOT the dot MAC.** Real dp4a
(v_dot4) replaced the 4 int MACs/group and the kernel **did not get faster** (39% vs fp 40%). So the bottleneck
is the **format-mandated UNPACK ALU**: extracting 8×4-bit nibbles per word (shift/mask) + decoding the Q4_K
6-bit group scales (`_q4k_group_params`), done per weight/group. dp4a removes the *dot*, not the *unpack* — the
same structural wall as the Q6_K dp4a refutation. Worse, q8_1 adds a 29.6 µs pack and the dp4a path's inline-asm
CUSTOMI scheduled *worse* than fp coop.

Even an idealized dp4a kernel matching fp coop's 48% (the unpack-ALU ceiling) + amortized q8 pack would be
~0.96× pair — still below 1.3×. **The unpack ALU caps this role at ~48% peak (1.18× over base), below the gate;
no bounded kernel (dataflow OR dp4a) breaks past it.**

## Verdict

- **Family A (q8_1+dp4a): REFUTED** (whole-linear 0.82×; dp4a doesn't address the unpack ALU).
- **Family B (dataflow): REFUTED** earlier by the audit (coop = 48%).
- **Q4_K ffn_gate/up is unpack-ALU-bound at a ~48% ceiling** — the largest remaining decode role (44% of weight
  traffic) has **no bounded kernel lever**. llama's ~70% comes from a more ILP-efficient unpack that tinygrad's
  custom_kernel codegen does not reach via these primitives (a deeper codegen/ILP problem, not a kernel-shape
  knob).

## What remains after this arc

The MMVQ_COOP campaign shipped every role where the bottleneck was *coalescing* (lm_head/ffn_down Q6_K,
attn_q/o Q4_K → decode ~48%→~68% of llama). The roles where the bottleneck is the *unpack ALU* (Q4_K
ffn_gate/up, ffn_down; Q6_K split-K dp4a) are all refuted — dp4a/dataflow can't remove the format-mandated
unpack. **The remaining decode gap to llama (~32%) is unpack-ALU/codegen-ILP-structural, not a bounded primitive.**
Next levers are out of decode-kernel scope: prefill WMMA (different phase), 14B/32B (more GPU-bound), or a deep
codegen-ILP effort on the Q4_K unpack (very high risk).

## Comparison (Q4_K ffn_gate/up)
| | GB/s | % current tinygrad | % llama (~626) | % XTX roofline (900) |
|---|---|---|---|---|
| base fp | 363 | 100% | 58% | 40% |
| fp coop (best correct) | 431 | 119% | 69% | 48% |
| dp4a Family A | 354 | 97% | 57% | 39% |
| READRAW (roofline w/o dequant) | 632 | 174% | 101% | 70% |

## Files / commits
`extra/q4_k_gemv_primitive.py` (prototype added then removed — refuted), this doc (`[docs]`). Artifacts in
`bench/qk-mmvq-q4k-ffn-full/`. No `[nn]`/`[codegen]` retained (Family A refuted).
