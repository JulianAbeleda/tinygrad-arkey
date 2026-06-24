# Q4_K MMVQ int-dot line — session closeout / bank (2026-06-17 → 2026-06-18)

Consolidated record of the MMVQ / int-dot investigation. RX 7900 XTX (gfx1100), Qwen3-8B-Q4_K_M. The line is
**complete and CLOSED**: one durable shipped capability (correct native signed dot4), the rest refuted with
quantified reasons. Shipped decode is unchanged at **~66-69% of llama** via the byte-identical coop + flash-decode
routes.

## SHIPPED / durable (banked)
| win | what | commit |
|---|---|---|
| **`_sdot4` → native signed dot4** | `_sdot4` renderer helper now lowers via `__builtin_amdgcn_sudot4(true,a,false,b,c,false)` → native `v_dot4_i32_iu8` + `neg_lo` modifier = correct a(signed)·b(unsigned). Fixes a latent bug: the prior bare-asm helper silently computed UNSIGNED×UNSIGNED. | `0adb9f55b` |
| **value-level dot4 test** | `test_sdot4_lowering.py` now checks the computed VALUE (signed×unsigned, incl negatives) + guards the unsigned regression + locks the sdot4-builtin scalar-fallback fact. 3/3 + 8/8 coop suite pass. | `d9be577d3` |

These were shipped EARLIER and remain the production decode wins (untouched this arc): MMVQ_COOP routes (Q6_K
lm_head + ffn_down, Q4_K attn_q/o), flash-decode threshold-512, hoisted-exp, gqa_coop_vec, PREFILL_V2.

## REFUTED with quantified reasons (banked, not shipped)
| arc | result | doc |
|---|---|---|
| MMVQ codegen / deep-linearizer / scale-hoist | dot4 emits fine; failing layer is scheduling, not lowering | `qk-mmvq-deep-linearizer-*`, `qk-mmvq-codegen-arc-*` |
| fused cooperative-row quadrant | ceiling ~53-54%, fails gate | `qk-mmvq-fused-coop-row-verdict-*` |
| llama-scheduler probe (128-thread/row) | shape correct + expressible; the fast 55% was a `_sdot4` *correctness artifact* (then fixed → 57% correct) | `qk-mmvq-llama-scheduler-probe-verdict-*` |
| sudot4 full-linear | kernel 57% correct (beats opaque 52%) but whole-linear 0.96× coop — q8 pack eats it; also lossy | `qk-mmvq-sudot4-full-linear-arc-*` |
| q8 activation lifecycle | reuse ceiling=2 (gate+up only) + per-kernel floor ~7µs > 5µs break-even → int-dot FFN refuted | `qk-q8-activation-lifecycle-verdict-*` |

## Durable findings (the campaign knowledge)
- **RDNA3 dot4 ISA:** only `v_dot4_i32_iu8` (signed×unsigned, needs `neg_lo` modifier — bare asm defaults to
  unsigned) and `v_dot4_u32_u8`. No signed×signed. `__builtin_amdgcn_sdot4` (dot1-insts) scalar-fallbacks on
  gfx1100; **`__builtin_amdgcn_sudot4` is the native path** (= llama's RDNA3 `ggml_cuda_dp4a`).
- **llama uses NO native dot4 was wrong — llama uses sudot4** (native, via sudot4 builtin); its `mmvq.cu.o` is
  host-only so the device code lives in `libggml-hip.so`.
- **llama MMVQ scheduler:** 128 threads/row, 16 K-blocks parallelized across threads (no serial loop), in-kernel
  warp-shuffle (`__shfl_xor`) + small shared reduce, one write.
- **The q8-pack wall:** any int-dot path pays a q8 activation-quant cost (~7µs/kernel floor; reuse ceiling 2 for
  Q4_K) that ≈ the kernel speedup → the byte-identical fp coop (no pack) stays competitive. This sank dp4a,
  Family-A, and sudot4 alike.
- **Kernel-level Q4_K ffn_gate/up ladder:** base fp 41% · fp coop 48% (byte-identical) · opaque asm 52% · 8-thread
  sudot4 50% · 128-thread sudot4 **57%** · llama 70%. The 57→70 residual is per-thread codegen (clang vs
  custom_kernel) — tinygrad-internals territory.
- **Lesson banked:** a dot4 lowering test MUST validate the computed value, not just instruction emission.

## Only theoretical reopen (not pursued; standing 14B no-pivot honored)
q8 as a **zero-extra-kernel epilogue of the prior RMSNorm** → ~1.20× coop, all gates pass — but a deep
activation-lifecycle change, still q8-lossy (needs dNLL), best-case decode EV ~+3-4% (gate+up = 2 of 7
linears/layer). Scope separately only if a byte-identical or higher-EV motivation appears.

## Status: Q4_K int-dot FFN line CLOSED. Frontier within target = per-thread codegen (high-risk, internals).
