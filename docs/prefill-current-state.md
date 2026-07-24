# 8B Prefill Current State

This is the compact authority for the shipped Qwen3-8B gfx1100 prefill route. Historical scopes and failed benchmark
banks live in Git history, not on the active repository surface.

Last updated: 2026-07-24 (aligned three-way benchmark + fused-attention contribution added; original pinned authority below unchanged).

## Shipped route

- Route: `prefill_wmma_lds_dbuf_generated`.
- Status: promoted default for the exact pp512 roles `attn_qo`, `attn_kv`, `ffn_down`, and `ffn_gate_up`.
- Ownership: ordinary tinygrad matmul lowered from a typed `KernelCandidateContext`; there is no hand-emitted pipe/LDS2
  kernel on the runtime path.
- Pipeline: fp16 operands, fp32 accumulation, two LDS slots, 256 threads, and exact role/shape/target admission from the
  canonical candidate set.
- Device: normal `DEV=AMD` / HIP renderer. `DEV=AMD:ISA` is optional compiler-analysis tooling and is not imported or
  selected by production execution.
- Rollback: `PREFILL_GRAPH_GEMM=0` returns to the ordinary tinygrad scheduler. Retired raw-kernel selectors fail loud.

## Durable evidence

The only promotion evidence retained in-tree is under
`bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/`:

- `candidate-set.json`: four exact admitted payloads and canonical identities.
- `whole-model-quality.json`: PASS, three deterministic greedy cases, baseline/candidate token parity, route bound, and
  healthy GPU after both isolated children.
- `whole-prefill-pinned.json`: clean authority at commit `8045efcef`, pinned pp512 `3561.32 tok/s`, all four candidate
  identities observed, no missing or unexpected bindings.

The current raw-versus-practical placement is owned by BoltBeam in
`BoltBeam/docs/qwen3-8b-current-dual-roofline-20260713.md`. The retired
`bench/qk-prefill-theoretical-ceiling/latest.json` missing-evidence placeholder is intentionally removed.

## Aligned benchmark: fused vs pre-fused baseline + vs llama.cpp (2026-07-24)

Same device (RX 7900 XTX / gfx1100, ROCm), **unpinned**, guard-serialized, best-of-bursts.
8B only (14B tinygrad blocked by the packed-WMMA canary GPU HW-fault, see
`BOLTBEAM_GPU_HANG_DIAGNOSIS_HANDOFF_20260724.md`; run under
`TINYGRAD_PREFILL_PACKED_WMMA=0`, keeping 8B on graph-GEMM). The correct "what did fused
add" baseline is the **pre-fused commit `533c0aa00` (07-21, last before the flash
campaign), measured fresh on the same hardware** — NOT a fused-off patch. (An earlier
draft of this section used `_should_use_custom_kernel_prefill_attn=False`, which lands on a
pathologically slow SDPA path that collapses to DNF at long context and grossly INFLATED
the fused contribution to "+61% → +319%". That column is RETRACTED as a measurement error.)

| pp   | 07-21 pre-fused | current fused | fused gain | llama (best) | fused vs llama |
|------|-----------------|---------------|------------|--------------|----------------|
| 512  | 3461            | 3623          | +4.7%      | 3571         | +1.5%          |
| 1024 | 3223            | 3515          | +9.1%      | 3470         | +1.3%          |
| 2048 | 2819            | 3314          | +17.6%     | 3338         | −0.7%          |
| 4096 | 2213            | 3026          | +36.7%     | 3160         | −4.4%          |

Findings (honest, corrected):
1. **No regression.** Current fused ≥ the pre-fused baseline at every context (07-21 3461 →
   fused 3623 at pp512, +4.7%). At pp512 attention is not the bottleneck, so fused only
   nudges the whole-model number (~+5%) — which is why "the pp512 number barely moved" is
   the correct observation, not a regression.
2. **Fused's real contribution grows with context: +5% (pp512) → +37% (pp4096)** as the
   pre-fused attention decays (3461 → 2213, −36%). Real flash-attention thesis, but MODEST.
   The retracted "+61% → +319%" was measured against a crippled fused-off SDPA arm and is
   WRONG.
3. **Fused's biggest actual value is closing the long-context gap to llama.** Pre-fused was
   **−30% behind llama** at pp4096 (2213 vs 3160); fused pulls that to **−4.4%** (3026).
   Short context we edge llama (+1.5%/+1.3%); long context llama still edges us
   (−0.7%/−4.4%) — a crossover.
   **CORRECTION (2026-07-24, per-kernel trace):** the earlier claim here — "the residual
   long-context deficit is the graph-GEMM / per-chunk path, not attention" — is WRONG for
   the *intra-model* context decay. A `PROFILE=1` per-kernel trace (8B, gfx1100, start_pos
   0→3584; artifact `/tmp/...tax_trace_final.json`, method below) shows the per-chunk
   decomposition is **GEMM-bound and flat with context** (QKV/O + FFN ≈ 470 ms/chunk,
   93%→74% of per-chunk time) while **fused attention is the ONLY component that grows with
   KV** — 4% (pp512) → 23% (pp4096), ~8×. So our own throughput decay (3564→2976 tok/s,
   −16.5%) is **entirely attention's KV-length scaling**, not GEMM. GEMM owns the *baseline*
   ceiling (~3564 tok/s); attention owns the *long-context* decay. **Caveat:** this
   attributes our own decay, not the gap vs llama — that needs a llama-side per-kernel trace
   (in progress). But it makes our attention KV-scaling the prime suspect for the vs-llama
   long-context crossover, the opposite of the retracted "not attention" claim.
   **Next levers, ranked:** (1) fused-attention KV-scaling (owns the long-context deficit);
   (2) per-layer GEMM efficiency (owns the ~3564 baseline ceiling).
   *Trace method:* `PROFILE=1 TINYGRAD_PREFILL_PACKED_WMMA=0 prefill_whole_synced.py --mode
   authority --logits-only` populates `device_profile[sp].by_name` (per-kernel `device_ms`);
   exclude the `r_16_2374_*` kernel (the 512×151936 LM-head is a `--logits-only` artifact,
   absent from real generation). Non-`--logits-only` gives representative throughput but the
   profiler returns 0 events on the argmax path.
4. **Reconciliation:** unpinned-best fused 3623 ≈ pinned 3561 ≈ llama-best 3571 (all
   best-case ~3.6k). The historical "≈145% of llama" was an artifact of the invalid
   ~4408/4413 tinygrad number (see below) plus an un-artifacted "~3050" llama estimate; the
   real aligned figure is near-parity-to-slightly-ahead short, slightly behind long.

## Closed branches

- Historical `~4413` and recreated `~4099` pipe results were invalid: leaked LDS geometry launched only 1/16 of the
  pipe-owned output. Correcting geometry and buffer effects produced parity but only `2095.70 tok/s`.
- The raw LDS2 oracle hung the GPU and is not benchmark-eligible.
- The old S9/S10, hybrid, raw pipe/LDS2, single-buffer, and environment-driven local-stage experiments are deleted from
  the active tree. Their conclusions remain in `docs/prefill-lessons-ledger.md` and Git history.

## Change rule

A replacement must supply an exact typed candidate, isolated correctness and GPU-health evidence, whole-model parity,
route-census identity, and pinned timing before it can replace the current default. A faster incomplete or unverified
kernel is not a performance authority.
