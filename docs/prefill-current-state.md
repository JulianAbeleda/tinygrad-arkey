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

## Aligned three-way benchmark: fused attention + vs llama.cpp (2026-07-24)

Same device (RX 7900 XTX / gfx1100, ROCm), **unpinned**, guard-serialized, both metrics
reported. `best` = fastest burst (tinygrad min-of-bursts / llama max sample); `avg` =
mean (tinygrad mean-of-bursts / llama `avg_ts` over `-r 5`). **8B only** — the 14B
tinygrad arm is blocked by the packed-WMMA canary GPU HW-fault (see
`BOLTBEAM_GPU_HANG_DIAGNOSIS_HANDOFF_20260724.md`); this run used
`TINYGRAD_PREFILL_PACKED_WMMA=0`, which keeps 8B on graph-GEMM (its fast path) and avoids
the fault. `tg-SDPA` = fused attention off (`_should_use_custom_kernel_prefill_attn`
patched False), identical graph-GEMM otherwise — so `tg-FUSED` vs `tg-SDPA` cleanly
isolates fused attention's whole-model contribution.

| pp   | llama best/avg | tg-FUSED best/avg | tg-SDPA best/avg | fused vs SDPA | tg-FUSED vs llama (best) |
|------|---------------|-------------------|------------------|---------------|--------------------------|
| 512  | 3571 / 3399   | 3623 / 3612       | 2247 / 2242      | +61%          | +1.5% (avg +6%)          |
| 1024 | 3470 / 3436   | 3515 / 3509       | 1631 / 1629      | +116% (2.16x) | +1.3% (avg +2%)          |
| 2048 | 3338 / 3330   | 3314 / 3311       |  790 /  789      | +319% (4.2x)  | −0.7%                    |
| 4096 | 3160 / 3158   | 3026 / 3021       | DNF (timeout)    | SDPA unusable | −4.4%                    |

Findings (honest):
1. **Fused attention is the big lever, and its whole-model win grows with context.** SDPA
   collapses quadratically (2247 → 1631 → 790 → DNF at pp4096) while fused holds
   (3623 → 3026). Whole-model contribution: +61% (pp512) → +116% (pp1024) → +319%
   (pp2048). The flash-attention thesis at the whole-model level.
2. **Vs llama.cpp it is a crossover, NOT a clean win.** tinygrad-fused is **ahead at short
   context** (pp512 +1.5%/+6%, pp1024 +1.3%/+2%) and **behind at long context** (pp2048
   −0.7%, pp4096 −4.4%). Both decay with length; llama decays less (−12% vs our −16% over
   512→4096). Since fused attention is ~33× SDPA at kv=4096, the long-context deficit is
   **not** attention — it is the graph-GEMM / per-chunk path losing ground to llama's
   kernels at length. That is the next optimization target.
3. **Reconciliation:** our unpinned-best 3623 ≈ pinned 3561 ≈ llama-best 3571 — all
   best-case, clustered ~3.6k; consistent. The historical "≈145% of llama" was an artifact
   of the invalid ~4408/4413 tinygrad number (see below) plus an un-artifacted "~3050"
   llama estimate; the real aligned figure is near-parity-to-slightly-ahead short, behind
   long. tinygrad also shows lower run-to-run variance (best≈avg) than llama (best≫avg).

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
