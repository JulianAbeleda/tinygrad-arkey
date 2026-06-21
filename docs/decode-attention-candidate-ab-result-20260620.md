# Decode Attention Candidate A/B Result (Deliverable 2)

Date: 2026-06-20

Verdict: `NO_CHEAP_ATTENTION_CANDIDATE_PASSES` — the cheap, env-only attention levers all regress or fail to move
the dominant bucket. The default flash-decode path (`gqa_coop_vec`, `FLASH_L=128`, threshold 512) is already the
tuned optimum. The real lever (flash reduction / online-softmax-stat fusion) is a custom-kernel rewrite, specced
below and **deferred** (not a cheap candidate). Default decode behavior NOT changed.

## Target (from Deliverable 1)

Attention's dominant cost is `reduce_fixup` (flash-decode partial-reduction/fixup, 1.79 ms@1024 → 2.43 ms@4096)
plus `softmax_stats` (online-softmax kernels, grows fastest with ctx: 0.94 → 1.85 ms). `partial_compute` (the
QK·V flash compute) is small/flat (~0.8 ms). At ctx1024, reduce_fixup + softmax_stats = 2.73 ms = 78% of
attention. These scale with the number of KV chunks (`S = ceil(KV/L)`).

## Candidates tested (same-process / clean clock-pinned A/B via `extra/qk_decode_attention_cost_split.py`)

### C1 — Chunk policy tuning (`FLASH_L` 128 → 256 / 512)

Hypothesis: fewer/larger chunks → fewer partials to reduce → less reduce_fixup + softmax_stats.

| config | ctx4096 wall | ctx4096 tok/s | note |
|---|---:|---:|---|
| `FLASH_L=128` (default) | 16.45 ms | **60.8** | partial 0.90 / reduce 2.43 / softmax 1.85 |
| `FLASH_L=512` | 18.56 ms | 53.9 | partial **2.95** (parallelism collapse) / reduce 2.47 / softmax 1.79 |

Result: **regresses.** Larger L collapses `partial_compute` parallelism (fewer workgroups = lower batch-1 GPU
occupancy) faster than reduce/stat shrinks. `FLASH_L=256` showed the same trend. → default L=128 is optimal.

### C2 — Short-context SDPA bypass (`FLASH_DECODE=0`, force SDPA)

Hypothesis: at short KV, SDPA avoids flash's reduction overhead.

| config | ctx512 | ctx1024 | ctx2048 | ctx4096 |
|---|---:|---:|---:|---:|
| flash (default) | 68.0 | 66.5 | 63.5 | 60.8 tok/s |
| SDPA (`FLASH_DECODE=0`) | 14.3 | 10.4 | 5.3 | 4.3 tok/s |

Result: **catastrophic regress** (batch-1 SDPA has ~1% GPU occupancy at decode — the reason flash-decode exists).
The threshold-512 cutover is already correct; SDPA is never competitive at these contexts.

## Conclusion

No cheap candidate clears the local gate (≥0.5 ms/tok @1024 or ≥1.0 ms/tok @4096). The flash-decode policy
surface (variant, L, threshold) is already tuned. The recoverable mass (reduce_fixup + softmax_stats ≈ 2.7 ms) is
real but lives in the **number of separate flash kernels**: tinygrad's flash-decode emits ~8 kernels for the
online softmax (`flash_max`, `flash_den`, `flash_prob`, `flash_gmax`, `flash_combine` + partial reduces `r_*`),
where llama's flash_attn uses ~3 (`flash_attn_tile`, `stream_k_fixup`, `combine_results`). Decode is
GPU-execution-bound on many tiny kernels (D≈W, host-sync 0%), so the lever is **reducing kernel count by fusing
the online-softmax statistics + partial reductions** in `extra/qk_flash_decode.py`'s kernel generators.

### Deferred real candidate (build spec)

- Fuse `flash_max` + `flash_den` + `flash_prob` (+ `flash_gmax`) into one online-softmax-stat kernel per chunk,
  and fold the cross-chunk `combine` into the partial-reduce, in `extra/qk_flash_decode.py` (the `gqa_coop_vec`
  variant generators). Target: collapse ~8 attention kernels → ~3-4, recovering part of the 2.7 ms reduce+stat.
- Gate: same-process A/B ≥0.5 ms@1024 / ≥1.0 ms@4096, exact-greedy (flash is exact-vs-SDPA up to fp
  reassociation), then W==D promotion (≥3% @1024, ≥5% @4096, no ctx512 regress >1%).
- Effort: multi-day custom-kernel-generator rewrite — not a cheap env candidate. Recommended as the #1 decode
  build, but out of this session's bounded-candidate scope.

## Commands

```bash
# chunk policy A/B
FLASH_L=256 PYTHONPATH=. python3 extra/qk_decode_attention_cost_split.py --modes baseline --ckpts 1024 4096 ...
FLASH_L=512 PYTHONPATH=. python3 extra/qk_decode_attention_cost_split.py --modes baseline --ckpts 1024 4096 ...
# SDPA bypass A/B
FLASH_DECODE=0 PYTHONPATH=. python3 extra/qk_decode_attention_cost_split.py --modes baseline --ckpts 512 1024 2048 4096 ...
```

## Boundary

No decode default changed. Clock pinned for measurement; `auto` restored after (verified).
