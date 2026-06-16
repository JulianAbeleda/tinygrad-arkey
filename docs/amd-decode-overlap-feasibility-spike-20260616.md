# Decode overlap-feasibility spike — 2026-06-16

Phase 2 decision spike. **Question:** the decode token is a single-queue sum (GEMV 72% +
non-GEMV 28%); building a second compute queue to *overlap* the non-GEMV behind the weight
stream is major tinygrad infra and a prior session deferred it as "weight-read bound." Is the
HBM slack to overlap into actually real and reclaimable — i.e. should we build the infra?

**Method:** analytical byte/bandwidth accounting (rocprofv3 PMC counters don't work on
tinygrad's AMD backend — see `amd-decode-prefill-plan.md:219-232`), cross-checked against the
cli's measured `global_mem`/token and the existing named per-kernel profile
(`amd-decode-sequential-tax-profile-20260616.md`). Model: Qwen3-8B-Q4_K_M, RX 7900 XTX, HBM
peak 859 GB/s, default-on primitives (no demote).

## Byte ledger (per decode token) — reconciles to within 1.2%

| | GB/token | share |
|---|---:|---:|
| Q4_K weights (attn_q/k/o, ffn_gate/up all layers; ffn_down/attn_v ×18) | 3.355 | 71.8% |
| Q6_K weights (ffn_down ×18, attn_v ×18, lm_head) | 1.316 | 28.2% |
| **Total weight read (analytical)** | **4.671** | |
| measured `global_mem`/token (cli) | 4.726 | ✓ reconciles (Δ = KV+activations+embd) |

Weight-byte split (71.8/28.2) ≈ the kernel-*time* split (72.3/27.7) → both GEMV buckets run
at ~the same GB/s, as expected.

KV-cache read scales with context but stays small vs weights:

| context | KV read/token | % of weight bytes |
|---:|---:|---:|
| 8 | 1.2 MB | 0.03% |
| 1024 | 151 MB | 3.2% |
| 3072 | 453 MB | 9.7% |

## The slack is real

- **GEMV** reads 4.67 GB in ~13.0 ms (72.3% of an ~18 ms GPU-busy token) ⇒ **359 GB/s = 42% of
  peak**. So **HBM is ~58% idle during the weight read** — a ~6.5 GB byte-budget of unused
  bandwidth over the GEMV window. This is the B1 occupancy ceiling made concrete: batch-1
  GEMV can't saturate HBM, so the bandwidth is there.
- **Non-GEMV** (~5.0 ms, 27.7%) reads almost **no HBM**: norms/rope/sampling are
  cache-resident, and KV is ≤ 0.45 GB even at ctx 3072 — all of which fits inside the 6.5 GB
  idle budget. So the non-GEMV is **compute/cache-bound, not bandwidth-competing** → it can
  run in the GEMV's idle bandwidth. (Its compute also has CUs to use: the GEMV is
  occupancy-bound, leaving idle CUs.)

## Verdict: **GREENLIGHT overlap infra** (with eyes open)

The slack is real and the non-GEMV is overlappable across the whole practical context range.
Analytical ceiling if non-GEMV fully hides behind the weight stream
(`token → max(GEMV, non-GEMV)`):

```
token 18.0 ms → 13.0 ms   |   55.6 → 76.8 tok/s  (+38%)   |   53% → 73% of llama.cpp
```

That is the largest single decode lever still on the table, and it is the lever the roadmap
named (B2). **Recommend scoping the second-queue / cross-layer-interleave build as the next
plan** — overlap layer-N non-GEMV with layer-(N+1) weight-GEMV (legal: N+1 weights are
independent of N activations; the dependency is only *within* a layer).

### Honest caveats (size the next plan around these)
1. **+38% is the ceiling, not the estimate.** Realizable gain is a fraction of it: second-queue
   scheduling overhead, signal/sync cost, and partial CU-occupancy contention (the non-GEMV
   needs CUs too — favorable since GEMV leaves idle CUs, but not free).
2. **Major infra.** tinygrad has one compute queue per device, no reordering hook, no fusion
   pass (research-confirmed). The build adds a second compute queue + a cross-layer issue/
   interleave in `runtime/graph/hcq.py` + `engine/jit.py` — a real hot-path change, gated,
   exact-output (greedy token parity), AMD-validated.
3. **Cheaper de-risking increment first:** fusing the norm bucket (11.3%) into the adjacent/
   fused GEMV (in-pattern custom kernel, no scheduler change) *reduces* non-GEMV directly and
   captures part of the win (~+10%) with far less risk. Recommended as Increment 1 before the
   full second-queue build — it both ships value and validates the exactness/gating harness on
   a small change.

### Not a pivot
- **Not bandwidth-bound** (42% peak during GEMV) → B3 sub-4-bit is *not* forced here (it
  remains a separate beyond-llama lever, orthogonal to overlap).
- Attention stays overlappable even long-ctx (KV ≤ 0.45 GB ≪ 6.5 GB slack); flash already
  handles its *compute* growth.

## Next plan
Scope **Increment 1 (norm-into-GEMV fusion, in-pattern, exact)** as the first build — it
captures ~+10% with low risk and stands up the gating/exactness harness — then the **second-
queue overlap infra** for the remaining ceiling. Both gated, AMD-validated, `[nn]`/`[codegen]`.

Anchors: `amd-decode-sequential-tax-profile-20260616.md` (72/28 split + single-queue proof),
`amd-decode-beyond-llama-roadmap.md` (B2), `amd-decode-measurement-confounds.md`,
`machine-search-decode-context-plan-2026-06-16.md` (Track 1).
