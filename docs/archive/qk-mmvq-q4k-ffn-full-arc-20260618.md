# Q4_K ffn_gate/up full-MMVQ arc — Phase 0/1 audit (2026-06-18)

The largest remaining decode lever: Q4_K ffn_gate/up (72 linears, 12288×4096, parts=1, **~44% of weight
traffic**). Local coop was refuted (41%→47%, 1.18×). This audit asks: is the ceiling coalescing-bound (Family B)
or dequant-ALU-bound (Family A = q8_1+dp4a)? **Answer: dequant-ALU-bound — Family A is the lever, and the
roofline for this role is ~70% peak (proven by READRAW).** Commit `1ed3d109b`. No kernels built this phase.

## Phase 0 — baseline

Defaults on: Q6_K lm_head/ffn_down coop, Q4_K attn_q/o coop; Q4_K ffn_gate/up = default fp-dequant path.
Decode: ctx512/1024/4096 = 68.3/66.3/60.9 tok/s (69/68/66% of llama). ffn_gate/up isolated (real weights, fresh
input, fp-reassoc-exact):

| variant | µs | GB/s | % HBM peak |
|---|---|---|---|
| base fp-dequant (default) | 77.6 | 365 | 41% |
| coop fp-dequant (lane4→LOCAL coalesced) | 65.7 | 431 | 48% |
| **READRAW (coalesced, no dequant)** | 44.8 | **632** | **70%** |

`bench/qk-mmvq-q4k-ffn-full/baseline.json`.

## Phase 1 — the decisive diagnosis

**READRAW (70%) >> coop (48%) → fp-dequant-ALU-bound.** The coalescing lever is already captured (base 41% →
coop 48%); the schedule itself can read at **632 GB/s (70%)** — so the remaining 48%→70% gap is **not** dataflow,
it is the **fp dequant ALU** (per-nibble int→fp convert + fp affine + fp MAC per weight). READRAW = 70% is
exactly llama's measured MMVQ level: llama reaches ~70% precisely because q8_1+dp4a makes the dequant nearly
free, so it runs at the read roofline. **The roofline for this role with a coalesced schedule is ~70% peak, and
it is reachable only by making the dequant cheap (int8/dp4a) — Family A.**

### tinygrad vs llama MMVQ (Q4_K ffn_gate/up)

| # | feature | tinygrad current | llama MMVQ | perf effect | search knob |
|---|---|---|---|---|---|
| 1 | weight access | row-major, coop-coalesced (lane4→LOCAL) | GGUF coalesced | parity now | B (done) |
| 2 | Q4_K unpack | per-nibble → **fp32** + fp affine **per weight** | unpack → **int8**, affine on block sums | **major (dequant ALU)** | **A** |
| 3 | activation | **fp16** | **q8_1 int8** (once/linear) | enables int dot | **A** |
| 4 | dot | fp dequant × fp MAC | **int8 dp4a (sdot4)** | **major** | **A** |
| 5 | accumulation | fp32 | int32 (dp4a) + fp affine epilogue | int accum cheaper | **A** |
| 6 | workgroup/tile | row_tile×8 lanes | block-tiled | minor | A/B |
| 7 | vector load width | scalar word/lane (coalesced) | vectorized | small headroom | B (small) |
| 8 | reduction/epilogue | stage-2 `.sum` (extra kernel) | in-kernel | small | C |
| 9 | kernels/linear | 2 (partial + sum) | 1 | small | C |
| 10 | graph/runtime | TinyJit | HIP graph | **not the limiter** (W==D, GPU-bound) | — |

The whole 48%→70% gap is items 2/3/4/5 = **the dequant ALU**. Items 1/6/7 (Family B) are exhausted (coop = 48%,
READRAW proves the schedule is fine). Items 8/9 (Family C) are small.

## Verdict / gate

The gap **plausibly exceeds +3% e2e**: ffn_gate/up is 44% of weight traffic; moving 41%→70% peak is ~1.7× on
the role → Amdahl ≈ +15-20% on this role's share → **~+5-12% e2e** (matches the estimate). **Proceed to Family A.**

- **Family B (q4k_ffn_unpack_dataflow): REFUTED by audit** — coop already reached 48%; READRAW shows the
  schedule can do 70% without dequant, so dataflow is not the limiter. Do not prototype B.
- **Family A (q4k_ffn_mmvq_q8: q8_1 activations + dp4a int-dot, in the coalesced coop structure): EARNED** — it
  directly attacks the dequant ALU, and READRAW proves the 70% roofline is reachable. **Caution:** dp4a *as a
  toggle on the uncoalesced fp kernel* was +1% (refuted) — A must be the *full co-design* (q8_1 + dp4a + coalesced
  lane mapping + int accum + fp affine epilogue), not a toggle. This is a new-kernel-family build, high-risk.

See `qk-mmvq-q4k-ffn-search-rows-20260618.md` for the rows. Next: the minimal Family-A prototype (q8_1+dp4a
coop), isolated-gated ≥1.3× whole-linear (q8 quant cost included), then in-model W==D ≥5%.
