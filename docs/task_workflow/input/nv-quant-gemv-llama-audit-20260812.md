# NV quant GEMV llama audit - per-shape deficit cross-reference (rank-1 row)

Date: 2026-08-12
Branch: `nvidia-bringup-20260731` (HEAD `28665531d`, post M1 NO-GO + cost-gate fix)
Status: **audit record. Traces llama's live decode quant path per shape, cross-references
the current native census and every closed/landed mechanism, and ranks the remaining
unclaimed mass. Authorizes no implementation: the next step is a diagnostic microbench
under its own scope.** Process: trace llama -> arithmetic-validate -> e2e dependency map
-> implement (standing process, `0515f2539`).

## 1. Why this audit exists

The M1 norm-absorption NO-GO closed the last epilogue-folding idea on the norms row. The
remaining rank-1 mass is the quant GEMV class (~4053.74 us/token census at the current
HEAD, 217 kernels). Every named mechanism on this row was measured and closed or booked
between 08-03 and 08-09 (MC1/MC2/MC3 diagnostics, shared-Q8 DP4A substrate, M2/M4
epilogue folds). Before opening yet another candidate, this audit re-derives the deficit
per shape from llama's live d512 path and asks: which shapes still have an untried,
llama-anchored mechanism?

## 2. Llama per-shape path (llama.cpp `ac4cddeb0`, d512, Qwen3-8B-Q4_K_M)

Every projection node is `mul_mat_vec_q<type,1,has_fusion>` (MMVQ): Q8_1 quantized
activation + DP4A, one CTA per output row, 4 warps/CTA, block `(32,4,1)`, grid `(N,1,1)`,
zero dynamic smem (`mmvq.cu`). The fused gate/up is ONE mmvq call with SWIGLU in-kernel
(37.856 us/node, 36 nodes, 1364.038 us/token); O and down run residual/bias adds
in-kernel (`has_fusion=true`). Quantize_q8_1 runs per matmul (217 nodes, 552.101 us
node-sum; 445.954 us hidden behind the mmvq union, ~106 us exposed).

| population | quant | N x K | us/node | nodes | llama total us/token |
| --- | --- | ---: | ---: | ---: | ---: |
| attention Q | Q4_K | 4096x4096 | 9.536 | 36 | 342.881 |
| attention K | Q4_K | 1024x4096 | 3.328 | 36 | 117.376 |
| attention V (Q4 layers) | Q4_K | 1024x4096 | 3.328 | 18 | 75.838 |
| attention V (Q6 layers) | Q6_K | 1024x4096 | 4.896 | 18 | 89.437 |
| attention O | Q4_K | 4096x4096 | 11.776 | 36 | 418.464 |
| fused gate/up | Q4_K | 12288x4096 | 37.856 | 36 | 1364.038 |
| FFN down (Q4 layers) | Q4_K | 4096x12288 | 11.776 | 18 | 346.209 |
| FFN down (Q6 layers) | Q6_K | 4096x12288 | 28.753 | 18 | 520.836 |
| vocabulary | Q6_K | 151936x4096 | 303.618 | 1 | 303.618 |

## 3. Native census at HEAD (control arm, 594 kernels, 5494.76 us/token)

Quant GEMV class (q4k + q6k + shared-Q8 consumers): 217 kernels / 4053.74 us/token.
The class row matches the ~4050 us cited by the epilogue-absorption route scope.

| kernel family | count | median us | total us/token |
| --- | ---: | ---: | ---: |
| q4k_g3_lanemap_gemv_w1w3fused16_12288_4096 (landed MC3+M2a) | 36 | 38.735 | 1394.46 |
| q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288 (M2b) | 18 | 26.75 | 481.50 |
| q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd (M2+M2b) | 18 | 35.17 | 633.06 |
| q4k_g3_lanemap_gemv_epi_resadd_4096_4096 (M4) | 36 | 9.825 | 353.70 |
| q4k_g3_lanemap_gemv_4096_4096 (plain q/o) | 19 | 9.63 | 182.97 |
| q4k_g3_lanemap_gemv_1024_4096 (plain k/v) | 28 | 4.88 | 136.64 |
| q6k_gen_partial_1024_4096_4 | 10 | 17.89 | 178.90 |
| q6k_gen_coop_151936_4096_inkernel (vocab) | 1 | 330.21 | 330.21 |
| q4k_warp_coop_q8_dp4a_partial_1024_4096 (shared-Q8) | 26 | 3.87 | 100.62 |
| q4k_warp_coop_q8_dp4a_partial_4096_4096 (shared-Q8) | 17 | 9.12 | 155.04 |
| q6k_q8_dp4a_1024_4096 (shared-Q8 Q6) | 8 | 13.33 | 106.64 |

## 4. Closed and landed mechanisms on this row (no re-litigation)

| mechanism | status / measured |
| --- | --- |
| MC3 w1+w3 fused gate/up (scalar) | LANDED 08-03, +1.7-2% tok/s |
| shared-Q8 Q4 attention lease | BOOKED +49.62 us d512 (08-09) |
| shared-Q8 Q4 g12 + max17 | booked 24.676 + 12.462 us (incremental) |
| M4 o-proj residual fold | BOOKED +32.61 us |
| fp32 q/k route | BOOKED +83.5 us |
| MC2 partial (Q6 1024x4096) | NO-GO: installed split-4 is local optimum (7.36-7.39 us vs 3.3 us floor), all 14 swept shapes worse |
| MC2 coop-down (Q6 4096x12288) | NO-GO: installed control is local optimum (26.48-26.59 standalone), all 17 shapes worse |
| MC2 q4k gate/up quad u128 | NO-GO in-loop: standalone 10.0-10.15 us but fused in-loop 49.2 us (-5% wall) |
| MC1 dp4a instruction mapping | closed: vector spelling compiles identical to scalar FP32 recurrence |
| Q4 FFN subset (shared-Q8) | WALL NO-GO +6.205 us/token (layer-8 singleton) |
| Q6 shared-Q8 consumer | WALL NO-GO +7.009 us/token (g12 settled), re-bracket +27.28 us (08-09) |
| FFN-down shared-Q8 | WALL NO-GO +151.192 us/token |
| reduce-output phase6 | NO-GO: candidate 18.5 us SLOWER (5.420 vs 5.401 ms) |
| M1 norm absorption | NO-GO: +81.92 us/token, cost gate CONTRADICTED (this audit's trigger) |

## 5. Per-shape deficit cross-reference (the finding)

Llama floors (us/node) vs native medians (us), matched by population:

| shape | llama us/node | native median | ratio | native total us | closed? |
| --- | ---: | ---: | ---: | ---: | --- |
| FFN down Q4 4096x12288 | 11.776 | 26.75 | **2.27x** | 481.50 | **NEVER SWEPT** |
| FFN down Q6 4096x12288 | 28.753 | 35.17 | 1.22x | 633.06 | MC2 coop NO-GO |
| attention V Q6 1024x4096 | 4.896 | 17.89 | 3.65x | 178.90 (10 k) | L2/MC2 partial NO-GO |
| attention K Q4 1024x4096 | 3.328 | 4.88 / 3.87 | 1.47x / 1.16x | 136.64 + 100.62 | shared-Q8 landed on 26/54 |
| gate/up 12288x4096 | 37.856 | 38.735 | 1.02x | 1394.46 | landed, at parity |
| vocab 151936x4096 | 303.618 | 330.21 | 1.09x | 330.21 | landed, near parity |

**The gap: the Q4 FFN-down shape (4096x12288) was never load-pattern swept.** MC2 swept
the partial Q6 shape, the coop-down Q6 shape, and the gate/up Q4 shape - but NOT the Q4
down shape. It is our largest remaining per-shape ratio deficit (2.27x llama) and the
largest untouched kernel family by total mass (481.50 us/token, 18 kernels). llama's Q4
down node runs the same MMVQ 4-warp/DP4A geometry as everything else, so the floor is
llama-anchored, not invented.

## 6. Ranked candidates (from the cross-reference, not from memory)

| rank | mechanism | mass | llama floor | expected ceiling | status |
| --- | --- | ---: | ---: | --- | --- |
| 1 | **Q4 FFN-down 4096x12288 load-pattern sweep** (vec width / rows-per-block / smem x-staging / prefetch on the installed `q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` body) | 481.50 us | 11.776 us/node | ~212 us node-sum at floor (-269 us census, INFERRED); in-loop offset 1.26-1.7x keeps even partial wins positive | **untried surface**; next step is the MC2-style diagnostic microbench with the same controls and the in-loop census gate |
| 2 | Q6 attention-V partial (10 kernels) | 178.90 us | 4.896 us/node | L2/MC2 swept: installed split-4 is the local optimum | closed NO-GO, no new mechanism |
| 3 | Q4 attention-K remainder | ~35 us | 3.328 us/node | shared-Q8 already on 26/54; remainder is non-admitted blocks | closed-default lease; tail expansion NO-GO |
| 4 | vocab | 330.21 us | 303.618 us/node | 1.09x, near parity | landed, no lever |

## 7. Decisive gate for rank 1

The MC2 pattern applies unchanged: extend `l2_q6k_partial_sweep.cu` with a q4k-down
family (4096x12288, k_blocks=48, same installed body via the `epi_ffnresadd` kernel
replica), sweep vec / rows-per-block / xsmem / pf, reproduce the installed control
(26.75 us in-loop; standalone target per the 1.26-1.7x offset), gate on the llama-class
floor (11.776 us/node, i.e. standalone ~8-9 us at the same offset), and keep the MC2
lesson: a standalone winner is NOT evidence until it survives the in-loop census. If a
row clears the floor in-loop, it becomes an additive-route candidate with the full
smoke/logits/census/wall gate; if not, the Q4 down row closes NO-GO with a floor table
and the quant row is exhausted except for closed mechanisms.

## 8. References

- `nv-decode-llama-live-gemv-route-audit-20260805.md` (llama path, per-shape medians)
- `nv-decode-llama-d512-timeline-ledger-20260804.json` (quantize_q8_1 552.101 us, hidden 445.954)
- `mc2-load-pattern-measurement-record-20260803.md` (sweep surface; the q4k section is gate/up only)
- `mc3-w1w3-fusion-measurement-record-20260803.md` (gate/up fusion, landed)
- `nv-decode-parity-campaign-reconciled-ledger-20260805.md` (booked/closed rows)
- `nv-gemv-substrate-landing-scope-20260808.md`, `nv-decode-gate-green-parity-accounting-20260809.md` (shared-Q8)
- Current census: control arm of the M1 cost-gate campaign record (`/tmp/m1_costgate_ab_fixed.json`)
