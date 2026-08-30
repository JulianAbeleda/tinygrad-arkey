# NV installed-island Phase 7 partitioned K/V projection routes

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

Evidence: `docs/task_workflow/evidence/nv-installed-islands-20260822/phase7/`

## Findings, ordered by wall severity

`MEASURED` The mixed K/V bucket resolves cleanly into 36 K projections and 36
V projections, plus 26 shared-Q8 completion kernels. The K-versus-V split is
authoritative from tensor metadata and DAG completion consumers:

| route | K | V | K sum | V sum |
| --- | ---: | ---: | ---: | ---: |
| `q4k_g3_lanemap_gemv_1024_4096` | 19 | 9 | 90.432 us | 38.080 us |
| `q4k_warp_coop_q8_dp4a_partial_1024_4096` | 9 | 17 | 29.856 us | 63.840 us |
| `q6k_q8_warp_direct_1024_4096` | 8 | 0 | 30.048 us | 0 |
| `q6k_v_four_warp_fp16_direct_1024_4096` | 0 | 10 | 0 | 44.128 us |
| `r_8_32_4_4` completion | 9 | 17 | 12.160 us | 18.688 us |
| total | 36 | 36 | 162.496 us | 164.736 us |

`MEASURED` The node-sum deltas close against llama exactly:

```text
K  tinygrad 162.496 - llama 114.592 = +47.904 us
V  tinygrad 164.736 - llama 100.417 = +64.319 us
combined                               +112.223 us  (matches frozen census)
```

These are disjoint census rows, not additive wall recovery.

## Exact-body and clean-HCQ decomposition

`MEASURED` Every route was replayed with its exact production cubin (nsys
2000 reps) and its clean chained-HCQ drain slope:

| kernel | body B | clean HCQ C | D = C-B | production P | R = P-C | llama body |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q4 g3 1024 (K/V) | 3.328 | 3.798 | 0.470 | 4.768 | 0.970 | K 3.200 / V 2.784 |
| q4 coop 1024 (K/V) | 2.016 | 2.310 | 0.294 | 3.712 | 1.402 | same |
| q6 q8 1024 (K) | 2.112 | 2.394 | 0.282 | 3.808 | 1.414 | 3.200 |
| v four-warp 1024 (V) | 3.104 | 3.439 | 0.335 | 4.480 | 1.041 | 2.784 |
| r8 completion | 0.576 | 0.899 | 0.323 | 1.120 | 0.221 | n/a |

`MEASURED` tinygrad's K/V bodies are at or below llama's body, but the
production command interval is higher in every route. The positive installed
delta is dominated by the production-conditioned residual `R` (0.97-1.41
us/call) and the small clean dispatch `D` (0.28-0.47 us/call), not arithmetic.
The same pattern held for O (Phase 4) and the FFN support rows.

`MEASURED` The completion kernel split is a structural difference, not a
body problem: tinygrad runs 26 separate `r_8_32_4_4` completion kernels
(9 K + 17 V) while llama's K and V are single fused GEMVs with no completion
node. The completion body is `0.576 us` but its installed interval is
`1.120 us`, so each one also carries an install cost.

## Verdicts

```text
K body         BODY_PARITY     (weighted ~3.1-3.3 vs llama 3.200)
V body         BODY_PARITY     (weighted ~2.8 vs llama 2.784)
K mechanism    INSTALL_MIXED   (R + D dominate; body at parity)
V mechanism    INSTALL_MIXED   (R + D dominate; body at parity)
completion     STRUCTURAL      (26 extra nodes, no llama counterpart)
```

No single partition is arithmetic-bound. A family-wide recommendation across
K and V is therefore not justified; the two sides share the same mechanism
(production-conditioned residual plus completion structure), so one
handoff/launch-elimination scope covers both.

## Manifest correction

`MEASURED` The phase2 `I_V` bucket mislabeled `q6k_q8_warp_direct_1024_4096`
as `v_proj_shared_q8`. It is the K projection (Q6_K K, 8 nodes, 30.048 us).
The corrected V route is `q4_g3` (9) + `v_four` (10) + `q4_coop` (17) = 36,
and the corrected K route is `q4_g3` (19) + `q6_q8` (8) + `q4_coop` (9) = 36.

## Wall sensitivity

`INFERRED` Legal, non-double-counted ceilings per token:

```text
K installed delta  47.904 us   (R+D; body at parity)
V installed delta  64.319 us   (R+D; body at parity)
completion nodes   26 x (1.120 - 0.576) = 14.1 us install, plus structural
```

The projected ceiling is the residual-and-dispatch term, not a body rewrite.
Nothing here is booked.

## Decision

`INSTALL_MIXED`. One K/V projection handoff/launch-elimination scope targeting
the production-conditioned residual and the 26 completion kernels is the
follow-on, with the activation-quant provider advantage preserved as a frozen
control. Arithmetic/topology rewriting is prohibited by the BODY_PARITY
verdicts.

## Q projection partition (added for the Phase 10 ledger)

`MEASURED` The Q projection row (`+84.412 us`, 53 tinygrad kernels versus 36
llama Q nodes) closes with the same signature as K/V: the cooperative route's
body is faster than llama, and the install term throws the advantage away.

| kernel | nodes | P | B | C | D | R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q4k_g3_lanemap_gemv_4096_4096` | 19 | 8.704 | 7.488 | 7.751 | 0.263 | 0.953 |
| `q4k_warp_coop_q8_dp4a_partial_4096_4096` | 17 | 8.416 | 4.800 | 5.309 | 0.509 | 3.107 |
| `r_32_32_4_4` completion | 17 | 1.344 | 0.608 | 1.061 | 0.453 | 0.283 |

`MEASURED` llama Q body is `6.919 us/call`. The cooperative route body
(`4.800 us`) is `2.119 us/call` faster, but its production-conditioned
residual `R = 3.107 us/call` and the 17 completion kernels (`1.344 us` each)
invert the advantage. The Q deficit is install-dominated: coop install
`61.5 us`, G3 install `23.1 us`, completion `22.8 us`, against a net body
delta of only `-14.4 us`.

## Ledger snapshot

```text
node_sum   = 4677.920 us (tinygrad) / 3878.254 us (llama)
union      = 4671.500 us (tinygrad) / 3878.254 us (llama PDL-off)
overlap    = 6.420 us (tinygrad) / 0 us (llama PDL-off)
wall       = 4771.423 us (fresh control)
host_gap   = unmeasured single-domain
useful_body = unmeasured
booked_recovery = 0.000 us
remaining_to_240 = 604.756 us
```
