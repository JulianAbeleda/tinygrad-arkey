# pipe_tm2_tn2 promotion hardening

**Verdict: PIPE_TM2_TN2_HARDEN_PASS_PROMOTE_READY**

## H1 long-context guard (PRIMARY) — PASS, holds TIER_A to 8192
| ctx | default | pipe | Δ | tier |
|---|---|---|---|---|
| 512 | 3596 | 4291 | +19.3% | TIER_A_MAJOR |
| 1024 | 3502 | 4089 | +16.8% | TIER_A_MAJOR |
| 2048 | 3249 | 3711 | +14.2% | TIER_A_MAJOR |
| 4096 | 2818 | 3137 | +11.3% | TIER_A_MAJOR |
| 8192 | 2234.5 | 2423.5 | +8.5% | TIER_A_MAJOR |

8192 feasible (no OOM); delta erodes +19.3%→+8.5% as non-GEMM grows, never regresses.

## H2 correctness — PASS (output-equivalent)
m.logits argmax 198, sum 524309.75, max 14.3638 — byte-identical default vs pipe. (m.forward's [1,1] token diverged only as gumbel sampling noise, not a logit diff.)

## H3 role mechanism — pipe lifts sub-BLAS roles, regresses saturated gate_up
| role | default %BLAS | pipe %BLAS | Δ |
|---|---|---|---|
| attn_kv | 51.5% | 106.4% | +106% |
| attn_qo | 67.7% | 107.7% | +59% |
| ffn_down | 76.4% | 91.5% | +20% |
| ffn_gate_up | 107.3% | 89.0% | **−17%** |

PREFILL_GEMM_PIPELINE TM=2 TN=2 = a 2x2 register micro-tile per thread (more independent FMAs in flight -> higher ILP/arithmetic-intensity, hides MFMA/load latency). LATENCY/ALU-bound roles WITH headroom (attn_kv 1024x4096, attn_qo 4096x4096, ffn_down 4096x12288) jump toward/past BLAS (+106/+59/+20%). But ffn_gate_up (12288x4096) was ALREADY at 107% BLAS (saturated, well-tuned) -> the extra register pressure of the 2x2 tile REGRESSES it -17%. Net whole-prefill is positive because qo+kv+down (~50% of wall) gains outweigh the gate_up loss (35% of wall).

**Follow-on:** role-SELECTIVE pipe (apply TM2TN2 to qo/kv/down ONLY, keep gate_up on the saturated default route) should recover the -17% gate_up loss -> potentially >+19%. A P3 candidate, not done here.

## H4 recommendation
PROMOTE-READY. Correct, TIER_A across all ctx incl. long-context (the user's promotion-critical gate), clean rollback, default stays off. Follow-on (P7): BubbleBeam/search-binding to select it without manual flags; and a role-selective-pipe P3 candidate (exclude gate_up) for further upside.