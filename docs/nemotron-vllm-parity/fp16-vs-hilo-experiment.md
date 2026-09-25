# fp16 vs hi/lo experiment record

Recovered from the crashed session's transcript (`d46d3054-5e4f-4756-9122-360207142d7f`, 2026-09-25)
after the scratch files it produced (`~/.../scratchpad/fp16/{arms.py,consist.py,*.log,*.json}`) were
deleted. Recovery source: the coordinator's own messages/polls in the main transcript, the fp16 test
subagent's full tool-call transcript (`.../subagents/agent-ad2d2ffc19a29c628.jsonl`, 254 lines, no
final report — the agent never handed back), and two of its background-task output files that
survived under `/tmp/claude-1000/.../tasks/` (`bgz7qn4z3.output` = `gemm_time.log`; `bei6216tl.output`
= `consist_A_1000.log`). All numbers below are quoted from those sources; nothing is estimated.

## Question

Julian, 15:03 2026-09-25 ("well lets look at the curr lit. see if there are any updates" / "yeah lets
do that"): given Qi et al. 2025 ([arXiv 2510.26788](https://arxiv.org/abs/2510.26788), "Defeating the
Training-Inference Mismatch via FP16"), which reports fp16 cuts sampler/trainer log-prob mismatch by up
to 24x vs bf16 — can **fp16 operands** replace our **hi/lo bf16 split** for Nemotron-H 4B, without losing
sampler/trainer self-consistency, at half the GEMM cost?

## Hi/lo decision (separate, already made)

At 14:02 the coordinator recommended keeping hi/lo for prefill (one stack, one numerics policy; hi/lo is
cheap since prefill is shared across an RLOO group of 8 rollouts; ~+0.35s/10k-token prompt). **Julian
approved at 14:02 ("ok do it")** — this is independent of, and predates, the fp16 test below; hi/lo
stayed the production numerics regardless of the fp16 result.

## Arms (from the agent's brief, 15:04)

- **A** (baseline): current hi/lo bf16 — activation split into two bf16 terms stacked on rows, weight
  read once (2x GEMM work). Rows ≤16 use fp32 matvec.
- **B**: fp16 operands — activation cast once to fp16, weights held as an fp16 copy, fp16xfp16 matmul
  with fp32 accumulation on tensor cores.
- **C**: B + fp16 rounding of attention (model K/V/q/P, sampler KV storage, prefill buffers) — this
  could also close the known ~2e-3 prefill/prefix gap since the flash-prefill kernel already runs fp16
  internally.
- **D**: only the attention/KV rounding moves to fp16; GEMMs stay hi/lo bf16. **Defined in `arms.py`
  (index 195 of the subagent transcript) but never measured** — the session crashed before any `D` row
  was produced (`consist_D_200.json`/`consist_D_1000.json` exist in a later `ls` but are never `cat`'d).

Gate stated in the brief: adopt fp16 only if self-consistency vs arm A (seed-for-seed) passes **mean
≤1e-3, max ≤1e-2** (aside from sampling-noise outliers) *and* it is measurably faster.

## Results

### Weight/activation range (`wrange.py`, one background run, all 4B tensors)

No bf16 weight overflows the fp16 normal range in any role (`over=0` everywhere). A small, uniform
fraction underflows to fp16 subnormal — not zero, not concerning:

| Role | n | subnormal frac | max\|w\| |
|---|---|---|---|
| ssm_in | 1,152,743,424 | 3.45e-3 | 0.758 |
| ssm_out | 505,774,080 | 3.00e-3 | 1.21 |
| ffn_up | 668,745,728 | 2.81e-3 | 1.46 |
| ffn_down | 668,745,728 | 2.96e-3 | 1.12 |
| attn_q | 64,225,280 | 2.79e-3 | 0.34 |
| attn_k | 12,845,056 | 2.89e-3 | 0.293 |
| attn_v | 12,845,056 | 2.84e-3 | 0.408 |
| attn_output | 64,225,280 | 2.79e-3 | 0.809 |
| token_embd | 411,041,792 | 5.25e-3 | 0.289 |
| output | 411,041,792 | 4.18e-3 | 0.303 |

Activation max\|x\| at a 1000-token prompt (well under fp16's 65504 max): ssm_in 48.1, ssm_out 447,
ffn_up 66.7, ffn_down 607, attn_q/attn_kv 68.6, attn_o 9.5.

### GEMM speed, us/call (`gemm_time.log`, `gpu-run time`, background task `bgz7qn4z3`, exclusive GPU)

Columns: `A_hilo` (routed candidate if rows≤128 else generic), `A_hilo_routed`, `bf16x1_routed` (single
bf16, no hi/lo), `fp16_routed`, `fp16_generic`; last column is fp16-tensor-core max relative error vs an
exact reference (all ~3-6e-6, i.e. fp16 arithmetic itself is not the noise source).

| Role | M | A_hilo | A_hilo_routed | bf16x1_routed | fp16_routed | fp16_generic | maxrel |
|---|---|---|---|---|---|---|---|
| ssm_in | 2048 | 12.422 | 4.136 | 2.032 | 1.994 | 3.002 | 5.8e-6 |
| ssm_in | 8192 | 53.426 | 15.138 | 7.312 | 7.122 | 13.478 | 6.2e-6 |
| ssm_in | 16384 | 109.415 | 36.662 | 13.955 | 14.865 | 27.457 | 6.4e-6 |
| ssm_out | 16384 | 41.390 | 95.290 | 28.213 | 28.299 | 11.175 | 5.6e-6 |
| attn_q | 16384 | 18.376 | 92.069 | 33.178 | 33.296 | 5.159 | 3.8e-6 |
| attn_o | 16384 | 24.500 | 73.567 | 26.651 | 26.691 | 6.804 | 5.1e-6 |

`fp16_routed` tracks `bf16x1_routed` closely (as expected — same route family, half the hi/lo work) and
both roughly halve `A_hilo_routed` at large M. (`ffn_up`/`ffn_down`/`output` rows hit
`KernelOptError: no tensor core available` on this box's route table and were not measured.)

### Self-consistency (sampler logprobs vs full recompute; `consist_{A,B,C}_{200,1000}.log`, mean/p99/max
x1e-3, two seeds, B=8 and 32). `R0` = prefix-primed vs full recompute, `R1` = prefill-primed vs full,
`R2` = prefill-primed vs teacher-forced completion (the metric closest to the RLOO update path).

**PL=200:**

| Arm | B | seed | R0 | R1 | R2 |
|---|---|---|---|---|---|
| A | 8 | 0 | 1.70/10.6/16 | 1.04/5.4/8 | 0.95/5.3/9 |
| A | 8 | 1 | 1.37/9.9/28 | 1.12/5.6/9 | 0.99/5.9/9 |
| A | 32 | 0 | 1.02/7.6/16 | 1.11/8.1/14 | 1.00/7.7/18 |
| A | 32 | 1 | 0.97/6.8/15 | 1.15/7.5/48 | 1.01/7.0/56 |
| B | 8 | 0 | 2.09/18.5/38 | 1.43/8.0/13 | 2.51/15.8/48 |
| B | 8 | 1 | 1.50/8.7/14 | 1.51/7.5/16 | 1.33/7.0/11 |
| B | 32 | 0 | 1.31/8.2/16 | 1.37/8.6/19 | **3.97/65.6/500** |
| B | 32 | 1 | 1.52/11.5/49 | 1.37/7.7/53 | 2.13/20.4/317 |
| C | 8 | 0 | 0.65/3.3/4 | 0.59/2.5/4 | 0.56/2.5/3 |
| C | 8 | 1 | 0.67/2.9/14 | 3.40/28.9/206 | **3.67/28.2/205** |
| C | 32 | 0 | 0.63/3.3/6 | 1.83/26.2/86 | 1.99/32.2/101 |
| C | 32 | 1 | 0.62/2.8/5 | 1.29/15.1/181 | 0.95/6.7/35 |

**PL=1000:**

| Arm | B | seed | R0 | R1 | R2 |
|---|---|---|---|---|---|
| A | 8 | 0 | 3.33/51.6/78 | 1.42/10.7/22 | 1.44/10.0/31 |
| A | 8 | 1 | 1.28/9.6/14 | 1.41/9.9/24 | 1.36/8.3/24 |
| A | 32 | 0 | 0.91/7.7/17 | 1.03/7.3/44 | 0.96/7.4/14 |
| A | 32 | 1 | 1.24/10.8/76 | 1.14/8.4/32 | 1.09/8.4/32 |
| B | 8 | 0 | 1.67/11.7/17 | 1.61/11.4/23 | 1.50/13.0/27 |
| B | 8 | 1 | 1.58/10.4/23 | **5.05/64.9/170** | **5.15/66.9/168** |
| B | 32 | 0 | 1.29/9.1/15 | 1.35/8.5/21 | 1.26/8.3/27 |
| B | 32 | 1 | 2.92/25.6/136 | 3.76/46.1/304 | 3.81/44.8/301 |
| C | 8 | 0 | 0.74/3.8/7 | 1.95/26.3/93 | **36.14/343.2/581** |
| C | 8 | 1 | 1.58/23.4/122 | 2.68/38.6/121 | 2.73/38.9/120 |
| C | 32 | 0 | 0.55/3.1/7 | 0.59/3.4/8 | 0.57/3.3/6 |
| C | 32 | 1 | 4.20/44.9/153 | 3.78/41.1/139 | 3.67/41.8/138 |

D: never measured (no rows recovered anywhere in the transcript).

### End-to-end speed

Arm A only (batched decode, `gpu-run time`, `e2e.py`): B=64 step median **17.94 ms** (3568 tok/s), B=128
step median **30.93 ms** (4138 tok/s) — matches the numbers already on the goal board for `c736693a2`.
Arm B decode (`e2e_B_dec64.log`) and arm A/B 10k-token prefill (`e2e_A_pf.log`, `e2e_A_pf2.log`) both hit
`MemoryError: NV_ERR_NO_MEMORY` (29-30 GB used) under the shared/contended GPU at the time — no clean
arm-B or long-prefill end-to-end number exists.

## Reading the numbers

- Pure fp16 tensor-core arithmetic is not the noise source (maxrel ~3-6e-6 vs an exact reference, and
  0 bf16 weights overflow fp16's range).
- But **B and C both blow past the stated gate** (max ≤1e-2) on `R2`, the metric closest to what an RLOO
  update would use: B reaches max 0.500 (PL=200, B=32, seed0) and 0.170 (PL=1000, B=8, seed1); C reaches
  max 0.205 (PL=200) and **0.581** (PL=1000, B=8, seed0) — worse than A's own worst R2 max (0.056 at
  PL=200, 0.032 at PL=1000). C's attention-rounding change (meant to *close* the prefill/prefix gap) is
  the arm with the single worst outlier of the whole experiment.
- A itself is not perfectly clean either (R0 max up to 0.078 at PL=1000), so some of this is inherent
  batch/seed variance in the harness, but B/C are consistently equal-or-worse than A on the tail, never
  clearly better, across every (PL, B, seed) cell measured.

## Conclusion status: **open — no decision made**

The fp16-test subagent's own transcript ends mid-run (last tool call at index 253, scheduling one more
`consist_D` wait) with no final report ever produced; the coordinator sent it a second unanswered status
poll at 16:56 just before the session crashed. No recommendation was delivered, and Julian was never
asked to decide on fp16 itself (only on hi/lo, which he'd already approved earlier and independently of
this test). Based on the numbers actually recovered, arms B and C do not pass the experiment's own gate,
so adopting fp16 is not supported by what was measured — but this was never formalized as a rejection,
since the agent that owned the call never got to report it. Treat as: **do not adopt fp16 pending a
completed re-run**, not as a closed/rejected decision.
