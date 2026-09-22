# NV GEMV substrate re-verification record - Q8_1 + DP4A four-warp package at current HEAD

Date: 2026-08-08
Branch: tinygrad `nvidia-bringup-20260731`, HEAD `fb522cd17` (M4 resadd promoted,
census-fix stack `a7944410e`/`123ea1b4c`/`c855309fc` in tree)
Status: **measurement record. Re-runs the two surviving isolated substrate gates at
the current HEAD. Both still win; direction and magnitude reproduce the 08-05
records. No route change, no promotion, no default flip.**

## 1. Why this record exists

The reconciled ledger's ranked step 1 is the distinct exact-native Q4/Q6 DP4A
substrate (the llama MMVQ package: Q8_1 activation representation + DP4A +
four-warp ownership). The 08-05 isolated included-cost wins were measured at
`a1a51c349`. Since then the M4 landing and the census-fix stack changed
union-resolve, transport elision, and the scheduler; a promotion scope must not
be written on stale isolated evidence. This record re-runs the two gates that
won at 08-05 on the current tree.

## 2. Gates re-run (all under `/tmp/gpu-bench.lock`, native NV sm_120)

### 2.1 Q4 cooperative Q8+DP4A four-warp (`q4k_warp_cooperative_dynamic_microgate.py`)

Same protocol as the 08-05 corrected retry: correctness first, then A/B/A, 200
replays, seven samples/arm. Production Qwen shape rows=4096, K=4096, LOCAL=128,
runtime block bound 4. Candidate included cost = three programs per replay:
`q8_1_llama_provider_4096`, `q4k_warp_coop_q8_dp4a_partial_4096_4096`,
`partial_sum_4096x4`.

| check | observed | gate |
| --- | --- | --- |
| baseline vs CPU fp16 ref, max abs | 4.7684e-06 | tol 0.0981 PASS |
| candidate vs CPU Q8 ref, max abs | 0.0025558 | tol 0.0981 PASS |
| candidate vs baseline fp16, rel L2 | 0.004799 | characterized |
| control A/C midpoint median | 123.341 us/replay | |
| candidate median | 67.061 us/replay | |
| delta | **-56.280 us/replay (-45.6%)** | **PASS** |

08-05 reference: `-56.957 us/replay (-45.70%)`. Direction and magnitude
reproduce on the current tree (delta within 0.68 us / 0.1 points of the 08-05
row). Correctness contract unchanged.

Artifact: `/tmp/q4k_coop_current_head_20260808.json`
(`tinygrad.q4k_warp_coop_dynamic_microgate.v3`, git_commit `fb522cd17`).

### 2.2 Q6 flat four-warp Q8+DP4A (`q6k_q8_warp_direct_microgate.py`)

Same protocol as 08-05 G1: correctness first, then interleaved A/B/A, 100
replays, seven samples/arm. Production Qwen Q6 V shape rows=1024, K=4096.

| check | observed | gate |
| --- | --- | --- |
| Q8 oracle max abs | 0.010935 | tol 0.02 PASS |
| direct vs partial bitwise | true (max abs 0.0) | PASS |
| partial midpoint | 65.543 us/replay | |
| installed midpoint | 66.352 us/replay | |
| partial vs installed | **-0.808 us** | win |
| direct vs partial | -0.249 us | win |
| direct vs installed | **-0.937 us** | win |

08-05 reference: flat control beat installed by 1.403-2.481 us in three
sessions. Direction reproduces; magnitude is smaller on this session (0.81-0.94
us). Verdict shape: `MEASURED_WIN_REQUIRES_FULL_LOGIT_GATE`.

Artifact: `/tmp/q6k_warp_direct_current_head_20260808.json`
(`tinygrad.q6k_q8_warp_direct_microgate.v1`, git_commit `fb522cd17`).

### 2.3 Q6 row_tile=2 single-kernel spelling (`q6k_q8_dp4a_microgate.py`)

Not a win: delta **+0.405 us/replay** vs installed partial+sum. This is the
row_tile=2 (two-warp) spelling, not the four-warp ownership. It confirms the
four-warp/128-thread ownership is the load-bearing part of the Q6 package, and
is recorded so the scope does not reopen the two-warp spelling.

Artifact: `/tmp/q6k_q8_dp4a_current_head_20260808.json`.

## 3. What this does NOT do

- No route, policy, or default change; the checked-in
  `decode-q4k-epilogue-resadd-route-policy.json` promotion from the M4 gate is
  untouched by this record.
- No token-wall claim. These are isolated included-cost gates; the model-level
  semantic + composed-wall ladder is the landing scope's business, and every
  prior model-level attempt (Q4 g12/max17 booked; Q6 direct g12 WALL NO-GO;
  FFN-down singleton WALL NO-GO) is cited there.
- No new Q8 quantize tax: the provider is the already-qualified
  `_emit_q8_provider` / `_emit_rmsnorm_q8_provider`; its cost is inside every
  number above.

## 4. Evidence chain

- 08-05 isolated Q4 win: `nv-q4-warp-cooperative-dynamic-static-record-20260805.md`
- 08-05 isolated Q6 win: `nv-q6k-mmvq-instruction-delta-and-one-change-record-20260805.md`
- Model-level Q4 cooperative progression (booked): `nv-q4-cooperative-shared-q8-g1-integration-record-20260805.md`,
  `nv-q4-cooperative-subset-precision-budget-record-20260805.md`
- Model-level Q6 direct: `nv-q6-direct-shared-q8-g12-record-20260805.md` (WALL NO-GO)
- FFN-down: `nv-q4k-ffn-down-mmvq-included-cost-and-one-layer-record-20260805.md`
