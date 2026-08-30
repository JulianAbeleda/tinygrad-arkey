# NV Phase 0 current-ledger rebuild (2026-08-23)

Date: 2026-08-23  
Branch: `nvidia-bringup-20260731`  
HEAD: `6570abc025514273faa100c66b979e531585a1e1`  
GPU: RTX 5090 (`sm_120`), locked SM 2790 / memory 14001 MHz, observed SM 2715-2782 MHz  
Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, single-token decode, depth 512  
Routes: production (both accepted routes enabled), no research override

## Findings first

1. **[MEASURED] Route state is exact.** All 36 blocks carry the semantic Q/K
   RMSNorm+RoPE fusion; 18 blocks carry the Q6_K FFN-down four-warp fp16
   direct route. Current decode graph is 560 nodes/token
   (`32 + 64 + 128 + 256 + 80`).
2. **[MEASURED] Current device ledger closes exactly.**
   `node_sum = 4543.024 us`, `union = 4539.750 us`, `overlap = 3.274 us`.
   Identity residual `node_sum - overlap - union = 0.000 us`.
3. **[MEASURED] The profiled host wall is inflated by instrumentation.**
   In the same `PROFILE=1` capture the settled host wall is
   `6375.726 us/token`; the profiled `wall = union + host_gap` identity closes
   with `host_gap = 1835.976 us`, most of which is the per-replay
   `collect_timestamps` JSON export at graph boundaries (prior authority:
   `d14e6964e`). This host wall is not a decode-speed signal.
4. **[MEASURED/REFERENCE] The authoritative unprofiled wall remains
   `4697.289 us/token`** (`212.889 tok/s`, composition reverse bracket).
5. **[INFERRED] Real host gap is small.** `4697.289 - 4539.750 = 157.54 us`
   mixes the unprofiled wall authority with the profiled device union across
   sessions/clocks, so it is an inference, not a booked term.
6. **[MEASURED] Current-vs-llama reconciliation closes.** Representative
   steady replay node_sum `4543.01 us` versus llama PDL-off `3878.25 us`,
   delta `664.75 us`.

## Current census, ranked by recoverable term

Node-sum domain, current tinygrad minus retained llama PDL-off oracle:

| row | tinygrad us | llama us | delta us | tinygrad calls |
| --- | ---: | ---: | ---: | ---: |
| K/V projections + completion | 333.28 | 215.01 | **+118.27** | 98 |
| attn/ffn/final 4096 norm | 317.95 | 203.36 | +114.59 | 94 |
| gate/up GEMV | 1358.69 | 1268.37 | **+90.32** | 36 |
| down GEMV | 935.01 | 855.82 | **+79.19** | 36 |
| Q projection + completion | 327.39 | 249.09 | **+78.30** | 53 |
| flash score | 239.87 | 162.95 | **+76.92** | 36 |
| O projection | 335.26 | 259.81 | **+75.45** | 36 |
| flash combine | 102.50 | 37.06 | **+65.44** | 36 |
| vocab main + tail | 368.80 | 303.91 | **+64.89** | 5 |
| Q head norm (fused body) | 69.57 | 41.28 | +28.29 | 36 |
| K head norm (fused body) | 69.06 | 40.83 | +28.22 | 36 |
| misc / embedding | 9.50 | 2.78 | +6.72 | 4 |
| rope + K/V store | 44.83 | 92.87 | **-48.03** | 37 |
| activation quant | 31.30 | 145.12 | **-113.82** | 17 |

The `reduce_output_rmsnorm_rope_{32,8}_128` rows fold RMSNorm **and** RoPE
into one kernel while llama reports norm and rope separately, so the Q/K head
norm deltas are not like-for-like. Combining Q norm + K norm + rope/store:
tinygrad `183.46 us` versus llama `174.98 us`, a residual of `+8.48 us`. The
semantic fusion has effectively closed the former `104.8 us` clean-norm gap.

The 4096-norm row is still coupled to the activation-quant advantage:
`+114.59 - 113.82 = +0.77 us` net. It is not an independent recovery target.

## Where the remaining 530 us actually is

The wall gap to 240 (`530.62 us`) is not an admission/wait mystery. After
removing the now-parity norm/provider rows, the node-sum delta concentrates in
named kernel-residence rows:

```text
GEMV projections (K/V, gate/up, down, Q, O)   440 us
flash score + combine                         142 us
vocab main + tail                              65 us
misc + norm/rope residual                      15 us
total (node_sum domain)                        ~662 us
```

The K/V and O occurrence-0 exact-live closures already measured installed
excess `P-C5` of only `+0.520` and `+1.032 us/call` (count-weighted
`13.5` and `37.2 us/token`), and Q closed at `+0.008 us/call`. These rows are
therefore **body/DRAM/codegen** deficits, not scheduler admission or dependency
wait. This supersedes the prior "hundreds of microseconds of admission/wait in
Q/O/K/V" priority; that pool is measured to be small at occurrence zero.

## Verdict

`240_UNMEASURED`

The remaining gap is dominated by GEMV projection streaming efficiency
(`~440 us` across K/V, gate/up, down, Q, O), flash score/combine body
(`~142 us`), and vocab tail (`~65 us`). Phase 1 should partition the
un-closed flash rows and confirm body dominance; Phase 2 should target generic
NV quantized-GEMV streaming and the flash combine topology.

## Evidence

`docs/task_workflow/evidence/nv-phase0-current-ledger-20260823/`

- `production.profile.jsonl` - raw HCQ graph profile per replay.
- `production.child.json` - settled windows, token SHA, GPU state, routes.
- `ledger.json` - closure identities and per-name census.
- `llama-reconcile.json` - current tinygrad vs llama role reconciliation.
- `sha256.txt` - manifest over every retained artifact.

Tools: `extra/llm_research/decode/nv_phase0_current_ledger.py`,
`extra/llm_research/decode/nv_phase0_llama_reconcile.py`.

No production, renderer, scheduler, runtime, or route code was changed by this
Phase 0 capture.
