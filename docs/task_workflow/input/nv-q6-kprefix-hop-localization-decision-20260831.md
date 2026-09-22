# NV Q6_K real-fixture K256 hop-localization decision

Date: 2026-08-31  
Evidence: `docs/task_workflow/evidence/nv-q6-oracle-kprefix-hop-20260831/result.json`  
Verdict: `PASS_REPAIR_FULL_AB`

## Decision

The first proven divergence is at **depth 1, epoch 0, `WEIGHT_SCALE_CONTRACT`**.

The legacy broad body computes an integer scale-weighted dot and applies block `d` afterward. The trusted direct-wide contract first rounds each `d * signed_q6_scale` product to FP16, converts it back to FP32, and applies the two independently rounded weight scales to their corresponding integer IMMA accumulators.

The isolated `trusted_fp16` arm repairs correctness at every tested prefix and on the full 128-tile route. It is warranted as the required numerical contract, but its current explicit implementation is **not performance-admitted**: it regresses the locked R31 median by 16.928 us. No commit is authorized from this gate.

```text
first_divergence = WEIGHT_SCALE_CONTRACT
first_depth = 1
repair_prefix_correct = true
repair_full_correct = true
repair_trace_exact = true
repair_timing_admitted = false
commit_admitted = false
```

## Executed command

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_kprefix_trace.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --depths 1,2,4,8,16,24,32,40,48 \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-kprefix-hop-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-kprefix-hop-20260831/artifacts
```

The final run exited 0 after the prefix traces and conditional full-route R31 A/B. No GPU run remains active.

## Proven hop ledger

The primary probe is tile `(0,0)`, output row 0/column 0, with its earliest differing scale product at epoch 0/group 0.

| Hop | Evidence | Verdict |
|---|---|---|
| Packed real Q6/Q8 prefixes | Each depth is repacked with its actual row stride; matching wide and broad records derive from the same full fixture | exact input contract |
| Q6 shared publication | Selected 76-word row, written-word mask, sentinel padding, and FP32 `d` bits | exact |
| Q8 publication | Both shared panels and traced B/Q8 carriers | exact |
| Decoded Q6 values/scales | Host canonical decode versus production trace | exact |
| A fragment carrier | Bounded indirectly by exact Q6 publication and exact actual IMMA results; direct vector observation was removed because CUDA rendering cannot scalar-bitcast `signed_char8` safely | indirect exact |
| IMMA `c0/c1` | Actual production accumulators versus host integer replay | exact |
| Integer corrections | `s0`, `s1`, both scale-weighted contributions, combined dot | exact |
| Lane/output mapping | CTA/thread/warp/lane/accumulator to output row/column | exact |
| Legacy weight-scale contract | Does not perform the trusted FP16 rounding | first mismatch |
| Repaired FP32 steps | FP16-rounded `d*s0`, `d*s1`, contribution and accumulator bit replay | exact |
| Phase/epoch accumulator chain | Every traced accumulator before/after edge | exact |
| Final accumulator versus output | Final traced bits versus copied mapped output element | exact |

The repaired depth-48 tile reports every required trace check true:

```text
q6_publication_exact
q8_both_panels_exact
decoded_q6_values_exact
decoded_q6_scales_exact
d_fp32_bits_exact
traced_q8_carriers_exact
imma_c0_c1_exact
integer_corrections_exact
repair_fp32_steps_exact
mapping_exact
accumulator_chain_bits_exact
final_accumulator_output_bits_exact
```

## Correct repair contract

For each Q6 scale pair and its corresponding IMMA results:

```text
d = fp32(shared_q6_d)
w0 = fp32(fp16_rn(fp32(d * fp32(s0))))
w1 = fp32(fp16_rn(fp32(d * fp32(s1))))
t0 = fp32(fp32(c0) * w0)
t1 = fp32(fp32(c1) * w1)
weighted_dot = fp32(t0 + t1)
scaled = fp32(weighted_dot * yscale)
acc = fp32(acc + scaled)
```

`d` is published as FP32 bits in shared word 64 for this arm. The FP16 round trips for `w0` and `w1` are semantic, not storage optimizations, and cannot be replaced with `d * (s0*c0 + s1*c1)`.

## Population impact

Across `blk.0.ffn_down.weight`:

| Metric | Value |
|---|---:|
| Q6 `d * scale` products | 3,145,728 |
| Products changed by FP16 rounding | 2,064,507 |
| Changed fraction | `0.6562891006469727` |
| Maximum per-scale absolute change | `0.000003814697265625` |
| Mean per-scale absolute change | `1.1423609436178594e-7` |

The individual changes are small but affect 65.63% of scale products and accumulate across K, explaining the systematic full-output residual.

## Prefix sweep

Depths `1,2,4,8,16,24,32,40,48` used actual packed model prefixes and independently qualified matching wide-direct references.

- The legacy contract first diverges at depth 1.
- Every depth classifies the first semantic mismatch as `WEIGHT_SCALE_CONTRACT`.
- The repaired arm passes every prefix.
- At depth 48 on the traced tile, repaired max absolute error is `0.00006103515625`, mean absolute error is `3.2260140869766474e-7`, and failing count is zero.

## Full 128-tile correctness

| Metric | Legacy baseline | `trusted_fp16` repair |
|---|---:|---:|
| Finite | true | true |
| Max absolute error | `0.187255859375` | `0.0001220703125` |
| Mean absolute error | `0.013687682338058949` | `3.190014581377909e-7` |
| Failing count | 1,758,882 | 0 |
| Reference pass | false | true |

This reduces maximum error by more than three orders of magnitude and removes every tolerance failure.

## Locked R31 timing

The arms run in the same process with alternating order.

| Metric | Legacy baseline | Repair | Repair minus baseline |
|---|---:|---:|---:|
| Median | `282.112 us` | `299.168 us` | `+17.056 us` |
| Paired median delta | - | - | `+16.928 us` |
| Repair wins | - | 0/31 | - |

The repair is approximately 6.0% slower in this explicit formulation. Correctness is mandatory, but this implementation does not satisfy a positive timing promotion threshold.

## Normalized SASS/resources

| Metric | Legacy baseline | Repair |
|---|---:|---:|
| Cubin SHA-256 | `839a95b6b09fc9d0974ad9b5d19a2ba971a9d8ef768a871e1c044d0bb2efb430` | `e2da8ad1aa67caa698c227f731231e1848debb172b529dd4bbd50081c967248f` |
| Static instructions | 4,088 | 5,352 |
| Registers | 255 | 255 |
| Stack | 64 B | 64 B |
| LDL / STL | 15 / 30 | 16 / 16 |
| IMMA / LDSM | 256 / 32 | 256 / 32 |
| LDG / STS | 105 / 71 | 105 / 71 |
| BAR | 5 | 5 |
| FFMA | 512 | 0 |
| FMUL / FADD | 512 / 0 | 1,600 / 1,024 |
| F2FP / I2FP | 0 / 512 | 64 / 1,024 |

The repair preserves tensor-core, global, shared, and barrier counts. Its cost is explicit conversion and scalar FP32 arithmetic. The next arithmetic implementation experiment must preserve the FP16-rounded per-scale contract while recovering contraction/scheduling; it must compare bit/tolerance correctness and emitted SASS, not algebraically refactor `d` outward again.

## Promotion state and next gate

The repaired semantics become the correctness anchor for subsequent isolated experiments, but the current repair implementation remains uncommitted.

Before or alongside the nested-runtime-RANGE substrate gate, test a small finite implementation set for the same repaired semantics:

1. Preserve `w0/w1 = fp32(fp16(d*s))` exactly.
2. Use verified FMA only after those rounded values exist.
3. Compare `fma(c0,w0,c1*w1)` and the trusted wide grouping, retaining the output tolerance.
4. Require unchanged 256 IMMA, 32 LDSM, 105 LDG, 71 STS, and five barriers.
5. Require zero failing outputs and a positive R31 timing result before implementation promotion.

No nested-RANGE, prefetch, publication-barrier, or reduction result should be interpreted against the legacy incorrect arithmetic.
