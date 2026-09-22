# NV Q6_K packed FP16 weight-scale contract decision

Date: 2026-08-31  
Evidence: `docs/task_workflow/evidence/nv-q6-oracle-kprefix-hop-20260831/packed-result.json`  
Verdict: `PROMOTE_TRUSTED_FP16_PACKED`

## Decision

The `trusted_fp16_packed` contract is admitted as the correctness/performance anchor.

It preserves the proven FP16 rounding of each `d * signed_q6_scale`, produces output bit-identical to the explicit correct repair at all nine prefixes and across the full output, retains the existing 76-word Q6 shared row and 57,344-byte dynamic shared allocation, beats both comparison arms in 31/31 paired rounds, and eliminates local spills.

```text
trusted_reference = pass
explicit_repair_bit_equal = true
prefix_trace_gate = pass
full_output_gate = pass
paired_timing_gate = pass
resource_gate = pass
promotion_eligible = true
```

## Shared ABI

| Q6 row words | Meaning |
|---|---|
| `0..63` | Existing signed Q6 publication |
| `64` | FP32 `d` bits |
| `65..72` | Eight packed pairs of FP16-rounded `d * signed_scale` values |
| `73..75` | Padding/sentinel |

The consumer scalar-unpacks the two FP16 values from each scale word and converts them to FP32. It does not factor `d` back outside the two IMMA terms.

## Validation

Focused command:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  test/unit/test_nv_q6_oracle_weight_scale_contract.py
```

Result: `2 passed in 0.58s`.

Locked GPU command:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_kprefix_trace.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --depths 1,2,4,8,16,24,32,40,48 \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-kprefix-hop-20260831/packed-result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-kprefix-hop-20260831/packed-artifacts
```

Result: exit 0 with `promotion_eligible=true`.

## Correctness

All packed prefix traces pass the packed table, Q6/Q8 publication, decoded value/scale, actual IMMA, weighted-term, mapping, accumulator-chain, and final-output gates.

| Full-route metric | Incorrect legacy | Explicit repair | Packed |
|---|---:|---:|---:|
| Maximum absolute error | `0.187255859375` | `0.0001220703125` | `0.0001220703125` |
| Mean absolute error | `0.013687682338058949` | `3.190014581377909e-7` | `3.190014581377909e-7` |
| Failing outputs | 1,758,882 | 0 | 0 |
| Packed equals explicit repair | - | - | bit-exact |

## R31 timing

The three arms use a six-order rotating/reversed schedule in one locked process.

| Arm | Median |
|---|---:|
| Incorrect legacy | `283.104 us` |
| Explicit correct repair | `299.520 us` |
| Packed correct contract | `271.616 us` |

| Paired comparison | Median delta | Packed wins |
|---|---:|---:|
| Packed minus explicit repair | `-28.032 us` | 31/31 |
| Packed minus incorrect legacy | `-11.072 us` | 31/31 |

## SASS/resources

| Metric | Legacy | Explicit repair | Packed |
|---|---:|---:|---:|
| Static instructions | 4,088 | 5,352 | 5,056 |
| Registers | 255 | 255 | 255 |
| Stack | 64 B | 64 B | 0 B |
| LDL / STL | 15 / 30 | 16 / 16 | 0 / 0 |
| IMMA / LDSM | 256 / 32 | 256 / 32 | 256 / 32 |
| LDG / STS | 105 / 71 | 105 / 71 | 109 / 73 |
| BAR | 5 | 5 | 5 |
| FMUL / FADD | 512 / 0 | 1,600 / 1,024 | 1,544 / 1,024 |
| FP16 rounding ops | 0 | 64 consumer-side | 8 publisher-side |

Packed cubin SHA-256:

```text
6388869e98e05d4cd147a42c154e34f41353fb5f22675384cbceae5918850d88
```

The four extra LDG and two extra STS are the publication-boundary scale-table traffic. They replace repeated consumer rounding and eliminate all local-memory operations.

## Carry-forward contract

Every subsequent nested-RANGE, one-body, prefetch, publication, and reduction experiment must use `trusted_fp16_packed`. The legacy direct-`dA` arm remains useful only as uncommitted performance evidence and must never be labeled reference-correct.

No route is promoted unless it retains:

- Zero trusted-reference failures.
- Bit identity to this packed anchor when arithmetic/order is unchanged.
- The 76-word Q6 shared ABI and explicit slot/ownership ABI.
- Normalized IMMA/LDSM and resource evidence.
- Locked R31 timing evidence.
