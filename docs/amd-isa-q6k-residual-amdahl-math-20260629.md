# AMD ISA Q6_K Residual Math Note

Date: 2026-06-29

Status: audit/math note. No implementation directive, no default change.

## Purpose

This note explains the equation behind the current Q6_K residual decision. It is meant to keep the next implementation honest:

1. compute the maximum W==D gain from a candidate speedup,
2. separate proven Q6_K time from mixed reduce time,
3. avoid treating the raw 820 GB/s streaming-copy number as the decode ceiling,
4. decide whether a direct/warp Q6_K route has enough measured upside to justify implementation.

The external model matches standard performance-analysis practice:

- Amdahl's law: total speedup from optimizing one component is limited by that component's fraction of total time. Formula: `S = 1 / ((1 - p) + p / s)`, where `p` is the optimized fraction and `s` is that fraction's speedup. Source: Wikipedia, "Amdahl's law" (`https://en.wikipedia.org/wiki/Amdahl%27s_law`) and CS61C parallel performance notes (`https://notes.cs61c.org/content/parallel-performance/`).
- Roofline model: attainable performance is bounded by machine peak compute and bandwidth, and is interpreted using arithmetic intensity / bytes moved rather than raw percent-of-peak alone. Source: NERSC roofline documentation (`https://docs.nersc.gov/tools/performance/roofline/`) and JAX scaling book roofline chapter (`https://jax-ml.github.io/scaling-book/roofline/`).

Local benchmark inputs are from:

- `bench/amd-isa-backend-system-residual-ceiling/latest.json`
- `bench/amd-isa-backend-system-residual-ceiling/kernel_taxonomy.json`
- `bench/amd-isa-backend-g3-weight-promotion/latest.json`
- `bench/amd-isa-backend-weight-path-ceiling/latest.json`

## Variables

Use these variables for the current G3-promoted route.

```text
T0(ctx)      = current wall time per token at context ctx
R0(ctx)      = current tok/s = 1 / T0(ctx)

p_i(ctx)     = fraction of total time spent in bucket i
s_i          = speedup factor for bucket i after optimization
S_total      = total speedup from optimizing bucket i
R_new(ctx)   = new tok/s after optimization

B_memcpy     = measured streaming-copy bandwidth
B_q4k        = measured Q4_K G3 effective bandwidth
B_q6k        = measured Q6_K effective bandwidth
M_weights    = model weight bytes used by the naive floor
```

Current measured values:

```text
B_memcpy  = 820 GB/s
M_weights = 5.027783488 GB

R0(512)   = 103.93 tok/s
R0(1024)  = 102.04 tok/s
R0(2048)  = 99.74 tok/s
R0(4096)  = 94.44 tok/s
```

At ctx512:

```text
p_q4k_g3_gemv       = 0.427
p_q6k_gemv          = 0.136
p_lm_head           = 0.057
p_reduce_partial    = 0.224   # mixed bucket; not fully role-resolved
p_attention_total   = 0.033 + 0.020 = 0.053
p_norm_rope_elem    = 0.049

B_q4k = 650.4 GB/s
B_q6k = 503.1 GB/s
B_lm_head = 761.4 GB/s
```

At ctx4096:

```text
p_q4k_g3_gemv       = 0.394
p_q6k_gemv          = 0.125
p_lm_head           = 0.053
p_reduce_partial    = 0.207   # mixed bucket
p_attention_total   = 0.111 + 0.019 = 0.130
p_norm_rope_elem    = 0.042

B_q4k = 649.8 GB/s
B_q6k = 506.4 GB/s
B_lm_head = 761.7 GB/s
```

## Equation 1: naive memory floor

The naive streaming floor is:

```text
R_stream = B_memcpy / M_weights
```

With local measurements:

```text
R_stream = 820 GB/s / 5.027783488 GB = 163.1 tok/s
```

But this is not a valid full-decode target. It assumes the token step is one clean streaming read of the weights. The actual decode step also pays:

- quant unpack/dequant/dot work,
- Q6_K route overhead,
- lm_head,
- reductions,
- norm/rope/elementwise kernels,
- attention,
- activation/partial/output traffic,
- graph and synchronization effects.

So the useful conclusion is not "we must reach 163 tok/s." The useful conclusion is:

```text
R0 / R_stream = 103.93 / 163.1 = 63.7%
```

The missing 36.3% must be decomposed by measured bucket time, not assigned blindly to Q4_K.

## Equation 2: per-bucket effective bandwidth

For a memory-like bucket:

```text
B_i = bytes_i / time_i
```

The current Q4_K G3 route is:

```text
B_q4k = 650.4 GB/s
B_q4k / B_memcpy = 650.4 / 820 = 79.3%
```

The strongest individual Q4_K role, `down`, was reported around 707 GB/s:

```text
707 / 820 = 86.2%
```

This supports the audit conclusion that Q4_K is near its practical dequant-GEMV ceiling. It does not prove Q4_K has zero possible improvement, but it refutes Q4_K as the best next target while G3 matches owned and already reaches this bandwidth range.

Q6_K is lower:

```text
B_q6k = 503.1 GB/s
B_q6k / B_memcpy = 61.4%
B_q6k / B_q4k = 503.1 / 650.4 = 77.4%
```

So Q6_K is a plausible target: it is both a non-trivial time bucket and materially below Q4_K's effective route bandwidth.

## Equation 3: Amdahl speedup for one bucket

For a candidate optimization that speeds up fraction `p` by factor `s`:

```text
S_total = 1 / ((1 - p) + p / s)
R_new   = R0 * S_total
```

Equivalent form using "fraction removed" `r`, where `r = 1 - 1/s`:

```text
S_total = 1 / (1 - p * r)
R_new   = R0 / (1 - p * r)
```

Examples:

```text
25% faster bucket: s = 1.333, r = 0.25
50% faster bucket: s = 2.000, r = 0.50
infinite bucket speed: s = infinity, r = 1.00
```

## Equation 4: proven Q6_K-only lower bound

The proven Q6_K-related share at ctx512, excluding the mixed reduce bucket, is:

```text
p_q6k_proven = p_q6k_gemv + p_lm_head
             = 0.136 + 0.057
             = 0.193
```

This is the conservative target share. It is enough to justify investigation because it is above 10%.

If the Q6_K route were impossible-fast, the absolute upper bound would be:

```text
S_max = 1 / (1 - 0.193) = 1.239
R_max = 103.93 * 1.239 = 128.8 tok/s
```

That is not realistic; it only bounds the possible win.

More realistic cases:

```text
If r = 0.25:
S = 1 / (1 - 0.193 * 0.25) = 1.051
R = 103.93 * 1.051 = 109.2 tok/s

If r = 0.50:
S = 1 / (1 - 0.193 * 0.50) = 1.107
R = 103.93 * 1.107 = 115.1 tok/s
```

So the conservative math gives:

```text
Q6_K proven-only expected range: roughly 109-115 tok/s if the route removes 25-50% of Q6_K/lm_head time.
```

This matches Claude's `+5-12%` estimate.

## Equation 5: mixed reduce bucket sensitivity

The uncertainty is `reduce_partial`.

Measured:

```text
p_reduce_partial = 0.224
```

But this bucket is mixed. It includes some combination of:

- Q6_K coop partials and sum,
- RMSNorm reductions,
- flash/attention reductions,
- possibly other `r_*` reductions.

Define:

```text
a = fraction of reduce_partial actually caused by Q6_K coop partials+sum
```

Then total Q6_K-affected share is:

```text
p_q6k_total(a) = p_q6k_proven + a * p_reduce_partial
               = 0.193 + a * 0.224
```

Sensitivity table at ctx512:

| assumption | a | p_q6k_total | if r=25% | tok/s | if r=50% | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| no reduce belongs to Q6_K | 0.00 | 0.193 | 1.051x | 109.2 | 1.107x | 115.1 |
| quarter reduce belongs to Q6_K | 0.25 | 0.249 | 1.066x | 110.8 | 1.142x | 118.7 |
| half reduce belongs to Q6_K | 0.50 | 0.305 | 1.082x | 112.4 | 1.180x | 122.7 |
| all reduce belongs to Q6_K | 1.00 | 0.417 | 1.116x | 116.0 | 1.264x | 131.4 |

This is why the next audit must role-resolve the reduce bucket before a large implementation. The expected gain ranges from "nice" to "large" depending on `a`.

## Equation 6: ctx4096 check

At ctx4096:

```text
R0(4096) = 94.44 tok/s
p_q6k_proven = 0.125 + 0.053 = 0.178
p_reduce_partial = 0.207
```

Conservative cases:

```text
If r = 0.25:
S = 1 / (1 - 0.178 * 0.25) = 1.047
R = 94.44 * 1.047 = 98.9 tok/s

If r = 0.50:
S = 1 / (1 - 0.178 * 0.50) = 1.098
R = 94.44 * 1.098 = 103.7 tok/s
```

Sensitivity including reduce:

| assumption | a | p_q6k_total | if r=25% | tok/s | if r=50% | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| no reduce belongs to Q6_K | 0.00 | 0.178 | 1.047x | 98.9 | 1.098x | 103.7 |
| quarter reduce belongs to Q6_K | 0.25 | 0.230 | 1.061x | 100.2 | 1.130x | 106.7 |
| half reduce belongs to Q6_K | 0.50 | 0.282 | 1.076x | 101.6 | 1.164x | 109.9 |
| all reduce belongs to Q6_K | 1.00 | 0.385 | 1.106x | 104.4 | 1.239x | 117.0 |

The ctx4096 math is consistent with ctx512: Q6_K route work is a plausible 5-10% conservative win, with larger upside only if reduce_partial is mostly Q6_K-owned.

## What The Current Evidence Proves

The evidence proves:

1. Q4_K G3 is not the live residual while it matches owned and runs at 650-707 GB/s.
2. The 820 GB/s copy number is a sanity ceiling, not a reachable full-decode target.
3. Q6_K/lm_head is a real measured bucket:

```text
ctx512: 19.3% GPU-time
ctx4096: 17.8% GPU-time
```

4. Q6_K effective bandwidth is below Q4_K:

```text
503 GB/s vs 650 GB/s aggregate Q4_K
```

5. Therefore Q6_K route efficiency is the best next suspect.

## What The Current Evidence Does Not Prove

The evidence does not yet prove:

1. that most of `reduce_partial` belongs to Q6_K,
2. that a direct/warp Q6_K route will remove 25-50% of Q6_K time,
3. that Q6_K has the same route ceiling as Q4_K,
4. that the `lm_head` row is slow; its measured bandwidth is already high at ~761 GB/s,
5. that +12% W==D is guaranteed.

So the correct engineering stance is:

```text
Q6_K is the next target to audit/prove, not yet a guaranteed implementation win.
```

## Required Next Gate

Before implementing a direct/warp Q6_K kernel, run a role-resolved proof gate:

```text
extra/amd_isa_q6k_residual_math_gate.py
```

Required artifacts:

```text
bench/amd-isa-backend-q6k-residual-math/latest.json
bench/amd-isa-backend-q6k-residual-math/summary.md
bench/amd-isa-backend-q6k-residual-math/reduce_role_split.json
bench/amd-isa-backend-q6k-residual-math/amdahl_sensitivity.json
bench/amd-isa-backend-q6k-residual-math/q6k_route_candidates.json
```

The gate must answer:

1. Which exact `reduce_partial` kernels belong to Q6_K coop partials+sum?
2. What is `a`, the fraction of `reduce_partial` attributable to Q6_K?
3. What is `p_q6k_total(a)` at ctx512 and ctx4096?
4. What W==D range follows from `r in {0.25, 0.50, 1.00}`?
5. Is Q6_K slow because of the coop+reduce route, or because the Q6_K quant math/layout is intrinsically worse?
6. Is `lm_head` actually improvable, or already near its own practical ceiling?

Promotion rule:

```text
If p_q6k_total(a) >= 0.10 and the direct/warp route has a credible r >= 0.25:
  select direct/warp Q6_K implementation.
Else:
  do not implement Q6_K yet; reclassify the residual.
```

## Claude Prompt

Use this exact prompt for the next audit:

```text
You are working in /home/ubuntu/tinygrad-arkey.

Task: build an audit-only Q6_K residual math/proof gate. Do not implement kernels, do not change defaults, do not modify autogen files.

Background:
- Current best route is G3-promoted Q4_K, speed-equivalent to owned.
- Current best tok/s: ctx512=103.93, ctx1024=102.04, ctx2048=99.74, ctx4096=94.44.
- System residual audit selected Q6_K/lm_head because q6k_gemv+lm_head is 19.3% GPU-time at ctx512 and 17.8% at ctx4096.
- The weak point is reduce_partial: 22.4% @ctx512 / 20.7% @ctx4096 is mixed and not role-resolved.

Build extra/amd_isa_q6k_residual_math_gate.py.

Inputs:
- bench/amd-isa-backend-system-residual-ceiling/latest.json
- bench/amd-isa-backend-system-residual-ceiling/kernel_taxonomy.json
- bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
- bench/amd-isa-backend-g3-weight-promotion/latest.json

Outputs:
- bench/amd-isa-backend-q6k-residual-math/latest.json
- bench/amd-isa-backend-q6k-residual-math/summary.md
- bench/amd-isa-backend-q6k-residual-math/reduce_role_split.json
- bench/amd-isa-backend-q6k-residual-math/amdahl_sensitivity.json
- bench/amd-isa-backend-q6k-residual-math/q6k_route_candidates.json

Required math:
- Use Amdahl's law: S = 1 / ((1 - p) + p / s).
- Also report removed-fraction form: S = 1 / (1 - p*r), where r = 1 - 1/s.
- Compute conservative p = q6k_gemv + lm_head.
- Compute p(a) = q6k_gemv + lm_head + a * reduce_partial for a in {0, 0.25, 0.5, 1.0}.
- Report tok/s projections for r in {0.25, 0.50, 1.00} at ctx512 and ctx4096.
- Role-resolve reduce_partial as far as current route_attribution allows. If not possible, say so and keep reduce as sensitivity, not proof.

Required verdicts:
- AMD_ISA_Q6K_RESIDUAL_PASS_DIRECT_ROUTE_JUSTIFIED if role-resolved Q6_K affected share >=10% and credible removable fraction >=25%.
- AMD_ISA_Q6K_RESIDUAL_INCONCLUSIVE_REDUCE_NOT_ROLE_RESOLVED if reduce attribution is too mixed and q6k_gemv alone does not prove route upside.
- AMD_ISA_Q6K_RESIDUAL_PASS_RECLASSIFY_TARGET if another bucket has higher proven W==D upside.

Discipline:
- Do not claim reduce_partial belongs to Q6_K unless role attribution proves it.
- Do not treat 820 GB/s memcpy as the full decode ceiling.
- Do not include Q4_K layout reshuffle as live unless new data refutes G3 parity.
- Do not implement the Q6_K route in this phase.
```

