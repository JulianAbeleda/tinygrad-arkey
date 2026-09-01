# NV Q6 one-body publication gates decision (2026-08-31)

Status: `PROMOTE_COMBINED_ONLY`

## Fixed anchor

The control is the admitted one-physical-body packed route. Its frozen
early/separate cubin SHA-256 remained:

```text
1df61553f7ebb9904108c2ed14b0c256abdce067a2ae3a1bfe45fcc86a243e1f
```

The admitted early/combined candidate cubin SHA-256 is:

~~~text
6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137
~~~

All arms retain 170 owner CTAs, nested segment/K256 ranges, the trusted packed
FP16 weight-scale contract, plane-major all-partials ABI, and the standalone
deterministic fixup.

## Isolated arms

| arm | Q8 panel-1 schedule | initial publication |
|---|---|---|
| `early_separate` | admitted early preload | separate Q6 and Q8 barriers |
| `late_separate` | load after half-0 column group 6 | separate barriers |
| `early_combined` | admitted early preload | common Q6+Q8 barrier |
| `late_combined` | load after column group 6 | common barrier |

The four-arm run preserves the rejected late variants as evidence. Executable
late scheduling was removed after Gate 3 failed its span, resource, and timing
requirements. The reproducible admission harness now contains only
`early_separate` and `early_combined`.

Q6 word 64 already contains FP32 `d` in the admitted packed contract. The old
raw-FP16-versus-FP32 publication gate is therefore recorded as
`SATISFIED_BY_TRUSTED_FP16_PACKED`, not rerun as a numerically irrelevant ABI
probe.

## Fail-closed gates

- Trusted-reference failing count is zero for every arm.
- GPU fixup is bit-exact to the CPU slot recurrence.
- Every candidate's partial workspace and final output are uint32-bit-exact to
  the frozen anchor.
- Every body retains 109 LDG, 73 STS, 256 IMMA, and 32 LDSM instructions.
- Separate and combined bodies contain five and four BAR instructions,
  respectively.
- The classifier finds exactly 18 panel-1 loads and 18 panel-1 shared stores.
- Each late arm has a normalized first-load-to-first-store span below 160
  instructions.
- Registers, stack, static local/shared allocation, LDL, and STL do not regress
  against the corresponding separate-publication base.
- Promotion requires at least 3 microseconds median end-to-end improvement and
  at least 24 wins in 31 paired rounds.

## Locked command

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock env PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_publication_gates.py \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-one-body-publication-gates-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-one-body-publication-gates-20260831/artifacts
```

The final run exited zero with `experiment_valid=true`,
`promotion_eligible=true`, and `verdict=PROMOTE_COMBINED_ONLY`.

## Correctness

All four arms were finite, GPU-fixup/CPU bit-exact, and uint32-bit-exact to the
frozen anchor for both partial workspace and final output.

| arm | maximum absolute error | mean absolute error | failures |
|---|---:|---:|---:|
| `early_separate` | 0.00067138671875 | 0.00002147154009435326 | 0 |
| `late_separate` | 0.00067138671875 | 0.00002147154009435326 | 0 |
| `early_combined` | 0.00067138671875 | 0.00002147154009435326 | 0 |
| `late_combined` | 0.00067138671875 | 0.00002147154009435326 | 0 |

## Normalized SASS and resources

Every arm retained 109 LDG, 73 STS, 256 IMMA, 32 LDSM, 1,544 FMUL,
1,024 FADD, zero FFMA, 1,024 static shared bytes, zero static local bytes, and
an exact classified 18-load/18-store panel-1 publication.

| arm | instructions | registers | stack | LDL / STL | BAR | panel-1 span |
|---|---:|---:|---:|---:|---:|---:|
| `early_separate` | 5,192 | 255 | 48 B | 12 / 12 | 5 | 2,221 |
| `late_separate` | 5,368 | 255 | 248 B | 99 / 131 | 5 | 443 |
| `early_combined` | 5,136 | 255 | 0 B | 0 / 0 | 4 | 2,225 |
| `late_combined` | 5,144 | 255 | 0 B | 0 / 0 | 4 | 1,827 |

## Locked R31 timing

| arm | main | fixup | end-to-end pair |
|---|---:|---:|---:|
| `early_separate` | 246.752 us | 25.792 us | 272.448 us |
| `late_separate` | 295.520 us | 26.432 us | 321.888 us |
| `early_combined` | 231.264 us | 25.344 us | 256.672 us |
| `late_combined` | 226.432 us | 25.568 us | 252.224 us |

Paired candidate-minus-control medians:

| comparison | main delta / wins | fixup delta / wins | pair delta / wins |
|---|---:|---:|---:|
| late vs early, separate | +48.768 us / 0 of 31 | +0.704 us / 6 of 31 | +49.440 us / 0 of 31 |
| combined vs separate, early | -15.616 us / 31 of 31 | -0.384 us / 28 of 31 | -16.192 us / 31 of 31 |
| combined vs separate, late | -68.576 us / 31 of 31 | -0.928 us / 30 of 31 | -69.664 us / 31 of 31 |

## Decision

Gate 3 is `NO_GO_LATE_PREFETCH`: the isolated late candidate missed the
less-than-160-instruction span target, increased stack from 48 to 248 bytes,
increased LDL/STL from 12/12 to 99/131, and lost every end-to-end pair.

Gate 5 is `PROMOTE_COMBINED_ONLY`: early/combined removes one barrier, removes
all stack and local operations, preserves exact traffic and arithmetic, and
wins by 16.192 microseconds median end-to-end in 31 of 31 pairs.

Late/combined is not sequentially eligible because the late scheduling gate
failed before composition. FP32 Q6 `d` publication remains
`SATISFIED_BY_TRUSTED_FP16_PACKED`.

Machine-readable result and retained artifacts are under
`docs/task_workflow/evidence/nv-q6-oracle-publication-gates-20260831/`.
