# NV Q6 Oracle Fixup Geometry Gate 8 Decision

Date: 2026-08-31

## Decision

`RETAIN_ONE_TILE_256`

The isolated four-slice scattered fixup is correct and wins every paired timing sample, but its `1.024 us` R31 median improvement does not clear the fail-closed `3.000 us` materiality gate. Grid granularity alone is therefore not an admitted reduction. No builder/main file was changed, no harness repair was used, and no commit was made.

The result falsifies CTA underfill as the dominant explanation for the pinned fixup residual. Relative to the pinned `25.056 us` all-partials fixup and `8.640 us` standalone lower bound, the paired gain removes about `6.24%` of the `16.416 us` residual. The remaining candidate median is `23.840 us`, still `15.200 us` above that lower bound. The next safe isolated experiment should target destination-store coalescing while preserving descriptor order, exact fold order, scratch traffic, and the unique-writer contract.

## Released command

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env NV_Q6_GATE8_GPU_RELEASED=1 NV_Q6_GPU_LOCK_HELD=1 PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_fixup_geometry.py \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-fixup-geometry-gate8-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-fixup-geometry-gate8-20260831/artifacts
```

- GPU lock: `/tmp/nv-q6-oracle-gpu.lock`, acquired successfully with `flock -w 1200`.
- Release interlock: `NV_Q6_GATE8_GPU_RELEASED=1`, observed released.
- Harness lock assertion: `NV_Q6_GPU_LOCK_HELD=1`.
- Process exit: `1`, the expected fail-closed performance verdict rather than a correctness, compilation, or lock failure.

## Frozen A/B contract

| Property | Admitted control | Four-slice candidate |
|---|---:|---:|
| Grid | `(256,1,1)` | `(128,4,1)` |
| Block | `(256,1,1)` | `(128,1,1)` |
| Outputs per active thread | 64 | 32 |
| Active blocks | 256 | 512 |
| Warps | 2,048 | 2,048 |
| Output policy | all partials | all partials |
| Descriptor order | frozen 294-entry order | identical |
| Scratch layout | `slot*16384+z` | identical |
| Destination mapping | `(mt*128+mc)*4096+nt*128+wr` | identical |
| Destination writes | 2,097,152 fp32 | 2,097,152 fp32 |
| Reset | none | none |

Candidate slices are the disjoint intervals `[0,4096)`, `[4096,8192)`, `[8192,12288)`, and `[12288,16384)`. Within each slice, `lane + 128*iteration` is a bijection, proving one writer for every destination element. There are no atomics, memory barriers, spin loops, counters, resets, or inter-CTA dependencies.

Scratch read traffic is unchanged: one fp32 value for every valid descriptor and output element. The main producer and its all-partials scratch writes are shared by the A/B. Scratch layout SHA-256 is represented by the frozen descriptor SHA-256 `e366033b4945a810c441e5b50d6e7ecc9e8e238472f40142e52f1a1177f42983`; the observed partials SHA-256 was `14d509484f4db97f1196aa958632d77e1e12abfe06797a79b21b08c2cf1ae3d8` before and after both fixups.

## Correctness gates

| Check | Result |
|---|---:|
| Candidate versus admitted control | bit exact |
| Candidate repeat | bit exact |
| Control versus CPU ordered fold | bit exact |
| Scratch partials unchanged | pass |
| Finite output | pass |
| Trusted max absolute error | `0.00067138671875` |
| Trusted mean absolute error | `0.000021467494661919773` |
| Trusted failing elements | `0` |

Control and candidate output SHA-256 are both `51ab501f46be3f395263a7655bd204c2397d385496eb3f1d440a3c3e4ef11205`.

## SASS and resources

| Metric | Admitted control | Four-slice candidate | Candidate gate |
|---|---:|---:|---:|
| Instructions | 672 | 64 | informational |
| Registers | 255 | 22 | `<=84`, pass |
| Stack bytes | 16 | 0 | `0`, pass |
| Static shared bytes | 0 | 0 | pass |
| Static local bytes | 0 | 0 | pass |
| `LDG` | 196 | 7 | informational |
| `STG` | 64 | 1 | informational |
| `FADD` | 192 | 3 | informational |
| `IADD` | 2 | 5 | informational |
| `IMAD` | 8 | 11 | informational |
| `ISETP` | 8 | 3 | informational |
| `BRA` | 10 | 2 | informational |
| `LDL` | 4 | 0 | `0`, pass |
| `STL` | 8 | 0 | `0`, pass |
| `ATOM` | 0 | 0 | `0`, pass |
| `MEMBAR` | 0 | 0 | `0`, pass |
| `BAR` | 0 | 0 | `0`, pass |

Binary identities:

| Kernel | Symbol | Source SHA-256 | CUBIN SHA-256 |
|---|---|---|---|
| Main anchor | `nv_q6_oracle_broad_cta_prefetch_combined_publish_oracle_publisher_trusted_fp16_packed_ws_segments_in_cta_streamk_s0` | `40cca7e5d0b11d37c7df5843206eaf0a27cbb128dd556f879f7b4f43ace324d3` | `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137` |
| Control fixup | `nv_q6_ordered_fixup_all_partials` | `232cdd9fd88d51326419983712f08fdf2962f75fec77119a3faf14bfc7d582a4` | `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514` |
| Candidate fixup | `nv_q6_ordered_fixup_all_partials_four_slice_scatter` | `b77f3beaa2ba9ef3e90e21dd115f517a66a6ff0f96e6dc9c1b2142563d21bb4f` | `2b3eb17efae198c2137bac107a52de070aa5c3c045d3619f1fbdc18e55e5cbbb` |

## Timing

All values are microseconds. R9 is the first nine pairs of the R31 sequence.

| Component | R9 min / median / max | R31 min / median / max |
|---|---:|---:|
| Shared main | `228.800 / 229.952 / 231.456` | `228.800 / 230.208 / 231.456` |
| Control fixup | `24.544 / 24.864 / 25.056` | `24.544 / 24.896 / 25.248` |
| Candidate fixup | `23.712 / 23.840 / 24.192` | `23.616 / 23.840 / 24.192` |
| Control total | `253.664 / 254.944 / 256.512` | `253.664 / 255.072 / 256.512` |
| Candidate total | `252.608 / 253.888 / 255.648` | `252.608 / 254.016 / 255.648` |

| Paired gate | R9 | R31 |
|---|---:|---:|
| Candidate minus control median | `-0.896 us` | `-1.024 us` |
| Candidate minus control min / max | `-1.248 / -0.640 us` | `-1.408 / -0.448 us` |
| Candidate wins | `9/9` | `31/31` |
| Paired MAD | n/a | `0.160 us` |
| Required improvement | `>=3.000 us` | `>=max(3.000, 3*MAD)=3.000 us` |
| Required wins | `>=7/9` | `>=24/31` |
| Verdict | fail magnitude | fail magnitude |

## Gate interpretation and next safe reduction

The all-negative paired distribution proves that increasing independent CTAs reduces the current scattered fixup slightly. The gain is stable but too small to explain the residual: the experiment passes correctness, liveness, binary-resource, and win-count gates while failing only the predeclared magnitude gates.

Ranked next experiment: keep the admitted control and exact fold invariant, but isolate destination-store coalescing in a scratch-output staging harness. The experiment must preserve the 294-descriptor traversal, all-partials scratch reads, arithmetic sequence, unique writer, zero reset, and no synchronization. Admit only if it remains bit exact, has stack/`LDL`/`STL` zero, uses at most 84 registers, contains no `ATOM`/`MEMBAR`/`BAR`, and clears the same R9/R31 paired gates. Do not combine it with geometry or main-kernel changes until that isolated result passes.

## Evidence

- Result: `docs/task_workflow/evidence/nv-q6-oracle-fixup-geometry-gate8-20260831/result.json`
- Main CUBIN: `docs/task_workflow/evidence/nv-q6-oracle-fixup-geometry-gate8-20260831/artifacts/main_gate8_anchor/main_gate8_anchor.cubin`
- Control CUBIN: `docs/task_workflow/evidence/nv-q6-oracle-fixup-geometry-gate8-20260831/artifacts/fixup_gate8_control/fixup_gate8_control.cubin`
- Candidate CUBIN: `docs/task_workflow/evidence/nv-q6-oracle-fixup-geometry-gate8-20260831/artifacts/four_slice_128/four_slice_128.cubin`
