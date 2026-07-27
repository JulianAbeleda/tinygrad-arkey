# End-to-end validation of the lowering refactor + cut

Branch `refactor/lowering-architecture` @ `7173f247f` vs master `5574e409c`. All numbers measured on the same
machine, same session, minutes apart, each run serialised under `flock /tmp/gpu-bench.lock`. Nothing here is
compared against a remembered number.

This was the first execution evidence on the branch. Everything before it — 30 AMD kernels byte-identical, 10 CPU
fingerprints, 981 collapsed pass steps, 12 variant-arm kernels, the unit suite at parity — is compile-only by
construction and cannot detect a change that survives codegen unchanged but behaves differently at runtime.

## Correctness — PASS

`extra/qk/prefill/prefill_flash_e2e_parity.py`, real weights, next-token argmax, fused route vs SDPA baseline.

| model | SDPA | fused | verdict |
|---|---|---|---|
| 8B  Qwen3-Q4_K_M | 198 | 198 | MATCH, `AUTHORITY_GATE: PASS` |
| 14B Qwen3-Q4_K_M | 90310 | 90310 | MATCH, `AUTHORITY_GATE: PASS` |

14B was run on the DEFAULT path, deliberately not with `TINYGRAD_PREFILL_PACKED_WMMA=0`, despite that harness's
docstring still recommending it. That docstring predates fix `7463a6774`; following it is what produced two
earlier phantom "14B prefill faults". The default path completed cleanly.

## Prefill throughput — no regression

`extra/qk/prefill/prefill_whole_synced.py`, 8B, K=8 warmups=4 rounds=3, `GRAPH_GEMM=True`.

| start_pos | master | branch | delta |
|---|---|---|---|
| 0 | 3744 | 3760 | +0.43% |
| 512 | 3638 | 3639 | +0.03% |
| 1024 | 3529 | 3542 | +0.37% |
| 2048 | 3323 | 3331 | +0.24% |
| 3584 | 3076 | 3081 | +0.16% |

Whole-prefill: branch 3760/3698/3589/3384 vs master 3744/3690/3579/3376 at 512/1024/2048/4096.

Branch is marginally ahead everywhere. This is reported as **no regression**, not as a speedup: sub-0.5% is
inside run-to-run variance, and a consistent sign at that magnitude is not evidence of a real effect.

## Decode throughput — no regression, established by re-running

`extra/qk/bench.py --decode`, 8B. The first branch run came in below master at all four contexts, which on one
A/B pair is indistinguishable from a small real regression. It was re-run rather than argued about.

| ctx | master | branch #1 | branch #2 | delta #1 | delta #2 |
|---|---|---|---|---|---|
| 128 | 95.45 | 94.83 | 95.09 | -0.65% | -0.38% |
| 512 | 113.99 | 113.58 | 114.00 | -0.36% | **+0.01%** |
| 1024 | 111.62 | 111.38 | 111.76 | -0.22% | **+0.13%** |
| 4096 | 102.84 | 102.69 | 102.96 | -0.15% | **+0.12%** |

The sign reverses at 512/1024/4096 between the two branch runs, so run #1's deficit was variance. Route
selection is also correct and unchanged: `sdpa` at ctx128, `flash` at 512 and above.

**The one residual:** ctx128 read below master in BOTH branch runs (-0.65%, -0.38%). Branch-to-branch spread
there is ~0.3%, so it is inside noise, but it is the single place a sub-percent effect cannot be ruled out from
this data. It is the `sdpa` (non-flash) route. Not actionable at this magnitude; recorded rather than rounded
away, and a third pair would settle it if it ever matters.

## What this validates

The refactor AND the ~900-line cut together — the measurements were taken after all four cut tiers. If a
regression had appeared, the tiers are separate commits and bisectable:

    5ce3970b4  BufferizeOpts home, buffer_plan deleted
    c9b9ce929  passes.py / kernel_specs.py / plan.py dead half
    3ba50a236  recorders + kernel_pipeline builders
    7173f247f  private helpers + re-export shims
