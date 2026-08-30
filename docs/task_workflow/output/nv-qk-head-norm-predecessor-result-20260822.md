# NV Q/K head-norm predecessor-conditioned result (2026-08-22)

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

## Verdict

`LAUNCH_DISPATCH_DOMINANT`. The Q/K head-norm gap is **not** compiler codegen,
**not** RMSNorm body arithmetic, and **not** L2 cache state in the production
steady-state path. It is per-kernel NV queue dispatch overhead that the
tinygrad HCQ profile folds into each kernel's own duration slot.

The NVRTC-compiler hypothesis is rejected: the exact production cubins, run
through `cuModuleLoad` + `cuLaunchKernel` under nsys, execute in about
`1.18 us`. Llama's nsys kernel body is `1.15 us` (Q) and `1.13 us` (K). The
`2.2x`-style gap in the census was an apples-to-oranges comparison between
tinygrad's profiled command wall and llama's pure GPU kernel time.

## Measurements

The exact production cubins were captured without changing production code,
then launched with their captured grid/block geometry in a CUDA driver
context. Pure kernel medians from nsys `cuda_gpu_kern_sum`:

| kernel | grid | block | nsys mean | nsys median |
| --- | ---: | ---: | ---: | ---: |
| `reduce_output_rmsnorm_32_128` (Q) | 32x1x1 | 4x8x1 | 1.190 us | 1.184 us |
| `reduce_output_rmsnorm_8_128` (K) | 8x1x1 | 2x16x1 | 1.196 us | 1.184 us |

The retained production profile reports Q/K command walls of `2.687 us` and
`2.506 us`, while llama's nsys PDL-off kernel durations are `1.147 us` and
`1.134 us`. The standalone nvcc microgate already showed the same source at
`1.120 us` hot (llama copy `1.088 us`), and producer-conditioned (`fill`)
timing was equal for both sides. Only the L2-evicted (`flush`) case favored
llama, and that is not the steady-state decode path.

The nvcc proxy hot medians from `nv_qk_hot_nsys.sqlite`:

| kernel | nsys median |
| --- | ---: |
| tinygrad `reduce_output_rmsnorm_32_128` | 1.120 us |
| tinygrad `reduce_output_rmsnorm_8_128` | 1.120 us |
| llama `rms_norm_qk` copy | 1.088 us |

## Decomposition

Subtracting the exact pure-GPU body from the production command wall and
reconciling against the frozen census:

```text
Q command wall              2.687 us
Q pure GPU body            -1.190 us
Q dispatch overhead        =1.497 us/kernel

K command wall              2.506 us
K pure GPU body            -1.196 us
K dispatch overhead        =1.310 us/kernel

dispatch overhead x 72      101.041 us   (96.4%)
residual body gap vs llama    3.790 us   ( 3.6%)
reconciled Q/K gap          104.831 us   (= census 55.455 + 49.376)
```

The residual `3.79 us` body gap is the true RMSNorm arithmetic difference at
the current association. The other `101.04 us` is launch/dispatch overhead
paid per standalone Q/K norm kernel.

## Conclusion

The Q/K head-norm recovery ceiling is launch-path work, not a faster RMSNorm
algorithm. Fusing the Q/K norm into its producer or consumer, or otherwise
removing the two standalone launches per layer, is the only mechanism that
can recover the `~101 us`; changing the reduction itself has a hard ceiling of
about `3.8 us` and risks flipping token SHA.

No production model, renderer, scheduler, or runtime code was changed. The
census accounting remains frozen at SHA
`0326f0d21e10059a92196a439431f5bd58fb04353a6b20d972e94b3cece494cf`.

## Next Gate

The next experiment is a production candidate that eliminates the standalone
Q/K norm launches. It must pass a control/candidate/control wall bracket with
matching token SHA before promotion. Because the current `reduce_output`
association is bitwise-pinned to the ordinary RMSNorm, the candidate may keep
the exact fp32 reduction order but fold the launch away, or it must reproduce
the token SHA in the bracket.

## Evidence

- Exact-cubin summary: `docs/task_workflow/evidence/nv-qk-head-norm-predecessor-20260822/exact-cubin.json`
- Retained cubins: `reduce_output_rmsnorm_32_128.cubin`, `reduce_output_rmsnorm_8_128.cubin`
- nsys reports: `nv_qk_q_exact.sqlite`, `nv_qk_k_exact.sqlite`
- nvcc proxy hot report: `nv_qk_hot_nsys.sqlite`
- Capture split: `docs/task_workflow/evidence/nv-qk-head-norm-predecessor-20260822/capture-split.json`
- Microgate: `docs/task_workflow/evidence/nv-qk-head-norm-predecessor-20260822/microgate.json`
- Tools: `nv_cubin_capture.py`, `nv_cubin_ncu_launcher.py`, `nv_qk_head_norm_microgate.py`
