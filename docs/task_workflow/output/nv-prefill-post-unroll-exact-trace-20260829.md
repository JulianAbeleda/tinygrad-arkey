# NVIDIA pp512 post-unroll exact lifecycle trace — 2026-08-29

Status: **PASS for the captured device-timestamp accounting; not a wall-performance run.**

The current combined gate/up + K + Q/O route with `NV_COMPILER_Q4_IMMA_UNROLL=4` was captured under `PROFILE=1` while holding `/tmp/gpu-bench.lock`. The final six HCQ graph segments were parsed as one complete pp512 invocation. The route arm itself reported exact census and 20/20 deep replay, with token 198 and no overlays/copies/fixups.

## Exact route and clock closure

| item | measured |
|---|---:|
| launches classified | 1,449 |
| unknown launches | 0 |
| device span | 541,417.728 us |
| device interval union | 541,401.088 us |
| device idle inside span | 16.640 us |
| overlap / duplicate charge | 0 us |
| closure | 541,417.728 = 541,401.088 + 16.640 us |

The profile exporter adds substantial instrumentation cost (the arm's profiled host wall was 541.026 ms minimum), so this trace is used for event accounting only. It is not substituted for the unprofiled 69.205 ms candidate wall.

## Classified active service

| region | tiny active us | llama active us | tiny − llama us |
|---|---:|---:|---:|
| input/embed and setup | 6.112 | 0 | +6.112 |
| RMSNorm/conversion | 1,376.896 | 2,145.511 | −768.615 |
| Q | 4,492.064 | 2,514.450 | +1,977.614 |
| K | 2,221.312 | 1,251.141 | +970.171 |
| V | 0 | 1,053.720 | −1,053.720 |
| Flash score/reduction | 3,330.976 | 1,657.447 | +1,673.529 |
| O | 4,489.760 | 2,397.213 | +2,092.547 |
| gate | 10,344.864 | 6,086.529 | +4,258.335 |
| up | 10,603.680 | 6,111.329 | +4,492.351 |
| activation/multiply | 499.520 | 903.843 | −404.323 |
| down / residual/support | 501,106.784 | 6,940.482 + 1,379.275 | not uniquely separable in this trace |
| vocabulary | 2,919.808 | 313.154 | +2,606.654 |
| output/token | 9.312 | 0 | +9.312 |

The tiny trace has zero V and down-specific custom identities because those routes are not admitted in this arm; their executed kernels are correctly classified into the common support bucket. Therefore a down-only debt cannot be claimed from this capture. The large support bucket includes the ordinary FP16 lifecycle and is the dominant measured service region, but ownership must be split by a richer graph-node/shape map before it can drive an optimization.

## Ranked measured debt ledger

1. **Support ownership / boundary attribution — substrate required.** The trace has 937 support launches and 501,106.784 us active time, but the current parser cannot distinguish down from residual/RoPE/KV/support for this admitted route. Add graph-node or argument-shape ownership, then rerun the same locked capture.
2. **Gate + up — measured active debt +8,750.686 us.** The unroll4 route is exact and retained; this is the largest separable dense debt after support.
3. **Q + O — +4,070.161 us combined.**
4. **Vocabulary — +2,606.654 us.**
5. **Flash — +1,673.529 us.**
6. **K — +970.171 us.**
7. **Input/output — +15.424 us combined.**

Negative rows are tinygrad wins in active service and are not optimization targets. These are active-service differences only; they are not scaled to unprofiled wall and do not imply token-rate recovery.

## Artifacts

- `docs/task_workflow/evidence/nv-prefill-post-unroll-trace-20260829/candidate.profile.jsonl`
- `docs/task_workflow/evidence/nv-prefill-post-unroll-trace-20260829/candidate.json`
- `docs/task_workflow/evidence/nv-prefill-post-unroll-trace-20260829/candidate.accounting.json`
- llama authority: `docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/llama/llama-accounting.json`
