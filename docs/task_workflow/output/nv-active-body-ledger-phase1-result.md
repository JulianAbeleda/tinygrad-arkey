# Common-protocol active-body ledger: phase 1

## Decision

Flash score and native norms are not tinygrad body debts. Vocabulary is the
first old losing row that survives a common CUDA/CUPTI boundary.

| region | tinygrad exact CUPTI body | charged llama CUPTI body | tiny - llama | verdict |
|---|---:|---:|---:|---|
| Flash score, hot | 4.191 us/layer | 4.264 us/layer | -0.074 us/layer | tiny wins |
| Flash score, 96-MiB disturbed | 4.256 us/layer | 4.384 us/layer | -0.128 us/layer | tiny wins |
| native 4096 norms | 110.880 us/token (55 x 2.016) | 203.778 us/token | -92.898 us/token before provider allocation | tiny wins |
| vocabulary main body | 317.402 us/token | 300.930 us/token | +16.472 us/token | real body debt |
| vocabulary plus llama quant | 317.402 us/token | 301.602 us/token | **+15.800 us/token** | clean next target |

The tinygrad norm row excludes the 18-call shared-Q8 provider. That provider
also owns Q/K/V quantization and cannot be charged wholly to norms. Even an
intentionally over-conservative charge of its complete old 31.360-us HCQ row
would leave tinygrad norms about 61.5 us/token ahead; the old +28-us norm debt
is therefore closed without depending on an exact provider allocation.

The vocabulary comparison is like-for-like enough to retain: both rows perform
the complete quantized 151936 x 4096 projection. Llama's separate 0.672-us
vocabulary activation quantization is charged to its lifecycle. Tinygrad's
native argmax remains a separate tail and is not hidden in the 15.8-us body
difference.

## Endpoint translation

The installed endpoint remains 4060.523 us/token / 246.274 tok/s. Recovering
the full measured vocabulary body difference would imply:

```text
4060.523 - 15.800 = 4044.723 us/token
1e6 / 4044.723     = 247.236 tok/s
```

This is a body ceiling, not a booked wall result. A vocabulary construction
must still pass exactness and an unprofiled reverse token-wall bracket.

## Exact tinygrad authorities

- `rmsnorm_native_1_4096`: cubin SHA-256
  `e9615606370f73faa85245380cb6f05661536c3909a727f64db1dbc3b252bc24`,
  grid `(1,1,1)`, block `(32,16,1)`.
- `q6k_gen_coop_151936_4096_inkernel`: cubin SHA-256
  `46ae07b81bad8bd00b1c06bb8fe5af9abbcf0c3f85d936423fdf79687bdaf4a1`,
  grid `(75968,1,1)`, block `(2,16,1)`.

## Consequence

The active-body campaign should proceed as follows:

1. Treat vocabulary's approximately 15.8 us/token as the first confirmed body
   target and test service-rate constructions there.
2. Continue common-protocol reconciliation for Q/O and the projection families.
3. Keep native norms and Flash closed unless a new wall-level mechanism appears.
4. Do not use the old mixed-boundary 47.570-us device node-sum delta as a
   recovery ledger.

## Evidence

- `docs/task_workflow/evidence/nv-active-body-ledger-20260827/tiny-vocab-norm-capture.json`
- `docs/task_workflow/evidence/nv-active-body-ledger-20260827/tiny-vocab-norm-cupti.json`
- `docs/task_workflow/evidence/nv-lifecycle-recovery-tests-20260826/llama-pdl-ab/pdl-off-dag.json`
- `docs/task_workflow/output/nv-flash-score-common-protocol-result.md`
