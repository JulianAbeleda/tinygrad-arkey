# NV 227 push: shared-Q8 Q4/Q4 K/V producer result

Date: 2026-08-24
Base checkpoint: `b7ddda44b`
GPU: RTX 5090 (`NV sm_120`), model: Qwen3-8B-Q4_K_M

## Outcome

The dual-output cooperative Q4/Q8 K/V producer is promoted for the nine
Q4/Q4 blocks inside the existing shared-Q8 lease. It is bit-exact, removes
exactly nine decode launches, and passes a stabilized reps=9 production wall
bracket with the ordinary pair and terminal cache sink enabled in every arm.

The same-session bracket recovers `15.915 us/token`: `221.517` to `222.301
tok/s`, or `+0.784 tok/s`. The conservative no-rollback reps=15 endpoint is
`4.515396 ms/token = 221.465 tok/s`, leaving `110.109 us/token` or `5.535
tok/s` to 227. Only `2.752 us/token` (`+0.135 tok/s`) is visible across the
two conservative endpoint sessions, so the larger causal bracket delta is
not added to the conservative floor.

## Native gate

The CUDA-event control launches the installed direct cooperative Q4/Q8
consumer twice. The candidate launches one producer with separate K and V
outputs.

| row | control pair | candidate pair | change |
| --- | ---: | ---: | ---: |
| median, 500 passes x 9 | 4.339776 us | 3.084608 us | -1.255168 us (-28.9%) |
| projected over nine pairs | - | - | -11.296512 us/token ceiling |

All 2048 fp32 words match bit-for-bit. Both kernels use 40 registers; the
candidate uses 32 bytes shared memory, one barrier, and zero spill
loads/stores.

## Production profile

The candidate preserves the existing Q8 provider and producer-owned cache
sink. K and V remain separate direct AFTER outputs, so the K RMSNorm+RoPE
semantic route does not expand.

| row | control | candidate | delta |
| --- | ---: | ---: | ---: |
| nodes | 471 | 462 | -9 |
| node sum | 4284.000 us | 4269.696 us | -14.304 us |
| device union | 4281.000 us | 4257.750 us | -23.250 us |

The union delta is larger than the isolated ceiling and is not treated as a
causal projection. The installed profile reports `462` nodes, `4271.072 us`
node sum, `4258.500 us` union, and `13.202 us` overlap.

## Wall qualification

The first reps=7 bracket was retained as a no-go despite a positive midpoint:

```text
control A  4.495592 ms/token
candidate  4.496002 ms/token
control C  4.508255 ms/token
midpoint   4.501923 ms/token
recovery       5.921 us/token
verdict    NO_GO_WALL (candidate missed control A by 0.410 us)
```

The stabilized reps=9 confirmation passed both controls:

```text
control A  4.522569 ms/token
candidate  4.498406 ms/token
control C  4.506073 ms/token
midpoint   4.514321 ms/token
recovery      15.915 us/token
verdict    WALL_PASS
```

All profile and wall arms preserve their token-stream hashes.

## Promotion boundary

`decode-shared-q8-q4kv-pair-route-policy.json` promotes only `NV sm_120`.
The loader additionally requires the existing shared-Q8 lease, the promoted
cooperative direct-output Q4 consumer, and exact Q4_K K and V primitives.
`TINYGRAD_SHARED_Q8_Q4KV_PAIR_DISABLE=1` restores the two direct consumers at
model load. Mixed Q4/Q6 blocks, ordinary blocks, and other targets miss.

Evidence: `docs/task_workflow/evidence/nv-227-shared-q8-kv-pair-20260824/`.

Verdict: `PROMOTED_SHARED_Q8_Q4Q4_KV_PAIR_221_465_TOK_S_227_NOT_REACHED`.
