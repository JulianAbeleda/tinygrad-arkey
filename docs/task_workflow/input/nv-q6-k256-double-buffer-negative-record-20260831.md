# NV Q6 K256 double-buffer negative record

## Question

Can the promoted wide Q6_K producer reduce its phase overhead by replacing the
single 20,480-byte shared window and two barriers per K64 phase with bounded
ping/pong windows and one steady-state barrier?

## Matched experiment

Both arms execute one K256 epoch for every output tile using the canonical
packed Q6_K weights, packed Q8 records, 128 CTAs, 256 threads per CTA, unroll 2,
and the same IMMA/FP32 arithmetic.  The candidate changes only the phase
lifetime:

```text
prologue: publish phase 0
steady:   publish phase N+1 to ping/pong[N+1]
          consume phase N from ping/pong[N]
          synchronize
drain:    consume the final phase
```

This is not the rejected persistent canonical-Q6 cache.  Only two decoded
K64 publication windows exist, and each is released before reuse.

Command:

```sh
PYTHONPATH=. python3 extra/llm_research/prefill/bench_nv_q6_k256_double_buffer.py \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-k256-double-buffer-20260831/result-r31.json \
  --artifacts docs/task_workflow/evidence/nv-q6-k256-double-buffer-20260831/artifacts-r31
```

## Result

| Measurement | Unroll-2 control | Ping/pong candidate |
|---|---:|---:|
| median | 12.992 us | 13.760 us |
| minimum | 12.896 us | 13.664 us |
| static IMMA | 128 | 128 |
| static BAR | 4 | 2 |
| registers | 254 | 220 |
| stack / local | 0 / 0 | 0 / 0 |
| shared bytes | 21,504 | 41,984 |
| max absolute difference | - | 0.0 |

The candidate is bit-exact and spill-free, but its median is 5.91% slower.
It fails the investment gate and must not be integrated into the full 170-owner
Stream-K route.

## Derived constraint

Removing the overwrite barrier through decoded-phase double buffering is not
the missing lever.  The next producer experiment must reduce the canonical
Q6 load/decode or shared-publication instruction work itself; increasing the
decoded shared footprint to overlap unchanged work is insufficient.

Evidence: `docs/task_workflow/evidence/nv-q6-k256-double-buffer-20260831/result-r31.json`.
