# NV producer-owned K/V cache sink scope

Date: 2026-08-24
Repo: `/home/ubuntu/tinygrad-arkey`
Target: Qwen3-8B-Q4_K_M single-token decode on NV `sm_120`

## Decision this scope must make

Test whether the terminal K producer can own the current-token cache write and
absorb the already-final V value, deleting the 36 generic K/V store joins
without changing cache bytes or token output. This is the smallest executable
producer-side construction. It does not claim the full measured K/V-to-flash
readiness envelope.

The current installed graph has 516 nodes. Its generic `E_8_8_16_2` K/V cache
store occurs 36 times and contributes 42.336 us/token at the authoritative
median. Flash score begins a median 89.250 us/token after the later of the K/V
producer dependencies and the cache store. The 131.586 us sum is an upper
bound on the boundary, not expected recovery.

## Phase 1 construction

For each decode block, replace this terminal chain:

```text
K projection -> reduce_output_rmsnorm_rope_8_128 -> fp32 K
V projection ------------------------------------> fp32 V
fp32 K + fp32 V -> generic cache store -> cache AFTER -> flash score
```

with:

```text
K projection -> reduce_output_rmsnorm_rope_kv_cache_8_128 -> cache AFTER
V projection ----------------------------^                    |
                                                               +-> flash score
```

The fused K terminal producer retains the installed RMSNorm reduction
association and RoPE arithmetic verbatim. Its epilogue casts K and V exactly
once to the cache's own dtype and writes only
`cache[0|1, 0, kv_head, start_pos, element]`.

This first construction intentionally hosts both stores in the K terminal
producer. It removes the launch while retaining the existing K/V join at that
producer. Independent K- and V-producer writes are a follow-up only if the
profile shows that waiting for V delays the K producer or leaves the readiness
gap intact.

## Exactness contract

The candidate must pass all of these before a production route can be opened:

1. For cache dtypes fp16 and fp32, every byte of the full cache allocation
   after the candidate launch equals a control consisting of the installed
   fused K RMSNorm+RoPE producer followed by the legacy K/V cache store.
2. Test `start_pos` at the first slot, an interior slot, and the last slot.
   Sentinel bytes outside the selected slot must remain unchanged.
3. Test nontrivial deterministic K/V values plus signed zero and finite fp16
   extrema. NaN payload preservation is not part of the model contract.
4. The production candidate must retain the current token-stream SHA and the
   ordinary path must remain unchanged when its closed-default admission is
   absent.

## Structural and performance gates

The production candidate advances only if:

- scheduled nodes move from 516 to 480;
- `E_8_8_16_2` cache-store calls move from 36 to 0;
- exactly 36 `reduce_output_rmsnorm_rope_kv_cache_8_128` calls appear;
- no replacement copy, cast, stack, or cache transport kernel appears;
- every flash score call depends on the producer-owned cache AFTER;
- a fresh installed profile improves device union; and
- a reps>=7 plain reverse bracket beats both controls with identical token
  streams.

The guaranteed device target is deletion of the measured 42.336 us/token
store row. The 89.250 us/token readiness gap is remeasured after the graph is
changed; it is not booked in advance.

## Fail-closed boundary

The candidate is limited to decode `B=T=1`, full-head RoPE, qk_norm equal to
head_dim, fp16/fp32 cache, no KV quantization, no rope-at-read/ring mode, and
the exact 8 heads x 128 elements K shape. It is disabled by default and must
have an explicit NV `sm_120` research lease during qualification. Prefill,
other targets, other shapes, and ordinary model loads keep the current graph.

## Explicitly out of scope

- changing GEMV arithmetic, weight bytes, or projection topology;
- widening flash combine again;
- making flash consume fresh K/V registers or transient producer buffers;
- independent concurrent K and V cache stores before the first candidate is
  profiled; and
- claiming any portion of the 89.250 us readiness gap without a new causal
  profile.
