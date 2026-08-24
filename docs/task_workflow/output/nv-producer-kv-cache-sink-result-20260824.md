# NV producer-owned K/V cache sink result

Date: 2026-08-24
Repo: `/home/ubuntu/tinygrad-arkey`
GPU: RTX 5090 (`NV sm_120`), model: Qwen3-8B-Q4_K_M

## Outcome

The first producer-side construction passes and is promoted for NV `sm_120`.
The terminal K RMSNorm+RoPE producer now absorbs final V and writes both values
directly to the current cache slot. This removes all 36 generic K/V cache-store
launches without changing cache bytes or token output.

The accepted reps=7 reverse bracket recovers `32.658 us/token`. The new
conservative reps=15 installed endpoint is `4.556418 ms/token`, or
`219.471 tok/s`, leaving `389.751 us/token` to 240.

## Exactness gate

The isolated gate used the same NVRTC compiler as production and compared:

```text
control: installed K reduce_output RMSNorm+RoPE -> legacy K/V cache store
candidate: terminal K producer -> K/V cache slot
```

It compared the entire cache allocation byte-for-byte for fp16 and fp32 cache
dtypes at `start_pos=0`, an interior slot, and the last slot. All six cases
passed with zero mismatched bytes and zero changed bytes outside the selected
slot. The production-core emitter then re-passed the same six cases.

## Structural gate

At depth 512:

| row | control | candidate | delta |
| --- | ---: | ---: | ---: |
| scheduled nodes | 516 | 480 | -36 |
| generic `E_8_8_16_2` stores | 36 | 0 | -36 |
| producer cache sinks | 0 | 36 | +36 |
| node sum | 4348.016 us | 4300.976 us | -47.040 us |
| device union | 4343.875 us | 4298.625 us | -45.250 us |

The first profile report incorrectly showed 512 candidate nodes because the
generic replay parser inferred the next token's 32-node prefix as a replacement
tail after the real 36-node tail disappeared. Exact topology parsing gives the
480-node result above. The parser now knows that control is
`32+64+128+256+36`, while candidate is `32+64+128+256`; prefill prefixes
followed by the 394-node compile tail are excluded.

The final no-override installed profile independently reports 480 nodes,
`4301.744 us` node sum, `4298.500 us` union, 36 producer sinks, and zero generic
stores.

## Wall qualification

Every arm is a fresh process with 32 timed tokens per repetition:

| arm | median ms/token |
| --- | ---: |
| control A | 4.553372 |
| producer sink | 4.516326 |
| control C | 4.544594 |

The candidate beats both controls and recovers `32.658 us/token` versus their
midpoint. All arms have identical token-stream hashes.

The final reps=15 installed run is intentionally reported conservatively. Its
samples enter a slower thermal/power regime after the first seven repetitions;
the retained full median is `4.556418 ms/token`, not the faster early window.
Relative to the preceding comparable `4.578813 ms/token` endpoint, that moves
the ledger by `22.395 us/token` after cross-session thermal variation.

## Promotion and rollback

`decode-producer-kv-cache-sink-route-policy.json` promotes only
`("NV", "sm_120")`. The model call site also requires decode `B=T=1`, full-head
RoPE, qk_norm equal to head_dim, an fp16/fp32 cache, no KV quantization, and no
rope-at-read path. Other targets and shapes retain the legacy store.

`TINYGRAD_PRODUCER_KV_CACHE_SINK_DISABLE=1` restores the complete legacy chain
at model load.

## What the new graph says is next

The producer boundary is not exhausted:

- the new K terminal body contributes `67.360 us/token`;
- it begins immediately after the later K/V projection dependency;
- V is the later dependency, and the aggregate K/V projection-end spread is
  `139.250 us/token` median; and
- after the K cache producer is complete, flash score still waits a median
  `93.875 us/token` before launch (p10-p90 `88.750-99.250`).

This identifies the next bounded construction: independent producer writes.
K norm+RoPE should write only K directly, while each admitted V projection
writes V directly in its own terminal epilogue; flash joins both cache AFTERs.
The construction removes the explicit V-to-K-producer dependency and gives the
67.360 us K terminal body room to fit under the measured V lag. It must keep
480 nodes and add zero transports. The surviving ~94 us ready-to-flash gap is
a separate scheduling/placement target and is not booked as recovery.

## Verification and evidence

- 57 focused unit tests pass.
- Six cache-byte microgate cases pass twice (research candidate and production
  core emitter).
- The promoted normal-load profile has the required 480-node topology.
- The reps=7 wall bracket is token-exact and passes against both controls.

Evidence: `docs/task_workflow/evidence/nv-producer-kv-cache-sink-20260824/`.

Verdict: `PROMOTED_PRODUCER_CACHE_SINK_219_471_TOK_S_NEXT_INDEPENDENT_WRITERS`.
