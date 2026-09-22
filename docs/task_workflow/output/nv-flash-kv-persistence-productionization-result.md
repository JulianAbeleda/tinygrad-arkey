# NV Flash K/V persistence productionization result

## Outcome

The CUDA access-policy-window substrate is real, and its context reservation
mechanism has been decoded and replayed successfully on tinygrad's native HCQ
channel. The mechanism is an MME call, not a QMD field:

```text
SET_MME_SHADOW_SCRATCH[0..2] = {0, reservation_units, 0x1f0000}
SET_FALCON04                 = 0x17e2ac
```

For a 60 MiB reservation CUDA changes only `reservation_units` from zero to
`0x000a0000`. The copied native transaction recovers most of an intentionally
disturbed 72 MiB persisting footprint. The native reservation substrate is
therefore **solved**.

It is not promoted for Flash. Two exact settled-token brackets show that the
large primitive recovery does not convert on the dense production lifecycle:
full persistence and capacity-matched fractional persistence both lose to
both controls. Production remains unchanged.

## Mechanism and translation ledger

| construction | result | disposition |
|---|---:|---|
| CUDA stream access-policy window, aggregate 72 MiB footprint | disturbed reload 47.104 -> 20.480 us | hardware substrate pass |
| HCQ cache-hinted K/V loads | hot 5.312 -> 5.408 us/layer; cold 5.792 -> 6.016 us/layer | primitive no-go |
| one-time real-cache priming | 4.066318 -> 4.067415 ms/token; -0.066 tok/s | wall no-go |
| CUDA command capture | identical 11-word PB; only word 2 changes `0 -> 0x000a0000` | mechanism identified |
| native six-word MME replay | zero and 60 MiB arms complete on HCQ | construction pass |
| native reservation + persisting aggregate footprint | recovery share 0.098 -> 0.871 | primitive pass |
| native reservation + full static Flash K/V persistence | 4058.535 -> 4076.971 us/token; -1.114 tok/s | wall no-go |
| native reservation + capacity-matched numerator 13 | 4057.379 -> 4070.791 us/token; -0.812 tok/s | wall no-go |

The original 11-word CUDA pushbuffer ended with a five-word completion
semaphore. Literal replay faulted because that tail referenced CUDA VA
`0x206a0fff0`; it was correctly removed and replaced by tinygrad's own HCQ
timeline signal. The remaining six words are the self-contained reservation
transaction.

Both native token candidates preserved the complete token-stream hash. The
full arm lost 18.436 us/token; the capacity-matched arm lost 13.412 us/token.
The latter tests the accounting-derived `60/72` policy fraction, so simple
capacity over-subscription is not the entire negative. Reserving L2 for K/V
displaces data that is at least as valuable elsewhere in the token lifecycle.

## Closure and reuse boundary

The generic native reservation mechanism is reusable research substrate, but
Flash K/V persistence is closed for promotion on this measured dense path.
Reopen only with a new ownership hypothesis that specifies which competing L2
population is evicted and predicts a positive full-token balance. More static
fraction sweeps without that accounting are not admissible.

The earlier conclusion that a stream-backed path or unknown QMD encoding was
required is superseded. Periodic explicit priming and per-load dynamic
`evict_last` remain closed by their own primitive and wall gates.

## Evidence

- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/l2-persisting-window-r1.json`
- `docs/task_workflow/evidence/nv-flash-kv-persistence-20260827/evictlast-counter-r1.json`
- `docs/task_workflow/evidence/nv-flash-kv-persistence-20260827/retention-wall-r1.json`
- `docs/task_workflow/evidence/nv-flash-kv-persistence-20260827/native-reserve-wall-r1.json`
- `docs/task_workflow/evidence/nv-flash-kv-persistence-20260827/native-reserve-n13-wall-r1.json`
- `extra/llm_research/decode/nv_l2_reservation_mapping_probe.cu`
- `extra/llm_research/decode/nv_l2_native_reservation_replay.py`
- `extra/llm_research/decode/nv_flash_kv_native_reserve_wall.py`
- `extra/llm_research/nv_capture/nv_reservation_capture.c`
