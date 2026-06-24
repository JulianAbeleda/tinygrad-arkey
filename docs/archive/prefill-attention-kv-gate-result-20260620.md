# Prefill Attention/KV Gate Result - 2026-06-20

Verdict: `PARK_PREFILL_ATTENTION_KV_LOW_AMDAHL_FOR_PP512`

Run:

```bash
PYTHONPATH=. python3 extra/qk_prefill_attention_kv_gate.py
```

## Result

| item | value |
|---|---:|
| attention/KV share of PREFILL_V2 graph span | `25.21%` |
| full speedup if attention/KV gets `1.25x` | `1.0531x` |
| full speedup if attention/KV gets `2.00x` | `1.1442x` |
| required attention/KV speedup for `1.10x` full prefill | `1.5641x` |
| required attention/KV speedup for `1.15x` full prefill | `2.0723x` |

Decision: park pp512 attention/KV as the next prefill move. It only becomes worth reopening if long-context
changes the share or a bounded `>=2x` route appears.
