# Decode MMVQ runtime/cache identity

Verdict: `B2_CLOSED_NO_RUNTIME_CACHE_MISMATCH`.

Representative high-share roles reuse the same program/cache/launch identity in-model and direct-call; remaining gap is not a bounded runtime-cache wiring issue.

| role | identity match | in-model variants | standalone variants |
|---|---:|---:|---:|
| `attn_k/v` | `True` | `1` | `1` |
| `attn_q/o` | `True` | `1` | `1` |
| `ffn_down` | `True` | `2` | `2` |
| `ffn_gate/up` | `True` | `1` | `1` |
| `lm_head` | `True` | `1` | `1` |

Close B2; continue only with renderer/scheduler or artifact/import for large decode movement.
