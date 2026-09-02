# NV live-context wide vector Flash promotion

## Decision

Promote the llama-derived wide-KV vector Flash family for dense Qwen3-8B batch-1 decode on NV `sm_120`.
The selector is keyed by live context, and every geometry owns a separate captured graph identity.

| live context | split | control us/token | candidate us/token | recovered us/token | candidate tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 6 | 4135.601 | 4055.173 | 80.428 | 246.599 |
| 1024 | 10 | 4271.189 | 4083.013 | 188.176 | 244.917 |
| 2048 | 18 | 4428.785 | 4194.251 | 234.535 | 238.422 |
| 4096 | 34 | 4805.906 | 4406.013 | 399.893 | 226.963 |

Every reverse bracket used control/candidate/control fresh processes and produced identical token-stream hashes.
The retained `S=8` route covers `Tc=769..1024`; the new point tests qualify the surrounding live-context family.

## Admission

- Target: `NV`, `sm_120`.
- Topology: dense, 36 blocks, 32 query heads, 8 KV heads, head dimension 128, fp16 KV.
- Bands: `S6` through 768, `S8` through 1024, `S10` through 1280, `S18` through 2304, `S34` through 4352.
- Explicit request-horizon `S64` selection takes precedence.
- Contexts outside the qualified range retain the existing production route.
- Physical `max_context` is not capped: a 40K-capable model uses the promoted live bands through `Tc=4352` and safely falls back above them.
- Rollback: `TINYGRAD_FLASH_ACTIVE_HORIZON_DISABLE=1`.
- Prewarming is request-horizon scoped when `expected_output_tokens` is supplied; an unspecified horizon conservatively captures every reachable band.

## Cross-runtime interpretation

At depth 4096 the candidate measured 226.963 tok/s, while the fresh llama authority measured 226.022 tok/s.
This closes the former deep-context attention slope experimentally. A final strict production endpoint sweep remains
the authority for the integrated route because the causal brackets intentionally used the common full-logits path.
