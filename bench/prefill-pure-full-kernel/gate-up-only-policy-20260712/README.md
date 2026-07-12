# Gate/up-only buffer2 policy

This run applies the proven 40 KB LDS buffer2 candidate only to `ffn_gate_up`.
`attn_qo`, `ffn_down`, and `attn_kv` fall through to the existing lean
handwritten route. The measured policy is therefore **hybrid role-selective**,
not pure. It is controlled by `BOLTBEAM_FULL_KERNEL_CANDIDATE_ROLES` and
defaults to `ffn_gate_up`.

Pinned smoke authority: Qwen3-8B Q4_K_M, `K=8`, four warmups, three rounds,
512-token chunks, clock pin enabled, strict-pure route, no rollback.

| Context | tok/s |
|---:|---:|
| 512 | 4,012 |
| 1,024 | 3,835 |
| 2,048 | 3,444 |
| 4,096 | 2,865 |

The ctx512 result is 127.6 ms, within roughly 3% of the hybrid reference
(124.9 ms) and far ahead of the all-four buffer2 result (147.05 ms).
