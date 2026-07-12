# Prefill 4k breakthrough: gate/up-only buffer2 policy

The first role-specific pure policy has crossed 4,000 tok/s on the RX 7900 XTX.

## Pinned result

Qwen3-8B Q4_K_M, AMD gfx1100, 512-token chunk, `K=8`, four warmups, three
rounds, synchronized whole-prefill authority, and peak clock pinning:

| Context | Throughput |
|---:|---:|
| 512 | **4,012 tok/s** |
| 1,024 | 3,835 tok/s |
| 2,048 | 3,444 tok/s |
| 4,096 | 2,865 tok/s |

The ctx512 timing is 127.6 ms. This is above the 4,000 tok/s milestone and is
within roughly 3% of the historical S9 result at 124.9 ms. The 4.4k target is
separate: it requires approximately 116.36 ms, leaving about 11.2 ms from this
run.

## Exact policy

The proven 40 KB LDS buffer2 candidate is selected only for `ffn_gate_up`.
`attn_qo`, `ffn_down`, and `attn_kv` fall through to the existing lean route.
The policy is explicit and reversible through
`BOLTBEAM_FULL_KERNEL_CANDIDATE_ROLES`; its default is `ffn_gate_up`.

The candidate-set census passed with one expected and one selected exact entry,
and the route remained strict-pure with rollback disabled. The all-four buffer2
policy measured 147.05 ms at ctx512, so this result directly validates the
role-specific LDS diagnosis rather than a clock or model change.

Evidence: [gate-up-only-policy-20260712](../bench/prefill-pure-full-kernel/gate-up-only-policy-20260712/README.md).
