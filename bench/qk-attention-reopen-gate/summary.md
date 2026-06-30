# PMS-R7 Decode Attention Reopen Gate

Verdict: **PMS_R7_PASS_ATTENTION_REOPEN_GATE** -> gate: **DO_NOT_REOPEN_ATTENTION**

Active threshold: 5.0% whole-decode gain. Realizable gain: **0.0%** (best available route = 68.3% of owned, all levers walled).

| ctx | attention tile wall-share (Amdahl) | perfect-parity gain ceiling % | native vs owned % |
|---:|---:|---:|---:|
| 512 | 0.102 | 10.2 | 68.3 |
| 4096 | 0.03 | 3.0 | 60.1 |

DO_NOT_REOPEN_ATTENTION: (1) realizable whole-decode gain = 0% -- the best available attention route is 68.3% of owned (BELOW owned), and every attention search axis is refuted/exhausted (['attention_combine_fused_lifecycle', 'native_attention_as_default', 'n1b_scalar_address_path', 'occupancy_lds_only_attention_tuning', 'scheduler_only_attention_tuning']), so no candidate can realize the Amdahl ceiling; (2) even the theoretical ceiling is small and ctx-decaying (10.2%@ctx512 -> 3.0%@ctx4096), below the 5.0% reopen bar at the representative long-ctx operating point; the dominant decode wall is the weight-memory-bound FFN/projection (HBM), not attention.

What would make attention active again:
- much longer context with a large KV cache that materially raises the attention tile wall-share
- a different KV layout / dtype that raises the KV-read floor relative to weight read
- larger Hq/Hkv/Hd (head_dim) increasing attention FLOPs/bytes per token
- MoE or a larger FFN-sparsity that shrinks the dense-FFN wall-share so attention dominates
- a different GPU with higher HBM bandwidth making the weight read less dominant
- a NEW attention primitive premise distinct from the refuted axes (not native-as-default, not combine/fused-lifecycle, not scheduler/occupancy/LDS-only, not N1B scalar-address)
