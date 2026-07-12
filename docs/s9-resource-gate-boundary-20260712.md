# S9 resource-gate boundary

Exact pinned authority (`K=8`, warmups=4, rounds=3, ctx512) in isolated
worktree `c317132ef` measured **116.0 ms / 4,413 tok/s** on the S9 route,
matching the historical 4.4k band. This commit includes the attn_kv pipe
resource gate and is therefore the first narrowed boundary that restores the
historical result; commits after it must be bisected for the regression.

The adjacent `8ab3ee80c` worktree could not admit the model (VRAM admission
reported max_context=0), so that run is invalid and needs a clean-device
repeat. No conclusion is drawn from it.
