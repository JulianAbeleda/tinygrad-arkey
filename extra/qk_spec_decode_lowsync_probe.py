"""Low-sync speculative-decode proposal graph (Phase 3/4 PROVEN). NOT routed; probe/reference.

The reusable K-step draft proposal graph: one captured TinyJit, device-token feedback (argmax fed into the next
step on-device, no .item()), K DISTINCT symbolic start_pos vars rebindable per pass. Byte-exact vs host-stepped
draft greedy, reusable across advancing L with NO recompile (flat ~25ms across rebinds), one sync for K proposals.

The key (Phase 4): use K distinct symbolic vars sp0..sp_{K-1}, bind sp_i = base+i per pass. A SINGLE var with
+i offset conflicts ("bind mismatch 5 != 6"); concrete positions recompile per L. Distinct vars rebind the cache
read length correctly per step. See docs/spec-decode-low-sync-{arc,phase4}-20260618.md.
"""
from tinygrad import Tensor, UOp, TinyJit, dtypes

def make_proposal_graph(draft, K:int, max_ctx:int):
  sps = [UOp.variable(f"sp{k}", 0, max_ctx-1) for k in range(K)]
  @TinyJit
  def propose(tok0, *bs):
    t = tok0; outs = []
    for k in range(K):
      t = draft.logits(t, bs[k])[:, -1:, :].argmax(-1).cast(dtypes.int32)
      outs.append(t.reshape(1, 1))
    return outs[0].cat(*outs[1:], dim=1).realize()   # [1, K], one sync
  def call(tok0_id:int, base:int):
    return propose(Tensor([[tok0_id]], dtype="int32").contiguous(), *[sps[k].bind(base+k) for k in range(K)])
  return call
