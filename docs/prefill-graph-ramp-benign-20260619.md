# LEARNING — the prefill 5-graph batch-ramp is BENIGN (not a lever); exhausts the dispatch/graph hypothesis

Investigated the 5-graph-per-forward structure (the path chosen for max learning).

## What the 5 graphs are
Every prefill forward emits exactly 5 jit graphs: **32, 64, 128, 256, 249 kernels** (=729 total), repeating per
forward (1 forward -> [32,64,128,256,249]; 3 forwards -> that x3). Cause: tinygrad's JIT graph-batching
(`engine/jit.py` `graph_split_rewrite`/`flush_batch`): starts at `JIT_BATCH_SIZE=32` and **doubles after each
flush** (32->64->128->256->512). A deliberate ramp: the small first graph launches fast so the GPU starts while
the host builds the bigger ones (host/GPU-overlap heuristic).

## Is it a lever? NO.
A/B of `JIT_BATCH_SIZE` (controls the ramp -> 2048 = one graph of 729): **32 / 256 / 2048 all = 1537 tok/s (333ms),
identical.** Collapsing 5 graphs -> 1 gives 0% change. The inter-graph syncs chain seamlessly on replay; the ramp
adds no measurable overhead. **Not a lever.**

## What this exhausts (the value)
Combined with the prior solid results, prefill's NON-attention hypotheses are now all REFUTED:
- matmul kernel speed (Tensile/schedule/transpose-free): ~1.00x
- dispatch (cProfile: submit cheap 0.002s; wall is busy-WAIT for GPU)
- graph count / batch ramp: 1.00x (this result)
-> **prefill is GPU-bound (busy-wait), and the ONLY validated lever is concrete-KV/attention (1.24x).** The matmul,
   the dispatch, and the graph structure are all confirmed non-levers.

## Remaining honest gap (unchanged)
WHAT dominates the ~333ms GPU (attention vs the .contiguous() copies vs per-layer structure) is still not cleanly
measurable (PMC perturbs, ProfileGraphEvent timestamps corrupt). But the LEVER is known: concrete-KV (attention).

## Files
A/B inline; jit mechanism `tinygrad/engine/jit.py:31-58`. Prior: `prefill-wall-CORRECTION-20260619.md`,
`prefill-l1-l2-result-20260619.md`.
