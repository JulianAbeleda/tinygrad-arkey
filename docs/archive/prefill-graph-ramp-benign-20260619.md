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

## WHY benign (mechanism — code + cProfile confirmed, not inferred)
1. The 5 graphs are the SAME 729-kernel forward chunked by submission order (graph_split walks the linear once,
   flushes 32/64/128/256/512). Not functional units. Collapse to 1 = identical kernels/order = identical work.
2. Per-graph submit is CHEAP: cProfile submit/__call__ = 0.002s for 25 replays.
3. Back-to-back, no inter-graph host-wait: HCQGraph.__call__'s start-`wait` is for the graph's OWN previous replay
   (buffer-reuse safety), satisfied immediately in steady state; the 5 graphs chain via the device TIMELINE signal
   (GPU orders them). cProfile: 6 waits/forward = 5 cheap own-prev waits + 1 expensive final synchronize (the
   483K-poll busy-wait for the GPU finishing all 729 kernels).
4. The 0%-change A/B IS the proof of no inter-graph overhead (collapsing would close any gap; it didn't).
5. Why the ramp exists: CAPTURE-time host/GPU overlap (small first graph starts GPU while host builds the rest);
   irrelevant on warm replay (all pre-built, submit cheap).
COROLLARY: the earlier "115ms GPU" was ONE graph (the 249-kernel one); total GPU = all 729 kernels (5 graphs)
run back-to-back; the 333ms wall = final busy-wait for that. Prefill is GPU-bound on the full 729-kernel forward;
matmul not the lever -> non-matmul (attention) dominates the GPU time (concrete-KV 1.24x attacks it).
