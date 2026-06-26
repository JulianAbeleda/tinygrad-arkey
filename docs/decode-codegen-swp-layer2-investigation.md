# Codegen scheduling capability — layer 2 (cross-iteration SWP) investigation (2026-06-26)

Layer 1 (the latency-aware list scheduler, `SCHED_LIST=1`) is built and verified
(`docs/decode-codegen-list-scheduler-result.md`) but moves nothing on its own (7023→7075 µs) because the
hot loop has no independent work within one iteration's basic block. Layer 2 must create cross-iteration
ILP for layer 1 to interleave. This note records the precise obstacle and the two viable implementations,
with evidence, so the next build is unambiguous.

## The obstacle (evidence-backed)

The online-softmax recurrence is represented as **`Ops.AFTER(acc, range)`** — a loop-carry tied to the
`Ops.RANGE` (`tinygrad/uop/ops.py:533` `.after`; `:406` AFTER ended-ranges; the block-tile body at
`extra/qk_flash_decode.py:957-967` uses `acc.after(tt)`, `mx.after(tt)`, `dotp.after(b, tt)`).

`UOp.substitute` (`tinygrad/uop/ops.py:452`) can bind a loop variable to a constant — the established
unroll primitive (`tinygrad/schedule/multi.py:11` does `x.substitute({var: var.const_like(i)})`;
`tinygrad/codegen/opt/postrange.py:80,100` substitute ranges). **But substituting `tt → const` breaks the
recurrence:** `acc.after(tt)` becomes `acc.after(const)`, which is not the loop-carry. A correct scalar
unroll-by-U must, for each copy `u`, replace the loop-carry read `acc.after(range)` with **copy u−1's store
result** (and copy 0's with the outer-loop carry). That is custom AFTER-chain reconstruction, not a
substitution — this is precisely why a naive `AxisType.UNROLL` (which the expander *vectorizes*,
`tinygrad/codegen/late/expander.py:do_expand`) also fails: a serial `m/l/acc` recurrence cannot be
vectorized.

## Two viable layer-2 implementations

### (1) Scalar unroll-by-U with AFTER-chain reconstruction
Replace the `tt` REDUCE range (size N) with an outer range (size N/U); instantiate the body U times via
`substitute(tt → U·tt_outer + u)`; then **re-thread the recurrence**: in copy u, every loop-carry read
`acc.after(tt)` / `mx.after(tt)` is rewired to copy u−1's corresponding store (copy 0 → the outer carry).
The U copies' independent prologues (loads + fdot2 + cross-lane reduce → `sc`, which do NOT read the
recurrence) then coexist in one basic block, and layer 1 interleaves them. Generic, env-gated. Risk: the
"corresponding store" matching across copies must be exact; the block-tile microgate is the correctness
oracle.

### (2) Loop-split: independent-prologue loop + serial-recurrence loop
Split the reduction loop into: (a) a fully-independent loop computing all N `sc` (load + fdot2 +
cross-lane reduce) into a small REG/LDS array — its iterations are independent, so comgr + layer 1
pipeline them; then (b) a cheap serial loop running the online-softmax recurrence over the precomputed
`sc`. This decouples the independent work from the recurrence WITHOUT unrolling the recurrence. Generically
this needs a pass that partitions the loop body into "does-not-read-the-recurrence" vs "recurrence" — a
real but more local analysis than (1)'s rewiring.

## Status and next build

- Layer 1: DONE + verified (`extra/qk_codegen_list_scheduler.py`, `SCHED_LIST`).
- Layer 2: the AFTER-chain reconstruction (1) or loop-split (2) is the focused next compiler sub-project,
  now precisely defined. The harness is ready: `SCHED_LIST` (the consumer of the ILP), the isolated-timing
  method (currently flat at 7023 µs — it WILL move the moment layer 2 exposes real ILP), and
  `extra/qk_decode_attention_block_tile_microgate.py` (the correctness oracle that catches any mis-threaded
  recurrence).
- Honest scope: this is the hard core of the capability — a recurrence-aware loop transform — and it must
  be built and verified iteratively against the microgate, not rushed. It is the right long-term build
  (the generic capability that lets the *machine* produce latency-hidden kernels), with layer 1 as its
  foundation. Label: `SEARCH_PROGRESS__CODEGEN_SCHEDULER_LAYER1_DONE__LAYER2_DEFINED`.
