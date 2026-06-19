# Scope — TPE-6b runtime helper: single-dispatch graph capture for the extracted Tensile kernel

TPE-6 REDIRECT proved the extracted Tensile kernels route a full FFN block **exact + copy-free at 1.53× the
PREFILL_V2 plateau on GPU time**, but naive per-op routing adds ~6.2 ms host overhead (each `.realize()` /
`wait=True` is a separate schedule + GPU sync). That overhead is a **JIT-less probe artifact**: the real model
forward is TinyJit-captured into one HCQGraph with no per-op host sync. TPE-6b builds/validates the runtime helper
that lets the Tensile launch live inside a single GPU dispatch so the GPU win survives end-to-end.

## Mechanism (grounded in the code)
`HCQProgram.__call__` (hcq.py:374-380) builds one queue: `q = hw_compute_queue_t().wait(timeline, v-1).memory_barrier();
q.exec(prg, args_state, gws, lws); q.signal(timeline, next).submit(dev)`. **Multiple `q.exec()` calls before one
`signal().submit()` batch N kernels into ONE submit + ONE host sync.** `NamedAMDProgram` already exposes the exec
contract (prog_addr, rsrc1/2/3, group_segment, args_state via `fill_kernargs`). So a single queue can enqueue several
Tensile execs with `memory_barrier()` between dependent ones — no host round-trip between kernels.

Two tiers:
- **Tier A — standalone batch primitive (this phase, achievable now):** a reusable single-queue launcher that
  enqueues N Tensile execs in one submit/wait, with persistent baked kernarg args_states. Proves the Tensile
  *dispatch* batches with negligible host overhead (≈ summed device time), i.e. the removable cost in TPE-6 was the
  per-kernel sync, not the launch.
- **Tier B — in-model graph capture (TPE-7):** make the Tensile launch a node the model's existing TinyJit/HCQGraph
  captures alongside the tinygrad transpose/SiLU·mul ops, so the *whole* block is one dispatch. The tinygrad
  elementwise/transpose kernels are already JIT-captured in PREFILL_V2's forward; only the Tensile launch must join
  that graph. Extracting/re-enqueuing tinygrad's UOp-`Ops.PROGRAM` kernels into a *standalone* manual queue is not
  cheap, so full-block single-dispatch validation is deferred to the in-model path.

## Phases
- **RH-0** — confirm the batch-exec contract: persistent args_state (call `fill_kernargs` once, reuse), `memory_barrier`
  between dependent execs, one `signal().submit()`, one `synchronize()`.
- **RH-1** — build `batch_launch(dev, items)` and measure the 3 Tensile matmuls (gate, up, down) batched in ONE
  submit/wait vs the per-kernel `wait=True` path. Gate: batched wall ≈ summed device time (≈ 2.5 ms), i.e. ≪ the naive
  per-op wall — proving the per-kernel sync was the removable host cost.
- **RH-2** — project the graph-integrated block: batched-matmul GPU time + the measured GPU time of the SiLU·mul +
  transposes (each realized once, device time only) = the single-dispatch block estimate; compare to the PREFILL_V2
  block. This is the honest end-to-end-if-graph-captured number.
- **RH-3 (TPE-7, separate)** — in-model: route the block behind a research flag inside the TinyJit'd forward so the
  Tensile launch is graph-captured with the tinygrad ops; measure warm pp512 + dNLL.

## Gates
- RH-1 PASS: 3-matmul batched wall ≤ 1.3× their summed device time (host overhead per kernel ~eliminated).
- RH-2 PASS: projected single-dispatch block ≥ 1.20× the PREFILL_V2-plateau block for the same work.
- KILL: batching does not reduce the wall (dispatch itself carries irreducible host overhead), or the projected
  block is < 1.15×.

## Constraints
No model.py route in Tier A; no defaults; decode untouched; reuse committed kernarg captures + NamedAMDProgram; keep
the launcher probe-local. Tier B (in-model) is TPE-7 and needs the separate external-artifact policy decision.

## Deliverables
`extra/qk_tensile_block_graph.py` (batch launcher + RH-1/RH-2 measurement), `bench/qk-tensile-extraction/block_graph.json`,
result appended to `prefill-tensile-tpe6b-runtime-helper-result-20260619.md`.
