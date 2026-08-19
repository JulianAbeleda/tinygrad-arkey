# NV vocab top-1 codegen diff: tinygrad packed argmax vs llama.cpp top-1

Date: 2026-08-19
Status: read-only source analysis. No code changes.
Question: is the packed-u64 argmax 2x regression a real decode wall cost, or
hidden mass? What lowering change would actually close it?

## TL;DR verdict

The `142.6 us` vs `71.9 us` packed-argmax microgate number is an
included-cost, isolated graph timing, not a decode wall debit. In the full
decode graph the entire vocab argmax tail is already hidden behind the
`~320 us` vocab GEMV, so the honest recoverable wall is **~5.6 us/token
(~+0.25 tok/s)**, and the packed route as implemented is wall-neutral to a
small regression, never a 70 us wall win.

The real codegen miss is packing `(max, index)` into one `uint64`. That is the
opposite of what llama.cpp does: llama reduces a plain `(float max, int index)`
pair, in either a single CUDA kernel or on the host. tinygrad should carry a
native `(fp32 value, int32 index)` pair, not synthesize and reduce a 64-bit key.

## 1. How llama.cpp computes the final top-1

llama.cpp has two decode-time paths, and the benchmark evidence in this repo is
the CPU-sampler path.

### Path A - default/CPU sampler (host argmax over the 151936 floats)

In the CPU-sampler path, the final logits row is made available on the host and
the sampler scans it there:

- The graph still writes the vocab logits, then
  `llama_context::output_reorder` copies them to the host:
  `ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0,
  n_tokens*n_vocab*sizeof(float))`
  (`src/llama-context.cpp:1420-1425`). For decode that is one
  `151936 * 4 = 607744` byte D2H copy.
- `llama_sampler_sample` reads that host row through
  `llama_context::get_logits_ith`
  (`src/llama-context.cpp:845-854`, `src/llama-sampler.cpp:806-855`) and runs
  the CPU `.apply` chain.
- Greedy `.apply` is a straight serial scan over the vocabulary:
  `llama_sampler_greedy_apply` (`src/llama-sampler.cpp:963-970`). With the
  common sampler chain, the equivalent zero-temperature top-1 is the
  temperature sampler's host branch.

Cost: an `O(V)` host scan plus a 607 KB async D2H. The host scan is not part
of llama's GPU node ledger and is hidden under its graph replay/overlap. This
is what the repo's measured comparison observed: llama's nsys vocab node is
only the `303.6 us` mmvq, with no separate GPU argmax node
(`docs/task_workflow/input/nv-decode-gap-attribution-same-session-20260812.md`
section 3, and `nv-three-front-audit-findings-20260818.md` section 3).

### Path B - backend sampler chain (GPU reduce, then 4-byte host copy)

When a sampler chain is offloaded through `llama_context_params.samplers`
(`src/llama-context.cpp:1150-1185`), sampling is folded into the CUDA graph by
`llm_graph_context::build_sampling` (`src/llama-graph.cpp:3031-3090`):

- The greedy sampler's `backend_apply` emits `ggml_argmax(ctx, data->logits)`
  (`src/llama-sampler.cpp:984-995`).
- That lowers to `GGML_OP_ARGMAX`, dispatched to `ggml_cuda_argmax`
  (`ggml/src/ggml-cuda/ggml-cuda.cu:2831`) and the kernel in
  `ggml/src/ggml-cuda/argmax.cu:8-90`.
- `argmax_f32` is one block per row, up to 1024 threads. It strides over
  `ncols=151936` with a per-thread `(float maxval, int argmax)`, warp-shuffles
  that pair, then reduces across warps in shared memory, and writes one
  `int32` per row (`argmax.cu:15-66`).
- The result is copied to host with
  `copy_tensor_async_ints`, which transfers only `sizeof(sampled.data[row])`
  per token (`src/llama-context.cpp:1542-1562`), invoked at
  `llama-context.cpp:1948-1953`.

So Path B is the answer to "GPU reduce to a small buffer then host top-1?": yes,
the device reduces the full fp32 row to one int, and the host reads one int.
There is no in-GEMV top-1 fusion in llama.cpp; the argmax is a separate graph
node, but it is a single small memory-bound kernel (one 607 KB read), not a
multi-kernel tail.

## 2. How tinygrad computes it today

### Legacy `Tensor.argmax` (the shipped greedy path)

`Tensor.argmax` in `tinygrad/mixin/__init__.py:835-860` lowers to:

1. `m = x.eq(x.max(axis))` - one fp32 MAX reduction plus the equal mask.
2. `idx = m * arange(shape[axis], 0, -1)` - mask the reversed index vector.
3. `(shape[axis] - idx.max(axis)).cast(int32)` - a second reduction and unpack.

For `[1, 151936]` fp32 logits this becomes the four-kernel tail recorded in the
08-18 ledger (`nv-fuse-hide-eliminate-ledger-20260818.md`, L2 row):

| kernel | launches | med us | node us |
| --- | ---: | ---: | ---: |
| `E_1187_32_4` | 2 | 3.65 | 7.29 |
| `r_32_4_1187` | 1 | 39.14 | 39.14 |
| `r_128_16_8_1187` | 1 | 11.10 | 11.10 |
| `r_16_8` | 1 | 1.70 | 1.70 |
| tail total | 5 | | 59.23 |

The vocab head itself is `q6k_gen_coop_151936_4096_inkernel`, ~319.87 us at
HEAD. The greedy call sites are `Transformer.forward_greedy` and
`forward_greedy_with_logits` (`tinygrad/llm/model.py:1417-1438`).

### Packed-u64 standalone route (the microgate subject)

`packed_argmax_finite_fp32` (`tinygrad/llm/packed_argmax.py:13-49`) reorders the
fp32 bit pattern into a monotonic `uint32`, casts it to `uint64`, ORs the
inverted index into the low 32 bits, performs one `uint64` MAX, and unpacks the
index. It is gated by `_decode_packed_argmax_promoted`
(`tinygrad/llm/model.py:1426-1427`), default false.

The microgate
(`extra/llm_research/decode/nv_packed_argmax_microgate.py:20-49`) compares the
full included-cost graph of this route against `Tensor.argmax` on
`[1, 151936]` fp32. It is explicit that this is direction-only: line 49 says
"wall debit requires a later full-token capture."

### Packed-u64 in-GEMV P1 fusion (the L2/E1 target)

The research route in `tinygrad/llm/decode_routes.py:474-514` swaps the vocab
GEMV epilogue to emit one packed u64 key per `row_tile`-wide warp tile, then
runs one cross-tile u64 MAX:

- The GEMV epilogue packs `(ordered f32 bits)<<32 | (rows-1-row)` and
  warp-reduces it with `_q6k_coop_tile_key_reduce_max`
  (`tinygrad/llm/decode_kernels.py:544-557`, `798-807`).
- The second stage is `packed_argmax_from_tile_keys`
  (`tinygrad/llm/packed_argmax.py:91-113`), gated by `_decode_vocab_top1_lease`
  (`tinygrad/llm/model.py:1420-1424`).

Important measured detail: the scheduler does not remove the tail; it renames
4 kernels into 2 (`E_1187_16_4` + `r_16_4_1187`). The u64 cross-tile reduce
reads its freshly written 607 KB key buffer L2-cold after the 315 us GEMV
evicts L2, so the fused route measured **+25.5 us/token** end-to-end until a
2.2 us `keys.clone()` warm-up copy was added, which flips it to ~11 us faster
than the legacy tail (`decode_routes.py:507-513`,
`nv-vocab-top1-fusion-head-recheck-20260816.md` sections 0-1).

## 3. Is the 2x microgate a wall cost or hidden mass?

It is hidden/renamed mass, not a 70 us wall debit.

- Isolated included-cost: `142.647 us` packed vs `71.874 us` legacy, delta
  `+70.773 us` (`nv-packed-argmax-microgate-record-20260805.md`). This includes
  key construction and every ordinary reduction, not just the final kernel.
- Wall transfer: the 08-17 A/B removed ~16.5 us of tail node and moved wall
  only **-1.55 us** (`nv-l2-vocab-aux-ab-verdict-20260817.md` section 2).
  Scaling to the full 59.23 us tail gives an honest wall ceiling of **~5.6 us
  (~+0.25 tok/s)** (`nv-three-front-audit-findings-20260818.md` section 3).
- The P1 fused route was bit-exact but **+25.5 us** in the 08-16 A/B (L2-cold
  u64 reduce), and -1.55 us after the warm-up copy was understood. Both are far
  below the +50 us promotion bar.

Conclusion: the tail is dominated by the vocab GEMV anchor, so the tail kernels
overlap. The packed-u64 route does not make a ~70 us wall cost visible, but it
also cannot produce a meaningful win, because its "one MAX" optimization is
bought with 64-bit key construction/reduce and an L2-residency penalty. The
correct codegen answer is a native fp32 argmax, not a packed key.

## 4. What closing fix actually looks like

Carry `(max, index)` as two native values and reduce them with the existing
multi-slot composite-reduce machinery, instead of bit-packing them into u64:

- `tinygrad/uop/ops.py` already has `AccumulatorSlot(op, dtype, identity, name)`
  and `CompositeReduce.slots`
  (`tinygrad/uop/ops.py:1318-1322`, `1425-1432`), and
  `tinygrad/codegen/late/composite_combines.py` has a combine registry
  (`composite_combines.py:106-110`). An `argmax` combine with two slots
  (`f32 MAX` for the value, `int32` "carry winning index" slot) would lower to
  the same per-thread `(maxval, argmax)` pair llama's `argmax_f32` uses.
- That makes `Tensor.argmax` a single-pass value+index reduce instead of the
  current two-reduction `eq/max` chain. It does not require the 64-bit
  bitcast/shift/OR/index-cast preamble in `packed_argmax.py:36-41`.
- For the in-GEMV fusion specifically, the second cross-tile kernel cannot be
  removed today because the PTX renderer has no atomics/grid sync for a
  single-pass cross-tile reduce (`nv-three-front-audit-findings-20260818.md`
  section 3). The realistic fix is to keep the second kernel but make it an
  fp32+int32 pair reduce, and retain the warm-up copy if the key buffer is
  still a separate allocation.

## 5. Concrete changes needed

1. `tinygrad/mixin/__init__.py` (`argmax`, lines 835-860): lower argmax through
   a first-class value+index composite reduce rather than
   `eq(max) * arange -> max -> subtract`. This removes the extra index material
   and the second reduction.
2. `tinygrad/uop/ops.py` and `tinygrad/codegen/late/composite_combines.py`: add
   an `argmax` combine/slot definition (`f32 MAX` value slot + `int32` index
   slot) and register it in `COMBINE_REGISTRY`/`COMBINE_STEP_REGISTRY`.
3. `tinygrad/llm/decode_kernels.py` (`_emit_q6k_coop` vocab epilogue, lines
   798-807, and `_q6k_coop_tile_key_reduce_max` lines 544-557): replace the
   packed-u64 epilogue with a native `(max, index)` pair carry so the emitter
   can use `fmax` plus a 32-bit select instead of a 64-bit `MAX`/shuffle.
4. `tinygrad/llm/decode_routes.py` (`q6k_vocab_top1_call`, lines 474-514):
   emit the pair-shaped output and drop the `packed_argmax_from_tile_keys` u64
   unpack; keep the `keys.clone()` L2 warm-up only if the second-stage buffer is
   still a fresh per-token allocation.
5. `tinygrad/llm/packed_argmax.py`: retire or demote the u64 helpers. Do not
   promote `_decode_packed_argmax_promoted` in `tinygrad/llm/model.py`; keep
   `_decode_vocab_top1_lease` closed until change 3 lands.
6. Optional renderer work (`tinygrad/codegen`/NV renderer): if a true
   single-pass in-GEMV cross-tile argmax is desired, add the missing
   cross-tile atomic/grid-sync primitive; otherwise the two-kernel pair reduce
   is the ceiling.

## 6. Evidence files

- `docs/task_workflow/input/nv-fuse-hide-eliminate-ledger-20260818.md` (rows L2
  and E1).
- `docs/task_workflow/input/nv-three-front-audit-findings-20260818.md` (section
  3, vocab_aux).
- `docs/task_workflow/input/nv-packed-argmax-microgate-record-20260805.md`.
- `docs/task_workflow/input/nv-l2-vocab-aux-ab-verdict-20260817.md`.
- `docs/task_workflow/input/nv-vocab-top1-fusion-head-recheck-20260816.md`.
- `docs/task_workflow/input/nv-decode-gap-attribution-same-session-20260812.md`
  (vocab class, llama mmq 303.6 us with no GPU argmax node).
