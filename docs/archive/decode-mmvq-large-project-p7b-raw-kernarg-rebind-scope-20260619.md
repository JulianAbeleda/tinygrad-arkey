# Decode MMVQ large project P7b raw-kernarg rebind scope - 2026-06-19

Purpose: scope the next build after P7a redirected. P7a proved that the imported llama Q4_K MMVQ primitive cannot be
made graph-safe by a plain runtime-cache wrapper. P7b is the minimal runtime capability needed to make imported raw
kernargs rebindable in TinyJit/HCQGraph.

## Problem Statement

P3-P6 prove the imported Q4_K consumer:

- direct HCQ launch works;
- raw `144` byte llama kernarg can be patched eagerly;
- q8 producer + imported consumer lifecycle is correct;
- Q4 shape matrix covers `attn_output`, `ffn_gate`, and `ffn_up`.

P7a failed only at graph replay:

- `HCQGraph.__init__` pre-fills kernargs once into graph-owned `kernargs_bufs`;
- replay reuses those graph arg buffers;
- normal tinygrad kernels are rebindable because `HCQArgsState` knows buffer arguments;
- the imported llama kernel carries opaque raw bytes, with pointer fields hidden at offsets `0`, `8`, and `56`.

So the missing primitive is **raw-kernarg pointer rebinding**:

```text
raw kernarg template
+ declared pointer patches: [(offset, arg_index)]
+ graph-owned args buffer
=> graph-safe kernarg with symbolic/replayable buffer VAs
```

## Target Contract

Add a tiny, explicit imported-kernel ABI object, local to the research path at first:

```python
RawKernargPatch(offset:int, arg_index:int, kind:"ptr")
RawKernargProgram(raw_template:bytes, patches:list[RawKernargPatch], launch, local)
```

For the imported llama Q4_K no-fusion template:

| kernarg offset | meaning | bound arg |
|---:|---|---|
| `0` | Q4_K weight pointer | `q4` |
| `8` | `block_q8_1` activation pointer | `q8` |
| `16` | ids pointer | constant null |
| `56` | output pointer | `out` |

All other scalar fields stay copied from the captured llama launch.

## Implementation Shape

### P7b-0 - static runtime audit

Confirm the exact call path:

- `get_runtime` returns the swapped imported runtime;
- `HCQGraph.__init__` calls `runtime.fill_kernargs(self.hcq_bufs[j], vars, argsbuf)`;
- `self.hcq_bufs[j]` may contain fake `HCQBuffer(UOp.variable(...), size)` for graph inputs;
- `AMDComputeQueue.exec` only binds `args_state.bind_data` and then launches with `args_state.buf.va_addr`.

Gate: written note in the result artifact with the files/lines used.

### P7b-1 - rebindable raw args state

Build a probe-local `RawRebindArgsState` or equivalent wrapper that:

- writes the raw template into the supplied `argsbuf`;
- writes each pointer patch into the raw template at the declared byte offset;
- if a buffer VA is a concrete int, writes it immediately;
- if a buffer VA is symbolic, records a `bind_data` entry so `bind_args_state` patches it at graph replay.

The existing `HWQueue.bind_args_state` path already handles `bind_data`:

```python
for vals, mem, fmt in args_state.bind_data:
  self.bind_sints_to_mem(*vals, mem=mem, fmt=fmt)
```

Gate: a CPU-side artifact proves the graph-owned kernarg buffer contains the expected concrete or symbolic patch entries
before any imported kernel launch.

### P7b-2 - eager parity

Replace `ImportedQ4MMVQRunner.fill_kernargs` with the rebindable path, but run only eager direct HCQ:

- same `blk.0.attn_output.weight`;
- same q8 producer;
- correctness vs q8 CPU reference;
- no TinyJit yet.

Gate: same correctness as P5 (`max_abs <= 2e-2`) and no MMU fault.

### P7b-3 - graph micro-smoke without model

Use synthetic persistent `x`, `q8`, `q4`, `out` tensors:

- TinyJit captures q8 producer -> imported Q4 MMVQ consumer;
- replay at least `5` times;
- compare replay outputs to eager output;
- run with fixed side buffers first.

Gate: replay `max_abs <= 1e-6` vs eager for calls `3+`, no fault.

### P7b-4 - real activation graph proof

Repeat P7b-3 with real `blk.0.attn_output` activation from Qwen3-8B block 0.

Gate:

- replay stable;
- imported output correct vs q8 reference;
- no fault;
- no model default change.

### P7b-5 - one-block model route decision

Only after P7b-4 passes:

- route one Q4 role behind `DECODE_MMVQ_IMPORT_Q4=1`;
- use persistent side buffers owned by the block/linear route;
- measure clock-controlled one-role or one-block decode;
- do not route by default.

Gate:

- no graph replay fault;
- role output passes q8/dNLL policy;
- projected W==D movement remains `>=5%` before full model route.

## Kill Conditions

Kill or redirect P7b if any of these happen:

- symbolic fake `HCQBuffer` VAs cannot be written through `bind_sints_to_mem`;
- graph-owned kernarg buffer cannot carry per-offset pointer patching without runtime-wide changes;
- eager rebindable path regresses P5 correctness;
- graph replay still faults after symbolic patching;
- the required runtime change is broad enough to affect normal tinygrad kernel args.

If killed, the imported route remains an eager research primitive and the project should switch to native MMVQ lowering.

## Expected Outcome

Best case:

- P7b turns imported raw-kernarg kernels into graph-safe programs;
- Q4 route can move to P7c W==D/dNLL;
- the same capability applies to Q6 and future imported backend kernels.

Likely risk:

- tinygrad graph input replacement uses symbolic `UOp.variable` addresses in fake buffers, and the current raw patch path
  needs careful `bind_data` integration to avoid freezing stale addresses.

This is a runtime capability build, not a kernel search build.
