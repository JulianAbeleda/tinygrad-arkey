# NVIDIA prefill ordered scratch ownership gate

## Verdict

**PASS for the generic scheduler substrate and the finalized native pp512
chain.** One physical partial/id workspace can be reused by 72 captured
producer/main/fixup chains when each call's returned scratch value is threaded
as the next explicit epoch. Raw reuse without that epoch remains rejected.

This does not promote the packed model route. The default-off research binding
has now adopted the contract and passes the model graph/correctness gates, but
its synchronized wall remains slower than the FP16 route.

## Root cause

`Tensor.uop_program` returns one `AFTER` per argument. That includes read-only
arguments so a future writer can wait for a read to finish. The rangeify WAR
repair previously treated every `AFTER` as a write and stored only one writer
per physical buffer. Consequently this valid chain looked circular:

```text
main 0 writes scratch
  -> fixup 0 reads scratch and returns a read-completion AFTER
  -> main 1 writes the same scratch
```

The scheduler now gets writable slots from the opaque SINK or finalized
PROGRAM ABI. For an opaque buffer with multiple writers, it admits reuse only
when all declared call access epochs form one dependency chain. An unordered
raw alias raises `unordered repeated write epochs`. Ordinary STORE/assignment
graphs retain the existing WAR path.

## Qualified contract

```python
partial_epoch, id_epoch = partials, ids
for projection in projections:
  out, partial_epoch, id_epoch, *_ = out.uop_program(
    partial_epoch, id_epoch, words, q8, scales, sums, fxn=main_program)
  out, partial_epoch, _ = out.uop_program(
    partial_epoch, slotmap, fxn=fixup_program)
```

Every access to the reused workspace must remain inside an opaque call with a
declared input/output ABI and must use the returned epoch. Passing the original
raw buffer to a later writer is not admissible.

## Hardware evidence

The two-chain gate uses the real Qwen3-8B gate Q4_K weight, two distinct
nonzero activations, the compact Q8 producer, finalized v4 IMMA main, and
fixup. TinyJit capture/replay and swapped input rebinding are bit-exact for all
checked outputs. The capture contains exactly two producer, two main, and two
fixup programs, with one partial base and one id base.

The 72-chain gate captures exactly 72 producer, 72 main, and 72 fixup programs.
Representative first, second, and last outputs are finite and bit-exact after
input rebinding. It retains one `22,283,600`-byte workspace instead of
`1,604,419,200` bytes for 72 distinct workspaces, a reduction of
`1,582,135,600` bytes. Hot research-gate replay is about 33.9 ms in this
same-weight fixture; this is a substrate timing, not a full-model wall claim.

Evidence:

- `docs/task_workflow/evidence/nv-prefill-ordered-scratch-20260828/native-two-chain.json`
- `docs/task_workflow/evidence/nv-prefill-ordered-scratch-20260828/native-72-chain.json`

Harness:

- `extra/llm_research/prefill/nv_q4_imma_ordered_scratch_gate.py`

## Regression gates

- ordered two-chain producer/main/fixup exactness;
- unordered raw reuse rejection;
- 72-chain exactness with one workspace identity;
- TinyJit capture/replay input rebinding;
- finalized PROGRAM output-ABI classification;
- multi-CALL AFTER classification is order-independent: read-then-write and
  write-then-read are writable, while an all-read dependency set is read-only;
- rangeify, reduce-output, memory planner, JIT replay, multi-stream, and graph
  admission suites: 98 passed.

## Whole-model integration result

The default-off binding resets `partial_epoch` and `id_epoch` once at graph
construction and threads the returned epochs through every gate/up projection.
The production-shaped captured graph contains exactly 72 compact-Q8 producers,
72 v4 mains, and 72 fixups. All native calls share one physical partial base
and one physical id base.

The computed-input fix is also live in this graph: the 72 producers consume
zero-byte `SLICE` views of exactly 36 physical per-block norm-result buffers.
There is no norm-to-Q8 adapter launch or extra materialized norm buffer.

Full last-token logits versus a separate-process FP16 comparator are finite,
select the same token (`198`), and pass the established `rtol=0.02, atol=0.5`
gate. Max and mean absolute differences are `0.0675683` and `0.0143403`.

The synchronized unprofiled R9 minimum is `91.9953 ms` (`5565.5 tok/s`),
versus the recorded `83.793 ms` FP16 comparator. This is an `8.2023 ms`
deficit, so performance promotion remains **NO_GO**. Relative to the prior
72-workspace v4 arm at `93.2 ms`, ordered reuse recovers about `1.2 ms` while
reducing the scratch allocation from `1,604,419,200` bytes to `22,283,600`
bytes.

Evidence:

- `docs/task_workflow/evidence/nv-prefill-ordered-scratch-20260828/candidate-model-r9.json`
- `docs/task_workflow/evidence/nv-prefill-ordered-scratch-20260828/candidate-logits.json`
- `docs/task_workflow/evidence/nv-prefill-ordered-scratch-20260828/fp16-logits.json`
- `docs/task_workflow/evidence/nv-prefill-ordered-scratch-20260828/full-logit-quality.json`

Harness:

- `extra/llm_research/prefill/nv_q4_imma_ordered_model_gate.py`

## Exact surviving integration wall

Matched `PROFILE=1` HCQ graph traces show that scratch is no longer the main
debt. The ordered packed lifecycle totals about `40.903 ms` of profiled GPU
busy (`39.886 ms` main, `0.224 ms` Q8 producer, `0.793 ms` fixup), essentially
equal to the FP16 gate/up body's `40.947 ms` in the matched comparator.

The largest candidate-only row is 72 canonical packed-carrier copies, kernel
`E_73728_32_3_c81631061fb6bc9bbfa2c6c6e9d2e39116b6103d26b3711633b9886187464d4e`.
Those copies total `4.807 ms` of profiled GPU busy per forward. Their element
count is exactly the canonical Q4_K gate/up word carrier
(`12288 * 16 * 36 = 7,077,888`). Candidate total profiled GPU busy is
`101.077 ms` versus `95.648 ms` for FP16; the copies explain most of that
`5.429 ms` busy-time delta.

An independent zero-copy discriminator corrected the initial source
attribution. The binding admitted only an exact one-dimensional uint32 NV
buffer with direct `MODEL_PARAMETER` lifetime, exact carrier size, and a
read-only main-PROGRAM ABI. All 72 model carriers passed; all 72 main pointers
were the exact canonical bases. Removing the binding's unconditional
`words.contiguous()` reduced the copy census only from 72 to 70, not to zero,
and R9 did not improve (`92.7128 ms` minimum). Inspection of a surviving copy
showed a canonical packed buffer as input and a noncanonical `SLICE` as output.
Therefore at most two copies belonged to the binding boundary; 70 arise later
from captured/callified carrier ownership. The failed binding experiment was
reverted. Eliminating the remaining copies is a separate callify ownership
gate, not part of the ordered-scratch substrate.

Profile evidence:

- `docs/task_workflow/evidence/nv-prefill-ordered-scratch-20260828/candidate-graph-profile.jsonl`
- `docs/task_workflow/evidence/nv-prefill-ordered-scratch-20260828/fp16-graph-profile.jsonl`
- `docs/task_workflow/evidence/nv-prefill-ordered-scratch-20260828/zero-copy-weight-r9.json`
