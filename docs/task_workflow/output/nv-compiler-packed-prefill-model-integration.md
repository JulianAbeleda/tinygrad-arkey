# Compiler-packed dense prefill model integration

## Outcome

PASS as a default-off research route.  The exact dense Qwen3-8B, pp512 gate/up
path now executes this lifecycle:

```text
fp16 normalized activation
  -> exact compact Q8 record producer
  -> ordinary compiler-generated packed Q4_K IMMA PROGRAM
  -> direct fp32 output
```

There is no handwritten main kernel, old fixup call, expanded gate/up weight,
or partial/fixup workspace.  The generated main consumes the producer record
and the canonical model-owned Q4_K words directly.  Ordinary behavior is
unchanged unless `NV_COMPILER_Q4_IMMA_PP512=1` is set, and the binding rejects
all model/shape identities outside the exact dense Qwen3-8B pp512 topology.

## Why the reusable compiler asset is finalized

The ordinary carrier matmul is first compiled through tinygrad's standard
compiler with the admitted packed fragment and accumulator contracts.  The
resulting compiler PROGRAM is then retained as an opaque reusable invocation,
including its compiler-emitted source, binary, launch ABI, tensor-core ledger,
and candidate identity.

This explicit finalization is necessary at a nested model boundary.  Before
code generation, the ordinary matmul does not yet expose `ProgramInfo.ins`, so
an enclosing precompiled function cannot prove that its compact-Q8 result is a
read-only generated-program input.  Reusing the finalized PROGRAM makes that
ABI explicit.  The generic computed-input ownership pass can then preserve the
exact producer allocation through nested functions without inserting a copy;
writable, aliased, partial, or non-program uses still fail closed.

## Whole-model gates

| Gate | Result |
|---|---:|
| Compact-Q8 producer calls | 72 |
| Compiler-generated direct-output main calls | 72 |
| Old fixup calls | 0 |
| Packed-weight transport copies | 0 |
| Main weight arguments / unique canonical bases | 72 / 72 |
| Expanded gate/up FP16 weight bytes | 0 |
| Q8 records | 72 |
| Partial workspace bytes | 0 |
| Same-activation capture/replay | exact |
| Distinct activation changes output | yes |

The full-vocabulary comparison against a fresh resident-FP16 control is finite,
selects the same token, and passes `rtol=0.02, atol=0.5`.  The observed maximum
absolute logit difference is 0.1144 and the mean absolute difference is 0.00829.

The clean synchronized R9 wall is 74.695 ms for the compiler-packed arm versus
83.503 ms for the resident-FP16 control.  That is 8.808 ms less latency (10.55%)
and 6855 versus 6132 prompt tokens/s (11.79% more throughput) for this pp512
workload.  These are prefill chunk rates, not decode token rates.

An independent profiled three-replay capture retained 3,267 kernel events.  It
accounts for 216 compact-Q8 launches and 216 generated-main launches across the
three replays, matching 72 of each per replay.  Profiling increases host wall,
so the unprofiled R9 artifacts remain the timing authority.

## Interpretation and remaining boundary

This closes the model-lifecycle substrate gate: a computed compact activation
can flow into a compiler-generated packed-weight kernel, through nested
precompiled functions and capture/replay, with canonical weights and no
transport/fixup copies.  It also wins decisively over the current resident-FP16
gate/up control.

It does not establish a production-default promotion or a final prefill
roofline.  The compiler-generated main still owns the dominant service time,
so future recovery belongs to compiler kernel geometry/pipeline quality and to
other lifecycle regions, not to another model-side binding or copy workaround.
The route remains default-off pending a separate promotion decision.

## Evidence

- `docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/model_candidate_r9.json`
- `docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/model_fp16_r9.json`
- `docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/model_compare.json`
- `docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/model_candidate_profile.json`
