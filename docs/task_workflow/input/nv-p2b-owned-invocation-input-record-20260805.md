# P2b owned invocation-input boundary record

Date: 2026-08-05. Target: native NV d512, Qwen3-8B-Q4_K_M.
Status: **the generic boundary removes one copy per fused attention-O block,
but the composed topology gate still fails; route closed and zero recovery
credit.** No policy/default was promoted and no logits or wall timing followed.

## Exact causal class found on CPU

The rendered `E_32_32_4_86a23...` source was recovered from the compiler cache.
It is not a cast, permutation, or arithmetic epilogue: it loads one `float4`
from a 4096-element fp32 input and stores it unchanged to a 4096-element fp32
output. Each invocation moves exactly 16 KiB.

The old hermetic construction connected a precompiled producer directly to an
opaque projection epilogue and therefore missed the real nesting. The exact
failing spelling is:

```text
precompiled producer output
  -> precompiled consumer FUNCTION invocation input
  -> equal-span RESHAPE(PARAM)
  -> opaque attention-O CALL read argument
```

Before the amendment this CPU graph compiled as producer, two fp32 identity
copies, fused epilogue, and consumer output. This reproduces the two-copy
population rather than inferring it from the real-model count.

## Closed-default generic construction

`CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT` remains default zero. Under the
existing opt-in only, callify now exposes precompiled consumers top-down while
their concrete invocation arguments remain visible. It admits a direct input
only when all of these facts hold:

1. the concrete invocation argument is an exact, uniquely owned precompiled
   output with identical dtype and element span;
2. the opaque consumer argument is literally
   `CONTIGUOUS(RESHAPE*(PARAM))`;
3. the PARAM is read-only in that CALL and is not an output slot;
4. the same PARAM is not repeated as another CALL argument;
5. no permute, shrink, offset, cast, or unequal-span movement is crossed.

The dependency remains the producer's `AFTER(output, CALL)`. Default-off
construction uses the original bottom-up pass and input normalization exactly.
The exact nested CPU graph loses both `E_86a2` copies with the gate on, while
the default-off arm retains two. Movement and offset cases fail closed.

Focused callify, projection-boundary, reduce-output, multioutput replay, and
ping-pong replay coverage passes: **54 tests**. The final focused projection
file alone passes **17 tests**.

## Bounded real-model census

The expensive copy observer was disabled for the first accepted topology run;
the rendered hash count is the construction gate. Artifact:
`docs/task_workflow/output/nv-attention-o-owned-invocation-input-census-20260805.json`.

| arm | programs | E programs | `E_86a2` | residual adds | fused O | token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prior composed | 874 | 418 | 71 | 1 | 35 | 38835 |
| owned invocation input | 841 | 385 | 36 | 1 | 35 | 38835 |
| required | <=804 | -- | 1 | 1 | 35 | 38835 |

The construction removes exactly 35 identity copies, one per fused block, and
33 net programs. It leaves one new `E_86a2` per fused block plus the common
baseline instance. Therefore it fails the predeclared topology gate by 37
programs and 35 copies. Token equality is direction only and cannot promote a
topology failure.

## Remaining attribution is unresolved

The post-callify observer originally recursively sorted every rendered PROGRAM
body and held the GPU without output. It was narrowed to the one compiled
LINEAR containing both `E_86a2` and `epi_resadd_4096_4096`, then changed to use
the LINEAR's direct ordered CALL sources rather than traversing instruction
UOps. CPU trace tests remained green.

Two bounded real attempts still produced no exact slot/chain record: the first
was stopped at about 180 seconds, and the final strict attempt exited 124 at
150 seconds. The GPU lock was released after each. Under the explicit gate,
the remaining 35-copy class is **UNRESOLVED**, not labeled residual or
activation by inference.

## Verdict

`TOPOLOGY_NO_GO`. The generic support work is real but insufficient for the
attention-O composition. No full logits, no A/B/A wall run, no promotion, no
commit, and no push are authorized by this record. The next reopen requires a
bounded exact writer-to-consumer slot trace or an independently hermetic
construction which reproduces the surviving one-copy-per-block class and
removes it without adapters.
