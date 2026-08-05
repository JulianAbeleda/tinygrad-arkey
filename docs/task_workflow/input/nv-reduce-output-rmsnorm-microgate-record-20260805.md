# NV REDUCE_OUTPUT RMSNorm microgate

Status: **isolated construction and included-cost gate PASS; production route
NO-GO because the safe contract admits zero decode norms. Shipped policy remains closed.**

The new default-off `Ops.REDUCE_OUTPUT` carrier keeps the exact ordinary RMSNorm
as source zero.  At the concrete callified STORE, rangeify admits only an
identity-preserving realized x/weight/output binding on NV/CUDA.  It replaces
that STORE with one normal UOp SINK/CALL containing the reduction, barrier,
restored output phase and stores.  Lazy or movement inputs return the untouched
fallback.  There is no `UOp.custom_kernel`, source string, or assembly.

Production integration is independently closed by
`decode-reduce-output-rmsnorm-route-policy.json` (`promoted_targets: []`). Model
load resolves the record once from backend/architecture facts and installs the
decision on the model and blocks. Only explicit decode call sites, gated by
`not _prefill`, may attach the marker. `nn.RMSNorm` itself cannot enable this
route. Marker creation and concrete STORE admission remain separate fail-closed
checks; the existing Path-3 policy is unchanged.

Hermetic contracts pass:

* realized fp16 `(1,4096)` topology: two ordinary programs -> one
  `reduce_output_rmsnorm_1_4096` program;
* lazy `x+x`: fails closed, with no materialization inserted by this route;
* AST contains a REDUCE range, one barrier, and a LOOP range reusing axis id 2
  after the reduction; no CUSTOM op;
* result is bitwise identical to ordinary tinygrad RMSNorm.

The first synchronized host replay showed the familiar +58-us artifact because
tinygrad deliberately elides a graph containing only one kernel.  The fair A/B
therefore used the existing diagnostic `GRAPH_ONE_KERNEL=1` for both arms:

| arm | median us/replay |
| --- | ---: |
| ordinary two-program RMSNorm | 55.374 |
| REDUCE_OUTPUT one-program RMSNorm | 54.121 |
| candidate delta | **-1.253** |

The candidate device body measured 3.01 us in the compile trace, consistent
with the prior hand-body evidence.  The isolated route therefore clears its
construction, correctness, topology and included-cost gates.

## Unsafe first real-token attempt

The setup-only prefill route was pinned to the historical non-fused authority,
then a d512 control completed normally. The first forced candidate hit an Xid
31 because its late view predicate stripped dependency-bearing movement and
treated a bare callified PARAM as sufficient ownership evidence.

The forced candidate is a hard NO-GO. During its first full-token process the
driver recorded:

`NVRM Xid 31: MMU Fault, FAULT_PTE, ACCESS_TYPE_VIRT_READ @ 0x20_0a064000`

No candidate wall or token result is qualified. GPU testing stopped immediately.
The isolated admission predicate is insufficient for a callified model graph:
callification can expose a lazy/aliased invocation view as a PARAM/RESHAPE even
though stripping that movement node does not prove byte offset, span, alias
lifetime, or dependency ownership for a new opaque CALL. Standalone realized
buffers satisfy those assumptions; the full token does not.

This identifies the remaining required substrate precisely: the late
reduction-output call must bind through an invocation-local typed view carrying
base allocation, byte offset/span, and producer dependency, instead of reducing
the value to its bare buffer. Until that contract exists and passes hermetic
offset/alias/lifetime tests, the model policy must stay empty and the route must
not be force-opened. The isolated -1.253-us result remains a body/graph-potential
measurement only, not recoverable token credit.

## Safe follow-up and decode-only correction

The admission was tightened after the fault. `ReduceOutputSpec` now records
`input_identity_at_marker` before callify; late lowering may not infer safety
from a PARAM created by callification. Binding accepts only an equal-span
RESHAPE over one invocation PARAM/BUFFER. AFTER (lost dependency),
MEMORY_SEMANTIC, PERMUTE, SHRINK, EXPAND and offset views are rejected.
Hermetic tests cover each rejection.

The admission was tightened, but the next forced run exposed a separate model
integration error: `[151936, 151936]` was not an authority token stream. The
vocabulary extent is 151936, so token 151936 is out of range. The marker flag
was stored on `nn.RMSNorm`, whose `__call__` also runs during prefill; candidate
prefill was therefore not the control graph. This completely accounts for the
apparent control drift and invalidates that candidate wall row.

The marker was moved to explicit model decode call sites for block attention,
FFN, q/k and output norms. Every call site gates it with `not _prefill`; model
load stores no per-norm marker flag. A direct hermetic test proves that setting
the old RMSNorm attribute cannot create `REDUCE_OUTPUT`.

## Qualified finite/logit and wall gates

A fresh-process d512 control/candidate full-logit gate used fixed valid feedback
token 1 after populating the production KV prefix. Both arms had prelude token
13876, returned finite float32 logits of shape `(2,1,151936)`, produced in-range
argmax tokens `[64461,4710]`, and were bitwise identical: zero mismatches,
max-abs zero, SHA-256
`506ee637c17af7d8c62f4785fc870870bb4b851bd0e8d9483d93a73a421f609b`.
No new Xid was recorded.

Only after that pass, a timeout-bounded greedy d512 A/B/A wall bracket ran with
three repetitions of eight decode tokens per arm:

| arm | wall ms/token | graph calls | REDUCE_OUTPUT calls |
| --- | ---: | ---: | ---: |
| control A | 5.560499 | 946 | 0 |
| candidate B | 5.588062 | 946 | 0 |
| control A2 | 5.562211 | 946 | 0 |

All repetitions in all arms produced the same eight tokens. Candidate B is
0.480% slower than the control midpoint, but this is not a candidate cost:
the graphs are identical and ordinary run variance owns the difference. The
causal result is **zero production admissions, zero programs removed, and zero
recoverable wall credit**.

The safe predicate is doing its job. The first production census therefore
declined every marker. P2a remained NO-GO and the policy stayed empty.

## Shared precompiled-output substrate follow-up

P2b subsequently added a generic pre-callify predicate for only an exact
`GETTUPLE(FUNCTION(precompile=True))` (plus zero-offset reshape/semantic
wrappers). P2a added the matching post-callify validator for only an exact
equal-span `AFTER(BUFFER, CALL)`: the buffer must be the final CALL argument,
the body must STORE through the PARAM for that same slot, and the downstream
ordinary CALL retains the complete AFTER dependency. Non-precompiled function
outputs, permutation/shrink views, arbitrary AFTER dependencies and mismatched
output slots fail hermetic tests. The combined focused suite passes 17 tests.

A census-only forced production capture still contained 946 calls and zero
`reduce_output_rmsnorm` programs. A call-site observer accounted for all 290
promoted norm invocations (the trace observes ignore/capture construction):

| count | shape | exact marker input |
| ---: | --- | --- |
| 72 | `(1,1,4096)` | `MEMORY_SEMANTIC(CONTIGUOUS(...))` |
| 72 | `(1,1,4096)` | `MEMORY_SEMANTIC(ADD(...))` |
| 2 | `(1,1,4096)` | `MEMORY_SEMANTIC(CAST(...))` |
| 72 | `(1,32,1,128)` | `PERMUTE(...AFTER...)` |
| 72 | `(1,8,1,128)` | `PERMUTE(...AFTER...)` |

Both identity predicates were false for every row. The shared GETTUPLE proof
does not reach production because `FFNBlock.__call__` explicitly wraps the
precompiled `_run(...)` result in `.contiguous()` before attaching its semantic
role. That wrapper is the sole plausible next reopen: prove explicit
CONTIGUOUS as an invocation-owned materialization across replay, then admit
only the 72 block-output rows. ADD/CAST/PERMUTE remain outside the contract.
No second full-logit or wall run was authorized because the real graph did not
change. Policy remains closed.

## Realization-stage closure

Late tracing then accounted for why even identity-qualified markers did not
reach the selector. The production graph structurally deduplicated to seven
markers; three carried `input_identity_at_marker=true`, but every one hit the
fallback rule. `lower_reduce_output_store` was never invoked because no
`STORE(REDUCE_OUTPUT)` existed at that rangeify stage. A realization-map hook
also fired zero times: generic realization does not enter the precompiled
FUNCTION body containing these markers.

One final narrow construction wrapped only identity-qualified decode markers
in the ordinary CONTIGUOUS materialization primitive inside that body. The
census grew from 946 to 982 calls—exactly 36 extra generic programs—while still
containing zero `reduce_output_rmsnorm` programs. This is a decisive NO-GO:
forcing the boundary exposes fallback work but does not move selector timing
past STORE creation. The construction and dead realization hook were reverted;
no logits or wall test followed.

The ordinary REDUCE_OUTPUT route is therefore closed at the scheduler-stage
boundary, not at CUDA code generation. Reopening it requires a first-class
post-bufferization selector or a consumer-owned fused route; more view
broadening or forced CONTIGUOUS nodes cannot solve it.
