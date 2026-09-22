# Flash two-branch causal result

## Outcome

Both remaining Flash branches were tested through their proper accounting
boundary. Neither is promoted.

| branch | semantic/mechanism result | paid result | disposition |
|---|---|---|---|
| cross-head K/V sharing | QG2 is bit-exact and only one of two sibling query-head warps loads the common K/V tile | device score time is 4.224 us versus 3.520 us control | primitive no-go |
| reuse-class producer admission | bit-exact; ql/qh payload is evict-first while scale/d metadata remains ordinary; full-entry Flash penalty moves from +0.608 to -0.016 us/layer | 4.138186 ms/token versus 4.068309 control midpoint, +69.877 us / -4.151 tok/s | wall no-go |

## What the tests establish

The grouped-Q test is now causal rather than topological. It reduces duplicate
global K/V loaders, but the full shared tile, two phase barriers, and shared
reads cost 0.704 us/layer more than the requests they replace. The result does
not reject every GQA algorithm; it rejects this explicit shared-memory lifetime.

The admission test separates quantized data by reuse class. The one-touch Q6
ql/qh payload is addressed through an evict-first alias of the immutable weight
allocation. Scale and `d` metadata keep ordinary caching. This completely
removes the measured predecessor-conditioning penalty, proving that the cache
displacement is controllable. It still fails the token contract: the changed
pointer ABI and streaming-load construction cost roughly 70 us/token in the
installed graph. Residency recovery is therefore real but not free.

The next admissible cache construction must preserve the existing kernel ABI
and producer service, for example a per-access cache semantic on the original
pointer or a native K/V persisting window. It must pass the same entry ledger
and reverse wall. No endpoint or tok/s claim changes from these tests.

## Evidence

- `docs/task_workflow/evidence/nv-flash-two-causal-branches-20260827/cross-head-kv-share-r1.json`
- `docs/task_workflow/evidence/nv-flash-two-causal-branches-20260827/cross-head-kv-share-r2.json`
- `docs/task_workflow/evidence/nv-flash-two-causal-branches-20260827/q6-reuse-class-r1.json`
- `docs/task_workflow/evidence/nv-flash-two-causal-branches-20260827/reuse-class-entry-control.json`
- `docs/task_workflow/evidence/nv-flash-two-causal-branches-20260827/reuse-class-entry-candidate.json`
- `docs/task_workflow/evidence/nv-flash-two-causal-branches-20260827/q6-reuse-class-wall-r1.json`
