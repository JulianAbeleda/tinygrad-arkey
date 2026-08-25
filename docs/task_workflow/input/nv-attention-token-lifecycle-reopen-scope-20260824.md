# NV attention token-lifecycle reopen scope

Date: 2026-08-24

## Objective

Recover measured single-token decode wall time from the installed dense
attention chain, using the current promoted graph as the only control:

```text
attention norm/provider -> Q/K/V projections and cache ownership
  -> flash score -> flash combine -> O projection/residual
```

The campaign must distinguish a fast isolated kernel from a fast token. A
kernel change is not promotable unless its causal device movement survives a
fresh production wall bracket.

## Accounting gate

1. Capture a fresh unprofiled depth-512 endpoint and a fresh profiled graph.
2. Reconstruct every steady attention chain on one device clock, including
   direct dependencies, edge intervals, per-role duration, node cardinality,
   and device union.
3. Separate body time from boundary time. Zero-duration dependency edges are
   closed; missing dependency information is a wall to investigate, never an
   independence claim.
4. Reconcile the current rows against the retained same-build llama ledger.

## Candidate admission

Do not retry the closed wider-combine or single-stage flash constructions
without new information. Admit a candidate only when the fresh accounting
shows one of:

- a measurable producer/consumer boundary or redundant materialization;
- compulsory bytes that can legally be removed;
- a cold production-rate deficit with a specific scheduling/codegen cause;
- exposed independent work supported by the exact dependency graph.

Every candidate requires an exact numerical gate, an isolated or counter
mechanism gate, unchanged semantics, a topology census, and an unprofiled
control/candidate/control bracket with at least seven measured repetitions.
Book only recovery against the faster control. If an expected pass meets an
information wall, promote the question and repair the accounting before
closing the mechanism.

## Promotion and completion

Accepted behavior must be target-scoped to NV `sm_120`, fail closed, carry an
explicit rollback, and include focused tests. After each accepted change,
rerun the installed endpoint and full role ledger. Finish with the remaining
latency to 240 tok/s and to the retained llama authority, then commit the
implementation, summarized evidence, and result document. Raw profiler dumps
may remain local when a compact committed summary preserves their hashes and
controlling measurements.
