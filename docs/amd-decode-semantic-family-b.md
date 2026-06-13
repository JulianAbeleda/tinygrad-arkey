# AMD Decode Semantic Family B

Date: 2026-06-13

Status: pre-registered bounded probe. This is not a promoted optimization path
unless the committed gates say so.

## Purpose

Family B tests whether a richer semantic schedule/codegen surface can beat the
current Q4_K/Q6_K primitive families after two prior surfaces saturated near
noise:

- descriptor-level `parts`/`LOCAL` search: exhausted;
- semantic schedule v0: isolated attention win, full-decode rejection;
- semantic codegen v1: Q4 direct output tied or regressed at microbench.

The goal is not "try more knobs." The goal is to test one concrete memory and
schedule mechanism that the previous surfaces did not exercise.

## Mechanism

Current Q4_K partial GEMV maps each output row independently across the packed
K blocks. For decode, the activation vector is shared by all output rows, while
the packed weights are row-specific. A row-grouped schedule gives the renderer a
different loop shape:

- group adjacent output rows under one outer row-group axis;
- keep the same packed Q4_K storage and split-K partial reduction;
- reuse the same activation slice across multiple adjacent row lanes inside the
group;
- keep quant-weight bytes unchanged, but try to reduce redundant activation
traffic and improve occupancy/scheduling around the packed-weight traversal.

This probe is therefore a memory/schedule test, not a compute-instruction test.
It is expected to help only if activation traffic or row-axis scheduling is a
meaningful part of the current Q4_K `ffn_down` cost. If the current kernel is
already dominated by row-specific packed-weight traffic and instruction issue,
row grouping should tie or regress.

## Target

Start with Q4_K `ffn_down` only:

- 8B Q4_K `ffn_down`: v1 direct-output regressed badly, so split-K/reduction is
  doing useful work and row/K scheduling may matter.
- 14B Q4_K `ffn_down`: v1 direct-output tied, so the same tensor family sits
  closer to the boundary where schedule changes could matter.

Do not blanket `ffn_gate`, `attn_q`, or `attn_k` in the first pass. If `ffn_down`
does not show a strong signal, broadening the same schedule family is not
justified.

## Candidate Surface

The first surface is `semantic-codegen-v2`, with exact-tensor candidates:

- family: `q4_k_packed_u32_grouped`;
- format: Q4_K only;
- role: `ffn_down` only;
- reduction: split-K partial;
- storage delta: zero, reuses existing packed Q4_K storage;
- row groups: `2` and `4`;
- parts: inherit the accepted current descriptor;
- local opts: inherit current local row-axis setting, plus row-lane upcast.

This is microbench-supported first. Runtime full-decode installation is a
separate step and is only justified if the microbench result is strong enough.

## Acceptance

The harness status remains mechanical:

- `>=3%` microbench gain: `raw_accept`;
- within `+-3%`: tie;
- below that: reject.

Strategic interpretation is stricter:

- `3-5%`: weak raw accept, likely full-decode tie after model dilution;
- `5-10%`: worth considering for full-decode only if it is on a dominant tensor;
- `>=10%`: strong raw accept and the first credible reason to wire runtime
  full-decode support.

No result is promoted from microbench alone. Promotion requires full-decode A/B
and a matching confirmation rerun.

## Escalation Rule

If Family B also ties or rejects on 8B and 14B, the conclusion is not "try knob
#4." The conclusion is:

> Semantic schedule knobs over the current hand-seeded primitive family are
> saturated within measurement noise; further progress toward llama.cpp-class
> performance likely requires a different kernel class, such as WMMA/wide-K or
> MMVQ-style lowering, or a deeper compiler representation change.

That escalation is compiler research. It should not be mixed into this bounded
probe.
