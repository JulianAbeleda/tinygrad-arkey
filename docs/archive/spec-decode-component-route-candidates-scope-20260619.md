# Spec decode component-route candidates scope - 2026-06-19

Purpose: define what would count as a concrete component route after TBF-0..2 stopped before implementation.

This is the next layer down from `spec-decode-tcheap-batched-forward-project-scope-20260619.md`. It is decode-only,
does not touch prefill, does not route `SPEC_DECODE`, and does not start TBF-3. It scopes the candidates that could
earn another TBF-2 run.

## Current Gate State

TBF-2 current baseline:

| component | current T5/T1 | required | state |
|---|---:|---:|---|
| Q4_K GEMM | `2.916x` | `<=1.5x` | fail |
| Q6_K/lm_head | `5.831x` | `<=1.5x` | fail |
| attention/reduces | `3.061x` | `<=1.5x` | fail |
| linears group | `3.523x` | `<=1.5x` | fail |

Full spec verify also needs T=5 verify near `<=1.3-1.5x` one T=1 pass. Passing only one component is not enough; it
only earns the next component probe.

## Candidate Classes

### Candidate L - grouped short-block quantized linears

Goal:

```text
all target quantized linears for T=K+1 run at <=1.5x their T=1-equivalent cost
```

Required roles:

- Q4_K ffn_gate/up;
- Q4_K attention q/o;
- Q4_K ffn_down if present in the role map;
- Q6_K ffn_down;
- Q6_K lm_head;
- any Q6_K projections that appear in the verify component map.

Plausible implementation families:

| family | idea | first proof | risk |
|---|---|---|---|
| L1 native short-T MMVQ | each packed weight block feeds T accumulators before moving on | one Q4_K + one Q6_K role at T=5, exact, no q8 activation change | current batched GEMM already exists and is still too linear |
| L2 artifact/import | find or import a mature small-T quantized GGUF-style kernel | standalone Q4_K/Q6_K T=5 role kernel | likely no mature Tensile equivalent for GGUF Q4_K/Q6_K |
| L3 renderer project | teach tinygrad a grouped short-T GEMV/GEMM schedule | grouped linears reach gate across Q4_K and Q6_K | project-level scheduler/register work |

Gate to rerun TBF-2:

- Q4_K role sample `<=1.5x` T1-equivalent at T=5;
- Q6_K/lm_head sample `<=1.5x` T1-equivalent at T=5;
- exact argmax/logit tolerance preserved;
- no extra activation-format quality risk.

Kill:

- Q4_K-only win;
- Q6_K-only win;
- any route that still reads/reduces weights approximately once per T column;
- any route that requires q8 activation lifecycle work before proving T-cheapness.

### Candidate A - short-block causal verify attention

Goal:

```text
verify attention/reduces for T=K+1 run at <=1.5x their T=1-equivalent cost
```

Required semantics:

- existing KV prefix;
- proposed block K/V;
- causal lower-right mask inside the proposed block;
- GQA;
- exact enough for greedy byte identity;
- no prefill-style large-T path.

Plausible implementation families:

| family | idea | first proof | risk |
|---|---|---|---|
| A1 flash-decode generalized to short-T | extend current T==1 flash decode to process T=3/4/5 queries together | T=5 attention-only exactness and `<=1.5x` T1-equivalent | current flash-decode kernels assume one query |
| A2 block-local mini-flash | specialized short-block attention over existing KV plus intra-block KV | standalone attention/reduce component gate | attention alone is insufficient for whole verify |
| A3 reuse current SDPA with graph/layout fixes | reduce current attention/reduce overhead without new kernel family | component gate from `3.061x` to `<=1.5x` | prior reuse-free attention attempts were refuted |

Gate to rerun TBF-2:

- T=5 attention/reduces `<=1.5x` T1-equivalent;
- exact vs current SDPA/target verify;
- no KV corruption;
- no dependency on full prefill route.

Kill:

- reuse-free global reread pattern;
- route only improves one of attention or reduce while combined attention/reduces remains >1.5x;
- requires changing normal decode attention.

### Candidate C - combined short-block verify

Goal:

```text
linears + attention together make T=5 verify plausibly <=1.3-1.5x one pass
```

This is the only candidate class that can directly earn TBF-3/TBF-4. It can be composed from Candidate L and Candidate
A, but neither one alone is enough.

Gate:

- projected T=5 verify `<=1.5x` one pass using measured component timings;
- both linears group and attention/reduces meet their component gates;
- accept/commit plan avoids host reads.

Kill:

- if either linears or attention remains T-linear.

## Recommended Sequence

Do not build both components at once. Sequence the cheapest discriminator first:

1. **A0: attention feasibility audit.**
   - Reason: attention/reduces is the largest T=5 share (`48.6%`) and has a clear existing T==1 flash-decode reference.
   - Output: can current flash-decode architecture generalize to T=3/5 without full rewrite?
   - Gate: if no, Candidate A is project-level and cannot rescue spec.

2. **L0: grouped-linears feasibility audit.**
   - Reason: linears group is also far over gate (`3.523x`) and spans Q4_K + Q6_K/lm_head.
   - Output: can a single schedule idea cover both Q4_K and Q6_K?
   - Gate: if no, Candidate L is project-level.

3. **C0: combined projection.**
   - Use measured/proposed A and L ceilings to recompute full verify.
   - Gate: only if projected verify `<=1.5x` one pass should implementation begin.

## Phase Plan

### SCR-0 - candidate inventory

Deliverable:

- `bench/qk-spec-component-routes/candidates.json`
- rows for L, A, and C with gates, risks, and evidence.

Gate:

- every candidate names the component gate it must satisfy.

### SCR-1 - attention feasibility audit

Read existing code/docs:

- `extra/qk_flash_decode.py`;
- `tinygrad/llm/model.py::_attention`;
- prior flash-decode results and attention refutations.

Deliverable:

- decide whether A1/A2 has a bounded proof surface.

Gate:

- PASS only if a T=3/5 attention-only proof can be scoped without altering normal decode.

### SCR-2 - grouped-linears feasibility audit

Read existing code/docs:

- Q4_K/Q6_K `K != 1` batched paths in `tinygrad/llm/model.py`;
- `extra/q4_k_gemv_primitive.py`;
- `extra/q6_k_gemv_primitive.py`;
- SDB/TBF component artifacts.

Deliverable:

- decide whether L1 has a bounded proof surface across Q4_K and Q6_K.

Gate:

- PASS only if the same schedule family plausibly covers both Q4_K and Q6_K/lm_head.

### SCR-3 - combined projection

Deliverable:

- `bench/qk-spec-component-routes/projection.json`

Gate:

- PASS only if combined proposed ceilings reach T=5 verify `<=1.5x` one pass.

### SCR-4 - implementation decision

Outcomes:

- `BUILD_ATTENTION_PROOF` if A passes and L has a credible path;
- `BUILD_GROUPED_LINEARS_PROOF` if L passes and A has a credible path;
- `PROJECT_LEVEL_CLOSE` if either component lacks a bounded proof surface;
- `REST` if no component route beats the current TBF gates.

## Current Expectation

Based on current evidence, the likely result is `PROJECT_LEVEL_CLOSE`:

- Q4_K-only is already known insufficient.
- Q6_K/lm_head is more T-linear than Q4_K.
- attention/reduces is the largest share, but attention alone is insufficient.
- generic prefill-style fixes are the wrong regime.

The purpose of SCR-0..3 is to make that expectation explicit and prevent accidental implementation work before a
component route earns it.
