# LUNA-050 BoltBeam constrained search contract

Verdict: `SUPPORTED` as a contract, `NOT_RUN` as a search. Actual entry/schema boundary: `/home/ubuntu/BoltBeam` emits `boltbeam.candidate_set.v1`; tinygrad consumes JSON-only runner requests and returns `tinygrad.search_gate_batch.v1` evidence. BoltBeam must not import tinygrad runtime or encode llama source as a candidate.

## Immutable request facts

- Model identity, GGUF digest, commit/worktree, gfx1100/wave32, power state, token fixture, exact context, route identity, and trace/resource artifact hashes are mandatory.
- Candidate records must state useful-work invariants: Hq=40, Hkv=8, head dim=128, G=5, causal masking, KV positions/layout, output/token parity, and no duplicated KV traffic.
- Evidence classification is mandatory: measured, source-proven, inferred, or unresolved. An unresolved field cannot justify an axis.

## Candidate families and axes

| Track | Admitted family | Axis | Justification | Current state |
|---|---|---|---|---|
| A ctx128 | None | None | LUNA-043 has no semantic owner. | Search prohibited. |
| B depth | G5 compiler lifetime/allocation primitive | expression lifetime/order or address recomputation, represented as a reusable tinygrad primitive | Retained G5 deep cliff; current alternatives are refuted/unsafe. | Compile-only only until resource attribution. |

Excluded axes: raw HIP/llama kernels, split size, K_ONLY, G reduction/fifth-head serialization, second-pass KV work, benchmark context exclusion, and unrelated scheduler toggles.

## Gate order

1. Schema/provenance validation rejects missing immutable fields or unknown axis justification.
2. Compile-only rejects invalid code, illegal STORE semantics, bounds/route mismatch, or resource regression.
3. Numerical gate requires deterministic token parity, finite values, cache bounds, and exact route identity.
4. Bounded kernel timing requires a positive kernel/route match and ordinary timing separate from profiler instrumentation.
5. Whole-model authority requires fresh processes, same-session interleaved baseline/candidate pairs, ctx512 non-regression within noise, and ctx4096 target comparison to llama.
6. Promotion writes only a reusable primitive plus evidence hashes; otherwise record a rejected candidate and delete temporary probes.

## Controls

Positive controls: a known packed-WMMA selection at 14B ctx512; an allowed direct-packed non-14B control; expected G5 flash tile/combine match; and a known llama MMQ/attention dispatch once tracing exists.

Negative controls: candidate with altered causal mask/KV bounds; candidate with changed route identity; missing trace match; profiler-only timing; and a synthetic-only prompt that omits `"hi"`/ctx128. Any control failure is rejection, not an optimization result.

## Blockers

LUNA-021 `TOOL_FAILURE` blocks source-to-dispatch validation and all trace-derived axes. Before any search, provide LUNA-021/022/023 bounded traces, LUNA-030/031/032/033 tinygrad artifacts, LUNA-034 fallback verdict, and an LUNA-044 revision with a causal Track A owner and/or G5 resource owner.
