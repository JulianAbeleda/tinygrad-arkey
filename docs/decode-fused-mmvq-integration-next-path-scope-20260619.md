# Decode fused-MMVQ integration next path scope - 2026-06-19

Purpose: scope the next decode path after `decode-bandwidth-bound-pmu-learning-20260619.md` converged the mechanism:
tinygrad's standalone GEMV kernel is not the problem; the in-model integration loses the bandwidth.

This is decode-only. It does not touch Claude's prefill work, does not reopen spec-decode T-cheap verify, and does not
change defaults.

## Authority

Latest authority:

`decode-bandwidth-bound-pmu-learning-20260619.md`

Key numbers:

| engine | standalone GEMV | in-model weight-GEMV |
|---|---:|---:|
| tinygrad | `76%` HBM peak | `~44%` HBM peak |
| llama.cpp | `57%` HBM peak | `~54%` HBM peak |

Interpretation:

tinygrad has the better standalone GEMV kernel, but loses `32` HBM-points when embedded in the model. llama loses only
`3` points. The next decode path is therefore **fused-MMVQ integration**:

1. amortize activation to `Q8_1` once across input-sharing GEMVs;
2. preserve max-occupancy, low-VGPR, high-grid launch behavior in the real model.

Target:

```text
in-model weight-GEMV bandwidth: ~44% -> >=54% HBM peak
decode speed: >=5% W==D sustained improvement, strong >=10%
quality: greedy byte-exact for byte-identical routes, dNLL-gated for q8 routes
```

## Non-Goals

- Do not build another standalone GEMV kernel first. Standalone already clears the reference.
- Do not start spec-decode TBF-3. SCR-0..4 closed that bounded path.
- Do not make a default-on lossy route.
- Do not treat q8 dot availability alone as a primitive. The primitive is producer format + reuse + consumer + launch
  shape inside the model.

## The Route

This path has two tracks that must be measured separately before combining.

### Track A - Activation/Q8 Lifecycle Integration

Question:

```text
Can tinygrad produce Q8_1 once per shared activation and reuse it across the matching MMVQ consumers without paying
the old per-linear quantization tax?
```

Known reuse map:

| activation | candidate consumers | q8 reuse count | state |
|---|---|---:|---|
| `ffn_norm(h)` | `ffn_gate`, `ffn_up` | `2` | highest-value q8 route |
| `attn_norm(h)` | `attn_q`, `attn_k`, `attn_v` | mixed Q4/Q6 | not first; consumer mismatch |
| attention output | `attn_o` | `1` | no amortization |
| SwiGLU output | `ffn_down` | `1` | no amortization |
| final hidden | `lm_head` | Q6_K | not Q4 q8 route |

Prior evidence:

- `q8-mmvq-lifecycle-deep-result-20260619.md`: native fused producer is blocked by multi-granularity
  reduce + multi-output expressibility.
- the q8 artifact/research route proves the lifecycle concept, but remains default-off and policy-bound.
- expected decode EV from gate/up q8 alone is modest (`~3-6%`) because reuse is only `2`.

Track A should not ask "can dot4 run?" It should ask whether the full lifecycle moves W==D decode.

### Track B - In-Model Occupancy / Launch-Shape Preservation

Question:

```text
Why does the same class of tinygrad GEMV work hit ~76% standalone but only ~44% in-model, and can the model route
preserve the standalone launch/resource shape?
```

llama's measured launch signature:

- `mul_mat_vec_q`;
- grid roughly `131072`;
- workgroup `32`;
- low VGPR, roughly `24-40`;
- `lds=0`;
- huge number of tiny one-wave workgroups to saturate memory-level parallelism.

tinygrad must be audited role-by-role in-model, not just from standalone harnesses.

Track B is byte-identical if it only changes launch shape/scheduling. It should be attempted before any new lossy q8
default decision, because it targets the `44% -> 54%` gap without a quality gate.

## Phases

### FMI-0 - authority and stale-frame cleanup

Goal: prevent older "better kernel" or "spec is next" framings from steering work.

Work:

- mark the latest mechanism in source-of-truth docs;
- list stale docs that are now provenance-only;
- keep spec-decode and prefill as orthogonal rows.

Deliverable:

- this scope;
- README/source-of-truth pointer.

Gate:

- next work is framed as in-model integration, not standalone kernel search.

### FMI-1 - in-model GEMV loss atlas

Goal: localize the `76% -> 44%` collapse.

Work:

- run/refresh a warm W==D decode with per-role kernel attribution;
- for each role, record device time, bytes read, effective GB/s, `%HBM`, launch grid, workgroup, VGPR, LDS, spills if
  available;
- compare against the best standalone reference for the same role shape;
- tag roles as `activation_lifecycle`, `occupancy_launch`, `coverage`, or `already_close`.

Deliverables:

- `bench/qk-decode-fused-mmvq-integration/inmodel_loss_atlas.json`
- `bench/qk-decode-fused-mmvq-integration/summary.md`

Gate:

- at least one role group with `>=5%` projected e2e movement and a named mechanism.

Kill:

- if the `44%` aggregate cannot be reproduced or no role group has measurable room.

### FMI-2 - llama/tinygrad launch-contract diff

Goal: make "sustained max occupancy" concrete.

Work:

- capture llama `mul_mat_vec_q` launch metadata by role;
- capture tinygrad in-model Q4_K/Q6_K launch metadata by role;
- diff grid/workgroup, VGPR, LDS, occupancy, waves/CU, instruction counts if available;
- identify whether tinygrad's standalone launch contract differs from its in-model launch contract.

Deliverable:

- `bench/qk-decode-fused-mmvq-integration/launch_contract_diff.json`

Gate:

- a concrete tinygrad-side delta exists, such as higher VGPR, lower grid count, smaller occupancy, extra kernels, or
  role path fallback.

Kill:

- if tinygrad in-model launch contract already matches standalone and llama, then the remaining gap is not launch
  shape.

### FMI-3 - activation-amortized q8 replay

Goal: isolate Track A in-model movement using the least new machinery.

Work:

- use the already proven research/artifact q8 route or an equivalent replay route;
- route only `ffn_gate`/`ffn_up` behind a flag;
- quantize once per `ffn_norm` activation and reuse for both consumers;
- measure W==D ctx sweep and dNLL.

Deliverables:

- `bench/qk-decode-fused-mmvq-integration/q8_gateup_replay.json`
- result doc if it moves.

Gate:

- W==D sustained `>=3%` decode improvement;
- strong `>=5%`;
- dNLL within the established q8 gate.

Kill:

- if q8 reuse still only moves `<=2%`, keep q8 artifact research-only and do not fund native producer work.

### FMI-4 - occupancy-preserving tinygrad route

Goal: isolate Track B without changing activation format.

Possible implementation families:

| family | idea | first proof | risk |
|---|---|---|---|
| B1 schedule/opts replay | force the standalone-winning launch/resource shape into the in-model role | one role group reaches near standalone BW in-model | may be blocked by graph/context differences |
| B2 runtime/cache route | ensure in-model route uses the same compiled program/metadata as standalone | exact role kernel identity + BW recovery | runtime cache/key surgery |
| B3 renderer project | teach AMD renderer to preserve low-VGPR/high-grid MMVQ launch contracts | role group `>=54%` in-model | project-level scheduler/codegen |
| B4 artifact/import | import known-good MMVQ artifact if one exists | HCQ launch exactness + BW | external artifact policy; likely harder than fp16 Tensile |

Gate:

- one high-share role group moves `>=10%` relative isolated in-model or projects `>=5%` W==D.

Kill:

- if only standalone improves or if the route breaks graph capture.

### FMI-5 - combined fused-MMVQ integration flag

Goal: combine the passing parts of Track A and Track B only after both are independently measured.

Work:

- research flag only, default off;
- W==D ctx sweep;
- greedy exact for byte-identical parts;
- dNLL for q8 parts;
- PMU atlas after routing to confirm weight-GEMV `%HBM` moved toward `>=54%`.

Gate:

- sustained decode `>=5%`, strong `>=10%`;
- in-model weight-GEMV `>=54%` HBM or a measured explanation for why tok/s moved without that;
- no prefill regression;
- no default route.

## Expected Outcomes

| outcome | meaning | decision |
|---|---|---|
| Track A passes, Track B fails | q8 lifecycle gives small real win, but `44% -> 54%` remains | keep q8 research flag; do not claim llama parity |
| Track B passes, Track A fails | byte-identical launch/integration was the main gap | fund native occupancy route first |
| both pass | true fused-MMVQ lifecycle exists in tinygrad | integrate behind research flag and rerun full decode suite |
| both fail | decode learning is complete; rest on shipped decode + spec/project-level future | stop decode implementation |

## Practical Recommendation

Start with **FMI-1 and FMI-2**. They are measurement-only and decide whether Track B is concrete. Do not jump straight
to q8 producer work: prior q8 work already showed the native producer is codegen-walled and the EV is capped unless
the occupancy/integration loss also moves.

If FMI-1/2 identify a concrete launch-contract delta, pursue Track B first because it is byte-identical. If they do
not, run FMI-3 once to decide whether the measured q8 artifact route is worth keeping as the only remaining decode
research flag.
