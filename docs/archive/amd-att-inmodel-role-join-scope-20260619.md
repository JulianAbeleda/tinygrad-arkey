# AMD ATT In-Model Role Join Scope

Date: 2026-06-19

## Purpose

Use the working ATT interval tracer on one real in-model decode role and join the trace to the HCQ program identities
launched inside that interval.

This phase is the direct successor to `amd-att-primitive-attribution-result-20260619.md`. The prior pass proved ATT can
body-attribute primitive surfaces. It did not prove whether the real model route preserves or changes the same primitive
contract. This phase answers that for one high-share role.

## Target

Start with:

```text
blk.0.attn_output
```

Reasons:

- Q4_K, shape `4096 x 4096`;
- decode path uses the native Q4_K coop primitive plus a stage-2 sum;
- imported llama Q4_K MMVQ has an existing comparable trace surface;
- the activation can be captured from the actual block attention path without changing defaults.

## Work

1. Load Qwen3-8B Q4_K_M on AMD with decode primitives enabled.
2. Run enough of block 0 attention to capture the real `attn_output` activation.
3. Warm the role call once outside ATT.
4. Open an ATT interval around `blk.0.attn_output(captured_activation)`.
5. Monkeypatch `HCQProgram.__call__` during that interval to record:
   - program name;
   - global/local launch shape;
   - code hash;
   - runtime object id;
   - role classification;
   - call count.
6. Compare the in-model role interval against prior primitive-surface traces:
   - native tinygrad Q4_K coop;
   - imported llama Q4_K MMVQ.

## Gates

Pass:

- ATT start/stop sync;
- nonzero trace;
- decodable body packets;
- at least one Q4_K role program captured inside the interval;
- joined program identity shows whether the interval is native tinygrad Q4_K coop, fallback, imported MMVQ, or mixed.

Kill:

- if the role interval cannot be body-attributed even though the primitive-surface atlas passed;
- if no HCQ program calls are captured inside the interval;
- if activation capture changes the route or disables decode primitives.

## Decision Table

| Finding | Decision |
|---|---|
| in-model interval launches the same native coop program as the standalone surface | runtime/cache identity is not the gap for this role; look at graph-level scheduling, stage2 reduce, or broader role mix |
| in-model interval launches fallback/dense or different program identity | fund runtime/cache route before scheduler work |
| in-model interval uses imported llama route only when flag is enabled and is still slower/no e2e win | artifact route is not enough; lifecycle/scheduling remains |
| ATT body packets differ materially but program identity matches | fund scheduler/resource project only if Amdahl clears |

## Deliverables

- Probe: `extra/qk_att_inmodel_role_join.py`
- Artifact: `bench/qk-att-inmodel-role-join/result.json`
- Summary: `bench/qk-att-inmodel-role-join/summary.md`
- Result doc: `docs/amd-att-inmodel-role-join-result-20260619.md`

