# Decode MMVQ Schedule Object Structural Scope

Date: 2026-06-20

## Verdict

`PASS_DECODE_MMVQ_SCHEDULE_OBJECT_STRUCTURAL_NATIVE_BLOCKED`

This pass creates the decode equivalent of the prefill schedule-object gate: a first-class, unwired
`DecodeMMVQScheduleObject` plus a probe that instantiates the current decode surfaces against it.

It is structural metadata only. It changes no defaults, lowers no ISA, launches no kernels, and makes no performance
claim.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_mmvq_schedule_object_probe.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_mmvq_schedule_object_result.json
```

## Object Contract

The object encodes the decode primitive stages:

```text
activation_prepare
activation_q8_producer
activation_reuse
packed_weight_load
packed_extract
dot_or_dequant_dot
scale_apply
partial_reduce
output_store
route_policy
```

The four instantiated rows are:

| row | quant | status |
|---|---|---|
| `ffn_gate/up` | Q4_K | imported Q4 consumer proven, graph/default route not promoted |
| `attn_q/o` | Q4_K | promoted default coop route |
| `ffn_down/lm_head` | Q6_K | promoted default for selected roles, imported/source parity still open |
| `ffn_gate/up_q8_artifact` | q8 artifact | hardened opt-in, lossy/default-off |

## Structural Gates

The gate checks:

- small-batch decode policy (`batch <= 8`);
- known quant and activation formats;
- all named MMVQ stages present and ordered;
- packed weight load precedes dot/dequant-dot;
- partial reduction precedes output;
- q8 lifecycle and lossy default policy are explicit;
- llama source contract and tinygrad route presence are labeled;
- performance claim is false.

Passing this gate means the decode primitive is now representable and auditable as one object. It does **not** mean a
native decode renderer is ready.

## Native Renderer Boundary

Native decode remains blocked because:

- no timing-grade `>=30us` attributed native scheduler feature exists;
- q8 `ffn_gate/up` role-joined body/counter evidence is not closed enough for codegen authority;
- Q6 imported/source-contract coverage is still open;
- the current lifecycle is split across default Q4/Q6 primitives, imported Q4 evidence, and a q8 external artifact.

So the decode state is:

```text
primitive transfer matrix: done
schedule object metadata: done
native lowering/emission: not authorized yet
```

## Next

The next non-looping step is role-contract normalization: produce one table with role, shape, quant format, lifecycle,
launch/resource, timing, quality, and ownership labels for llama and tinygrad. After that, choose one of two tracks:

1. source-import graph route for Q4/Q6 coverage; or
2. native renderer only if the attribution gate clears or broad decode backend work is explicitly accepted.
