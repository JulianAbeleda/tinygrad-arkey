# Decode Llama Primitive Transfer Scope

Date: 2026-06-20

## Verdict

`PASS_DECODE_PRIMITIVE_TRANSFER_SCOPED_NATIVE_OBJECT_BLOCKED`

Decode now has the same primitive-first framing we used for prefill, but it does **not** use the same primitive.
Prefill's native target is a dense GEMM schedule. Decode's native target is a small-batch quantized
MMVQ/q8-lifecycle contract:

```text
activation -> q8_1 producer/cache
packed Q4_K/Q6_K weight load
packed int unpack/extract
dp4a/sdot4 dot + q8 sum/min correction
per-group scale application
row/reduction/output policy
```

This pass scopes and freezes that transfer matrix. It changes no runtime route, builds no kernel, and makes no
performance claim.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_primitive_transfer_probe.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_primitive_transfer_result.json
```

## Current Chain

| Layer | State |
|---|---|
| shipped decode stack | promoted default, about `~67%` llama in the banked reconciliation |
| Q4/Q6 tinygrad primitives | present and default-routed by role |
| flash-decode | promoted for long-context decode |
| q8 FFN route | hardened opt-in, `+5.1-6.3%`, default off |
| llama Q4_K MMVQ inner loop | audited: packed nibble extract + `sdot4` + per-group scale |
| llama MMVQ source/object inventory | present in local llama.cpp build, not a Tensile-like standalone `.co` family |
| imported Q4 consumer lifecycle | proven for `attn_output`, `ffn_gate`, `ffn_up` |
| native decode scheduler start gate | still blocked: no `>=30us` timing-grade attributed feature |

## Transfer Matrix

| llama decode primitive / feature | llama evidence | tinygrad today | missing tinygrad primitive surface | gate |
|---|---|---|---|---|
| MMVQ lifecycle selection | `ggml-cuda/mmvq.cu`, `mul_mat_vec_q`, `MMVQ_MAX_BATCH_SIZE=8` | `decode_enabled` Q4/Q6 primitive linears and role policies | one `DecodeMMVQScheduleObject` owning lifecycle, role, shape, quant format, q8 producer, consumer, and reduction policy | normalized role contract table |
| `block_q8_1` activation producer | llama quantize/MMVQ lifecycle | `Q4K_VDOT` experiment, q8 FFN artifact, imported Q4 lifecycle probe | owned activation quant lifecycle with reuse and quality policy | byte-exact q8 producer, reuse count, dNLL/W==D policy |
| Q4_K x Q8_1 packed int dot | `vec_dot_q4_K_q8_1_impl_vmmq`, `VDR_Q4_K_Q8_1_MMVQ=2` | fp-dequant Q4 kernels, coop paths, vdot experiment, imported Q4 consumer | packed q4 extraction, dp4a q8 dot, q8-sum/min correction, per-group scale as one primitive | op-mix structural gate + correctness/W==D |
| Q6_K x Q8_1 packed int dot | `vec_dot_q6_K_q8_1_impl_mmvq`, `VDR_Q6_K_Q8_1_MMVQ=1` | Q6 fp-dequant/coop paths for high-share roles | Q6 packed extract + scale + q8 dot contract | Q6 imported/source coverage |
| small-batch policy | llama MMVQ per-arch tables | T==1 guards, K<=32 fallback, flash ctx policy | batch/context route policy inside the primitive contract | ctx/batch route matrix |
| row/reduction/output contract | llama `mul_mat_vec_q` and fusion hooks | custom partials + separate sums in many paths; q8 artifact fuses a narrow FFN path | direct-output/reduction topology as a first-class choice | W==D movement with quality unchanged |

## What This Means

The prefill work changed the native GEMM path because it proved:

```text
contract -> K-loop -> lowering plan -> structural emission
```

For decode, this pass gets us only to:

```text
primitive transfer matrix -> native object still blocked
```

That is still useful. It prevents the wrong next move: starting BEAM/register search or a generic scheduler rewrite
without a decode primitive contract.

## Why Native Decode Is Still Blocked

The blocking rows are not philosophical; they are concrete:

- no timing-grade `>=30us` attributed q8/native scheduler feature;
- q8 `ffn_gate/up` role-joined body/counter evidence is not closed enough for native codegen authority;
- Q6 imported/source-contract coverage is still open;
- decode lifecycle is split across shipped Q4/Q6 primitives, imported Q4 source path, and q8 artifact policy rather
  than one owned schedule object.

So the answer is not "we know nothing." The answer is:

```text
we know the decode primitive class, but we have not unified it into an owned schedule object with native start gates.
```

## Next Phases

1. **DPT-1 contract normalization**: one table for llama and tinygrad decode roles with shape, quant format,
   lifecycle, launch, resource, timing, and quality labels.
2. **DPT-2 metadata object**: build a `DecodeMMVQScheduleObject` as metadata only: q8 producer, packed weight load,
   dequant/dot, reduction/output, route policy.
3. **DPT-3 structural probe**: validate existing tinygrad Q4/Q6/q8 routes against that object.
4. **DPT-4 source-import track**: continue graph-safe Q4 route and Q6 coverage separately from native renderer work.
5. **DPT-5 native renderer gate**: start native renderer only if readiness finds timing-grade movement or the project
   explicitly accepts broad decode backend work.

## Boundary

No BEAM/search yet. Search becomes useful only after the decode primitive object exists and can be structurally gated,
the same way prefill search only becomes useful after the GEMM schedule object can lower.
