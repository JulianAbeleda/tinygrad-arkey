# Shared-Q8 native-NV local-memory boundary isolation

Date: 2026-08-05
Verdict: **no generic QMD/local-memory or typed-boundary defect found; production route remains closed**

## Question

An earlier default-off model-route experiment produced native-driver Xid 13
`SKEDCHECK05_LOCAL_MEMORY_TOTAL_SIZE`. Is that caused by the new Q4/Q6
shared-Q8 consumer binaries, by `OutputSpec` allocation, or by a generic
native-NV QMD local-memory encoding error?

## Isolated construction

`extra/llm_research/decode/nv_shared_q8_boundary_localmem_probe.py` uses real
Qwen3-8B Q4_K_M payloads, without loading a model route:

- blk.0 `attn_q`, `attn_k`, and `attn_v` packed GGUF payloads;
- the dequantized GGUF token-1 embedding as the activation;
- CPU Q8_1 packing solely to avoid admitting a producer graph; and
- the exact experimental `emit_q4` / `emit_q6` consumer emitters.

Each arm lowers every NV program and records resource/QMD state *before* the
first submit. The `research` arm uses explicit
`execute_research_program(Tensor.empty(...))`; the `promoted` arm uses the
same emitter plus `OutputSpec` through `execute_promoted_program(None, ...)`.
All launches ran in fresh processes under `/tmp/gpu-bench.lock` on the native
RTX 5090.

## Results

The one-consumer Q4 arm passed in both forms, bit-identically. Both compiled
to source/lib SHA `589efc72d84980ce6bf59648a92ddf2c561e9473cdba5af4f07f426c0db22164`,
with 47 registers, 1024 B shared memory, 576 B local memory, and Blackwell QMD
`shader_local_memory_high_size_shifted4=36` (576 B). The output SHA was
`6f4294a313c542b20ece5e49a375fe56ff77d4d44e3e90332583b2db99574882` in both
forms.

The authorized escalation, a single scheduler graph containing real Q4/Q4/Q6
Q/K/V consumers plus a scalar completion sink, also passed in both forms. The
three consumer programs all had 576 B local memory and QMD local-high 36; the
completion reduction had the same local-memory field and 2176 B shared
memory. The complete pre-submit metadata lists were identical between arms;
the final output SHA was
`df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119` in both.

## Causal conclusion

The evidence rules out the proposed generic failure mechanisms:

1. The Q4 and Q6 experimental binaries have a valid, matching local-memory
   allocation and QMD field on this device.
2. `OutputSpec` allocation/promotion does not change binary, resource, QMD,
   or output for this emitter.
3. Three distinct local-memory consumers can coexist in one native scheduler
   graph.

Therefore no runtime/QMD fix is authorized. The earlier Xid needs a distinct
reproducer that retains the prior model graph's lifetime/allocator/scheduling
condition; it cannot be attributed to shared-Q8 or repaired by changing local
memory fields.

## Bounded-hook follow-up

The qualification-only integration is now `tinygrad.llm.shared_q8_attention`.
It has no load policy, environment switch, or default admission: only an
explicit `SharedQ8AttentionAdmission(block_index)` attached to one block can
reach it. The call independently rechecks decode shape `(1,1,4096)`, exact
Q4/Q4/Q6 primitive types, output rows `(4096,1024,1024)`, and each legacy
primitive admission. Any failed check returns `None`, preserving the three
ordinary calls.

Hermetic topology tests establish one provider (`xp`, `xs` UOp identity shared)
and exactly three generated consumers. Lowering has the same resource tuple as
the isolated PASS: Q4 uses 47 registers / 1024 B shared / 576 B local and Q6
uses 66 / 1024 / 576. The kernel names were deliberately kept
`q4k_q8_dp4a_*` / `q6k_q8_dp4a_*`, making the consumer source hashes match the
passing emitters instead of creating a source-only distinction.

On a healthy RTX 5090 (`595.84`, 46 C, idle), a bounded `timeout 120s flock
-w 90 /tmp/gpu-bench.lock` subprocess ran one real-payload blk.0 Q/K/V group
through the new promoted boundary and completed `PASS`: output shapes were
`(1,1,4096)`, `(1,1,1024)`, `(1,1,1024)` and scalar checksums were
`-2.7255945`, `-12.0255909`, `1.1837995`. This is an isolated transport and
resource gate, not a model correctness or wall recovery credit.

A progressive full-logit model qualification may resume only after the
independent P0 cache correctness gate is green, beginning with this one
admitted group rather than the known-faulting full graph.

## Corrected progressive model gate

After P0 gained an identity-correct diagnostic clone, fresh g0/g1 native-NV
children enforced `sampled_id == returned_logits.argmax()` on every one of
eight capture/replay iterations. Both arms were finite and token/argmax
identical. The one-group arm nevertheless failed both admission gates:

- strict full-logit `atol=0.01`: **FAIL**, max absolute error
  `0.0165755749` (mean `0.0017600919`); 816 of 1,215,488 elements exceeded
  0.01, with the maximum at decode row 2 / vocab index 100263;
- topology: **FAIL**, 947 -> 950 calls for one lease.

The ordered call diff accounts for the topology exactly. Baseline projection
ownership is one activation preparation, three consumers, and the Q6 parts
reduction (five calls). Candidate ownership is four Q8-provider calls
(packing/cast, staged absmax reductions, scale/pack), three shared-Q8
consumers, and one downstream adapter (eight calls). Thus the candidate
removes the legacy activation preparation, three GEMVs, and Q6 parts reduction
but replaces them with eight calls, a net `+3`. The exact candidate consumer
census is one each of `q4k_q8_dp4a_4096_4096`,
`q4k_q8_dp4a_1024_4096`, and `q6k_q8_dp4a_1024_4096`.

The progressive sequence therefore stops at g1. No g2/g4/g18 or wall A/B is
admissible. Reopen requires both a provider construction with a net topology /
included-cost win (most directly, one fused Q8 producer rather than four
scheduler programs and no new downstream adapter) and a numerical construction
that passes the predeclared full-logit tolerance.

## One-program llama-CUDA semantic follow-up

The four-program Tensor provider was replaced by one ordinary promoted program,
`q8_1_llama_provider_4096`. Its packed output is 1024 `int8x4` words followed
by 128 fp16 `d|s` metadata words. The provider rounds the normalized activation
to fp16 in-kernel, uses llama CUDA's fp32 `d = amax/127`, reproduces its
warp-shuffle sum association, and stores llama's live CUDA meaning of Q8_1
`s`: the raw fp32 activation sum rounded to fp16. The Q4 consumer does not use
that stored `s`; like llama's live MMVQ implementation it forms the Q4 minimum
correction from the integer Q8 lane sum times `d`.

`extra/llm_research/decode/nv_shared_q8_llama_oracle.py` launches extracted
llama CUDA Q8_1 and Q4_K/Q6_K MMVQ cubins on the same RTX 5090, then launches
the native-NV provider and consumers against the same activation and packed
weights. The result is exact at the provider ABI and effectively exact at the
consumer outputs:

- provider: 0 mismatched bytes out of 4608 (`d=0`, `s=0`, `qs=0`);
- Q4 consumer: max absolute delta `2.6226044e-6`, mean `8.2515180e-7`;
- Q6 consumer: max absolute delta `3.8146973e-6`, mean `8.8429078e-7`.

This closes the earlier ambiguity between the model-file CPU Q8_1 convention
and the live CUDA decode convention. The one-program route now implements the
latter exactly; remaining full-model differences from the ordinary tinygrad
path are the expected consequence of introducing llama-style activation
quantization, not an unidentified Q8 packing error.

Fresh post-identity-contract g0/g1 children then produced:

| arm | calls | shared consumers | finite | tokens / argmax |
|---|---:|---:|---|---|
| g0 | 947 | 0 | yes | reference |
| g1 | 948 | 3 | yes | exact match |

The g1 full-logit delta improved to max `0.01363945`, mean `0.0019740802`,
with 589 elements above the predeclared `0.01` tolerance. All eight greedy
tokens and full-logit argmaxes remained equal; the smallest candidate top-1
margin was `0.13689995`. This is useful stability evidence, but it does not
override the predeclared numerical gate: g1 is still **FAIL**.

### Exact topology cause

Compiled-source inspection disproved the remaining-"adapter" interpretation.
The candidate's `E_8_8_16_2_34c2...` is not a copy. It is the required fused
K-RoPE plus K/V cache-store kernel and reads a final 1024-element V projection.
The baseline's `r_8_8_16_2_4_c6bed...` is the same cache-store operation with
the legacy Q6 kernel's four V partials reduced in the store kernel. Therefore
the first-block ownership is:

- baseline: Q4-Q + Q4-K + Q6-partial + K/V-store-with-V-reduce = **4 calls**;
- candidate: Q8 provider + Q4-Q + Q4-K + Q6-final + K/V-store = **5 calls**.

The input-side norm epilogue remains one call in either arm. It writes fp16 in
the baseline because the legacy projection preludes request fp16; it writes
fp32 in the candidate because the exact llama rounding is performed inside the
provider. The promised precompiled-output identity contract removes no member
of the five-call candidate set because none is a redundant materialization.

The resulting structural verdict is stronger than the earlier progressive
stop: a standalone shared-Q8 provider plus three standalone consumers cannot
meet the current `g1 <= g0` topology prerequisite. An admissible reopen must
change the operation partition, for example by fusing the norm epilogue into
the Q8 provider or fusing the Q6 consumer into the K/V store. Relaxing the
topology gate to time the five-call construction would be a separate scope
decision; it is not authorized by this record. No g2/g4/g18 or wall run was
performed.

## Scoped structural successor: attention RMSNorm -> Q8

The ranked successor is to replace the attention RMSNorm output epilogue with
the Q8 provider, rather than attempt to delete the mandatory cache-store. The
construction has a narrow ownership seam:

1. Only an explicit one-block `SharedQ8AttentionAdmission` may additionally
   attach the existing decode-only `REDUCE_OUTPUT` semantic marker to that
   block's attention norm. The global reduce-output policy remains closed.
2. The shared-Q8 boundary may consume only a visible `REDUCE_OUTPUT` marker and
   its declared `(fallback, x, weight)` sources. It may not rediscover or
   pattern-match generic RMSNorm arithmetic. A miss keeps the ordinary marker
   fallback.
3. The leased block owns a load-time materialized fp16 norm weight. This avoids
   turning the GGUF weight view into hidden per-token preparation work.
4. One 16-warp workgroup first computes RMS sumsq/rsqrt, then assigns eight
   consecutive Q8 groups to each warp. Each activation remains on one lane so
   llama CUDA's 32-lane max and sum association is preserved. The affine RMSNorm
   result is rounded to fp16 at the ordinary projection-prelude cast point
   before Q8 quantization.

This successor is not yet qualified. Its mandatory order is:

- hermetic byte comparison against ordinary RMSNorm -> fp16 -> the already
  byte-exact standalone llama-Q8 provider;
- hermetic included-cost census proving fused one-program ownership and no
  weight/input adapter;
- only if those pass, fresh g0/g1 full-logit qualification at the unchanged
  `atol=0.01` and `g1 <= g0` gates;
- only if g1 passes, one-block wall A/B/A, then progressive g2/g4/g18.

No tolerance relaxation, all-layer admission, or production policy change is
part of this successor. Q6-to-KV-store fusion remains secondary: the baseline
already absorbs its Q6 partial reduction into that store, so it is less likely
to improve the operation ledger.
