# llama.cpp kernel residual primitive audit scope

Status: executed; see `llama-kernel-residual-primitive-audit-20260619.md` for the fresh `rocprofv3` redo and current
verdicts. The ledger below is the pre-redo input scope, not the final measured result.

Goal: audit **what llama.cpp itself may still leave on the table** on RX 7900 XTX / Qwen3-8B-Q4_K_M. This is
different from the tinygrad-vs-llama gap audit. The existing docs explain why llama is faster than tinygrad; this
scope asks whether llama's own primitives are near their hardware/algorithmic ceilings, and which unexplored
primitive ideas might still be plausible.

No tinygrad kernels should be built for this scope. This is a measurement/source audit.

## Pre-redo input llama ledger

From `qk-llama-token-primitive-accounting-20260617.md`:

| llama primitive | decode share | current explanation |
|---|---:|---|
| Q4_K/Q6_K MMVQ | ~73.4% | q8_1 activation + signed dot4 + packed QK decode + block affine + MMVQ scheduler, ~626 GB/s / ~70% HBM peak |
| decode attention | ~7.5% | flash_attn_tile + stream_k_fixup + combine |
| q8_1 activation quant | ~3.8% | quantize once per activation, reused by MMVQ |
| RMSNorm | ~5.0% | separate kernels |
| RoPE | ~2.5% | separate kernels |
| elementwise/residual/SwiGLU | ~1.0% | separate/fused small ops |
| graph/launch | amortized | HIP graphs, ~260 kernels/token |

Known fact: llama's MMVQ at ~70% HBM peak is good, but not a proof of optimality. The missing audit is the residual
from ~70% to the practical ceiling for that primitive on this card, plus the Amdahl value of reducing non-MMVQ
shares.

## Scope question

For every major llama primitive, answer:

```text
share of token time
+ achieved hardware efficiency
+ likely limiter
+ plausible unexplored primitive
+ Amdahl ceiling
+ evidence needed to close
```

The audit should distinguish:

- **hardware ceiling:** HBM peak, instruction throughput, occupancy, LDS/register limits;
- **algorithmic ceiling:** required q4/q6 unpack, q8 quant, affine correction, reductions;
- **implementation ceiling:** scheduler, tile shape, fusion boundary, backend portability;
- **Amdahl ceiling:** whether improving the primitive matters to token/s.

## Required deliverable

Create:

`docs/llama-kernel-residual-primitive-audit-20260619.md`

with a table like:

| primitive | share [M/I] | achieved efficiency [M/I] | residual to ceiling | likely limiter | unexplored idea | Amdahl max | verdict |
|---|---:|---:|---:|---|---|---:|---|
| MMVQ Q4_K/Q6_K aggregate | ~73.4% [M] | ~70% HBM [M] | TBD | TBD | TBD | TBD | open until audited |
| q8_1 activation quant | ~3.8% [M] | TBD | TBD | launch/reduction/write? | producer-fused q8? | <=3.8% | TBD |
| decode attention | ~7.5% [M] | TBD | TBD | KV traffic / reduction / launch? | fused/shape-specific flash-decode? | <=7.5% | TBD |
| RMSNorm | ~5.0% [M] | TBD | TBD | memory/reduction/launch? | norm+q8 producer fusion? | <=5.0% | TBD |
| RoPE | ~2.5% [M] | TBD | TBD | memory/elementwise? | fuse with Q/K prep? | <=2.5% | TBD |
| elementwise/SwiGLU/residual | ~1.0% [M] | TBD | TBD | memory/launch? | already low | <=1.0% | likely closed |
| graph/kernel boundaries | amortized [M] | TBD | TBD | graph launch vs GPU work | deeper fusion/persistent decode? | TBD | TBD |
| prefill GEMM/attention | separate phase | TBD | TBD | tile/autotune/LDS/attention | shape-specific tuning | TBD | separate section |

Every number must be tagged:

- `[M]` measured;
- `[I]` inferred;
- `[H]` hypothetical.

## Audit tracks

### LRA-0 — source/version/provenance lock

Goal: make llama results reproducible.

Record:
- llama.cpp commit used;
- build flags (`GGML_HIP_GRAPHS`, hipBLAS/rocBLAS/Tensile use, GPU arch flags);
- ROCm version;
- model;
- prompt/context settings;
- exact command;
- profiler command and raw trace location.

Gate:
- decode tok/s matches the known ~92-98 tok/s band for ctx 512/1024/4096.

Kill if:
- build path differs from the one used in prior accounting and cannot be reconciled.

### LRA-1 — per-primitive time ledger refresh

Goal: refresh the llama token-time breakdown on current source/build.

Measure:
- MMVQ kernels by role if names/arguments allow;
- q8 quant kernels;
- RMSNorm;
- RoPE;
- attention kernels;
- elementwise/residual/SwiGLU;
- graph/launch or unclassified tail.

Output:
- ms/token and share per primitive;
- kernel count per token;
- measured vs prior accounting delta.

Gate:
- unclassified time <=5% or explicitly explained.

### LRA-2 — MMVQ residual-to-peak audit

Goal: explain why llama MMVQ is ~70% HBM peak and not higher.

Questions:
- Is MMVQ truly HBM-bound at the achieved rate, or partly ALU/instruction-bound from q4/q6 unpack and affine?
- Are loads fully coalesced and wide enough?
- What are the VGPR count, occupancy, LDS/shared use, and spills?
- How much time is qsum/min correction vs dot vs packed extraction?
- Do Q4_K and Q6_K roles differ materially?
- Do small rows/large rows/lm_head have different efficiency?

Measurements:
- achieved GB/s per MMVQ role if profiler/kernel args allow;
- instruction mix / ISA excerpts for Q4_K and Q6_K;
- occupancy/VGPR/spill info from compiler/profiler where available;
- compare measured bandwidth to raw copy/READRAW ceiling on same card.

Possible residual explanations:
- required unpack/affine instruction overhead keeps practical ceiling near ~70%;
- memory transaction inefficiency / cache behavior;
- scheduler occupancy or register pressure;
- generic kernel shape not fully tuned for each role;
- q8 activation load/scales/qsum overhead.

Close criterion:
- one of:
  - "MMVQ is near practical ceiling; residual <10% e2e after Amdahl";
  - "specific role has >=5% e2e possible and named limiter";
  - "need hardware counters not available; leave as deferred measurement."

### LRA-3 — q8 activation quant residual

Goal: test whether llama's ~3.8% q8 quant cost has a plausible lifecycle improvement.

Questions:
- Which activations are quantized and how many times per token?
- Is q8 quant fused with any producer today?
- Is cost dominated by reduction, packing, memory writes, or kernel launch?
- Could q8 be emitted by RMSNorm/apply or another producer, and would that preserve reuse?

Do not assume tinygrad's side-channel economics transfer to llama; measure llama's actual producer/consumer graph.

Close criterion:
- if q8 quant remains <=3.8% and any fusion cannot exceed a small Amdahl gain, mark low-EV;
- if producer fusion could remove most of it and share is stable, create a live llama-side lifecycle row.

### LRA-4 — decode attention residual

Goal: decide whether llama decode attention is already below meaningful Amdahl.

Questions:
- Does attention share grow with ctx on the measured range?
- Are `flash_attn_tile`, `stream_k_fixup`, and `combine` individually meaningful?
- Is there a single attention kernel boundary/fusion that could remove fixup/combine?
- Is the path memory-bound on KV traffic or overhead-bound on small reductions?

Close criterion:
- residual <=2-3% e2e, or a specific attention primitive with >=5% long-context upside is named.

### LRA-5 — norm/RoPE/elementwise fusion residual

Goal: decide whether small-op fusion is meaningful inside llama itself.

Questions:
- RMSNorm is ~5% and RoPE ~2.5%; are these memory-bound, launch-bound, or reduction-bound?
- Could RMSNorm also produce q8 and remove q8 quant kernels?
- Could RoPE fuse into Q/K projection or KV write path?
- Are residual/SwiGLU already fused enough that Amdahl is negligible?

Close criterion:
- all small-op ideas are <=3% e2e and closed as low-EV; or one fused producer/consumer primitive is named with
  >=5% e2e or strategic relevance to q8 lifecycle.

### LRA-6 — graph/kernel-boundary audit

Goal: distinguish actual overhead from useful kernel granularity.

Questions:
- Does disabling HIP graphs materially change tok/s?
- Is graph launch overhead visible or fully amortized?
- Are ~260 kernels/token a launch problem, a necessary primitive boundary, or an optimization frontier?
- Would persistent decode or deeper block fusion reduce GPU work, not just launches?

Close criterion:
- graph/boundary overhead is measured and either closed as low-EV or assigned a specific primitive.

### LRA-7 — llama prefill residual audit

Goal: do not assume llama prefill is optimal just because it beats tinygrad.

Measure separately:
- pp512/pp1024/long-prompt throughput;
- GEMM/attention share;
- rocBLAS/Tensile kernel names and achieved TFLOPS/GB/s where available;
- attention share at long prompt.

Questions:
- Is llama prefill GEMM near Tensor Core/WMMA practical peak for the actual shapes?
- Is attention the limiter at long prompts?
- Are there shape-specific tile/autotune misses?

Close criterion:
- prefill residual table exists, or this is explicitly deferred as a separate phase.

## Candidate unexplored llama-side primitive ideas

These are hypotheses to audit, not conclusions:

| idea | why plausible | first kill question |
|---|---|---|
| producer-fused q8 activation | q8 quant is ~3.8%; producer fusion might remove separate quant kernels | is q8 quant cost big enough and producer-compatible enough to matter? |
| role-specialized MMVQ kernels | aggregate MMVQ is ~70%, not peak; some roles may be worse | do per-role MMVQ efficiencies differ by >=5% e2e Amdahl? |
| deeper norm/q8/MMVQ fusion | RMSNorm + q8 quant together are ~8.8% | can reductions/layouts be fused without hurting MMVQ scheduling? |
| attention fixup/combine fusion | attention uses multiple kernels | do fixup/combine have meaningful share? |
| persistent/deeper decode block | could reduce boundaries and reuse activations | does boundary overhead show up after HIP graphs? |
| long-prompt prefill attention tuning | attention often grows with prompt length | does attention dominate at long pp? |
| shape-specific rocBLAS/Tensile/autotune | library kernels are strong but not always perfect | are any prefill shapes below expected TFLOPS by enough Amdahl? |

## Decision rules

- Do not call an idea open unless it has both a plausible primitive and Amdahl room.
- If a primitive has <3% e2e max, mark low-EV unless it composes with another row.
- If a primitive needs source changes in llama.cpp, keep it as "llama-side hypothesis"; do not convert it into a
  tinygrad build task.
- If an idea maps to tinygrad too, create a separate tinygrad row only after the llama-side audit proves the
  primitive and the tinygrad boundary are both relevant.

## Expected outcome

Most likely:
- llama decode MMVQ is not perfect, but near the practical ceiling for a portable GGML/HIP MMVQ primitive;
- q8 quant and small ops have limited Amdahl unless fused with a broader producer lifecycle;
- attention residual is small for decode but may matter for long contexts;
- prefill may have shape-specific headroom, but requires a separate phase.

The deliverable should make that measured rather than assumed.
