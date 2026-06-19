# q8 FFN codegen/ASM transfer scope (2026-06-19)

This is Track B after `q8-ffn-handwritten-a4-decode-result-20260619.md`.

Track A proved the route:

- dense decode `ffn_gate/up` only, Qwen3-8B Q4_K_M shape `4096 -> 12288`;
- fused RMSNorm + q8_1 side-channel producer;
- fused Q4_K x q8_1 gate/up consumer;
- graph-injected route behind `Q8_FFN_HANDWRITTEN=1`;
- W==D decode speedup `1.051x-1.063x`;
- dNLL `+0.002887`.

The remaining question is ownership: can tinygrad generate or host the primitive without relying on external hipcc/LLD
artifacts?

## Target

Keep the A4 behavior, remove the external compiler artifact dependency.

| gate | target |
|---|---:|
| producer + gate + up isolated lifecycle | `<=129.2us` |
| final W==D decode speedup | `>=3%` sustained |
| dNLL | `<=0.01` |
| default route | unchanged/off |
| in-process HIP runtime | none |

The fast artifact remains the oracle, not the product.

## B0/B1 audit result

Executed:

- probe: `extra/q8_ffn_codegen_transfer_audit.py`;
- artifact: `bench/q8-ffn-codegen-transfer/audit.json`;
- scope: no kernel execution, compile/disassemble/load-contract only.

Objects audited:

| object | source | role |
|---|---|---|
| `fast_producer_hipcc_lld` | hipcc/LLD oracle | fused RMSNorm + q8 producer |
| `fast_gateup_hipcc_lld` | hipcc/LLD oracle | fused Q4_K x q8 gate/up consumer |
| `comgr_producer_raw_c` | tinygrad COMGR raw C | slower producer baseline |
| `comgr_mmvq_raw_c` | tinygrad COMGR raw C | slower one-output consumer baseline |

Key counts:

| object | dot4 | VALU | SALU | barriers | global loads | global stores | loads in `AMDProgram` |
|---|---:|---:|---:|---:|---:|---:|---|
| fast producer | 0 | 653 | 509 | 12 | 19 | 4 | yes |
| fast gate/up | 16 | 120 | 197 | 1 | 11 | 1 | yes |
| COMGR producer | 0 | 879 | 474 | 10 | 19 | 4 | yes |
| COMGR MMVQ | 16 | 167 | 299 | 1 | 12 | 1 | yes |

Launch contract:

| object | kernarg | LDS | private |
|---|---:|---:|---:|
| fast producer | 32 B | 4096 B | 0 |
| fast gate/up | 40 B | 16 B | 0 |
| COMGR producer | 32 B | 1024 B | 0 |
| COMGR MMVQ | 24 B | 16 B | 0 |

Verdict: **B0/B1 PASS as an audit, not as a build.**

The important finding is that the current slow route is not missing dot4. Both consumers emit 16 `v_dot4_i32_iu8`
instructions. The gap is the full primitive contract:

- fused gate/up in one consumer launch;
- 1024-thread producer shape and 4 KB LDS reduction contract;
- clang/hipcc scheduling around the same dot4 math;
- graph-visible q8 side-channel lifetime;
- exact launch dimensions supplied by the runtime-cache route.

So the codegen transfer should not chase a generic "add dot4" patch. That is already present. It should either hand-own
the exact primitive or add the compiler capability that lets tinygrad generate it.

## Track B2 — tinygrad-owned consumer first

Purpose: own the dominant q8 consumer without hipcc/LLD.

Why consumer first:

- it contains the dot4 MMVQ inner loop that creates the decode gain;
- it has a simple 40-byte kernarg contract in the fast fused form;
- it has only 16 B LDS and one barrier;
- it avoids the producer's staged reduction + multi-output-store problem.

Candidate implementations:

1. `Ops.PROGRAM` raw AMD assembler using the existing `assemble_linear` path.
2. A tinygrad AMD DSL kernel that explicitly emits the q8 MMVQ loop and reduction.
3. A minimal source-to-ASM hand port from the oracle disassembly only if the DSL route is too slow.

Gate:

- correctness vs the existing q8 proxy on real GGUF `ffn_gate/up`;
- fused gate/up consumer `<=60us`;
- no HIP runtime;
- no external hipcc/LLD artifact;
- generated/assembled object loads through `AMDProgram`;
- no default route change.

Kill:

- if reproducing the consumer requires a broad HIP-to-ASM importer or LLVM-level scheduler, stop and classify as
  project-level compiler work.

## Track B3 — producer capability

Purpose: generate the fused RMSNorm + q8 side-channel producer natively.

Required primitive shape:

- one decode row, 4096 floats;
- sum-of-squares reduction;
- LDS/barrier broadcast of `rinv`;
- fp normalized output store;
- per-32 q8 max/quantize;
- q8 block store with scale and 32 signed bytes;
- q8 buffer lifetime visible to the gate/up consumer.

The previous Q8L UOp attempt failed because current expression could not cleanly represent a flash-style staged
reduction followed by post-barrier multi-output stores. This audit sharpens that wall: the fast producer is not large,
but it is structurally a lifecycle producer, not a simple elementwise kernel.

Gate:

- standalone producer matches the hipcc/LLD producer;
- producer `<=10us` or combined lifecycle `<=129.2us`;
- one kernel, not staged pack kernels;
- no graph-visible extra q8 pack pass.

Kill:

- if current UOps still cannot express staged reduction + post-barrier multi-output stores, do not keep schedule-searching
  around it. Escalate to a renderer capability.

## Track B4 — graph-native integration

Purpose: replace the artifact runner, not the math.

Tasks:

- create graph-visible tinygrad-owned `PROGRAM` nodes for producer and fused gate/up;
- remove `Q8ArtifactRunner` launch-dimension monkeypatch for this route;
- keep the same buffer order and side-channel lifetime as A4;
- route only dense FFN gate/up under an env flag.

Gate:

- TinyJit/HCQGraph replay passes;
- W==D ctx sweep still clears `>=3%`;
- dNLL still `<=0.01`;
- flag off is byte-for-byte unchanged.

## Track B5 — renderer capability if B2/B3 cannot stay local

Minimal capability set:

- explicit workgroup-local staged reductions;
- barrier-aware local-memory lifetime;
- post-barrier multi-output stores;
- stable launch-contract control for non-rectangular side-channel buffers;
- enough AMD DSL/renderer surface to express `v_dot4_i32_iu8` MMVQ without C/hipcc.

Non-goals:

- no broad HIP importer;
- no general dynamic linker;
- no default route;
- no reopen of q8 quality unless the speed gate passes.

## Decision matrix

| result | decision |
|---|---|
| B2 consumer passes, B3 producer passes | replace artifact route with tinygrad-owned env-flag route and rerun A4 |
| B2 passes, B3 fails | keep consumer as an asset; producer capability is the blocker |
| B2 fails but B3 passes | producer alone is likely sub-gate; do not route unless W==D says otherwise |
| B2/B3 both require broad compiler work | classify as project-level renderer/ASM transfer, not a small kernel primitive |
| B4 passes but speed is <3% | keep the oracle and close the route as not worth maintenance |

## Recommendation

Proceed to **B2a: tinygrad-owned fused gate/up consumer**.

The audit says dot4 exists, so a dot-only patch is not meaningful. The smallest useful build is a standalone fused
gate/up consumer that keeps the fast route's lifecycle contract but replaces the external artifact with tinygrad-owned
ASM/DSL. If that clears `<=60us`, the producer capability becomes worth funding. If it does not, the remaining work is
compiler scheduling/project-level, not primitive search.
