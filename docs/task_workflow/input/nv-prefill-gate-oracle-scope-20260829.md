# NV pp512 gate/up critical-path oracle scope

## Decision

Falsify or qualify investment in a llama-style packed gate/up primitive by measuring the maximum wall time recoverable from exactly 72 gate/up main programs on the current best composed route.

## Frozen route

- Qwen3-8B Q4_K_M, pp512, token seed `20260617`.
- Compiler gate/up, K, Q/O, serialized Q4 V, and graph-owned Q6 V.
- `HCQ_NUM_COMPUTE=2`, ready placement `0`, and the qualified combined cut policy.
- Control/oracle/control, nine timed samples per arm, three warmups, and 20 replay cycles.

## Intervention

- Preserve all 72 ordinary Q8 activation producers.
- Replace only the 72 gate/up main programs.
- Preserve the `(out, record, packed_weight)` buffer ABI, dependency edges, graph-owned allocations, launch positions, and canonical packed weight arguments.
- The replacement writes deterministic zeros to the complete FP32 output and performs no packed matmul work.
- Correct model logits are intentionally out of scope; stable execution, exact replay, and structural isolation are required.

## Structural PASS

- Exactly 72 `nv_gate_oracle_zero` calls and zero compiler gate/up mains.
- Exact unchanged populations: K 36, Q/O 72, Q4 V 18, Q6 V 18.
- Exactly 216 main calls and 216 Q8 producers in the admitted route.
- Exactly 216 canonical, unique packed weight arguments.
- Zero weight-copy kernels and zero old fixups.
- Exact graph-owned record/output census and exact 20-cycle replay.

## Wall decision

- `PASS`: oracle median saves at least 8 ms against both controls. Build the packed stream-K gate/up primitive.
- `WEAK`: saves 3-8 ms against both controls. Build only a representative single-projection prototype.
- `STOP`: saves less than 3 ms against either control. Do not invest; the regional service is not sufficiently exposed on the model critical path.

The oracle is diagnostic-only and default-off. It must never be selected by the model loader or a production policy.

## Follow-on primitive gates

Passing this oracle qualifies only the single-projection gate/up kernel body. It does not qualify the remaining proposed primitives.

### FFN down

- Replace exactly 36 down-projection mains at the post-gate activation boundary while preserving activation, weight, output, and residual dependency positions.
- Test Q4_K and Q6_K populations separately before a combined arm.
- Require finite stable execution, unchanged non-down populations, and control/oracle/control R9.
- `PASS` at 6 ms or more recoverable median wall, `WEAK` at 2-6 ms, otherwise `STOP`.

### Gate/up Q8 activation reuse

- First measure the exact service of the 72 gate/up Q8 producers in the finalized current-best graph.
- Then alias one record per layer into the paired gate/up consumers, leaving exactly 36 producers and 72 mains.
- Require bit-identical Q8 records for the two consumers, exact full logits, and no extra copies or materialization.
- `PASS` only if full-model median improves by at least 0.5 ms against both controls.

### Role epilogues

- Inventory finalized graph calls for SiLU/multiply, down residual, Q/O conversion/reshape, and K/V cache-layout work.
- A role with no standalone critical-path call is already fused and is `STOP` without an oracle.
- For each remaining population, bypass only that population with an ABI-compatible graph-owned provider.
- Require at least 0.5 ms recoverable median wall per role, or 1 ms for a deliberately grouped epilogue substrate.

### Output precision

- FP16 and FP32 output modes are correctness/performance variants of a passing packed matmul, not independent levers.
- Require full-logit `rtol=0.02`, `atol=0.5`, identical token, finite replay, and a wall win against both FP32 controls before retaining FP16 output.
