# q8 FFN B2a COMGR fused gate/up result (2026-06-19)

This executes the first ownership probe from `q8-ffn-codegen-asm-transfer-scope-20260619.md`.

Goal: test whether tinygrad-owned COMGR compilation plus the same fused gate/up lifecycle is enough to replace the
external hipcc/LLD fast artifact.

Probe:

- `extra/q8_ffn_comgr_fused_gateup_probe.py`;
- artifact: `bench/q8-ffn-codegen-transfer/comgr_fused_gateup.json`;
- source: raw C compiled by `Device["AMD"].compiler.compile`, not hipcc/LLD;
- shape: Qwen3-8B Q4_K_M `ffn_gate/up`, `4096 -> 12288`;
- launch: `global=(12288,2,1)`, `local=(32,4,1)`;
- default route: unchanged/off.

## Result

**FAIL_PERF.**

| gate | result | verdict |
|---|---:|---|
| gate correctness | max_abs `7.15e-7` | PASS |
| up correctness | max_abs `1.43e-6` | PASS |
| no external hipcc/LLD artifact | yes | PASS |
| fused gate/up consumer | `146.88us` | FAIL (`<=60us` target) |
| producer + fused gate/up lifecycle | `177.72us` | FAIL (`<=129.2us` target) |

This is a performance-only failure, not a math/layout failure.

## Interpretation

The previous B0/B1 audit showed both the fast hipcc/LLD oracle and this COMGR-family path emit the same key inner-loop
class: 16 `v_dot4_i32_iu8` instructions. B2a now shows that adding fusion to the COMGR source still does not transfer
the fast primitive.

Therefore the missing piece is not:

- q8 correctness;
- Q4_K layout;
- native signed dot4;
- fused gate/up as an abstract dataflow;
- HCQ launchability.

The missing piece is lower-level code quality/scheduling around the primitive:

- fewer SALU/control instructions;
- better control-flow shape;
- better wait/scheduling around memory and dot4;
- possibly hand-authored AMD DSL/ASM or renderer support, not C-source variants.

## Decision

Close the **COMGR fused-C sublane**.

Do not keep trying source-level reshuffles of the same raw C kernel unless a specific ISA diff predicts a large change.
The next meaningful B2 build is an explicit tinygrad AMD DSL/ASM consumer, with the fast hipcc/LLD artifact as the
oracle and this COMGR result as the negative control.

## Next

Proceed only if we accept a hand-owned kernel build:

1. Author a standalone `Ops.PROGRAM` / AMD DSL fused gate/up consumer.
2. Keep the exact A4 buffer contract: `dst_gate`, `dst_up`, `gate_words`, `up_words`, `q8`.
3. Gate on real-GGUF correctness and `<=60us` consumer time.
4. If it cannot be kept local, classify the q8 ownership route as project-level compiler/ASM work.
