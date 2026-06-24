# q8 FFN AMD scheduler S0 result (2026-06-19)

S0 is the terminal audit for the native tinygrad q8 decode ownership route.

Context:

- A4 proved the q8 route in-model as a research artifact:
  - W==D decode `1.051-1.063x`;
  - dNLL `+0.002887`;
  - default off.
- B2b proved a tinygrad-owned AMD DSL/ASM fused `ffn_gate/up` consumer is correct on real GGUF:
  - gate max_abs `9.54e-7`;
  - up max_abs `1.43e-6`.
- B2b failed the speed gate:
  - tinygrad AMD DSL/ASM fused gate/up `166.649us`;
  - target `<=60us`;
  - COMGR fused-C was also slow at `146.88us`.

S0 asked whether that miss has a clear bounded cause in the generated code object.

## Probe

Executed:

- probe: `extra/q8_ffn_asm_schedule_audit.py`;
- artifact: `bench/q8-ffn-codegen-transfer/asm_schedule_audit.json`.

The probe builds the passing AMD DSL/ASM full-row consumer, extracts its instruction stream before `assemble_linear`
packing, and compares it against:

- the hipcc/LLD fast `q8_mmvq_gateup` artifact;
- the COMGR fused-C `q8_mmvq_gateup` object.

Important tooling note: `assemble_linear` emits a minimal ELF accepted by tinygrad `AMDProgram`, but rejected by
LLVM `objdump/readelf`. For the tinygrad ASM object, S0 counts the AMD instruction objects directly. For hipcc/LLD and
COMGR, S0 uses the existing LLVM-disassembly parser.

## Result

| object | median gate/up | static instructions | dot4 | global loads | DS | waitcnt |
|---|---:|---:|---:|---:|---:|---:|
| tinygrad AMD DSL/ASM | `166.649us` | `218` | `16` | `22` | `10` | `17` |
| hipcc/LLD oracle | `<=60us` target | `336` | `16` | `11` | `7` | `20` |
| COMGR fused-C | `146.88us` | `482` | `16` | `12` | `2` | `9` |

Top grouped deltas, tinygrad ASM minus hipcc/LLD:

| group | delta |
|---|---:|
| SALU | `-169` |
| VALU | `+37` |
| global_load | `+11` |
| branch | `-5` |
| FMA | `-5` |
| DS | `+3` |
| waitcnt | `-3` |
| convert | `+1` |

Top tinygrad ASM mnemonics:

| mnemonic | count |
|---|---:|
| `v_and_b32_e32` | `30` |
| `v_add_nc_u32_e32` | `28` |
| `s_waitcnt` | `17` |
| `global_load_b32` | `16` |
| `v_dot4_i32_iu8` | `16` |
| `v_lshrrev_b32_e32` | `14` |
| `v_lshlrev_b32_e32` | `11` |
| `v_cndmask_b32_e32` | `10` |

## Interpretation

This is not a missing-instruction problem.

Both the fast oracle and tinygrad ASM emit the same `16` native signed dot4 operations. The tinygrad ASM object also has
fewer static instructions than the hipcc/LLD oracle. The slow path is not failing because it lacks `v_dot4` or because
the hand-owned primitive expanded into a huge instruction stream.

The concrete visible deltas are:

1. tinygrad ASM has twice as many global load instructions (`22` vs `11`);
2. tinygrad ASM has more address/bit-manipulation VALU (`+37` grouped VALU, especially `v_add_nc`, `v_and`, shifts);
3. tinygrad ASM has slightly more LDS/reduction traffic (`+3` DS), but the reduction shape is not a large standalone
   delta;
4. static instruction count does not explain a `166.649us` vs `<=60us` miss.

The likely gap is instruction scheduling and memory-latency hiding: hipcc/LLD emits a wider/coalesced load shape,
compiler-managed dependency ordering, and target-specific scheduling annotations. The tinygrad AMD DSL consumer is
correct but does not own enough of that scheduler behavior to approach the oracle.

## Verdict

**S0_CLOSE_PROJECT_LEVEL_SCHEDULER.**

Per the scope's kill rule, stop the native q8 decode ownership route here. S0 did not reveal a bounded primitive edit
with a credible path from `166.649us` to `<=60us`. Continuing into S1-S4 would be blind local tuning unless the project
first commits to broader AMD scheduler/codegen work or PMU-level observability.

What is closed:

- q8 primitive validity: proven;
- q8 research artifact route: proven and useful;
- tinygrad AMD DSL/ASM consumer correctness: proven;
- bounded native tinygrad q8 decode ownership: closed at scheduler/codegen quality.

What remains open only as project-level work:

- vector/coalesced load selection for this consumer;
- latency-aware load/wait/dot scheduling;
- AMD descriptor/local-id ergonomics if needed by a future scheduler;
- a renderer/codegen path that can emit hipcc/Tensile-class schedules without external artifacts.

## Consequence

For decode, the honest state is:

- use the A4 artifact route only as a research flag;
- do not promote q8 to default;
- do not fund producer ownership until the consumer scheduler wall is solved;
- treat the remaining native route as compiler roadmap, not machine-search primitive work.
