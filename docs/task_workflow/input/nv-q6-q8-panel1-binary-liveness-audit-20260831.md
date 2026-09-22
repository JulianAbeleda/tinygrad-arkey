# NV Q6 Q8 panel-1 binary liveness audit

Status: **binary mapping complete; source ordering was not used**.

## Scope and method

This audit reads the frozen cubins with nvdisasm, cuobjdump, and readelf.
Panel entries are classified only by their scalar Q8 global-address stride,
shared-address stride, PC window, and identical LDG-destination/STS-source
register. The deterministic parser asserts 18 load/store pairs per panel, all
logical word indices 0..17, identical pair registers, and no redefinition
before publication.

## Frozen identities

| role | cubin SHA-256 | exact function symbol | ELF symbol extent | resources |
|---|---|---|---|---|
| pinned llama main | 04eb9bcb2edef62c672b5496d743a98c57e3236558b88f2ff117964b7fbb91ca | _Z15dense_mul_mat_qIL9ggml_type14ELi128ELb0EEvPKcPKiPfS5_5uint3iiiiiS6_S6_iiiS6_S6_iiiS6_ | 0x0..0x21c80 | REG255, STACK72, SHARED1024, LOCAL0 |
| admitted candidate main | 6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137 | nv_q6_oracle_broad_cta_prefetch_combined_publish_oracle_publisher_trusted_fp16_packed_ws_segments_in_cta_streamk_s0 | 0x0..0x14100 | REG255, STACK0, SHARED1024, LOCAL0 |

The llama direct K body is 0x0d80..0xeb50; its partial body is
0x12880..0x20780. The candidate loop backedge bounds one physical body at
0x09a0..0x13ad0.

## Exact panel boundaries

| body | panel-0 LDG | panel-0 STS | initial publication BAR | panel-1 LDG | overwrite BAR | panel-1 STS | panel-1 publication BAR | lifecycle BAR |
|---|---|---|---|---|---|---|---|---|
| llama direct | 0x13b0..0x15d0 | 0x3720..0x3830 | 0x3840 | 0x80e0..0x8290 | 0x8620 | 0x86e0..0x8870 | 0x8890 | 0xeb40 |
| llama partial | 0x13110..0x13720 | 0x15350..0x15470 | 0x15480 | 0x19d00..0x19e20 | 0x1a230 | 0x1a2f0..0x1a470 | 0x1a4d0 | 0x20770 |
| candidate | 0x1c80..0x1d90 | 0x2f20..0x3050 | 0x3060 | 0x1e80..0x2280 | 0xa930 | 0xa990..0xaaa0 | 0xaab0 | 0x109b0 |

The candidate loads both Q8 panels before 0x3060. Llama loads panel 1 near
the tail of phase 0, after the initial publication barrier and before the
overwrite barrier.

## Normalized spans and overlap window

Ordinals are PC / 16; span is destination STS ordinal minus source LDG
ordinal.

| body | first LDG ordinal | first STS ordinal | first-to-first span | first-to-last span | per-value span min..max | instructions before overwrite BAR | IMMA / LDSM / scalar LDS in that window |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama direct | 2062 | 2158 | 96 | 121 | 94..103 | 83 | 6 / 0 / 9 |
| llama partial | 6608 | 6703 | 95 | 119 | 94..101 | 82 | 7 / 0 / 8 |
| candidate | 488 | 2713 | 2225 | 2242 | 2178..2225 | 2218 | 123 / 16 / 96 |

## Exact llama direct panel-1 register table

| word | global offset | LDG PC | register | STS PC | shared offset | span | last value use | first static redefinition |
|---:|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 0x0 | 0x80e0 | R49 | 0x86e0 | 0x200 | 96 | 0x86e0 | 0x86f0 MOV |
| 1 | 0x400 | 0x8100 | R51 | 0x8770 | 0x600 | 103 | 0x8770 | 0x92d0 IMMA.16816.S8.S8 |
| 2 | 0x800 | 0x8120 | R52 | 0x8780 | 0xa00 | 102 | 0x8780 | 0x89b0 LDS |
| 3 | 0xc00 | 0x8150 | R53 | 0x8790 | 0xe00 | 100 | 0x8790 | 0x9940 IMMA.16816.S8.S8 |
| 4 | 0x1000 | 0x8160 | R54 | 0x87a0 | 0x1200 | 100 | 0x87a0 | 0x9940 IMMA.16816.S8.S8 |
| 5 | 0x1400 | 0x8170 | R55 | 0x87b0 | 0x1600 | 100 | 0x87b0 | 0x9940 IMMA.16816.S8.S8 |
| 6 | 0x1800 | 0x8180 | R64 | 0x87c0 | 0x1a00 | 100 | 0x87c0 | 0x8f80 IMMA.16816.S8.S8 |
| 7 | 0x1c00 | 0x81b0 | R65 | 0x87d0 | 0x1e00 | 98 | 0x87d0 | 0x8f80 IMMA.16816.S8.S8 |
| 8 | 0x2000 | 0x81e0 | R177 | 0x87e0 | 0x2200 | 96 | 0x87e0 | 0x8a80 MOV |
| 9 | 0x2400 | 0x8210 | R178 | 0x87f0 | 0x2600 | 94 | 0x87f0 | 0x8a60 MOV |
| 10 | 0x2800 | 0x8220 | R179 | 0x8800 | 0x2a00 | 94 | 0x8800 | 0x97c0 FFMA |
| 11 | 0x2c00 | 0x8230 | R47 | 0x8810 | 0x2e00 | 94 | 0x8810 | 0x8910 IMMA.16816.S8.S8 |
| 12 | 0x3000 | 0x8240 | R50 | 0x8820 | 0x3200 | 94 | 0x8820 | 0x92d0 IMMA.16816.S8.S8 |
| 13 | 0x3400 | 0x8250 | R45 | 0x8830 | 0x3600 | 94 | 0x8830 | 0x8910 IMMA.16816.S8.S8 |
| 14 | 0x3800 | 0x8260 | R46 | 0x8840 | 0x3a00 | 94 | 0x8840 | 0x8910 IMMA.16816.S8.S8 |
| 15 | 0x3c00 | 0x8270 | R44 | 0x8850 | 0x3e00 | 94 | 0x8850 | 0x8910 IMMA.16816.S8.S8 |
| 16 | 0x4000 | 0x8280 | R40 | 0x8860 | 0x4200 | 94 | 0x8860 | 0x89c0 IMMA.16816.S8.S8 |
| 17 | 0x4400 | 0x8290 | R42 | 0x8870 | 0x4600 | 94 | 0x8870 | 0x89c0 IMMA.16816.S8.S8 |

## Exact llama partial panel-1 register table

| word | global offset | LDG PC | register | STS PC | shared offset | span | last value use | first static redefinition |
|---:|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 0x0 | 0x19d70 | R51 | 0x1a3a0 | 0x200 | 99 | 0x1a3a0 | 0x1a620 IMMA.16816.S8.S8 |
| 1 | 0x400 | 0x19d00 | R55 | 0x1a2f0 | 0x600 | 95 | 0x1a2f0 | 0x1a310 MOV |
| 2 | 0x800 | 0x19d60 | R59 | 0x1a380 | 0xa00 | 98 | 0x1a380 | 0x1a3b0 MOV |
| 3 | 0xc00 | 0x19d80 | R96 | 0x1a3c0 | 0xe00 | 100 | 0x1a3c0 | 0x1ac60 IMMA.16816.S8.S8 |
| 4 | 0x1000 | 0x19d90 | R97 | 0x1a3e0 | 0x1200 | 101 | 0x1a3e0 | 0x1ac60 IMMA.16816.S8.S8 |
| 5 | 0x1400 | 0x19da0 | R98 | 0x1a3f0 | 0x1600 | 101 | 0x1a3f0 | 0x1ac60 IMMA.16816.S8.S8 |
| 6 | 0x1800 | 0x19db0 | R99 | 0x1a400 | 0x1a00 | 101 | 0x1a400 | 0x1ac60 IMMA.16816.S8.S8 |
| 7 | 0x1c00 | 0x19dc0 | R100 | 0x1a410 | 0x1e00 | 101 | 0x1a410 | 0x1a600 MOV |
| 8 | 0x2000 | 0x19dd0 | R101 | 0x1a420 | 0x2200 | 101 | 0x1a420 | 0x1ae20 IMMA.16816.S8.S8 |
| 9 | 0x2400 | 0x19de0 | R102 | 0x1a430 | 0x2600 | 101 | 0x1a430 | 0x1ae20 IMMA.16816.S8.S8 |
| 10 | 0x2800 | 0x19df0 | R103 | 0x1a440 | 0x2a00 | 101 | 0x1a440 | 0x1ae20 IMMA.16816.S8.S8 |
| 11 | 0x2c00 | 0x19d20 | R46 | 0x1a300 | 0x2e00 | 94 | 0x1a300 | 0x1a390 IMMA.16816.S8.S8 |
| 12 | 0x3000 | 0x19d30 | R47 | 0x1a330 | 0x3200 | 96 | 0x1a330 | 0x1a390 IMMA.16816.S8.S8 |
| 13 | 0x3400 | 0x19d40 | R44 | 0x1a340 | 0x3600 | 96 | 0x1a340 | 0x1a390 IMMA.16816.S8.S8 |
| 14 | 0x3800 | 0x19d50 | R45 | 0x1a360 | 0x3a00 | 97 | 0x1a360 | 0x1a390 IMMA.16816.S8.S8 |
| 15 | 0x3c00 | 0x19e00 | R50 | 0x1a450 | 0x3e00 | 101 | 0x1a450 | 0x1a620 IMMA.16816.S8.S8 |
| 16 | 0x4000 | 0x19e10 | R48 | 0x1a460 | 0x4200 | 101 | 0x1a460 | 0x1a620 IMMA.16816.S8.S8 |
| 17 | 0x4400 | 0x19e20 | R49 | 0x1a470 | 0x4600 | 101 | 0x1a470 | 0x1a620 IMMA.16816.S8.S8 |

## Exact admitted-candidate panel-1 register table

| word | global offset | LDG PC | register | STS PC | shared offset | span | last value use | first static redefinition |
|---:|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 0x4800 | 0x1e80 | R192 | 0xa990 | 0x9800 | 2225 | 0xa990 | 0xb470 LDS |
| 1 | 0x4c00 | 0x1eb0 | R191 | 0xa9a0 | 0x9c00 | 2223 | 0xa9a0 | 0xb360 LDS |
| 2 | 0x5000 | 0x1ee0 | R190 | 0xa9b0 | 0xa000 | 2221 | 0xa9b0 | 0xb700 LDS |
| 3 | 0x5400 | 0x1f10 | R189 | 0xa9c0 | 0xa400 | 2219 | 0xa9c0 | 0xb380 LDS |
| 4 | 0x5800 | 0x1f40 | R188 | 0xa9d0 | 0xa800 | 2217 | 0xa9d0 | 0xb860 I2FP.F32.S32 |
| 5 | 0x5c00 | 0x1f70 | R187 | 0xa9e0 | 0xac00 | 2215 | 0xa9e0 | 0xb300 HADD2.F32 |
| 6 | 0x6000 | 0x1fa0 | R186 | 0xa9f0 | 0xb000 | 2213 | 0xa9f0 | 0xb7d0 I2FP.F32.S32 |
| 7 | 0x6400 | 0x1fd0 | R185 | 0xaa00 | 0xb400 | 2211 | 0xaa00 | 0xbdd0 I2FP.F32.S32 |
| 8 | 0x6800 | 0x2020 | R184 | 0xaa10 | 0xb800 | 2207 | 0xaa10 | 0xb7b0 I2FP.F32.S32 |
| 9 | 0x6c00 | 0x2050 | R183 | 0xaa20 | 0xbc00 | 2205 | 0xaa20 | 0xbde0 I2FP.F32.S32 |
| 10 | 0x7000 | 0x20b0 | R119 | 0xaa30 | 0xc000 | 2200 | 0xaa30 | 0xad10 LDSM.16.M88.2 |
| 11 | 0x7400 | 0x20e0 | R182 | 0xaa40 | 0xc400 | 2198 | 0xaa40 | 0xb6f0 I2FP.F32.S32 |
| 12 | 0x7800 | 0x2110 | R178 | 0xaa50 | 0xc800 | 2196 | 0xaa50 | 0xb6c0 I2FP.F32.S32 |
| 13 | 0x7c00 | 0x2170 | R177 | 0xaa60 | 0xcc00 | 2191 | 0xaa60 | 0xafc0 FMUL |
| 14 | 0x8000 | 0x21b0 | R176 | 0xaa70 | 0xd000 | 2188 | 0xaa70 | 0xb010 FADD |
| 15 | 0x8400 | 0x21f0 | R175 | 0xaa80 | 0xd400 | 2185 | 0xaa80 | 0xafd0 FMUL |
| 16 | 0x8800 | 0x2220 | R174 | 0xaa90 | 0xd800 | 2183 | 0xaa90 | 0xaf40 FADD |
| 17 | 0x8c00 | 0x2280 | R173 | 0xaaa0 | 0xdc00 | 2178 | 0xaaa0 | 0xada0 I2FP.F32.S32 |

## Liveness and dependency classification

**Proven binary facts**

- Every one of the 54 panel-1 LDG values has zero textual register occurrence
  and zero redefinition between its LDG and matching STS.
- The matching STS is therefore the last use and value-death point for every
  global-load register.
- None of those LDG destination registers is a textual source operand of an
  intervening LDSM or IMMA.
- Llama uses scattered registers and starts recycling them immediately after
  publication. In the direct body, R49 is stored at 0x86e0 and redefined
  by MOV at 0x86f0; other published registers become IMMA fragments,
  scalar temporaries, or shared-load destinations.
- The candidate retains a mostly contiguous high bank
  R192..R173 plus R119/R178/R177 across nearly all phase-0 work. Its first
  publication is at 0xa990; first redefinitions do not begin until 0xad10
  and continue through 0xbdd0.
- Llama's loads overlap a short independent tail: 83/82 static instructions
  before the direct/partial overwrite barrier. The candidate holds panel 1
  across 2,218 instructions, including 123 IMMA, 16 LDSM, and 96 scalar LDS.

The LDG registers only transport panel 1 to shared memory. After STS and the
publication barrier, Q8 consumption proceeds through new scalar shared-load
registers. LDSM carries Q6 fragments, not these global-load registers.
Consequently the global-register identity cannot be extended through shared
memory into a later IMMA operand.

**Inference**

The shorter llama lifetime reduces simultaneous register-role pressure and
allows aggressive physical-register reuse. Static SASS alone does not prove
how much LDG latency is hidden or how much runtime improvement comes from that
reuse.

## Current tinygrad controllable boundary

Currently expressible:

- AFTER carries a graph dependency while it survives lowering.
- GROUP bundles effects.
- BARRIER emits actual synchronization and correctly orders shared
  publication/consumption.
- REG placeholders preserve a value until a consumer.

Not currently proven expressible:

- A zero-runtime-instruction, final-machine-order constraint preventing an
  independent LDG from being hoisted before a chosen arithmetic token.
- A two-sided placement window that survives symbolic rewriting, linearization,
  C/PTX rendering, and downstream ptxas.

C-style and PTX renderers alias AFTER to its first source and emit no
instruction; they skip GROUP. The optional list scheduler treats them as
structural only if they reach its input. Gate 7 supplied two different UOp
dependency spellings and different generated-source hashes, but both produced
the identical fc11face14a8df4ff5f193110679d7cbd834567bcb0a0d0aa7fb2411ffe52df8
cubin and the same rejected 1,827-instruction span. The exact pass that
neutralized the edge is not established.

## One minimal backend-neutral primitive and gate

Add strict_after(value, dependency): a non-droppable scheduling edge distinct
from ordinary AFTER.

Semantics:

~~~text
strict_after(value, dependency) == value
final emitted value/load must occur after dependency
no CTA synchronization or memory-value semantics
backend must lower the edge or reject compilation; silent erasure is invalid
~~~

First use a one-LDG/one-STS microtest. Hard gates:

- LDG ordinal is strictly after the lower sentinel and before its STS consumer.
- Constrained and unconstrained controls produce different machine ordering.
- Exactly one intended LDG and STS; no extra global/shared traffic.
- No hardware BAR, stack, LDL, or STL is introduced.
- Output is bit-exact.

Only then retry the full Q6 arm. Hard gates:

- Exactly 18 panel-1 LDG and 18 matching STS.
- First LDG is after the initial combined barrier and before overwrite.
- First-load-to-first-store span is at most 160 instructions.
- IMMA/LDSM/LDS/LDG/STS/STG/BAR =
  256/32/176/109/73/64/4.
- I2FP/FMUL/FADD/FFMA = 1024/1544/1024/0.
- Instructions <=5144, registers <=255, and stack/local/LDL/STL all zero.
- Trusted exactness plus partial/final uint32 identity.
- Locked alternating R31: both main and total paired medians <= -3 us with
  at least 24/31 wins.

## Limitations

- Static SASS does not measure dynamic LDG completion, scoreboard stalls, or
  achieved latency overlap.
- Value identity ends at STS; this audit does not invent an identity across
  shared memory.
- Textual LDSM/IMMA operand identity is exact; undocumented implicit operand
  grouping is not used to create dependencies.
- The location of the Gate-7 edge loss inside the compiler pipeline remains
  unisolated.

## Reproduction

~~~bash
sha256sum \
  docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-mmq-dense.sm_120a.cubin \
  docs/task_workflow/evidence/nv-q6-true-late-q8-panel1-gate7-20260831/artifacts/early_combined_all_partials/early_combined_all_partials.cubin

/usr/local/cuda-13.2/bin/cuobjdump --dump-elf-symbols <cubin>
/usr/local/cuda-13.2/bin/cuobjdump --dump-resource-usage <cubin>
readelf -sW <cubin>
.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm -c <cubin>

PYTHONPATH=. .venv/bin/python \
  docs/task_workflow/evidence/nv-q6-q8-panel1-binary-liveness-audit-20260831/parse_panel_liveness.py
~~~

## Evidence paths

- JSON: docs/task_workflow/evidence/nv-q6-q8-panel1-binary-liveness-audit-20260831/audit.json
- Parser: docs/task_workflow/evidence/nv-q6-q8-panel1-binary-liveness-audit-20260831/parse_panel_liveness.py
- Raw SASS: llama.nvdisasm, candidate.nvdisasm
- Symbols/resources: *.symbols.txt, *.readelf-symbols.txt, *.resources.txt
- Tool identity: nvdisasm-version.txt
