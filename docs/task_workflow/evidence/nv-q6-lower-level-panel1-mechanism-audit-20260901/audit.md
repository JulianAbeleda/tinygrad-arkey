# Q6 Q8-panel1 lower-level ordering-mechanism audit

Date: 2026-09-01  
Mode: read-only binary/source audit; no compile, GPU run, implementation edit, test edit, or commit  
Decision: hold real Q6 until the isolated qualifier x launch/architecture synthetic is released

## Bottom line

The pinned llama kernel does not use inline PTX, a CUDA load intrinsic, a volatile access, an explicit cache operator, a register constraint, or a separate/noinline copy function for Q8 panel1. Its copy is ordinary CUDA C++ in a `static __device__ __forceinline__` template. The Q8 input is declared `const int *__restrict__`; the instantiated kernel is `__launch_bounds__(256, 1)`; and the available local build metadata uses nvcc/ptxas 13.2.86, `sm_120a`, `-O3`, and `-use_fast_math`.

In final SASS, llama hoists all 18 Q8 loads above the first source copy barrier, emits them as `LDG.E.CONSTANT`, keeps the 18 destination registers unused across the barrier, and consumes each exactly once with an `STS`. Direct and partial paths have first-load-to-first-store spans of 96 and 95 normalized instructions. Gate12 uses ordinary direct C dereferences too, but its Q8 parameter is mutable and unqualified. Its 18 `LDG.E` instructions remain after the first barrier, share a base address held live from PC `0x1a50`, and have a 251-instruction first-load-to-first-store span; the whole symbol also has a 32-byte frame and 8 `LDL`/8 `STL` spill operations.

This proves a source/binary correlation, not the causal compiler rule. No PTX or NVVM payload survived in either cubin, and the available llama build trees do not identify the exact command that produced the extracted cubin. The smallest evidence-grounded next mechanism is therefore **none**: isolate `const __restrict__` from the `__launch_bounds__(256,1)`/`sm_120a` package in a synthetic compiler A/B. Inline PTX is only the first fallback experiment if that matrix fails; it is not the mechanism used by llama.

## Frozen inputs

| Role | Path | SHA-256 / identity |
|---|---|---|
| llama source | `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh` at commit `ac4cddeb0dbd778f650bf568f6f08344a06abe3a` | `6d153a9d6f293a4ff5f11e7886a48bf765b21d74075d73b2097a2b2a9149de6f` |
| llama cubin | `/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-mmq-dense.sm_120a.cubin` | `04eb9bcb2edef62c672b5496d743a98c57e3236558b88f2ff117964b7fbb91ca` |
| llama normalized disassembly | `/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-q6-q8-panel1-binary-liveness-audit-20260831/llama.nvdisasm` | `c97e22ee7b3fa81c7322eb52f07111f6c7aef3b7b1391c4f559ccf675e9a4802` |
| Gate12 source | `/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-q6-region-copy-panel1-gate12-20260831/fresh-artifacts/candidate/candidate.cu` | `206ebe0ea6214fccfa6c389c19e6b4e6f1d9e0fcc38557495552710555e90017` |
| Gate12 cubin | `/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-q6-region-copy-panel1-gate12-20260831/fresh-artifacts/candidate/candidate.cubin` | `dbecf56c280a10016ea73c4406d68dcb094c22691dc6e235b216c2e831347a24` |
| admitted main anchor | frozen external admission identity | `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137` |
| admitted all-partials fixup | frozen external admission identity | `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514` |

Pinned llama entry symbol:

```text
_Z15dense_mul_mat_qIL9ggml_type14ELi128ELb0EEvPKcPKiPfS5_5uint3iiiiiS6_S6_iiiS6_S6_iiiS6_
```

Gate12 entry symbol:

```text
nv_q6_oracle_broad_cta_prefetch_region_copy_q8_panel1_combined_publish_oracle_publisher_trusted_fp16_packed_ws_segments_in_cta_streamk_s0
```

## Source and function-boundary evidence

### llama

- `mul_mat_q_process_tile` is `static __device__ __forceinline__` at `mmq.cuh:3447`; its parameters include `const char *__restrict__ x` and `const int *__restrict__ y` at `mmq.cuh:3448`.
- The copy at `mmq.cuh:3501-3513` is an ordinary source sequence: `__syncthreads();`, calculate `const int *by0`, unrolled `tile_y[l] = by0[l];`, then `__syncthreads();`.
- There is no copy-local inline assembly, CUDA load intrinsic, `volatile`, explicit cache operator, named register constraint, or source data/control dependency on the phase-0 arithmetic. The two `__syncthreads()` calls are the only explicit ordering operations around the copy.
- Direct (`mmq.cuh:3710-3713`) and partial/fixup (`mmq.cuh:3779-3782`) call the same force-inlined template. They are control paths inside one instantiated global kernel, not separate copy functions.
- The instantiated kernel is `static __global__ __launch_bounds__(warp_size*nwarps, 1)`; here that is `__launch_bounds__(256,1)`.
- The cubin symbol table contains the entry and `$__internal_0_$__cuda_sm20_div_u64`, but no `mul_mat_q_process_tile` or copy-body symbol. There is no separate/noinline copy boundary in the binary.

### Gate12

- The entry is `extern "C" __global__ void __launch_bounds__(256)` and all three pointer arguments are mutable and lack `const`/`restrict`.
- The panel copy at `candidate.cu:1231-1248` is 18 direct expressions of the form `*(buf0+dst)=*(data2+index)`, covering source element offsets 4608 through 8960 in steps of 256. It is between the existing source barriers at lines 1229 and 1250.
- There are zero named panel-load temporaries, no copy-local inline PTX, no volatile/cache operator, and no added synchronization. Inline PTX helpers elsewhere in the source implement `ldmatrix`/MMA only and prove that the CUDA renderer can emit operand constraints; they do not constrain this copy.
- The cubin has only the entry function symbol. Therefore a separate function/template boundary is not the differentiator.

## Intermediate and toolchain availability

| Fact | llama | Gate12 |
|---|---|---|
| file form | final ELF64 CUDA executable cubin | final ELF64 CUDA executable cubin |
| ELF flags | `0x6007802` | `0x6007802` |
| preserved `.ptx` section | none | none |
| preserved `.nvvm` section | none | none |
| preserved fatbin section | none | none |
| `cuobjdump --dump-ptx` | empty | empty |
| `.nv.info` language tag | `PTX` | `PTX` |
| embedded ptxas | 13.2.86, build `cuda_13.2.r13.2/compiler.37953736_0` | same |
| embedded target string | `-arch sm_120a -m 64` | `-arch sm_120` |
| producer front end | exact producing command unavailable; local tree has nvcc commands | NVRTC direct-cubin path in tinygrad `NVRTCCompiler` |

The `.nv.info` `Language: PTX` tag records the input language seen by ptxas; it does not preserve the PTX program. `.nv.merc.*` sections are final-binary Mercury metadata, not NVVM IR or recoverable PTX.

Available local llama commands compile `mmq.cu` with nvcc 13.2.86, `-O3 -DNDEBUG -std=c++17`, `--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]`, `-use_fast_math`, `-extended-lambda`, and `-compress-mode=size`; one local tree adds `-rdc=true`. No extraction manifest proves which local build produced the pinned cubin, so only the embedded ptxas/version/architecture strings are promoted to binary facts.

Gate12 uses tinygrad's NVRTC direct-cubin path with `--gpu-architecture=sm_120`, the CUDA include path, and `--minimal` for NVRTC >= 12.4. Its source lacks `TINYGRAD_NV_USE_FAST_MATH`, so that path does not add `--use_fast_math`. No available command or source supplies `--maxrregcount` for either artifact.

## ELF attributes and resources

| Attribute | llama | Gate12 |
|---|---:|---:|
| `.text` bytes | `0x21c80` | `0x14300` |
| static `.nv.shared` bytes | 1024 | 1024 |
| register count | 255 | 255 |
| `EIATTR_MAXREG_COUNT` | 255 | 255 |
| frame/min-stack bytes | 72 / 72 | 32 / 32 |
| reported local bytes | 0 | 0 |
| decoded max threads | `(256,1,1)` | `(256,1,1)` |
| decoded barriers | 1 | 1 |
| whole-symbol `LDL`/`STL` | not attributed to the copy window here | 8 / 8 in Gate12 evidence |

`EIATTR_MAXREG_COUNT=255` is present in both final binaries and is not evidence of a `--maxrregcount` command. `cuobjdump` exposes the same `MAX_THREADS` tuple for both but does not expose a distinct decoded minimum-block-count attribute. Thus llama's second launch-bounds argument is source-proven, not distinguishable in this decoded ELF view.

## Exact SASS transport liveness

PCs are hexadecimal instruction addresses. Normalized spans are instruction ordinals, not byte distances. Every row below has zero occurrences of the transport register between its `LDG` definition and its `STS` use; the `STS` is its first and only intervening consumer. `first reuse` is the first post-store definition/use identified by the deterministic liveness parser.

### llama direct

Address base is defined immediately before the loads by `LEA R4,P0,R64,R248,0x2` at `0x80b0` and `LEA.HI.X R5,R64,R247,R65,0x2,P0` at `0x80c0`. All 18 loads are `LDG.E.CONSTANT` through descriptor `UR8` using `R4.64`. `R4` is reused by `IMMA` at `0x8410`; `R5` is reused by `IMAD` at `0x8490`, both after the last load. The source's first/overwrite barrier is at `0x8620`, the stores are after it, and the publication barrier is at `0x8890`. There are 6 tail `IMMA` and 9 tail `LDS` instructions between the load group and overwrite barrier. First-load-to-first-store span is 96; per-value spans are 94-103.

| i | LDG PC | dst | STS PC | first reuse PC | opcode |
|---:|---:|---:|---:|---:|---|
| 0 | `0x80e0` | R49 | `0x86e0` | `0x86f0` | MOV |
| 1 | `0x8100` | R51 | `0x8770` | `0x92d0` | IMMA |
| 2 | `0x8120` | R52 | `0x8780` | `0x89b0` | LDS |
| 3 | `0x8150` | R53 | `0x8790` | `0x9940` | IMMA |
| 4 | `0x8160` | R54 | `0x87a0` | `0x9940` | IMMA |
| 5 | `0x8170` | R55 | `0x87b0` | `0x9940` | IMMA |
| 6 | `0x8180` | R64 | `0x87c0` | `0x8f80` | IMMA |
| 7 | `0x81b0` | R65 | `0x87d0` | `0x8f80` | IMMA |
| 8 | `0x81e0` | R177 | `0x87e0` | `0x8a80` | MOV |
| 9 | `0x8210` | R178 | `0x87f0` | `0x8a60` | MOV |
| 10 | `0x8220` | R179 | `0x8800` | `0x97c0` | FFMA |
| 11 | `0x8230` | R47 | `0x8810` | `0x8910` | IMMA |
| 12 | `0x8240` | R50 | `0x8820` | `0x92d0` | IMMA |
| 13 | `0x8250` | R45 | `0x8830` | `0x8910` | IMMA |
| 14 | `0x8260` | R46 | `0x8840` | `0x8910` | IMMA |
| 15 | `0x8270` | R44 | `0x8850` | `0x8910` | IMMA |
| 16 | `0x8280` | R40 | `0x8860` | `0x89c0` | IMMA |
| 17 | `0x8290` | R42 | `0x8870` | `0x89c0` | IMMA |

### llama partial/fixup

Address base is defined by `LEA R4,...` at `0x19cd0` and `LEA.HI.X R5,...` at `0x19ce0`; all 18 loads are `LDG.E.CONSTANT` through `UR8` using `R4.64`. `R4` is first overwritten by `IMMA` at `0x19fe0`, after all loads. The first/overwrite barrier is `0x1a230`, stores follow it, and publication barrier is `0x1a4d0`. There are 7 tail `IMMA` and 8 tail `LDS` instructions between the group and overwrite barrier. First-load-to-first-store span is 95; per-value spans are 94-101.

| i | LDG PC | dst | STS PC | first reuse PC | opcode |
|---:|---:|---:|---:|---:|---|
| 0 | `0x19d70` | R51 | `0x1a3a0` | `0x1a620` | IMMA |
| 1 | `0x19d00` | R55 | `0x1a2f0` | `0x1a310` | MOV |
| 2 | `0x19d60` | R59 | `0x1a380` | `0x1a3b0` | MOV |
| 3 | `0x19d80` | R96 | `0x1a3c0` | `0x1ac60` | IMMA |
| 4 | `0x19d90` | R97 | `0x1a3e0` | `0x1ac60` | IMMA |
| 5 | `0x19da0` | R98 | `0x1a3f0` | `0x1ac60` | IMMA |
| 6 | `0x19db0` | R99 | `0x1a400` | `0x1ac60` | IMMA |
| 7 | `0x19dc0` | R100 | `0x1a410` | `0x1a600` | MOV |
| 8 | `0x19dd0` | R101 | `0x1a420` | `0x1ae20` | IMMA |
| 9 | `0x19de0` | R102 | `0x1a430` | `0x1ae20` | IMMA |
| 10 | `0x19df0` | R103 | `0x1a440` | `0x1ae20` | IMMA |
| 11 | `0x19d20` | R46 | `0x1a300` | `0x1a390` | IMMA |
| 12 | `0x19d30` | R47 | `0x1a330` | `0x1a390` | IMMA |
| 13 | `0x19d40` | R44 | `0x1a340` | `0x1a390` | IMMA |
| 14 | `0x19d50` | R45 | `0x1a360` | `0x1a390` | IMMA |
| 15 | `0x19e00` | R50 | `0x1a450` | `0x1a620` | IMMA |
| 16 | `0x19e10` | R48 | `0x1a460` | `0x1a620` | IMMA |
| 17 | `0x19e20` | R49 | `0x1a470` | `0x1a620` | IMMA |

### Gate12 candidate

The Q8 kernel parameter is loaded by `LDC.64 R50,c[0][0x390]` at `0x1830`; `IMAD.WIDE R50,R238,0x4,R50` at `0x1a50` forms the panel base. That `R50.64` base remains live until all 18 panel1 loads at `0x9830-0x9940`, with byte offsets `0x4800-0x8c00` in steps of `0x400`, through descriptor `UR14`. The first component is reused by `FMUL` at `0x9aa0` and the high component by `FMUL` at `0x9c50`. Unlike llama, all loads are ordinary `LDG.E` and occur after the first/overwrite barrier at `0x9820`. Publication barrier is `0xa930`. First-load-to-first-store span is 251.

| i | address offset | LDG PC | dst | STS PC | first reuse PC | opcode |
|---:|---:|---:|---:|---:|---:|---|
| 0 | `0x4800` | `0x9830` | R126 | `0xa7e0` | `0xb080` | LDSM |
| 1 | `0x4c00` | `0x9840` | R127 | `0xa800` | `0x12780` | I2FP |
| 2 | `0x5000` | `0x9850` | R128 | `0xa820` | `0xb0b0` | LDSM |
| 3 | `0x5400` | `0x9860` | R129 | `0xa830` | `0x124d0` | MOV |
| 4 | `0x5800` | `0x9870` | R130 | `0xa840` | `0xb170` | LDSM |
| 5 | `0x5c00` | `0x9880` | R131 | `0xa850` | `0x12ca0` | I2FP |
| 6 | `0x6000` | `0x9890` | R141 | `0xa860` | `0x10590` | FADD |
| 7 | `0x6400` | `0x98a0` | R142 | `0xa870` | `0x105a0` | FADD |
| 8 | `0x6800` | `0x98b0` | R145 | `0xa880` | `0x12690` | MOV |
| 9 | `0x6c00` | `0x98c0` | R146 | `0xa890` | `0xb0d0` | LDS.64 |
| 10 | `0x7000` | `0x98d0` | R147 | `0xa8a0` | `0xc8d0` | HADD2 |
| 11 | `0x7400` | `0x98e0` | R148 | `0xa8b0` | `0xbbd0` | HADD2 |
| 12 | `0x7800` | `0x98f0` | R149 | `0xa8c0` | `0xc230` | HADD2 |
| 13 | `0x7c00` | `0x9900` | R150 | `0xa8d0` | `0xbb80` | HADD2 |
| 14 | `0x8000` | `0x9910` | R117 | `0xa8e0` | `0x10ec0` | I2FP |
| 15 | `0x8400` | `0x9920` | R118 | `0xa8f0` | `0x111c0` | I2FP |
| 16 | `0x8800` | `0x9930` | R119 | `0xa900` | `0x11340` | I2FP |
| 17 | `0x8c00` | `0x9940` | R124 | `0xa910` | `0xb050` | LDSM |

## Proven, inferred, and unknown

### Proven

- llama's source copy uses ordinary CUDA C++, a `const __restrict__` Q8 pointer, a force-inlined template, and `__launch_bounds__(256,1)`.
- Gate12's copy uses ordinary CUDA C++ direct assignments, a mutable unqualified Q8 pointer, and `__launch_bounds__(256)`.
- llama emits 18 pre-barrier `LDG.E.CONSTANT`; Gate12 emits 18 post-barrier `LDG.E`.
- Neither transport chain has an explicit value/control dependency on the surrounding MMA/FADD computation. In each binary, transport registers have zero intervening uses before their `STS`.
- Both kernels use 255 registers. The extracted llama has a 72-byte frame; Gate12 has a 32-byte frame and 8 `LDL`/8 `STL` operations. The copy transport registers themselves are not directly spilled in the listed windows.
- Neither final cubin preserves PTX/NVVM or contains a separate copy symbol.

### Inferred, requiring the released synthetic to prove or reject

- `const __restrict__` plus the read-only `LDG.E.CONSTANT` path may give the compiler enough non-alias/immutability information to hoist Q8 loads across a shared-memory barrier while preserving stores after it.
- `__launch_bounds__(256,1)`, `sm_120a`, nvcc versus NVRTC, and the full register-pressure graph may affect that schedule and register allocation.
- llama's immediately formed/reused `R4.64` base, versus Gate12's `R50.64` lifetime from `0x1a50` to `0x9940`, likely lowers address-register pressure. No preserved IR proves which pass created this difference.

### Unknown / unavailable

- Exact llama PTX, NVVM IR, ptxas scheduling directives, and the producing nvcc command.
- Whether qualifier, launch-bound minimum blocks, `sm_120a`, nvcc versus NVRTC, or their interaction is the necessary cause.
- Whether `LDG.E.CONSTANT` itself enables hoisting or is only another consequence of the same alias analysis.
- A CUDA C source construct that guarantees a particular physical register allocation or the <=160 SASS span. CUDA C cannot supply that guarantee; only the compiled-binary gate can.

## Mechanism ranking

1. **None: qualifier/launch/architecture A/B.** This matches llama's actual source mechanism and is already supported. Release only if the synthetic independently demonstrates the target pre-barrier load/post-barrier store shape without spill or traffic changes.
2. **Inline PTX load constraint.** Existing generated CUDA already uses inline-PTX operand constraints for MMA/LDSM, so a scalar `ld.global.u32` output constraint is mechanically supported. It is not llama-derived and ptxas still owns physical register allocation and scheduling; it must remain a synthetic fallback.
3. **Renderer PTX load/store region.** A single volatile inline-PTX region could make source ordering more opaque, but it would overconstrain the desired latency overlap, is not used by llama, and would require stronger address-space/type/predicate gates.
4. **Separate device/noinline body.** Ruled out as the llama mechanism by source attributes and symbol tables. It risks `CALL`/`RET`, ABI registers, frame growth, and lost scheduling overlap; do not test on real Q6.

## Exact synthetic microgate contract

The approved synthetic is a 2x2 matrix, held outside the Q6 builder:

| lane | Q8 pointer | launch / architecture package |
|---|---|---|
| A | mutable, unqualified | `__launch_bounds__(256)`, `sm_120` |
| B | `const __restrict__` | `__launch_bounds__(256)`, `sm_120` |
| C | mutable, unqualified | `__launch_bounds__(256,1)`, `sm_120a` |
| D | `const __restrict__` | `__launch_bounds__(256,1)`, `sm_120a` |

Freeze compiler version, the 18 affine immutable u32 addresses, source barriers, arithmetic DAG, shared destinations, and all non-axis source bytes. The combined launch/architecture package can establish an eligible configuration but cannot causally distinguish launch bounds from architecture; if C or D alone wins, split those axes in a second synthetic before claiming causality.

Hard synthetic gates:

- exactly one physical `LDG` per logical load and exactly 18 `LDG`/18 `STS` classified to the panel;
- all 18 loads before the existing overwrite barrier, all 18 stores after it, and the next publication barrier after all stores;
- first panel `LDG` to first panel `STS` <=160 normalized instructions;
- no added `BAR`, `MEMBAR`, `ATOM`, `CALL`, `RET`, scheduling-only `LOP3`, or extra global/shared/local traffic;
- stack/frame 0, local bytes 0, `LDL=0`, `STL=0`, and no register-count increase versus lane A;
- exact uint32 output identity and deterministic repeated cubin SHA for the selected lane.

If no C-level lane passes, the next isolated microgate may replace only one ordinary load with:

```cuda
uint32_t v;
asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(addr) : "memory");
```

Scale to 18 only after the one-load cubin proves exactly one `LDG`, no extra traffic/control, no spill, and the intended barrier side. This is a test contract, not an implementation recommendation or a claim about llama.

## Held real-Q6 binding plan

Do not bind until the synthetic release identifies an eligible lane. Then make one isolated candidate route while keeping the default admitted route byte-for-byte unchanged.

File scope:

- `extra/llm_research/prefill/nv_q6_oracle_broad_cta.py`: isolated Q8 panel1 candidate arm only; preserve arithmetic, ownership, combined publication, and all-partials reduction.
- `tinygrad/renderer/cstyle.py`: consume the compiler agent's released qualifier/launch API only; no new local renderer behavior in the real-Q6 experiment.
- `tinygrad/runtime/support/compiler_cuda.py` or the isolated harness: select `sm_120a` only through the released compiler option; default remains `sm_120`.
- A new Gate13-specific harness/evidence namespace: source, cubin, disassembly, census, correctness, timing, and decision ledger. Do not reuse Gate12 cubins.

Candidate source gates:

- only the immutable Q8 argument (`data2`) becomes `const unsigned int *__restrict__`; no store may target it;
- only the candidate kernel becomes `__launch_bounds__(256,1)` and only its compiler target becomes `sm_120a` if that exact synthetic lane wins;
- the default kernel and compiler command must rebuild the admitted main SHA `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137`;
- preserve the fixup SHA `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514`;
- retain exactly 18 direct Q8 panel1 assignments, zero named panel-load temporaries, and existing barriers only.

Candidate binary gates under `flock -w 1200 /tmp/nv-q6-oracle-gpu.lock`:

- exact Q8 panel1 `LDG/STS=18/18`, one physical load/store per logical copy, all loads pre-overwrite-barrier and stores post-barrier, span <=160;
- whole main `IMMA/LDSM/LDS/LDG/STS/STG/BAR = 256/32/176/109/73/64/4`;
- arithmetic census `1024/1544/1024/0`, unchanged from the admitted route;
- no added `LOP3`, `MEMBAR`, `ATOM`, `CALL`, or `RET`; instructions <=5144;
- registers <=255, stack/frame 0, local bytes 0, `LDL=0`, `STL=0`;
- trusted-reference exactness, partial and final uint32 identity, and CPU/GPU fixup identity before timing;
- same-process alternating locked R31 against the admitted 256.256 us route: candidate improves both main and total by at least 3 us and wins at least 24/31 pairs.

Any frozen-hash, source, SASS, resource, or correctness failure stops before R31. No real-Q6 repair is authorized by this plan; return to a smaller synthetic on failure.

## Reproduction commands

```bash
cd /home/ubuntu/tinygrad-arkey
LL=docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-mmq-dense.sm_120a.cubin
G12=docs/task_workflow/evidence/nv-q6-region-copy-panel1-gate12-20260831/fresh-artifacts/candidate/candidate.cubin
G12CU=docs/task_workflow/evidence/nv-q6-region-copy-panel1-gate12-20260831/fresh-artifacts/candidate/candidate.cu
LLSASS=docs/task_workflow/evidence/nv-q6-q8-panel1-binary-liveness-audit-20260831/llama.nvdisasm

sha256sum "$LL" "$G12" "$G12CU" "$LLSASS" /home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh
file "$LL" "$G12"
readelf -hSW "$LL"
readelf -hSW "$G12"
readelf -Ws "$LL"
readelf -Ws "$G12"
/usr/local/cuda-13.2/bin/cuobjdump --dump-ptx "$LL"
/usr/local/cuda-13.2/bin/cuobjdump --dump-ptx "$G12"
/usr/local/cuda-13.2/bin/cuobjdump --dump-resource-usage "$LL"
/usr/local/cuda-13.2/bin/cuobjdump --dump-resource-usage "$G12"
/usr/local/cuda-13.2/bin/cuobjdump --dump-elf "$LL" | rg -i -C2 'EIATTR|MAX_THREADS|MAXREG|STACK|FRAME|REGCOUNT|SHARED'
/usr/local/cuda-13.2/bin/cuobjdump --dump-elf "$G12" | rg -i -C2 'EIATTR|MAX_THREADS|MAXREG|STACK|FRAME|REGCOUNT|SHARED'
strings -a "$LL" | rg 'ptxas|Cuda compilation tools|Build cuda_|-arch|-m 64'
strings -a "$G12" | rg 'ptxas|Cuda compilation tools|Build cuda_|-arch|-m 64'

nl -ba /home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh | sed -n '3440,3535p;3695,3790p'
nl -ba "$G12CU" | sed -n '1,30p;1220,1260p'
rg -n 'asm|volatile|__restrict__|__launch_bounds__|by0|tile_y\[l\]|data2|__syncthreads' /home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh "$G12CU"
rg -n -- '-use_fast_math|-rdc=true|compute_120a|sm_120a|maxrregcount|mmq\.cu' /home/ubuntu/env/llama.cpp/build*/compile_commands.json
rg -n 'NVRTCCompiler|--gpu-architecture|--minimal|TINYGRAD_NV_USE_FAST_MATH' tinygrad/runtime/support/compiler_cuda.py

rg '0x(80b0|80c0|80e0|8100|8120|8150|8160|8170|8180|81b0|81e0|8210|8220|8230|8240|8250|8260|8270|8280|8290|8410|8490|8620|86e0|8770|8780|8790|87a0|87b0|87c0|87d0|87e0|87f0|8800|8810|8820|8830|8840|8850|8860|8870|8890)' "$LLSASS"
rg '0x(19cd0|19ce0|19d00|19d20|19d30|19d40|19d50|19d60|19d70|19d80|19d90|19da0|19db0|19dc0|19dd0|19de0|19df0|19e00|19e10|19e20|19fe0|1a230|1a2f0|1a300|1a330|1a340|1a360|1a380|1a3a0|1a3c0|1a3e0|1a3f0|1a400|1a410|1a420|1a430|1a440|1a450|1a460|1a470|1a4d0)' "$LLSASS"
```

The normalized llama disassembly was made previously with Triton's nvdisasm 12.8.55; that is an analysis-tool version, not the producing compiler. CUDA 13.2.86 supplied `cuobjdump`; `/usr/local/cuda-13.2/bin/nvdisasm` is not installed. The absence of preserved PTX/NVVM prevents a source-to-IR pass-level attribution, so this audit stops at the explicit synthetic contract above.
