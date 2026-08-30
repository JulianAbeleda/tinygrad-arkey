# NVIDIA all-native packed inference stack scope

Date: 2026-08-30

Status: executable implementation scope

Target: NVIDIA RTX 5090, `NV`, `sm_120`, Qwen3-8B Q4_K_M

Agent class: low-effort agents for atomic implementation packets; root agent owns integration and promotion.

## 1. Goal

Reproduce the llama-competitive `pp512` endpoint with executable binaries generated from tinygrad-owned IR or source.

The completed route must not require a llama-extracted cubin, a llama kernel symbol, or an evidence-directory binary at runtime. NVIDIA may still execute cubins, but every cubin must be reproducibly compiled from tinygrad-owned input through the tinygrad compiler/runtime stack.

The current llama-cubin route remains an oracle until the all-native route passes the full integration gate. It is not the implementation target.

## 2. Ground truth

The current promoted endpoint is not all-native:

- `NV_LLAMA_FULL_PACKED_PP512` defaults on for the qualified NV `pp512` shape.
- Q4_K and Q6_K main/fixup programs read llama-extracted cubins from evidence directories.
- the promoted whole-tile Flash program is an opaque cubin; the separate clean-room path compiles CUDA source with NVCC but is not a tinygrad UOp program.
- the latest endpoint measured tinygrad `35.1865905 ms` median versus llama `35.334424 ms` median.
- the last `2.177346 ms` recovery came from single-owner opaque-program lifecycle work. Native replacement must preserve this graph property.

The native substrate is partially implemented:

- Q4 gate/up: compiler-owned direct-output IMMA, `M512 N12288 K4096`.
- Q4 K: compiler-owned direct-output IMMA, `M512 N1024 K4096`.
- Q4 Q/O: compiler-owned direct-output IMMA, `M512 N4096 K4096`.
- Q4 down: compiler-owned candidate, `M512 N4096 K12288`.
- Q6 V and down: compiler-owned candidates using canonical Q6_K weights and graph-owned Q8 records.
- decode Flash has tinygrad-owned generated online-softmax/PV emitters that can inform, but not silently substitute for, the prefill contract.

No existing artifact proves that the complete compiler-owned composition reaches the `35 ms` endpoint.

## 3. Definitions

`llama_extracted`:

- executable bytes copied or extracted from llama.cpp or its process.
- a production binding reads an evidence `.cubin` containing a llama implementation.

`external_source_compiled`:

- source is locally owned, but an opaque CUDA/NVCC/NVRTC program bypasses the tinygrad UOp/compiler ownership contract.

`tinygrad_generated`:

- program body is produced from tinygrad-owned UOps, scheduler IR, or a typed `KernelProgram` emitter.
- source identity and compiled-binary identity are recorded.
- the binary can be regenerated without llama.cpp or an evidence binary.

The final route requires `tinygrad_generated` for every hot-path program. An intermediate source-owned producer may remain `external_source_compiled` only while its replacement packet is open; it cannot pass the final gate.

## 4. Non-negotiable invariants

1. Canonical packed GGUF Q4_K/Q6_K weights are direct inputs.
2. No expanded FP16 weight overlay or hot-path weight copy is allowed.
3. Q8 records, outputs, and workspace are graph-owned.
4. One physical program invocation has one graph owner, including multi-output main programs.
5. A role swap cannot change another role, prompt fixture, graph policy, or allocation policy.
6. Correctness uses complete output data, not a top-1 proxy alone.
7. Isolated kernel timing cannot promote a route.
8. Every whole-model timing is same-session control/candidate/control, serialized with `/tmp/gpu-bench.lock`.
9. Profiling measurements are not wall authority.
10. The oracle route is removed only after the all-native integration gate passes.

## 5. Authority fixture

All pp512 packets use:

- model `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`.
- device `NV`, architecture `sm_120`.
- `B=1`, `T=512`, `start_pos=0`, `Hq=32`, `Hkv=8`, `Hd=128`.
- the existing deterministic model-arm prompt and temperature-zero output contract.
- R9 wall samples for promotion decisions.
- full logits, selected token, finite check, per-role launch census, program identity, and source/binary provenance.

The oracle baseline is the default `NV_LLAMA_FULL_PACKED_PP512=1` route. The rollback baseline disables only the packet under test.

## 6. Dependency graph

```text
P0 provenance gate
 |
 +--> Q4-1 gate-only --> Q4-2 up-only --> Q4-3 gate/up pair
 |                                      |
 |                                      +--> Q4-4 K
 |                                      +--> Q4-5 Q/O
 |                                      +--> Q4-6 V
 |                                      +--> Q4-7 down
 |
 +--> Q6-1 provenance/direct-output --> Q6-2 V --> Q6-3 down --> Q6-4 vocab
 |
 +--> F1 Flash mathematical ABI --> F2 UOp body --> F3 dynamic contract --> F4 model integration

Q4-7 + Q6-4 + F4 --> I1 all-native composition --> I2 matched wall --> I3 length generalization --> I4 oracle removal
```

Packets at the same graph depth may run in parallel only when their permitted file lists do not overlap. `tinygrad/llm/model.py` integration packets are serialized.

## 7. Packet P0: provenance gate

Permitted files:

- new `extra/llm_research/prefill/nv_all_native_provenance_gate.py` only.

Implementation:

- classify every default pp512 family as `llama_extracted`, `external_source_compiled`, or `tinygrad_generated`.
- audit production selection, binding module, executable input, symbol, and reproducible source identity.
- provide audit mode and `--require-all-native` mode.

Required JSON:

- schema and repository root.
- family, role, production call site, binding file, population, binary origin, source identity, binary identity, violations.
- aggregate counts by origin and verdict.

Pass:

- audit mode reports current violations deterministically and exits zero.
- `--require-all-native` fails while any production-selected family reads an extracted/evidence cubin, imports a llama binding, or lacks a tinygrad provenance identity.

Stop:

- do not infer provenance from a program name.
- do not open a GPU or compile programs.

### P0 observed violations (2026-08-30)

The static audit artifacts are `docs/task_workflow/evidence/nv-all-native-stack-20260830/provenance-audit.json` and `provenance-require-all-native.json`. The ordered current violations and owning packets are:

1. `q4_gate_up`: imports `nv_llama_packed_q4k_pp512_binding` and reads extracted Q4 main/fixup cubins. Owner: `Q4-1`, then `Q4-2`, `Q4-3`.
2. `qkv`: imports llama Q4/Q6 bindings and reads extracted Q4/Q6 main/fixup cubins. Owner: `Q4-4`, `Q4-6`.
3. `q4_attention_o`: imports `nv_llama_packed_q4k_o_pp512_binding` and reads extracted Q4 cubins. Owner: `Q4-5`.
4. `q4_ffn_down`: imports `nv_llama_packed_q4k_down_pp512_binding` and reads extracted Q4 cubins. Owner: `Q4-7`.
5. `q6_ffn_down`: imports `nv_llama_packed_q6k_down_pp512_binding` and reads extracted Q6 cubins. Owner: `Q6-1`, `Q6-3`.
6. `flash_attention`: imports `nv_llama_fattn_mma_pp512_binding` and reads the specialized evidence cubin. Owner: `F1`, `F2`, `F3`, `F4`.
7. `q6_vocabulary`: imports `nv_llama_q6k_vocab_pp512_binding` and reads the evidence Q6 vocabulary cubin. Owner: `Q6-4`.

Audit mode exits zero with these violations reported; `--require-all-native` exits one until all owning packets pass.

## 8. Q4 packets

### Q4-1: gate-only native swap

Permitted files:

- `tinygrad/llm/model.py`.
- `extra/llm_research/prefill/nv_compiler_q4k_pp512_binding.py`.
- one new gate-only model harness.

Delta:

- add one default-off flag that replaces only the 36 `ffn_gate` projections.
- keep `ffn_up`, Q/K/V/O, down, Flash, vocabulary, and lifecycle unchanged.

Pass:

- 36 compiler Q8 producers and 36 compiler main programs.
- zero llama gate main/fixup programs in the swapped population.
- exact expected llama/ordinary census for every other role.
- canonical Q4 weight bases, no copies, no FP16 overlay.
- finite complete output, unchanged token, accepted logit tolerance.
- same-session whole-model candidate does not regress beyond 5% versus the oracle arm before proceeding.

Stop:

- any second capture pool, eager materialization, or unrelated role change.

### Q4-2: up-only native swap

Repeat Q4-1 for `ffn_up`. Use a distinct default-off flag and the same ABI. Do not compose gate/up yet.

### Q4-3: paired gate/up native route

Dependencies: Q4-1 and Q4-2 pass.

Delta:

- quantize the shared activation once where the existing record ABI permits it.
- consume the graph-owned record from both projections.
- preserve single-owner invocation semantics.

Pass:

- 36 Q8 providers, 72 Q4 consumers, no duplicate provider.
- no llama gate/up main/fixup symbols.
- full model correctness and same-session wall bracket.

### Q4-4: attention K

Use `nv_compiler_q4k_k_pp512_binding.py`, exact shape `M512 N1024 K4096`, 36 projections. Replace K only and retain Q/V/Flash/O controls.

### Q4-5: attention Q/O

Use `nv_compiler_q4k_qo_binding.py`, exact shape `M512 N4096 K4096`, 36 Q plus 36 O projections. Test Q and O independently before composition. O must preserve its residual-aware output ABI and single-owner graph contract.

### Q4-6: attention V

Replace only the Q4_K V population. Q6_K V remains the Q6 track's responsibility. Record the exact per-block type census.

### Q4-7: FFN down

Use the compiler-owned down binding for the 18 Q4_K blocks. Remove any eager `record.realize()` from the candidate lifecycle before timing; dependency ordering must remain graph-owned.

## 9. Q6 packets

### Q6-1: provenance and direct-output contract

Permitted files:

- `extra/llm_research/prefill/nv_compiler_q6k_pp512_binding.py`.
- one new Q6 provenance/census harness.

Delta:

- assert canonical `uint16` Q6_K weights and compiler candidate identity.
- assert no llama import, cubin read, partial workspace, or fixup program.
- give `attn_v` the same direct-output ownership contract as down when this is isolated to the binding.

Pass:

- deterministic role identities and census.
- complete output correctness fields.
- zero extracted programs and zero partial/fixup workspace.

### Q6-2: V model integration

Dependency: Q6-1.

- replace only the 18 Q6_K V projections.
- retain Q4_K V and every other attention role.
- require exact type/population census and full logits.

### Q6-3: down model integration

Dependency: Q6-1.

- replace only the 18 Q6_K down projections.
- preserve down residual semantics and canonical weight storage.

### Q6-4: vocabulary

- replace the extracted Q6 vocabulary cubin with a compiler-generated full-logit service.
- output all 151936 logits before argmax qualification.
- do not promote a final-row or top-1-only shortcut.

## 10. Flash packets

### F1: mathematical ABI freeze

Inputs:

- clean-room CUDA reference.
- current promoted opaque Flash fixture.
- tinygrad generated decode Flash emitter as representation guidance only.

Output:

- typed input/output ABI, causal mask rule, GQA ownership, accumulator precision, online-softmax recurrence, tile mapping, and output layout.

No executable llama instruction sequence is an implementation input.

### F2: tinygrad UOp emitter

- express the F1 contract as a `KernelProgram` emitter with declared tinygrad provenance.
- compile through the normal NV PTX/nvJitLink path.
- preserve 36 calls and graph-owned outputs.

Pass:

- fixture parity, finite output, full-model token/logit contract, no external cubin read.

### F3: dynamic contract

Generalize the emitter before promotion:

- `T` in `128, 512, 1024`.
- `start_pos` in `0, 256` where legal.
- live `Tc=start_pos+T` rather than an allocation-derived workload.
- non-ring first; ring is a separate sub-packet if its semantics differ.

### F4: production integration

- add a tinygrad-generated selector with its own default-off promotion record.
- keep the opaque cubin route as an explicit oracle flag.
- no production import from an `nv_llama_*` binding in the candidate arm.

## 11. Integration packets

### I1: all-native composition

Dependencies: Q4-7, Q6-4, F4.

Requirements:

- one flag selects the complete native stack.
- partial mixtures are test-only and cannot become the default accidentally.
- P0 `--require-all-native` passes.
- exact route census contains no llama symbol and no evidence-binary source.
- graph has no duplicated multi-output main invocation.

### I2: matched endpoint gate

Run interleaved R9:

```text
oracle / all-native / oracle
all-native / llama / all-native
```

Pass:

- complete correctness gate passes.
- all-native median is within 5% of the llama-cubin oracle and within 5% of same-session llama.
- minimum and median are both reported; promotion uses median.
- cold compile time is reported separately and never included in warm throughput.

### I3: length generalization

After pp512 passes:

- prefill `pp128, pp256, pp512, pp1024, pp2048, pp4096`.
- prefix-prefill matrix crosses query chunk `128,256,512` with `start_pos=0,512,2048,4096` where legal.
- decode is a separate route census at `d512,d1024,d2048,d4096`; it must not inherit a prefill kernel by name alone.

Every row records route identity, fallback reason, correctness, latency, throughput, and achieved-bandwidth interpretation.

### I4: oracle removal

Only after I2 and I3 pass:

- default the all-native route on for its qualified shapes.
- default the llama-cubin route off and mark it research-only.
- remove production imports of llama bindings.
- retain oracle artifacts and reproduction harnesses outside the runtime.

## 12. Low-agent operating contract

Every task prompt must contain:

- one packet ID.
- exact permitted files.
- prerequisite artifact paths.
- exact flag and control arm.
- required JSON schema fields.
- pass and hard-stop conditions.

Low-effort agents must not:

- edit files outside the packet.
- change benchmark fixtures or tolerances.
- run git commands.
- promote defaults.
- claim wall recovery from isolated timing.
- repair an unexpected dependency by broadening scope.

If a packet cannot be completed inside its permitted files, the agent returns `BLOCKED` with the missing interface, proposed owner, and minimal new dependency. Root then creates a new packet; the agent does not improvise one.

## 13. Completion definition

The campaign is complete when:

1. P0 reports every production-selected pp512 program as `tinygrad_generated`.
2. I1 contains no llama symbol, extracted/evidence cubin read, hot-path copy, or duplicated main invocation.
3. Full logits and tokens pass the established correctness contract.
4. I2 is within 5% of both the oracle stack and same-session llama.
5. I3 has no unexplained route cliff across qualified prompt lengths and decode depths.
6. I4 removes the llama binary path from production selection while retaining a reproducible research oracle.
