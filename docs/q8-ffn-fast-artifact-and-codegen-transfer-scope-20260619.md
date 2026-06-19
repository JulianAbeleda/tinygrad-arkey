# q8 FFN fast-artifact + codegen-transfer scope (2026-06-19)

This is the forward scope after `q8-dual-track-route-and-codegen-scope-20260619.md` reached the A2 split verdict:

- **Correctness:** PASS. The one-block route matches the graph q8 proxy (`max_abs=0.00137`).
- **Quality:** PASS proxy. `dNLL=+0.00165` on the 160-token q8 gate/up proxy.
- **Economics in HIP oracle:** PASS. Producer + gate + up was `~107.6us`, implying about `~1.05x` decode EV.
- **Economics in HCQ-loadable COMGR artifact:** FAIL. One-block eager route measured `~195.1us`.

So the q8 lifecycle is not the blocker. The blocker is transferring the fast kernel into tinygrad's runtime or making
tinygrad generate an equivalent primitive.

Do **not** proceed to A3 graph capture with the slow COMGR artifact. A3 only makes sense after one of the two paths
below clears its isolated lifecycle gate.

## Target

Restore the oracle economics inside tinygrad-owned execution:

| unit | target |
|---|---:|
| producer + gate + up isolated lifecycle | <=129.2us (HIP oracle 107.6us + 20% route slack) |
| one-block route vs q8 proxy | max_abs <=2e-2 |
| W==D decode EV to justify final routing | >=3% sustained |
| default behavior | unchanged |

The two paths are complementary. Path A is an artifact-loader path. Path B is a tinygrad-owned generation path.

## Path A — fast artifact loader

Purpose: use the already-fast clang/hipcc-quality handwritten kernels through tinygrad HCQ, without HIP runtime in the
model process.

### A-F0 — characterize the hipcc object

Current evidence:

- `HIPCCCompiler("gfx1100").compile(MMVQ_SOURCE)` produces an ELF code object.
- `AMDProgram` rejects it with `unknown AMD reloc 10`.
- The COMGR object loads, but is too slow.

Tasks:

- Dump the hipcc object sections, symbols, relocation records, `.rodata` descriptor, and metadata.
- Compare against the COMGR object that `AMDProgram` accepts.
- Identify exactly what relocation type 10 means for AMDGPU in this object and which section it targets.

Gate:

- Produce `bench/q8-ffn-handwritten-oracle/hipcc_object_audit.json`.
- Name the relocation type, target section, addend semantics, and whether it can be applied by tinygrad's current image
  model.

Kill:

- If relocation type 10 targets unsupported external runtime state or device libraries that cannot be resolved without
  HIP runtime, close Path A.

### A-F1 — minimal loader support or object normalization

Try the smallest working route first:

1. **Object normalization:** adjust hipcc flags or post-link steps so the emitted object uses only relocations
   `AMDProgram` already supports.
2. **Loader support:** add probe-local handling for the identified relocation type, then decide whether it belongs in
   runtime code.

Constraints:

- No HIP runtime loaded in-process.
- No model route.
- No broad ELF-loader rewrite.
- Loader change must be justified by a concrete relocation record from A-F0.

Gate:

- A probe-local `FastAMDProgram` loads the hipcc object and launches the producer/consumer on tinygrad-owned buffers.
- A1 correctness still passes for `blk.0.ffn_gate.weight` and `blk.0.ffn_up.weight`.

Kill:

- If the relocation requires a general dynamic linker or runtime library state, stop. Do not build a partial dynamic
  linker for this q8 route.

### A-F2 — isolated performance parity

Measure the hipcc-quality HCQ artifact with the same harness as A1/A2.

Gates:

- Consumer per role <=60us, preferably near the HIP oracle `~50us`.
- Producer <=10us with the final in-model dtype contract.
- Producer + gate + up <=129.2us.
- No HIP runtime in `/proc/self/maps`.

Artifact:

- `bench/q8-ffn-handwritten-oracle/fast_artifact_perf.json`.

If this fails near COMGR timing, the issue is not only loader quality; close Path A and use Path B.

### A-F3 — one-block route replay

Rerun `extra/q8_ffn_oneblock_route.py` using the fast artifact.

Gates:

- Route vs q8 proxy max_abs <=2e-2.
- Producer + gate + up <=129.2us.
- No default change.

Only after this passes does A3 graph capture reopen.

## Path B — tinygrad-owned generation

Purpose: make the q8 FFN lifecycle a native tinygrad primitive rather than a backend artifact.

This path has two sublanes:

- **B1 raw-code/ASM transfer:** fastest way to own the kernel without relying on hipcc object loading.
- **B2 UOp/renderer capability:** broader compiler feature that can generate the producer/consumer family.

### B-G0 — final kernel contract

Write the precise contract before implementing anything:

- Producer input: decode row `[4096]`, RMSNorm weight, norm epsilon.
- Producer output: fp normalized activation `[4096]` plus q8_1 blocks `[128 * 36 bytes]`.
- Consumer input: Q4_K weight rows `[12288,4096]`, q8_1 activation.
- Consumer output: fp32 gate/up `[12288]`.
- Required dtype handling: avoid the A2 fp16-weight-to-float cast tax by either accepting fp16 norm weight or proving the
  cast is outside the measured lifecycle.
- Lifetime: q8 buffer is token-local, consumed exactly by `ffn_gate` and `ffn_up`, never cached across tokens.

Gate:

- Contract doc section and one JSON schema row for buffer layout, launch shape, and error tolerances.

### B-G1 — raw-code/ASM transfer spike

Use tinygrad's AMD raw assembler path (`renderer/amd/elf.py:assemble_linear` precedent) or a minimal hand-written
AMDGPU source path to produce an HCQ-loadable object with no hipcc relocations.

Scope:

- Start with the consumer, because A2's consumer dominates the slowdown (`~82us` vs HIP `~50us`).
- Producer second, because producer correctness depends on dtype contract and side-channel stores.

Gates:

- A1 correctness for gate/up.
- Consumer <=60us.
- No HIP runtime in-process.

Kill:

- If raw-code transfer becomes a broad assembler project rather than a minimal q8/MMVQ primitive, stop and move to B-G2.

### B-G2 — producer expressibility retry

Retry the fused producer as a current-UOp custom kernel using the exact B-G0 contract.

Required structure:

- One kernel.
- One per-row reduction for RMSNorm sumsq.
- Barrier/LDS broadcast of `rinv`.
- Per-32 max/quantize.
- Multi-output stores.

Gate:

- UOp verification passes.
- DEBUG source has one producer kernel, not staged kernels.
- Producer output matches handwritten producer.
- Producer <=10us.

Kill:

- If current UOps still cannot express the staged reduction plus post-barrier multi-output stores, do not keep trying
  schedule variants. Escalate to B-G3.

### B-G3 — minimal renderer capability

Add the smallest compiler feature that makes B-G2 legal:

- staged local reductions with barriers;
- post-barrier multi-output stores;
- explicit local-memory lifetime in the range model.

Non-goals:

- no general flash-attention compiler;
- no broad graph scheduler rewrite;
- no default model route.

Gate:

- standalone producer test passes;
- no existing custom-kernel regressions in the local test set;
- generated source shape matches the handwritten contract.

### B-G4 — generated consumer parity

Decide whether tinygrad owns the consumer too:

- **B-G4a:** generated producer + existing q8 consumer. Lower risk, likely not enough speed.
- **B-G4b:** generated/raw consumer matching llama-style Q4_K x q8_1 MMVQ. Higher risk, required for the measured EV.

Gate:

- gate/up lifecycle >=1.15x over current fp coop.
- one-block route <=129.2us producer+gate+up.

### B-G5 — model route

Only after B-G4 passes:

- add `Q8_FFN_HANDWRITTEN` or successor env flag;
- route dense FFN gate/up only;
- keep default off;
- run dNLL and W==D ctx sweep.

Final gate:

- dNLL <=0.01;
- W==D decode speedup >=3%;
- flag-off output unchanged.

## Decision matrix

| outcome | decision |
|---|---|
| Path A loads hipcc object and clears <=129.2us | reopen A3 graph capture as research flag |
| Path A loads but stays slow | close artifact path; evidence says compiler/runtime object quality is not enough |
| Path A blocked by runtime reloc/dynamic-link state | close artifact path unless project wants a broader AMD loader |
| Path B producer passes but consumer lags | keep as codegen asset unless W==D still clears >=3% |
| Path B consumer+producer match oracle | proceed to env-gated model route |
| both pass | prefer Path B for project ownership; keep Path A as oracle/regression target |

## Recommendation

Run Path A-F0 first because it is a bounded diagnostic: one object, one relocation failure, clear kill gate. If it
passes, it may restore the fast route quickly. In parallel or next, run B-G0/B-G1 for ownership: even if Path A works,
the long-term tinygrad primitive is the raw/codegen transfer, not a brittle external object.
