# Kernel program execution boundary scope

Date: 2026-07-29

Promotion direction: implement and validate on `exp`, promote the production-only commits to `dev`, validate, then promote the same production changes to `master`.

## Problem

`Tensor.uop_program` is a low-level graph-construction mechanism. Its name does not describe who authored a program, how a configuration was selected, or whether the call is production, oracle, or research-only. The current promoted LLM routes call it directly, which can make a machine-generated/search-selected program look like a hand-tuned kernel.

The method itself is not wrong and remains a generic tinygrad API. The architectural error is exposing that transport primitive at every route call site without typed lifecycle/provenance context. `Tensor.custom_kernel` is retained silently as the upstream-compatibility spelling.

## Outcome

Production code reads `execute_promoted_program(...)`. EXP fallback/oracle code reads `execute_oracle_program(...)`. Unpromoted experiments read `execute_research_program(...)`. All three use `Tensor.uop_program` only inside one small boundary module.

This refactor changes no emitter, UOp graph, selected route, tensor shape, ABI, scheduling option, fallback, environment gate, or performance claim.

## Non-goals

- Do not rename or change the `UOp.custom_kernel` substrate globally.
- Do not modify kernel bodies or machine-search artifacts.
- Do not reclassify route provenance based on naming alone.
- Do not promote an EXP oracle or experiment into master.
- Do not add online autotuning.
- Do not touch the eGPU; AMD recertification remains a final evidence refresh.

## Pinned API

New production-owned module: `tinygrad/llm/kernel_program.py`.

```python
class KernelProgramProvenance(StrEnum):
  MACHINE_SEARCH_GENERATED = "machine_search_generated"
  TINYGRAD_SCHEDULER_GENERATED = "tinygrad_scheduler_generated"
  HAND_AUTHORED_ORACLE = "hand_authored_oracle"
  RESEARCH_ONLY = "research_only"

@dataclass(frozen=True)
class KernelProgram:
  route_id: str
  program_id: str
  provenance: KernelProgramProvenance
  emitter: Callable

  def to_dict(self) -> dict[str, str]: ...

def execute_promoted_program(output: Tensor, *inputs: Tensor, program: KernelProgram) -> Tensor: ...
def execute_oracle_program(output: Tensor, *inputs: Tensor, program: KernelProgram) -> Tensor: ...
def execute_research_program(output: Tensor, *inputs: Tensor, program: KernelProgram) -> Tensor: ...
```

Contract:

- Identifiers are non-empty strings.
- `emitter` is callable.
- `execute_promoted_program` accepts only `MACHINE_SEARCH_GENERATED` or `TINYGRAD_SCHEDULER_GENERATED`.
- `execute_oracle_program` accepts only `HAND_AUTHORED_ORACLE`.
- `execute_research_program` accepts only `RESEARCH_ONLY`.
- Each function delegates exactly once to `output.uop_program(*inputs, fxn=program.emitter)` and returns element zero, preserving the present output contract.
- `to_dict()` excludes the callable and emits stable route/program/provenance trace facts.
- The boundary does not inspect names to infer provenance.

Master may contain the vocabulary and boundary, but master production call sites use only `execute_promoted_program`. Oracle and research functions are generic enforcement tools with no oracle implementation or selected fallback on master.

## Work packages

### KP0 — Scope and baselines

Owner: orchestrator.

- Record branch heads and clean status.
- Record every direct `.uop_program(` and legacy `.custom_kernel(` call under `tinygrad/llm` and `extra/llm_research`.
- Preserve existing UOp/frozen identity tests as the behavioral baseline.

Acceptance: scope committed on EXP before implementation.

### KP1 — Core typed boundary

Files:

- `tinygrad/llm/kernel_program.py`
- `test/unit/test_llm_kernel_program.py`

Tests:

- Valid promoted machine-search and scheduler programs delegate once and return output zero.
- Promoted execution rejects oracle/research provenance.
- Oracle execution rejects promoted/research provenance.
- Research execution rejects promoted/oracle provenance.
- Empty identifiers and non-callable emitters fail at construction.
- `to_dict()` is stable and contains no emitter.

Acceptance: pure CPU tests pass; no other file changes.

### KP2 — Production hot-path migration

Files:

- `tinygrad/llm/decode_routes.py`
- `tinygrad/llm/flash_decode_attention.py`
- `tinygrad/llm/fused_attention.py`
- focused existing tests and, only if required, new production-boundary assertions

Routes:

- Q4_K G3 decode GEMV: route `decode_q4k_g3_generated`; program ID derived from the selected candidate ID.
- Q6_K decode stage one: route `decode_q6k_coop_generated`; distinct `.gemv` program ID.
- Q6_K large-vocabulary reduction: same route; distinct `.vocab_reduce` program ID.
- Flash G4/G5 tile and combine: route/config from `FLASH_DECODE_G4/G5`; distinct `.tile` and `.combine` program IDs.
- Promoted fused prefill attention: route `prefill_flash_attention_generated`; program ID includes its stable selected geometry identity.

All production explicit programs use `MACHINE_SEARCH_GENERATED`, matching the existing promoted manifest classification. Ordinary scheduler-generated graph GEMMs do not call this boundary merely to acquire a label.

Flash constraint:

- Production flash execution must resolve an admitted `FlashDecodeRouteConfig` for the actual Hq/Hkv/Hd/device geometry.
- Unsupported exploratory geometry must not be mislabeled as a promoted route. EXP HD sweeps move to KP3 research execution.

Documentation in `fused_attention.py` must call this a promoted program boundary. It may explain that `Tensor.uop_program` is the internal transport once, but must not present transport as provenance.

Acceptance:

- `rg '\.custom_kernel\(' tinygrad/llm` returns no callers; `rg '\.uop_program\(' tinygrad/llm` returns only `kernel_program.py`.
- Existing decode/flash/fused UOp identity tests pass unchanged.
- Route IDs, program names, shapes, and fallback behavior remain unchanged.

### KP3 — EXP qualification, research, and oracle migration

Files: `extra/llm_research/**` and focused EXP tests only.

Classification:

- Production qualification adapters that execute a master promoted emitter use `execute_promoted_program` with the production route ID.
- Explicit independent handwritten fallback/reference implementations use `HAND_AUTHORED_ORACLE` and `execute_oracle_program`.
- Candidate sweeps, microbenchmarks, resource probes, MMQ experiments, and unsupported geometry experiments use `RESEARCH_ONLY` and `execute_research_program`.

Known direct sources to migrate:

- `decode/current_decode_execution_adapter.py`
- `decode/decode_hd_sweep_numerics.py` (must build/execute its descriptor as research, not call the admitted production executor for unsupported geometry)
- `benchmark_split_shared_attention.py`
- `mmq_ds4_logical_emitter.py`
- `mmq_q4k_q8_atom.py`
- `phase_abi_v1_resource_probe.py`

Any additional `extra/llm_research` hit discovered by static search is classified explicitly. Tests that directly exercise the generic Tensor mechanism use the canonical `.uop_program`; runtime, benchmark, qualification, and campaign source may not use the legacy spelling.

Acceptance:

- No direct legacy `.custom_kernel(` remains under `extra/llm_research`.
- No qualification path labels unsupported geometry promoted.
- Current production qualification still imports production emitters.
- CPU-focused EXP tests and syntax checks pass.

### KP4 — Static provenance and identity gates

Add tests that prove:

- Production LLM files do not call `.custom_kernel` directly and use `.uop_program` only in the typed boundary.
- Master production files do not call oracle/research execution functions.
- EXP research sources have no direct legacy `.custom_kernel` calls.
- Every explicit program object has a non-empty route/program identity.
- Existing production emitter UOp keys/frozen digests remain identical.

Do not create a second route manifest or global selector.

### KP5 — EXP validation

Required before promotion:

1. Boundary unit tests.
2. Decode route and decode-kernel identity tests.
3. Production flash/oracle parity tests.
4. Fused-attention structural tests that are CPU-capable.
5. Current decode adapter CPU/compile tests that are available; missing ROCm tooling is reported separately.
6. Repository static boundary audit.
7. `git diff --check` and clean commit boundaries.

### KP6 — Promotion EXP → dev → master

Commit boundaries:

1. `[docs] scope kernel program execution boundary` — EXP only.
2. `[runtime] add typed kernel program boundary` — production commit, promote.
3. `[runtime] route promoted programs through typed boundary` — production commit, promote.
4. `[runtime] classify EXP explicit program execution` — EXP only.
5. `[test] enforce explicit kernel program provenance` — split production/EXP portions as necessary.
6. `[docs] record kernel program boundary result` — EXP only.

Promotion rules:

- No squashing that obscures production versus EXP-only changes.
- Promote the production commits unchanged where possible.
- Dev validates first and retains its fallback/oracle implementations; those implementations must call the oracle boundary.
- Master receives no EXP implementation, benchmark, campaign, or oracle.
- A conflict is resolved by preserving the target branch's legitimate extra functionality while applying the same production API and hot-path semantics.

### KP7 — Final validation and publication

EXP:

- All affected tests green.
- Full historical suite sampled; hardware/tooling failures separated from regressions.

Dev:

- Boundary, route, identity, and fallback-focused tests green.
- No production direct legacy `.custom_kernel` call.

Master:

- Full `test/unit` suite green.
- CLI help smoke passes.
- No master import of EXP.
- Only `tinygrad/llm/kernel_program.py` contains the low-level call.

Push EXP, dev, and master only after their gates pass and verify local HEAD equals the corresponding remote.

## Completion definition

The refactor is complete when a new reader sees explicit promoted/oracle/research execution at every LLM program call site; `Tensor.uop_program` is confined to one transport boundary in production and the legacy Tensor spelling is absent from EXP campaign source; all production identities and behavior are unchanged; validated commits have been promoted EXP → dev → master; branches are clean and pushed; and AMD recertification is documented as evidence refresh rather than unfinished code.
