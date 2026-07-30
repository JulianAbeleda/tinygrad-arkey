# Metal semantic role identity inventory

Date: 2026-07-30

Status: MR5 inventory only. No runtime change or hardware claim.

## Existing reusable authorities

- `tinygrad.llm.model_facts` is the single production vocabulary authority:
  `QK_ROUTE_ROLES`, `normalize_route_role`, `TensorFact`, and the Qwen GGUF
  resolver provide model tensor name, module path, Q4_K/Q6_K label, logical
  rows/cols, and role. Reuse it; do not add a Metal role enum.
- `tinygrad.llm.model_route_plan` converts those facts into
  `PrimitiveRouteEntry(name,module_path,quant_label,rows,cols,role,parts,opts,
  family)`. This is the existing route/shape/quant attachment inventory.
- `tinygrad.llm.model` already brackets semantic graph regions with
  `role_metadata` (`rms_norm`, `residual`, attention score/mask/softmax/AV,
  rope, etc.) and retains the loaded GGUF metadata through model construction.
- `tinygrad.llm.memory_semantics` and `KernelInfo.memory_semantic_slots` carry
  allocation ownership independently of role. They must remain separate.
- `KernelInfo.candidate_context` and `ProgramInfo.candidate_context` preserve
  selected compiler context where it exists. `CallInfo.metadata` is propagated
  by `tinygrad.engine.jit` when graph calls are batched.

## Exact gaps

1. A packed-linear `PrimitiveRouteEntry` is not currently attached as structured
   call/program metadata during decode capture. The compiler sees a program name
   and buffer ownership, but a portable trace cannot reliably join every program
   to tensor name, role, M/N/K, quant, scalar/accumulator dtype.
2. `role_metadata` covers broad operation regions but does not carry exact model
   tensor identity or quant layout. Labels cannot be parsed back into policy.
3. JIT batching preserves `CallInfo.metadata`, but there is no normalized
   trace-export projection that joins it with `ProgramInfo`, graph membership,
   and the source/binary identity.
4. Model facts provide rows/cols; decode candidate identity needs explicit
   logical M=1 plus dtype/layout facts at the linear construction boundary.

## Minimal seams

1. Add one immutable, target-neutral semantic identity dataclass beside existing
   model facts (not a Metal class): model/tensor identity, normalized role,
   phase, logical M/N/K, quant storage/layout, scalar input/output/accumulator
   dtypes. Construct it only from `TensorFact` + the linear invocation.
2. Attach that identity once to the corresponding `KernelInfo`/`CallInfo`
   metadata at packed linear construction. Preserve it through existing JIT
   graph batching; do not encode it into a program label.
3. Let MR1 observation export the existing metadata as optional fields and emit
   `metadata_unavailable` when no identity reaches a call. The graph-admission
   census owns admission reasons; this seam owns only semantic identity.

## Tests

- Unit: Qwen GGUF tensor rows resolve through `model_facts` to the existing role
  vocabulary and exact quant/shape facts.
- Unit: attached identity survives a single call and a batched graph call without
  changing executable UOps or memory-semantic ownership.
- Integration/mock: census export joins a call to identity when present and
  explicitly reports unavailable metadata when absent; no kernel-label parsing.
- Metal smoke after MR1: trace identity fields match the loaded model hash and
  selected program source/binary hashes, without asserting per-dispatch time.

## Non-goals

No new route selector, target table, label parser, Metal-specific vocabulary,
candidate policy, or runtime performance alteration is proposed here.
