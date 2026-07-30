# Model program semantic identity design

Date: 2026-07-30

Status: MR5 design only. The real graph-admission census currently has empty
call metadata; this document proposes the smallest propagation seam.

## Authority and ownership

`tinygrad.llm.model_facts` remains the sole role/quant/model-tensor authority:
`TensorFact` gives tensor name/module path, normalized role, GGUF quant and
logical matrix rows/cols. `model_route_plan.PrimitiveRouteEntry` is the existing
runtime attachment for packed route shape/quant/role. Neither graph observation
nor Metal may infer this information from a kernel label.

The generic compiler/JIT owns transport only: `KernelInfo`, `CallInfo.metadata`,
and `tinygrad.engine.jit` graph batching already preserve metadata. The census
export owns serialization of metadata that reached a call. BoltBeam owns joins,
ranking and policy; it receives explicit unavailable fields when no identity is
present.

## Proposed type and API

Add one frozen target-neutral value type in `tinygrad.llm.model_facts`:

```python
@dataclass(frozen=True)
class ProgramSemanticIdentity:
  phase: Literal["decode", "prefill"]
  tensor_name: str
  module_path: str
  role: str                    # normalized by normalize_route_role
  logical_m: int
  logical_n: int
  logical_k: int
  source_quant_storage: str    # TensorFact.quant_label, or "none"
  source_weight_layout: str    # e.g. gguf_packed_row_major / dense_row_major
  execution_representation: str # e.g. generic_tensor / qk_packed_primitive
  input_dtype: str
  output_dtype: str
  accumulator_dtype: str | None
```

It validates positive M/N/K and uses the existing role normalization. It does
not contain target, schedule, candidate, source hash, buffer offset, timing or
route-selection policy.

### Actual metadata transport

`role_metadata` currently accepts only `str` and creates the frozen
`helpers.Metadata(name, caller, backward)` record; it cannot transport rich
fields. The smallest concrete extension is a frozen `ProgramIdentityMetadata`
subclass of `Metadata` in `tinygrad.llm.model_facts` (or a small shared metadata
module if import layering requires it). It retains the existing three base
fields and adds the identity fields above. `role_metadata` changes its argument
from `str` to `str | Metadata`, preserving its current string behavior and
installing the supplied immutable record unchanged. `CallInfo.metadata` already
accepts `tuple[Metadata,...]`, and JIT already merges those tuples. Census
serialization then recognizes this subtype; unrecognized ordinary `Metadata`
continues to serialize as name/caller/backward. This is one additive transport
change, not a parallel side table or label encoding.

One constructor, `program_identity_for_linear(tensor_fact, *, phase, m,
input_dtype, output_dtype, accumulator_dtype, source_layout,
execution_representation)`, belongs next to `TensorFact`. A genuinely dense
source passes `source_quant_storage="none"`. A Q4_K/Q6_K generic fallback and
its packed primitive pass the same source fact and layout; they differ only in
the explicit execution representation (and in dtypes when that is actually
true).

## Call sites

1. During `Transformer.from_gguf`, retain the already-built `ModelFacts` lookup
   by tensor name alongside the existing `ModelRoutePlan`; do not rescan GGUF or
   add a parallel role map. Perform attachment only after QK primitive
   installation has completed, using the same module paths but without replacing
   modules again.
2. Add an optional `call_metadata_factory` field to existing `nn.Linear` and the
   existing QK primitive linear base, with one shared helper owning the call
   context: `with linear_call_metadata(module, x): return existing_call(x)`.
   The default factory is `None`, so global nn behavior is byte-for-byte
   unchanged. This preserves parameter names, `state_dict` paths, `_module_at`,
   `_set_module_at`, and QK registry behavior.
3. Attach a `TensorFact` plus the shared factory to the final module object
   after primitive replacement. Generic `nn.Linear.__call__` and QK primitive
   `__call__` invoke the same helper around their existing expressions; neither
   wraps a module nor changes its parameter ownership.
4. Derive `m` at invocation from the activation's concrete logical leading
   extent; `n/k`, role and **source storage** quant/layout come from `TensorFact`.
   Keep source storage (`Q4_K` GGUF packed, for example) separate from execution
   representation (ordinary dequantized Tensor lowering versus a packed
   primitive). Thus an ordinary fallback of a Q4_K GGUF tensor reports
   `source_quant_storage="Q4_K"` and `execution_representation="generic_tensor"`,
   not a false dense/none source claim. This must not consult target/backend.
5. Preserve metadata through normal Tensor lowering. `KernelInfo` and
`function.call(...metadata=...)` already carry it; JIT graph batching merges
member `CallInfo.metadata`. No changes to names, labels, or Metal renderer.
6. MR1 census export maps recognized identity metadata to structured optional
semantic fields. If absent, it emits `metadata_unavailable`; it never parses
`program_name`.

## Tests

- `model_facts`: constructor accepts Q4_K/Q6_K and dense facts, rejects unknown
  role/nonpositive dimensions, and preserves existing normalized vocabulary.
- model/route-plan: one QK primitive linear and one generic fallback linear
  retain identical module/state-dict paths and source TensorFact; their identity
  differs only in explicit execution representation/layout as dictated by their
  actual source.
- schedule/JIT: identity reaches a direct call and each graph member without
  altering UOp keys, buffer ownership, program source, or candidate context.
- census: structured fields are emitted from metadata; absent identity remains
  explicitly unavailable; label strings cannot supply identity.
- integration: fixed-depth decode census has identities for attributable linear
  calls and no claim for unrelated kernels.

## Migration risks and guards

- A fused program may include several semantic linears. Existing
  `CallInfo.metadata` is already a tuple: transport one `ProgramIdentityMetadata`
  per source linear in source order, deduplicate exact equal records only, and
  census reports an ordered `semantic_identities` list. It never selects one or
  allocates graph time between members.
- Metadata must be immutable/JSON-safe enough for tracing but must not influence
  schedule selection. Keep it out of `KernelInfo` executable fields and
  candidate hash unless a later explicit contract adds it.
- The generic fallback must not be skipped merely because no `PrimitiveRouteEntry`
  exists. It attaches source truth from `TensorFact`; only a genuinely dense
  source is labeled dense/`none`. Missing source facts remain unavailable.
- No target/backend conditional belongs in this API. Metal-specific offset and
  graph admission facts remain MR1 backend evidence, orthogonal to identity.
