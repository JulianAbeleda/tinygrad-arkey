# Budget Reduction Lanes 3-5 Scope - 2026-07-10

Baseline after the route-plan prune:

```text
python3 sz.py
AUTHORED budgeted lines: 27938 / 31000
tinygrad/llm: 3129 lines in 16 files
```

This scope covers the next three budget-reduction lanes only:

1. Q4/Q6 primitive install-loop dedupe.
2. `model.py` admission / fp16-overlay arithmetic extraction.
3. AMD ISA / prefill machine-search extraction from framework core.

## Shared Rules

- Do not delete live runtime behavior to hit a line count.
- Completion proof is deletion or moved ownership, not adding an additional abstraction beside the old path.
- Runtime selection must route from facts, quant, role, shape, architecture, and policy data; not model-size/name labels.
- CUDA/NV thin runtime/renderer support stays.
- Each slice must report `python3 sz.py` before/after.

## Lane 3 - Q4/Q6 Primitive Install Dedupe

### Current Shape

Budgeted file:

```text
tinygrad/llm/qk_primitives.py: 446 raw lines
```

The Q4 and Q6 install loops are parallel:

- tensor row filtering: quant type, 2D, `.weight`;
- route-plan or generated-policy lookup;
- byte alignment;
- module lookup and bias checks;
- budget reservation;
- shared/sidecar storage materialization;
- debug skip/install summary.

Differences that must remain explicit:

- Q4 type `typ == 12`, storage dtype `uint32`, bytes per 256 = `144`, align `4`;
- Q6 type `typ == 14`, storage dtype `uint16`, bytes per 256 = `210`, align `2`;
- Q4 generated families allow `q4_k_packed_u32` and `q4_k_packed_u32_direct`;
- Q6 generated family is `q6_k_packed_u16`;
- Q4 supports `kernel_mode` and direct-out validation;
- Q6 has no direct-out kernel mode today;
- Q4 `q4_ondemand` nonpersistent mode exists; Q6 does not.

### Target

Introduce one narrow internal iterator/helper that validates rows and resolves route/generated policy metadata, then keep
the final Q4/Q6 constructor calls explicit.

Do not create a generic "quant primitive framework" that hides Q4/Q6 semantics.

Expected budget win: `50-100` lines if the helper removes real duplication. If the helper saves fewer than about
`30` lines, stop and keep the duplication.

### Acceptance

- Q4/Q6 generated-policy behavior is unchanged.
- RoutePlan remains the only non-generated install-policy source.
- Focused tests:

```bash
python3 -m pytest test/unit/test_model_route_plan.py test/unit/test_llm_decode_routes.py test/unit/test_qk_route_purity.py
```

- E2E smoke after integration:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/llm/model_e2e_bench.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --id qwen3-8b-qk-dedupe \
  --max_context 512 --decode-tokens 8 --warmup-skip 2 --prefill 0 \
  --out /tmp/qwen3-8b-qk-dedupe.json
```

## Lane 4 - `model.py` Admission Extraction

### Current Shape

Budgeted file:

```text
tinygrad/llm/model.py: 1205 raw lines
```

`Transformer.from_gguf` has two admission blocks:

- path-GGUF pre-load metadata path;
- fallback post-load path for preloaded Tensor / no pre-load metadata.

Both compute:

- block count and trained context;
- Q4 bytes / fp16 overlay estimate;
- KV bytes per token;
- prefill-v2 auto decision;
- direct-packed availability;
- resident fp16 admission;
- chunked overlay resident bytes;
- prefill peak bytes and flash scratch;
- KV-quant/ring support;
- final `resolve_max_context_admission` call and reporting.

### Target

Move the duplicated arithmetic into pure helpers in `tinygrad/llm/admission.py` or a small sibling module. Prefer a
single dataclass result such as:

```text
AdmissionInputs / AdmissionPlan
```

The helper may take already-derived facts; it must not import `Transformer`, GGUF loaders, or route code.

Keep in `model.py`:

- loading metadata/state tensors;
- applying `_set_prefill_v2`;
- printing the final user-facing admission line;
- assigning `_kv_quant`, `_ring_admitted`, `max_context`.

Expected budget win: `80-150` lines if both blocks collapse to shared helper calls.

### Acceptance

- Existing admission semantics preserved for:
  - path GGUF auto;
  - explicit context;
  - preloaded Tensor fallback;
  - KV quant tier;
  - ring tier;
  - `PREFILL_V2=auto`;
  - `PREFILL_CHUNKED`.
- Focused tests:

```bash
python3 -m pytest test/unit/test_llm_context_admission.py test/unit/test_model_route_plan.py test/unit/test_llm_decode_routes.py
```

- Direct generate smoke:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python - <<'PY'
from tinygrad import Tensor, GlobalCounters
from tinygrad.llm.model import Transformer
model, kv = Transformer.from_gguf('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf', 512)
Tensor.manual_seed(0)
g = model.generate([0])
for _ in range(3):
  GlobalCounters.reset()
  print(next(g))
PY
```

## Lane 5 - AMD ISA / Prefill Machine-Search Extraction

### Current Shape

Largest relevant budgeted files:

```text
tinygrad/renderer/isa/amd.py:         2106 raw lines
tinygrad/codegen/opt/postrange.py:     758 raw lines
tinygrad/codegen/late/devectorizer.py: 642 raw lines
tinygrad/codegen/__init__.py:          317 raw lines
```

Core appears to carry prefill/QK-specific search machinery:

- prefill local-stage policy;
- DBUF peel / DBUF role scoping;
- WMMA proof tags and proof-key reuse;
- D3A stage/audit markers;
- K-major phase/stage-steal logic;
- prefill-specific devectorizer predicates for buffer ids `990/991/993`;
- QK named codegen hooks currently wired through core.

### Target

Do not start by moving large renderer code. Start with an inert extension boundary and byte-identical proof harness:

1. Identify every prefill/QK-specific branch in the four files above.
2. Classify each branch:
   - generic AMD substrate, must stay;
   - QK/prefill policy, move candidate;
   - debug/proof marker, move/delete candidate;
   - unknown, needs proof.
3. Add one narrow extension interface if needed, but only if it replaces a real branch.
4. Move one low-risk predicate/policy through the interface.
5. Prove emitted output unchanged for representative routes.

Expected budget win:

```text
first safe slice: 50-150 lines
full staged extraction: 900-1500 lines
```

### Non-Negotiable Proof

Before any meaningful code movement:

```bash
python3 -m pytest test/unit/test_amd_isa_extension_interfaces.py \
  test/unit/test_backend_intrinsic_lowering_allowlist.py \
  test/unit/test_prefill_route_spec.py \
  test/unit/test_qk_route_purity.py
```

For moved renderer/lowering logic:

- capture representative emitted source/hashes before and after;
- include direct 2x2, 4x2, 2x4;
- include k-major 2x2, 4x2, 2x4, 4x4 when locally available;
- no route attribution changes.

### Stop Conditions

- If the change cannot prove byte-identical emitted source/hash, stop at audit/doc.
- If the extension layer adds more lines than it removes in core, stop.
- If CUDA/NV code paths need special-case changes, stop and rescope.

## Integration Order

1. Land Lane 3 if it reduces lines and focused tests pass.
2. Land Lane 4 only after direct generate smoke passes.
3. For Lane 5, land only the first byte-identical slice; otherwise land the audit and proof matrix only.
4. Run:

```bash
python3 -m pytest
python3 sz.py
```

5. Ratchet the budget only after the tree is green.
