# M4 residual boundary fold probe record - ordinary-producer typed-view fold

Date: 2026-08-06
Status: **probe record, PASS at the CPU-only hermetic gate. Authorizes no
implementation, no route-record change, no promotion.** Branch boundary:
tinygrad `nvidia-bringup-20260731`, HEAD `3f0784e7b`. Authority:
`m4-variant-reopen-boundary-p0-scope-20260806.md` section 4, probe 1 (HARD STOP
section 7 applies: probes only).

## 1. Question

Can the o-proj `residual_add` slot
`epi_inputs["residual"][:, 0, :].reshape(N).cast(fp32)` (`decode_routes.py:97-98`)
fold to a zero-copy view of its ordinary producer under an EXTENDED typed-view
contract, without weakening the M5 combine contract? The answer is the
schedule itself: with the fold applied, the E_32_32_4_86a2-shaped opaque
boundary copy must vanish from the linearized graph; without the ABI it must
remain. Probe-owned predicate only; `kernel_program.py` / `decode_routes.py` /
`decode_kernels.py` are untouched.

## 2. What the real chain is

The residual producer is the previous block's output. With the epilogue open,
the block boundary is `@function(precompile=True, allow_implicit=True)`
(`model.py:599-600`), the model applies `.contiguous()` before returning
(`model.py:658`), and the runtime-activation wrapper sits on top. The
decomposed producer uop is
`MS(CONTIGUOUS(GETTUPLE(precompiled FUNCTION)))`; the residual chain is pure
movement (`RESHAPE(4096) <- RESHAPE(1,4096) <- MS <- CONTIGUOUS <- GETTUPLE <-
FUNCTION`). There is no CAST uop (the identity cast is elided). The walk stops
at `CONTIGUOUS` (not in `GroupOp.Movement`), and `base.src[0]` (the GETTUPLE)
reports `has_precompiled_output_identity() == True`.

Today the opaque boundary's `.contiguous()` materializes one E_32_32_4_86a2-
shaped copy per residual slot (matches the M4 census copy class). The layer-0
contrast: `tokens @ emb` fp16 upcast to fp32 is a CAST over a REDUCE with no
buffer identity and no precompiled-output identity; the extended contract must
reject it.

## 3. Probe and gate evidence

Probe: `extra/llm_research/decode/m4_residual_boundary_fold_probe.py`
(CPU-only, hermetic, no GPU). It rebuilds the exact producer->consumer UOp
chain, evaluates the probe-owned `extended_validated_typed_view` contract
(pure-movement leg check, offset-0 row-major view, dtype/numel preservation,
residual-slot opt-in gated to route_role `attn_qo` / kind `residual_add` /
dtype fp32), and schedules the consumer GEMV with and without the fold.

```text
form             validator                        without ABI   with fold
block_output     fold=true  base=CONTIGUOUS        1 copy        0 copies
layer0_embedding fold=false producer has no buffer/precompiled-output identity
plain_buffer     fold=true  base=BUFFER            0 copies      0 copies
```

With the fold, the folded residual binds the producer's flat `(4096,)` fp32
buffer directly into the GEMV PARAM list; zero copy-class calls remain in the
linearized schedule. Without the ABI, the exact `block_output` chain keeps the
single boundary copy. The layer-0 embedding form is the documented fail-closed
contrast: the extended contract rejects it (`producer has no buffer/
precompiled-output identity`) rather than folding a view with no buffer
backing. `plain_buffer` folds as the control (BUFFER identity).

Unit gate: `test/unit/test_m4_residual_boundary_fold_probe.py` - 10 tests,
all passing (real-chain purity, validator fires, copy-removal contrast,
fail-closed, layer-0 rejection, plain-buffer control, opt-in specificity
slot/role/kind/dtype, impure-permute rejection, M5 validator still rejects,
probe schema).

## 4. M5 contract untouched

`test/unit/test_m5_typed_boundary.py` - 27 tests, all passing on the same
tree. The M5 validator (`kernel_program.py:_validated_typed_view`) still
rejects this residual chain (base is not a declared typed AFTER), and the
probe's extended contract is a probe-owned dataclass, deliberately NOT a
`TypedViewRequest`, so no production ABI moved.

## 5. Verdict and boundary

- **PASS**: the ordinary-producer residual slot is a provably pure, offset-0
  view chain, and the fold removes the boundary copy class in the hermetic
  graph with zero materialization.
- Fail-closed holds: every mismatch (role, slot, dtype, impure movement,
  non-buffer base) rejects to the generic flat-buffer ABI.
- HARD STOP respected: no production code changed, no record changed, no
  promotion. The section 6 full gate (fixed-depth wall + census assertions +
  pins 3/3) is a separate, subsequent decision.

## 6. References

- `m4-variant-reopen-boundary-p0-scope-20260806.md` sections 2-4, 7
- `m4-decomposition-measurement-record-20260803.md` section 6
- `m5-typed-boundary-p0-implementation-record-20260803.md`
- `tinygrad/llm/decode_routes.py` (`_Q4KDecodeCandidate.execute`,
  `decode_routes.py:89-107`)
- `tinygrad/llm/decode_kernels.py` (`Q4KGEMVEpilogue` residual_add body)
- `tinygrad/llm/kernel_program.py` (`_validated_typed_view`)
