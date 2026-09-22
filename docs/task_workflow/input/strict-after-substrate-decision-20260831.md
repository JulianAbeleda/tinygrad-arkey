# Strict-after substrate decision (2026-08-31)

## Verdict

`MICROGATE_PASS_FULL_DAG_REJECT`

The CUDA microgate proves an exact compiler dependency can order an address-dependent load without device synchronization. The current substrate is not accepted for the full Q6 DAG because compilation does not reach CUDA generation or SASS.

## Semantic contract

`value.strict_after(dependency)` returns `value` bit-exactly while requiring every consumer of the result to remain compiler/data-dependent on `dependency`.

- It is compiler ordering, not inter-thread synchronization.
- It emits no atomic, memory fence, `MEMBAR`, CTA barrier, or device barrier.
- It must be applied to the ordered load's address or index. Wrapping an already-loaded value is too late to order that load.
- The IR marker is backend-neutral. A renderer must explicitly implement a dependency-preserving lowering or compilation fails closed.

## Core and renderer changes

- `tinygrad/uop/ops.py`: defines the hashable `STRICT_AFTER` argument marker and `UOp.strict_after` API.
- `tinygrad/uop/symbolic.py`: excludes strict AFTER from ordinary AFTER dependency flattening.
- `tinygrad/uop/spec.py`: validates the narrow scalar value/dependency contract.
- `tinygrad/codegen/__init__.py`: rejects a strict edge before lowering when the renderer lacks `supports_strict_after`.
- `tinygrad/codegen/late/devectorizer.py`: preserves the marker while rewriting `GEP(AFTER(...))`.
- `tinygrad/renderer/cstyle.py`: dispatches strict AFTER to an explicit renderer implementation; ordinary AFTER remains an alias of source zero.
- `tinygrad/renderer/cuda.py`: opts CUDA in and lowers a 32-bit ordered index through a reversible two-XOR inline-PTX dependency.
- `test/unit/test_strict_after.py`: covers post-rewrite retention, the ordinary-AFTER negative control, generated source order, final SASS order, and absence of synchronization/spill instructions.

## Microgate evidence

The live dependency is independently consumed after the ordered store, so NVRTC cannot satisfy the fixture by deleting its producer. Strict AFTER remains the only ordering relationship.

Generated CUDA signature:

```cuda
float val0 = (*(data1_32+lidx0));
float alu0 = (val0+1.0f);
int strict_after_0 = ((int)(lidx0));
asm volatile(
  "xor.b32 %0, %0, %1; xor.b32 %0, %0, %1;"
  : "+r"(strict_after_0)
  : "r"(__float_as_uint(alu0))
  : "memory");
unsigned int val1 = (*(data2_32+strict_after_0));
*(buf0+lidx0) = val1;
```

Final `sm_120` SASS dependency chain:

```text
0x0060  LDG.E       R2, ...          dependency load
0x0080  FADD        R9, R2, 1        live dependency producer
0x00c0  LOP3.LUT    R7, R9, ...      identity dependency token
0x00d0  IMAD.WIDE   R4, R7, ...      ordered address calculation
0x0110  LDG.E       R0, ...[R4]      ordered load
0x0140  STS         [...], R0         consumer store
```

No `BAR`, `MEMBAR`, `LDL`, or `STL` is present. The focused microgate result was `3 passed`; the remaining output was three unrelated pytest configuration warnings.

Unsupported renderers fail before lowering with `RuntimeError: <Renderer> cannot preserve strict_after compiler ordering`. CUDA is currently the only opted-in renderer.

## Rejected probes

- Empty compiler assembly: `asm volatile("" :: "r"(dependency) : "memory")` produced the desired source order but no machine data edge. With a dead dependency, NVRTC removed its load. With the dependency independently live, NVRTC/PTXAS still placed the ordered load before the dependency load.
- Self move: a real `mov.u32 %0, %0` did not consume the dependency and NVRTC removed or detached the dependency producer. It therefore could not prove the ordered load was downstream of the token.

These failures rule out source sequencing and a compiler-only barrier. The accepted microgate works because the ordered address is in the dependency's machine-level def-use chain.

## Full-Q6 Gate 9 blocker

- Initial full-Q6 attempt ran for 8.6 minutes and was terminated in `simplify_valid/backward_slice`.
- The scalar-REG repair timed out after 240 seconds in the initial symbolic `_commutative_key -> norm(u.tuplize)` path.
- Neither attempt reached CUDA source generation, NVRTC, or SASS, so the microgate result cannot be promoted to a full-Q6 pass.

## Next minimal primitive requirement

Introduce a dependency identity that is opaque to canonicalization after semantic reachability has been established. Its representation must prevent `_commutative_key` and tuple normalization from recursively expanding the full dependency DAG while retaining the ordering edge through final lowering.

Before another full-Q6 run, add a large-DAG compile-scaling test that exercises the strict dependency with Q6-scale shared subgraphs and gates symbolic/rewrite time and graph growth. Passing the small source/SASS microgate alone is insufficient.

## Uncommitted path inventory

Strict-after substrate:

```text
tinygrad/uop/ops.py
tinygrad/uop/symbolic.py
tinygrad/uop/spec.py
tinygrad/codegen/__init__.py
tinygrad/codegen/late/devectorizer.py
tinygrad/renderer/cstyle.py
tinygrad/renderer/cuda.py
test/unit/test_strict_after.py
docs/task_workflow/input/strict-after-substrate-decision-20260831.md
```

Separate nested runtime RANGE work already present in the same uncommitted workspace:

```text
tinygrad/codegen/simplify.py
tinygrad/codegen/late/linearizer.py
test/unit/test_nested_runtime_range_lifecycle.py
```
