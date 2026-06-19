# q8 FFN fast artifact vs raw-code result (2026-06-19)

This executes the two-way scope in `q8-ffn-fast-artifact-and-codegen-transfer-scope-20260619.md`.

Question:

- **Path A:** can the fast HIP/clang-quality q8 kernels be normalized into HCQ-loadable artifacts with no HIP runtime
  in-process?
- **Path B:** is the current tinygrad-owned raw-code/COMGR form already good enough?

Verdict:

- **Path A: PASS as a fused gate/up lifecycle primitive.**
- **Path B: FAIL for performance in its current raw-code/COMGR form.**

## Path A — hipcc artifact through HCQ: PASS

Probe:

- `extra/q8_ffn_fast_artifact_probe.py`

Artifacts:

- `bench/q8-ffn-handwritten-oracle/hipcc_object_audit.json`
- `bench/q8-ffn-handwritten-oracle/fast_artifact_perf.json`

The loader problem was narrower than the earlier A2 redirect implied.

The header-free raw-C hipcc object emits unsupported `R_AMDGPU_REL32_LO/HI` relocations, matching the original
`unknown AMD reloc 10` failure. But the HIP-style oracle device kernel does not have that problem after LLD
normalization:

| object | loader status | relocation status |
|---|---|---|
| COMGR raw C baseline | loads | no relocations |
| hipcc raw C relocatable | not the chosen path | has unsupported REL32_LO/HI class |
| hipcc HIP-style oracle relocatable | loads | only supported REL64 in this run |
| hipcc HIP-style oracle linked by `ld.lld -shared` | loads | no relocations |

No HIP runtime is loaded in the tinygrad process (`libamdhip64.so` absent from `/proc/self/maps`).

## Path A performance

The correct lifecycle unit is:

`fused RMSNorm/q8 producer -> fused gate/up q8 MMVQ consumer`

not:

`producer -> separate gate consumer -> separate up consumer`

The q8 activation is token-local and shared by gate/up, so the natural research primitive is one producer and one
gate/up consumer launch.

Measured with `--warmups 8 --iters 24`:

| unit | median |
|---|---:|
| producer, 1024-thread HIP-style artifact | 21.26 us |
| fused gate/up consumer | 92.86 us |
| total lifecycle | **114.12 us** |
| target | <=129.2 us |

Correctness:

| output | max_abs |
|---|---:|
| producer fp output | 0.0 |
| gate vs q8 reference | 9.54e-7 |
| up vs q8 reference | 1.43e-6 |

So Path A clears the isolated lifecycle gate.

Separate consumers are still useful as a diagnostic:

| unit | median |
|---|---:|
| producer | ~21.5 us |
| gate consumer | ~56 us |
| up consumer | ~56 us |
| separate lifecycle | ~133-134 us |

That misses the strict lifecycle target, but only because it models two independent consumers instead of the shared
gate/up primitive.

## Path B — current raw-code/COMGR route: FAIL

Probe:

- `extra/q8_ffn_oneblock_route.py`

Artifact:

- `bench/q8-ffn-handwritten-oracle/oneblock_route.json`

The current tinygrad-owned raw-code/COMGR kernels are correct but too slow:

| unit | median |
|---|---:|
| producer | 30.00 us |
| gate consumer | 82.48 us |
| up consumer | 82.32 us |
| total lifecycle | **194.80 us** |
| target | <=129.2 us |

Correctness remains good:

| check | value |
|---|---:|
| one-block route vs graph q8 proxy max_abs | 0.00137 |
| one-block route vs graph q8 proxy mean_abs | 4.44e-5 |

This means the current raw-code route is not a model route candidate. It remains useful as a correctness/reference
asset, not as the fast primitive.

## Comparison

| path | ownership | lifecycle | verdict |
|---|---|---:|---|
| A: HIP-style artifact, separate consumers | external artifact | ~133-134 us | below consumer target but misses lifecycle |
| A: HIP-style artifact, fused gate/up | external artifact | **114.12 us** | **PASS** |
| B: current COMGR raw code | tinygrad-owned raw code | **194.80 us** | **FAIL** |

Path A is currently better by about `194.8 / 114.1 = 1.71x` on the q8 gate/up lifecycle.

## What this means

The q8 lifecycle idea is not dead. A2 failed because the loadable COMGR artifact did not preserve the HIP oracle
kernel quality and because it modeled gate/up as two separate consumers. Once the HIP-style artifact is normalized and
gate/up are fused as one lifecycle primitive, the isolated speed gate passes.

The remaining risk moves from "is the primitive fast enough?" to "can it be routed through TinyJit/HCQGraph in-model
without losing the win?"

## Next gate

Reopen Track A A3, but with the new primitive:

1. promote the probe-local hipcc+LLD artifact loader to a research-only helper;
2. route one FFN block as `producer + fused_gateup + existing ffn_down`;
3. make the route graph-capturable;
4. run W==D decode and dNLL with the flag off by default.

Gate:

- W==D decode speedup >=3%;
- dNLL <=0.01;
- no HIP runtime in-process;
- flag-off output unchanged.

Track B should not continue from the current COMGR raw source. Its next useful form is a codegen/ASM transfer that
matches the HIP-style artifact's producer and fused gate/up consumer. The passing artifact is now the oracle for that
transfer.
