# Phase-enabled attention: negative register-pressure result

## Result

The phase-enabled one-pass attention probe is structurally valid but does not pass resource admission.

| Backend | VGPR | SGPR | LDS | Scratch | Spills | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| HIP physical code object | 197 | 26 | 10752 B | 0 B | 0 | over the 192-VGPR admission limit |
| AMD ISA | over budget | - | - | - | - | register allocation rejects the candidate |

The prior non-phase microgate used 8704 bytes of LDS and also allocated 197 VGPRs. Phase mode therefore adds exactly 2048 bytes of LDS while reducing no VGPRs:

```text
10752 B - 8704 B = 2048 B
```

This is a negative result. It proves typed `StateHandle` publication/reload and lane-major local storage can coexist with the attention graph, but it does not prove a phase lifetime reduction and is not eligible for replay timing or promotion.

## Why publication did not transfer ownership

The phase-enabled path in `amd_gfx1100_q16_grid_hd128_loop_attention` still creates all three original register owners:

```text
mreg = DEFINE_REG float[8], id 9601
lreg = DEFINE_REG float[8], id 9602
creg = DEFINE_REG float[acc_blocks*8], id 9603
```

The `m` and `l` declarations remain authoritative throughout the loop:

1. `mi` initializes eight `mreg` lanes through `AMD_ATTENTION_LOOP_STATE(role="m", access="init")`.
2. `li` initializes eight `lreg` lanes through `AMD_ATTENTION_LOOP_STATE(role="l", access="init")`.
3. `om` and `ol` are first read from those registers through `rd(mreg, ...)` and `rd(lreg, ...)`.
4. Phase mode publishes those already-register-backed vectors into `phase_lds`; it does not replace their owners.
5. The reload supplies the row-softmax consumer, but `nm` and `nl` are written back to `mreg` and `lreg` through the original loop-state operations.
6. The next loop iteration again reads the register declarations before publishing another LDS copy.
7. Final normalization reads `lreg` again through `AMD_ATTENTION_LOOP_STATE(access="final_read")`.

The native AMD lowering makes the retained ownership explicit:

```text
role "m"   -> fixed base 72 -> v72:v79
role "l"   -> fixed base 80 -> v80:v87
role "acc" -> fixed base 8  -> v8:...
```

`lower_amd_attention_loop_state` selects those fixed aliases from the role alone. It has no phase-owned alternative. Consequently, adding `StateHandle(old_m)` and `StateHandle(old_l)` creates an LDS mirror while preserving the complete fixed-register recurrence.

## Exact source of the additional 2048 bytes

Phase mode declares:

```text
phase_lds: DEFINE_LOCAL float[512]
lane_stride: 16 float elements
old_m: lanes [0, 8)
old_l: lanes [8, 16)
wave lanes: 32
```

The allocation is therefore:

```text
32 lanes * 16 fp32 elements/lane * 4 bytes = 2048 bytes
```

The typed handles correctly distinguish the regions and the `qk_preload -> row_softmax` boundary. The error is not layout, overlap, wait ordering, or spilling. The error is that the handle owns only a copy, while `AMD_ATTENTION_LOOP_STATE` continues to own the recurrence.

## Required phase-owned loop-state ABI

The next implementation must make one owner authoritative. For phase-enabled `m/l`, that owner is the storage-backed `StateHandle`; no parallel register owner is permitted.

### State declaration

- Keep the accumulator `creg` declaration unchanged for this step.
- Do not emit `mreg` or `lreg` `DEFINE_REG` nodes when phase ownership is selected.
- Declare one local region with two non-overlapping typed handles:
  - `old_m`: scalar `float`, vector shape `(8,)`, lane stride 16, element offset 0.
  - `old_l`: scalar `float`, vector shape `(8,)`, lane stride 16, element offset 8.
- Bind both handles to the same generic directed boundary and generation. Phase names are identifiers, not backend behavior.

### Loop recurrence

For each KV iteration, including the first:

1. The previous iteration's state publication, or the initial `(-inf, 0)` publication, owns the current `m/l` values in LDS.
2. QK executes without either `m/l` value in its backward slice.
3. A typed wait orders the state publication before reload.
4. `StateHandle.reload` reconstructs exactly `float.vec(8)` for the row-softmax merge.
5. Row softmax computes `new_m/new_l`.
6. `StateHandle.publish(new_m/new_l)` commits the next iteration's state directly to LDS.
7. The loop `END` depends on that publication, making it the recurrence edge.
8. Final normalization reloads `l` from its handle after the last committed publication.

At no point may `old_m`, `old_l`, `new_m`, or `new_l` be routed through an `AMD_ATTENTION_LOOP_STATE` whose base is a `DEFINE_REG`.

### Ownership metadata

The loop-state descriptor needs a phase-owned form that carries or references all of the following:

- StateHandle identity: region, publish phase, reload phase, boundary ordinal, generation.
- Storage identity and capacity.
- Runtime lane owner, lane stride, and element offset.
- Exact scalar dtype and vector lane count.
- Access kind: initialization publish, iteration reload, iteration publish, or final reload.
- Logical role `m` or `l`, without implying a fixed physical register base.

The old register-backed form remains valid for non-phase kernels and for `acc`. A phase-owned `m/l` descriptor must fail closed if it also references a `DEFINE_REG`, and a register-backed descriptor must fail closed if it claims a storage-backed handle.

### Lowering contract

The phase-owned path must consume the generic publish/reload carrier before native loop-state fixed-alias selection. Fixed bases `72` and `80` are forbidden for phase-owned `m/l`. A backend that cannot preserve the typed LDS vector reload must reject phase ownership rather than silently recreating register state.

This describes a future compiler/lowering change only. No renderer change is part of this evidence commit.

## Falsifiable structural gate

Before compiling a phase-enabled candidate, its UOp graph must satisfy all of these conditions:

- Exactly one register declaration remains for loop state: `creg`.
- No `DEFINE_REG` of eight fp32 elements exists for `m` or `l`.
- Every `m/l` initialization, iterative write, iterative read, and final read is bound to one validated storage-backed `StateHandle`.
- The QK backward slice contains neither an `m/l` reload nor an `m/l` register declaration.
- Each row-softmax merge is dominated by the typed wait and reload.
- The loop recurrence and final drain depend on the corresponding typed publication.
- StateHandle ownership has no phase mismatch, shape mismatch, gap, logical overlap, or physical LDS overlap.

## Resource admission gate

Only after the structural gate passes should physical compilation run. The candidate may proceed to replay timing only if:

- HIP and AMD ISA both compile.
- VGPR allocation is at most 192.
- Scratch and spill counts remain zero.
- The expected 2048-byte `m/l` phase region is present and the total LDS allocation remains within the target limit.
- QK and PV WMMA role counts and full-output numerical gates remain unchanged.

If the graph removes the two register declarations but physical allocation remains 197, then `m/l` ownership was not the complete cause of the excess. The next diagnostic must inspect the exact physical live set at the final K-fragment load; adding another state copy is not a valid response.
