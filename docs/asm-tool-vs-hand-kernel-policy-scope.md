# ASM Tool vs Hand Kernel Policy

Date: 2026-07-08.

## Goal

Stop conflating "uses assembly" with "is a hand kernel."

The repo policy is:

```text
Generated/compiler routes may emit ASM.
Hand-authored full-kernel schedules are oracle/escape-hatch routes, not pure generated routes.
```

The practical target is:

```text
machine search over reusable compiler primitives
```

This is intentionally less brittle than "no ASM" and more disciplined than "write fine-tuned kernels by hand." Humans may
write backend/compiler primitives. Machine search chooses how to compose them into kernels.

## Classification

| Class | Definition | Default allowed? | Example |
|---|---|---:|---|
| `generated_backend` | Kernel topology comes from tinygrad graph, descriptor, or search; backend emits target ASM. | yes | `decode_q4k_g3_generated`, `decode_q6k_coop_generated`, `prefill_v2_scheduler_matmul_default` |
| `backend_asm_tool` | Reusable compiler/backend primitive for an instruction family. | yes | WMMA lowering, DS offset lowering, waitcnt policy |
| `compiler_primitive_spec_owned__asm_backend_atom` | Route/spec owns lifecycle data and primitive selection; a reusable backend atom may emit ASM for the selected primitive. | research/opt-in until strict generated proof | S10 LDS2 `ffn_gate/up` route |
| `asm_probe` | Temporary diagnostic hand ASM to learn hardware semantics. | no product default | WMMA/register/waitcnt probes |
| `asm_oracle` | Measured hand kernel kept as reference or escape hatch. | opt-in only | `PREFILL_GRAPH_GEMM=1` 8B prefill |
| `hand_kernel_product` | Hand-authored per-shape kernel schedule used as the shipped route. | only by explicit exception | any new route-local raw instruction stream |

## Preferred Compromise

The preferred route is not strict "pure from first principles." It is:

```text
search/spec owns:
  shape policy
  tile sizes
  role selection
  pipe vs LDS
  DBUF on/off
  wait policy
  primitive composition

compiler/backend primitives own:
  WMMA lowering
  b128 global/DS load-store lowering
  targeted waitcnt
  LDS staging operations
  DBUF scheduling idioms
  register/layout constraints
  DS offset folding
```

The forbidden case is a human-authored complete model-specific lifecycle:

```text
global loads -> LDS/VGPR staging -> waits -> WMMA loop -> epilogue stores
```

as a raw route-local instruction stream.

## Current 8B Prefill State

| Route | Classification | Evidence |
|---|---|---|
| `PREFILL_GRAPH_GEMM=1` / `prefill_pipe_role_selective_generated` | `asm_oracle` / escape hatch | Reproduces 8B pp512 around 5111 tok/s, but executes `extra/qk/prefill/wmma.py` through raw `Ops.INS`. |
| `PREFILL_GRAPH_GEMM=0` / `prefill_v2_scheduler_matmul_default` | `generated_backend` | Ordinary tinygrad scheduler/codegen path, slower but pure. |
| 2x2 LDS/DBUF generated work | `generated_backend` replacement path | Keep. This is the route to reproduce the hand oracle without a hand-authored kernel schedule. |

## Design Impact

This policy removes the wrong constraint.

We are not banning ASM. We are banning uncontrolled hand-authored full-kernel schedules from being called generated.

The replacement target for the 5k oracle is:

```text
machine search / schedule spec
  -> reusable compiler primitives
  -> AMD backend emits WMMA/LDS/DBUF/waitcnt ASM
  -> no route-local full-kernel Ops.INS instruction-list
```

## Keep / Stop

Keep:

- the 5k hand prefill route as an opt-in oracle and benchmark target,
- 2x2 generated WMMA work,
- LDS staging and DBUF work,
- backend-owned WMMA, DS, waitcnt, register-allocation, and address-lifetime primitives,
- machine-search policy over those primitives,
- purity/census audits that expose the distinction.

Stop:

- expanding raw `Ops.INS` prefill kernels as if they are the final generated design,
- naming a route pure because only its schedule selector is generated,
- treating reusable compiler primitives as hand kernels just because they emit ASM,
- deleting 2x2/LDS/DBUF work just because the hand oracle already exists.

## Done Criteria

- `pure_kernel_surface_audit.py` reports both `asm_usage` and `kernel_authorship`.
- `prefill_pipe_role_selective_generated` is classified as `raw_instruction_or_binary_injection` +
  `hand_authored_full_kernel_schedule`.
- Generated routes can remain pure even when their backend emits target instructions.
- Docs distinguish ASM as a backend tool from hand-authored kernel schedules.
- The project target is documented as machine search over compiler primitives, with hand kernels reserved for oracle or
  explicit escape-hatch use.
