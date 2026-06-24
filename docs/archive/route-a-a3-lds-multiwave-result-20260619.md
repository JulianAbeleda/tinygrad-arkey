# Route A / A3 LDS multi-wave result (2026-06-19)

Executed the first A3 gates from `route-a-a3-lds-multiwave-scope-20260619.md`.

Verdict: **P0_PASS_P1_BLOCKED_MMU**.

The single-wave LDS plumbing works. The first multi-wave LDS GEMM faults structurally before correctness or timing can
be evaluated. Do not proceed to P2 pipeline/tuning until P1 address/base mapping is fixed.

## Artifact

- `bench/qk-codegen-wmma/route_a_a3_lds_multiwave.json`

Source under test:

- `extra/gemm/rdna3_wmma_matmul.py`

Note: that source file was already dirty before this result was recorded. This result records execution only; it does
not stage or commit that in-progress implementation.

## P0 — LDS tile smoke

Command:

```bash
LDSTILE=1 GEMM=0 DEV=AMD PYTHONPATH=. python3 extra/gemm/rdna3_wmma_matmul.py
```

Result:

| check | result |
|---|---:|
| device time | `19.36us` |
| relative RMSE | `0.000209` |
| verdict | PASS |

Interpretation:

RDNA3 `DEFINE_LOCAL` + `global_load` -> `ds_store` -> `s_barrier` -> `ds_load` -> `v_wmma` -> global store works for
a single wave/tile. The basic LDS/barrier/lgkm plumbing is not the blocker.

## P1 — multi-wave LDS GEMM

Commands:

```bash
LDSGEMM=1 DEV=AMD PYTHONPATH=. N=2048 M=2048 K=2048 CNT=10 python3 extra/gemm/rdna3_wmma_matmul.py
LDSGEMM=1 DEV=AMD PYTHONPATH=. N=128 M=128 K=128 CNT=1 python3 extra/gemm/rdna3_wmma_matmul.py
```

Both fail with MMU faults:

| shape | result |
|---|---|
| `2048x2048x2048` | `MMU fault: 0x7B4F50DC3000 | NotPresent=1 ReadOnly=1 NoExecute=0 imprecise=0` |
| `128x128x128` | `MMU fault: 0x7689C0EA6000 | NotPresent=1 ReadOnly=1 NoExecute=0 imprecise=0` |

The smallest legal shape also faults, so this is not a large-shape loop-bound issue. It is structural in the P1
multi-wave LDS/global address path: likely global base/kernarg use, cooperative-load address math, or C-store address
mapping. GPU recovery smokes passed after both faults.

## Decision

Stop A3 at P1 for now.

Next valid step:

- build a smaller **multi-wave store-only / load-only LDS probe** that isolates kernarg/base mapping and cooperative
  global address math before reattempting full GEMM;
- only after P1 correctness passes should P2 double-buffering/software-pipeline work start.

This preserves the principle boundary: correctness before timing, and no tuning while the kernel can fault.
