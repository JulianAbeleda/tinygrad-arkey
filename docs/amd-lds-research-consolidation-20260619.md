# AMD LDS Research Consolidation

Date: 2026-06-19

Generator:

- `extra/qk_amd_lds_research_consolidation.py`

Artifact:

- `bench/amd-broad-backend-roadmap/lds_research_consolidation_result.json`

## Verdict

`LDS_RESEARCH_CONSOLIDATED_DO_NOT_LOOP`.

The old LDS docs and the new BB-5a.9 result are not contradictory. They are about different levels of the stack:

- old LDS work refuted **plain LDS tiling**, **hand multi-wave LDS GEMM tuning**, and **manual UOp prefetch** as
  bounded session levers;
- BB-5a.9 keeps open only **Tensile-class renderer capability**: staged LDS layout, wide LDS reads, software-pipelined
  K-loop, semantic waits/barriers, and resource policy.

Do not restart the closed variants.

## Current Authority

The canonical same-kernel evidence is BB-5a.8/BB-5a.9:

| fact | tinygrad captured | Tensile oracle |
|---|---:|---:|
| TFLOPS | `43.026` | `65.6` |
| WMMA | `64` `v_wmma` | `13810` `v_wmma` |
| LDS bytes | `0` | `1LDSB0` schedule |
| `ds_load_b128` | `0` | `9324` |
| `ds_store_b128` | `0` | `2144` |
| `s_barrier` | `0` | `2112` |
| scratch | `0` | no-spill oracle note |

Conclusion: the captured tinygrad authority kernel already uses WMMA and does not spill. The gap is not “make WMMA
appear” and not “fix scratch.” The open target is renderer-level staged LDS plus software-pipelined scheduling.

## Closed Rows

| row | status | do not reopen as |
|---|---|---|
| LDS expressibility | available | “can tinygrad express LDS at all?” |
| plain hand-LDS WMMA tiling | refuted | “try LDS tiling again” |
| multi-wave hand-LDS GEMM | refuted | “tune DBUF/PAD/BK/occupancy again” |
| POWN config sweep | refuted | “try more waves, bigger tiles, BLOCK_K, or no-LDS” |
| manual UOp prefetch | refuted | “write prefetch UOps and expect schedule movement” |

Provenance:

- `docs/amd-lds-tiling-existing-primitives-20260617.md`
- `docs/prefill-wmma-lds-tiling-result-20260619.md`
- `docs/route-a-a3-p2-p3-lds-refuted-20260619.md`
- `docs/prefill-own-wmma-kernel-result-20260619.md`
- `docs/prefill-codegen-software-pipeline-result-20260619.md`

## Open Rows

Only these remain open:

| track | target | minimum pass |
|---|---|---|
| B LDS layout | real authority-shape LDS tile allocation | nonzero LDS in ELF and DS traffic in disasm |
| B wide LDS reads | vectorized LDS reads | `ds_load_b128` feeding WMMA |
| C K-loop scheduler | two-stage global-to-LDS-to-WMMA loop | prologue plus steady-state alternating LDS slots |
| C waits/barriers | semantic waits over staged LDS traffic | correctness-preserving `vmcnt`/`lgkmcnt` waits and barriers |
| D resource policy | reject bad candidates before timing | no scratch/private spill or deterministic rejection |

The first candidate gate remains blocked until B/C/D produce a real staged authority-shape kernel. Q8 transfer remains
blocked until pure tinygrad prefill reaches `>=60 TFLOPS`.

## Reconciliation Rule

When a future LDS idea appears, classify it before building:

| if it is... | then |
|---|---|
| plain LDS operand tiling | closed by PWLT-A2 |
| multi-wave hand-LDS GEMM tuning | closed by A3 P2/P3 |
| wave/tile/BLOCK_K/no-LDS sweep | closed by POWN-1 |
| manual UOp prefetch | closed by CG-1 byte-identical ISA |
| real renderer-level staged LDS + software pipeline | allowed under BB-5a.10 |

This is the consolidation checkpoint before BB-5a.10. Build only the open renderer-capability tracks.
