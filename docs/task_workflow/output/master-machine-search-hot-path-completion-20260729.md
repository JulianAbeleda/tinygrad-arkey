# Master machine-search hot-path completion — 2026-07-29

## Outcome

The production LLM hot path is now self-contained under `tinygrad/llm` and has no runtime import from
`extra.llm_research`. The selected prefill, quantized decode, and flash-decode implementations retain the exact
promoted configurations produced by the BoltBeam + BubbleBeam/FutureSight search workflow. Unsupported or declined
routes use ordinary tinygrad graph execution.

For this project, **machine searched** means the search system explored, measured, ranked, and selected the
performance-sensitive configuration. Human-authored search spaces, compiler primitives, descriptors, validation
gates, and promotion decisions do not change that classification. `Tensor.custom_kernel` is an execution mechanism,
not evidence of a handwritten fallback.

## Branch boundary

| Branch | Production runtime | Handwritten specialized fallback/oracles | Search development |
|---|---|---|---|
| `master` | yes | no | no |
| `dev` | yes | yes, qualification-only | retained |
| `exp` | yes | yes, qualification-only | retained |

Production-code baseline tips before this completion record:

- `master`: `bfd1d0544df741d2892685542bd5c30886dd89a9`
- `dev`: `6f8cbea6903061d5e71b1ab2342b67005a636afd`
- `exp`: `bc7cf8d001bc340ae7af63764751b68cb72d91bb`

Master removed all 97 tracked files under `extra/llm_research` plus research-only audit/test consumers: 124 files and
28,092 lines in the final prune commit. Dev and exp retain the Q4/Q6 direct-packed oracle specifications at
`extra/llm_research/prefill/q4k_prefill_route_spec.py` and
`extra/llm_research/prefill/q6k_prefill_route_spec.py`.

## Verification

- Master production, fallback, CLI, organization, and ownership matrix: **138 passed**.
- Dev focused production boundary matrix: **72 passed**.
- Exp focused production boundary matrix: **72 passed**.
- Master has zero tracked `extra/llm_research` files and zero production imports from that namespace.
- Dev and exp retain both direct-packed qualification oracles while their `tinygrad/llm` packages also have zero
  production imports from that namespace.
- The public `python -m tinygrad.llm --help` and metadata-only benchmark surfaces run from master.

Final allocator closure: master `60aff50d3f4bba47dbd8ac916b46461083f1237c` publishes the allocator fact consumed by
the selected-GGUF admission scan (4 KiB KFD, 2 MiB AM large-allocation tier). The expanded master matrix is
**141 passed**, and the exact memory-ledger matrix is **27 passed, 1 skipped**. Unknown backends still fail closed.

The 8B AMD recertification command was attempted from clean master under `/tmp/gpu-bench.lock` with the qualified host
environment. It did not reach model allocation or execute a kernel: the live probe returned
`No interface for AMD:0 is available`. Therefore this completion does not claim a new throughput number. The retained
README numbers remain explicitly labeled pre-migration comparison points. Rerun the documented 512/4096 commands when
the eGPU service is available; no code change or weakened admission guard is required.
