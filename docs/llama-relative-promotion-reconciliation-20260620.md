# Llama-Relative Promotion Reconciliation

Date: 2026-06-20

Artifact:
`bench/qk-llama-promotion/reconciliation.json`

Command:

```bash
python3 extra/qk_llama_relative_promotion_reconciliation.py
```

Verdict:
`PASS_LLAMA_RELATIVE_PROMOTION_RECONCILIATION`

## Purpose

This is the banked-wins reconciliation in the useful sense: what should be promoted, kept promoted, held as
research, or blocked when measured against the llama baseline.

It covers prefill, decode, and attention separately. Do not collapse these into one Tensile or one decode story.

## Promotion Summary

| Area | Candidate | Status | Llama-relative result | Promotion decision |
|---|---|---|---|---|
| prefill | concrete-KV WMMA dependency-free baseline | promoted baseline | `47-49%` llama pp512 | keep as dependency-free baseline; do not claim parity |
| prefill | external Tensile FFN+q/o route | policy-ready | `86-87%` llama pp512, byte-identical | promote only if vendored `.co` dependency policy is accepted |
| prefill | native Tensile-class codegen transfer | blocked native project | current P8 candidates are `~18-21 TFLOPS` vs `>=60` gate | run PTM-1 before choosing a native row |
| decode | banked default decode stack | promoted default | `68.2/66.4/60.7 tok/s` at ctx512/1024/4096, about `~67%` llama | keep as default authority |
| decode | q8 FFN handwritten/artifact route | hardened opt-in candidate | W==D `+5.1-6.3%`, multi-window `max dNLL 0.002225` | keep default-off behind `Q8_FFN_HANDWRITTEN=1`; default-on remains policy |
| decode | native q8 scheduler/renderer | blocked roadmap-only | no `>=30us` attributed feature | do not start from prefill/Tensile evidence |
| decode | MMVQ contract preservation/source import | project-level option | potential `~1.187x` decode from llama-like contract preservation | promote only as funded project |
| attention decode | Q4_K attn_q/o coop + flash stack | promoted default | included in reproduced `~67%` llama decode line | keep promoted |
| attention decode | `decode_attention_v3` WMMA/GQA V-reuse | deep-codegen candidate | projected `+4-10%` short ctx, `+12-36%` ctx4096 | not promoted; scope only if taking codegen risk |
| attention prefill | score-free/reuse-free flash prefill | closed | correct but not performant | do not promote |
| spec decode | T-cheap verify forward | closed | loses current speed model | do not promote |

## What To Promote

Keep promoted:

- decode default stack and W==D measurement authority;
- concrete-KV WMMA as the dependency-free prefill baseline;
- current decode attention defaults: Q4_K attn_q/o coop and flash-decode stack.

Policy-ready:

- external Tensile prefill route. This is the only immediate large llama-relative promotion candidate. It is not a
  measurement problem anymore; it is an external artifact policy decision.

## What Not To Promote

- native Tensile-class tinygrad codegen before PTM-1;
- standalone LDS;
- current P8 LDS/no-LDS candidates;
- native q8 scheduler/renderer from prefill-only evidence;
- q8 artifact route as default-on; it is now hardened opt-in only;
- score-free flash prefill;
- speculative/T-cheap verify.

## Required Gates

Native Tensile/prefill gate:

- PTM-1 same-harness authority bridge;
- then choose one row: software-pipelined K-loop, spill-free resource policy, or timing/launch correction.

Decode transfer gate:

- q8 same-binary primitive ablation;
- `>=30us` timing-grade movement;
- W==D quality unchanged;
- packed q8 format and lifecycle preserved;
- role-joined gate/up evidence.

q8 artifact promotion gate:

- Q8P-1..Q8P-6 passed on 2026-06-20;
- accepted promotion is hardened opt-in, not default-on;
- default-on requires explicit maintainer/user acceptance of a lossy external-artifact route.

Attention gate:

- for short decode, current attention rows are already promoted;
- for long context, accept a long-context target first;
- for `decode_attention_v3`, treat it as deep codegen, not a bounded promotion.

## Decision

There are three distinct promotion paths:

1. Accept external Tensile `.co` for prefill and promote toward `86-87%` llama.
2. Fund project-level decode MMVQ/source-contract work to move the `~67%` llama decode line.
3. Keep dependency-free native work scoped to PTM-1 before any new P8/Tensile implementation.

Do not mix these paths. They have different risk, policy, and evidence gates.
