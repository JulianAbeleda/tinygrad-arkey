# AMD ATT Primitive Attribution Result

Date: 2026-06-19

Artifacts:

- Probe: `extra/qk_att_primitive_atlas.py`
- Result: `bench/qk-att-primitive-atlas/result.json`
- Decode artifact: `bench/qk-att-primitive-atlas/decode_mmvq.json`
- Prefill artifact: `bench/qk-att-primitive-atlas/prefill_nonmatmul.json`
- Summary: `bench/qk-att-primitive-atlas/summary.md`

## Verdict

`PASS_ATT_PRIMITIVE_ATTRIBUTION`, with interpretation `ATT_USABLE_NOT_DECISIVE_FOR_INMODEL_GAP`.

The R1-P2 ATT replay path now works as a reusable primitive-local interval tracer. It can body-attribute:

- a smoke HCQ kernel;
- tinygrad's native Q4_K coop decode primitive;
- the imported llama Q4_K MMVQ artifact;
- tinygrad's pp512 SDPA/prefill-attention surface.

This clears the observability/tooling blocker. It does **not** by itself change the measured performance map, because
ATT is instruction/resource evidence, not timing authority.

## Gates

| Gate | Result |
|---|---:|
| v2 AQLprofile helper export | PASS |
| adapter smoke body attribution | PASS |
| decode MMVQ body attribution | PASS |
| prefill attention body attribution | PASS |

## Trace Results

| Target | Body-like packets | `VALUINST` | `INST` | `WAVESTART` | `WAVEEND` | Notes |
|---|---:|---:|---:|---:|---:|---|
| smoke body | 172,368 | n/a | n/a | n/a | n/a | adapter positive control |
| tinygrad Q4_K coop attn-o | 168,693 | 163,872 | 3,442 | 1,022 | 1,534 | native tinygrad decode primitive |
| imported llama Q4_K MMVQ attn-o | 163,942 | 153,537 | 8,598 | 1,534 | 2,053 | mature MMVQ contract through HCQ |
| tinygrad pp512 SDPA surface | 135,442 | 98,740 | 31,619 | 467 | 467 | prefill non-matmul attention surface |

All traced targets had:

- start packet sync success;
- target execution success;
- stop packet sync success;
- nonzero SQTT output;
- decodable body packets.

## Decode Meaning

This pass proves ATT can see both sides of the decode primitive comparison:

1. tinygrad native Q4_K coop;
2. imported llama Q4_K MMVQ.

That is enough to retire "we cannot inspect HCQ bodies" as a blocker. The next decode question is now narrower:

```text
Can we trace the exact in-model role interval and compare it against the standalone/native/imported surfaces?
```

This pass does **not** yet explain the full `76% standalone -> 44% in-model` collapse, because it traces primitive
surfaces rather than a full in-model JIT/graph role interval with program-identity joins. It supports the existing
conclusion, though: the problem is not that tinygrad bodies are invisible or untraceable; the remaining gap is at the
MMVQ lifecycle/contract-preservation boundary.

Concrete next decode step if continuing:

- join ATT intervals to the HCQ attribution ledger for one real model role (`blk.0.attn_output`, then `ffn_down` or
  `lm_head`);
- record whether the in-model role uses the same program identity and wave/resource behavior as the standalone surface;
- only then decide between runtime/cache identity, scheduler/resource work, or closing native decode primitives.

## Prefill Meaning

ATT can body-attribute the pp512 SDPA surface, so prefill attention is inspectable.

It does not overturn the current prefill conclusion:

- matmul is already near the fast-kernel ceiling in-model;
- the non-matmul residual is led by attention plus smaller glue/runtime costs;
- the known prefill opportunity is small/project-level, not a missing GEMM primitive.

This reinforces `docs/prefill-nonmatmul-missing-primitive-result-20260619.md`: prefill's remaining gap is not a
missing primitive analogous to Tensile. The only candidate left is attention/runtime overlap class work, and its Amdahl
payoff is modest.

## What Changed

Before this result:

```text
tinygrad HCQ ATT body attribution was blocked at packet plumbing.
```

After this result:

```text
ATT body attribution works on real tinygrad decode/prefill primitive surfaces.
```

So the toolchain state changes from `blocked` to `usable`. The performance story does not change yet.

## Remaining Boundaries

- ATT is sampled by WGP/SIMD target; short kernels still need enlarged or repeated dispatches.
- The trace is not timing authority; continue to use clock-controlled A/B and PMC for speed claims.
- Full in-model role attribution still needs program-identity joins across TinyJit/HCQ graph replay.
- The current PM4 relocation logic is a gfx1100 ATT command-shape patcher, not a general PM4 relocation framework.

## Decision

Do not reopen packet plumbing. The next useful decode work is a **role-joined in-model ATT pass** for one high-share
MMVQ role. If that still does not produce a concrete label for `44%` in-model HBM, close ATT as explanatory-only and
return to the measured build choices: spec-decode or project-level MMVQ integration.

