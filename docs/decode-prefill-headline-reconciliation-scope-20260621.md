# Decode / Prefill Headline Reconciliation Scope

Date: 2026-06-21

Owner: next executor

Status: scope only

## Context

Prefill policy integration is shipped:

- `PREFILL_V2=auto` is opt-in and VRAM-gated.
- `PREFILL_SERVER_PROFILE=1` is the server / long-prompt fast path.
- `PREFILL_REMAINDER_FIX` is default-on but only active under `PREFILL_V2`.
- Global default `PREFILL_V2` remains off.

The remaining owner call is whether to flip global default prefill from `PREFILL_V2=0` to `PREFILL_V2=auto`.

Separately, a new report claims:

> Default decode is 87.6 tok/s (matches the banked headline ~86) -- no regression, no errors, VRAM 4.8GB.

That conflicts with the current canonical decode report, which says default decode is about
`68.0 / 66.5 / 63.5 / 60.8 tok/s @ctx512/1024/2048/4096`, roughly `67%` of llama. The number `87.6` also appears
in older artifacts as a `decode_ms` value, not `tok/s`, so the unit and benchmark route must be reconciled before
the project headline changes.

## Goal

Produce one trustworthy headline table that answers:

1. What is current default decode speed, by context?
2. Is `87.6` a tok/s result, a ms/token value, a different route, a different context, or a stale artifact?
3. Did prefill policy integration change decode behavior?
4. Should the README/project headline still say decode is `~67% llama`, or should it move to the `~86-90 tok/s`
   class?
5. Is global `PREFILL_V2=auto` ready to become the default?

No implementation changes are allowed until this reconciliation is complete.

## Authority Rules

Use three timing classes and label them explicitly:

| class | use | authority |
|---|---|---|
| clean wall | promotion headline | `PROFILE=0`, no debug/profiling, median-of-5 or better |
| pinned diagnostic | cost attribution | peak-clock pin allowed, but cannot become user headline by itself |
| profiled split | attribution only | `PROFILE=1`, `ProfileGraphEvent`, rescaled only for family attribution |

Do not mix these classes in one row. Do not compare pinned diagnostic tinygrad against unpinned llama unless the
row says so.

## Phase 1 — Artifact Triage

Collect and classify every recent decode number currently being cited.

Required rows:

| source | value | unit | ctx | mode | clock | profile | route | headline-safe? |
|---|---:|---|---:|---|---|---|---|:--:|

Inputs to inspect:

- `docs/prefill-policy-integration-result-20260620.md`
- `docs/decode-attention-elementwise-result-20260620.md`
- `docs/decode-current-route-attribution-result-20260620.md`
- `docs/decode-role-tensor-kernel-attribution-result-20260620.md`
- `bench/qk-prefill-policy-integration/*`
- `bench/qk-decode-attention-elementwise/*`
- any artifact or commit that contains the claimed `87.6 tok/s`

Gate:

- PASS only if the `87.6` claim is traced to an exact artifact/command/commit and classified as either:
  - valid clean-wall tok/s;
  - valid but different route/context;
  - diagnostic-only;
  - unit error;
  - stale artifact.

## Phase 2 — Clean Decode Re-run Matrix

Run clean wall decode with current HEAD and no profiling.

Required matrix:

| row | env | ctx512 | ctx1024 | ctx2048 | ctx4096 | VRAM | notes |
|---|---|---:|---:|---:|---:|---:|---|
| default | unset prefill/decode env | | | | | | |
| prefill auto | `PREFILL_V2=auto` | | | | | | decode should match default |
| server profile | `PREFILL_SERVER_PROFILE=1` | | | | | | decode should match default |
| q8 opt-in | `Q8_FFN_HANDWRITTEN=1` | | | | | | opt-in only |

Measurement requirements:

- median-of-5 minimum;
- same prompt/model/token count across rows;
- report context lengths exactly;
- report whether clocks were auto or pinned;
- report VRAM peak;
- report output/tok0 equality for default vs prefill-policy rows;
- keep llama comparison in the same timing policy or label it as historical.

Gate:

- prefill policy rows must not regress default decode by more than `1%` at any context;
- if `87.6 tok/s` is reproduced, it must be clear which row/context produced it;
- if not reproduced, the old claim must be marked non-headline.

## Phase 3 — Prefill Default Decision Matrix

Validate the owner call: global default `PREFILL_V2=auto` vs current default off.

Required matrix:

| hardware class | expected `PREFILL_V2=auto` decision | pass condition |
|---|---|---|
| unknown VRAM | off | no OOM, clear reason |
| <=16GB | off | no fp16 allocation attempt |
| 24GB RX 7900 XTX | on | fits Q4 + fp16-covered + KV + margin |
| explicit `PREFILL_V2=0` | off | explicit override wins |
| explicit `PREFILL_V2=1` | on or clear OOM | explicit override wins |

Also validate CLI/profile behavior:

- default long-prompt path prints a fast-path hint, but does not auto-enable.
- `PREFILL_V2=auto` prints the selected decision.
- `PREFILL_SERVER_PROFILE=1` implies `PREFILL_V2=auto` and concrete-KV when it fits.
- `PREFILL_REMAINDER_FIX=0` reverts the shifted-chunk route.

Gate:

- recommend flipping global default only if 16GB/unknown stay off and 24GB stays on with no decode regression.

## Phase 4 — Update Canonical Tables

Only after Phases 1-3 pass, update:

- `docs/README.md`
- `bench/README.md`
- `docs/prefill-policy-integration-result-20260620.md` if the decode sentence is stale
- the current handoff doc

The new headline must use this shape:

| area | default | opt-in/server | llama-relative status | next action |
|---|---|---|---|---|
| prefill | | | | |
| decode | | | | |

Rules:

- Prefill and decode must be separated. Prefill policy does not imply decode speed changes.
- If decode remains `~67% llama`, keep the current attention/elementwise fusion scope as the frontier.
- If default decode is now `~87.6 tok/s`, rerank decode work because the remaining gap is much smaller.
- If `87.6` was a unit/context mistake, explicitly state that in the result doc to prevent future reuse.

## Phase 5 — Result Doc

Write:

- `docs/decode-prefill-headline-reconciliation-result-20260621.md`
- artifact JSON under `bench/qk-headline-reconciliation/`

Minimum result sections:

1. `87.6` provenance and verdict.
2. clean wall decode table.
3. prefill policy/default decision table.
4. updated project headline table.
5. owner recommendation on global `PREFILL_V2=auto`.

## Stop Conditions

Stop and do not update headlines if:

- `87.6` cannot be traced to an artifact or exact command;
- clean decode reruns are contaminated by profiling/debug/clock pinning without labels;
- llama comparison is not using comparable timing authority;
- prefill policy rows change decode output or regress decode by more than `1%`;
- 16GB/unknown VRAM behavior cannot be simulated or directly verified.

## Expected Outcomes

Likely outcomes:

1. `87.6` is valid but refers to a different row/context/measurement policy. Keep decode headline conservative.
2. `87.6` is a unit mixup from `decode_ms`. Mark stale/non-headline and keep `~67% llama`.
3. `87.6 tok/s` is a real current clean-wall default result. Update the decode headline and rerank decode fusion ROI.

In all cases, prefill remains kernel-solved and policy-shipped. The only prefill decision is whether `auto` becomes
the global default.
