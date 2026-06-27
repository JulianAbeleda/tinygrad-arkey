# Pure machine-search gap audit result

Canonical tool: `extra/qk_pure_machine_search_gap_audit.py`.

Canonical artifact: `bench/qk-pure-machine-search-gap/latest.json`.

## Verdict

`PURE_MACHINE_SEARCH_PARTIAL__DECODE_GEMV_CLOSE__DECODE_ATTENTION_AND_PREFILL_STILL_NOT_FULLY_SEARCH_OWNED`

Overall score: **67 / 100**.

This is the top-level decode + prefill wrapper. It does not replace the decode-attention child audit; it consumes it and adds decode GEMV plus prefill.

## Scores

| Area | Score | Status |
|---|---:|---|
| Decode attention | 60 | Generated stack transfers, but manual flags and owned default performance remain. |
| Decode GEMV | 85 | Tracked Q4_K GEMV roles are effectively pure/search-generated under BubbleBeam G3. |
| Prefill | 62 | Eightwave baseline is stable and fast, but search provenance/aggressive-bound closure is incomplete. |

## Current benchmark meaning

| Area | Current evidence |
|---|---|
| Decode attention | Full generated stack improves over generated no-stack by +72.6% / +77.1%, but owned is still 3.15x / 15.13x faster at ctx512 / ctx4096. |
| Decode GEMV | BubbleBeam/FutureSight measured route is near current lifecycle baseline and tracked Q4_K roles are classified pure generated in the GEMV scope. |
| Prefill | Current baseline/eightwave is stable across ctx512..8192 at ~3570 tok/s; long-context growth is fixed in the graph lane. |

## CTX regression explanation

The ctx regression is **not prefill**. Prefill graph/eightwave is flat across long context.

It is also not "flash attention is bad" in general. The regression is the generated decode flash-style implementation exposing the outer `b`-block online-softmax carry. The owned/tuned kernel handles the same broad flash/split-KV idea with a pressure-aware structure that keeps the long-context slope controlled.

| Hypothesis | Verdict |
|---|---|
| Prefill regresses with ctx | Refuted by long-context hardening and canonical baseline. |
| Flash attention itself causes the gap | Too broad / inaccurate. |
| Generated flash decode tile exposes outer block-loop carry | Current best explanation. |
| Owned kernels were good because they avoided this cost structurally | Yes. |

## Missing pure-search pieces

| Area | Missing or not search-owned |
|---|---|
| Decode attention | Occupancy guardrail, split-aware hotloop audit, LDS-staged outer-`b` split-combine primitive, pressure-aware scheduling. |
| Prefill | Search provenance binding for eightwave/current baseline, aggressive-bound proof, attention/copy decomposition blind spot. |
| Shared | Manual flags need BubbleBeam/FutureSight candidate provenance before they count as pure machine search. |

## Next actions

| Rank | Action |
|---:|---|
| 1 | Build decode-attention occupancy guardrail and split-aware hotloop audit. |
| 2 | Add/search LDS-staged outer-`b` split-combine primitive. |
| 3 | Bind prefill eightwave/current baseline to explicit search provenance. |
| 4 | Put manual flags/primitives into BubbleBeam/FutureSight candidate provenance. |
