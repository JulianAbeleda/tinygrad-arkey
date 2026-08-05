# NV P5 sampler and feedback tail record

Date: 2026-08-05. Scope: native `DEV=NV`, Qwen3-8B-Q4_K_M, d512 greedy
decode. Status: **SUPERSEDED IN PART — sampler census remains valid; feedback copy solved by later ping-pong qualification.**

This census correctly rejected deleting the alias firewall from one capture.
It did not test alternating two non-aliasing captures. That later construction
qualified an 89--94 us/token redirect-on recovery while keeping the generic
firewall unchanged; see
`nv-decode-feedback-pingpong-record-20260805.md`. The sampler-chain ownership
and indivisibility findings below remain current.

The corrected greedy production route already preserves device-resident token
feedback: after a decode call, `generate` passes `out` directly as the next
`decode_input`. The host observes `sampled.item()` only to satisfy its public
streaming/yield API and to append the returned token. It is not an H2D feedback
path. Removing that read would change the observable generator contract; it is
not a cheap performance switch.

P1's reusable shadow also cannot be removed here. The captured decode graph
writes its feedback input, so the copy and dependency are an alias firewall.
P1 has already replaced allocation/reconstruction overhead while retaining the
copy; an elision requires a separate lifetime proof and a correctness-passing
wall A/B.

The qualified 948-node native DAG has one remaining device candidate:

| boundary | nodes | device duration |
| --- | --- | ---: |
| token-feedback prefix | 0--1 | 5.386 us |
| post-LM-head greedy sampler | 944--947 | 56.160 us |

The sampler is an indivisible serial reduction chain after Q6_K LM head node
943: `943→944→945→946→947`, with `944→946` as an additional dependency. The
four programs have durations 3.808, 39.456, 11.328, and 1.568 us. A candidate
must replace the complete chain with an included-cost, deterministic argmax
primitive; fusing only a suffix cannot remove the 39.456 us first reduction.

This confirms the ledger's 71.215 us vocab/sampler/feedback ranking but does
not book it as recoverable. No GPU work was run: no topology-reducing primitive
exists yet, and the expected ceiling is below the active boundary, shared-Q8,
and projection work. The fail-closed parser is
`extra/llm_research/decode/nv_sampler_feedback_tail_census.py`; it rejects any
capture that is not the qualified template or changes a required dependency.
