# NV decode two-capture feedback qualification record

Date: 2026-08-05. Target: native `DEV=NV`, Qwen3-8B-Q4_K_M,
d512 greedy decode. Status: **qualified wall win; closed-default.**

## Construction

A single captured rollout has a fixed sampled-token return allocation. Feeding
that allocation into the same capture on the next token creates an
input/output alias, so `CapturedJit` correctly inserts its defensive shadow
copy. The candidate alternates two captures instead: A always reads B's fixed
return and writes A's, then B reads A's and writes B's. The generic firewall is
unchanged. The host still calls `sampled.item()` for the public streaming API.

Both captures proved distinct NV returns, identical input contracts, and zero
written-input shadows. Every generation starts at phase zero. A contract miss
disables the candidate and returns subsequent calls to the ordinary path.

## Gates

| callify redirect | arm | programs | pre-graph 4-byte copy | contract |
| ---: | --- | ---: | --- | --- |
| 0 | greedy control | 946 | present | n/a |
| 0 | ping-pong | 946 | absent | admitted, shadows `[0,0]` |
| 1 | greedy control | 875 | present | n/a |
| 1 | ping-pong | 875 | absent | admitted, shadows `[0,0]` |

At both redirect settings, greedy and ping-pong produced the same eight tokens
and exact full-logit SHA-256
`31e5cc2c03aa706f873397d35f54776d915727347df5220c5ae4a02fc14a46ed`.
Every sampled token equaled the full-logit argmax.

## Reverse wall brackets

Each process used three included samples of 16 settled tokens after six
capture/warm tokens. Token hashes were identical across every sample.

| redirect | A1 greedy | B ping-pong | A2 greedy | B recovery vs A1 | B recovery vs A2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.542271 | 5.438011 | 5.541361 | 104.260 us | 103.350 us |
| 1 | 5.463597 | 5.374517 | 5.468711 | 89.079 us | 94.194 us |

The decisive composed result is therefore an 89--94 us/token recovery. The
redirect reduces the isolated ping-pong benefit by roughly 10--15 us; the two
routes are not treated as additive. The graph itself has unchanged program
count: the recovery is specifically the pre-graph alias-firewall copy path.

The implementation and harness remain experimental. No policy or default is
promoted by this record.

## d4096 direct resident-KV regression

The composed redirect-on route was also qualified at d4096 without repeatedly
prefilling 4096 tokens.  One model was loaded with `max_context=4608`; all 36
physical fp16 KV caches had exact shape `[2,1,8,4608,128]`, occupied 36 unique
buffers, and were initialized by setup-only zero stores.  A full-span maximum
reduction returned `0.0` for every cache before admission.  Each independent
arm restored that identical resident state outside the measured window.

The harness then invoked the normal production decode entry points directly
with `start_pos=4096`, flash decode enabled, actual prior sampled-token
feedback, and ping-pong slots alternating `0,1`.  Six excluded calls populated
positions 4096--4101 and completed both captures; only positions 4102--4117
were timed.  Zeroing, cache verification, capture, and synchronization were
not included.  This synthetic all-zero prefix changes KV values, not decode
geometry, storage, program selection, feedback ownership, or host `item()`
semantics.

The diagnostic control and candidate each retained four complete
`[1,151936]` logit rows.  Arrays were bit-identical (`max_abs_diff=0.0`) with
SHA-256 `9438ddcb3c57c1b52046822218a534e9a48e69559f58ed44a1e25c6c762f3ba4`;
sampled tokens were `[256,256,256,256]` and equaled argmax.  Both diagnostic
and timing ping-pong pairs admitted
`distinct_fixed_returns_and_read_only_inputs`, with distinct fixed NV int32
`[1,1]` returns and written-input shadows `[0,0]`.

Same-process reverse A/B/A results, three independent 16-token samples per
arm:

| arm | samples (ms/token) | median | ping-pong recovery |
| --- | --- | ---: | ---: |
| A1 greedy | 6.157631, 6.154603, 6.154055 | 6.154603 | -- |
| B ping-pong | 6.070201, 6.067459, 6.068675 | 6.068675 | 85.927 us vs A1 |
| A2 greedy | 6.166465, 6.165393, 6.170447 | 6.166465 | 97.790 us vs A2 |

All nine included token streams had the same SHA-256
`2286e6b7c9eea06d84516678ab6093a903b875f472fec5b46f5516ebeb46cf82`.
The d4096 result therefore reproduces the d512 composed recovery within the
same approximately 86--98 us/token band.  It supports the causal claim that
the saved alias-firewall path is depth-independent CPU/pre-graph work; it does
not claim prompt-quality numerics for the synthetic prefix and does not change
the closed default.

## Depth regression

The accepted redirect-on construction was also checked at d2048
(`max_context=2304`) with the same 16-token timing window. Token hashes were
identical in every arm:
`9e6664fd1d67a6124e786daaa1d895bdb64b972c3991c54dd5fcc6cea16f6881`.

| depth | A1 greedy | B ping-pong | A2 greedy | control midpoint | recovery |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 5.757520 | 5.663911 | 5.737715 | 5.747617 | **83.706 us/token** |

A1 and B each contain two samples; the closing control contains one because
the original long outer run was interrupted after B and A2 was rerun
separately. This row is a positive depth non-regression, not a new d512 ledger
credit. The completed d4096 direct resident-KV gate is recorded above.
