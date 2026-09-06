# NVIDIA current252 gate/up diagnosis index (2026-09-06)

## Method and current authority

The canonical investigation loop is `docs/what-makes-inference-fast.md` §0.5:
fresh production wall and device ledger, rank total contribution, state a causal
accounting claim, run the smallest exact falsifier, investigate a flat lever,
then require a clean repeated production bracket and a new ledger before
promotion.

The current route is current252: 252 generated projection mains/producers, 252
canonical weight bases, and zero V/down FP16 overlays. Commit `4ba03c6f1` fixes
the role classifier and retains its HCQ ledger in
`docs/task_workflow/evidence/nv-prefill-current252-hcq-20260906`. The observer
closes all 1,629 physical launches and attributes 19.886720 ms to gate/up,
9.764992 ms to Q/O, 9.704928 ms to down, 2.517824 ms to V, 3.347968 ms to
Flash, 3.208096 ms to vocabulary, and 2.361056 ms to K. Observer-bearing wall
time is mapping evidence, not endpoint authority.

The generated gate/up main is the U8, 170-owner Stream-K program
`q4_qo_streamk`, invoked 72 times with a five-buffer ABI (output, partials,
partial IDs, compact Q8 record, canonical Q4 weight) and followed by active
fixups. Its source tile is 20 KiB; CUDA reports 21,504 allocated shared bytes.
The actual default U8 spelling reports 255 registers, 80 stack bytes, and
21,504 shared bytes. The default-off fragment load-to-use research spelling
reports 255 registers and 88 stack bytes; the combined fragment/interleave arm
reports 225 registers and zero stack. These distinct facts must not be merged
when interpreting a new NCU capture.

## Historical counter hypothesis, with topology boundary

The authoritative historical chain is:

- `nv-prefill-gate-up-service-audit-20260829.md`
- `nv-prefill-gateup-schedule-locator-20260829.md`
- `nv-prefill-gateup-schedule-discriminator-20260829.md`
- `nv-prefill-gateup-schedule-counters-20260829.md`

The old three-buffer wide U4 body had the same useful IMMA count as the matched
llama body, but only 14.65% tensor duty versus 31.71%, 0.554 versus 0.807
eligible warps/cycle, 0.535 versus 0.324 long-scoreboard exposure, and 24.4%
more instructions. That supports an issue/latency-hiding hypothesis. It does
not transfer those counter values to current252: the current route uses U8,
170-owner Stream-K, a five-buffer ABI, and fixups, whereas the historical bridge
used the wide three-buffer body and different ownership.

## Indexed current experiments

| mechanism | accounting claim | binary/resource signal | production result | decision/evidence |
|---|---|---|---|---|
| U6/U10/U12 outer-K unroll | improve instruction scheduling | no sufficient winning endpoint signal | late minima 54.762/55.847/58.196 ms; all miss U8 current route | STOP; `nv-prefill-current252-gate-intermediate-unroll-20260906` |
| pointer restrict | improve alias scheduling | source-only scalar toggle | +1.267% | STOP; commit `67209b281` |
| unsliced fixup | reduce fixup launch/slice overhead | changes fixup only | +1.399% | STOP; commit `e14beb16f` |
| 40 KiB parity double buffer | remove recycle barrier | one barrier removed; shared tile doubled | 53.506390 vs controls 53.302259/52.979670 ms | STOP; `nv-prefill-current252-gate-double-buffer-20260906` |
| fragment load-to-use reorder | shorten Q4 global-fragment lifetime | shared unchanged; 255 registers, 88-byte stack | 52.832851 ms between controls 52.979670/53.315076; 0.314522 ms (0.592%) vs mean control | STOP below frozen 0.5 ms threshold; default-off `nv-prefill-current252-gate-fragment-ltu-20260906` |
| all shared loads at pack use | remove scalar shared-load live ranges | 254 registers, zero stack | 56.741611 ms, decisively slower | STOP; `nv-prefill-current252-gate-shared-ltp-20260906` |
| fragment-only shared loads at pack use | reduce fragment scalar lifetime while retaining early scale loads | 255 registers, 88-byte stack | 53.215871 ms; no material win over adjacent promoted controls | STOP; same evidence directory |
| interleave eight WMMA results with 16 accumulator updates | cap live IMMA results while preserving each FP32 expression | 225 registers, zero stack, shared unchanged | 52.832851/52.734764/52.960819 ms bracket; 0.162071 ms vs mean control | STOP below frozen 0.5 ms threshold; `nv-prefill-current252-gate-interleave-20260906` |

Every completed model arm above requires the current252 census, canonical/read-
only weights, zero overlays, full logits/token comparison, and 20-cycle exact
replay. A resource improvement alone is not a performance result; the all-load
staging negative demonstrates this directly.

The broader negative corpus is indexed by
`nv-prefill-ranked-campaign-closure-20260829.md`,
`nv-prefill-substrate-test-ledger-20260829.md`,
`nv-prefill-h0.2-parity-ledger-20260829.md`, and the STOP list in
`nv-prefill-llama-path-theory-review-20260829.md`. Those documents prevent
repeating final-row pruning, scalar/metadata reorder sweeps, Q8 reuse below its
threshold, fused gate epilogue regressions, and unqualified cp.async/TMA claims.

## Missing discriminator and next gate

There is no current five-buffer U8 Stream-K NCU counter capture. The old NCU
bridge cannot launch the current cubin unchanged because its ABI and ownership
are different. Before another schedule family is tested, adapt the retained
bridge to the exact current five-buffer program and capture, under one matched
protocol:

- registers, stack/local traffic, shared bytes, and achieved occupancy;
- tensor and issue duty, eligible warps, long-scoreboard exposure;
- total instructions, DRAM/L2/shared traffic, and barrier stalls;
- exact main service with producer/fixup boundaries stated separately.

The interleave candidate advances only if its clean control/candidate/control
wall bracket wins and at least one causal current counter moves beyond the
declared noise threshold in the llama direction. Otherwise retain its resource
result as a negative and select the next mechanism from the refreshed current
counter delta rather than another scalar source toggle. Any full prefill parity
claim additionally requires a fresh matched llama endpoint; the historical
35--38 ms llama range is reference context only.

## Paused execution point

`extra/llm_research/prefill/nv_prefill_gateup_streamk_ncu_bridge.py` adapts the
retained CUDA-primary-context bridge to the exact five-buffer main plus active
fixup ABI. Current default and combined fragment/interleave sources/cubins were
built locally and report the resources above. The first G0 invocation stopped
before CUDA initialization because the inherited symbol parser required
`__launch_bounds__` on the fixup; the bridge now accepts either signature and
is CPU syntax/symbol checked. Per the user pause, no replacement GPU launch or
NCU run was started. Resume with the paired G0 control/interleave correctness
run, then profile only after both outputs match and all readonly/coverage gates
pass.
