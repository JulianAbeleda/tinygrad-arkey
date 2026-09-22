# NV L6 host-gap A/B at HEAD (2026-08-18)

Date: 2026-08-18
Branch: `nvidia-bringup-20260731`, HEAD `da4400e48`
Status: **measured. Bracketed B/D/B/D wall authority for the submit-ahead
decode route (L6), fresh at HEAD.**

## Result

Harness: `scratchpad/nv_host_gap_submit_ahead_bracket.py`, 40 samples/arm,
warmup 3, d512, Qwen3-8B-Q4_K_M, RTX 5090, bench-lock held.
Evidence: `docs/task_workflow/evidence/nv-host-gap-submit-ahead-bracket-head-20260818.json`.

| arm | median wall us/token |
| --- | ---: |
| baseline B1 | 4718.86 |
| submit-ahead D1 | 4700.12 |
| baseline B2 | 4698.31 |
| submit-ahead D2 | 4704.63 |
| **D vs B** | **-6.21 us** |

`engaged: true` (the promoted greedy + pingpong pair was captured and the
alias contract admitted), tokens bit-identical across all four arms, first
token stable.

## Reading

The ledger's L6 row carried a **~100.6 us ceiling** (host gap 269.0 vs llama
168.3 in the 08-17 exact account) and a ~213 tok/s ceiling at 1:1. The real
recoverable delta at HEAD is **-6.2 us (~+0.27 tok/s)**, not ~100.6 us. The
submit-ahead reorder engages and is token-safe, but the host gap is not
submit-order-bound at this HEAD: the ~100.6 us includes JIT capture/handoff
and single-sync overhead that this route does not remove.

## Verdict

L6 closes as a **measured-flat row**: engaged, correct, but worth ~6 us of
wall - far below the 100.6 us ceiling and below the +50 us promotion bar. It
joins L1 (closed), L2 (closed), L3 (structural), L5 (structural) as a
non-lever at HEAD. The remaining genuine upside rows are the two codegen /
search targets from the audit: searchable flash score shape and cheaper
packed-key argmax reduce.
