# Low-sync spec decode — PRODUCTION verdict (Phase 8): NO-GO (0.24x); host-sync-bound 2026-06-18

Integrated the proven low-sync spec loop and measured against the REAL production baseline (perf=high, clock
ramped). **Correctness PASSES (greedy byte-exact); production speed FAILS (0.24-0.26x << 1.20x gate).** The
algorithm + low-sync proposal graph are proven; the production speedup is not achieved due to the host-overhead
wall. Default decode untouched; nothing routed.

## Phase 0 — production baseline (valid)
`model.generate` tight loop @ perf=high (MCLK 1249): **62.8-83.1 tok/s** (prompt-dependent), 398 GB/s; cli
`--warmup --benchmark` = 81 tok/s. In the banked range. (Earlier ~9 tok/s was a host-bound *manual* harness +
idle MCLK — NOT the production path.) This is the only valid denominator.

## Phase 3-4 — production integration result
SPEC_DECODE greedy loop (proposal graph + verify T=K+1 + host accept + KV self-correction), perf=high:
| K | spec tok/s | vs production | greedy exact | accept/pass |
|---|---|---|---|---|
| 4 | 14.9 | **0.24x** | ✓ | 1.28 |
| 2 | 21.7 | **0.26x** | ✓ | 1.78 |

**Greedy byte-identical confirmed.** No per-pass recompile. But FAR below the 1.20x gate.

## Phase 6 — bottleneck attribution (measured)
Per-pass (K=4): propose 21.2ms | verify 21.3ms | accept-read 43.5ms | **total 86ms** for ~1.3 tokens. GPU work
is only ~18ms/pass; the rest (~68ms) is **host/sync latency**:
- The loop does **≥4 serial syncs/pass** (propose realize, verify realize, 2× tolist), each ~10-20ms *exposed*
  because the python loop is fully serial (realize → read → python → next dispatch).
- The production decode hits 62-83 tok/s because `model.generate` **pipelines** (1 cheap `.item()`/token; the
  host work overlaps the next dispatch). The spec loop's accept decision breaks that pipelining.
- Combining the 2 accept reads → 1 did NOT materially help (0.24→0.26x) — confirming the propose+verify realizes
  are themselves the syncs, not just the reads.

**Classification: Phase 6 B+D** — host accept/read AND serial-sync structure. This is the SAME host-overhead wall
the campaign keeps hitting (decode is GPU-bound only in the pipelined production loop; explicit serial loops
expose ~10-20ms/dispatch), here in an accept-dependent-serial-loop form that's intrinsically hard to pipeline.

## Verdict: NO-GO (do not ship); algorithm proven, production needs a fused spec graph
- Correctness: greedy byte-exact ✓. Low-sync proposal graph (Phase 4) ✓. KV protocol ✓.
- Production speed: **0.24-0.26x — FAIL.** The serial per-pass sync latency dominates.
- To realize the algorithm's ~1.4x GPU-work potential (per-pass GPU ~18ms for ~2 tokens → ~115 tok/s ceiling if
  host-free) requires a **fused one-sync-per-pass spec graph** (draft K + target verify + on-device accept in ONE
  capture, returning only accepted_count + tokens) or async dispatch pipelining. That is a deep runtime build
  (Phase D) — the two-model fused graph + on-device accept.

## Roadmap impact
Spec decode was ranked the highest-EV "beat llama" path because it's orthogonal to the kernel walls. It IS
orthogonal, but it hits the **same host-overhead wall** — the production decode is host-free via pipelining, and a
spec pass needs host involvement (accept) per ~2 tokens, which (in a serial loop) costs more than the target
passes it saves. **The "beat llama on 8B" goal now hinges on a fused/pipelined spec pass** (one sync/pass with
on-device accept) — the single remaining lever, and a substantial runtime arc. Until then, banked decode stays
~62-83 tok/s (production) and the low-sync algorithm is proven-correct-but-not-yet-faster.

## Files
`[test]` `extra/qk_spec_decode_lowsync.py`, `bench/qk-spec-decode-production/baseline.json`; `[docs]` this. No
kernel/model/default changes.
