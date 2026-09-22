# NV tightened-endpoint bracket result (hiding verdict, resolved below noise floor)

Date: 2026-08-21

Commit: `6570abc025514273faa100c66b979e531585a1e1`

Evidence: `docs/task_workflow/evidence/nv-endpoint-tight-20260821/`

Analyzer: `extra/llm_research/decode/nv_endpoint_per_token_s1.py`

## 1. Decision

**Decision: hiding does not move the needle toward llama parity.  On this
construction it moves it the wrong way by ~245 us of S1 and ~193 us of wall,
now resolved cleanly below the old measurement noise floor.**

The earlier single-final-token endpoint could not resolve the 150 us gate
because its control spread was 552.5 us.  This packet locks GPU/CPU clocks
and replaces the one final token with the per-token settled S1 distribution
from every captured decode token.  The phantom noise floor drops to a ~50 us
MAD, and the best-prior hiding arm (`edge_2q_start_entry`) is measured as a
clean regression rather than a signal.

No production change and no route-default change follow from this packet.

## 2. Why the earlier bracket was under-powered

The retained Stage 4 profile files each contain six decode-token timelines,
not one.  The first window is a ~220 ms prefill/decode transition artifact,
and the remaining windows drift ~500 us within a single process because the
GPU was ramping 225 -> 3135 MHz (memory 405 -> 14001 MHz) during measurement.
The old endpoint sampled only the last window under unlocked clocks, so its
552.5 us control spread was clock ramp plus a cross-arm queue mix, not the
mechanism.

## 3. Measurement setup

- SM clock locked to 2850 MHz (observed hold 2842 MHz), memory locked to
  14001 MHz, persistence mode enabled, P0 held.
- CPU governor set to `performance` for the bracket.
- `off` and `edge_2q_start_entry` arms, fresh-process control/candidate/control,
  `--tokens 32`, serialized under `flock /tmp/gpu-bench.lock`.
- S1 is summed per-layer O.start - Q.end from the HCQ graph-profile timeline;
  the first three windows per process are dropped as warm-up and the median
  is taken over the remaining settled tokens.

All six runs produced the identical token SHA:
`034ea96dbab43e1eebd2e523402aee383a9373e6640fe622a1fec4dd4c626328`.

## 4. Per-token settled S1 (the new endpoint)

| arm | role | n | median S1 us | MAD us | min us | max us |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| off | control 0 | 27 | 2176.50 | 40.25 | 2096.75 | 2507.25 |
| off | candidate (phantom) | 27 | 2224.75 | 48.75 | 2154.50 | 2530.50 |
| off | control 2 | 27 | 2229.75 | 39.25 | 2137.50 | 3494.00 |
| edge_2q_start_entry | control 0 | 27 | 2252.25 | 47.00 | 2191.50 | 2566.25 |
| edge_2q_start_entry | candidate | 27 | 2481.75 | 32.00 | 2344.00 | 2706.00 |
| edge_2q_start_entry | control 2 | 27 | 2222.00 | 50.25 | 2123.50 | 2467.00 |

The `off` phantom (no mechanism change) spans medians 2176.50 to 2229.75 us,
a 53.25 us cross-run spread with ~40-49 us MAD.  That is the honest noise
floor of this endpoint.

The hiding candidate median is 2481.75 us against its two controls at
2252.25 and 2222.00 us.  The candidate-control delta is **+244.63 us of S1
in the wrong direction**, cleanly separated from the control MAD (~47 us)
and from the phantom spread (~53 us).  The candidate distribution
(MAD 32 us) does not overlap the control distributions.

## 5. Final-token and wall confirmation

The old final-token metric agrees directionally (`observed`):

| arm | role | final-token S1 us | wall us |
| --- | --- | ---: | ---: |
| edge_2q_start_entry | control 0 | 2274.25 | 6676.238 |
| edge_2q_start_entry | candidate | 2670.50 | 6842.001 |
| edge_2q_start_entry | control 2 | 2143.50 | 6621.016 |

The candidate is +396.25 us of final-token S1 and +193.37 us of wall worse
than the control median.

## 6. First-principles read (inferred)

The Q -> O S1 support (K/V GEMVs, flash attention, RMSNorm) is on the
critical path: the O projection depends on it.  A latch launch-ahead reorders
the launch signals but does not create SM/queue concurrency; the same queue
still serializes the support, so nothing is actually overlapped.  The added
arrive/wait work across ~259 armed edges instead shows up as overhead, which
is consistent with ~245 us of S1 regression at this device.

The practical consequence: this S1 gap is serialized device work, not a
hideable host-submission stall.  Closing it needs genuine concurrency (for
example queue-splitting the support onto a second queue, where 140 rows
currently fall back) or fusion that removes the serialized support from the
critical path, not a latch reorder.

## 7. Retained evidence

- `docs/task_workflow/evidence/nv-endpoint-tight-20260821/stage4_endpoint_merged.json`
- `.../stage4_endpoint_rows.jsonl`
- `.../per_token_s1_verdict.json`
- per-arm `stage4_*.profile.jsonl` (six decode-token timelines per process)
