# Phase 0 progress (2026-08-22 session 21:39Z)

Status: in progress. No production edits made.

## Locked fresh wall endpoints

| endpoint | wall us/token | tok/s | token evidence |
| --- | ---: | ---: | --- |
| tinygrad W-method (`decode_runtime_overhead`, ckpts 512, nmeas 40, reps 5) | 4742.989 | 210.84 | SHA dbd3026bb808... across all 5 reps |
| llama (`llama-bench -p 512 -n 20 -d 512 -r 5`) | 4029.796 | 248.23 | n_gen 20 |
| **fresh gap** | **713.193** | | |

The locked historical gap was 729.430 us. The fresh same-session gap is
16.2 us smaller, driven by both endpoints running slightly faster this
session.

## Fresh tinygrad native node ledger

`full_token_dag_capture --capture --depth 512` produced 874 pre-split nodes
(first-token linear) and a 596-node steady-decode window:

| quantity | value |
| --- | ---: |
| steady decode node_sum | 4675.168 us |
| prior audit fresh node_sum | 4677.920 us |
| agreement | -2.752 us |

## Open items before the baseline ledger can close

1. `union` and `overlap` for the steady window need a clean contiguous
   capture. The retained `HCQ_GRAPH_PROFILE_JSON` last window (records 70-74)
   has inter-group timestamp gaps (span ~600 ms vs node_sum 4.675 ms), so its
   interval union cannot be read directly from this capture.
2. Wall-method discrepancy: this W-method wall (4742.989 us) is ~95 us higher
   than the prior audit's submit-ahead control wall (4647.86 us). The fresh
   baseline must pin which route is the production default before host_gap is
   derived from `wall - union`.
3. llama PDL-on and PDL-off nsys traces for the fresh session are still
   required (762-node topology, true interval union).

## Next

Re-run a clean steady-window tinygrad profile capture and capture llama
PDL-on/PDL-off in the same session, then close the baseline ledger
(node_sum / resident_union / resident_overlap / host_gap).
