# S1 body-vs-gap split result (2026-08-22)

## Verdict

The S1 window (Q anchor end to O anchor start) is **mostly kernel body, not
launch/serialization**. But the 634 us S1 gap is not one thing either. It
splits into three additive terms, and llama's overlap is the largest.

| side | S1 exposure | S1 body in window | dead (+) / overlap (-) |
| --- | ---: | ---: | ---: |
| tinygrad | 1152.250 us | 1042.500 us | +109.750 us dead |
| llama | 517.916 us | 844.050 us | -326.134 us overlap |

So our S1 window is 90.5% body and 9.5% dead time. The gap is not dominated
by idle bubbles; it is dominated by llama packing its S1 work into a much
shorter window.

## The 634.334 us gap, decomposed per 36 layers

| term | total us | us/layer | share of gap |
| --- | ---: | ---: | ---: |
| llama overlap (hiding) | 326.134 | 9.059 | 51.4% |
| tinygrad exposed-body delta | 198.450 | 5.512 | 31.3% |
| tinygrad dead device time | 109.750 | 3.049 | 17.3% |
| sum | 634.334 | 17.620 | 100% |

The terms sum exactly. The body delta is "exposed in-window mass," not a
claim that we do more total work: llama's weighted S1 dependency cost is
actually larger than ours (664.924 vs 558.912 us). Llama does not remove the
work; it hides it behind anchors and overlaps it inside its own window.

## What this resolves

The fusion conversation was treating S1 as one binary choice between "body"
and "launch/serialization." It is neither cleanly. It is a three-way split:

- ~326 us is hiding/overlap, the one llama-only mechanism.
- ~198 us is exposed body difference inside the window.
- ~110 us is dead time, the only part that is strictly launch/serialization.

This also explains why the recent micro-tests did not move the wall. Each one
attacked a small slice of one bucket: K four-warp was ~17 us of body, RMSNorm
copy-free was ~17 us of body. Against a 634 us three-bucket gap, no single
micro-change is big enough to register.

The next test should be chosen against the largest single term: llama's
326 us of S1 overlap. That means asking which S1 families are actually
independent of the Q/O spine and can overlap, rather than proposing another
body optimization.

## Evidence

- `docs/task_workflow/output/nv-s1-body-gap-split-20260822.json`
- source: `docs/task_workflow/output/nv-weighted-inter-anchor-ledger-20260820.json`
- tool: `extra/llm_research/decode/nv_s1_body_gap_split.py`
