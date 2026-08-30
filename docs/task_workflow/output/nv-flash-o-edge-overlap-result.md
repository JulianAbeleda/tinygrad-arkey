# Flash-to-O edge overlap result

## Verdict

Llama does not usefully overlap its O GEMV body with Flash. It launches the
dependent grids early, lets them reside and wait, and begins their dependent
work at the producer boundary. Tinygrad already has zero device gap from
score to combine and from combine to O, so there is no corresponding dispatch
bubble to recover.

No production recovery is booked.

## Matched edge test

The llama arm joins seven steady CUPTI graph replays to the retained
`%globaltimer` wait-exit ring. It contains 252 Flash/combine/O layer edges.
The tinygrad arm reads 8,910 production edges from the retained device-ledger
profile.

| Edge observation | llama | tinygrad |
| --- | ---: | ---: |
| score body | 5.152 us | 5.120 us |
| combine grid launch relative to score end | 4.673 us early | exactly adjacent |
| combine useful wait exit relative to score end | 0.210 us early | no wait phase |
| O-quant grid launch relative to combine end | 5.664 us early | no O-quant kernel |
| O-quant useful wait exit relative to combine end | 0.206 us early | no O-quant kernel |
| O-GEMV earliest useful wait exit relative to Flash end | **0.584 us late** | exactly adjacent |
| combine-to-O device gap | hidden inside wait residence | **0.000 us** |

For llama, the earliest O-GEMV wait exit was after Flash combine on every one
of the 252 edges. Its p10/median/p90 offsets were +0.440/+0.584/+0.726 us.
The O grid still appeared about 6.05 us before its quant producer ended, but
that interval was dependency residence, not O weight-streaming work.

For tinygrad, both adjacent gaps were 0.000 us at p10, median, and p90 over all
8,910 edges. Its score, combine, and O bodies were serialized without a dead
device interval.

## What transfers from llama

The transferable principle is to launch a dependent grid early, perform only
producer-independent setup, and wait immediately before the first dependent
load. That hides launch/residency overhead. It does not make two dependent,
bandwidth-heavy bodies execute for free.

At this particular boundary tinygrad has already removed the launch gap, and
it also avoids llama's separate O-quant kernel. The only llama-like useful
tail is roughly 0.21 us/layer at score-to-combine and combine-to-quant. The
former is the already-tested last-CTA combine class; the latter does not map to
tinygrad's FP16 O route. Even treating 0.21 us/layer as fully recoverable is
only about 7.6 us/token across 36 layers, before integration costs.

Deeper Flash-to-O segmentation is therefore not a transfer of llama's
mechanism. It is a new algorithmic topology that must pay for reduced O
parallelism, synchronization, and shared partial state. The tested exact
spellings did not pay those costs.

## Evidence

- `docs/task_workflow/evidence/nv-flash-o-edge-overlap-20260828/summary.json`
- `docs/task_workflow/evidence/nv-llama-useful-body-h1-20260821/`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/device-ledger/production.profile.jsonl`
- parser/calibration authority: `extra/llm_research/decode/nv_llama_useful_body_h1.py`
