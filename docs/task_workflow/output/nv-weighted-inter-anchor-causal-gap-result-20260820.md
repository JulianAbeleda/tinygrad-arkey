# NV weighted inter-anchor causal gap result (2026-08-20)

Status: measurement result, not a promoted performance change. No runtime
files were modified; the changes for this scope are measurement tooling only.

## 1. Headline

Unprofiled wall on the shared RTX 5090 session:

| route | us/token |
| --- | --- |
| tinygrad control (HEAD `6570abc02`, `DEV=NV`) | 4723.214 |
| llama.cpp (`ac4cddeb0` build 9592, CUDA, flash-attn on) | 4005.709 |
| gap | 717.505 |

The device-timeline contribution to that gap is 839.723 us, and the
host/launch residual is -122.218 us. The decisive segment is S1
(Q end -> O start): tinygrad exposes 1152.25 us there against llama's
517.916 us, a 634.334 us contribution.

Cause: llama overlaps the Q->O support work (norm, rope, quant, flash)
behind its MMQ anchors and with itself, and PDL launch-completion edges let
its device timeline compress below its logical critical path. Tinygrad
exposes 1042.5 us of support mass in the S1 windows plus all 109.75 us of
the token's dead device time, and hides zero support work behind anchors.

The K/V GEMV branch is not the cause. It is off the critical path in both
implementations (critical-path mass 0.0 in each) and nearly equal in exposed
device time: 298.75 us tinygrad vs 288.057 us llama.

## 2. Wall to device ledger

| row | tinygrad us | llama us | tiny minus llama us |
| --- | ---: | ---: | ---: |
| unprofiled wall | 4723.214 | 4005.709 | +717.505 |
| device union (profiled) | 4732.500 | 3892.777 | +839.723 |
| host / launch residual | -9.286 | +112.932 | -122.218 |

The tinygrad host residual is negative because its `PROFILE=1` device
timestamps absorb profiler tax, so this row is a lower bound, not a
measured host win. 839.723 - 122.218 = 717.505 exactly.

The device gap decomposes into rows that sum exactly:

| device gap row | tinygrad us | llama us | delta us |
| --- | ---: | ---: | ---: |
| Q/O/gate-up/down anchor union | 2943.750 | 2998.617 | -54.867 |
| per-layer support exposed | 1418.750 | 593.012 | +825.738 |
| vocab tail | 370.000 | 303.740 | +66.260 |
| interval overlap accounting residual | - | - | +2.592 |
| device union total | 4732.500 | 3892.777 | +839.723 |

Llama's anchor bodies are larger (the -54.867 us row); the entire gap comes
from tinygrad's exposed support and vocab tail. The 2.592 us residual is
boundary overlap accounting between the anchor/support/tail interval sets
and keeps the ledger an exact sum; it is not unassigned serialization time.

## 3. S0-S4 segments

Exposure totals over 36 layers, with the weighted dependency totals:

| segment | tinygrad exposure | llama exposure | delta | tinygrad weighted | llama weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | 181.000 | 31.299 | +149.701 | 179.744 | 219.326 |
| S1 | 1152.250 | 517.916 | +634.334 | 558.912 | 664.924 |
| S2 | 186.000 | 32.896 | +153.104 | 186.016 | 221.212 |
| S3 | 0.000 | 15.937 | -15.937 | 0.000 | 29.856 |
| S4 | 170.250 | 30.884 | +139.366 | 169.120 | 218.111 |
| tail after vocab | 56.250 | 0.000 | +56.250 | - | - |

S4 of layer L is the same device interval as S0 of layer L+1; the tiling
check counts it once. Segment rows do not map one-to-one onto the
support-exposed reconciliation row for the same reason plus the separate
tail. S3 is the one segment where tinygrad wins outright: its GLU is fused
into the down epilogue while llama runs a separate G->D kernel window.

Weighted totals are the duration on the longest dependency path between
anchors. In S1 llama's weighted cost is actually larger (664.924 vs
558.912 us); its exposure is smaller because launch-completion overlap
compresses the device window. The S1 difference is scheduling and overlap,
not less work.

## 4. S1 mechanism

Family mass inside the S1 exposure windows:

| family | tinygrad in-window mass us | llama in-window mass us |
| --- | ---: | ---: |
| gemv_kv | 298.750 | 288.057 |
| quant_provider | 0.000 | 132.768 |
| rope | 0.000 | 127.998 |
| rmsnorm | 186.750 | 93.535 |
| kv store/get | 0.000 | 75.453 |
| flash_score | 266.750 | 74.559 |
| flash_combine | 105.750 | 51.680 |
| residual | 129.750 | 0.000 |
| reduce | 54.750 | 0.000 |
| total | 1042.500 | 844.050 |
| window exposure | 1152.250 | 517.916 |

Tinygrad's S1 windows contain at least 109.75 us of dead device time,
which equals the token's entire span-minus-union dead time. Every other
segment is fully covered. Llama's S1 families overlap each other by
326.134 us, which is why 844.05 us of work fits in a 517.916 us window.

Hidden behind anchors (intersection of the family union with anchor
intervals, all five anchor kinds):

| family | llama hidden us |
| --- | ---: |
| quant_provider | 372.444 |
| rmsnorm | 153.759 |
| flash_combine | 138.463 |
| flash_score | 99.391 |
| total | 764.057 |

Tinygrad hides 0.000 us behind anchors in every family. Its support work
occupies exclusive device time between anchors; llama's runs during the
MMQ anchors.

Critical path vs device span:

| side | critical path us | device span us | nodes |
| --- | ---: | ---: | ---: |
| tinygrad | 4249.216 | 4842.250 | 596 |
| llama | 4443.435 | 3901.205 | 762 |

Llama's logical critical path exceeds its observed span by 542.230 us:
consumers start before producers logically complete through PDL
launch-completion edges. Tinygrad's span exceeds its critical path because
off-path support fills the gaps between spine nodes.

## 5. Mechanism labels

| label | site | evidence |
| --- | --- | --- |
| `FUSED_INTO` | GLU into down epilogue | tinygrad S3 exposure 0.0 us; llama 15.937 us |
| `OFF_PATH` | K/V projections | critical-path mass 0.0 in both; 298.75 vs 288.057 us exposed |
| `OVERLAPPED` | llama norm/quant/flash | 764.057 us hidden behind MMQ anchors; tinygrad hides 0.0 us |
| `OVERLAPPED` | llama S1 support | 326.134 us of in-window family overlap |
| `OVERLAPPED` | llama vocab output reduction | tail after vocab 0.0 us; tinygrad 56.25 us |

## 6. Ranked legal build list

Ranked by recoverable end-to-end wall on this route. Zero-cost ceilings are
ideal upper bounds with full critical-path recomputation after removal, so
alternate-path takeover is included; they are not claims that a legal
change recovers that amount.

| rank | family / change | mechanism | node mass us | CP mass us | zero-cost ceiling us | legal ceiling us | measured A/B us | confidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | rmsnorm: copy-free native fp16 | shorter body, no cast/copy | 324.608 | 234.528 | 234.528 | see A/B | +12.506 to +17.141 | measured |
| 2 | residual: elementwise add/mul fusion | fold into GEMV epilogue | 218.944 | 148.736 | 148.736 | unknown | none | simulated |
| 3 | reduce: output-reduction folding | llama folds the reduce in-kernel | 234.272 | 203.680 | 203.680 | unknown | prior brackets insufficient | simulated |
| 4 | vocab tail | fold post-vocab norm/quant/reduce | 313.632 | 313.632 | 313.632 | small | measured transfer too small | measured, ledger only |
| 5 | flash_score / flash_combine | llama overlaps, bodies are near-equal mass | 267.328 / 105.056 | 267.328 / 105.056 | 267.328 / 105.056 | none known | none | inferred |
| 6 | quant_provider | tinygrad already below llama exposure | 30.976 | 30.976 | 30.976 | small | none | inferred |
| 7 | gemv_kv | off-path in both, matched mass | 298.400 | 0.000 | 0.000 | none | none | measured, no recovery |

Row 1 detail: arms A (ffn) +12.506, B (attn) +16.604, C (attn,ffn)
+16.345, D (attn,ffn,output) +17.141 us, all below the 50 us promotion
gate, so nothing is promoted; the row stays in the explanatory ledger.

Rows 2 and 3 are the remaining positive ceilings after the measured arm,
but no legal change has been built and measured for them. Rows 5-7 explain
why big node/union masses are not big levers: llama does not remove that
work, it hides it, and tinygrad's K/V work is already off the spine.

## 7. Phase A verdict

Copy-free native fp16 RMSNorm, wall bracket per arm, 24 wall samples, 96
timed tokens, byte-identical token stream SHA
`1e73e557e48b0c2f0792318e1a306f06a1412cd9800ba7a1e667b9c09c4a1254`, no
rejected samples, settled continuous:

| arm | sites | control bracket ms | candidate ms | delta us | promoted |
| --- | --- | ---: | ---: | ---: | --- |
| A | ffn | 4.723214 | 4.710708 | +12.506 | no |
| B | attn | 4.737494 | 4.720890 | +16.604 | no |
| C | attn,ffn | 4.729505 | 4.713160 | +16.345 | no |
| D | attn,ffn,output | 4.725735 | 4.708594 | +17.141 | no |

All four are positive and token-identical but below the 50 us promotion
bar. The measured gain is consistent with the isolated body prediction and
does not contradict the 234.528 us ideal ceiling; the bracket value, not
the node-mass prediction, is the bookable number.

## 8. Closed levers

Unchanged by this trace:

- more than two compute GPFIFOs: current-DAG ideal ceiling below the
  promotion bar before real wait tax;
- replay merge (`JIT_BATCH_SIZE=1024`): measured +112.9 us slower on
  production wall;
- early `launch_dependents` START placement: no new overlap beyond the
  landed QMD latch behavior;
- coarse flash split S=4/S=2: measured substantially slower on `DEV=NV`.

## 9. Reconciliation checks

- Unprofiled wall = device union + host gap: tinygrad 4732.500 - 9.286 =
  4723.214; llama 3892.777 + 112.932 = 4005.709. Exact on both sides.
- Node sum - device union = overlap mass: tinygrad 4738.496 - 4732.500 =
  5.996 us; llama 5020.797 - 3892.777 = 1128.020 us. Exact.
- Weighted critical path <= device span: tinygrad 4249.216 <= 4842.250;
  llama 4443.435 > 3901.205, the documented PDL launch-completion
  exception where consumers start before logical completion.
- Segment tile + anchor exposure reconciles with the per-layer device path:
  `reconstructed_minus_span_us` = 0.0 for both implementations.
- Predicted ceiling >= measured gain: ideal rmsnorm ceiling 234.528 us >=
  measured 12.506-17.141 us; the binding legal ceiling is the measured
  bracket, which sits below the promotion gate.

## 10. Provenance

| item | value |
| --- | --- |
| tinygrad commit | `6570abc02` |
| llama commit / build | `ac4cddeb0` / 9592 |
| model | `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf` |
| context / decode | d512 prompt, decode depth 512, 36-layer token |
| GPU | RTX 5090, same session, `flock /tmp/gpu-bench.lock` |
| tinygrad route | `DEV=NV`, PROFILE=1 topology plus paired unprofiled wall |
| llama route | CUDA, flash-attn on, single graph per token |
| tinygrad control capture | 596 nodes, 1230 edges, 5 replay graph groups |
| llama DAG | 762 nodes, 834 data edges, steady span 3901.205 us |

The tinygrad capture's HCQ timestamp unit was repaired in this tool
(raw `start_ns`/`end_ns` are microseconds; the prior capture divided by
1000), the five per-group host launch gaps were collapsed into one
back-to-back device timeline, and the anchor census was repaired to
36/36/36/36.

## 11. Artifacts

- `docs/task_workflow/output/nv-rmsnorm-phaseB-control-20260820.json`
  canonical current-HEAD control capture.
- `docs/task_workflow/output/nv-weighted-inter-anchor-ledger-20260820.json`
  the reconciled ledger, segments, composition, and tiling check.
- `docs/task_workflow/output/nv-weighted-inter-anchor-wall-sensitivity-20260820.json`
  critical paths, family/node/edge ceilings, and build rank.

The tool is
`extra/llm_research/decode/nv_inter_anchor_analysis.py`; run it with
`python3 extra/llm_research/decode/nv_inter_anchor_analysis.py`.

## 12. Open items

- The -122.218 us host row is a lower bound because tinygrad's profiled
  device timestamps absorb `PROFILE=1` tax; closing it exactly needs an
  unprofiled device-timeline route.
- Llama zero-cost edge ceilings are not mechanically meaningful (removing
  an inferred logical data edge severs the reconstructed chain) and are
  retained only for structure audit, not ranking. Tinygrad edge ceilings
  are legal-edge results: gate-up -> down RAW dependencies at 44-48 us and
  the vocab RAW edge at 56.352 us.
- The 2.592 us interval overlap residual is fully attributed to boundary
  overlap accounting between the anchor/support/tail interval sets.
