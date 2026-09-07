# Gate/up physical-layout chain discriminator

`--gate-x4-chain` binds both gate and up to the generated Q4-A/x4 Stream-K body and returns their physical `(12288,512)` storage as logical transpose views. The full model smoke passes token 198, exact replay, canonical 252/252 weights, zero overlays/copies, and the unchanged generated program census. No standalone transpose service appears; SiLU/multiply and down's producer consume the views in the ordinary graph.

A fresh-process control/candidate/control R3 bracket measured 50.908677 / 52.285344 / 51.373079 ms. The control midpoint is 51.140878 ms, so the chain regresses 1.144466 ms with 0.464402 ms endpoint drift. It is not selected. Since the isolated x4 body wins and topology adds no program, the remaining loss is the strided logical view at the elementwise/down producer boundary. The next compatible hook is a typed swapped down consumer of the shared physical layout.
