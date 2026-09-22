# Gate/up physical-layout chain discriminator

`--gate-x4-chain` binds both gate and up to the generated Q4-A/x4 Stream-K body and returns their physical `(12288,512)` storage as logical transpose views. The full model smoke passes token 198, exact replay, canonical 252/252 weights, zero overlays/copies, and the unchanged generated program census. No standalone transpose service appears; SiLU/multiply and down's producer consume the views in the ordinary graph.

A fresh-process control/candidate/control R3 bracket measured 50.908677 / 52.285344 / 51.373079 ms. The control midpoint is 51.140878 ms, so the chain regresses 1.144466 ms with 0.464402 ms endpoint drift. It is not selected. Since the isolated x4 body wins and topology adds no program, the remaining loss is the strided logical view at the elementwise/down producer boundary. The next compatible hook is a typed swapped down consumer of the shared physical layout.

## Live service split

Nine hot samples for every live call separate the conversion boundary. The 72 x4 Stream-K mains sum to 18,941.280 us versus 19,650.560 us for conventional mains, a 709.280 us body win. The 36 transposed-layout SiLU/multiply calls sum to 705.760 us versus 590.848 us, a 114.912 us loss. The measured main-plus-epilogue boundary therefore still wins 594.368 us; the model regression occurs after this boundary in the fp16 cast/down producer. This evidence keeps the generated x4 body open and directs the next experiment to a typed coalesced logical C-store remap.
