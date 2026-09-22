# Coalesced logical output for swapped Q4-A gate/up

The typed swapped 96x4 physical Stream-K source now remaps corrected C fragments with unconditional full-warp XOR4 gathers and writes logical `(512,12288)` output directly. Direct and partial branches use the identical transposed 128x128 local layout; fixup maps physical tile `(tile//4,tile%4)` to logical `(tile%4,tile//4)`.

The canonical conventional-versus-coalesced Stream-K probe passes with max absolute error `4.2915344e-06`; both direct and split-owner tiles are covered. The mapping test enumerates all 32 lanes x 4 source components and proves a unique cover of the 8x16 MMA result.

SASS retains LDSM64, SHFL128, and 64 STG.E.64 across the mutually exclusive direct/partial source branches (32 vector stores per branch). The Stream-K transform already has local traffic: raw x4 is STL18/LDL28; coalescing is STL17/LDL25, so the remap does not add spill traffic. Selection remains default-off pending full-model timing.

## Model decision

The correctness smoke passes the complete current252 contract. An initial R5 control/candidate/control bracket shows a 0.639704 ms midpoint win but 1.460775 ms endpoint drift. A reverse R9 bracket shows a 0.398218 ms midpoint win with 0.209348 ms candidate drift. The stabilized warmup9 R9 bracket is 47.977542 / 47.438707 / 47.758907 ms: median midpoint win 0.4295175 ms, minimum midpoint win 0.573945 ms, endpoint drift 0.218635 ms. The direction is consistent and matches the isolated 0.709280 ms main-body gain, but the authoritative median does not clear the frozen 0.5 ms population threshold. The route remains default-off and is not selected.
