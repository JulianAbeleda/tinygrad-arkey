# Direct-wide Q Q4-A/x4 coalesced epilogue

The typed weight-A/x4 Q path keeps the compiler's existing direct-wide body and launch, replacing only its exact 32-store physical NxM terminal span with the validated XOR4 logical MxN epilogue. Canonical block0 Q is bit-exact against conventional Q (`max_abs=0`). The candidate SASS retains LDSM8 and has SHFL64, STG.E.64=32, LDL0, STL0. The conventional O source and cubin remain byte-identical.

The selected current252 lifecycle with Q/Q4V sharing formally passes: token198, exact replay3 and deep stage replay, canonical252/252 weight bases, 252 generated mains, Q8 producer count198, fixups90, and no overlays or weight copies. Timing selection remains pending a stabilized matched bracket.

The stabilized warmup9/R9 control/candidate/control bracket does not promote Q-only. Controls were `48.127924 ms` and `47.905265 ms` (midpoint `48.0165945 ms`); candidate was `48.109609 ms`, a `0.0930145 ms` regression. The direct coalesced substrate remains available default-off, but `--q-x4` is not selected.
