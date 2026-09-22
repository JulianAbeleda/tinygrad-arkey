# Generated gate/up packed Q8 DS4 loads

The typed Stream-K transform matches exactly eight Q8-B `char8` fragments and 32 Q8 DS4 half metadata values. It replaces 64 scalar fragment-byte declarations with 16 aligned uint32 loads and 64 scalar metadata-byte declarations with 32 aligned half loads. Every declaration has one consumer; Q4 metadata cast25..56 is excluded after its isolated neutral result. Mismatched source fails closed.

The production-shaped R15 comparison is bit exact and read-only. Baseline measured355.047us (min349.507), Q8 packed measured341.241us (min336.242), a13.806us/call win. SASS preserves IMMA256, LDG152, LDS448, zero stack/local/LDL/STL, while PRMT falls496→368 and registers250→207.

The first deep artifact omitted the selected scalar Q4-down command and is retained as command-topology evidence; all arithmetic/replay stages were exact. The corrected smoke formally passes with generated mains/canonical weights252, Q8 producers234, active fixups126, zero overlays/copies, token198, and exact three-cycle logits/stages.

Stable warmup9 control/candidate/control R9: control medians46.182695/45.959445ms (midpoint46.071070) versus candidate45.458594ms, win0.612476ms; minima45.674801/45.859808 (midpoint45.7673045) versus44.926032, win0.8412725ms. Endpoint drift is0.223250ms median and0.185007ms min. Both frozen0.5ms gates clear. The qualified compiler pp512 selector enables it ordinarily; rollback is `NV_COMPILER_Q4_GATE_Q8_PACKED_LOADS=0`.

`ordinary_selector_smoke.json` leaves the rollback variable unset and does not pass the research packed-load switch. The harness invokes the production selector predicates, which choose x4 plus packed Q8 loads. It formally passes the same252/234/126 canonical topology with zero copies/overlays and exact three-cycle deep replay.
