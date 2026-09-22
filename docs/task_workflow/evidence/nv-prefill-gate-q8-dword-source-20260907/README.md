# Gate Q8-B aligned dword source transform (closed)

A fail-closed Stream-K source transform matched exactly eight generated `signed_char8` activation fragments. Each fragment's eight unique scalar shared loads were proven to be two aligned byte runs `[b:b+4]` and `[b+16:b+20]`. The candidate replaced 64 scalar declarations with 16 aligned `uint32` loads and a bit-preserving `uint2 -> signed_char8` cast. No compiler verifier, devectorizer, or generic renderer rule changed.

Fresh-process canonical block-0 gate output was bit exact (`max_abs=mean_abs=0`); input and packed weight remained read-only. Static binary census preserved IMMA 256, LDS 448, PRMT 496, LDG 152, STS 64, SHFL 128, stack/local 0, and shared 21,504 bytes. Registers fell from 250 to 224, but the intended LDS/PRMT reduction did not materialize because NVRTC lowered the source form to the same instruction classes.

Matched alternating R15 complete lifecycle timing was 359.857 us control versus 355.559 us candidate, a 4.298 us/call win; minima were 351.931 and 347.043 us. Across 72 gate/up calls the projected 0.309 ms model exposure is below the frozen 0.5 ms promotion gate, and the instruction target was missed. The transform is rejected and no enable path remains.
