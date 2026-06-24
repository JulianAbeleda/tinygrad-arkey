# SCOPE — the occupancy frontier (the real 42->66 WMMA lever), and re-opening "LDS refuted"

After Lever A (refuted) and Lever B (pipelining = project-level + moot), the conclusion was "occupancy is the real
ceiling driver, and POWN's wave experiments regressed." This scopes THAT -- the genuinely-unexplored angle.

## Grounded
- Production prefill matmul (opts TC,UPCAST0:2,UPCAST1:4,UNROLL0:8): **local_size=(32,1,1) = ONE wave32/workgroup**
  (single-wave; matches the POWN/Route-A "single-wave, no inter-wave WMMA latency hiding" finding). global_size
  (8,384) = 3072 workgroups. VGPR/thread + waves-resident/CU = P0 (not exposed via ProgramInfo; needs code-object
  metadata / DEBUG disasm).

## The tension (why POWN's wave experiments regressed) -- the crux
RDNA3 occupancy is VGPR-limited (1536 VGPR/SIMD, 256/thread max). The 42-TFLOPS kernel uses 128 accumulators/thread
(float) -> heavy VGPR -> low occupancy BUT high per-thread arithmetic intensity. POWN's "more waves" (W4x2/W2x4 =
64 acc) raised wave count but HALVED accumulators -> smaller output tile/thread -> LOWER arithmetic intensity ->
28-31 (regressed). So the three are coupled: **occupancy^ requires VGPR_v requires accumulators_v requires
intensity_v** -> naive occupancy increases lose more intensity than they gain in latency hiding. The default (4
waves/128 acc in the heuristic kernel, or 1 wave here) is a local balance at ~42.

## The contradiction to RE-OPEN: "LDS refuted" vs "Tensile uses LDS for 66"
Tensile/rocBLAS hit 66 with **LDS operand staging**: operands live in LDS (not VGPR) -> frees VGPRs -> enables
BOTH more accumulators (intensity) AND more resident waves (occupancy) simultaneously -> breaks the tension above.
BUT our prior conclusion was "LDS refuted on RDNA3" from:
- A3 P2/P3: tinygrad LDS-staged GEMM = ~6 TFLOPS (5% peak).
- POWN: no-LDS 38 vs LDS 42 -> "~90% is global-direct."
**Both are the WRONG test of the Tensile mechanism:** A3's 6 TFLOPS is a BROKEN impl (5% peak -- not evidence LDS
can't help, evidence that impl was bad), and POWN measured "add LDS to a register-heavy SINGLE-wave kernel"
(marginal +10%) NOT "use LDS to RELIEVE register pressure so occupancy can rise." Tensile uses LDS for the latter.
**So "LDS refuted on RDNA3" does NOT actually rule out Tensile-style LDS-for-occupancy.** This deserves re-examination.

## Plan
- **P0 (ground the limiter):** extract VGPR/thread, LDS, and waves-resident/CU for (a) the production 42-kernel,
  (b) the POWN regressed configs. Via code-object metadata (llvm-objdump on the .hsaco / comgr) or rocprof
  occupancy counter. CONFIRM the kernel is occupancy=low and VGPR-bound, and that POWN's regression = intensity loss.
- **P1 (re-test LDS fairly):** profile the A3 6-TFLOPS LDS kernel -- WHY 6 (bank conflicts? barrier serialization?
  uncoalesced? occupancy still 1?). Determine if it's a fixable impl bug vs a real LDS-bandwidth limit on RDNA3.
- **P2 (the real question):** can tinygrad EXPRESS Tensile-style LDS-operand-staging that frees enough VGPR to raise
  occupancy 1->2+ waves WHILE keeping high accumulators (intensity)? Map the DEFINE_LOCAL + tiling + the
  load->LDS->WMMA path; identify if it's expressible with current primitives (DEFINE_LOCAL, BARRIER) or needs new
  capability. (Note: the Lever-B linearizer pin is about cross-iteration prefetch, a DIFFERENT issue from LDS staging.)
- **P3 (prototype):** if expressible, build the occupancy-balanced LDS config (a few candidate VGPR/wave/acc/LDS
  points, NOT POWN's naive sweep) and measure vs 42. GATE: >=1.2x isolated, transfers in-model.

## Risks / honesty
- Heavy overlap with POWN (configs) + A3 (LDS), both "refuted." The NEW angle is the VGPR/occupancy/intensity
  framing + the Tensile-LDS-for-OCCUPANCY mechanism, which neither actually tested. Real risk it still walls
  (tinygrad can't express the balanced tiling, or RDNA3 single-wave WMMA issue rate is the true limit even with LDS).
- But the "LDS refuted" conclusion is directly contradicted by Tensile getting 66 WITH LDS -> P1/P2 are warranted
  before declaring 42 the final ceiling. This is the one genuinely-unexamined lever.

## Files
POWN (prefill-own-wmma-kernel-result), A3 (route-a-a3-p2-p3-lds-refuted), wmma-both-levers-conclusion. P0 needs
code-object VGPR/LDS extraction (not in ProgramInfo).
