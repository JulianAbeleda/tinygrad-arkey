# NVIDIA lifecycle and Q6 round-major endgame

## Goal

Close the NVIDIA Qwen3-8B gap without relying on llama cubins: retain the
already-qualified BoltBeam/tinygrad decode and packed-Q6 arithmetic, reproduce
llama's bounded-live execution path, pass exact correctness, and reach the
matched performance gates in the full runtime.

## Track D: prefill-to-decode lifetime

Observed failure: production setup retains weights plus prefill/decode capture
state until device use reaches 30.20--30.24 GB, then a 10.62 or 24.00 MB
allocation fails. The steady decode kernel is not the failing allocation.

1. `D1`: build the fixed-depth KV prefix without proactive decode capture;
   restore live-band admission and lazily capture only the reachable band in
   discarded decode warmups.
2. `D2`: require stable token hashes and a depth-1024 endpoint within 5% of the
   retained llama authority. No generated-token measurement may include JIT
   capture or prompt construction.
3. `D3`: repeat in fresh processes at depths 512/1024/2048/4096. Physical test
   extents may be bounded, but production generation must retain fallback above
   the qualified S6/S8/S10/S18/S34 bands and must never acquire an artificial
   context cap.
4. `D4`: only after D1--D3 pass, expose a generic phase-lifetime handoff that
   preserves model weights, KV state, and returned token storage while releasing
   dead prefill-only graph/workspace ownership before decode capture.

## Track Q: packed Q6 round-major lowering

Frozen authority: `M=512,N=4096,K=12288`, packed trusted-FP16 scales, 170 CTA
one-body Stream-K, all-partials ascending fixup. Current total is 256.256 us;
llama is 209.856 us and the 5% gate is 220.3488 us.

1. `Q1`: build a synthetic chain-major versus round-major phase test through
   real UOps, CUDA/PTX lowering, NVRTC, and nvdisasm. Identical inputs must be
   bit-exact. The candidate must have zero stack/LDL/STL, preserve the requested
   round ordering, and keep the panel load-to-store span at or below 160
   instructions.
2. `Q2`: express ordering as backend-neutral BoltBeam schedule metadata/UOps.
   Unsupported renderers fail closed; CUDA/PTX may own target-specific spelling,
   but model or kernel builders may not contain `sm_120` instruction text.
3. `Q3`: bind the passing schedule to the existing packed-Q6 route without
   changing arithmetic, 170-CTA ownership, barriers, partial layout, or fixup.
4. `Q4`: require trusted-reference tolerance success, bit-exact partial/fixup
   contracts, the frozen IMMA/LDSM/LDG/STS/BAR census, zero local spills, and
   same-process R31 total at or below 220.3488 us.
5. `Q5`: promote only passing shapes, then run the complete pp512 lifecycle and
   record memory plus endpoint throughput against the same llama authority.

## Stop rules

- A failed synthetic gate does not reach the real Q6 kernel.
- A fast but numerically different arm is rejected.
- A correct arm above the 5% bound remains research-only.
- Generated cubins and model-weight fixtures remain local; compact JSON/SASS
  census and decision records are committed.
