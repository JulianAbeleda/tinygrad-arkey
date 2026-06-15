# Clean prefetched GEMV — beat tinygrad's 42% toward readraw's 54% (scope)

Date: 2026-06-15. We've demonstrated the lever (MLP/prefetch: naive load-serialized 16% -> 24%). Now: a
register-EFFICIENT load structure that approaches the readraw ceiling (54%), and whether it beats tinygrad's
real GEMV (~42%).

## The signal to chase
- readraw (no dequant) = 54% peak, `loads=9` -- the compiler auto-vectorizes 36 words -> 9 uint4 wide loads.
- naive fp = 16% peak, `loads=1` -- the tight word-loop serializes loads behind the dequant.
- my register-array prefetch = 24% but VALU-bloated (1298) -> partly compute-bound.
So the lever is WIDE (uint4) LOADS + modest load-ahead, with a LEAN dequant (keep VALU low so it stays
load-bound, overlapping under the wide loads).

## Variants to build (all lean dequant q*x, apples-to-apples; readraw=ceiling, naive fp=floor)
1. **fp_vec** -- cast weights to uint4*, load 9 uint4/block (wide loads like readraw), looped dequant (low VALU).
2. **fp_vec_u2 / u4** -- fp_vec + unroll the uint4 load loop by 2/4 (N uint4 in flight = deeper MLP), small
   register footprint (N*4 words live, not 36).
3. pick the best; report % of peak vs readraw(54%) and naive(16%).

## Make-or-break
- If the best clean variant approaches readraw (e.g. >=45%) -> the load structure (wide+MLP) is the whole
  story; a clean GEMV CAN saturate, and the 42->54% decode gap is reachable on RDNA3. Build the full-dequant
  (affine) version with the winning structure and measure decode tok/s.
- If it caps well below readraw (e.g. ~30%) even with lean dequant -> the dequant dependency fundamentally
  limits bandwidth; the load structure alone can't close it, and the gap needs more (async/cache hints) or
  is a harder wall.
- Context: also measure tinygrad's real q4k_gemv_partial (full dequant) on this session for the 42% bar.
  My lean kernels are an UPPER BOUND (less compute than the affine); if even the lean best can't beat 42%,
  the full GEMV can't.

## Honest caveats
- Degraded-GPU session: absolute % is low; RELATIVE (variant vs readraw vs naive, same session) is valid.
- Lean dequant (q*x) omits the Q4_K affine -> it's the best-case bandwidth; the real win must survive the
  full affine compute, measured in a follow-up.
- Register pressure vs MLP is a tradeoff: too-deep prefetch (36-word arrays) bloats VALU/regs and goes
  compute-bound (the 24% result). The sweet spot is modest depth + wide loads.

## If the make-or-break passes
Build the full-affine prefetched GEMV (wire the winning load structure into the real Q4_K dequant), measure
standalone Q4-GB/s vs tinygrad's 42% and e2e decode tok/s vs fp. That is the first decode kernel that targets
the MEASURED bottleneck (MLP), not a compute lever -- a potential ~1.3x decode win on our hardware.
