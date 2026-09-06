# NV generated corpus disposition

Date: 2026-09-06

The retained corpus is substantial, but authority, integration, and performance
must be kept separate. The BoltBeam NV qualification bundle contains 33 exact
decode routes. Many are isolated wins and the composed generated decode path is
at effective d512 parity. They are valid substrate inputs for decode closure.

For Qwen3-8B pp512, the current252 graph already integrates the retained
generated projection wins behind the explicit compiler composition:

- Q6 FFN-down Stream-K beats its live llama primitive, 224.992 us to 226.208 us,
  and recovered 5.249 ms in its original model bracket.
- Q4 FFN-down, Q6 V, and generated vocabulary each beat the previous tinygrad
  fallback when composed; current252 contains all 252 generated projections,
  canonical packed weights, and no V/down FP16 overlays.
- Fused prefill attention is exact-generated and correctness-qualified, but its
  installed population is about 3.32 ms versus about 1.66 ms for llama.
- Generated gate/up main plus fixup is about 271.62 us per role and 19.35 ms for
  72 roles, versus 14.15 ms for the reference-class population.

Therefore no unintegrated, end-to-end pp512 winner exists in the retained
artifacts. The ordinary 34.89 ms path beats llama by selecting native llama
binaries; current252 is 50.75 ms and remains 11.93 ms behind the fresh llama
reference. Promoting current252 as ordinary now would violate the generated
performance goal.

The old `nv_sm120_vkv_h4_t64_w4_online128_v1` prototype was re-run on the
current tree. It remains full-output correct (`max_abs=3.02e-5`, read-only
inputs, no unwritten output) but takes 7,577.664 us for one call. It implements
a scalar recurrence across all 64 query rows and is not an integration
candidate.

The next substrate boundary is concrete: generate a prefill Flash program that
combines tensor-core score work, vector K/V staging, bounded online-softmax
ownership, and direct FP16 output. It must first beat the installed generated
93 us/layer class in the frozen live fixture, then pass the 36-layer population
and model gates. Gate/up remains the next dense-body boundary; prior scalar
schedule toggles do not constitute a missing corpus win.

Ordinary promotion follows only after the generated composition reaches the
matched llama wall. Until then, the native default is an oracle and rollback,
not evidence that the generated goal is complete.
