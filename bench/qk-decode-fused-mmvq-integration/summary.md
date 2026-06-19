# Decode fused-MMVQ integration FMI-1/FMI-2

Verdict: `BUILD_TRACK_B_FIRST`.

FMI-1 passes: the in-model weight-GEMV bucket has enough movement. The authority aggregate is tinygrad `44%` vs llama `54%` HBM in-model, which projects about `1.187x` if recovered across the weight-GEMV bucket.

FMI-2 passes: llama's dominant MMVQ launch contract is q8 + wg32/large-grid/lds0, while tinygrad's in-model routes are mixed fp/Q4/Q6 custom kernels with partial-output reductions and do not retain standalone BW.

Decision: build Track B first. It is byte-identical and targets the larger integration loss. Track A q8 replay remains secondary and lossy/dNLL-gated.
