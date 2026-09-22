# Q4_K single-projection vector-load candidate result

Date: 2026-08-23
Repo: `/home/ubuntu/tinygrad-arkey`
Branch: `nvidia-bringup-20260731`, HEAD `6570abc025514273faa100c66b979e531585a1e1`
Backend: `DEV=NV`, RTX 5090 `sm_120`
Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, decode depth 512

## Verdict

`NO_GO_PROMOTION`. The candidate is bit-exact and structurally clean, but its
wall effect is within the measurement noise floor: two reverse brackets
disagree on sign. It is retained only as a research opt-in
(`TINYGRAD_Q4K_VECTOR_LOAD=1`); the installed scalar spelling remains the
closed production default.

## Candidate

Extend the already-landed gate/up (`w1w3`) bit-exact vector-load pattern to the
Q4_K single-projection GEMV `q4k_g3_lanemap_gemv_kernel`: one `uint4` header
load, each qpack word loaded once for its two groups, and the four per-group
fp16 activations loaded as one `half4`. The per-lane accumulation order is
unchanged, so the fp32 dot is bit-identical by construction. Applies to the
x-based epilogues only (`""`, `fp16_cast`, `residual_add`, `ffn_down_resadd`);
the `ffn_down_fused` prelude reads activations and keeps the scalar inner loop.

## Measurements

### Microgate (isolated, L2-resident)

- `[MEASURED]` 4096x4096 (Q/O): bitwise-identical (memcmp, non-zero data),
  control 8.05 us -> candidate 5.12 us per launch, ~36% faster.
- `[MEASURED]` 4096x12288 (down shape): bitwise-identical, control 21.4 us ->
  candidate 13.0 us per launch, ~39% faster.
- `[MEASURED]` Registers drop 61 -> 43; no spill, no stack.

### Structural census

- `[MEASURED]` The production schedule contains 83 Q4_K single-projection
  kernels per token-equivalent capture: `q4k_g3_lanemap_gemv_1024_4096` (28),
  `q4k_g3_lanemap_gemv_4096_4096` (19), `q4k_g3_lanemap_gemv_epi_resadd_4096_4096`
  (36). The vector spelling replaces all 83 with `_vec_` names and adds no
  materialization/copy/cast kernel. The shared-Q8 cooperative attention route
  (43 kernels) and the landed w1w3 vectorized route (36 kernels) are untouched.

### Wall brackets (production, DRAM-bound)

- `[MEASURED]` Bracket 1 (reps=5): control midpoint 4.70263 ms, candidate
  4.68820 ms, +14.43 us/token, token stream hash `f25083e5...` identical across
  all three arms, verdict `WALL_PASS`.
- `[MEASURED]` Bracket 2 (reps=7, higher SNR): control midpoint 4.70524 ms,
  candidate 4.70734 ms, -2.10 us/token, per-rep token hashes identical across
  all arms and across both brackets, verdict `NO_GO_WALL`.

## Interpretation

- `[MEASURED]` Correctness is exact in every arm. The per-window token hashes
  are byte-identical between scalar and vector, and between the two sessions.
- `[MEASURED]` The isolated ~36% speedup does not transfer to the installed
  wall. The first bracket's +14.43 us is inside the control-arm spread of
  ~13.8 us, and the second bracket reverses sign.
- `[INFERRED]` Single-token decode is DRAM-bandwidth-bound: per token it streams
  roughly the full ~5 GB of Q4_K weights, while the redundant header loads the
  vector spelling eliminates are L1/L2-resident and do not add DRAM traffic.
  The load-width win is therefore an instruction/L1-level effect that only
  appears when the working set is cache-resident (the back-to-back microgate),
  not when every layer's weights are streamed once per token.
- `[INFERRED]` This implies the "scalar-load DRAM streaming" pool attributed to
  the quantized GEMVs is not recoverable by load-width vectorization alone.
  Closing that pool requires reducing actual DRAM bytes (smaller activations,
  narrower intermediates, or better reuse), not just widening the load
  instructions.

## State change

- `decode_kernels.py`: added `_half4_lane`, `_q4k_block_dot_packed_load_vec`,
  and a `load_style="vector"` branch on the single-projection kernel (shared
  with the landed w1w3 vectorized spelling).
- `decode_routes.py`: the single-projection `q4k_load_style` stays `scalar` by
  default and opts into `vector` only when `TINYGRAD_Q4K_VECTOR_LOAD=1` on NV
  `sm_120`. AMD/Metal behavior is unchanged.

No wall recovery is booked. The current endpoint is unchanged at ~212.9 tok/s
(~4697 us/token authority), still ~530 us from the 240 target.

## Evidence

- `docs/task_workflow/evidence/nv-q4k-single-vector-load-microgate-20260823/`
- `docs/task_workflow/evidence/nv-q4k-single-vector-load-wall-20260823/`
- `docs/task_workflow/evidence/nv-q4k-single-vector-load-wall2-20260823/`
- `docs/task_workflow/evidence/nv-q4k-single-vector-load-census-20260823/`

## Next candidates (ranked by wall leverage)

1. The shared-Q8 cooperative attention GEMV (`q4k_warp_coop_q8_dp4a_partial_*`,
   43 kernels/token) is the larger attention-projection pool and uses a different
   DP4A/Q8 body; measure its admission/wait/body partition before touching it.
2. Q6_K down/vocab vectorization (byte-strided Q6 layout, harder than Q4_K).
3. Vocab tail topology (native fp32+int32 argmax).
4. Flash combine reduction topology (48-way parallel reduce).
