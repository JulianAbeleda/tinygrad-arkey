# NV Decode: Llama Full Path Trace + Complete Lever Ledger

Date: 2026-08-17. GPU: RTX 5090 (idle). Model: Qwen3-8B-Q4_K_M, d512, same session.

Primary sources (all measured, residual 0.0):

- `/tmp/nv_240_exact_wall_account_20260817.json` (tinygrad HEAD `07e9b2abe`, llama `ac4cddeb0`)
- `/tmp/llama_ledger_d512_20260817.json` (nsys graph 6, 762 nodes, 49 complete replays, 47 steady, single stream)
- Docs: `nv-240-exact-wall-account-20260817.md`, `nv-llama-pdl-launch-hiding-trace-record-20260816.md`,
  `nv-launch-hiding-substrate-exhaustive-account-20260813.md`, `nv-decode-path-pseudocode-fresh-ledger-20260816.md`,
  `nv-shadow-size-class-unfuse-substrate-scope-20260817.md`

## 1. The wall equation (exact)

Same-session measured wall:

| | tinygrad | llama | delta |
|---|---:|---:|---:|
| tok/s | 208.84 | 246.37 | +37.5 |
| wall ms | 4.7883 | 4.0589 | +729.4 us |
| host gap us | 268.97 | 168.32 | +100.6 |
| GPU busy us | 4519.3 | 3890.5 | +628.8 |
| node_sum us | 4519.3 | 5015.7 | -496.3 |
| overlap mass us | 0.0 | 1125.1 | +1125.1 |

Identity (verified, residual 0.0):

```text
wall = GPU busy + host_gap
GPU busy = node_sum - overlap_mass

wall_delta 729.4 = busy_delta 628.8 + host_delta 100.6
busy_delta 628.8 = node_sum_delta (-496.3) - overlap_delta (-1125.1)
```

The entire GPU-busy delta comes from one inversion: llama hides 1125.1 us of
overlap mass, tinygrad hides 0.0. Kernel mass alone we are BELOW llama by
496.3 us.

## 2. Llama's full per-token path trace

One token, one stream (single stream 45), 762 kernel launches. Anchor is the
Q*W GEMV chain (`mmq`, 217 nodes, 3588.3 us node_sum, zero hidden). Everything
else hides behind it via CUDA programmatic dependent launch (PDL:
`cudaLaunchAttributeProgrammaticStreamSerialization` +
`cudaTriggerProgrammaticLaunchCompletion`; Hopper+ ptx>=90). 543/761 launches
(71.4%) start before their predecessor finishes - same-stream PDL, no second
stream.

Llama per-class exposure ledger (node_sum = total kernel work in class;
hidden = work hidden behind mmq; exposed = work on the wall):

| class | node_sum us | hidden us | exposed us | nodes |
|---|---:|---:|---:|---:|
| mmq (Q*W anchor) | 3588.3 | 0 | 3588.3 | 217 |
| quantize_q8_1 | 549.7 | 443.9 | 105.6 | 217 |
| rms_norm | 307.8 | 156.0 | 151.5 | 145 |
| flash_combine | 189.7 | 137.7 | 52.2 | 36 |
| flash_score | 173.7 | 98.7 | 75.1 | 36 |
| rope | 127.3 | 32.8 | 94.4 | 72 |
| kv_set_rows | 74.7 | 0 | 74.7 | 36 |
| get_rows | 2.8 | 0 | 2.1 | 2 |
| elementwise | 1.9 | 0.6 | 1.2 | 1 |

Non-anchor aggregate: hidden 443.9 us, exposed 302.4 us, union 746.6 us.
Replay median: node_sum 5015.7 us, overlap 1125.0 us, span 3899.5 us
(22.3% span discount). Inter-replay bounded median gap 240.0 us is the host
submit gap; that is where the 168.3 us host-gap component comes from.

Measured llama kernel shapes (shape census):

| kernel | grid | block | med us | regs |
|---|---:|---:|---:|---:|
| flash_attn_ext_vec | (1,6,32) | (32,4,1) | 4.80 | 162 |
| flash_attn_combine_results | (1,32,1) | 128 | 5.25 | 40 |
| mul_mat_vec_q (37x) | (4096,1,1) | (32,4,1) | 9.54 | 56 |
| mul_mat_vec_q (36x) | (12288,1,1) | (32,4,1) | 37.82 | 48 |
| mul_mat_vec_q (54x) | (1024,1,1) | (32,4,1) | 3.42 | 56 |
| quantize_q8_1 (181x) | (16,1,1) | 256 | 3.01 | 24 |
| quantize_q8_1 (36x) | (48,1,1) | 256 | 0.83 | 24 |
| rms_norm_f32 (73x) | (1,1,1) | 1024 | 2.91 | 40 |
| rms_norm_f32 (36x) | (32,1,1) | 256 | 1.31 | 40 |
| rms_norm_f32 (36x) | (8,1,1) | 256 | 1.28 | 40 |
| rope_neox (36x) | (8,1,1) | (1,256,1) | 1.79 | 19 |
| rope_neox (36x) | (32,1,1) | (1,256,1) | 1.73 | 24 |
| k_set_rows (36x) | (4,1,1) | 256 | 2.08 | 18 |
| k_get_rows_float (2x) | (1,16,1) | 256 | 1.46 | 38 |

Llama spends 549.7 us quantizing Q to Q8_1 in a separate kernel (443.9 of it
hidden) because its GEMV path is not fused; llama's norm/rope/combine classes
are likewise free-standing kernels hidden behind the mmq anchor.

## 3. Tinygrad's path mapped 1:1

Same class taxonomy, both sides measured in the same session. delta = tg -
llama (negative = we are ahead):

| class | tg us | llama us | delta | status |
|---|---:|---:|---:|---|
| gemv (incl. folded quant) | 3477.4 | 4138.0 | -660.6 | ahead (fusion) |
| reduce_output | 312.1 | 0.0 | +312.1 | OPEN |
| norms | 206.2 | 307.7 | -101.4 | ahead (fusion) |
| flash_score | 213.1 | 173.6 | +39.4 | structural parity gap |
| flash_combine | 99.6 | 189.7 | -90.1 | ahead |
| rope_kv | 100.5 | 201.9 | -101.4 | ahead (fusion) |
| vocab_aux | 59.5 | 0.0 | +59.5 | OPEN (F5 keys.clone landed) |
| other | 49.1 | 1.9 | +47.2 | open (small) |
| residual_cast | 1.9 | 2.9 | -0.9 | ahead |
| class sum | 4519.3 | 5015.7 | -496.3 | residual 0.0 |

We are already ahead on every fused class (gemv -660.6, norms -101.4, rope_kv
-101.4, flash_combine -90.1 = -953.5 us combined). The only kernel-work rows
where we are behind are reduce_output (+312.1), vocab_aux (+59.5), flash_score
(+39.4), and other (+47.2).

## 4. The complete lever ledger

240 target = 4166.7 us/token. Every row states its measured mass, its
wall-impact ceiling, its status, and its tok/s effect at that ceiling. Effects
are computed against the current 4788.3 us wall and are ceilings, not
forecasts.

| # | lever | mass (node us) | wall ceiling | status | tok/s ceiling |
|---|---|---:|---:|---|---:|
| L1 | reduce_output elimination/absorption | 312.1 | +312.1 | OPEN | 223.4 |
| L2 | vocab_aux elimination | 59.5 | +59.5 | OPEN (F5 keys.clone landed, row open) | 211.5 |
| L3 | flash_score parity | 39.4 | +39.4 | structural (shape/vectorization) | 210.6 |
| L4 | other (residual launches) | 47.2 | +47.2 | open | 210.9 |
| L5 | overlap / shadow mass | llama 1125.1, tg 0 | NOT 1125.1; transferable ~17.9-33 (see 4.1) | size-class WALL | ~209.6-210.3 |
| L6 | host gap | 100.6 | +100.6 | open (submit-ahead landed, gate closed-default) | 213.3 |
| L7 | PDL programmatic launch | - | construction | CONSTRUCTION-REQUIRED | enables L5 |
| L8 | gemv / norms / rope_kv / combine | -953.5 | already won | built/fused | baseline |

Kernel-work rows L1+L2+L3 sum to 411.0 us node; at perfect 1:1 recovery that
is 228.4 tok/s (wall 4377.3 us), not 240 - and adding L4 (458.2 us total)
only reaches 230.9 tok/s. Wall-to-tok/s conversion is sublinear: the residual
729.4 us wall delta minus the 411.0 us of kernel rows leaves 318.4 us that
240 requires - overlap (L5) plus host-gap (L6) plus PDL construction (L7).

### 4.1 The shadow answer (direct)

The overlap number (1125.1 us hidden vs our 0) is the single largest term in
the wall equation, but it is NOT transferable 1:1, for two measured reasons:

1. Llama's hidden mass is mostly its own non-fused quantize/norm/rope
   (443.9 of 746.6 non-anchor union us = quantize_q8_1; 156.0 = rms_norm;
   137.7 = combine; 98.7 = flash_score; 32.8 = rope). We already captured that
   work by fusion: our node_sum is 496.3 us BELOW llama's. The launch-hiding
   audit (08-13) quantified this: 578 us of llama's hidden support kernels
   correspond to work tinygrad already fused away; the transferable exposed-
   and-hidden remainder is the flash pair (~143 us node).
2. The native shadow size-class gate (this session, committed `2c4e68e06`):
   merged per-layer support kernels render (step 1 PASS) but run 2.5 us median
   (q8+rmsnorm 2.0, rope+kv_store 1.75) against the 7-25 us band where an
   anchor's shadow slot is big enough to matter. Layer-batch widening is
   grid-parallel and flat in L (4.75-5.0 us at L=4/8/16). Row closes
   size-class-blocked (wall), not substrate.

So: llama's 1125 us is mostly structural (it pays for non-fusion with
overlap; we already paid the smaller bill by fusing). The genuinely open lever
rows are L1 + L2 (371.6 us node, +17.6 tok/s exact at 1:1) - real, buildable
wins. 240 needs overlap (L5) + host gap (L6), which stay construction-blocked
on PDL (L7) and the size-class wall.

## 5. Transferable-overlap ceiling (per launch-hiding audit 08-13)

Llama's flash pair is the only exposed-and-hidden class that is not already
absorbed by our fusion: flash_combine 137.7 hidden + flash_score 98.7 hidden.
Even at perfect recovery of that pair (plus the residual 302.4 us of non-anchor
exposed work that PDL would hide in llama), the honest transferable ceiling is
~17.9-33 us of wall, not 946 us. Numbers in
`nv-launch-hiding-substrate-exhaustive-account-20260813.md`.

## 6. What this means for the plan

1. The audit says we are NOT kernel-mass-bound against llama; we are already
   ahead there (-496.3 us). We are overlap-bound (0 vs 1125.1) and host-gap
   bound (+100.6).
2. The next real, buildable wins are L1 (reduce_output, +312.1 us) and L2
   (vocab_aux, +59.5 us). These are fusion/elimination work, not new GPU
   primitives.
3. The 240 path requires the shadow substrate: build PDL-equivalent
   programmatic launch (L7, CONSTRUCTION-REQUIRED on native - no griddep PTX,
   no WAIT_ON_LATCH/ARRIVE_AT_LATCH programming), and break the size-class
   wall (L5) so a shadow kernel slot can actually hide behind the mmq anchor.
