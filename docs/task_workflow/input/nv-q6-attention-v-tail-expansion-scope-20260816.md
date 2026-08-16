# NV Q6 attention-V tail expansion scope (2026-08-16)

Date: 2026-08-16
Branch: `nvidia-bringup-20260731` (HEAD `a7e690171`)
Status: **NO-GO. Precision-blocked at the shared-Q8 ABI; nothing promoted.**

## 1. Why this is the next step

The Q6 attention-V direct-output consumer
(`q6k_q8_warp_direct_1024_4096`, llama `vec_dot_q6_K_q8_1_impl_mmvq` geometry)
is already promoted for the eight real Q6-K V blocks inside the max17 shared-Q8
lease. The production census shows the promoted family is already below the
llama floor while the remaining Q6 V blocks still pay the ordinary partial
kernel:

| kernel | blocks | median us |
| --- | ---: | ---: |
| `q6k_q8_warp_direct_1024_4096` | 8 | 4.27 |
| `q6k_gen_partial_1024_4096_4` | 10 | 17.83 |
| llama floor | - | ~4.90 us/node |

The ten remaining `q6k_gen_partial_1024_4096_4` blocks are the largest remaining
attention-GEMV gap: roughly `10 x ~13.5us = ~135us` of node work. This scope
extends the already-promoted Q6-direct route to the remaining real Q6 V blocks
that are actually leasable, leaving the embedding boundary untouched.

## 2. Exact topology (CPU GGUF metadata, not guessed)

`extra/llm_research/decode/shared_q8_real_topology_census.py` on
`/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf` reports 36 blocks with Q/K always
`Q4_K`. The 18 Q6 V blocks are:

```text
0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31, 32, 33, 34, 35
```

The current max17 lease (`1-12` and `14-18`) already holds eight of them:

```text
1, 2, 3, 6, 9, 12, 15, 18
```

The remaining ten Q6 V blocks are:

```text
0, 21, 24, 27, 30, 31, 32, 33, 34, 35
```

Block `0` is the embedding boundary: its attention norm does not carry the
`REDUCE_OUTPUT` marker required by the fused `rmsnorm_q8_1_llama_provider_4096`
provider, and the loader/harness deliberately keep it unleased. It stays
ordinary in this scope. The nine actionable tail Q6 V blocks are:

```text
21, 24, 27, 30, 31, 32, 33, 34, 35
```

## 3. What changes

No new kernel is implemented for the test. The production emitter, admission,
and loader wiring already exist:

- `tinygrad/llm/shared_q8_attention.py`: `SharedQ8AttentionAdmission`,
  `shared_q8_attention_call`, and `_emit_q6_warp_direct`.
- `tinygrad/llm/model.py`: installs `SharedQ8AttentionAdmission(_idx,
  cooperative_q4=True, q6_direct_output=...)` for `_SHARED_Q8_LEASE`.
- `tinygrad/llm/generated/decode-q6-direct-shared-q8-attention-route-policy.json`:
  already promoted for `NV/sm_120`.

The candidate is the current production lease plus the nine tail Q6 V blocks,
all with `cooperative_q4=True` (Q/K stay on the already-promoted cooperative Q4
route) and `q6_direct_output=True` (Q6 V uses the direct consumer):

```text
candidate lease = 1-12, 14-18, 21, 24, 27, 30, 31, 32, 33, 34, 35
control lease   = 1-12, 14-18
```

This is a lease-extent delta only. Both arms keep `cooperative_q4=True` and
`q6_direct_output=True`, so the only inter-arm change is the nine added Q6 V
blocks moving from `q6k_gen_partial_1024_4096_4` to
`q6k_q8_warp_direct_1024_4096` (with the fused shared-Q8 provider).

## 4. Precision risk and prior NO-GO

`nv-q4-cooperative-tail-subset-extension-record-20260805.md` marked blocks
`19-35` tail-expansion NO-GO under the cooperative-Q4 route. That measurement
predates the Q6-direct consumer and used the Q6 partial V emitter. Its singleton
relative-L2 table for the nine target blocks:

| added block | relative L2 | old verdict |
|---:|---:|:---:|
| 21 | `0.000874874` | PASS |
| 24 | `0.000876298` | PASS |
| 27 | `0.000899477` | PASS |
| 30 | `0.000969715` | PASS |
| 31 | `0.000882946` | PASS |
| 32 | `0.000941336` | PASS |
| 33 | `0.000883147` | PASS |
| 34 | `0.001205941` | FAIL |
| 35 | `0.001171132` | FAIL |

The Q6-direct V consumer is bitwise-compatible with llama's MMVQ and is strictly
more precise than the old partial emitter, but Q/K cooperative-Q4 error is
unchanged. Blocks `34` and `35` are therefore real risk and must be decided by
a fresh semantic child, not by this table. The semantic gate is the authority;
if the full candidate fails, the fallback is to drop `34`/`35` (or test those
two blocks with the ordinary Q4 Q/K route), then re-bracket.

## 5. Decisive tests

Semantic authority is a fresh control-vs-candidate harness child comparison at
d512 (exact tokens, equal argmax, ordered top-10, relative L2 `<= 1e-3`,
`2*max_abs/min_top1_margin < 1.0`).

Wall authority is the settled-continuous reverse bracket (control A / candidate
/ control C, 32-token windows x5 reps, fresh model load per arm under
`flock -w 600 /tmp/gpu-bench.lock`) at d512/d2048/d4096.

Acceptance for promotion:

1. semantic gate passes;
2. all three wall arms produce the same token stream hash at every depth;
3. candidate median is below the control A/C midpoint at every depth (or, at
   minimum, the combined-depth median is clearly negative and no depth
   regresses beyond same-session noise);
4. production census confirms the nine added `q6k_q8_warp_direct_1024_4096`
   consumers (18 across two captures) with zero legacy shared-Q4 and zero
   duplicate providers.

## 6. Promotion if the test passes

1. Extend `_SHARED_Q8_LEASE` in `tinygrad/llm/model.py` to
   `tuple(range(1, 13)) + tuple(range(14, 19)) + (21, 24, 27, 30, 31, 32, 33, 34, 35)`.
2. Update the `decode-shared-q8-attention-route-policy.json` description to
   name the added Q6 V tail blocks and remove the blanket "19-35 NO-GO" wording.
3. Update `test/unit/test_shared_q8_attention_landing.py` so the lease assertion
   reflects the Q6-V-tail extension while `0`, `13`, and the Q4 V tail blocks
   stay ordinary.
4. Run unit tests and `python3 sz.py`, re-run the production census, then
   commit and push.

## 7. If the test is flat or slower

Do not promote. Record the bracket as NO-GO evidence and keep the lease at
max17. The nine-block device-time gap stays as an open ledger item with a
precision blocker, not a silent wall regression.

## 8. Measured outcome (2026-08-16) — NO-GO

The semantic gate (fresh control-vs-candidate harness children, d512/count8,
`composed`) is the authority. Control is the production max17 lease with
`cooperative_q4=True` and `q6_direct_output=True`. Every candidate keeps
`q6_direct_output=True` and only changes the lease extent (and, in one arm, the
Q4 Q/K cooperative flag on the tail blocks).

| candidate | relative L2 | gate (`<=1e-3`) | top-k order |
| --- | ---: | :---: | :---: |
| max17 + 9 tail, all coop Q4 | `1.1487e-3` | FAIL | false |
| max17 + 7 tail (drop 34,35), all coop Q4 | `1.0449e-3` | FAIL | true |
| max17 + 6 tail (drop 30,34,35), all coop Q4 | `1.0616e-3` | FAIL | true |
| max17 + 9 tail, tail Q/K non-cooperative | `1.1507e-3` | FAIL | false |

Every arm kept exact sampled tokens and argmax, finite logits, and a tiny
perturbation-vs-min-margin (`0.03-0.05`). The relative-L2 overage is small but
persistent, and the 7-block -> 6-block row shows the error is non-monotonic in
the leased subset (dropping block 30 made relative L2 slightly worse), matching
the 08-05 tail record's "precision budget is not monotonic" finding.

Root cause is localized by the mixed arm: switching the tail Q/K from the
four-warp cooperative Q4 consumer to the ordinary shared-Q8 Q4 consumer does
not move relative L2 (`1.1507e-3` vs `1.1487e-3`), so the error is not the Q4
cooperative reduction and not the Q6 V emitter (the direct consumer is already
bitwise-compatible with llama). It is the shared-Q8 activation ABI itself
(llama `Q8_1` `d=amax/127` quantization) versus the ordinary per-projection
quantized path, accumulated through the residual stream in the tail layers.

That activation-only hypothesis is now measured directly
(`extra/llm_research/decode/nv_q8_activation_quantization_error_microgate.py`).
It captures the real fp16 RMSNorm output for one attention block, quantizes it
exactly the way the shared provider does, then computes both errors with the
same dequantized Q6 V weights and fp64 accumulation (the only delta is
`x -> q8_1(x)`):

| block | activation rel L2 | V-projection rel L2 | activation amax |
|---:|---:|---:|---:|
| 18 (in-lease) | `6.457e-3` | `5.250e-3` | 5.01 |
| 21 (tail) | `6.093e-3` | `5.498e-3` | 6.18 |
| 30 (tail) | `5.402e-3` | `6.009e-3` | 14.90 |
| 34 (tail) | `6.661e-3` | `6.205e-3` | 66.06 |

The Q8_1 activation quantization costs a roughly uniform `~0.5-0.67%` relative
error per V projection, on both in-lease and tail blocks. The tail expansion
overshoots the final-logit `1e-3` gate not because the tail quantization is
worse, but because a fresh tail block's `~0.6%` projection error reaches the
output with fewer remaining residual adds to dilute it, while the in-lease
blocks' error is already present in both control and candidate and cancels in
the comparison. This is the same non-monotonic accumulation the 08-05 record
documented, now with the per-projection activation cost pinned to a number.

Census was correct in every arm (for the all-coop 9-block candidate: 34 direct
Q6 consumers, 52 fused providers, 0 legacy shared-Q4; for the mixed arm: 34
direct, 52 fused, 36 expected legacy shared-Q4), proving the wiring is sound
and the block is purely numerical.

## 9. Disposition

- Keep the max17 lease unchanged; do not add any tail Q6 V block.
- Book zero incremental recovery from blocks 0/21/24/27/30-35.
- The `~135us` Q6 V tail device-time gap is precision-gated, not a free kernel
  swap. Unlocking it needs a more accurate shared-Q8 activation path (or a
  re-justified relative-L2 authority), which is a separate scope.
- Block 0 remains unleased on the embedding/REDUCE_OUTPUT boundary.

Raw artifacts: `/tmp/nv_q6_v_tail_semantic_gate.json` (9-block),
`/tmp/nv_q6_v_tail_semantic_gate_7block.json`,
`/tmp/nv_q6_v_tail_semantic_gate_6block.json`,
`/tmp/nv_q6_v_tail_semantic_gate_mixed.json`.

## Evidence to produce

- semantic child comparison JSON (this run)
- reverse wall bracket JSON at d512/d2048/d4096 (this run)
- follow-up production census JSON (only if promoted)
- updated route policy description and unit test (only if promoted)
