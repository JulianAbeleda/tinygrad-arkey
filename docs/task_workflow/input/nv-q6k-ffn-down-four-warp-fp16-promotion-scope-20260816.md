# NV Q6_K FFN-down four-warp fp16 promotion: re-bracket scope (2026-08-16)

Date: 2026-08-16
Branch: `nvidia-bringup-20260731` (HEAD `dbc1fdee3`)
Status: **promoted. Reverse wall bracket WALL_PASS (-39.0 us/token), production
census 200.81 tok/s with the token sha unchanged.**

## 1. Why this is the next step

The capture-safe schedule prune (`dbc1fdee3`) removed the 72 dead capture
kernels and restored the production decode wall to ~197-199 tok/s. That was the
blocker named in the Q6 route policy: the Q6 four-warp kernel win was
wall-masked by the `df3dca075` regression, so the route was kept closed.

The route itself was already measured device-time faster before the regression:

| side | kernel | device time | DRAM |
| --- | --- | ---: | ---: |
| control | `q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd`, row_tile 2 | 31.0 us | ~7550 GB/s |
| candidate | `q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd`, four warps/row | 25.7 us | ~9630 GB/s |
| llama floor | `mul_mat_vec_q` Q6 FFN-down | 28.75 us | - |

The candidate is `-5.2 us/node` (`-17%`) with `+27%` DRAM bandwidth and lands
below the llama floor on this node. The Q4 sibling of this exact geometry change
already promoted (`-100.3 us/token`, `+2.01%`, token-exact), so this is the
highest-confidence remaining kernel-work lever on the 240 path.

## 2. What changes

Nothing is newly implemented for the test. The production emitter and model
wiring already exist:

- `tinygrad/llm/q6k_ffn_down_mmvq.py`: closed-default `Q6KFFNDownMMVQAdmission`
  and `emit_q6k_four_warp_fp16_direct`.
- `tinygrad/llm/decode_routes.py`: returns the candidate only behind the
  explicit admission lease.
- `tinygrad/llm/model.py`: installs the admission only when the route policy
  target is promoted.
- `tinygrad/llm/generated/decode-q6k-ffn-down-fp16-geometry-route-policy.json`:
  flipped to `"promoted_targets": [{"backend": "NV", "architecture": "sm_120"}]`.

The candidate swaps 18 Q6_K FFN-down blocks 1:1 from the installed row_tile-2
coop consumer to a 128-thread, four-warp fp16 direct consumer with the same Q6
dequant arithmetic, no Q8 provider, and the M2b residual add absorbed in-kernel.
Program count is expected to stay 596 on the production census.

## 3. Decisive test

The existing reverse wall bracket isolates the Q6 change:

```text
PYTHONPATH=. .venv/bin/python extra/llm_research/decode/q6k_ffn_down_four_warp_wall_bracket.py \
  --depth 512 --count 32 --max-context 1024 --reps 5 \
  --out /tmp/q6k_ffn_down_four_warp_wall_bracket_20260816.json
```

It runs control A / candidate / control C in three fresh, lock-bounded
processes. Both arms keep the already-promoted Q4 four-warp route and the M2b
residual-add absorption, so the only inter-arm delta is the Q6 admission.

Measured result (2026-08-16):

| arm | ms/token |
| --- | ---: |
| control A | 4.99931 |
| candidate | 4.96094 |
| control C | 5.00065 |
| control midpoint | 4.99998 |
| delta | **-0.03904 (-39.0 us)** |
| speedup | **+0.787%** |
| token streams | identical across all three arms |

Acceptance for promotion:

1. all three arms produce the same token stream hash;
2. candidate median is below the control A/C midpoint;
3. the measured delta is large enough to book (target `-30 us/token` or better,
   i.e. roughly `+0.6%` tok/s in this band, rather than a sub-noise negative);
4. kernel/program count stays 596 on a follow-up production census.

The single-node microgate was launch-bound and showed only `-0.25 us` on its
replay wall despite the `-5.2 us` device win. The wall bracket is the
authoritative test: it decides whether that device win survives graph launch
overhead in the real decode stream.

## 4. Promotion if the test passes

1. Set `promoted_targets` to `[{"backend":"NV","architecture":"sm_120"}]` in
   `tinygrad/llm/generated/decode-q6k-ffn-down-fp16-geometry-route-policy.json`.
2. Update `test/unit/test_q6k_ffn_down_mmvq.py` so the closed-default assertion
   becomes a promoted-target assertion for `("NV", "sm_120")` while remaining
   closed for other backends.
3. Re-run the production census and confirm 596 kernels, token sha
   `227ad3ce...`, and the Q6 down family swap.
4. Commit the bracket evidence, the policy flip, and the test together.

Follow-up production census (2026-08-16):

| metric | before promotion | after promotion |
| --- | ---: | ---: |
| kernels/token | 596 | 596 |
| token sha | `227ad3ce...` | `227ad3ce...` (exact) |
| census tok/s | 197.29-198.88 | **200.81** |
| Q6 FFN-down family | `q6k_gen_coop_..._epi_ffnresadd` x18, 35.1 us | `q6k_fp16_mmvq_direct_..._epi_ffnresadd` x18, 32.0 us |

The swap is 1:1, program count unchanged, and the token stream is bit-identical.

## 5. If the test is flat or slower

Do not promote. Keep the policy closed and record the wall bracket as the
NO-GO evidence. The next lever then shifts from this Q6 geometry swap to the
larger reduce-output/flash/vocab fusion work already scoped in the 220
composition review.

## Evidence to produce

- reverse wall bracket JSON (this run)
- follow-up production census JSON (only if promoted)
- updated route policy and unit test
