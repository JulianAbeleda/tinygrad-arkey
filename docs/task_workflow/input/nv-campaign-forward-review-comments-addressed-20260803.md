# NV campaign forward-path amendment - comments addressed

Date: 2026-08-03
Status: response record. This document records that every forward-path item in
`nv-campaign-forward-review-amendment-20260803.md` section 4 is now addressed by a
committed, verified deliverable. It responds to the reviewer amendment (the
response-of-record); it supersedes nothing and authorizes no implementation. Branch
boundary: tinygrad `nvidia-bringup-20260731`, pushed and in sync at `09cfb4c26`.

## 1. What was asked (amendment section 4, condensed)

Decode: (1) correct the state-of-record, (2) decompose M4 without changing default
behavior, (3) choose one minimal variant-reopen boundary P0 starting with M5, (4)
re-measure before composing, (5) prioritize the current wall authority with GEMV-
efficiency scopes ranked by measured wall opportunity, (6) publish no composed forecast
until isolated measurements exist.

Prefill: (1) keep the measured pp512+ parity record as baseline, (2) scope B3
independently with the required AMD control, (3) measure the polling change against both
NV and AMD before landing, without using decode progress as its gate.

## 2. What landed

| amendment item | deliverable | commit | verification |
| --- | --- | --- | --- |
| Decode 1 - state-of-record | parity record amended: M2 open for NV:sm_120, M3/M4/M5/Path 3 closed | `fcf3774f9` | parity doc now names the reproducible baseline |
| Decode 2 - decompose M4 | isolated per-variant census/wall rows; FFN-down recompute defect confirmed | `09cfb4c26` | controls reproduce the parity baseline; defect is measured, not inferred |
| Decode 3 - M5 boundary P0 | typed output-layout/view-preservation contract, closed-default, consumer-specific | `7a4acdece` | exact UOp chain and all pinned hashes verified against HEAD |
| Decode 5 - GEMV-efficiency scopes | L2/L4/flash substrate items ranked by measured wall opportunity | `d3a748450` | every evidence number and hash verified against the source records |
| Prefill 2 - scope B3 | independent prefill runtime lever scope with AMD control requirement | `54749a342` | wait-path mechanism and all campaign numbers verified against code |
| Decode 4, 6 and Prefill 1, 3 | recorded as gates, not done | n/a | no implementation, no record change, no forecast published |

All four commits are on `nvidia-bringup-20260731` only, pushed
`fcf3774f9..09cfb4c26`, branch in sync with origin. No commit touched
`master`/`dev`/`exp`. The untracked user-owned files
(`extra/llm_research/microbench/dp4a_peak_cuda*`,
`scratchpad/t6_metal_admission_probe.py`) were left untouched.

## 3. Per-item detail

**Decode 1.** `fcf3774f9` amends `nv-decode-parity-final-20260802.md` so the baseline is
reproducible: M2 `decode_epilogue_fusion` open/promoted for NV sm_120 only; M3 norm, M4
Q4K epilogue, M5 combine, and Path 3 semantic RMSNorm closed.

**Decode 2.** `09cfb4c26` is the M4 decomposition measurement record. One variant open at
a time on Qwen3-8B-Q4_K_M, d512 and d4096, nmeas=20, reps=3:

| variant | d512 delta | d512 tok/s | verdict |
| --- | ---: | ---: | --- |
| residual_add (o-proj) | +69 us, +36 kernels | -1.15% | clean; boundary P0 eligible |
| fp16_cast (k/v) | +14 us, 0 net kernels | -0.13% | run noise; overlaps M5 |
| ffn_down_fused | +1321 us, +18 kernels | -18.2% | rejected; recompute defect |

The FFN-down defect claim from amendment section 2.3 is now directly confirmed: fused
`q4k_g3_lanemap_gemv_epi_ffndown_4096_12288` is 98.16 us at d512 / 98.56 us at d4096 vs
legacy 26.23 / 26.34 us, a 3.74x per-kernel regression; the recompute mass is 18 x
(98.16 - 26.23) = 1294.7 us/token at d512. Removing boundary copies cannot make that
shape economical. The combined M4 record stays closed.

**Decode 3.** `7a4acdece` scopes the M5 variant-reopen boundary P0: the combine
`KernelProgram` output declares its typed layout (fp16 `(Hq*Hd,)` row-major), the o-proj
Q4K GEMV (`attn_qo`) opts in via a typed input ABI so its
`reshape(...).cast(fp16).contiguous()` folds to a view of the `AFTER`, and the
`E_32_32_4_3b0fcfbc` copy (36x, ~1.58 us, variant-only) is not emitted. Closed-default
(`decode_flash_combine_fusion` stays closed), fail-closed validator, `custom_kernel`'s
default flat-buffer contract untouched.

**Decode 5.** `d3a748450` ranks the decode GEMV-efficiency work by measured wall
opportunity: A L2 Q6K partial single-pass (~0.25 ms), B L4 vocab substrate fusion
(~0.14-0.24 ms), C flash score tile structure (~0.16 ms), D like-for-like cap discipline
(quantize-excluded 0.92 ms bounds the combined claims). Separate scopes, no composed
forecast, the landed `row_tile=2` values row explicitly kept.

**Prefill 2.** `54749a342` scopes B3 independently: measured evidence (44-46 ms warm
wall, 24.1 ms busy, 23.7-23.8 ms `wait()` CPU, 1.9x wall/busy vs llama's 1.15-1.35x
envelope), three ranked fix shapes (a) cache the signal view, (b) real blocking wait,
(c) whole-schedule graph replay, the AMD runtime control requirement, and a closed-
default gate.

## 4. Verification performed

- pg3 render-equality re-derived against HEAD
  (`scratchpad/pg3_decode_rendered_source_equality.py`): all ten legacy HIP rows
  byte-identical, plus `flash_fused_gmax_combine_f16_32_128` = `94d73c1e9650` and the M2
  fused row `add50a7aa43f` (src_len 9440) matching the docs' pin tables.
- M4 all-closed controls reproduce the wall authority within spread: 172.835 vs 172.80
  tok/s at d512, 149.175 vs 149.00 at d4096; token sha `9d6b3787...` and first token
  `151936` hold 3/3 in every row of every variant.
- Code line references in the scopes were checked against HEAD
  (`flash_decode_attention.py:205,526-530`, `decode_routes.py:104`,
  `ops.py:999-1003,1264-1268`, `hcq.py:262`, `ops_nv.py:27-31,567`).
- `git diff --check` clean on every commit; one-prefix-per-commit rule held
  (`[docs]` throughout).

One naming drift found during verification, recorded here so it is not re-found later:
the B3 evidence describes the per-poll view class as `HcqView`; the current tree names it
`MMIOInterface` (`tinygrad/runtime/support/hcq.py:18`). Same mechanism, same line
numbers, cosmetic only. The `ab3cb84c1` citation for the L4 `row_tile=2` landing
resolves to that commit's object and subject in the local object database.

## 5. What is still open (deliberately not done)

- No implementation. Every deliverable ends in HARD STOP; each next step is a separate,
  variant-specific implementation scope with its settling command, legacy hash controls,
  correctness pins, and fixed-depth wall gate.
- No wall re-measurement of any reopened variant, and no composed decode endpoint. M5's
  section 5 gate (d512/d2048/d4096 wall, all shas, pg3 re-derive) must pass before the
  `decode_flash_combine_fusion` record may change; M4 reopens only under its own scope
  with the FFN-down compute-once redesign.
- The AMD leg of B3 has not run: the polling change must be measured NV+AMD same-session
  before landing, and no AMD GPU exists on this machine.
- The ffn-down residual-only epilogue probe is not implemented (the current spec fuses
  prelude and residual); it does not share the prelude verdict by force.

## 6. References

- `nv-campaign-forward-review-amendment-20260803.md` (the comments being answered)
- `m4-decomposition-measurement-record-20260803.md`
- `m5-variant-reopen-boundary-p0-scope-20260803.md`
- `decode-gemv-efficiency-forward-scope-20260803.md`
- `b3-prefill-host-overhead-scope-20260803.md`
