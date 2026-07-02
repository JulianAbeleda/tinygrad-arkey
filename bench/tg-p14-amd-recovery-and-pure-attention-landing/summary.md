# TG-P14 Recovery And Scratch Census

TG-P14.0 verdict: **TG_P14_0_PASS_AMD_RECOVERED**

TG-P14.1 verdict: **TG_P14_1_PASS_SCRATCH_CENSUS**

The post-reboot AMD recovery gate is clean. No `python3 -`, `qk_tg`, `AMDKFD`, or `tinygrad` worker remained in the
process table when checked with a self-filtering pattern, and the timeout-guarded AMD smoke completed successfully:

```text
[2.]
```

The dirty codegen scratch is present and in scope for TG-P13/TG-P14 verification:

- `tinygrad/codegen/late/devectorizer.py`
- `tinygrad/codegen/__init__.py`
- `extra/qk_tg_p11_reduce_upcast_microgate.py`

The following dirty benchmark artifacts are unrelated to the compiler landing and remain excluded from any TG-P14
compiler-fix commit:

- `bench/qk-decode-runtime-overhead/result.json`
- `bench/qk-search-spaces/default_route_manifest.json`
- `bench/qk-search-spaces/refuted_axes.json`
- `bench/tg-p8-generated-8b-attention-parity/baseline.json`
- `bench/tg-p8-generated-8b-attention-parity/summary.md`

The current scratch diff is recorded in `scratch_codegen.diff`. The compiler patch is still untrusted until P11
fix-off/fix-on and P10 fixed-mode pass with `REDUCE_ACC_UPCAST_FIX=1`.

No compiler fix was committed. Owned HIP attention remains default.

## P11 Contract

Verdict: **TG_P14_2_PASS_P11_CONTRACT**

Fix-off reproduced the baseline failure exactly: all four P11 rows were `cfail` with invalid vector REG stores.
Fix-on passed all four rows with `REDUCE_ACC_UPCAST_FIX=1` and no invalid vector REG stores.

## P10 Fixed Mode

Verdict: **TG_P14_3_PASS_P10_FIXED_MODE**

The host was power-cycled (boot 2026-07-01 22:42) to clear the earlier AMD reset failure. Post-reboot, the P10
fixed-mode repro passes all required cases under `REDUCE_ACC_UPCAST_FIX=1`, verified deterministic across two
consecutive runs:

- `shipped_per_d_combine` (`qk_flash_decode.flash_state_combine_kernel`): compile ok, numeric ok.
- `shared_weight_combine` (`qk_live_split_geometry.flash_fused_gmax_combine_kernel`): compile ok, **numeric ok**.
- `fused_gmax_combine` (`qk_live_split_geometry.flash_inline_gm_combine_kernel`): compile ok, numeric ok.
- `reg_store_devec_compiles_nan`: not used as a success path (success comes from `REDUCE_ACC_UPCAST_FIX` +
  distinct-slot pass, not `REG_STORE_DEVEC`).

The previously-recorded shared-weight numeric failure (max abs err ~`2.44`, odd private output slots left zero) is
resolved. That case is built from `flash_fused_gmax_combine_kernel`, so this run also **verifies the previously
unverified scratch pin** (`opts_to_apply=()` on `flash_fused_gmax_combine_kernel`): the pin supplies the numeric fix,
the devectorizer distinct-slot pass supplies the compile fix.

Script note: `qk_tg_p10_reg_scalar_repro.py`'s top-line PASS verdict (`TG_P10_1_PASS_REG_REPRO_PINNED`) is a fix-OFF
"is the bug still present" pin (requires `fails!=[]` and `devec_nan=True`), so under the fix it prints
`TG_P10_1_BLOCKED_REPRO_NOT_MINIMAL` and exits 1. TG-P14.3 grades on the per-case compile+numeric data, which is the
intended success shape. See `p10_fixed_mode.json`.

Verified scratch now cleared for the eventual landing:

- `tinygrad/codegen/late/devectorizer.py` + `tinygrad/codegen/__init__.py` — `REDUCE_ACC_UPCAST_FIX` passes
  (default-off, AMD-gated, fail-closed).
- `extra/qk_live_split_geometry.py` — `opts_to_apply=()` pin on `flash_fused_gmax_combine_kernel`.

Per the handoff gate, P14.2 **and** P14.3 pass. BoltBeam reachability and the default-off regression ladder followed.

## BoltBeam Reachability (P14.5)

Verdict: **TG_P14_5_PASS_BOLTBEAM_REACHABLE**

`tests/test_reg_lowering.py` 6/6 pass. The real fixed `reg_scalar_lowering.v1` artifact normalizes+classifies as
`REACHABLE` through `boltbeam/artifacts/tinygrad.py` + `boltbeam/diagnostics/reg_lowering.py`; the old blocked fixture
still classifies `EMITTER_BLOCKED`; no-rows returns `TARGET_INCOMPLETE` (no success from missing rows). Candidate
`decode_attention_g5_8b_refuted` reopen condition updated to cite TG-P14 evidence.

## Default-Off Route Regression Ladder (P14.6)

Verdict: **TG_P14_6_PASS_DEFAULT_OFF_REGRESSION**

Default-off census: `PMS_R0_PASS_CENSUS_PINNED` (default path composition unchanged; the fix is env-gated default-off).
Protected routes with `REDUCE_ACC_UPCAST_FIX=1`:

- generated prefill schedule: byte-identical (`TG_P4_PASS`).
- generated G5 K-only attention: exact, `rel_rmse=0` (`GP3_PASS_MICROGATE`).
- Q6_K generated coop decode: identical (`TG_P3_PASS`).
- Q4_K G3 decode GEMV + owned HIP attention (default path): **NLL bit-identical** fix-off == fix-on =
  `2.8552600837294158` (deterministic teacher-forced `qk_nll_eval.py`).
- P11/P10 compiler repros: still pass.

Two findings during this phase:

1. **Compile crash found and fixed.** With the flag on, the pass crashed model-wide at `devectorizer.py:475`
   (`can't vectorize dtypes.float.vec(4) with size 4`) — it assumed the accumulator base was scalar. Added a
   fail-closed guard `if sdt.count != 1: continue` (generic, no kernel-name branch). Re-verified: P11 4/4, P10 4/4,
   NLL bit-identical.
2. **`qk_decode_token_match_check.py` is non-deterministic as invoked** — it never prefills the KV cache, so decode
   reads uninitialized memory (three identical-config runs gave three different token lists, incl. an all-`151936`
   garbage sample). Its comparisons were discarded; `qk_nll_eval.py` is the deterministic authority used instead.

**Coupled changes:** the codegen fix makes the combine compile; `extra/qk_live_split_geometry.py` (gacc init-dep
refactor + `opts_to_apply=()`) makes it numerically correct. Reverting the latter regresses P10 shared-weight/fused to
`numeric_ok=False`, so both are committed together. The default model is unaffected (bit-identical NLL; model uses only
the unchanged live-split block-tile kernel).

## Split-Preserving Combine Reopen (P14.8)

Verdict: **TG_P14_8_PASS_COMBINE_REOPENED**

The emitter block is lifted: `qk_tg_p9_combine_microgate.py` with the fix on returns `TG_P9_4_PASS_COMBINE_MICROGATE`
(all shapes compile; was `TG_P9_4_BLOCKED_EMITTER`). `qk_tg_p14_combine_reopen.py` confirms all three generated-UOp
combine shapes are **compile + numerically correct** (generated-UOp only, no `REG_STORE_DEVEC`), including the
previously-unverified two-stage fexp-free weighted-sum:

| shape | kernels | fexp | numeric | directional µs (non-authoritative) |
|---|---|---|---|---|
| shipped per-d | 2 | Hq·Hd·S = 147456 | ok | 1346 |
| fused lds-warp | 1 | Hq·S = 1152 | ok | 1085 (~19% faster) |
| two-stage fexp-free | 2 | Hq·S = 1152 | ok | 1463 |

**128× fexp reduction** vs shipped. The fused single-kernel combine is directionally faster; the 128× fexp cut does
not linearize into wall-clock (combine is lifecycle/launch-bound, not fexp-bound), consistent with the ctx4096
combine-overhead cap. Per methodology, isolated combine micro-timing is **not** the promotion authority — the
authoritative speed test is the full W==D at P14.9. Not a speed refute (fused shape is faster + 3→1 kernels + 128×
fexp), so P14.8 passes and P14.9 is warranted.
