# q4k w1+w3 fused QV implementation record - scalar shape LANDED, quad shape measured NO-GO in-loop

Date: 2026-08-03 (measured 2026-08-03, one flocked session per arm, arms interleaved)
Status: implementation record for the MC3 coordination handoff (mc3-w1w3-fusion-measurement-record-20260803.md
section 6) and the MC2 load-pattern handoff (mc2-load-pattern-measurement-record-20260803.md section 7),
authorized by `decode-gemv-instruction-bandwidth-scope-20260803.md` section 4.3 and its section 10
amendment (per-item HARD STOP lifted; stop only on a blocker with no named next step, or on the MC3
coordination handoff). Authority chain: `nv-decode-parity-final-20260802.md` (wall authority, harness
protocol) and `nv-parity-and-beyond-forward-scope-20260803.md` (canonical parity criterion).
Branch: tinygrad `nvidia-bringup-20260731`; this scope's commits sit on `4e9f5398f` (6 local commits
ahead of origin, pushed with this record). Tracked tree otherwise clean except the three user-owned
docs (never committed here); untracked user artifacts untouched.

## 1. Change summary

ONE fused decode GEMV route for the ffn gate/up pair: `z = silu(dot(gate_row, x)) * dot(up_row, x)`
in a single 12288-row kernel, replacing the gate GEMV + silu elementwise + up GEMV + mul elementwise
chain (72 -> 36 kernels/token, -36 GEMV-class kernels and -36 E kernels net).

- `decode_kernels.py`: new `q4k_g3_lanemap_gemv_w1w3_kernel(rows, k, load_style="scalar")` (two
  accumulators in one reduce loop via the production flash-decode pattern: one init chain, first update
  `.after()` without `.end`, last update ends the range; distinct REG slots 20/21, distinct shuffle
  staging bases 90-94/95-99). Silu uses `_silu_uop`, the exact Tensor.silu lowering. The scalar style
  reproduces the installed 32-lane/1-row-per-block map and the installed per-lane accumulation order.
  The quad u128 style (MC2 shape: 16 rows/block, 8 lanes/row, wc-quad, x smem staging, XOR ladder)
  is also emitted but is the standalone optimum only; see section 5 for its measured in-loop NO-GO.
  Legacy kernel hashes and rows untouched (pg3, section 3).
- `decode_routes.py`: new `q4k_gate_up_primitive_linear_call(gate, up, x, fallback)` selecting the
  fused program only when BOTH linears bind the legacy decode GEMV shape, T==1, K/N equal, and both
  carry `w1w3_fusion_admitted` (their own QKPrimitiveRouteAdmission field). Any mismatch falls back
  to the legacy graph unchanged.
- `qk_primitives.py`: `QKPrimitiveRouteAdmission.q4k_w1w3_fusion_promoted` field (closed default,
  separate record from M2/M4) + `w1w3_fusion_admitted` property, resolved once per install.
- `model.py`: fused branch in `_feed_forward` (decode only, before the legacy tail) behind the
  resolve-once `_decode_q4k_w1w3_fusion_promoted` block flag; the fallback lambda reproduces the
  legacy chain's z exactly.
- `model_route_plan.py`: `load_decode_q4k_w1w3_fusion_promotion` + `decode_q4k_w1w3_fusion_promoted`
  (closed default, boltbeam.route_policy.v1, separate JSON record).
- `generated/decode-q4k-w1w3-fusion-route-policy.json`: promotes `{("NV", "sm_120")}` only. The
  landed shape is the scalar load style; the quad style is documented measured NO-GO in-loop and is
  NOT selected by the route.
- NOT done, per the binding bans: no `prefill_routes.py`, no new subsystem, no
  `BOUNDED_PACKED_TILES`, no dtype/precision change, no change to any dtypes.* literal, no AMD/Metal
  admission (they need their own measured record plus pg3 hash parity).

## 2. Unit results

`pytest test/unit/test_q4k_w1w3_fusion_gate.py test/unit/test_q4k_epilogue_fusion_gate.py
test/unit/test_decode_epilogue_fusion_gate.py test/unit/test_flash_combine_fusion_gate.py
test/unit/test_llm_decode_kernels.py test/unit/test_llm_decode_routes.py
test/unit/test_llm_decode_correctness.py test/unit/test_qk_capability_policy_gate.py
test/unit/test_model_route_plan.py test/unit/test_qk_route_purity.py`: 100 passed, 0 failed.

New `test/unit/test_q4k_w1w3_fusion_gate.py` (11 tests, CPU-only): closed default when the record has
no `promoted_targets` key or an empty list; the loader names explicit targets only; the checked-in
record promotes exactly `{("NV", "sm_120")}`; `w1w3_fusion_admitted` never changes the legacy
`admitted` routes; the route falls back on off-target, off-shape, missing admission, K/N mismatch,
and non-single-token binds; both fused styles render through HIPRenderer and CUDARenderer without a
GPU (0-spill gate at nvcc 13.2 in the probe, not in unit).

One pre-existing failure unrelated to this scope, confirmed failing at HEAD before any of this
scope's code was present (verified by stash): `test/unit/test_decode_norm_fusion_gate.py::
test_spec_validate_rejects_bad_contracts` (the `x_rank=2` case does not raise). Not touched here.

`sz.py`: authored budgeted 40143 / 100000. `git diff --check`: clean.

## 3. pg3 decode render-equality (HIPRenderer gfx1100, render-only, CPU, no lock)

`scratchpad/pg3_decode_rendered_source_equality.py` re-run this session: all 21 rows byte-identical
to the pinned `/tmp/pg3.log` table, including the 10 legacy hashes (312422c73a49 / 27857cb8ca03 /
851760e2053c / 39ddb717ddd4 / cc38fbb3db92 / 5795e66a7292 / 344e1c388eeb / c708302aa2d2 /
66d4c4da3108 / c78e4651ad35) and the M2 promoted fused row `q6k_gen_coop_4096_12288_inkernel` =
`add50a7aa43f`. The w1w3 emitter is additive-only in `decode_kernels.py` (+142 lines, zero
modifications to existing functions); AMD/Metal rendered sources are unchanged by construction and by
this control.

## 4. Protocol (all arms, this session)

- Same RTX 5090 / sm_120 box as the wall authority, driver 595.84, CUDA 13.2, Qwen3-8B-Q4_K_M.
  Every GPU run under `flock /tmp/nv_gpu.lock`, 0% util confirmed at lock acquisition, fused prefill
  attention disabled (`tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()`).
- Harness: `extra/llm_research/decode/gemv_class_census_nv.py` (d512 fixed depth, DEBUG=2 prime token
  census, 20 measured tokens x 3 reps tok/s, token sha256 + first token pins). Arms differ only by
  the w1w3 route state: quad (route as initially written), closed (monkeypatched
  `decode_q4k_w1w3_fusion_promoted` -> False in all three import namespaces before model build),
  scalar (route forced to `load_style="scalar"`). Arms interleaved in one session: quad, closed, quad,
  scalar, scalar, closed, scalar. A final as-committed run (route now calls scalar explicitly) closes
  the record.
- Standalone: `extra/llm_research/microbench/w1w3_quad_render_probe.py`, emitted geometry grids,
  nvcc 13.2 `-arch=sm_120 -O3 --ptxas-options=-v` (0-spill gate), cuobjdump + nvdisasm SASS census,
  host-looped best-of timing, rounds interleaved, finite Q4_K fill.

## 5. Results

### ptxas / SASS (OBSERVED, standalone probe)

| kernel | regs | spills |
| --- | ---: | ---: |
| q4k_g3_lanemap_gemv_12288_4096 (pair arm) | 61 | 0 |
| q4k_g3_lanemap_gemv_w1w3fused_12288_4096 (scalar fused) | 74 | 0 |
| q4k_g3_lanemap_gemv_w1w3qv_12288_4096 (quad fused) | 72 | 0 |

Quad SASS: 10 LDG.128 weight loads, LDS=16 / STS=8 (x smem staging), FFMA=517, BAR=1, gated store
(`if ((alu7==0))`). No LDL/STL anywhere. Standalone timing: gate 17.1 us, up 17.1 us, pair 34.2 us,
scalar fused 23.2 us, quad fused 22.2 us.

### Same-session in-loop A/B (OBSERVED, census harness, one session, interleaved arms)

| run | route | tok/s median | kernels/token | kernel us/token | GEMV-class count / ms | fused kernel in-loop median |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | quad | 165.7 | 948 | 6318 | 216 / 4.137 | 49.2 us (n=36) |
| 2 | closed | 174.1 | 1020 | 6132 | 252 / 3.865 | - |
| 3 | quad | 166.1 | 948 | 6316 | 216 / 4.129 | 49.2 us (n=36) |
| 4 | scalar | 176.9 | 948 | 5963 | 216 / 3.775 | 39.4 us (n=36) |
| 5 | scalar | 177.1 | 948 | 5985 | 216 / 3.790 | 39.4 us (n=36) |
| 6 | closed | 173.7 | 1020 | 6133 | 252 / 3.867 | - |
| 7 | scalar | 176.8 | 948 | 5967 | 216 / 3.778 | 39.4 us (n=36) |
| 8 | scalar (as-committed) | 176.6 | 948 | 5980 | 216 / 3.789 | 39.4 us (n=36) |

Pins 3/3 in every arm: token sha256 `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9`,
first token `151936`.

Class arithmetic (INFERRED from the OBSERVED rows): the scalar fused kernel replaces the pair's
2 x 20.83 = 41.7 us with 39.4 us (-2.3 us x 36 = -83 us) and removes the silu + mul elementwise pair
(72 E kernels -> 36; the E_128_32_3 hash set changes from 2ba5/4a0d (36x each, the legacy chain's two
per-call fp16 x casts) to 580a (36x, the fused route's single x cast), -79 us), netting the observed
-166 us/token (6132 -> 5966 average). The quad fused kernel instead costs 49.2 us (+7.5 us x 36 =
+270 us) minus the same -79 us E removal, netting the observed +186 us/token (6132 -> 6317 average).
No `E_32_32_4_86a2` boundary-copy explosion appears in either arm (unlike M4): the fused route's z is
consumed by ffn_down through the same custom-kernel boundary the legacy chain already had.

Effective DRAM streaming (56.6 MB of weights per fused call or per pair, INFERRED division of
OBSERVED times): scalar fused 1.44 TB/s vs pair 1.36 TB/s vs quad 1.15 TB/s in-loop. Standalone
(L2-resident): quad 2.55 TB/s, scalar 2.44 TB/s, pair 1.66 TB/s. The quad's standalone-to-in-loop
penalty (2.2x) is anomalous versus the pair (1.2x) and scalar (1.7x); its mechanism (u128 load
pattern, 16-row blocking, smem staging, or occupancy) is NOT identified here and is left as an open
question, not a named cause.

### Numerics (OBSERVED, probe)

Scalar fused reproduces the composed pair output to the in-kernel exp2 approximation exactly like
MC3 (20/12288 bit mismatches vs host exp2f, max rel 1.97e-7); because the scalar style preserves the
installed per-lane accumulation order, in-graph z is bit-identical to the legacy chain whenever the
per-row dot totals match (and the fixed-depth pin holds 3/3). Quad differs by fp32 accumulation order
only (max abs 496 on ~1e8 rows, rel ~3e-6; catastrophic-cancellation rows up to 5% rel when the gate
dot is near zero, e.g. row 3032 pair_g=0.00977); the pin still holds 3/3 with quad active.

## 6. Deviations

- The initial working-tree route called the kernel with the quad default; the A/B showed quad
  regresses wall (-5%, reproduced twice), so the landed shape is the scalar style, with the route
  calling `load_style="scalar"` explicitly and the kernel default changed to scalar. The quad emitter
  stays (standalone optimum, kept for occupancy/load-pattern study) but is never selected by the route.
- The first A/B driver captured stdout via `sys.stdout` swap and produced an empty buffer despite a
  completed run (JSON was printed, capture failed); that attempt's numbers were discarded and the
  harness was switched to running the census copy as a subprocess with output redirection. The census
  loader prints a `max_context=...` line to stdout before the JSON; the analysis strips to the first
  `{` line. No repo code involved.
- The record's first census session (from the handoff) showed the same quad regression (216/4.129 ms,
  165.5-165.9 tok/s) and is consistent with runs 1/3; it is superseded by this session's interleaved
  table.
- `test_checked_in_record_promotes_only_the_measured_nv_target` asserts NV sm_120 promoted, which
  remains true after the scalar landing.

## 7. Closeout

The fused w1+w3 decode GEMV route is LANDED on NV sm_120 in the scalar load style: same-session wall
+1.7-2% tok/s (176.8-177.1 vs 173.7-174.1 closed), -166 us/token kernel time, pins 3/3, pg3
byte-identical, 100 unit tests green. The quad u128 style is recorded measured NO-GO in-loop (-5%
wall, reproduced twice) and is not promoted or selected; its in-loop penalty mechanism is open.
AMD/Metal admission requires their own measured record plus pg3 hash parity, per the binding contract.

## 8. References

- `mc3-w1w3-fusion-measurement-record-20260803.md` (fused shape viability, standalone numbers)
- `mc2-load-pattern-measurement-record-20260803.md` (quad u128 load pattern, standalone numbers)
- `decode-gemv-instruction-bandwidth-scope-20260803.md` sections 3, 4.3, 5, 10
- `m4-q4k-epilogue-measurement-record-20260802.md` (record style; the boundary-copy contrast)
- Probes: `extra/llm_research/microbench/w1w3_quad_render_probe.py`, `extra/llm_research/decode/gemv_class_census_nv.py`
