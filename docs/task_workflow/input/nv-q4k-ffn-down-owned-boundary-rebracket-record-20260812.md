# NV Q4_K FFN-down owned-fp32-boundary layer-8 re-bracket

Status: **WALL WIN for the layer-8 singleton; the predicted two-layer subset is
not advanced.**

Date: 2026-08-12
Branch: `nvidia-bringup-20260731`, HEAD `d63f0a7c9` (post owned-boundary
graph fix + arithmetic trace record `22d82f048`)

## 1. What this arm is

The arithmetic trace (`nv-q4k-q8-substrate-arithmetic-trace-20260812.md`)
confirmed the Q4_K x Q8_1 arithmetic byte-for-byte against pinned llama and
identified one untried, llama-anchored mechanism on the Q4 FFN-down row
(option A): the owned-fp32-boundary MMVQ (`owned_boundary_topology()` in
`tinygrad/llm/q4k_ffn_down_mmvq.py`). It replaces the 3-node control chain
(w1w3 fused fp32 -> fp16 materialize -> installed `q4k_g3_lanemap_gemv_4096_12288`)
with a 3-node candidate chain (w1w3 fused fp32 -> `q8_1_llama_provider_12288`
consuming the fused output directly -> `q4k_q8_mmvq_direct_4096_12288`),
net-zero graph members, and the fp32->fp16 rounding moves inside the provider
(byte-identical to the former materialized boundary, test
`test_owned_boundary_rounding_matches_the_existing_q8_input_authority`).

This arm runs the layer-8 singleton only, under the settled 08-09 reverse
bracket protocol, with the owned boundary leased on layer 8.

## 2. Graph fix found by the gate (OBSERVED)

The first candidate census failed the exact topology contract: the owned
route passed the fused w1w3 output's RESHAPE view into the opaque provider,
and `UOp.custom_kernel` conservatively materialized any non-AFTER input,
inserting a new `E_128_32_3` fp32 identity copy (+2 calls; 1874 -> 1876).
The fix hands the provider the AFTER itself after stripping equal-span
reshapes (`q4k_ffn_down_mmvq.py`, fail-closed: any other shape returns None
and the installed path runs). Regression test
`test_owned_boundary_strips_equal_span_reshapes_to_the_after_without_a_copy`.
The re-run census then showed the exact expected delta and net-zero count.

## 3. Census (post-fix, composed, count 8)

Same protocol as the 08-09 census child: d512, 8 x 1 x 151936 full logits,
composed ping-pong, accepted-attention max17, `CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1`,
`DEV=NV`, every child under `flock -w 600 /tmp/gpu-bench.lock`.

| arm | program calls | finite | topology pass |
| --- | ---: | --- | --- |
| control (no lease) | 1874 | true | true |
| layer-8 owned candidate | 1874 | true | true |

Exact program delta (control -> candidate): `E_128_32_3_580aa...` (fp16
materialize) -2, `q4k_g3_lanemap_gemv_4096_12288` -2,
`q8_1_llama_provider_12288` +2, `q4k_q8_mmvq_direct_4096_12288` +2; no other
delta. Net-zero graph confirmed. Token streams identical between arms.
`logits_sha256` differs (expected: the Q8 quantization delta, documented
precision gate; not the gate for this arm).

## 4. Settled reverse bracket

All arms: d512, 32-token uninterrupted windows, five repetitions, zero
rejected samples, composed ping-pong (P5), accepted-attention max17 (P1/P2,
`CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1`), Qwen3-8B-Q4_K_M, `DEV=NV`.
Control = installed chain; candidate = owned-boundary provider + direct
consumer leased on layer 8 only.

| arm | ms/token |
| --- | ---: |
| control A | 5.30868034375 |
| layer-8 owned candidate | 5.30278765625 |
| control C | 5.30545625 |
| control midpoint | 5.307068296875 |
| candidate minus midpoint | **-0.004280640625 ms/token** |

| row | value |
| --- | --- |
| candidate speedup | 0.0807% vs control midpoint |
| all token hashes equal | true |
| token stream hash | `f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905` |
| rejected samples | 0/5 per arm |
| verdict | **WALL_PASS** |

The fresh wall is **-4.280640625 us/token**, materially better than the
fp16-boundary re-bracket at the post-census HEAD (-3.190 us/token, 08-09) and
an inversion of the 08-05 NO-GO (+6.205 us/token) measured on the older
composition. The token stream hash matches the 08-09 bracket exactly.

## 5. Boundary (fail-closed)

This is a WIN for the layer-8 singleton only. The predicted two-layer subset
(blocks 8,16) stays closed: no second layer is advanced or recommended from
this arm, per the landing-scope hard stops. This record does not alter the
promotion record, the 08-05 FFN-down record, the shared-Q8 attention lease,
or any default-closed admission. No loader policy, no emitter change outside
the research admission, no promotion on another target. The precision gate
(rel L2 `0.0020060` vs `0.001` maximum) and the cooperative-provider wall
record (+151.192 us) stand unchanged. Any future FFN-down expansion must
return through a fresh semantic + settled wall gate.

## 6. Raw artifacts

- `/tmp/nv-ffn-down-owned-20260812/timing.json` SHA-256
  `8d4fc42cebf05bdfc7f05dbee54610283b465476f9a18478cd4d508439da8463`
- `/tmp/nv-ffn-down-owned-20260812/control.json` SHA-256
  `9e37bbd50460c1547253f60c2eaffcae16b4c72a5a22e2affdde388587646e9d`
- `/tmp/nv-ffn-down-owned-20260812/candidate2.json` SHA-256
  `00a6abb16995791ff3e12a9f0798c360ec394f20a8e6d56ff40eeda13fcb1e73`

## 7. References

- `nv-q4k-q8-substrate-arithmetic-trace-20260812.md` (arithmetic confirmed;
  option A as the next measured arm)
- `nv-q4k-ffn-down-layer8-rebracket-record-20260809.md` (fp16-boundary
  singleton, -3.190 us/token)
- `nv-q4k-ffn-down-mmvq-included-cost-and-one-layer-record-20260805.md`
  (precision gate fail, standalone 13.952 us)
- `nv-gemv-substrate-landing-scope-20260808.md` section 5.2 (one re-bracket
  authority; two-layer subset hard stop)
- commit `d63f0a7c9` (owned-boundary AFTER fix + harness
  `--owned-input-boundary` + regression test)

