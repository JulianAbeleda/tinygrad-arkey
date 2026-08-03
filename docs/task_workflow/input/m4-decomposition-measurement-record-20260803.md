# M4 decomposition measurement record - isolated per-variant rows

Date: 2026-08-03
Status: measurement record. Authorized by
`nv-campaign-forward-review-amendment-20260803.md` section 2.3 and section 4.1 item 2:
decompose M4 into isolated census/wall rows before any reopen claim. It changes no code
and no promotion record; the M4 combined record stays closed (section 6). Branch boundary:
tinygrad `nvidia-bringup-20260731`.

## 1. Why this record exists

The M4 combined record (`m4-q4k-epilogue-measurement-record-20260802.md`) attributes the
non-landing to boundary copies, but its own totals leave ~1075 us/token unexplained after
deleting every copy arithmetically (amendment section 2.3). The implementation hinted at
the missing mass: the `ffn_down_fused` variant loads `gate_out[idx]` and `up_out[idx]` and
evaluates `_silu_uop(g) * u` inside the reduction for every output row
(`decode_kernels.py:156-174`). This record isolates each of the three variants with one
open at a time, proves the FFN-down recompute is the dominant defect, and gives the
o-proj and k/v pieces their own clean rows so each can be judged independently.

## 2. Protocol

Probe: `/tmp/m4_decomp_probe.py` (`--variant {none,residual_add,fp16_cast,ffn_down_fused}
--depth N --nmeas 20 --reps 3`).

- Model: Qwen3-8B-Q4_K_M, max_context 4608, temperature 0, chunk_size 32, NV sm_120.
- One-variant-open admission: the q4k gate target set is opened, then each linear's
  `route_admission.q4k_epilogue_fusion_promoted` is forced True only for the variant's
  route_role (`attn_qo`, `attn_kv`, `ffn_down`) and False for every other role.
- All-closed control: `_DECODE_Q4K_EPILOGUE_FUSION_PROMOTED_TARGETS = frozenset()` and no
  admission surgery.
- Fused prefill attention is disabled (HEAD-broken on NV, house convention) in every run.
- Census: one decode token at DEBUG=2, per-kernel medians summed by class; wall: 20
  measured tokens x 3 reps, median tok/s; pins: token sha256 and first token per rep.

## 3. All-closed control (reproduces the parity baseline)

| depth | kernels/token | E_ | r_ | kernel us/token | tok/s median |
| --- | ---: | ---: | ---: | ---: | ---: |
| d512 | 1021 | 528 | 167 | 6181.6 | 172.835 |
| d4096 | 1021 | 528 | 167 | 7079.5 | 149.175 |

Matches the wall authority within spread: `nv-decode-parity-final-20260802.md` records
172.80 at d512 and 149.00 at d4096. Token sha256
`9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` 3/3 and first token
`151936` 3/3 at both depths.

## 4. Isolated per-variant rows

| variant | role | depth | kernels/token | E_ | kernel us/token | delta us | tok/s | delta tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| residual_add | attn_qo | d512 | 1057 (+36) | 564 | 6250.6 | +69.0 | 170.852 | -1.15% |
| residual_add | attn_qo | d4096 | 1057 (+36) | 564 | 7165.5 | +86.0 | 147.484 | -1.13% |
| fp16_cast | attn_kv | d512 | 1021 (0) | 528 | 6195.2 | +13.6 | 172.610 | -0.13% |
| fp16_cast | attn_kv | d4096 | 1021 (0) | 528 | 7096.1 | +16.6 | 149.053 | -0.08% |
| ffn_down_fused | ffn_down | d512 | 1039 (+18) | 546 | 7503.0 | +1321.4 | 141.375 | -18.2% |
| ffn_down_fused | ffn_down | d4096 | 1039 (+18) | 546 | 8414.1 | +1334.6 | 124.938 | -16.2% |

Pins hold in every row: token sha256 `9d6b3787...` 3/3, first token `151936` 3/3.

The ffn_down_fused piece alone accounts for the entire combined M4 regression (-18.2% of
the combined -18.8% at d512); the other two pieces together are ~-1.3%, and fp16_cast is
inside run noise. The combined record's +1264 us/token decomposes cleanly: +1321 us from
the FFN-down variant alone (with its gate/up/normed_h input copies inside), offset by the
small clean costs of the other two when all three are open.

## 5. The FFN-down defect is verified, not inferred

`q4k_g3_lanemap_gemv_epi_ffndown_4096_12288` (18x/token) measures **98.16 us at d512 /
98.56 us at d4096**, against the legacy `q4k_g3_lanemap_gemv_4096_12288` at **26.23 /
26.34 us** - a 3.74x per-kernel regression. The recompute mass is 18 x (98.16 - 26.23) =
1294.7 us/token at d512, which is the load-bearing component of the +1321.4 us delta; the
remainder (~27 us) is the boundary input copies for gate_out/up_out/normed_h (the isolated
census shows +18 E_ kernels over the control).

This is the direct confirmation of amendment section 2.3's defect claim: the fused prelude
recomputes the nonlinear activation across the 4096 output rows instead of computing the
12288-element activation once. Removing boundary copies cannot make that shape economical.
The rendered fused source (`/tmp/m4_ffndown_fused.cu`, captured during this session) shows
the per-row `_silu_uop` evaluation inside the reduction loop; the legacy source
(`/tmp/m4_ffndown_legacy.cu`) reads a materialized activation.

## 6. Per-piece verdicts (mapped to amendment 2.3)

- **o-proj residual epilogue** (`residual_add`): small, clean, isolated: +36 kernels,
  -1.1% at both depths, no downstream class churn. Eligible for a narrow boundary P0 and
  isolated measurement, exactly as the amendment says.
- **k/v fp16 output** (`fp16_cast`): net-zero kernel count, ~noise wall (-0.13% / -0.08%),
  fused kernel 4.83 us vs legacy 4.86 us at parity. Overlaps M5's producer/output-layout
  problem; it now has its own isolated output-contract measurement.
- **ffn-down SiLU/multiply prelude** (`ffn_down_fused`): rejected with direct evidence.
  The current shape recomputes the activation per output row (3.74x per kernel, +1295 us
  recompute mass); a redesign must compute the activation once (producer-side gate/up
  fusion or a separately materialized activation) before any reopen claim.
- **ffn-down residual epilogue**: not separable in the current spec - the only ffn_down
  variant fuses prelude and residual into one kernel (`Q4KGEMVEpilogue.kind`), so no
  residual-only row exists. It does not share the prelude verdict by force; a
  residual-only variant is a design probe for a future scope, not measured here.

## 7. Status and boundary

- The M4 combined record `decode_q4k_epilogue_fusion` stays closed. No code changed; no
  promotion record changed; the pg3 legacy hashes remain the pre-session bytes.
- This record authorizes no implementation and no record change. A successful M5 boundary
  P0 does not reopen M4; M4 reopens only under its own scope with the FFN-down redesign
  and the amendment's fixed-depth wall + sha gate.

## 8. References

- `nv-campaign-forward-review-amendment-20260803.md` sections 2.3, 4.1
- `m4-q4k-epilogue-measurement-record-20260802.md` (combined record, stays closed)
- `nv-decode-parity-final-20260802.md` (wall authority and pins)
- `decode_kernels.py` `Q4KGEMVEpilogue` (`decode_kernels.py:128-174`)
- `nv-campaign-forward-review-amendment-20260803.md` section 2.2 (M5 boundary P0 scope)
