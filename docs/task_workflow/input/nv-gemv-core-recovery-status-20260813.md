# NV GEMV-core recovery status - two unblocks measured, DP4A successor needs re-derivation (2026-08-13)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `d548629da`)
Status: **ledger update.** Integrates the two GPU-gated unblocks authorized today
into the corrected GEMV-core deficit (`nv-gemv-core-deficit-correction-20260813.md`).
Neither change is promoted; both are honest measurement records and the production
tok/s is unchanged.

## 0. Where we are in the ledger

| item | value |
| --- | --- |
| tinygrad decode tok/s (DEV=NV, d512) | ~193.5 |
| llama fresh / pair tok/s | 248.0 / 245.5 |
| wall gap | ~1130 us/token |
| GEMV core deficit (recoverable, +12 tok/s) | +302.8 us |
| flash body device gap (in-situ) | +68 us |
| launch-hiding transferable ceiling | ~33 us |

The GEMV core deficit is the per-shape subtraction of the pinned llama semantic join
(`nv-decode-llama-live-gemv-route-audit-20260805.md`): llama MMVQ core 3579.8 us vs
tinygrad native core 3882.6 us. Three rows dominate it.

| population | llama us | native us | native - llama | status after today |
| --- | ---: | ---: | ---: | --- |
| Q6 attention V (18) | 89.4 | 307.3 | +217.9 | open (Q6 substrate, not touched today) |
| Q4 FFN down (18) | 346.2 | 443.2 | +97.0 | load-pattern re-closed NO-GO; DP4A successor stale |
| Q6 FFN down (18) | 520.8 | 601.9 | +81.1 | open (Q6 substrate, not touched today) |
| Q4 attention K (36) | 117.4 | 152.4 | +35.0 | tail |
| Q6 vocab (1) | 303.6 | 314.4 | +10.8 | near parity |
| Q4 V / gate-up / Q / O | - | - | -0.7 .. -106.7 | ahead of llama |

## 1. Unblock 1 - Q4 FFN-down quad-u128-smem re-census: NO-GO in-loop

Commit `4337ff29e`. The 08-12 load-pattern NO-GO was written against a copy-pasted
wrong floor (11.776 us/node, attention-O's value). The correct floor is 19.23 us/node.
The standalone winner (`q4kd_16row_128thr_u128_quad_xsmem`, 11.43 us standalone) was
re-censused in the real d512 decode loop under the research-only admission hook.

Result: token-exact (sha256 and first token identical across all arms), but the quad
geometry regresses in-loop to **34.48 us median** vs the installed control
**26.24-26.29 us**, and the wall moves +38.3 us/token on 6 leased blocks. This is the
same MC2 pattern as the gate/up quad (standalone win, in-loop smem/occupancy regression).

Consequence: the Q4 FFN-down **load-pattern** row is closed NO-GO against the correct
floor. The quad emitter stays in the tree research-only (admission-gated, production
byte-identical) but is not promoted. The +97.0 us row must be recovered through DP4A,
not a faster scalar load pattern.

## 2. Unblock 2 - producer-owned Q8_1 DP4A successor: stale against M2a/M2b

Commit `d548629da`. The successor (`Q4KFFNDownMMVQAdmission(16,
owned_input_boundary=True)`) was gated and found to be unmeasurable as pre-wired: the
promoted control moved beneath it hours after the candidate commit.

Two promotions landed 08-12 after the successor commit `d63f0a7c9` (17:48):

| promotion | effect |
| --- | --- |
| M2a `decode_q4k_w1w3_fp16_store` (18:53) | w1w3 producer stores fp16 in-kernel (`w1w3fused16`); the fp32 AFTER the owned boundary consumes no longer exists |
| M2b `decode_ffn_down_resadd` (19:45) | ffn_down GEMV absorbs `h+ffn_out` in-kernel and threads `normed_h`; installed kernel renamed to `q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` |

Call-site probe on this HEAD: `x.dtype == fp16` (owned branch requires fp32) and
`epilogue_inputs == {"normed_h"}` (the call's first guard rejects any epilogue). Both
guards return `None`, so the candidate silently falls back to the installed route. The
harness control arm additionally fails its own stale topology check (expects
`q4k_g3_lanemap_gemv_4096_12288`, census shows 18x `_epi_ffnresadd` and 0x the old name).
The timing bracket therefore measured control-vs-control (+8.62 us inter-arm drift), not
the candidate.

Consequence: the DP4A successor is still unmeasured and closed by default. It is not a
candidate bug and there is no minimal candidate-only fix.

## 3. What the re-derivation must now target

The pre-M2a/M2b owned-boundary contract ("remove one fp16 materialize, net-zero nodes")
no longer matches production. Against the current promoted control the DP4A path must be
re-derived as:

- producer: fold llama Q8_1 quantization into the existing w1w3 fused kernel's epilogue
  (it already owns the silu*up result), replacing the fp16 store with a packed Q8_1 store,
  so no separate provider node is added;
- consumer: extend the four-warp DP4A consumer to absorb the M2b `h + ffn_out` residual
  add in-kernel (matching the installed `_epi_ffnresadd` epilogue), so it stays net-zero
  nodes against the M2b control;
- harness: update the control-kernel expectation to `_epi_ffnresadd` and the topology
  contract to the fp16-store + in-kernel-resadd control.

This is the actual substrate-to-parity work for the +302.8 us core deficit, not another
load-pattern probe. The three slow shapes (Q6 V, Q4 down, Q6 down) all need llama's
DP4A-with-folded-quant substrate; Q4 FFN-down is the natural first because its Q8_1
provider/consumer already exists and only the producer fold + resadd epilogue are missing.

## Evidence

- `nv-gemv-core-deficit-correction-20260813.md` (authoritative +302.8 us join)
- `nv-q4-down-quad-re-census-20260813.md` + `nv-q4-down-quad-re-census-20260813.json`
- `nv-q4-down-producer-owned-q8-gate-20260813.md` + `nv-q4-down-owned-q8-construct-diagnosis-20260813.json`
- `nv-q4-down-owned-q8-bracket-20260813.json` and its three arm JSONs
