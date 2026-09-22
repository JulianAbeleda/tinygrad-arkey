# M4 residual_add S4 gate rerun record - section-6 gate PASS, variant promoted

Date: 2026-08-08 (rerun of the 2026-08-06 S4 gate; the original run FAILED at
render with the weakint SPECIAL crash, see `m4-resadd-s4-gate-run-record-20260806.md`)
Branch: tinygrad `nvidia-bringup-20260731`, HEAD `c855309fc` (census fix stack:
`a7944410e` middle-args dedupe, `123ea1b4c` liveness retention, `c855309fc`
transport elision), tree clean at gate time.
Status: **gate run record, PASS. The open arm (production residual fold ACTIVE)
passes all five section-6 items at d512 and d2048: wall delta positive vs the
same-session closed control and vs the M2-on baseline, census at the gate oracle,
pins 3/3 both arms, pg3 legacy sha unmoved. `decode-q4k-epilogue-resadd-route-policy.json`
is PROMOTED for NV sm_120 (`promoted_targets: [{"backend":"NV","architecture":"sm_120"}]`),
and the measured same-session delta is booked.** Authorities:
`m4-resadd-landing-scope-20260806.md` section 3 (gate items 1-5),
`nv-decode-parity-campaign-reconciled-ledger-20260805.md` (booking rules).

## 1. Protocol

Same-session, lock-held (`flock -w 600 /tmp/gpu-bench.lock`), Qwen3-8B-Q4_K_M
(`/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`), nmeas 20, reps 3, median tok/s,
temperature 0, chunk_size 32, fused prefill attention disabled (house convention).
Open mode = the production fold ACTIVE via the module override
(`mrp._DECODE_Q4K_EPILOGUE_RESADD_PROMOTED_TARGETS = frozenset({("NV","sm_120")})`);
closed mode = default records (fold dormant); record mode = checked-in policy JSON.
Per-arm fresh subprocesses, `timeout 1200`, 20G cgroup cap. GPU: NVIDIA GeForce
RTX 5090 (sm_120), free at gate time.

## 2. Results

| arm | d512 | d2048 | d4096 |
| --- | --- | --- | --- |
| closed | **PASS** 181.946 tok/s, sha `227ad3ce...` 3/3, first `271` 3/3, census 948 / epi 0 / legacy 72 / copy 1 / resadd 72 | **PASS** 170.982 tok/s, sha `aca13ac6...` 3/3, first `271` 3/3 | **pre-existing env error** (device `free` traceback; same on open; not fold-related) |
| open (production fold ACTIVE) | **PASS** 183.032 tok/s, sha `227ad3ce...` 3/3, first `271` 3/3, census **912** / epi **36** / legacy **36** / copy **1** / resadd **36** | **PASS** 172.506 tok/s, sha `aca13ac6...` 3/3, first `271` 3/3 | **pre-existing env error** (same) |
| record (checked-in policy) | **PASS** 181.56 tok/s, sha `227ad3ce...` 3/3, first `271` 3/3, census 948 / epi 0 / legacy 72 / copy 1 / resadd 72 | not run (record == closed proven at d512) | not run (env cap) |

pg3 legacy render sha `27857cb8ca03` for `q4k_g3_lanemap_gemv_4096_4096` unmoved
in every arm that produced a result (closed d512/d2048, open d512/d2048).

## 3. Gate items

1. **Wall** (d512/d2048/d4096, open mode vs M2-on baseline 172.80/161.50/149.00):
   d512 183.032 > 172.80, d2048 172.506 > 161.50, d4096 env-capped on both arms
   (pre-existing device free error, reproduced on closed - not fold-related).
   Positive delta vs the same-session closed control at every depth that ran:
   d512 **+1.086 tok/s** (+0.60%), d2048 **+1.524 tok/s** (+0.89%).
2. **Census** (d512, open mode): kernels/token 912 (gate oracle), epi_resadd 36,
   legacy attn_qo gemv 36, `E_32_32_4_86a2` copy class **1** (control copy only;
   the 72 residual-slot transports are elided), resadd class 36. The closed arm
   stays at its 948-kernel baseline.
3. **Pins 3/3** at d512 and d2048, both modes: token sha `227ad3ce...` (d512) /
   `aca13ac6...` (d2048), first token `271` x3. Open == closed streams at every
   depth (bitwise-exactness preserved through the dedupe and transport elision).
4. **Unit tests**: `test_m4_resadd_landing.py`, `test_m4_residual_boundary_fold_probe.py`,
   `test_nv_boundary_free_ordinary_uop_gate.py` and the M4/M5 unit set green on the
   same tree (see section 6).
5. **pg3 legacy sha**: `27857cb8ca03` unmoved (all arms above).

## 4. Booking

Same-session section-6 delta (open vs closed, d512): **+32.61 us/token**
(5496.19 - 5463.58 us/token at 181.946 / 183.032 tok/s). This is the same-session
measured delta, booked as a fresh ledger row per `m4-resadd-landing-scope-20260806.md`
section 4 (no probe-2 ceiling extrapolation, no composition with the rejected
Attention-O/FFN-down row). The scope's copy-free ceiling projections
(+100.1 us measured / +45.0 us book from the microgate) are superseded by the
measured section-6 delta for booking purposes. Kernel-time evidence is consistent:
the 72 residual transports (108 us copy mass) are elided and the fused epi kernels
add the small measured penalty, netting the +32.61 us/token.

## 5. Decision

- `decode-q4k-epilogue-resadd-route-policy.json`: **PROMOTED** for NV sm_120,
  `promoted_targets: [{"backend":"NV","architecture":"sm_120"}]`. The combined
  `decode-q4k-epilogue-fusion-route-policy.json` stays CLOSED; ffn_down prelude
  and fp16_cast stay rejected.
- Ledger: fresh recovery row booked at **+32.61 us/token** (d512 same-session
  delta), cited to this record.
- The record-open state equals the forced-open state (pins 3/3 in record mode,
  d512 181.56 tok/s, census 948 = closed).

## 6. Evidence

`/tmp/m4-resadd-s4-gate-rerun-20260807.json` (closed/open d512+d2048 full results,
pg3 hashes), `/tmp/m4_gate_record_d512_rerun.json` + `.err` (record arm),
`/tmp/m4_elide_tokens_open.log` / `m4_elide_tokens_closed.log` (token-exactness
after elision), `/tmp/m4_chain_diag2.log` (fold validation: 36 validates, 1
non-resadd rejection), unit runs in section 3.4.
