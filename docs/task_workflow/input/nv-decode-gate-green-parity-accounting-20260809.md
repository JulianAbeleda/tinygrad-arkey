# NV decode gate-green to parity accounting

Date: 2026-08-09
Status: measurement and accounting record. Records how the position-invariant
decode graph (wedge fix) and the section-6 shared-Q8 gate PASS translate into
the decode parity ledger. Authorizes no implementation, no route-record
change, and no promotion. Branch boundary: tinygrad `nvidia-bringup-20260731`
at `61d076ab6`. This record follows the booking rules of
`nv-decode-parity-campaign-reconciled-ledger-20260805.md` and the lifecycle
states of `nv-parity-and-beyond-forward-scope-20260803.md`.

## 1. What happened

The depth-4096 decode wedge (Xid 109, unique function key per token, schedule
CACHE MISS per token, ~26 s/token) was root-caused to the bound start_pos
(BIND) node leaking into decode graph structure. The fix makes every
structural slice (KV store slot, rope reads, KV read extents, mask diagonal)
index the UNBOUND start_pos variable: capture once, pure exec replay at every
position, llama.cpp style (positions are launch-time data, never graph
structure). Commit `585c3e4f4`.

That fix is the scaffolding the GEMV substrate landing scope
(`nv-gemv-substrate-landing-scope-20260808.md`) requires: its section-6 full
gate needs deep-context runs (d4096) to stay alive and replay
deterministically. This record ratifies the first complete section-6 gate
PASS on the fixed graph.

## 2. Gate result (2026-08-09, artifact `/tmp/nv_shared_q8_section6_gate_final.json`)

Protocol: lock-held, Qwen3-8B-Q4_K_M, nmeas 20, reps 3, median tok/s, closed /
open / record arms at d512/d2048/d4096, per the section-6 full gate spec.

| depth | closed tok/s | open tok/s | record tok/s | M2 baseline | pin sha | first token |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| d512 | 183.267 | 184.949 | 182.872 | 172.80 | `227ad3ce` | 271 |
| d2048 | 172.822 | 174.673 | 172.742 | 161.50 | `aca13ac6` | 271 |
| d4096 | 160.665 | 162.786 | 161.086 | 149.00 | `d9f1700a` | 374 |

Section-6 conditions, all met:

1. Wall: open >= M2 baseline at every depth (d512 +12.149, d2048 +13.173,
   d4096 +13.786 tok/s) and positive open-vs-closed delta at every depth
   (d512 +1.682, d2048 +1.851, d4096 +2.121 tok/s).
2. Census (d512, open): fused provider count 34 (17 blocks x two captures),
   cooperative Q4 consumers 86, legacy shared-Q4 0, ordinary providers 0,
   exact max17 record oracle.
3. Semantic contract: PASS (`semantic_pass: true`, `gate_pass: true`);
   argmax equal, ordered top-10 sets equal, relative L2 `8.552e-4` <= 1e-3,
   all finite.
4. Pins 3/3 at every depth in every mode; closed/open/record token streams
   identical.
5. Unit tripwire: `test_shared_q8_attention*` + M4/M5 + decode graph
   position-invariance sets green on the same tree.
6. pg3 legacy sha `27857cb8ca03` unmoved (all arms).

## 3. Parity translation

The parity authority's llama rows
(`nv-parity-and-beyond-forward-scope-20260803.md` section 2.2, same-session
family) and the current fixed-depth rows:

| depth | llama tok/s | llama ms/token | tinygrad open tok/s | tinygrad ms/token | gap ms/token | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d512 | 248.20 | 4.029 | 184.949 | 5.407 | 1.378 | 0.745x |
| d2048 | 235.14 | 4.253 | 174.673 | 5.725 | 1.472 | 0.743x |
| d4096 | 225.95 | 4.426 | 162.786 | 6.143 | 1.717 | 0.720x |

No depth is parity-qualified. The composed d512 steady-state authority
(`nv-decode-final-composed-same-session-record-20260805.md`) remains
separate: llama 4.005677 ms/token vs native composed 5.324244 ms/token,
0.75235x, gap 1.318567 ms. This record does not convert the fixed-depth rows
into the composed row and does not mix protocols.

## 4. Where the gate sits in the ledger

The reconciled ledger books disjoint recoveries against the fixed authority:
P1 66.662 + P2 75.031 + P5 91.637 + Q4-g12 24.676 + max17 12.462 + M4
residual_add 32.610 = 303.078 us/token strict booked.

The section-6 gate above is the qualifying evidence for the shared-Q8 GEMV
substrate landing scope. Per its booking rule, the fresh same-session
section-6 delta is bookable as a new ledger row incremental on the composed
P1/P2/P5/Q4-g12/max17 baseline, cited to this record and its gate artifact.
The isolated primitive wins (-56.28 us/replay Q4 cooperative, -0.81 to -0.94
us/replay Q6 flat four-warp) are NOT independently bookable and are not
booked here.

Remaining path to parity, with the authority's scoped marginal contributions
(estimates from `nv-decode-exhaustive-forward-scope-20260805.md`, not booked
recoveries):

| Bucket | Authority attribution | Marginal if recovered | Status |
| --- | ---: | ---: | --- |
| Shared-Q8 GEMV substrate (quant cores) | 302.788 us | section-6 delta pending booking | Gate PASSED, promotion record + loader admission + booking next |
| Norms population | 495 us | scoped | Capability-blocked by C1 (no generic cooperative reduction-to-output primitive) |
| Fusion / dataflow | 662.128 us | +26.67 tok/s | Attribution only; exact-output A/B per population, GPU parked |
| Hidden overlap | 445.954 us | +17.17 tok/s | Attribution only; co-schedule candidate gate, GPU parked |
| Host / outside window | 239.805 us | +8.86 tok/s | Attribution only; full-logit equality gate, GPU parked |

## 5. Rules that bind this record

- Parity ratio is reported per target point from same-session rows only; the
  fixed-depth rows above are the same protocol family as the M2 baseline and
  the parity authority rows, not a synthetic composition.
- No subtraction-from-fixed-authority absolute claims and no synthetic
  tok/s extrapolation. The composed 1.318567 ms gap is the current absolute
  d512 authority.
- The section-6 delta is booked once, incrementally, and never combined with
  the rejected Attention-O/FFN-down row or the isolated microgate numbers.
- Promotion of the shared-Q8 lease (`NV sm_120`) happens only through the
  landing scope's record (`decode-shared-q8-attention-route-policy.json`),
  loader-installed 17-block admission, and a closed-vs-open equality re-run
  after the record gains the target.

## 6. Next actions

1. Land the shared-Q8 lease: promotion record gains `NV sm_120`, loader
   admission, equality re-run, book the section-6 delta row.
2. Three bounded reopen arms from the landing scope: Q6 V direct-output
   lease, Q4 FFN-down singleton re-bracket, block-13 precision localization
   (CPU-only).
3. Norms, host, and overlap remain attribution with their own HARD STOPs; no
   GPU arm without a CPU-verified forecast or an exact-output correctness
   contract.
