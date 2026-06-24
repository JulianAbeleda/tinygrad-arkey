# Decode Native Renderer DNR-3C7B PMC Ladder Result - 2026-06-20

## Verdict

`PASS_DNR3C7B_PMC_LADDER_CAPTURED_BLOCKED_NO_30US_COUNTER_CAUSE`

DNR-3C7B captures a same-harness PMC ladder for native decode renderer variants. The counter tool now works for
directional attribution, but the captured deltas do not name a proven 30us-class scheduling lever.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c7b_pmc_ladder.py --warmups 1 --timeout-s 360
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7b_pmc_ladder_result.json
```

## Counter Sets

The programmable counter sets that passed on this machine:

| pass | counters |
|---|---|
| issue/wait/cache | `SQ_BUSY_CYCLES`, `SQ_WAIT_ANY`, `SQ_INSTS_VALU`, `SQ_INSTS_SALU`, `GRBM_GUI_ACTIVE`, `GL2C_HIT`, `GL2C_MISS`, `SQ_INSTS_TEX_LOAD` |
| LDS/memory | `SQ_BUSY_CYCLES`, `SQC_LDS_IDX_ACTIVE`, `SQC_LDS_BANK_CONFLICT`, `SQ_INSTS_LDS`, `GRBM_GUI_ACTIVE`, `GL2C_HIT`, `GL2C_MISS`, `SQ_INSTS_SMEM` |

Rejected during bring-up:

| counter | reason |
|---|---|
| `SQ_INSTS_VMEM_RD` | not exposed by the runtime-supported PMC list |
| `TA_BUFFER_LOAD_WAVEFRONTS` | exposed in the counter list, but tinygrad's `pmc_start` has no `TA` group mapping |

## Issue/Wait/Cache Pass

Values are normalized by `GRBM_GUI_ACTIVE`; deltas are versus native.

| variant | correct | L2 hit % | SQ busy | delta | SQ wait | delta |
|---|---:|---:|---:|---:|---:|---:|
| native DNR-2 | yes | `18.461` | `9.797` | `0.000` | `424.325` | `0.000` |
| load-b128 DNR-3C2 | yes | `18.593` | `9.496` | `-0.300` | `318.867` | `-105.458` |
| best static, no markers | yes | `18.605` | `9.680` | `-0.117` | `324.473` | `-99.852` |
| DNR-3C4 marked | yes | `18.614` | `9.395` | `-0.402` | `307.212` | `-117.113` |

## LDS/Memory Pass

Values are normalized by `GRBM_GUI_ACTIVE`; deltas are versus native.

| variant | correct | L2 hit % | SQ busy | delta | LDS inst | LDS active | bank conflict |
|---|---:|---:|---:|---:|---:|---:|---:|
| native DNR-2 | yes | `18.470` | `9.819` | `0.000` | `0.891` | `1.336` | `0.000` |
| load-b128 DNR-3C2 | yes | `18.589` | `9.552` | `-0.267` | `0.906` | `1.358` | `0.000` |
| best static, no markers | yes | `18.606` | `8.963` | `-0.856` | `0.591` | `1.267` | `0.000` |
| DNR-3C4 marked | yes | `18.614` | `9.566` | `-0.253` | `0.628` | `1.345` | `0.000` |

## Interpretation

The counter ladder confirms the static rewrites are not wrong:

- all four variants remain numerically correct;
- b128 and ds-load reductions move SQ wait/busy in the expected direction;
- the best static path cuts normalized LDS instruction count from `0.891` to `0.591`;
- bank-conflict counters remain zero for all rows;
- L2 hit-rate movement is tiny, around `+0.14` percentage points versus native.

The same ladder also explains why this is still blocked:

- DNR-3C6 already showed the best static movement is only `8.346us`, leaving about `69.637us` to oracle;
- DNR-3C7B finds directional counter wins, but no counter family identifies a 30us-class native scheduling lever;
- PMC runs perturb timing enough that these counters should be used for direction, not final latency claims;
- static shape similarity remains refuted as the search objective.

## Next

DNR-3C7C is only justified if it builds an issue/interleaving objective from these counter families, not another
local count rewrite. The minimum useful next probe is a schedule-order experiment that changes overlap between
load/unpack/dot/scale/reduction while keeping correctness and the same PMC ladder.

If DNR-3C7C cannot move the same counter families materially, native decode should pause and the practical route
should remain the q8 artifact/oracle path until SQTT body timelines or oracle resource metadata are available.

No renderer defaults changed.
