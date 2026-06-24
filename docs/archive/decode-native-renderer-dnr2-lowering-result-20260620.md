# Decode Native Renderer DNR-2 Lowering Result - 2026-06-20

## Verdict

`PASS_DNR2_NATIVE_LOWERING_CORRECT_BLOCKED_DNR3_SCHEDULER_RESOURCE`

DNR-2 is complete enough to stop treating decode as an address/data-format mystery. The tinygrad-owned AMD DSL
candidate lowers the q8 gate/up data path and is numerically correct at the full authority shape. It is blocked on
performance, which belongs to DNR-3 scheduler/resource modeling.

Run:

```bash
PYTHONPATH=. python3 extra/q8_ffn_asm_gateup_full.py --warmups 1 --iters 3 \
  --out bench/qk-decode-primitive-transfer/decode_native_renderer_dnr2_gateup_full_fresh.json
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr2_lowering_closeout.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr2_lowering_result.json
```

## Fresh Run

The fresh short run used the existing native tinygrad AMD DSL candidate:

```text
q8_b2b_fullrow_reduce
```

Result:

| gate | result |
|---|---:|
| gate correctness | pass, max abs `9.536743e-07` |
| up correctness | pass, max abs `1.430511e-06` |
| full rows | `12288` |
| q8 bytes | `4608` |
| no external artifact | pass |
| consumer timing gate | fail |

The short-run median was about `193us`. The historical same candidate median is `166.649us`; the hipcc/LLD oracle
consumer is `93.54us`. The exact short-run timing is not used as promotion authority; it only reconfirms that this
native candidate is correct but slow.

## Lowered Native Contract

| piece | native lowering |
|---|---|
| work decomposition | `gidx0=row`, `gidx1=gate/up`, `lidx0=tid`; 128 threads per row |
| Q4_K address | `row * 2304 + kb * 144` |
| Q4_K scale/min | load `d`, `dmin`, scale bytes, select lt4/ge4 forms |
| Q4 nibble data | `qs + 16 + (sub/2)*32`, choose low/high by `sub&1` |
| q8 address | `(kb * 8 + sub) * 36`; load half scale plus 32 signed bytes |
| dot/min correction | `v_dot4_i32_iu8` for q4*q8 and ones*q8 |
| output | wave reduce, 4-wave LDS reduce, one float store to gate/up |

## Blocker

DNR-2 is not blocked by:

- Q4_K block addressing;
- q8 block addressing;
- scale/min extraction;
- dot4 instruction selection;
- gate/up output selection;
- numeric correctness.

The project is blocked at DNR-3:

- `s_clause` / `s_delay_alu` semantic placement;
- coalesced/vector load shape without correctness drift;
- register live-range/resource scheduling;
- instruction ordering and wait policy;
- branch/resource policy matching the hipcc/LLD oracle.

## Next

Start DNR-3 only as broad scheduler/resource work. Do not reopen BEAM/search or one-off load/wait/reduction patches from
DNR-2; those were already below the native N2 gate.
