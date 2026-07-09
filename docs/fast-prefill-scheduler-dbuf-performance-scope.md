# Fast Prefill Scheduler/DBUF Performance Scope

Date: 2026-07-08.

## Objective

Move from the route-bound 4x4 A+B LDS DBUF substrate toward production-shape native ISA performance.

Correction from the latest audit: the route-bound correctness gate is not native ISA evidence. Its runner forces
`DEV=AMD`, so the GPU correctness pass is HIP/C renderer evidence:

```text
extra/qk/prefill_graph_gemm_route_bound_stage_gate.py:
  env = {**os.environ, "DEV": "AMD", "DEBUG": "4", "PYTHONPATH": "."}
```

The structural probes still show the intended machine-code substrate, but native ISA runtime correctness must be proven
separately before DBUF performance conclusions are valid.

The desired end state remains:

```text
generated native ISA
  -> A and B staged through wide LDS
  -> DBUF slot/cadence visible
  -> no spills
  -> verifier clean
  -> GPU correct
```

This scope starts at the next real blocker: production table shapes such as `4096x4096`.

## Current Result

| Gate | Result | Meaning |
|---|---|---|
| Route-bound DBUF correctness | PASS in `DEV=AMD` HIP/C route: `rel_rmse_vs_ref=0.00020765016961377114`, `max_abs_vs_ref=0.03130340576171875`. | This no longer proves native ISA correctness because the gate hard-codes `DEV=AMD`. |
| Route-bound DBUF structural | PASS for broad shape: `REGALLOC_SPILLS: count=0`, WMMA operands from `ds_load_b128`, no scalar LDS stores. Strict D7 is downgraded because `src1` currently shows one LDS load address family. | The generated machine-code substrate is structurally close, but runtime correctness is still open. |
| Native ISA base schedule-search matmul, no DBUF/no LDS staging | PASS after the loop-backedge live-in fix: `512x512,u0=2,u1=2,loc=0,unr=8` returns `status=ok`; `4096x4096` bounded shape returns `status=ok`. | The prior MMU fault was register-allocation loop-live handling, not DBUF address math. |
| Native ISA final-stream address proof | PASS for the faulting base and bounded DBUF candidates: no global or LDS interval OOB found. Strengthened proof also reports 82 `GLOBAL` addr-high warnings on schedule-search streams. | The fault is not explained by a single linear pass over low-32-bit address bounds. `GLOBAL` addr-high was probed and refuted as the direct cause. |
| Native ISA dynamic backedge simulation | ROOT CAUSE FOUND/FIXED: the physical stream corrupted loop-carried address bases across the K-loop backedge. At `512x512`, iteration 1 previously reached `global_load_b128` with address `1047008` for a `524288` byte buffer. | Regalloc now keeps the `RANGE` live to the real `END` even when broad `END` source liveness is suppressed, and emits explicit address remats before the backedge when needed. |
| Production `4096x4096`, native ISA, DBUF A+B bounded shape | PASS for `u0=2,u1=2,loc=0,unr=8` with `REGALLOC_ADDR_REMAT=1`: `binary_group_segment_bytes=65536`, `status=ok`, measured `2.92 TFLOPS`. | Runtime correctness is unblocked; this exact-limit shape is not a final performance target. |
| Production `4096x4096`, native ISA, DBUF A-only/B-only bounded shape | PASS for `u0=2,u1=2,loc=0,unr=8` with `REGALLOC_ADDR_REMAT=1`: each compiles to `binary_group_segment_bytes=32768` and returns `status=ok` (`A-only 9.04 TFLOPS`, `B-only 5.95 TFLOPS`). | The previous one-operand MMU faults were the same backedge live-in bug. |
| Production `4096x4096`, native ISA, DBUF A+B table shape | Still over LDS: table-selected `u0=4,u1=4,loc=4,unr=8` compiles to `local_bytes=131072`, `binary_group_segment_bytes=131072`. | Remaining production blocker is true LDS tile footprint/schedule choice, not runtime address corruption. |
| Production `4096x4096`, native ISA, non-DBUF A-only/B-only | Spills 64 `GLOBAL_LOAD_B128` values. | Without DBUF/remat/lifetime changes, production native ISA still has broad global-load residency. |
| Production `4096x4096`, native ISA, non-DBUF A+B | Spills 192 values. | Production shape still needs streaming/lifetime control even before DBUF performance. |

## What Changed In This Pass

`REGALLOC_ADDR_REMAT=1` now rematerializes pure address roots in addition to address arithmetic:

```text
SPECIAL(gidx/lidx)
WG_ID / WI_ID / MOV_S2V
V_AND / V_IMUL / V_IADD / V_OFFSET / V_LSHR
```

This removed the production DBUF root spills:

```text
before: SPILL SPECIAL gidx0/gidx1/lidx1 and WI_ID
after:  REGALLOC_SPILLS: count=0 stack_size=0
```

The remaining failure is:

```text
RuntimeError: Too many resources requested: group_segment_size
```

The strengthened worker now reports exception detail and a compile-only resource summary on failure. Current production
A+B evidence with accumulator reclaim enabled:

```json
{
  "local_bytes": 131072,
  "reg_bytes_per_thread": 512,
  "reclaimable_reg_bytes_per_thread": 512,
  "effective_reg_bytes_per_thread": 0,
  "n_threads": 128,
  "group_segment_unreclaimed_bytes": 196608,
  "group_segment_estimated_bytes": 131072,
  "binary_group_segment_bytes": 131072,
  "over_limit": true
}
```

One-operand production evidence after the backedge fix:

| Stage | `binary_group_segment_bytes` | Result |
|---|---:|---|
| A-only DBUF bounded shape | 32768 | PASS: `status=ok`, `9.04 TFLOPS`. |
| B-only DBUF bounded shape | 32768 | PASS: `status=ok`, `5.95 TFLOPS`. |
| A+B DBUF bounded shape | 65536 | PASS: `status=ok`, `2.92 TFLOPS`; exactly at the limit and not a final target. |

So the runtime/address blocker is fixed for bounded native ISA DBUF. The correct production acceptance target is still not
`<= 65536`; it is a below-limit LDS footprint with correctness and competitive performance.

Additional schedule/resource probe:

```text
DEV=AMD:ISA AMD_ISA_REG_ACCUM=1 REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 PREFILL_DBUF=1 PREFILL_DBUF_LDS_CONST_IMM=1 \
PREFILL_DBUF_LDS_INDEX_SPLIT=1 PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 PREFILL_DBUF_LDS_ADDR_USE_DEP=1 \
AMD_ISA_WMMA_B128_FRAG=1 PYTHONPATH=. \
python3 extra/qk/prefill_v2_schedule_table_gate.py --compact --no-artifact \
  --resource-search --resource-run-below-limit --resource-run-limit 1 \
  --shapes 4096x4096 --resource-stages B --resource-u 2 --resource-loc 2 --resource-unr 8
```

Result: `B,u0=2,u1=2,loc=2,unr=8` compiles to `32768` LDS bytes and passes (`status=ok`, measured `10.04 TFLOPS`
in that run). This confirms the below-limit one-operand route is mechanically valid, but not a production performance win
relative to the no-staging bounded native ISA run measured at `16.98 TFLOPS`.

## Diagnosis

The current primary blocker is production LDS footprint/schedule shape, not native ISA generated matmul runtime correctness.

The table-selected production schedule (`M=512`, `4096x4096`, `u0=4 u1=4 loc=4 unr=8`) creates large LDS buffers. DBUF
then doubles the slotted local storage. The native ISA route can now allocate registers, and accumulator LDS accounting can
be reclaimed, but the true local tile allocation is still too large:

```text
A+B production DBUF after accumulator reclaim:
  true local LDS = 131072 bytes
  limit          = 65536 bytes

A+B bounded DBUF after accumulator reclaim:
  true local LDS = 65536 bytes
  result         = correctness pass, but exact-limit/slow
```

This means scheduler/waitcnt tuning would be premature for the full A+B DBUF production path: there is no below-limit
production A+B kernel to time under the promoted DBUF path. One-operand staging is valid and below-limit, but needs a
separate performance reason before promotion.

The strengthened isolation also shows:

```text
Native ISA, u0=2 u1=2 loc=0 unr=8:
  no staging / DBUF off       -> correctness pass
  A-only DBUF, 32768 B LDS    -> correctness pass
  B-only DBUF, 32768 B LDS    -> correctness pass
  A+B DBUF, 65536 B LDS       -> correctness pass, exact-limit/slow

Native ISA, scheduler disabled:
  no staging / DBUF off       -> backedge fault class no longer reproduced after the regalloc fix

Final-stream proof:
  low address intervals       -> in bounds
  GLOBAL addr high half       -> 82 warnings, but standalone high-half poison probe passes
  dynamic backedge simulation -> OOB on iteration 1 from corrupted physical address base
```

So the fault is upstream of DBUF slot/base math and upstream of scheduler ordering. The likely next class is native ISA
register allocation/loop-live semantics in the base generated matmul path. The physical final stream reuses registers
holding loop-carried address bases (`v74/v76` in the captured `512x512` stream) as in-loop shifted byte-address
temporaries before branching back to the loop top. The old proof missed this because it walked the stream linearly once
and did not execute the hardware backedge.

## Required Primitive

The next primitive is bounded LDS staging for production shapes:

```text
stage only the 16x16 WMMA fragment footprint needed by the current subtile/window,
not the full table-local tile footprint implied by loc=4 and DBUF slot duplication.
```

Equivalent acceptable fixes:

| Option | Description | Acceptance |
|---|---|---|
| S1. Fragment-scoped LDS allocation | Allocate LDS for the exact A/B WMMA fragment rows consumed by the current 4x4 subtile group. | Production `4096x4096` DBUF A/B launches below 65536 bytes and passes correctness. |
| S2. Tile-shape gate | Select a smaller native-ISA schedule for DBUF when table-local shape would exceed LDS budget. | No spill, below-limit group segment, correctness pass. |
| S3. One-operand staged production route | Keep only the profitable/reuse-heavy operand in LDS if both operands exceed LDS. | Must reduce below 65536 bytes, launch, pass correctness, and beat baseline. Exactly 65536 bytes is rejected by current evidence. |
| S4. LDS allocation estimator | Fail closed before compile when `(operand, slot_count, tile_shape)` exceeds LDS budget. | Prevents invalid production DBUF launch attempts and drives schedule choice. |

## Bounded Search Result

The centralized table gate now has a compile-only bounded resource search:

```bash
DEV=AMD:ISA AMD_ISA_REG_ACCUM=1 REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 PREFILL_DBUF_LDS_INDEX_SPLIT=1 \
PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 PREFILL_DBUF_DIRECT_B128_CHAIN=1 \
PREFILL_DBUF_LDS_ADDR_USE_DEP=1 AMD_ISA_WMMA_B128_FRAG=1 \
PYTHONPATH=. python3 -m extra.qk.prefill_v2_schedule_table_gate \
  --resource-search --resource-stages both,A,B --resource-u 2,4 \
  --resource-loc 0 --resource-unr 8 --compact --no-artifact --shapes 4096x4096
```

Result:

| Stage | `u0` | `u1` | `loc` | LDS bytes | Status |
|---|---:|---:|---:|---:|---|
| A | 2 | 2 | 0 | 32768 | Below limit. Runtime launch still faults with MMU fault. |
| A | 2 | 4 | 0 | 32768 | Below limit. Not launched in this pass. |
| B | 2 | 2 | 0 | 32768 | Below limit. Runtime launch still faults with MMU fault. |
| B | 4 | 2 | 0 | 32768 | Below limit. Not launched in this pass. |
| both | 2 | 2 | 0 | 65536 | Rejected: exactly full LDS is already known unsafe. |
| both | 2 | 4 | 0 | 98304 | Rejected: over limit. |
| both | 4 | 2 | 0 | 98304 | Rejected: over limit. |
| both | 4 | 4 | 0 | 131072 | Rejected: over limit. |

Conclusion: the backedge register-lifetime fault was the runtime blocker. The next blocker is selecting or generating a
production A+B DBUF shape whose LDS footprint is below the limit and whose performance is worth promoting.

Probe conclusion within the current schedule knobs:

| Candidate class | Best observed LDS floor | Correctness | Performance meaning |
|---|---:|---|---|
| A+B DBUF, full-tile staging | 65536 bytes (`u0=2,u1=2`) | PASS | Exact-limit and slow; not a safe production target. |
| A-only DBUF | 32768 bytes (`u0=2`) | PASS for `loc=0`; `loc=2` A-only was wrong in one probe. | Valid fallback class, but slower than no-staging bounded native ISA. |
| B-only DBUF | 32768 bytes (`u1=2`) | PASS for `loc=0` and `loc=2`. | Best safe DBUF probe so far, still not a win. |
| Table-selected A+B `u0=4,u1=4,loc=4` | 131072 bytes | Cannot launch under LDS limit. | Requires smaller LDS allocation strategy, not waitcnt tuning. |

## Primitive Probe: Fragment-Scoped Cooperative A+B

The next attempted primitive was to replace generic A staging + B tile-key staging with cooperative WMMA-fragment staging
for both operands, then progressively shrink the LDS identity:

| Probe | LDS bytes | Result | Meaning |
|---|---:|---|---|
| cooperative A+B, include GLOBAL tile ranges | 4194304 | Compile/resource over-limit | GLOBAL route ranges cannot be part of LDS identity. |
| cooperative A+B, drop GLOBAL | `loc=0` spills; `loc=2` B-only-ish path at 32768 | `loc=2` produced `WRONG rr=nan` | Dropping GLOBAL alone is not sufficient and can leave A unstaged on local schedules. |
| cooperative A+B, drop GLOBAL, include LOCAL | 98304 | Compile/resource over-limit | A local tile identity is still too large. |
| cooperative A+B, drop GLOBAL+LOCAL | 65536 | Correctness not promoted; still exact-limit | This only reaches the old unsafe floor. |
| cooperative A+B, drop GLOBAL+LOCAL+all UNROLL | 4096 | `WRONG rr=nan` | Unroll fragments are concurrently live; full unroll slot reuse aliases data. |
| cooperative A+B, drop only UNROLL size 2 | 32768 | `WRONG rr=nan` | Even the small unroll axis is semantically live. |
| cooperative A+B, drop only UNROLL size 8 | 8192 | `WRONG rr=nan` | The large unroll axis is also semantically live. |

Conclusion: the primitive is **not** a pure LDS allocation shrink over the existing unrolled body. Below-limit A+B requires
changing the producer/consumer schedule so fewer unrolled fragments are live at once, or a different staging policy. The
simple fragment-slot reuse probes are useful negative tests and must stay opt-in.

Immediate next scope:

| Step | Purpose | Acceptance |
|---|---|---|
| R1. Compile-only resource gate | Keep `--resource-search` as the fail-closed estimator. | Candidate table reports LDS bytes before any launch. |
| R2. Native ISA bounded runtime isolation | Run the same `u0/u1/loc/unr` candidate with DBUF off, one-operand DBUF, and A+B DBUF. | Done: all bounded candidates pass after the backedge live-in fix. |
| R3. Address-safety probe | Add final-stream/global-address summary for production candidates. | Done: faulting candidates prove no generated global/LDS interval OOB under the supported instruction subset. |
| R4. Minimal native ISA microprobe | `extra/qk/prefill/global_vaddr_high_probe.py`: `addr+1=0`, `0x10000`, and `0xffffffff`. | Done: all variants pass, so `GLOBAL` high-half poisoning is not the direct cause. |
| R5. Dynamic backedge proof | Simulate the physical final stream across the K-loop backedge. | Done: iteration 1 previously produced a `global_load_b128` OOB address. |
| R6. Primitive fix | Fix loop-carried address-base preservation/rematerialization in regalloc/lowering before resuming DBUF sizing/performance. | Done: no-staging, one-operand DBUF, and A+B bounded native ISA candidates launch and pass correctness. |
| R7. Integrated below-limit runtime probe | Extend `prefill_v2_schedule_table_gate.py` to benchmark resource-safe DBUF candidates. | Done: `--resource-run-below-limit` compiles, filters, and times below-limit candidates through `_run_config`. |

## Acceptance Gates

| Gate | Required result |
|---|---|
| P1. Native base correctness | No-staging `DEV=AMD:ISA` schedule-search matmul launches and passes correctness at route-bound size. |
| P2. Production compile | `DEV=AMD:ISA` `4096x4096` table shape compiles without spills. |
| P3. Production launch | Same production shape has `binary_group_segment_bytes < 65536` and no longer fails or faults at launch. |
| P4. Correctness | Production shape passes the schedule-table numeric gate. |
| P5. Performance | Same-clock TFLOPS beats the current generated table baseline, then test `5120x5120`. |
| P6. Scheduler tuning only after P1-P5 | Waitcnt/scheduler changes are considered only once the production DBUF kernel launches. |

## Commands

Route-bound DBUF correctness:

```bash
REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 PREFILL_DBUF_LDS_INDEX_SPLIT=1 \
PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 PREFILL_DBUF_DIRECT_B128_CHAIN=1 \
PREFILL_DBUF_LDS_ADDR_USE_DEP=1 AMD_ISA_WMMA_B128_FRAG=1 \
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_route_bound_stage_gate --run-amd --local-stage both --compact
```

Production native ISA DBUF probe:

```bash
DEV=AMD:ISA REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1 \
PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PREFILL_DBUF=1 PREFILL_DBUF_LDS_CONST_IMM=1 PREFILL_DBUF_LDS_INDEX_SPLIT=1 \
PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 PREFILL_DBUF_DIRECT_B128_CHAIN=1 \
PREFILL_DBUF_LDS_ADDR_USE_DEP=1 AMD_ISA_WMMA_B128_FRAG=1 \
PYTHONPATH=. python3 -m extra.qk.prefill_v2_schedule_table_gate --run-amd --compact --no-artifact --shapes 4096x4096
```

## Completion Definition

This scope is complete when production native ISA DBUF launches for at least `4096x4096`, passes correctness, and produces
a measured TFLOPS result. If it does not beat the existing generated table route, scheduler/waitcnt tuning can proceed only
after the LDS footprint primitive is in place.
