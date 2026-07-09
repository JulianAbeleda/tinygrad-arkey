# DBUF Safe DS Offset Folding Phase 5 Report

Date: 2026-07-08

Scope: compare materialized LDS address arithmetic against `PREFILL_DBUF_LDS_CONST_IMM=1` safe DS offset folding for
`2x2 both` and `4x2 both` on `512x5120x5120`, `DEV=AMD:ISA`.

## Command Bundle

Common passing bundle:

```bash
PYTHONPATH=. DEV=AMD:ISA AMD_ISA_REG_ACCUM=1 REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1 \
PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1 PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_INDEX_SPLIT=1 PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 PREFILL_DBUF_LDS_ADDR_USE_DEP=1 \
AMD_ISA_WAITCNT_TARGETED=1 AMD_ISA_WMMA_B128_FRAG=1
```

Runtime command shape:

```bash
$COMMON python3 - <<'PY'
import os, json, statistics, time
from tinygrad.helpers import getenv
from tinygrad.codegen import to_program_cache
from extra.qk.prefill_v2_schedule_search import _run_config
cases=[('2x2-materialized',2,2,False),('2x2-safe-fold',2,2,True),('4x2-materialized',4,2,False),('4x2-safe-fold',4,2,True)]
for label,u0,u1,fold in cases:
  vals=[]
  for rep in range(3):
    if fold: os.environ['PREFILL_DBUF_LDS_CONST_IMM']='1'
    else: os.environ.pop('PREFILL_DBUF_LDS_CONST_IMM',None)
    getenv.cache_clear(); to_program_cache.clear()
    t=time.time(); r=_run_config(512,5120,5120,u0,u1,2,2); r['elapsed_s']=round(time.time()-t,2); r['rep']=rep+1
    if r.get('status')=='ok': vals.append(float(r.get('tflops',0)))
    print('RUNTIME', json.dumps({'case':label, **r}, sort_keys=True), flush=True)
  print('SUMMARY', json.dumps({'case':label,'n_ok':len(vals),'tflops':vals,'min':min(vals) if vals else None,'median':statistics.median(vals) if vals else None,'max':max(vals) if vals else None}, sort_keys=True), flush=True)
PY
```

Instruction-count command used `_compile_native_program(512,5120,5120,u0,u1,2,2)`, `_final_stream(...)`, and counted
final instruction mnemonics plus DS `offset0` fields under the same common bundle.

## Runtime Results

| Case | Status | TFLOPS samples | Median TFLOPS | LDS bytes |
|---|---|---:|---:|---:|
| `2x2 both`, materialized | ok | 9.33, 9.20, 9.31 | 9.31 | 32768 |
| `2x2 both`, safe fold | ok | 7.99, 7.98, 8.00 | 7.99 | 32768 |
| `4x2 both`, materialized | ok | 6.63, 6.60, 6.61 | 6.61 | 49152 |
| `4x2 both`, safe fold | ok | 7.00, 6.97, 6.98 | 6.98 | 49152 |

The `2x2` safe-fold slowdown reproduced across all three repeated samples. It is real for this flag bundle, not timing
noise.

## Instruction Counts

| Case | Total inst | `v_add` | `v_lshl` | `v_mul` | `ds_store_b128` | `ds_load_b128` | DS ops with nonzero offset | DS nonzero offsets | `s_waitcnt` |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `2x2` materialized | 737 | 177 | 130 | 62 | 32 | 64 | 32 / 96 | 16 | 53 |
| `2x2` safe fold | 625 | 128 | 130 | 20 | 32 | 64 | 45 / 96 | 16..240 step 16 | 53 |
| `4x2` materialized | 1485 | 382 | 258 | 158 | 48 | 128 | 64 / 176 | 16 | 85 |
| `4x2` safe fold | 1133 | 260 | 226 | 26 | 48 | 128 | 45 / 176 | 16..240 step 16 | 85 |

Safe folding lowers raw VALU address arithmetic and total instruction count for both shapes. The `4x2` runtime follows
that direction. The `2x2` runtime moves opposite to instruction count.

## Conclusion

The `2x2 both` safe-fold slowdown is real under the full passing bundle, but it is not a raw instruction-count regression.
It is most consistent with a final scheduling or DS-address-form side effect: same LDS bytes, same waitcnt count, fewer
VALU instructions, more/larger DS immediate offsets, and lower TFLOPS.

Blocker: safe DS offset folding cannot be promoted unconditionally as a performance win. The next gate should either add
a shape/selection guard for `2x2 both`, or inspect final DS scheduling/bank behavior for the immediate-offset stream before
promotion.

