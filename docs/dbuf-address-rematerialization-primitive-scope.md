# DBUF Address Rematerialization Primitive Scope

Date: 2026-07-07.

## Objective

Make `PREFILL_DBUF=1` 4x4 generated AMD ISA compile no-spill by preventing staging address calculations from becoming
long-lived VGPR values across the reduce loop.

This is the primitive fix after the LDS-local serialization attempts failed to move pressure.

## Current Diagnosis

The failure is:

```text
NotImplementedError: Inc 0: no spills
```

The live-range evidence is now specific:

```text
both DBUF peak: 137 live VGPRs
composition: 64 V_OFFSET + 64 V_IADD + small fixed remainder
```

`REGALLOC_DEBUG_DETAIL=1` shows many address producers are:

- defined before the reduce `RANGE`;
- live until the reduce-loop `END`;
- consumed as address bases for staging memory ops much later.

So the compiler is effectively producing:

```python
addr0 = materialize_address(...)
addr1 = materialize_address(...)
...
addr127 = materialize_address(...)

for k in reduce_loop:
  global_load_b128(addr0)
  ds_store_b128(...)
  ...
```

The desired machine-code shape is:

```python
for k in reduce_loop:
  addr0 = materialize_address(...)
  global_load_b128(addr0)
  ds_store_b128(...)

  addr1 = materialize_address(...)
  global_load_b128(addr1)
  ds_store_b128(...)
```

## Primitive Thesis

Address expressions used only by memory ops should be rematerializable at the memory op, not normal values with long
virtual-register live ranges.

The primitive is:

```text
symbolic address carrier -> materialize V_OFFSET/V_IADD immediately at GLOBAL_LOAD/STORE or DS_LOAD/STORE selection
```

In this route, the primary target is packed global staging loads feeding LDS stores:

```text
INDEX(global_buf, symbolic_idx) -> GLOBAL_LOAD_B128 -> DS_STORE_B128
```

not the later LDS `ds_load_b128` into WMMA fragments.

## Success Definition

100% for this primitive means:

| Gate | Required Result |
|---|---|
| A-only DBUF | native probe compiles no-spill. |
| B-only DBUF | native probe compiles no-spill. |
| Both DBUF | native probe compiles no-spill. |
| Structural | `ds_store_b128=16`, `ds_load_b128=16`, zero scalar LDS stores. |
| Origins | all WMMA A/B operands originate from `ds_load_b128`. |
| Verifier | `SPEC=1` clean. |
| Correctness | central DBUF GPU correctness still passes. |
| Regression | non-DBUF both remains unchanged. |

The pressure target is not an exact number, but both-side peak must leave the no-spill band. The old bad signature:

```text
64 V_OFFSET + 64 V_IADD live through reduce END
```

must disappear.

## Work Packages

### R0. Keep Failed Experiments Guarded

Owner files:

- `tinygrad/codegen/opt/postrange.py`
- `tinygrad/renderer/isa/amd.py`

Keep these default-off until deleted or promoted:

- `PREFILL_DBUF_LDS_ADDR_SERIAL`
- `PREFILL_DBUF_LDS_LOAD_SERIAL`
- `PREFILL_DBUF_LDS_CONST_IMM`
- `PREFILL_DBUF_GLOBAL_ADDR_INLOOP`

Acceptance:

- default unit suite still passes;
- no default-path behavior change;
- failed flags are not part of the acceptance path unless a later change makes one useful.

### R1. Add Address-At-Use Carrier

Owner file:

- `tinygrad/renderer/isa/amd.py`

Introduce a fail-closed carrier for DBUF staging addresses. This should not be a global semantic change to every
`INDEX`. Scope it to packed staging paths first.

Candidate representation:

```python
UOp(Ops.NOOP, dtype, src=(base_ptr, symbolic_idx, order_or_dep), arg=("addr_at_use", itemsize, const_off))
```

or:

```python
UOp(Ops.NOOP, dtype, src=(base_ptr, symbolic_idx), arg=("global_b128_addr_expr", itemsize, const_off))
```

Lowering rule:

```python
def materialize_addr_at_use(carrier):
  vidx = _movs2v(ctx, symbolic_idx) if _is_sgpr(symbolic_idx) else symbolic_idx
  off = V_OFFSET(vidx, shift)
  if const_off:
    off = V_IADD(off, const_off)
  return off, base_ptr
```

The key is placement: this materialization must happen inside the `GLOBAL_LOAD_B128` construction path, not as an
ordinary `isel_index` result that can be hoisted and kept live.

Acceptance:

- `REGALLOC_DEBUG_DETAIL=1` no longer shows 64 staging `V_OFFSET` and 64 staging `V_IADD` live until reduce `END`;
- no malformed pointer-vector `PTRCAT`;
- no scalar LDS fallback.

### R2. Apply Carrier to Packed Global-B128 Staging

Owner locations:

- `_pack_withlocal_lds_stores`
- `_pack_b_tilekey_lds_stores`
- `_lds_b128_store_data`
- `_global_half8_base`

Current risky shape:

```python
carrier = NOOP(..., arg=("global_b128", gidx0))
idxc = isel_index(ctx, gidx0)
GLOBAL_LOAD_B128(off, ptr, imm)
DS_STORE_B128(lds_addr, loaded_pack)
```

Target shape:

```python
carrier = NOOP(..., arg=("global_b128_addr_expr", gidx0, dependency_key))
...
idx_expr = carrier.arg[1]
off, ptr = materialize_addr_at_use(idx_expr)
GLOBAL_LOAD_B128(off, ptr, imm)
DS_STORE_B128(lds_addr, loaded_pack)
```

The implementation must preserve:

- fixed `LDS_PACK_BASE..LDS_PACK_TOP` pack destination;
- existing B tile-key bridge shape `16 x GROUP(N stores)`;
- existing non-DBUF withlocal b128 path.

Acceptance:

- A-only DBUF peak drops from about `88`;
- B-only DBUF peak drops from about `89`;
- both-side peak drops materially from `137`;
- packed `global_load_b128` and `ds_store_b128` counts remain visible.

### R3. Rematerialize LDS Address Only If Still Needed

Owner locations:

- `_frag_b128_loads`
- `isel_load`
- `isel_store`

Do not start here. The previous tests show LDS load serialization and DS immediate folding did not move the bad peak.

Only revisit if R2 removes the loop-carried global staging address values but the new peak is still dominated by LDS
`V_OFFSET`/`V_IADD`.

Acceptance:

- any LDS rematerialization must preserve `ds_load_b128=16`;
- WMMA origins must stay `ds_load_b128`;
- no D2 slot aliasing.

### R4. Probe the Live-Range Class

Owner files:

- `tinygrad/codegen/late/regalloc.py`
- `extra/qk/prefill/native_isa_l4_stream_probe.py`

Keep `REGALLOC_DEBUG_DETAIL=1` available, but add a compact probe summary so we do not rely on huge stderr dumps.

Desired report:

```json
"regalloc_peak": {
  "peak": 137,
  "classes": {"V_OFFSET": 64, "V_IADD": 64},
  "live_to_reduce_end": {"V_OFFSET": 64, "V_IADD": 64}
}
```

Acceptance:

- probe can distinguish "address values live to reduce END" from "address values materialized near memory op";
- no final-stream-only dependency, because current failure happens before final stream.

### R5. If R2 Fails, Use Explicit Address Scratch

Owner file:

- `tinygrad/renderer/isa/amd.py`

This is fallback, not the primitive. Use only after R2 proves the symbolic carrier cannot constrain lifetime.

Shape:

```python
scratch = reserved_vgpr_outside_vpool
scratch = V_OFFSET(...)
scratch = V_IADD(...)
GLOBAL_LOAD_B128(scratch, ...)
```

Requirements:

- scratch register excluded from `_vpool`;
- scratch not in WMMA fragment ranges;
- explicit dependency prevents reuse before the memory op consumes it;
- scoped to DBUF staging global b128 only.

Risk:

- because this hides a temporary from normal regalloc, it can create correctness hazards if dependencies are incomplete.

## Anti-Goals

- Do not add spills.
- Do not tune waitcnt or scheduler before no-spill compile.
- Do not switch to handwritten assembly.
- Do not broadly change `isel_index` for all kernels.
- Do not rely on `PREFILL_DBUF_LDS_ADDR_SERIAL`, `PREFILL_DBUF_LDS_LOAD_SERIAL`, or DS immediate folding as the primary
  fix unless new evidence shows they affect the loop-carried address class.
- Do not accept lower pressure if packed LDS b128 structure regresses.

## Gate Matrix

Run after each implementation attempt.

Non-DBUF regression:

```bash
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

A-only DBUF:

```bash
REGALLOC_DEBUG=1 \
PREFILL_DBUF=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=a \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

B-only DBUF:

```bash
REGALLOC_DEBUG=1 \
PREFILL_DBUF=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=b \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

Both DBUF:

```bash
REGALLOC_DEBUG=1 \
PREFILL_DBUF=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

Detailed failing-class check:

```bash
REGALLOC_DEBUG=1 \
REGALLOC_DEBUG_DETAIL=1 \
PREFILL_DBUF=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

Verifier:

```bash
SPEC=1 \
PREFILL_DBUF=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

Central correctness:

```bash
PREFILL_DBUF=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_route_bound_stage_gate --run-amd --local-stage both --compact
```

Unit:

```bash
PYTHONPATH=. pytest -q test/unit/test_amd_isa_wmma.py
```

## Stop Conditions

| Stop | Meaning |
|---|---|
| Primitive solved | A/B/both DBUF compile no-spill and the loop-carried `64 V_OFFSET + 64 V_IADD` class disappears. |
| Symbolic carrier blocked | Carrier cannot be represented without verifier-invalid pointer vectors or broad `isel_index` changes. Move to R5 scratch. |
| Pressure moves but still spills | Add compact peak probe, identify new class, then decide whether LDS rematerialization is needed. |
| Structure regresses | Restore packed LDS b128 before continuing; no scalar LDS fallback is acceptable. |
| Correctness fails after compile | Debug address expression equivalence and DBUF slot identity before scheduler/waitcnt. |

## 2026-07-07 Sequential Push Result

The original DBUF pressure class is reduced but not solved.

| Experiment | Result | Decision |
|---|---|---|
| `PREFILL_DBUF_LDS_ADDR_USE_DEP=1` | Both-side peak drops from `137` to `82`; A-only `64`, B-only `58`. | Keep as the first useful primitive. |
| `PREFILL_DBUF_DIRECT_B128_CHAIN=1` | Direct `global_load_b128 -> ds_store_b128` stream becomes paired instead of 16 loads then 16 stores. | Keep as a useful primitive; fixes the global staging burst. |
| `REGALLOC_END_NO_SOURCE_LIVE=1` with both primitives | Both-side peak drops to `53`, but still spills `5` address bases. | Useful diagnostic; needs a principled DBUF-safe form before promotion. |
| `PREFILL_DBUF_LDS_STORE_IMM_FOLD=1` | Changes which address bases spill but still spills `5`. | Not sufficient. |
| `PREFILL_DBUF_DIRECT_B128_ADDR_REMAT=1` | Lowers peak further but increases spills substantially. | Do not use. |
| `PREFILL_LDS_PACK_ALLOW_POOL=1` | Worsens spills. | Reject; fixed b128 scratch must stay excluded from the pool. |

Current best failing signature:

```text
REGALLOC_DEBUG: 2682 uops, PEAK 53 live vregs
REGALLOC_SPILLS: count=5 stack_size=20
  SPILL ... AMDOps.V_IADD range=[early, later LDS/global staging uses]
```

Interpretation: the broad `64 V_OFFSET + 64 V_IADD` DBUF failure is gone. The remaining blocker is a tiny set of
long-lived address-base terms around the resident-fragment / LDS-load boundary. More generic rematerialization is too
blunt; the next primitive should target those base expressions specifically, or make the DBUF route free one fragment
scratch window without violating the fixed-register WMMA contract.

### Five Remaining Spills

Captured with:

```bash
REGALLOC_DEBUG=1 \
REGALLOC_DEBUG_DETAIL=1 \
REGALLOC_DEBUG_SPILLS=1 \
REGALLOC_END_NO_SOURCE_LIVE=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 \
PREFILL_DBUF_LDS_ADDR_USE_DEP=1 \
PREFILL_DBUF=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

| Spill | Definition | Live range | Use clusters | Likely role | Theory | Next test |
|---|---:|---:|---|---|---|---|
| `v72` | uop `87`: `V_IADD(V_IMUL, 48)` | `[87, 88, 663, 734, 805, 876]` | early store address at `88`, later address clusters around `663/734/805/876` | A/B LDS slot row base with constant `48` | One base is shared across four DBUF-phase/LDS-fragment uses; allocator keeps it alive instead of recomputing. | Clone/rematerialize this `V_IADD` at each later cluster, not at generic store address level. |
| `v104` | uop `95`: `V_IADD(V_IMUL, 80)` | `[95, 96, 947, 1018, 1089, 1160]` | early store address at `96`, later clusters around `947/1018/1089/1160` | A/B LDS slot row base with constant `80` | Same pattern as `v72`, but for the paired/other slot bank. | Same targeted clone; check if constants map to slot bank stride. |
| `v88` | uop `91`: `V_IADD(V_IMUL, 64)` | `[91, 92, 977, 1048, 1119, 1190]` | early store address at `92`, later clusters around `977/1048/1119/1190` | middle LDS row/slot base | Persistent base crosses multiple WMMA operand-load groups; generic LDS load serialization does not touch the base. | Apply remat at `_frag_b128_loads` index carrier, before `isel_index`, for DBUF LDS carriers only. |
| `v136` | uop `103`: `V_IADD(V_IMUL, 112)` | `[103, 104, 1231, 1302, 1373, 1444]` | early store address at `104`, later clusters around `1231/1302/1373/1444` | high LDS row/slot base | Same live-range shape; this likely corresponds to the final row group. | Same targeted clone; validate no extra offset explosion. |
| `v120` | uop `99`: `V_IADD(V_IMUL, 96)` | `[99, 100, 1261, 1332, 1403, 1474]` | early store address at `100`, later clusters around `1261/1332/1403/1474` | high/mirrored LDS row/slot base | Paired with `v136`; ordering suggests two-slot DBUF bank/row symmetry. | Same targeted clone; if all five disappear, promote as `PREFILL_DBUF_LDS_BASE_REMAT`. |

Common structure:

```text
base = lane_or_row_term * stride + const
addr = V_OFFSET(base, scale=1)
use addr for DS_STORE_B128 early
reuse base again for four later LDS/fragment address clusters
```

So the likely primitive is not more broad scheduling. It is a proof-based DBUF LDS-base rematerializer:

```text
if address_base is V_IADD(shared_term, CONST)
and all later uses are address-only V_OFFSET/V_IADD consumers
and the expression is value-neutral/pure:
  clone address_base at each later cluster
  keep the original only for the early store cluster
```

### Per-Spill Micro-Scopes

Each micro-scope must answer four questions:

1. Is the spilled base expression pure and safe to clone?
2. Which later use clusters consume it?
3. Is the later use address-only, or does it feed data/control?
4. What is the smallest rewrite that kills the long live range without increasing total spills?

| Scope | Spill | Success condition | Reject condition | Candidate implementation |
|---|---|---|---|---|
| `S72` | `v72 = V_IADD(V_IMUL, 48)` | Original range `[87..876]` becomes one early store range plus four local remat ranges; spill disappears. | Any cloned value feeds non-address op, or new spill count increases. | Clone `V_IADD(V_IMUL, 48)` at clusters `663/734/805/876` before their `V_OFFSET` consumers. |
| `S104` | `v104 = V_IADD(V_IMUL, 80)` | Range `[95..1160]` is split across the `947/1018/1089/1160` clusters; spill disappears. | It aliases a DBUF slot identity term that must remain pointer-identical for matching. | Clone constant-add base at `_frag_b128_loads` LDS carrier reconstruction. |
| `S88` | `v88 = V_IADD(V_IMUL, 64)` | Later clusters `977/1048/1119/1190` use local cloned bases; no generic LDS load/store remat explosion. | Cloning recursively remats the shared `V_IMUL` or creates extra long-lived offsets. | Clone only the final `V_IADD`; keep shared `V_IMUL`. |
| `S136` | `v136 = V_IADD(V_IMUL, 112)` | Range `[103..1444]` is split for clusters `1231/1302/1373/1444`; spill disappears. | The expression is part of a contiguous b128 pair proof and cloning breaks matcher identity. | Add proof that cloned expression preserves `base + const` equivalence, not object identity. |
| `S120` | `v120 = V_IADD(V_IMUL, 96)` | Range `[99..1474]` is split for clusters `1261/1332/1403/1474`; spill disappears. | Spill simply moves to the shared `V_IMUL`/`V_AND` base. | Pair with `S136` as a mirrored high-slot test; accept only if both improve. |

Cross-scope acceptance:

```text
REGALLOC_END_NO_SOURCE_LIVE=1
PREFILL_DBUF_DIRECT_B128_CHAIN=1
PREFILL_DBUF_LDS_ADDR_USE_DEP=1
PREFILL_DBUF=1
PREFILL_TC_LOCAL_STAGE=both
...

must report:
REGALLOC_SPILLS: count=0
```

If only one or two spills remain, inspect whether the new spills are the shared `V_IMUL` terms. If so, the final primitive
must clone one level deeper for only those constants, not enable broad recursive rematerialization.

### Store-Side Split Result

`PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1` clones only the final store-side `V_IADD(..., CONST)` under
`_withlocal_b128_store` before `DS_STORE_B128`. This removes the early store use from the five original spilled
constant-add ranges, but does not reach zero spills:

| Spill | Definition | Live range | Note |
|---|---|---|---|
| `v56` | `V_IADD` | `[113, 700, 771, 842, 913]` | later LDS fragment clusters only |
| `v159` | `V_IMUL` | `[109, 110, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 701, 985, 1269, 1553]` | shared multiplicative base now spills |
| `v72` | `V_IADD` | `[114, 670, 741, 812, 883]` | later LDS fragment clusters only |
| `v40` | `V_IADD` | `[112, 1522, 1593, 1664, 1735]` | later LDS fragment clusters only |
| `v8` | `V_AND` | `[108, 109, 134, 2047]` | shared masked lane/base term now spills |

Next cut point: clone the final LDS fragment-load-side constant-add carrier at each `_frag_b128_loads` use cluster, then
only consider cloning one level deeper for the shared `V_IMUL`/`V_AND` terms if those remain as the sole spills.

### Two-Level Load-Side Remat Attempt

`PREFILL_DBUF_LDS_BASE_REMAT_DEEP=1` extends `PREFILL_DBUF_LDS_BASE_REMAT=1` to recurse through the LDS load address:

```text
V_OFFSET(V_IADD(V_IMUL(V_AND(...), const), const), 1)
```

It clones:

- the final `V_IADD(..., CONST)`;
- one level deeper through `V_IMUL`;
- the `V_AND` source when reached through that `V_IMUL`.

Result with store-side split:

| Variant | Result | Decision |
|---|---|---|
| `PREFILL_DBUF_LDS_BASE_REMAT_DEEP=1` with order/WMMA dep | Still `REGALLOC_SPILLS: count=5`; same live ranges. | The clone exists but does not get a strong enough late scheduling anchor. |
| Same, but anchored to the reduce `RANGE` | Worsens to `REGALLOC_SPILLS: count=107`; peak moves to `74`. | Reject. Pulling all address work into the loop is too broad. |

Conclusion: the remaining fix is not simply "clone two levels down." It needs a narrower cluster-local anchor, likely the
specific previous `DS_LOAD_B128`/`V_WMMA` use boundary for each resident fragment reload, not the whole reduce `RANGE`.

### Cluster-Local Scheduling Anchor Attempt

`PREFILL_DBUF_LDS_RELOAD_ANCHOR=1` moves the load-side remat inside the two-`DS_LOAD_B128` loop in `_frag_b128_loads`
and uses the existing `PREFILL_DBUF_LDS_LOAD_SERIAL=1` dependency chain as the cluster-local anchor.

Result with:

```text
REGALLOC_END_NO_SOURCE_LIVE=1
PREFILL_DBUF_LDS_RELOAD_ANCHOR=1
PREFILL_DBUF_LDS_LOAD_SERIAL=1
PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1
PREFILL_DBUF_LDS_BASE_REMAT=1
PREFILL_DBUF_LDS_BASE_REMAT_DEEP=1
PREFILL_DBUF_DIRECT_B128_CHAIN=1
PREFILL_DBUF_LDS_ADDR_USE_DEP=1
PREFILL_DBUF=1
```

Still fails:

```text
REGALLOC_SPILLS: count=5 stack_size=20
  V_IADD ranges remain later-cluster-only
  V_IMUL/V_AND shared roots remain live
```

The centered debug window confirms the anchor affects some generated address nodes, but the spilled canonical bases are
still direct sources of the first `V_OFFSET` in each resident-fragment reload pair. So the next cut point is lower than
the current `_frag_b128_loads` remat wrapper: either the LDS index carrier entering `isel_index` must be structurally
split before instruction selection, or the first `V_OFFSET` producer itself needs a late-bound carrier that prevents it
from retaining the original base UOp.

### Pre-Isel LDS Index Split Attempt

`PREFILL_DBUF_LDS_INDEX_SPLIT=1` perturbs both store and load LDS `INDEX` carriers before `isel_index`:

- store side: `_pack_withlocal_lds_stores` and `_pack_b_tilekey_lds_stores` before `_index_after_dep(...).store(...)`;
- load side: `_frag_b128_loads` before `isel_index(ctx, idx0)`;
- recursive symbolic split tags `ADD/MUL/SHL/AND` offset subexpressions so UOp interning cannot reuse the original roots.

Result with the useful prior flags:

```text
REGALLOC_DEBUG: PEAK 60 live vregs
REGALLOC_SPILLS: count=4 stack_size=16
  three V_IADD later-cluster bases
  one V_AND shared root
```

This is the first identity-split experiment that changes the remaining failure class: the shared `V_IMUL` spill disappears
and the spill count drops from `5` to `4`, but peak rises from `53` to `60`.

Additional checks:

| Variant | Result | Decision |
|---|---|---|
| Add `PREFILL_DBUF_LDS_BASE_REMAT=1 PREFILL_DBUF_LDS_BASE_REMAT_DEEP=1` | Still `count=4`. | No extra benefit. |
| Add cluster-local `PREFILL_DBUF_LDS_RELOAD_ANCHOR=1 PREFILL_DBUF_LDS_LOAD_SERIAL=1` | Still `count=4`. | No extra benefit. |
| Add `PREFILL_LDS_PACK_ALLOW_POOL=1` | Worsens to `count=8`. | Not a pack-scratch reservation issue. |

Current interpretation: pre-isel identity splitting is directionally correct but increases local address instruction count
enough to expose a smaller capacity boundary. The next experiment should not add more clones broadly; it should either
target only the three surviving `V_IADD` families, or reduce resident-fragment/pinned pressure by a small amount.

### Final Local Push Before Stop

Additional experiments after recursive index splitting:

| Experiment | Result | Decision |
|---|---|---|
| `PREFILL_DBUF_LDS_CONST_IMM=1` before the B128 guard fix | Worsened to `REGALLOC_SPILLS: count=55`, with many `V_PACK` spills. | This was a bad fallback, not proof against immediate folding. |
| `PREFILL_DBUF_LDS_STORE_IMM_FOLD=1` | Still `REGALLOC_SPILLS: count=4`. | No benefit. |
| `REGALLOC_NO_LOOP_EXTEND_ADDR=1` | Still `REGALLOC_SPILLS: count=4`. | Not a loop-extension artifact. |
| `AMD_ISA_WMMA_LOW_SCRATCH=0` | Worsens to `REGALLOC_SPILLS: count=99`. | Low scratch is essential; this is capacity-sensitive. |
| `PREFILL_LDS_PACK_ALLOW_POOL=1` | Worsens to `REGALLOC_SPILLS: count=8`. | Do not relax the fixed b128 scratch reservation. |

### Fable Design Review Follow-Up: Immediate Offset Folding

External design review correctly pointed at the primitive ISA-level shape:

```text
DS op should see:       addr = dynamic_base, imm_offset = static_fragment_offset
Not:                   addr = dynamic_base + static_fragment_offset
```

The native ISA path already has an immediate field for `DS_LOAD_B128`, `DS_STORE_B128`, and `DS_STORE_B64`. The first
follow-up found a concrete experiment bug: `isel_index` can return an LDS carrier with a third immediate source when
`PREFILL_DBUF_LDS_CONST_IMM=1`, but `_frag_b128_loads` only accepted `len(idxc.src) == 2`. That forced the packed
fragment loader to reject the compact B128 path and fall back to scalar/V_PACK lowering, explaining the old `count=55`
result.

Patch:

```text
_frag_b128_loads: accept LDS carriers with len(idxc.src) in (2, 3)
```

Corrected result:

```text
REGALLOC_DEBUG: 2650 uops, PEAK 60 live vregs @ uop 280
REGALLOC_SPILLS: count=4 stack_size=16
  SPILL V_IADD range=[113, 692, 761, 830, 899]
  SPILL V_IADD range=[114, 663, 732, 801, 870]
  SPILL V_IADD range=[112, 1491, 1560, 1629, 1698]
  SPILL V_AND  range=[109, 110, 121, 142, 2008]
```

Interpretation:

- Immediate folding is real and now stays on the compact B128 path (`2697 -> 2650` uops).
- It does not eliminate the remaining four spills.
- The survivors are dynamic LDS base/address carriers (`V_IADD(V_IMUL, const)` and `V_AND(lidx, 15)`), not per-fragment
  static offsets.
- Re-testing deep remat plus reload anchoring after the B128 guard fix still leaves `count=4`.

Therefore the refined primitive is:

1. keep immediate folding as part of the correct DS/LDS representation;
2. do not expect immediate folding alone to solve 4x4 DBUF;
3. the remaining required primitive is allocator-aware rematerialization or a pseudo-op that keeps dynamic LDS base math
   out of ordinary long-lived VGPR intervals until the DS use cluster.

### Narrow Allocator Remat Implementation

The remaining four spills were scoped separately:

| Value | Shape | Solution |
|---|---|---|
| `V_IADD(V_IMUL, 16)` | dynamic LDS address base used by later `V_OFFSET -> DS_LOAD_B128` clusters | rematerialize pure address producer at the memory-address use |
| `V_IADD(V_IMUL, 32)` | same | rematerialize pure address producer at the memory-address use |
| `V_IADD(V_IMUL, 48)` | same | rematerialize pure address producer at the memory-address use |
| `V_AND(lidx, 15)` | shared lane/root term feeding address math | rematerialize only when used by the same pure address chain |

Implementation:

- `REGALLOC_ADDR_REMAT=1` adds a narrow allocator escape in `regalloc.py`.
- It applies only to integer `INS` producers with `AMDOps.V_AND`, `V_IMUL`, `V_IADD`, or `V_OFFSET`.
- It applies only when the current use is another pure address op or a memory address op (`DS_*`, `GATED_STORE_*`,
  `GLOBAL_*`).
- Instead of assigning a stack slot, it allocates a real register and emits a cloned producer chain immediately before
  the use.
- This is not a general spiller and does not rematerialize loads, stores, WMMA fragments, or packed data registers.

Result:

```text
REGALLOC_ADDR_REMAT=1
REGALLOC_DEBUG: 2730 uops, PEAK 60 live vregs @ uop 281
REGALLOC_SPILLS: count=0 stack_size=0
```

The first no-spill lowering exposed a second, independent legality bug: some folded DS offsets were `8192`, but this
RDNA3 encoder's `offset0` field is 8 bits. `_ds_addr_imm` now checks the field; offsets that fit stay encoded as DS
immediates, and larger offsets become a late `V_IADD(addr, imm)` with `offset0=0`.

Verification:

```text
python3 -m py_compile tinygrad/codegen/late/regalloc.py tinygrad/renderer/isa/amd.py
PYTHONPATH=. pytest -q test/unit/test_amd_isa_wmma.py
```

The structural stream probe now emits a final no-spill stream with WMMA operands coming from `ds_load_b128`.

Numerical boundary:

```text
REGALLOC_ADDR_REMAT=1 ... python3 -m extra.qk.prefill_graph_gemm_route_bound_stage_gate --run-amd --local-stage both --compact
finite=true
rel_rmse_vs_ref=1.218783974647522
```

So this pass closes the allocator/spill blocker. It does not close the both-side LDS layout/correctness blocker; the next
layer is the A/B LDS layout/read contract, not register pressure.

Updated stopping point: the allocator/spill layer is no longer the blocker with `REGALLOC_ADDR_REMAT=1`; the route
compiles to a final no-spill stream. The remaining failure is numerical correctness for both-side LDS staging, so the
next productive direction is the A/B LDS layout/read contract:

1. validate that each `ds_store_b128` writes the tile layout that `_frag_b128_loads` later reads;
2. compare A and B address-family keys against the logical row/column WMMA operand expectations;
3. keep `REGALLOC_ADDR_REMAT=1` and checked DS offsets enabled during that investigation so pressure does not mask the
   layout issue.
