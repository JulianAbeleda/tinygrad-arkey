# DBUF LDS Address Lifetime Serialization Scope

Date: 2026-07-07.

## Objective

Make the generated `PREFILL_DBUF=1` 4x4 WMMA LDS route emit a final native ISA stream without spills by reducing LDS
address live ranges before scheduler/waitcnt tuning.

This scope starts from the working non-DBUF substrate:

```text
A/B -> wide LDS stores -> barrier -> wide LDS loads -> WMMA
```

and targets the current DBUF blocker:

```text
Inc 0: no spills
REGALLOC_DEBUG peak dominated by V_OFFSET / V_IADD LDS address producers
```

## Thesis

DBUF is currently blocked before scheduling. The packed data path is mostly solved:

- A+B non-DBUF stages through `ds_store_b128` / `ds_load_b128`.
- B tile-key DBUF slot math exists.
- B tile-key bridge handles DBUF's `16 x GROUP(8 stores)` shape.
- DBUF central correctness passes.

The remaining blocker is address-side pressure. `PREFILL_DBUF=1` peels the K phase and exposes two slot phases in one
body. That creates many independent LDS address expressions for A/B, slot 0/1, stores/loads, and subtile heads. These
lower to `V_OFFSET` and `V_IADD` VGPRs, and too many stay live at the same time.

Scheduler/waitcnt tuning is out of scope until this route emits a final stream.

## Current Evidence

| Probe | Result |
|---|---|
| non-DBUF both A+B | PASS, about `81` live VGPRs. |
| DBUF A-only | FAIL, about `89` live VGPRs, dominated by address producers. |
| DBUF B-only after bridge widening | FAIL, about `89` live VGPRs, dominated by address producers. |
| DBUF both after bridge widening/address threading | FAIL, about `137` live VGPRs, dominated by `V_OFFSET`/`V_IADD`. |
| DBUF central correctness | PASS, finite, accepted RMSE envelope. |

## Attempt Log

The first implementation pass tested four guarded pressure reducers:

| Flag / Patch | Result |
|---|---|
| `PREFILL_DBUF_LDS_ADDR_SERIAL=1` read-side postrange `DEFINE_LOCAL.after(prev_store).after(bar).index(...)` | Verifier-clean after removing unsafe store-address chaining, but worsened B-only pressure (`89 -> 97`) and did not solve A-only. |
| Store-address postrange chaining | Rejected. It can produce verifier-invalid B tile-key `PTRCAT` shapes. |
| `PREFILL_DBUF_LDS_LOAD_SERIAL=1` renderer LDS b128 fragment-load serialization | No material pressure reduction; A-only stayed about `88`, B-only about `89`. |
| `PREFILL_DBUF_LDS_CONST_IMM=1` DS immediate folding | Verifier-clean after preserving default LDS carrier arity, but no material pressure reduction. |
| `PREFILL_DBUF_GLOBAL_ADDR_INLOOP=1` seed packed-store address chain with reduce `RANGE` | No material pressure reduction; both stayed about `137`. |

New diagnostic evidence from `REGALLOC_DEBUG_DETAIL=1`:

- both-side DBUF peak is exactly `64 V_OFFSET + 64 V_IADD`;
- many address bases are defined before the reduce `RANGE`;
- their live ranges extend to the reduce-loop `END`, so the failure is loop-carried address hoisting, not just final
  LDS `ds_load_b128` address adjacency.

This moves the next real fix from "local b128 load/store dependency threading" to "prevent staging address bases from
being hoisted across the reduce loop, or rematerialize them inside the loop in a way regalloc can see."

## Required Design

Implement one narrow primitive first:

```text
produce LDS address family
consume it with ds_store_b128 / ds_load_b128
force address temps to die
then produce the next LDS address family
```

Prefer verifier-clean graph/effect ordering in `postrange.py`. Use renderer fallback only if graph ordering cannot reduce
pressure.

## Agent Review Additions

Three low-reasoning side agents reviewed the plan. Their combined recommendation is:

- first try buffer-level graph ordering in postrange, using `AFTER(DEFINE_LOCAL, dep).index(...)` rather than value-level
  `AFTER`;
- chain the next LDS address family behind the previous family store/load group or barrier so address expressions are
  forced to materialize close to their consuming `ds_*` instruction;
- if graph ordering is insufficient, use renderer-local address serialization before considering fixed scratch VGPRs;
- add resolved DBUF slot identity to the probe before declaring D2 complete.

The important legal primitive is:

```text
prior_done = GROUP(previous_stage_stores_or_loads)
ordered_lds = DEFINE_LOCAL.after(prior_done)
addr = ordered_lds.index(slot + row*16 + frag)
use addr in ds_store_b128 / ds_load_b128
```

This differs from the existing shared-barrier shape because sibling loads/stores must not all attach to the same broad
barrier and then remain independently live.

## Work Packages

### A0. Lock Existing Baselines

Acceptance:

- unit suite passes;
- non-DBUF both remains `ds_store_b128=16`, `ds_load_b128=16`, no scalar LDS stores;
- DBUF central correctness remains passing.

### A1. Postrange Address Serialization Experiment

Owner files:

- `tinygrad/codegen/opt/postrange.py`
- tests only if the experiment passes

Try a guarded flag, for example:

```text
PREFILL_DBUF_LDS_ADDR_SERIAL=1
```

Candidate legal shapes:

```text
INDEX(AFTER(DEFINE_LOCAL, prior_store_or_group), idx)
STORE(INDEX(...), value, gate)
```

or, for grouped stores/loads:

```text
bsh_after = bsh.after(previous_group_or_barrier)
bsh_after.index(slot + row*16 + frag)
```

Do not use:

```text
AFTER(half_value, void_barrier)
GROUP(..., BARRIER(...), ...)
STORE(AFTER(INDEX(...), dep), value)
```

Acceptance:

- A-only DBUF peak drops and compiles no-spill;
- B-only DBUF peak drops and compiles no-spill;
- both DBUF peak drops materially;
- `SPEC=1` remains clean;
- non-DBUF output unchanged when the flag is off.

### A2. Narrow-Breadth Fallback

If A1 is insufficient, reduce simultaneous DBUF breadth before a renderer fallback:

- serialize A rows before B rows;
- serialize slot 0 family before slot 1 family;
- serialize store-address production before load-address production;
- prove A-only and B-only DBUF separately before `both`.

Acceptance:

- A-only and B-only DBUF compile no-spill independently;
- both-side failure, if any, has a smaller and explainable peak.

### A3. Renderer Address Fallback

Use only if A1/A2 leave pressure dominated by `V_OFFSET`/`V_IADD`.

Ranked fallback order:

1. Strengthen DBUF/local LDS b128 address-chain serialization in `_frag_b128_loads`, maintaining a small sequence per
   LDS family or fragment so each next `ds_load_b128` address depends on the previous fragment load/carrier.
2. Fold constant LDS byte offsets into DS instruction immediates where the DS offset range and byte/word units are
   proven correct.
3. Add a renderer-local LDS address scratch/reuse path with explicit RAW/WAR dependencies and scratch registers reserved
   outside the normal vpool.
4. Strengthen B tile-key store chaining per DBUF slot or tile phase if B-only still fails after load-side fixes.
5. Use lower-level fixed scratch in `lower_inst` only as a last resort, because it runs after regalloc and can hide real
   liveness unless the scratch is globally reserved.

Target locations:

- `tinygrad/renderer/isa/amd.py::isel_index`
- `_frag_b128_loads`
- `_pack_b_tilekey_lds_stores`
- packed LDS store/load bridge helpers

Allowed direction:

- fail-closed address scratch/reuse;
- dependency-threaded address production;
- small fixed address window if correctness and ordering are explicit.

Rejected direction:

- adding spills;
- broad global scheduling changes;
- waitcnt tuning.

## Gate Matrix

Run these after each implementation attempt.

Non-DBUF regression:

```bash
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

A-only DBUF pressure:

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

B-only DBUF pressure:

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

Both DBUF pressure:

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

## Probe and Test Additions

Current `lds_address_families` keys by address VGPR plus immediate. That is useful for pressure analysis, but it does not
prove resolved DBUF slot identity. Add a probe field that classifies LDS accesses as:

```text
{ operand: A|B, slot: 0|1, byte_offset_or_family: ... }
```

D2 is complete only when the probe can prove two slots for A and two slots for B, not just multiple address families.

After the DBUF native route compiles no-spill, add a focused full 4x4 structural test for:

- `PREFILL_DBUF=1`;
- `PREFILL_TC_LOCAL_STAGE=both`;
- `ds_store_b128=16` and `ds_load_b128=16`;
- zero scalar LDS stores;
- both WMMA operands sourced from `ds_load_b128`;
- `dbuf_gate_summary.D2/D3/D7.ok=true`;
- no spill.

## Stop Conditions

| Stop | Meaning |
|---|---|
| Solved for this scope | A-only, B-only, and both DBUF compile no-spill; probe can inspect D2/D3/D7 final-stream fields. |
| Verifier blocked | Legal effect ordering cannot be expressed in UOp graph. Move to renderer fallback. |
| Pressure still address-dominated | A1/A2 do not reduce `V_OFFSET`/`V_IADD` enough. Move to renderer address scratch/reuse. |
| Correctness blocked | DBUF compiles but central correctness fails. Debug slot address/read contract before scheduler tuning. |
| Cadence blocked | DBUF compiles and is correct but D3 has no body work. Scope explicit prologue/body/tail graph next. |

## Anti-Goals

- waitcnt tuning;
- scheduler latency tuning;
- TFLOPS promotion;
- handwritten assembly;
- spills;
- broad devectorizer guards;
- value-level half `AFTER`;
- direct B `global_load_b128` unless pressure evidence shifts back to data producers.
