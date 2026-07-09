# 8B Prefill `ffn_gate_up` LDS Primitive Scope

Date: 2026-07-09.

## Decision

Use hand ASM for `ffn_gate_up` LDS only as an oracle/executable spec. The endpoint is a compiler-owned LDS WMMA
primitive selected through a declarative `WMMALDSSpec`, not a permanent hand-written full kernel.

This is separate from Path 1 pipe-role completion:

- pipe roles complete: `attn_qo`, `attn_kv`, `ffn_down`,
- LDS role remaining: `ffn_gate_up`.

## Current Raw Route

Current call path:

```text
route_pf16_graph_gemm
  -> describe_prefill_schedule(out_f=12288, in_f=4096, role=ffn_gate_up)
  -> route_family="lds"
  -> emit_prefill_gemm_from_spec
  -> _emit_schedule
  -> extra/qk/prefill/wmma.py::build_gemm_lds2
  -> UOp(Ops.INS, ...)
  -> Tensor.custom_kernel
```

The resolved 8B shape is:

```text
M=512, N=12288, K=4096
tile_m=128, tile_n=128, tile_k=32
waves_m=4, waves_n=2, wm=2, wn=4
threads=256, dbuf=1, plra=0, plrab=1, pad=16, leanaddr=0
```

## Primitive Spec

New declarative contract:

```text
extra/qk/wmma_lds_spec.py::WMMALDSSpec
```

The spec owns:

- operand contract: fp16 A `[M,K]`, fp16 B-transposed `[N,K]`, fp16 C `[M,N]`,
- tile geometry: `tile_m`, `tile_n`, `tile_k`,
- wave geometry: `waves_m`, `waves_n`, `wm`, `wn`, `threads`,
- LDS layout: `stride_a_bytes`, `stride_b_bytes`, `lds_a_bytes`, `lds_buffer_bytes`, `lds_total_bytes`,
- cooperative mapping: `cpr`, `row_stride`, `loads_a`, `loads_b`,
- pipeline fields: `dbuf`, `plra`, `plrab`, `leanaddr`,
- resource counters: accumulator VGPRs and cooperative temp VGPRs,
- legality checks.

For the current 8B `ffn_gate_up` route, the extracted spec reports:

```text
row_stride=64
loads_a=2
loads_b=2
lds_total_bytes=40960
plr_mode=A+B
legality_errors=[]
```

## Artifact

Existing harness surface:

```text
extra/qk/prefill_pipe_mvp_artifact.py --lds-primitive
```

Output:

```text
bench/prefill-pipe-mvp/ffn-gate-up-lds-primitive.json
```

Current artifact status:

```text
PREFILL_LDS_PRIMITIVE_SCOPED_BLOCKED_ON_GENERATED_LOWERER
```

This is expected. It means the spec/oracle/gate are in place, and the remaining work is generated lowering.

## Prior LDS Work Reused

This scope does not restart LDS lowering. It wraps the existing LDS/DBUF substrate in a route-owned `ffn_gate_up`
contract so we can retire the raw oracle without copying it.

Existing work to reuse:

- `docs/generated-machine-code-lds-dbuf-100pct-scope.md`: generated LDS DBUF objective, blocker taxonomy, and gates.
- `docs/native-isa-l4-software-pipeline-scope.md`: current L4 status, wide LDS primitive split, and spill/pressure
  blockers.
- `docs/dbuf-safe-ds-offset-folding-scope.md`: `LDSAddr`, `decompose_lds_index`, safe DS offset folding, and current
  materialized-vs-immediate correctness matrix.
- `tinygrad/renderer/isa/amd.py`: `GLOBAL_LOAD_B128`, `DS_STORE_B128`, `DS_LOAD_B128`, `LDSAddr`, and DS offset proof
  helpers.
- `tinygrad/codegen/opt/postrange.py`: cooperative LDS staging and DBUF/local-stage rewrite machinery.
- `extra/qk/prefill/native_isa_l4_stream_probe.py` and `extra/qk/prefill/kernel_lifecycle_trace.py`: structural trace
  and lifecycle acceptance probes.

The non-duplicate part added here is only:

- the `ffn_gate_up`-specific `WMMALDSSpec`,
- the fail-closed `PREFILL_WMMA_LDS_PRIMITIVE=1` route seam,
- the artifact tying the existing oracle trace to generated-route ownership.

Do not build a second LDS lowerer beside the above substrate. L2 must adapt the existing `postrange.py` and AMD renderer
paths, with this spec as the route contract and acceptance surface.

## Oracle Trace

The oracle trace uses the existing lifecycle tracer:

```sh
PYTHONPATH=. DEV=AMD:ISA:gfx1100 \
  python3 extra/qk/prefill/kernel_lifecycle_trace.py \
  --kind hand-lds2 --m 512 --n 12288 --k 4096 \
  --waves-m 4 --waves-n 2 --wm 2 --wn 4 --bk 32 --pad 16 --dbuf 1 --plrab 1 --json
```

Current oracle structural counters:

| Counter | Value |
|---|---:|
| `global_load_b128` | 16 |
| `ds_store_b128` | 16 |
| `ds_load_b128` | 96 |
| `s_barrier` | 4 |
| `v_wmma_f32_16x16x16_f16` | 64 |
| scalar LDS fallback | 0 |
| WMMA operand origins | `ds_load_b128/ds_load_b128` |

## Completion Phases

### L0. Spec And Oracle Gate

Done when:

- `WMMALDSSpec` extracts from `PrefillGEMMScheduleSpec(route_family="lds")`,
- legality checks cover tile, threads, LDS, PLR, and invalid probes,
- artifact records oracle trace and explicitly marks generated lowering as not implemented.

### L1. Fail-Closed Generated Seam

Done when:

- `PREFILL_WMMA_LDS_PRIMITIVE=1` diverts only `route_family="lds"` to `lower_wmma_lds_spec`,
- unsupported specs do not call `build_gemm_lds2`,
- no selected generated route can silently fall back while claiming generated ownership.

### L2. Generated Single-Buffer LDS Candidate

First generated lowerer target:

```text
global_load_b128 -> ds_store_b128 -> s_barrier -> ds_load_b128 -> WMMA -> epilogue
```

Keep DBUF/PLR disabled for the first generated correctness candidate if necessary. Correctness comes before matching
the oracle cadence.

Implementation constraint: use the existing generated LDS substrate above. A valid L2 patch should route through
`postrange.py`/AMD renderer primitives and the centralized probes, not a new route-local `Ops.INS` body or a parallel
hand-shaped emitter.

Current L2 transport decision:

- `PREFILL_WMMA_LDS_PRIMITIVE=1` diverts `route_family="lds"` before `emit_prefill_gemm_from_spec` can call the raw
  oracle.
- The route executes ordinary generated matmul transport, analogous to the pipe primitive MVP.
- The env bundle reuses the existing single-buffer LDS substrate:
  `PREFILL_TC_LOCAL_STAGE=both`, `PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1`, `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1`,
  `PREFILL_LDS_PACK_WITHLOCAL_B128=1`, `AMD_ISA_WMMA_B128_FRAG=1`, `AMD_ISA_REG_ACCUM=1`.
- DBUF is intentionally not enabled by this default bundle. Prior commits showed DBUF correctness depends on
  slot-cadence and DS-offset folding proof; those stay in L3.

Required trace:

- `global_load_b128 > 0`,
- `ds_store_b128 > 0`,
- `ds_load_b128 > 0`,
- `s_barrier > 0`,
- `wmma > 0`,
- scalar LDS fallback total is zero,
- WMMA operands originate from `ds_load_b128`.

### L3. DBUF + PLRAB Parity

After L2 correctness:

- add two-slot LDS identity,
- add future-slot work before current compute,
- add PLRAB fragment lifetime separation,
- prove no LDS byte-window aliasing.

The current lifecycle tracer has structural address-family proof, not resolved byte-window proof. Byte-window proof is a
known strengthening requirement.

### L4. Route-Bound `ffn_gate_up`

Done when:

- `route_pf16_graph_gemm(... role=ffn_gate_up ...)` with `PREFILL_WMMA_LDS_PRIMITIVE=1` executes through generated
  lowering,
- correctness passes against fp32 sampled reference,
- artifact reports `generated_lds_selected=true`, `uses_hand_lds_oracle=false`.

### L5. Whole-Prefill Mixed Smoke

Done when:

- whole-prefill smoke runs with pipe roles generated and `ffn_gate_up` generated LDS,
- route attribution reports no raw graph-GEMM oracle for the fp16 prefill roles.

### L6. Promotion Timing

Only after L4/L5:

- compare against the raw LDS oracle at same clocks,
- keep the raw route as rollback/oracle until generated throughput is acceptable.

## Hard Stop Conditions

Stop and do not promote if:

- the generated path calls `extra/qk/prefill/wmma.py::build_gemm_lds2`,
- route-local full-kernel `UOp(Ops.INS)` remains in the selected generated path,
- correctness fails or emits NaN/Inf,
- LDS total exceeds 64 KiB,
- scalar LDS fallback appears,
- WMMA operands do not come from generated LDS loads.

## Current Blocker

The generated LDS lowerer is not implemented. The repo now has:

- spec extraction,
- legality/resource checks,
- fail-closed lowering seam,
- oracle trace artifact.

The next concrete implementation step is L2: a generated single-buffer LDS candidate.
