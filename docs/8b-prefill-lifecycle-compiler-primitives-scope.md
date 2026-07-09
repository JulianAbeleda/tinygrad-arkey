# 8B Prefill Lifecycle To Compiler Primitives Scope

Date: 2026-07-08.

## Thesis

The target is not "no ASM" and not "fine-tuned hand kernels everywhere."

The target is:

```text
machine search over reusable compiler primitives
```

Humans may implement reusable compiler/backend primitives. Machine search/specs choose how to compose them. A route-local
implementation becomes a hand kernel only when it manually emits the complete model/shape-specific lifecycle:

```text
global loads -> LDS/VGPR staging -> waits/barriers -> WMMA loop -> epilogue stores
```

The current 8B `PREFILL_GRAPH_GEMM=1` route is the oracle/escape hatch. It reproduced:

| whole prefill | tok/s |
|---:|---:|
| 512 | 5111 |
| 1024 | 4910 |
| 2048 | 4428 |
| 4096 | 3677 |

Artifact: `bench/prefill-whole-synced/graph-gemm-8b-refresh-20260708.json`.

## Current Oracle Role Map

Resolved through `describe_prefill_schedule(out_f, in_f, role=...)`.

| Role | Shape `(M,N,K)` | Oracle family | Hand builder | Primary lifecycle |
|---|---:|---|---|---|
| `attn_qo` | `512,4096,4096` | `pipe` | `build_gemm_pipe` | global b128 -> VGPR fragments -> targeted vmcnt -> WMMA -> store |
| `attn_kv` | `512,1024,4096` | `pipe` | `build_gemm_pipe` | global b128 -> VGPR fragments -> targeted vmcnt -> WMMA -> store |
| `ffn_down` | `512,4096,12288` | `pipe` | `build_gemm_pipe` | global b128 -> VGPR fragments -> targeted vmcnt -> WMMA -> store |
| `ffn_gate_up` | `512,12288,4096` | `lds` | `build_gemm_lds2` | global b128 -> LDS stage -> barrier -> ds_load fragments -> lgkm wait -> WMMA -> store |

Default spec fields for all four roles:

```text
tile_m=128, tile_n=128, tile_k=32
waves_m=4, waves_n=2, wm=2, wn=4
threads=256, dbuf=1, pad=16
pipe_tm=2, pipe_tn=2, waitcnt_policy=targeted_vmcnt
```

`ffn_gate_up` is protected from the pipe route by `out_f == 12288`, because the pipe regressed that saturated role.

Important nuance: the `pipe` builder consumes only `M,N,K,TM,TN`. The spec still carries `bm/bn/threads/bk/pad/dbuf/plrab`
because the resolver is shared, but `build_gemm_pipe` itself uses the register-resident pipe lifecycle rather than the
LDS fields. The LDS builder consumes `WAVES_M/WAVES_N/WM/WN/BK/PAD/DBUF/PLRA/PLRAB/LEANADDR`.

## Ranked Lifecycle Plan

Generate from the edges inward. Do not start by cloning `build_gemm_lds2` as a giant macro.

| Rank | Lifecycle piece | Target classification | Action | Why |
|---:|---|---|---|---|
| 1 | Route/shape/role policy | search/spec | Keep and harden | Already data-owned by route/spec. Search should own role selection. |
| 2 | Tile/schedule parameters | search/spec | Keep and expand | `PrefillGEMMScheduleSpec` already captures most knobs. |
| 3 | WMMA instruction lowering | compiler primitive | Keep/grow backend-owned lowering | Reusable across all WMMA kernels; not a hand kernel by itself. |
| 4 | Targeted waitcnt | compiler primitive | Extract from hand helper into backend policy | Small, reusable, high leverage; `waitcnt_vm/lgkm` prove the needed encoding. |
| 5 | b128 global/DS load-store lowering | compiler primitive | Use backend-owned load/store primitives | Needed to close instruction-count gap. |
| 6 | DS offset folding / address lifetime | compiler primitive | Continue current work | Prevents VGPR address-carrier bloat in generated LDS paths. |
| 7 | WMMA fragment/register layout | compiler/regalloc primitive | Continue current work | Necessary for correctness and pressure, but harder than wait/load primitives. |
| 8 | Epilogue store pattern | generated backend | Generate | Isolated fp32 accumulator -> fp16 output store; less coupled than staging. |
| 9 | F0/F1 two-stage pipe idiom | scheduler primitive | Extract first | Best first middle-pipeline primitive; avoids LDS complexity. |
| 10 | Basic LDS staging | scheduler primitive | Extract after pipe | Necessary for `ffn_gate_up`, but adds barriers/LDS offsets/lifetimes. |
| 11 | DBUF LDS ping-pong | scheduler primitive later | Keep hand ASM for now | High value, but couples waits, barriers, LDS slots, and register lifetimes. |
| 12 | Full LDS2 lifecycle | oracle/escape hatch | Keep hand ASM for now | Full lifecycle equals hand kernel; do not rename it as generated. |
| 13 | Q4_K fused decode -> LDS -> WMMA | research/oracle | Keep out of first 8B fp16 path | Too specialized and quant-coupled. |
| 14 | Full role-specific raw kernel body | oracle only | Do not expand | This is the anti-target. |

## 2026-07-09 Ownership Extraction Checkpoint

Goal for this pass: remove human ownership of pieces of the `build_gemm_lds2` lifecycle without changing the emitted
default kernel or claiming the route is generated.

Implemented in `extra/qk/prefill/wmma.py`:

| Slice | New owner object | Default behavior |
|---|---|---|
| LDS2 physical VGPR ranges | `LDS2RegLayout` + `default_lds2_reg_layout(...)` | Same formulas as the prior inline `FA/FB/ACCb/CTA/CTB/SCR/FB2` arithmetic. |
| LDS2 memory layout | `LDS2MemoryLayout` + `default_lds2_memory_layout(...)` | Same `SA/SB/LDS_A/BUFSZ/NBUF` formulas as the prior inline arithmetic. |
| LDS2 wait counts | `LDS2WaitPolicy` + `default_lds2_wait_policy()` | Same `vmcnt(0)` / `lgkmcnt(0)` counts at the same placements. |
| LDS2 cadence selector | `LDS2Cadence` + `default_lds2_cadence(...)` | Same single-buffer vs DBUF branch as the prior `if not DBUF` control flow. |
| LDS2 lifecycle template | `LDS2LifecycleTemplate` + `default_lds2_lifecycle_template(...)` | Same prologue/body/tail sequence, including DBUF slot alternation, K-advance, loop branch, and tail compute. |
| LDS2 primitive emitter | `LDS2PrimitiveEmitter` | Same cooperative loads/stores, WMMA compute, PLR variants, K-advance, and lifecycle-step lowering, moved out of `build_gemm_lds2` local closures. |
| LDS2 shell/epilogue emitter | `LDS2PrimitiveEmitter.emit_kernel_prologue`, `emit_tile_setup`, `zero_accumulators`, `emit_epilogue`, `emit_kernel_end` | Same setup, indexing, accumulator init, output stores, and program end, moved out of the route-local shell. |
| LDS2 lowerer boundary | `lower_lds2_gemm_kernel(...)` | `build_gemm_lds2(...)` is now a compatibility wrapper around the named lowerer surface. |
| S9 wait-policy search | `lds2_s9_wait_search.py` + `PREFILL_LDS2_WAIT_*` env knobs | First non-byte-identical search axis. Current layout/lifecycle held fixed; only wait counts vary. |

Proofs:

| Gate | Result |
|---|---|
| Unit byte identity | `PYTHONPATH=. pytest -q test/unit/test_prefill_wmma_lds2_reg_layout.py test/unit/test_wmma_lds_spec.py` -> `24 passed`. |
| Active LDS2 stream | `build_gemm_lds2(512,12288,4096,4,2,2,4,32,16,1,PLRAB=1)` remains `3732` bytes; implicit defaults equal explicit layout/policy/cadence/lifecycle objects. |
| Authority after S1 | `bench/prefill-whole-synced/raw-hand-s1-reg-layout-authority.json`: pp512 `4411 tok/s`, pp4096 `3231 tok/s`. |
| Authority after S2 | `bench/prefill-whole-synced/raw-hand-s2-layout-authority.json`: pp512 `4409 tok/s`, pp4096 `3235 tok/s`. |
| Authority after S3 | `bench/prefill-whole-synced/raw-hand-s3-wait-policy-authority.json`: pp512 `4408 tok/s`, pp4096 `3229 tok/s`. |
| Authority after S4 | `bench/prefill-whole-synced/raw-hand-s4-cadence-authority.json`: pp512 `4408 tok/s`, pp4096 `3234 tok/s`. |
| Authority after S5 | `bench/prefill-whole-synced/raw-hand-s5-lifecycle-template-authority.json`: pp512 `4415 tok/s`, pp4096 `3235 tok/s`. |
| Authority after S6 | `bench/prefill-whole-synced/raw-hand-s6-primitive-emitter-authority.json`: pp512 `4414 tok/s`, pp4096 `3233 tok/s`. |
| Authority after S7 | `bench/prefill-whole-synced/raw-hand-s7-shell-emitter-authority.json`: pp512 `4403 tok/s`, pp4096 `3227 tok/s`. |
| Authority after S8 | `bench/prefill-whole-synced/raw-hand-s8-lowerer-wrapper-authority.json`: pp512 `4392 tok/s`, pp4096 `3225 tok/s`. |
| S9 wait micro search | `bench/prefill-lds2-s9/wait-search.json` and `wait-search-repeat.json`: VMEM wait relaxation is wrong; LDS-side wait relaxation is correct but only small microkernel movement. |
| S9 store-wait authority | `bench/prefill-whole-synced/raw-hand-s9-wait-store2-authority.json`: `PREFILL_LDS2_WAIT_LGKM_COOP_STORE=2`, pp512 `4416 tok/s`, pp4096 `3237 tok/s`. |
| S9 frag-wait authority | `bench/prefill-whole-synced/raw-hand-s9-wait-frag2-authority.json`: `PREFILL_LDS2_WAIT_LGKM_FRAG_LOAD=2`, pp512 `4385 tok/s`, pp4096 `3222 tok/s`. |
| S9 interleaved repeat | `s9-repeat-default-a.json`, `s9-repeat-coop-store2-a.json`, `s9-repeat-frag-load2-a.json`: default pp512 `4397`; store2 pp512 `4421`; frag2 pp512 `4390`. |

Interpretation:

- This is a successful ownership reduction checkpoint: register layout, LDS layout, wait-count choices, the
  single-buffer/DBUF cadence selector, and the current DBUF prologue/body/tail lifecycle are now explicit data objects;
  LDS2 middle-pipeline primitive emission, shell setup, and epilogue emission are now behind a reusable emitter object
  instead of route-local closures.
- This does not make the route pure. It still executes the raw `wmma.py` lifecycle through `Ops.INS`, so it remains an
  `asm_oracle` / escape hatch under `docs/asm-tool-vs-hand-kernel-policy-scope.md`.
- The byte-preserving extraction path is now at the useful boundary: `build_gemm_lds2` is only a compatibility wrapper
  around the lowerer. S7/S8 measured a small pp512 dip, but the emitted active LDS2 stream remains byte-identical, so this
  is measurement noise rather than a real performance change. A meaningful performance difference now requires a
  non-default search choice or replacing raw `Ops.INS` emission with backend/codegen emission.
- S9 wait-only search found the first intentional performance movement:
  - `vm_after_coop_load > 0` is incorrect on the real active shape (`NaN`), so VMEM global-load waits are not relaxable.
  - `lgkm_after_coop_store=2` is correct and lands in the top baseline band, but does not clearly beat the oracle.
  - `lgkm_after_frag_load=2` is correct in the microkernel but loses in whole prefill. The low-roll hypothesis was tested
    with a repeat and the loss repeated: median pp512 over two authority runs was about `4387` versus `4418` for store2.
  - Conclusion: wait-only S9 is exhausted except for keeping `lgkm_after_coop_store=2` as an opt-in/promotion candidate.
    The next S9 axis must be lifecycle/template or layout search, not more wait-count tuning.

Follow-up scope:

```text
docs/8b-prefill-s9-exhaustive-search-scope.md
```

That document owns the exhaustive S9 search plan across wait policy, lifecycle templates, register layout, later LDS
memory layout, and promotion/reporting. S10 remains separate: backend/codegen emission replacing raw `Ops.INS`.

## Phase Plan

### P0. Pin Oracle And Inventory

Goal: make the current hand route measurable and non-ambiguous.

Tasks:

- Keep `PREFILL_GRAPH_GEMM=1` opt-in.
- Keep `extra/qk/prefill/wmma.py` as oracle source.
- Record per-role builder, tile params, instruction counts, waits/WMMA, and throughput.
- Ensure purity audit classifies this route as `asm_oracle` / `external_raw_or_binary`, not generated.

Done when:

- `bench/prefill-whole-synced/graph-gemm-8b-refresh-20260708.json` is cited.
- The lifecycle table above matches route/spec output.

### P1. Primitive Inventory

Goal: identify what can be reused before writing new lowering.

Existing substrate:

| Primitive area | Existing evidence | Gap |
|---|---|---|
| WMMA lowering | `tinygrad/renderer/isa/amd.py` lowers contiguous WMMA fragments and emits `v_wmma_f32_16x16x16_f16`; `test_amd_isa_wmma.py` has b128/WMMA structural coverage. | Pipe needs explicit two-stage fragment lifetime, not only structural WMMA chains. |
| b128 global/DS ops | `GLOBAL_LOAD_B128`, `DS_LOAD_B128`, `DS_STORE_B128`, and gated `DS_STORE_B128` lower in AMD ISA renderer; tests cover b128 fragment loads and packed LDS stores. | Packed global-load -> LDS-store carrier selection is still narrow and shape/lifetime sensitive. |
| waitcnt | `_waitcnt_simm16` and `_insert_waitcnt` support encoded `vmcnt/lgkmcnt/expcnt`, targeted mode, and span-aware hazards. | Hand-quality pipe needs "leave future-stage loads outstanding" rather than draining too much. |
| LDS staging | Route-bound/local staging probes and `kernel_lifecycle_trace.py` can see LDS loads/stores, barriers, waits, operand origins, and DBUF cadence. | Structural staging can pass without being numerically correct or performant. |
| DS offset/address proof | `LDSAddr` / `decompose_lds_index` and safe DS offset folding exist with tests. | Safe fold is shape-dependent; some cases improve and some slow down, so search must choose. |
| DBUF trace/probes | Tests distinguish current-slot consumption from true future-slot staging; D3 target is explicitly marked as not complete. | Full two-slot cadence and lifetime split are missing for the generated route. |

Done when:

- Each primitive has owner files, tests/gates, and known gaps.

### P2. Pipe-First Primitive

Goal: extract the simpler `build_gemm_pipe` lifecycle as compiler/search primitives without making a complete hand-kernel
macro.

Hand oracle sequence:

```text
load F0 for k=0
for k pairs:
  load F1
  targeted wait for F0
  WMMA F0
  load next F0
  targeted wait for F1
  WMMA F1
store accumulators
```

Generate as primitives:

- `wmma_fragment_load_b128`
- `wmma_pipe_2stage_schedule`
- `targeted_vmcnt_wait`
- `wmma_accumulate`
- generated epilogue store

Do not generate as:

- one route-local `emit_pipe_kernel(...)` that emits the full GEMM instruction list.

First role:

```text
attn_qo: M=512, N=4096, K=4096
```

Done when:

- One pipe role is route-bound without raw `Ops.INS` full-kernel injection.
- It is numerically correct.
- Instruction/wait counts are compared to `build_gemm_pipe`.

Candidate API sketch:

```python
@dataclass(frozen=True)
class WMMAPipeSpec:
  m: int
  n: int
  k: int
  tile_m: int        # TM * 16
  tile_n: int        # TN * 16
  k_step: int = 16
  stages: int = 2
  operand_a: str = "global_row_major_fp16"
  operand_b: str = "global_row_major_bt_fp16"
  wait_policy: str = "targeted_vmcnt"
```

Compiler-visible metadata required:

- fragment stage identity: `stage=0/1`, `operand=A/B`, `tile_m/tile_n`,
- accumulator identity: stable accumulator per output subtile,
- wait policy: targeted `vmcnt(LPB)` where `LPB = TM*2 + TN*2`,
- WAR guard: reload of a fragment bank cannot precede WMMA consumption of that bank.

Minimum tests:

- pipe spec build must not call `build_gemm_pipe` when the generated path is selected,
- rendered stream has `global_load_b128`, `v_wmma`, targeted `s_waitcnt`, and no route-local raw `Ops.INS` list,
- small `K=64` structural order alternates load/consume/reload/consume,
- targeted `vmcnt` leaves future-stage loads outstanding,
- A/B fragment banks are disjoint and accumulator ranges are stable,
- route purity stays impure until the actual runtime pipe path stops wrapping `extra/qk/prefill/wmma.py`.

Task B first extraction seam:

- `extra/qk/wmma_pipe_spec.py::WMMAPipeSpec` is the narrow compiler-primitive contract for the register-resident
  two-stage pipe.
- `extra/qk/wmma_pipe_spec.py::extract_wmma_pipe_spec` accepts only resolved `PrefillGEMMScheduleSpec` values with
  `route_family == "pipe"`, `pipeline_depth == 2`, and `waitcnt_policy == "targeted_vmcnt"`.
- The minimal generated-lowering insertion point is `extra/qk/prefill_schedule_spec.py::emit_prefill_gemm_from_spec`:
  divert only the pipe branch to a backend-owned WMMAPipeSpec lowerer, leaving the LDS branch on `_emit_schedule`.
- Do not copy `extra/qk/prefill/wmma.py::build_gemm_pipe` instruction lists into the route. The current raw lowering
  remains `emit_prefill_gemm_from_spec -> _emit_schedule -> build_gemm_pipe -> UOp(Ops.INS, ...)` until the backend
  pipe primitive can emit through Tinygrad IR/backend primitives.
- Blockers before runtime diversion: generated fragment bank lifetime, WAR guard for reload-before-consume, targeted
  vmcnt that leaves future-stage loads outstanding, and generated epilogue stores.

### P3. Pipe Role Search

Goal: let machine search/spec select pipe primitive parameters.

Search knobs:

- `pipe_tm`
- `pipe_tn`
- `tile_m`
- `tile_n`
- `tile_k`
- wait policy
- epilogue policy

First exposed candidate space:

| Knob | First values | Gate notes |
|---|---|---|
| `route_family` | `pipe` only for `attn_qo`, `attn_kv`, `ffn_down`; `lds` remains forced for `ffn_gate_up` | Do not let search promote pipe for `out_f == 12288` until a role-specific authority run beats the LDS oracle. |
| `pipe_tm` | `2`, then `4` only after register/lifecycle proof | `2` is the current oracle-equivalent seed. `4` must prove no spills and stable fragment-bank ownership. |
| `pipe_tn` | `2`, then `4` only after register/lifecycle proof | Same rule as `pipe_tm`; do not expose `4x4` as an active gfx1100 promotion candidate while the generated 4x4 path is parked. |
| `tile_m` | `pipe_tm * 16` logical primitive tile, current route tile remains `128` | `WMMAPipeSpec` owns the primitive tile; `PrefillGEMMScheduleSpec.tile_m` keeps the route/block tile. Candidate records must store both when they differ. |
| `tile_n` | `pipe_tn * 16` logical primitive tile, current route tile remains `128` | Same storage rule as `tile_m`. |
| `tile_k` / `k_step` | `32` route tile, `16` WMMA step | Search may vary `tile_k` only after the structural order gate proves the two-stage load/consume/reload cadence for the new K grouping. |
| `wait_policy` | `targeted_vmcnt` only | Promotion requires non-full `s_waitcnt vmcnt(n)` where `n == pipe_tm*2 + pipe_tn*2`; `full_vmcnt` is diagnostic only. |
| `epilogue_policy` | `fp32_accum_to_fp16_store` | Keep fixed until pipe scheduling is route-bound and correct; epilogue tuning is not the first search axis. |

Do not expose first:

- `dbuf`, `pad`, `plra`, `plrab`, `leanaddr`, `waves_m`, `waves_n`, `wm`, `wn`, and `threads` for pipe promotion.
  They are carried by `PrefillGEMMScheduleSpec` because the resolver is shared, but the register-resident pipe primitive
  does not consume the LDS fields. Keep these fields in candidate artifacts as context, not active pipe knobs.
- `route_family=lds` search for `ffn_gate_up`; that belongs to P4/P5 and the existing DBUF/resource search gate.

Gate thresholds:

| Gate | Threshold |
|---|---|
| Candidate legality | Candidate must serialize from `PrefillGEMMScheduleSpec` into a pipe primitive spec without calling `extra/qk/prefill/wmma.py::build_gemm_pipe` on the generated path. |
| Correctness | Same numeric threshold as `prefill_v2_schedule_search.py`: finite `rel_rmse <= 2e-2` against fp32 numpy reference for the isolated `M=512` matmul shape. |
| Table application | Warm-start/spec application count must be nonzero; no `no-apply` result is promotable. |
| Resource safety | No spills/scratch and LDS footprint under `65536` bytes when LDS is present. Pipe candidates should report `0` LDS bytes. |
| Lifecycle | Rendered stream must contain `global_load_b128`, `v_wmma`, targeted `s_waitcnt`, stable accumulator ranges, disjoint A/B fragment banks, and the two-stage order `load F0 -> load F1 -> wait/consume F0 -> reload F0 -> wait/consume F1`. |
| Wait quality | Targeted waits must leave the future stage outstanding: for the first seed, `pipe_tm=2, pipe_tn=2` implies `vmcnt(8)`, not a full drain. |
| Route binding | Strict route attribution must show the selected role used the generated pipe primitive and no hidden fallback. |
| Throughput promotion | Per-shape candidate TFLOPS must be at least the frozen table default for that shape and at least `0.95x` the current pipe oracle for that role before whole-prefill promotion. |
| Whole-prefill promotion | Whole-prefill authority must pass on `512,1024,2048,4096` contexts with no correctness regression and worst context at least `0.98x` the current role-selective oracle before replacing the oracle/default. |
| Strict purity | Manifest provenance can move to `machine_authored_generated` only after the executing path has no `Ops.INS`, source-string, precompiled-binary, or route-local hand UOp full-kernel body. Until then the pipe route remains `external_handwritten_kernel` even if the schedule is search-selected. |

Candidate/result storage:

- Frozen warm-start scheduler results stay in `extra/qk/prefill_v2_schedule_table.json`; this remains the table consumed by
  `_build_prefill_v2_warmstart` and validated by `prefill_v2_schedule_table_gate.py`.
- Schedule-table gate artifacts stay in `bench/prefill-v2-schedule-table/latest.json`; resource-search candidates belong
  under the existing `resource_search` object rather than a second artifact family.
- Pipe primitive candidate ledgers should be separate from the frozen warm-start table until promoted:
  `bench/prefill-pipe-spec-search/latest.json` for the latest run and timestamped siblings for durable comparisons.
  Each row must include `route_id`, `role`, `shape`, the full `PrefillGEMMScheduleSpec.to_json()`, the extracted
  pipe primitive spec (`WMMAPipeSpec.to_json()` or equivalent), correctness/resource/lifecycle fields, route attribution,
  oracle TFLOPS, candidate TFLOPS, and whole-prefill linkage when available.
- Route/search-space manifests in `bench/qk-search-spaces/` should store only stable route state and profile context.
  Do not write transient candidate sweeps there; update `default_route_manifest.json` only when a gate changes route
  status/provenance or adds a durable promotion/refutation artifact.

Roles:

- `attn_qo`
- `attn_kv`
- `ffn_down`

Done when:

- Search/spec can choose the primitive composition per role.
- At least one generated pipe role moves materially toward oracle throughput.

### P4. Basic LDS Staging Primitive

Goal: cover the non-pipe lifecycle without DBUF first.

Generated primitive sequence:

```text
global b128 A/B tile loads
targeted vmcnt
ds_store A/B into LDS
targeted lgkm
barrier
ds_load A/B fragments
targeted lgkm
WMMA
barrier
epilogue
```

First role:

```text
ffn_gate_up: M=512, N=12288, K=4096
```

Done when:

- Single-buffer LDS staging is correct and route-bound for a bounded shape.
- No raw `Ops.INS` full-kernel injection.

### P5. DBUF And Address Lifetime

Goal: only after P4 works, add LDS buffer ping-pong.

Search/primitive knobs:

- `dbuf=0/1`
- LDS slot identity
- producer/consumer epoch
- DS immediate offset folding
- address rematerialization
- barrier/wait placement

Do not proceed unless:

- Slot/phase/epoch proof exists or the route fails closed to single-buffer.
- Register pressure stays below gfx1100 limits.
- The generated trace distinguishes current-slot `ds_load_b128` from true future-slot staging.
- Structural overlap is paired with numeric correctness and same-clock TFLOPS gates.

Done when:

- DBUF improves or holds throughput without MMU/runtime faults.
- Trace shows bounded address carriers and no unproven cross-slot reuse.

Keep hand ASM for now:

- full `build_gemm_lds2` DBUF cadence,
- slot cadence plus barrier/wait placement as one fused pipeline,
- both-operand staged 4x4 / large-tile route,
- Q4_K fused decode -> LDS -> WMMA.

Promote later as compiler primitives:

- safe DS address proof/folding, guarded by shape/search,
- wide LDS lowerers,
- packed staging matchers,
- lifecycle trace/probe gates.

### P6. Replace Or Keep Oracle

Decision:

- If generated primitive composition reaches acceptable throughput, replace the oracle role by role.
- If it does not, keep `PREFILL_GRAPH_GEMM=1` as an explicit escape hatch and keep generated path as default/purity route.

Acceptable outcomes:

| Outcome | Meaning |
|---|---|
| Generated reaches oracle class | Promote generated backend route; retire hand oracle for that role. |
| Generated reaches useful partial win | Keep both; generated default for purity, hand route for performance. |
| Generated remains far behind | Keep hand oracle and continue primitive extraction only where reusable. |

## Parallel Work Items

| Worker | Scope | Output |
|---|---|---|
| A. Lifecycle inventory | `wmma.py`, `prefill_graph_gemm_route.py`, `prefill_schedule_spec.py` | role/shape/builder/parameter table |
| B. Primitive substrate audit | AMD ISA WMMA tests, waitcnt, b128, LDS, DBUF docs | existing primitives and gaps |
| C. Pipe-first design | `build_gemm_pipe` only | minimal primitive API and test plan |
| D. LDS/DBUF boundary | `build_gemm_lds2` + DBUF/address docs | keep-hand vs generate-later ranking |

## Execution Orchestration

The work should run in parallel only up to the point where results have independent write scopes. The generated pipe
path itself is sequential: baseline first, smoke lowering second, role expansion third, search fourth.

| Lane | Can run now? | Owns | Must not touch | Required output |
|---|---:|---|---|---|
| Baseline/measurement | yes | benchmark commands, result locations, oracle-vs-generated comparison notes | compiler lowering | repeatable command set for the 5k oracle and current generated candidate |
| Pipe primitive extraction | yes | `WMMAPipeSpec` API target, insertion point, minimal skeleton if safe | LDS/DBUF route behavior | one-shape `attn_qo` lowering plan that does not copy the full hand instruction stream |
| Trace/smoke gates | yes | existing lifecycle/audit/probe metadata and tests | benchmark policy | structural gate for b128 loads, WMMA, targeted waitcnt, and no route-local full raw kernel |
| Search/promotion gates | yes | schedule/search knobs, result storage, promotion thresholds | primitive implementation | first search space and role promotion criteria |
| Role expansion | no | `attn_kv`, `ffn_down` generated pipe routes | `ffn_gate_up` LDS oracle | starts only after `attn_qo` smoke is correct and measurable |
| LDS/DBUF extraction | no | DS address proof, wide LDS lowerers, DBUF slot semantics | pipe-first critical path | starts only after pipe route either succeeds or is proven insufficient |

Sequential gates:

1. **G0 baseline pinned**: same-clock oracle and current generated numbers are reproducible, with artifact paths recorded.
2. **G1 pipe smoke compiles**: a generated/spec-owned pipe candidate renders without calling the full `build_gemm_pipe`
   raw instruction-list route.
3. **G2 pipe smoke is correct**: `attn_qo` numerics pass against the reference path.
4. **G3 pipe smoke moves**: instruction/wait counts and TFLOPS move toward the oracle. If not, stop and diagnose before
   role expansion.
5. **G4 role expansion passes**: `attn_qo`, `attn_kv`, and `ffn_down` have generated pipe candidates with correctness and
   throughput records.
6. **G5 search owns selection**: search/spec chooses pipe parameters and stores candidate results rather than hard-coding
   role-local instruction streams.
7. **G6 promotion decision**: each role is promoted, kept dual-route, or left on the hand oracle according to the P6 table.

Merge rules:

- Docs and audit metadata may merge immediately if they improve classification or repeatability.
- Code changes that affect lowering must land behind an opt-in env flag until G2 passes.
- Any candidate that uses `Ops.INS` / raw binary injection for the full GEMM lifecycle remains an oracle, not a generated
  replacement.
- A performance win is not promotable unless the route also passes provenance and correctness gates.
- A pure/generated route that is slow is still useful as a replacement substrate; a fast route-local raw hand kernel is only
  an oracle or explicit escape hatch.

## Guardrails

- Do not introduce a new route-local full GEMM instruction emitter.
- Do not call generated schedule selection over `wmma.py` pure.
- Do not delete 2x2/LDS/DBUF work; that is the replacement path.
- Do not block the 5k oracle on generated replacement work.
- Treat backend ASM as a tool, not as evidence of a hand kernel.
