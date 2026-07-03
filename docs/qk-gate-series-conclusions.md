# QK gate/experiment series conclusions + retirement ledger

Durable record of what the `extra/qk` experiment series proved, and the ledger for
files retired when their series was collapsed into a parameterized `gate_registry`
module. **This file is the banked record**: `bench/**` is gitignored scratch, so a
bench JSON does not preserve a verdict across a clean checkout — a conclusion is only
"banked" once it is written here (tracked).

Provenance: as of 2026-07-03 the per-experiment result docs these scripts fed
(`docs/decode-attention-online-state-pv-tile-p{8..12}-*.md`, `-a3-2b-*.md`, the
per-series PALL/score-broadcast/tg-p9 result docs) were pruned; `pure-machine-search-roadmap.md`
still lists their filenames but the files are gone, and no per-family `bench/` artifact
dir survives. The conclusions below were transcribed from the script docstrings +
in-body verdict logic before the scripts were collapsed.

## Cross-cutting finding (the load-bearing one)

Two independent arcs — Cluster A's combine repro (TG-P10.1) and Cluster E's combine
microgate (TG-P9.4) — terminate at the **same AMD-backend compiler wall**, `EMITTER_BLOCKED`:
any decode-attention combine that shares softmax weights across `d` or fuses the gmax
max-reduce lowers the reduction-accumulator REG into a non-assignable vectorized
`make_float4(...) = ...` store; `REG_STORE_DEVEC=1` trades the compile error for NaN
(the max-reduce mis-lowers). Only the shipped single-reduce per-`d` combine compiles
correctly. **Reopen condition:** a tinygrad codegen fix that keeps the reduction
accumulator REG scalar, or a DEVEC path that lowers the max-reduce correctly. This is
a compiler-primitive gap, not a decode-attention design failure — the math primitives
are all proven correct (Cluster A P11–P14).

## Cluster A — online-state x-lane family (P8–P15 + TG-P10.1)

Arc: P9 (scalar tile vs NumPy, the base) → P8/P10 (x-lane state / final-output vs ref)
→ when the full x-lane route failed, decomposed into synthetic micro-proofs P11 (merge),
P12 (merge components), P13 (reducer matrix), P14 (recurrence matrix), all PASS →
localized the fault to route-level indexing/layout, not the primitives. P15 re-expressed
as a split-state pipeline (PASS). TG-P10.1 pins the terminal combine blocker.

- **P9** `ONLINE_STATE_PV_P9_NUMERIC_PASS` — scalar online-state+PV whole-cache tile matches NumPy flash ref across ragged ctx (Tc=128,130,32,256, L=64; tol score/m/l 2e-3, pv/out 5e-3).
- **P8** `ONLINE_STATE_PV_P8_NUMERIC_PASS` — x-lane tile == scalar tile (m/l/PV + out; tol m/l 1e-4, pv/out 2e-3). Hazard: raw state buffers legitimately hold NaN in inactive-Smax slots; gate on finite active-column errors + final output, never a blanket state-buffer NaN check.
- **P10** `ONLINE_STATE_PV_P10_XLANE_OUTPUT_PASS` — x-lane final output == NumPy ref and == scalar route (tol 5e-3).
- **P11** `ONLINE_STATE_PV_P11_MERGE_PASS` — staged cross-lane online-softmax merge (`warp_reduce_max` + `_warp_reduce_sum_staged`, 32 lanes) correct in isolation (err ≤ 2e-4) → fault is per-lane state generation, not merge.
- **P12** `ONLINE_STATE_PV_P12_COMPONENTS_PASS` — max/sum/den/LSE sub-components each correct (per-col tol 2e-4).
- **P13** `XLANE_REDUCER_MATRIX_PASS` — 8-arm stressor sweep; reducers + feature/column axes sound. Finding: a GLOBAL output-feature axis whose input depends on the feature, or a 2nd GLOBAL output column, can change reducer semantics / store placement; masked (zeroed-inactive-lane) sums may not be preserved.
- **P14** `XLANE_RECURRENCE_MATRIX_PASS` — per-lane multi-token online recurrence (R=3/lane, TMAX=96, ragged tails 96/70/33/1) reproduces softmax·V (tol 2e-4) → recurrence primitive sound.
- **P15** `SPLIT_XLANE_OUTPUT_PASS` — split-state pipeline (max → xlane PV-from-m → gmax → den → combine, W=Hd+1) matches ref/scalar/xlane and per-split NumPy ref (tol 5e-3).
- **TG-P10.1** `TG_P10_1_PASS_REG_REPRO_PINNED` — minimal generated-UOp repro of the combine blocker (emits `tinygrad.reg_scalar_lowering.v1`): control per-d combine compiles+correct; shared-weight combine and fused-gmax combine → `invalid_reg_vector_store` at compile; shared-weight under `REG_STORE_DEVEC=1` → compiles but NaN. `REG_STORE_DEVEC` is a memoized compile-time getenv → the DEVEC case must run in a fresh subprocess. Env: `DEV=AMD`, `REG_STORE_DEVEC=1`.

## Cluster B — prefill ASM instruction scheduler (Inc0–Inc3)

Strictly linear; operates on `build_gemm_lds2` streams (`prefill/wmma.py`) via `asm_scheduler.py`.
Correctness gate throughout rel_rmse ≤ 3e-4; timing informational, clock-pinned. Env `DEV=AMD`, `MNK` (default 512).

- **Inc0** `INC0 ALL_PASS` — IR + dependency-DAG lift faithful: P1 identity byte-identical, P2 operand/reg-range coverage, P3 DAG legality (orig order is valid topo sort), P4 layout preserved, P5 identity runs correct, P6 a dependency-respecting asap reorder still computes correctly (the strong test).
- **Inc1** — s_waitcnt model (vmcnt/lgkmcnt). Q1 hand waits already minimal (slack ≤ 2), Q2 recompute-in-place correct, Q3 identity wait-correct, Q4 gate discriminates (rejects drains-removed stream, simm16=0x3FF0), Q5 wait model composes with reorder. **Q6 `WAIT_CORRECTNESS_NECESSARY_NOT_SUFFICIENT` (key hazard):** a fence_only reorder that moves memory ops is register-legal AND wait-correct yet computes WRONG on hardware. Inc1 originally blamed an "RDNA3 spacing/scoreboard hazard" and kept cross-motion OFF — **overturned by Inc2.**
- **Inc2** `INC2 CORRECTNESS_PASS` — **refutes Inc1-Q6:** the miscompile was `build_regions` not modeling the backward-branch TARGET (loop entry); the reorder crossed the prologue/loop-body boundary. Adding branch-target boundaries makes fence_only cross-motion byte-identical-correct across 4 route configs × asap/critical. ISA finding: RDNA3 interlocks VALU/VMEM deps, so register-legal + wait-correct reorder can't corrupt via spacing (s_delay_alu is perf-only). Latency reorder is perf-neutral on the hand-tuned kernel.
- **Inc3** `INC3 CORRECTNESS_PASS -- first non-neutral result` — waitcnt RELOCATION (remove per-block lgkm(0) drain, insert minimal per-WMMA lgkmcnt to overlap WMMA with LDS-load tail; forces branch-offset recompute). S1 relocation correct across configs + NBLK 16..128, S2 non-mutating. **Numbers: DBUF1 ~+6%, PLRA ~+2%, kv_halved regresses** → config-dependent; the sole promotion candidate, gated on per-config validation + whole-prefill confirmation.

## Cluster C — physical-tile family (P1 → PALL route → lifecycle → scaling → all-primitives)

Env `DEV=AMD`, `V_DOT2_LOWERING=1`. ISA captured via runtime hook + llvm-objdump, flagged for v_dot2/lds/cross_lane/barrier/spill. Writes `bench/qk-decode-primitive-space/`.

- **P1 crosslane** `P1_CROSSLANE_PASS__LANEMAP_CROSSLANE_VISIBLE` (or `..._EXTRA_PRIMITIVES_PRESENT`) — generated UOp emits lane-sharded q·k + cross-lane ds_bpermute reduce, detected by primitive tooling (max_abs 1e-4, rmse 1e-5).
- **all-primitives** `PALL_PRIMITIVES_VISIBLE__ROUTE_INTEGRATION_NEXT` — all four missing primitive classes independently emit/detect-visible (bundles P1 + a3_1 vdot2 probe + minimal LDS probe). Note: a minimal same-lane LDS probe can legitimately elide the barrier → barrier tracked separately, not a failure.
- **PALL route** `PALL_ROUTE_BUILDER_READY__ROUTE_NEXT` — LDS-K-stage + lane-sharded q·k + cross-lane reduce + fdot2 compose in ONE generated score builder with correct numerics + required ISA (`Ops.CUSTOMI __builtin_amdgcn_fdot2`).
- **PALL lifecycle** `PALL_LIFECYCLE_BUILDER_READY__ROUTE_NEXT` — q·k score + online-softmax state + PV accumulation in one lifecycle kernel retaining LDS+crosslane+fdot2 (max_abs ≤ 1e-3, rel_rmse ≤ 1e-5, spill guard). **Known limit:** recomputes q·k per output column — generated axis ownership cannot reuse one lane-sharded score across the PV output-column axis.
- **lifecycle scaling probe** `PALL_LIFECYCLE_SCALING_CONFIRMS_COLUMN_RECOMPUTE` — runtime scales with PV output columns Wp∈{1,2,8,32,130} → the per-column q·k recompute (not route overhead) is the W==D timeout source. Hands off a single named next primitive: **score reuse across PV output columns.**

## Cluster D — score-broadcast family (the answer to Cluster C's gap)

Env `V_DOT2_LOWERING=1`, `DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE=1`. Numeric gate finite ∧ max_abs ≤ 1e-3 ∧ rel_rmse ≤ 1e-5. Writes `bench/qk-decode-primitive-space/`.

- **reuse-paths probe** `SCORE_REUSE_PATHS_PASS__BROADCAST_PROBE_READY` — Path A score-once state (expressible, no PV) + Path B score-broadcast PV columns (q·k once, updates several PV cols); Wp∈{1,8,32,128}, 32col/1col runtime multiple < 16 (sublinear). Path B is the needed primitive.
- **direct gate** `SCORE_BROADCAST_DIRECT_READY__MODEL_CAPTURE_NEXT` — route works end-to-end through shipped `flash_decode_attention_whole_cache` (eager/JIT/varJIT), compute in child subprocess.
- **chain gate** `SCORE_BROADCAST_CHAIN_READY__ROUTE_NEXT` — standalone chain (score_once_state → 4× score_broadcast_pv_cols @0/32/64/96 → combine4) numerically clean.
- **varjit chain** `SCORE_BROADCAST_VARJIT_CHAIN_READY__ROUTE_NEXT` — chain survives variable-bound TinyJit warmup→capture→replay, chunks∈{1,2,4}, finite in all phases.
- **model-cache-view gate** `SCORE_BROADCAST_MODEL_CACHE_VIEW_READY__ATTENTION_ONLY_NEXT` — route correct on the model's real `assigned_kv` cache view. Purpose: **refutes** the assigned_kv view as the full-model MMU root cause.
- **control matrix** `SCORE_BROADCAST_CONTROL_MATRIX_RECORDED` — diagnostic 2×2 of graph-barrier (`..._NO_GRAPH`) vs persistent scratch (`..._SCRATCH`), never W==D, never promotes. NOTE: now a stub — every case reports `failure_class="stale_replay_removed"` (the old JIT-phase child replay is not part of the compact repo surface); concrete pass/fail data did not persist.

## Cluster E — TG-P9 trio (live-context split geometry)

Env `DEV=AMD`; 8B geometry Hq=32,Hkv=8,Hd=128,MAXC=4608,S=36. Symbolic Tc via the carry trick. Writes `bench/tg-p9-pure-attention-primitive-route/`.

- **P9.1 live-split** `TG_P9_1_PASS_LIVE_TC_SPLIT_IR` — live-context split geometry (fixed S workgroups, symbolic per=ceildiv(Tc,S), nb=ceildiv(per,TK)) lowers; every token in [0,Tc) covered exactly once, no spill on [Tc,MAXC); grid=S fixed (not ceildiv(MAXC,L)).
- **P9.2 live-split tile** `TG_P9_2_PASS_LIVE_SPLIT_TILE` — live-split tile numerically == fixed-L=128 g5 tile (rel < 2e-2) AND reduces tile work at low ctx (DEBUG=2 per-kernel wall, live < 0.9× fixed). Guard outcome `TG_P9_2_REFUTE_LIVE_SPLIT_NO_MOVEMENT` if correct but no timing win.
- **P9.3/9.4 combine** `TG_P9_4_BLOCKED_EMITTER` — split-preserving LSE combine to remove the per-d `fexp` redundancy (the 556us/fwd ctx4096 cap) without collapsing Hq·S or Hq·Hd parallelism: three designs (LDS weight-share warp, inline-gmax single-kernel, two-stage fexp-free) all trip the same AMD codegen wall (see cross-cutting finding). `no_parallelism_collapse=true`; only shipped single-reduce per-d combine compiles. Reopen: tinygrad codegen fix keeping the accumulator REG scalar or DEVEC lowering the max-reduce correctly.

## Retirement ledger

Files deleted after their series was collapsed into a parameterized module + registry rows.
Format: `series | files retired | collapsed into | commit`. (Appended per cluster as collapses land.)

- **Cluster B (asm scheduler)** | `asm_scheduler_inc0_test.py`, `asm_scheduler_inc1_test.py`, `asm_scheduler_inc2_test.py`, `asm_scheduler_inc3_test.py` | `extra/qk/asm_scheduler_proofs.py` (VARIANTS + `build_inc0..3`) | commit 5a195c0e7
- **Cluster E (tg_p9 trio)** | `tg_p9_live_split_microgate.py`, `tg_p9_live_split_tile_microgate.py`, `tg_p9_combine_microgate.py` | `extra/qk/tg_p9_live_split.py` (`build_live_split/_tile/_combine`) | this commit. NOTE: combine gate no longer hits EMITTER_BLOCKED at retirement time — all 3 designs compile (`TG_P9_4_PASS_COMBINE_MICROGATE`); the cross-cutting wall may have moved (recheck vs Cluster A TG-P10.1).
