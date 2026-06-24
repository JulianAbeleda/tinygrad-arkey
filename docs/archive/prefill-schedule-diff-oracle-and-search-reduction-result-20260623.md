# Prefill Schedule-Diff Oracle & Search Reduction — Result (2026-06-23)

## Verdict: `PREFILL_HAND_ASM_SCHEDULING_REQUIRED` (machine search does NOT reopen for prefill speed)
Reduced "Tensile wins by ~4–5%" into **one named, measured schedule primitive** — **K-loop software pipelining /
prefetch interleave** — and confirmed it is **not a bounded search knob** (the config knobs are exhausted; the full
pipeline needs a register pool = hand-asm/renderer). All other candidate primitives are ruled out or matched. No
default flip, no vendored-Tensile promotion, no tile-config search repeat.

## 1. Is the ~4–5% gap real under authority? — YES
Clock-pinned (`rocm-smi --setperflevel high`), 3-round interleaved synced whole-prefill: gap to Tensile = **+4.5–5.9%
@ctx512, +4.1–5.2% @ctx1024** (stable). `PREFILL_SCHEDULE_DIFF_AUTHORITY_LOCKED`.

## 2. Which role(s)? — ffn_down (4096×12288) and qo_proj (4096×4096)
Both below-parity and **well-occupied** (~128 workgroups — not WG-starved like the fixed kv_proj). gate_up is at
parity; kv_proj fixed.

## 3. Exact schedule-primitive diff (static ISA + dynamic PMC)
**Static** (steady K-loop region): `build_gemm_lds2` (down) is **PHASED** — all **0/8 global loads happen *before* the
WMMA region**, separated by 2 `s_barrier` (load → ds_store → barrier → wmma+ds_load → barrier). Tensile (13810 v_wmma;
densest segment) is **PIPELINED** — **3/4 global loads + 76/76 ds-loads are *inside* the wmma span** (next-K-tile
loads interleaved with the current WMMA group; PGR1/PLR1 in the `.dat`). **This is the residual primitive.**

**Dynamic** (reused PMC-hard-audit + LEANADDR, build_gemm_lds2-specific):
| primitive | status |
|---|---|
| VALU address arithmetic (+23%) | **ruled out** — LEANADDR matched it to Tensile, throughput **neutral** |
| occupancy / waves | **ruled out** — well-occupied; *more* occupancy (DBUF) **regresses** (L2-contention interior optimum) |
| LDS staging | **matched** — build_gemm_lds2 uses LDS (not the old tinygrad-WMMA LDS=0/DRAM-bound case) |
| L2 / DRAM traffic | **ruled out** — ours ≥ Tensile L2 hit |
| **K-loop software pipeline** | **ACTIVE residual** — Tensile overlaps loads with compute; build_gemm_lds2 phases |
| **VGPR register pool** | **ACTIVE enabler** — Tensile uses a register *pool* (dynamic VGPR lifetime) to afford full A+B prefetch; build_gemm_lds2 static-allocates and hits the 256-VGPR wall (PLRAB unbuildable) |

## 4–5. Critical path / ruled out
On the critical path: **software pipelining** + its enabler **register pool**. Ruled out (measurable but
non-transferring): VALU, occupancy, LDS-presence, L2. (The famous "tinygrad WMMA re-streams from DRAM, 6.5× stalls"
PMC finding was the **old** native-WMMA kernel — *not* build_gemm_lds2, which already stages through LDS.)

## 6. Searchable now? — NONE
The pipeline knobs in `build_gemm_lds2` are **exhausted**: `PLRA` (A-prefetch) is **already on** (= the +9–11% lever);
`DBUF` (double-buffer) **regresses** in-model (more LDS → lower occupancy → L2 contention); `PLRAB` (full A+B prefetch)
is **VGPR-blocked** (256-VGPR static wall). No remaining bounded knob recovers the gap. `SEARCHABLE_NOW = []`.

## 7. Requires hand-asm / renderer
Closing the gap means giving `build_gemm_lds2` a **register pool + true software-pipelined K-loop** (interleave the
next tile's loads with the current WMMA group across the whole steady region) — a structural hand-asm rewrite, or a
native renderer that emits software-pipelined GEMM. Both are deterministic engineering, **not machine search**.

## 8. Expected upside if solved — ~4–5% whole-prefill
Bounded above by the Tensile gap. (Whole-prefill, not isolated GEMM.)

## 9. Does this reopen prefill machine search? — NO
`PREFILL_HAND_ASM_SCHEDULING_REQUIRED`. Prefill speed search stays gated. Only **codegen-microprimitive learning**
rows are allowed (a native software-pipelining lowering would be such a capability — a long-term renderer target,
ISA-evidenced, no W==D promotion claim).

## 10. Next executable task
Either (a) **dependency policy**: accept vendored Tensile for prefill (declined so far), or (b) **renderer capability**:
software-pipelined GEMM lowering (a codegen-microprimitive target alongside v_dot2/cross-lane), or (c) **leave prefill
at ~96% of Tensile / at-above llama** and spend effort elsewhere (cross-shape, decode). No tile-config or occupancy
search — that surface is closed.

## Files changed
New: this doc + 6 artifacts under `bench/qk-prefill-schedule-diff-oracle/` (authority, kernel_pair_manifest,
static_isa_diff, dynamic_counter_diff, primitive_reduction, search_surface_decision) + 1 ledger entry. **No
`tinygrad/` source, no default flip, no vendored-Tensile promotion, no hand-asm kernel implemented (the scope
forbids it until a primitive is named + owner authorizes — it is now named; implementation awaits authorization).**

## Git status
Clean before; adds 1 doc + 6 artifacts + 1 ledger line. Defaults unchanged.
