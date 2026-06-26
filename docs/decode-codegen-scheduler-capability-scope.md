# Codegen scheduling capability: expose the wall, then bring machine search to it — scope (2026-06-26)

Principle: the goal is **machine search**, not a faster hand-written kernel. Hand-restructuring the
attention tile for ILP (Solution A) would produce more hand-owned code — which, per the project's own
lesson ("hand-owned code is not evidence the generic search space can express the primitive"), does NOT
advance machine search. The aligned move is to give **codegen** a generic latency-hiding/scheduling
capability so the *machine* can produce competitive kernels — for decode attention, prefill GEMM, and
future ops alike. This scope: (1) EXPOSE the scheduler wall authoritatively, then (2) build the minimal
capability that confronts it, with an honest terminal. Do not implement until asked.

## The wall, precisely (verified this campaign)

- Generated block tile = owned algorithm and structure (4 warps, TK=16, 8 KB LDS, fdot2, per-token
  cross-lane reduce — owned reduces per token too; static counts ~equal 557 vs 608).
- comgr ALREADY pipelines the global loads (staggered `s_waitcnt vmcnt(30…8)`), so load latency is hidden.
- Residual exposed latency = the **per-token cross-lane reduce (`ds_bpermute` ladder, `lgkmcnt`) on the
  serial online-softmax recurrence** (`acc.after(tt)`/`mx.after(tt)`). It is a true serial dependency chain
  with no independent work to overlap it; tinygrad's linearizer is a topological sort with **no instruction
  scheduler / software pipeliner**, so the chain is emitted in-order and every reduce/merge latency is fully
  exposed → isolated tile is **131–238× slower than owned** (whole-decode 5–27×, diluted by shared GEMVs).
- Occupancy is NOT the lever (forcing S=48 at ctx512 made it worse, 19→14.9 — more copies of a stalled loop
  don't hide its latency). The algorithm is NOT the issue (owned reduces per token too).
- This is the SAME wall as prefill GEMM (perf-state: "fine instruction-scheduling … needs an asm scheduler
  or vendored .co"). One missing capability gates both remaining hand-written kernels.

`SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING` is the correct, narrow blocker.

## Phase 1 — EXPOSE the wall (authoritative, do first; concrete and bounded)

Produce the artifact that makes "this is a scheduling wall" measured fact, not argument. New gate
`extra/qk_decode_hotloop_schedule_diff.py` (reuse the disasm helpers in
`extra/qk_decode_attention_fused_score_state_pv_attribution.py` and `extra/qk_amdgpu_isa_primitive_audit.py`):
- Disassemble `owned_flash_tile_gqa_whole` and `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`,
  identify the hot loop (the backward-branch body), and report per kernel: instructions in the loop body,
  count of `s_waitcnt lgkmcnt(0)`/`vmcnt(0)` **inside** the loop, the **issue-distance** from each
  `ds_bpermute`/load to its first consumer (latency exposed vs overlapped), and whether independent
  next-iteration work is interleaved into the current iteration's reduce-wait.
- If available, a dynamic confirmation: `PROFILE=1` (or sqtt via `extra/sqtt/roc.py`) on both kernels to
  attribute cycles to memory/LDS stall vs VALU busy.
Verdict: `HOTLOOP_SCHEDULE_DIFF__SCHEDULING_BOUND` (generated has exposed serial reduce/recurrence latency
owned overlaps) vs `__INCONCLUSIVE`. This is the machine-search-blocker artifact and the baseline the
capability must beat. Command:
`DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_hotloop_schedule_diff.py`.

## Phase 2 — build the real codegen scheduling capability (the long-term machine-search enabler)

This is the committed long-horizon investment, not a throwaway probe. The thing tinygrad is missing — and
the thing standing between *generated* kernels and competitive ISA for decode attention, prefill GEMM, and
every future latency-bound op — is a real **instruction scheduler / software pipeliner in codegen.** Build
it as a generic capability, designed for the general case (any recurrence/reduction loop, any generated
kernel, composing with the search), not a one-off decode-tile patch. Two architectural arms; build the one
that is the correct long-term foundation, not the quickest:

### Arm A — a generic software-pipelining pass in tinygrad codegen (the primary long-term architecture)
A real loop-software-pipelining pass that restructures a recurrence/reduction loop to expose cross-iteration
ILP — generically: a prologue that issues iteration N+1's independent work (loads, q·k dot, the reduce that
does not depend on the running state) while iteration N's serial merge (online-softmax `m/l/acc`) executes,
so memory/LDS/cross-lane latency overlaps independent work. Targets `tinygrad/codegen/late/linearizer.py`
(today a dependency topo-sort with no scheduling) — add a true pipelining/scheduling stage, env-gated + in
the `to_program` cache key (`tinygrad/codegen/__init__.py:255`), modeled structurally on the existing
opt-in lowerings (`extra/qk_fdot2_lowering.py`, `extra/qk_warp_reduce_lowering.py`). Build it to generalize:
parameterize the pipeline depth and the independent/serial partition so it applies to any reduction loop,
and so it can later become a searchable decision (Phase 3). If comgr re-schedules and undermines the
UOp-level ordering, that is itself a finding that forces Arm B — but the generic pass is the right
foundation and the first thing to build properly.

### Arm B — tinygrad's own instruction scheduler on the `Ops.INS` path (the deeper long-term independence)
The most durable long-term answer is tinygrad emitting **scheduled ISA itself**, independent of comgr's
quality. `extra/qk_asm_scheduler.py` already builds a register def/use DAG over `list[Inst]` and can reorder
fence-delimited regions; it is dormant. Mature it into a real scheduler and wire it on the
`Ops.INS → Ops.LINEAR → assemble_linear` path (`tinygrad/renderer/amd/elf.py`). This is the bigger
architectural commitment, but it is the capability the perf-state memory has already named as required for
prefill GEMM too — so it is one foundation that retires BOTH remaining hand-written kernels and makes the
search's output quality tinygrad's own, not comgr's. Build Arm A first as the generic codegen layer; mature
Arm B as the scheduler that backs it where comgr is insufficient.

### Validation (proves the capability is real and generic), in order
1. Isolated per-kernel timing (Phase-1 method): generated tile drops materially from the 131–238× baseline.
2. `extra/qk_decode_attention_block_tile_microgate.py` → `BLOCK_TILE_MICROGATE_PASS` (numeric unchanged).
3. Route gate clean; ISA-vec gate `ISA_VEC_AUTHORITATIVE_PASS`.
4. `extra/qk_decode_hotloop_schedule_diff.py`: the loop's exposed-latency count drops toward owned.
5. W==D moves materially toward baseline (82.4/103.5/101.8/94.6).
6. **Generality proof:** the SAME pass moves the prefill-GEMM hot loop (the second proving ground) — this is
   the long-term test that it is a capability, not a kernel hack.

## Phase 3 — make scheduling a first-class searchable codegen decision

Lift the pipeline depth / independent-vs-serial partition into BubbleBeam/FutureSight as a searched decision,
so the *machine* selects and applies latency-hiding across ops. This is the end state that "brings machine
search": once codegen can schedule and the search can drive it, the last two hand-written kernels are
retired and competitive kernels are generated, not authored. Treat decode attention + prefill GEMM as the
two anchor cases the search must reproduce.

## Stop condition / terminal labels

This is a long-term build; the only abort is correctness/regression, never cost.
- `SEARCH_PROGRESS__CODEGEN_SCHEDULER` — Arm A (or B) moves the isolated timing + W==D on the decode tile.
  Continue to the generality proof (prefill) and Phase 3.
- `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING` (confirmed deep) — only if a properly-built scheduler still cannot
  approach hipcc's pipelining on the recurrence. Even then the conclusion is "the scheduler is the named
  long-horizon enabler to keep building," not "give up"; machine search continues to be proven on the
  generated GEMV/lane-map routes in parallel, NOT abandoned.
- `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULER_NOT_WIRABLE` — only if neither arm can hook without a default-path
  regression (a wiring problem to solve, not a reason to drop the capability).

## Constraints

Default-off (env-gated + cache key); shipped default route + q4k GEMVs byte-for-byte unchanged;
correctness-first; do not edit `tinygrad/runtime/autogen/**`; do not hand-restructure the attention kernel
for speed (Solution A, off-principle — it produces hand-owned code, not a search capability); do not add
another attention layout. Bracketed-prefix commits. Codex prompt:
`docs/decode-codegen-scheduler-capability-codex-prompt.md`.

## Why this is the right long-term solution

Every cheaper path dead-ends: hand-tuning the attention kernel produces more hand-owned code (off-principle);
forcing occupancy made it worse; swapping the reduction form didn't move the ISA. The single capability that
generalizes — that turns *generated* kernels into competitive ones for decode attention, prefill GEMM, and
whatever comes next — is a real instruction scheduler / software pipeliner in codegen. It is a substantial
compiler investment, and that is exactly why it is worth doing properly rather than dodged: it is the thing
that makes machine search actually produce fast kernels instead of routing among hand-written ones. Build
the generic capability; let the decode tile and prefill GEMM be its first two proofs; then hand it to the
search.
