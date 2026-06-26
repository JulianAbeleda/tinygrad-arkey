# Codex task prompt — build the codegen scheduling capability (the long-term machine-search enabler)

Copy below the line into Codex. Full rationale: `docs/decode-codegen-scheduler-capability-scope.md`.
This is a committed long-term capability build, NOT a cheap probe. Do not hand-restructure the attention
kernel for speed (that produces hand-owned code, off-principle). Build the generic codegen capability.

---

Repo `/home/ubuntu/tinygrad-arkey` (AMD gfx1100; hardware present, `DEV=AMD JIT=1 PYTHONPATH=.`).

## Goal

tinygrad's codegen has no instruction scheduler / software pipeliner. The generated decode block tile has
the owned algorithm and structure, and comgr already pipelines its global loads, but the per-token
cross-lane reduce + online-softmax recurrence is emitted as a serial dependency chain with no independent
work overlapping it, so its latency is fully exposed → the generated tile is **131–238× slower than owned**
in isolation. This is the SAME wall as prefill GEMM ("needs an asm scheduler"). Build the generic codegen
scheduling/software-pipelining capability that lets *generated* kernels hide this latency — the capability
that retires the last two hand-written kernels and lets machine search produce fast kernels. Everything
default-off, correctness-first, shipped default route + q4k GEMVs byte-for-byte unchanged.

## PHASE 1 — expose the wall (build the authoritative diff first)

New gate `extra/qk_decode_hotloop_schedule_diff.py` (reuse `_disasm`/`_hist`/`_parse_desc` from
`extra/qk_decode_attention_fused_score_state_pv_attribution.py`; markers from
`extra/qk_decode_attention_isa_diff_gate.py`). Disassemble `owned_flash_tile_gqa_whole` and
`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`, isolate the hot loop (backward-branch body), and
report per kernel: loop-body instruction count; `s_waitcnt lgkmcnt(0)`/`vmcnt(0)` INSIDE the loop;
issue-distance from each `ds_bpermute`/load to its first consumer (exposed vs overlapped latency); whether
next-iteration independent work is interleaved into the current reduce-wait. Add a `PROFILE=1`/sqtt
(`extra/sqtt/roc.py`) dynamic confirmation attributing cycles to LDS/memory stall vs VALU busy.
Verdict `HOTLOOP_SCHEDULE_DIFF__SCHEDULING_BOUND` vs `__INCONCLUSIVE`. Run:
`DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_hotloop_schedule_diff.py`. This is the baseline the
capability must beat and the machine-search-blocker artifact.

## PHASE 2 — build the generic scheduling capability

### Arm A (primary, the long-term codegen layer): software-pipelining pass
Build a real loop-software-pipelining pass that restructures a recurrence/reduction loop to expose
cross-iteration ILP: a prologue issuing iteration N+1's independent work (loads, q·k dot, the reduce that
does not depend on the running state) overlapping iteration N's serial online-softmax merge. Target
`tinygrad/codegen/late/linearizer.py` (currently a topo-sort with no scheduling). Hook it env-gated like
`V_DOT2_LOWERING` (`tinygrad/codegen/__init__.py:112-114`) and add the flag to the cache key (`:255`);
model the module structure on `extra/qk_fdot2_lowering.py`. Design for GENERALITY: parameterize pipeline
depth and the independent-vs-serial partition so it applies to any reduction loop and can later become a
searched decision — do not special-case the decode tile.

### Arm B (deeper, build where comgr is insufficient): tinygrad's own Inst scheduler
`extra/qk_asm_scheduler.py` builds a register def/use DAG over `list[Inst]` and can reorder fence-delimited
regions; it is dormant. Mature it into a real scheduler and wire it on the
`Ops.INS → Ops.LINEAR → assemble_linear` path (`tinygrad/renderer/amd/elf.py`) so tinygrad emits scheduled
ISA itself, independent of comgr. This is the durable foundation that also resolves prefill GEMM. Build Arm
A first; turn to Arm B if comgr re-schedules away Arm A's ordering (record that finding).

### Validation (in order; capability must be real AND generic)
1. Isolated per-kernel timing (Phase-1 method): generated tile drops materially from 131–238×.
2. `extra/qk_decode_attention_block_tile_microgate.py` → `BLOCK_TILE_MICROGATE_PASS` (numeric unchanged).
3. Route gate clean; `extra/qk_decode_isa_vectorization_gate.py` → `ISA_VEC_AUTHORITATIVE_PASS`.
4. `extra/qk_decode_hotloop_schedule_diff.py`: exposed-latency count drops toward owned.
5. W==D toward baseline 82.4/103.5/101.8/94.6.
6. **Generality proof (required):** the same pass moves the prefill-GEMM hot loop — the test that this is a
   capability, not a kernel hack.

## PHASE 3 — make scheduling a searchable codegen decision

Lift pipeline depth / partition into BubbleBeam/FutureSight so the machine selects and applies latency-hiding
across ops; decode attention + prefill GEMM are the anchor cases the search must reproduce.

## Commands

```
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_hotloop_schedule_diff.py
DEV=AMD JIT=1 DECODE_ATTN_BLOCK_TILE=1 <SCHED_FLAG>=1 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py
DEV=AMD JIT=1 DECODE_ATTN_BLOCK_TILE=1 <SCHED_FLAG>=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py
DEV=AMD JIT=1 DECODE_ATTN_BLOCK_TILE=1 <SCHED_FLAG>=1 PYTHONPATH=. python3 extra/qk_decode_isa_vectorization_gate.py
DEV=AMD JIT=1 DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 DECODE_ATTN_BLOCK_TILE=1 <SCHED_FLAG>=1 V_DOT2_LOWERING=1 \
  PYTHONPATH=. python3 extra/qk_decode_runtime_overhead.py
# isolated timing gen vs owned at ctx 512 and 4096 (eager custom_kernel + DEBUG=2)
```

## Labels / terminal (the only abort is correctness/regression, never cost)

- `SEARCH_PROGRESS__CODEGEN_SCHEDULER` — capability moves isolated timing + W==D; continue to generality
  proof + Phase 3.
- `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING` (confirmed deep) — only if a properly-built scheduler still can't
  approach hipcc pipelining; conclusion is "keep building the named enabler," not "give up."
- `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULER_NOT_WIRABLE` — neither arm hooks without a default regression.

## Constraints

Default-off (env-gated + cache key); shipped default + GEMVs unchanged; correctness-first; do not edit
`tinygrad/runtime/autogen/**`; do not hand-restructure the attention kernel for speed; do not add another
attention layout. Do NOT claim a step worked unless an ISA/schedule marker moved AND W==D moved.
Bracketed-prefix commits with gate verdicts. Report isolated timing, schedule-diff, ISA markers, and W==D
before/after at each milestone.
