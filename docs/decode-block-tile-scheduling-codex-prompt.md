# Codex task prompt — close the block-tile decode scheduling/dataflow gap

Copy below the line into Codex. Full analysis: `docs/decode-block-tile-scheduling-resolution-plan.md`.

---

You are in the tinygrad fork `/home/ubuntu/tinygrad-arkey` (AMD gfx1100; hardware present, run real jobs
`DEV=AMD JIT=1 PYTHONPATH=.`). The generated block-tiled decode tile is structurally correct (4 warps,
TK=16, 8 KB LDS, wide loads, fdot2, clean route, microgate PASS) but W==D is still far below owned and the
gap GROWS with ctx: 82.4 / 19.0 / 11.8 / 3.5 tok/s @ ctx 128/512/1024/4096 vs baseline 82.4 / 103.5 / 101.8
/ 94.6. Do NOT add another attention layout, do NOT revive score-broadcast, keep everything default-off and
correctness-gated. Do NOT claim "machine search solved decode" — it has not.

## Root cause (already localized — verify, don't re-derive)

Owned-vs-generated ISA: static counts ~equal (557 vs 608); the generated global loads ARE pipelined
(staggered `s_waitcnt vmcnt(30…8)`). So this is NOT a naive load stall. Two real causes:
1. **DATAFLOW (dominant):** the generated tile runs `_warp_reduce_sum_staged(partial, lane, 32)` PER TOKEN
   inside the serial online-softmax recurrence (`extra/qk_flash_decode.py` `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`,
   ~line 956 for the reduce; `acc.after(tt)`/`mx.after(tt)` recurrence ~lines 965-967). That is a serial
   cross-lane LDS ladder (`ds_bpermute` 10 vs owned 5) on a true dependency chain — latency that scheduling
   cannot hide. Owned (`extra/qk_owned_flash_decode.hip:225` `owned_flash_tile_gqa_whole`) has cross_lane=5,
   i.e. it does NOT reduce per token — it token-shards the q·k.
2. **OCCUPANCY (cheap, separate):** the route sets `l_route = max(1, _ceildiv(MAXC, target_s))` with
   MAXC=4608 (`extra/qk_flash_decode.py:1312`), so the split count is derived from MAXC not the running ctx
   → at ctx512 `s_route=ceil(512/96)=6` → 8·6=48 workgroups → 0.5 wg/CU (STARVED); at ctx4096 `s_route=43`
   → 3.58 wg/CU (~matched). So ctx512 is partly occupancy-starved; ctx4096's 27× gap is at matched occupancy.

## STEP 0 — measure first (decisive, do before any change)

Isolated per-kernel GPU timing of the generated block tile vs `owned_flash_tile_gqa_whole` at **ctx512 and
ctx4096**, using the eager `custom_kernel` + DEBUG=2 method in `docs/decode-block-tile-codegen-result.md`
Part A (the JIT path reports 0; eager prints per-kernel `us`/`ms`). Report a small table: gen_us, owned_us,
ratio, at each ctx. Purpose: confirm the gap is a constant-ish throughput factor at MATCHED occupancy
(ctx4096), not super-linear. This number gates whether STEP 2 (the rewrite) is worth it. Commit STEP 0 as a
result doc.

## STEP 1 — H2 occupancy fix (cheap, do next)

In the block-tile route branch (`extra/qk_flash_decode.py` ~1310-1316), make the split count track the
actual ctx instead of MAXC: prefer a **concrete `S=48` grid with a symbolic per-split `L=ceildiv(Tc_u,48)`**
(do NOT Python-eval the symbolic `Tc_u` — keep `L` derived from a concrete target and `S` symbolic only if
needed; mirror how the other routes keep `L` concrete and `S=(Tc_u+L-1)//L`). Default-off (behind the
existing block-tile flag). Validate: route gate clean, ISA-vec gate `ISA_VEC_AUTHORITATIVE_PASS`, and W==D
at ctx512 moves up from 19.0. This only helps short ctx; it does not address the ctx4096 dataflow gap.

## STEP 2 — H1 token-shard the q·k (the real lever; only if STEP 0 shows matched-occupancy gap)

Eliminate the per-token cross-lane reduce. New kernel variant
`flash_block_tiled_xlane_score_pv_tile_whole_cache_tokshard_kernel` behind a new sub-flag (e.g.
`DECODE_ATTN_BLOCK_TILE_TOKSHARD=1`), default-off. Design intent: **each lane computes a FULL Hd q·k dot for
its own token(s) from LDS-resident K — no cross-lane per token** — then the TK scores are reduced/merged
for the online softmax with at most one cross-lane per block (or a scalar serial pass over TK), and PV is
accumulated. Read `extra/qk_owned_flash_decode.hip:225` to see the exact lane layout owned uses to reach
cross_lane=5; mirror its q·k/score structure (lanes own tokens for the dot; the d-shard PV and the
combine stay as in the current tile). Validate STRICTLY in this order:
1. New microgate (clone `extra/qk_decode_attention_block_tile_microgate.py`) — numeric vs NumPy per-split
   oracle, fp16 `max_abs ≤ 2e-5` across Tc 32/128/130/256 → `BLOCK_TILE_MICROGATE_PASS`.
2. Route gate clean (`DECODE_ATTN_BLOCK_TILE=1` + the tokshard sub-flag) →
   `FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT`.
3. ISA-vec gate → `ISA_VEC_AUTHORITATIVE_PASS`, and `ds_bpermute` count drops from 10 toward owned's 5.
4. W==D@4096 moves materially up from 3.5.
If the tokshard layout cannot be expressed/verified, record `SEARCH_BLOCKED_BY_CODEGEN__TOKSHARD_NOT_EXPRESSED`
and stop — do not force it.

## STEP 3 — H3 predication cleanup (optional, if cndmask still high)

Replace the per-element `in_r.where(...)` masks with a clean per-block valid-token bound
(`min(TK, Tc - (s*L + b*TK))`). Re-diff: `v_cndmask` should fall from 15 toward owned's 4. Validate with the
microgate + ISA-vec gate.

## Gates / commands (run after every change; assert the pass-strings)

```
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py                       # BLOCK_TILE_MICROGATE_PASS
DEV=AMD JIT=1 DECODE_ATTN_BLOCK_TILE=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py  # ROUTE_CLEAN__ECONOMICS_NEXT
DEV=AMD JIT=1 DECODE_ATTN_BLOCK_TILE=1 PYTHONPATH=. python3 extra/qk_decode_isa_vectorization_gate.py       # ISA_VEC_AUTHORITATIVE_PASS; read ds_bpermute / cndmask / wg_per_cu
DEV=AMD JIT=1 DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 DECODE_ATTN_BLOCK_TILE=1 V_DOT2_LOWERING=1 \
  PYTHONPATH=. python3 extra/qk_decode_runtime_overhead.py    # vs 82.4/103.5/101.8/94.6
# STEP 0 isolated timing: eager custom_kernel + DEBUG=2, gen block tile vs owned, ctx 512 and 4096
```

## Stop condition / terminal labels

Conclude it is a **codegen-scheduler wall** (and STOP hand-iterating the kernel) when ALL hold: occupancy
matched (STEP 1), per-token cross-lane removed (STEP 2 tokshard), predication cleaned (STEP 3), yet the
isolated ctx4096 gen/owned time ratio is still ≫1 and the disasm residual is unhidden serial latency
(`s_waitcnt lgkmcnt/vmcnt(0)` on a true dependency chain comgr cannot pipeline). Then:
`SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING` is CONFIRMED — note it is the SAME wall as the prefill GEMM (needs a
tinygrad instruction scheduler; the dormant `extra/qk_asm_scheduler.py` / Track-B `Ops.INS` path is the only
resolution, a separate large project), and machine-search purity should be proven on the simpler generated
GEMV/lane-map routes first. Other labels: `SEARCH_BLOCKED_BY_CODEGEN__TOKSHARD_NOT_EXPRESSED` (STEP 2 can't
be built). If STEP 1+STEP 2 move W==D materially toward baseline, the wall was the per-token-cross-lane
dataflow, not the scheduler — keep going.

## Constraints

Default-off (env-gated + cache key for any codegen change); shipped default route + q4k GEMVs byte-for-byte
unchanged; correctness-first; microgate before model; do not edit `tinygrad/runtime/autogen/**`. Report W==D
and ISA markers (ds_bpermute, cndmask, wg_per_cu, LDS) before/after each step. Do NOT claim a step "worked"
unless the ISA marker moved AND W==D moved. Bracketed-prefix commits (e.g. `[nn]`, `[codegen]`) with the
gate verdicts in the message.

Deliverable: STEP 0 timing table; STEP 1 result (W==D@512 before/after); STEP 2 microgate + ISA + W==D@4096
before/after (or the TOKSHARD_NOT_EXPRESSED finding); and an explicit verdict on whether the scheduler stop
condition is met.
