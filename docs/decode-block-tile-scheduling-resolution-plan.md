# Resolution plan: generated block-tile decode `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING` (2026-06-26)

The generated block tile now matches owned topology (4 warps, TK=16, 8 KB LDS, wide loads, fdot2, clean
route). W==D still 5–27× below owned and the gap GROWS with ctx (0.18× @512 → 0.04× @4096), while ctx128 is
at parity. This plan localizes the gap from the owned-vs-generated ISA + kernel structure, ranks the causes,
and gives the smallest next experiment + a stop condition. Do not implement until asked.

## 1. Owned vs generated differences (from disasm + kernel source)

| Dimension | Owned `owned_flash_tile_gqa_whole` | Generated block tile | Plausible impact |
|---|---:|---:|---|
| total static insts | 557 | 608 | ~equal → gap is dynamic, not volume |
| **ds_bpermute (cross-lane)** | **5** | **10** | **per-token warp_reduce on the serial critical path** |
| q·k reduction strategy | no per-token cross-lane (cross_lane=5 ⇒ token-sharded / amortized) | **e-shard + `_warp_reduce_sum_staged` per token `tt`** (kernel `:956-957`) | serial LDS ladder, can't be latency-hidden |
| online-softmax recurrence | — | `acc.after(tt)`, `mx.after(tt)` serialize tokens (`:965-967`) | next token waits on this token's reduce |
| **v_cndmask** | **4** | **15** | per-element `in_r.where(...)` masking vs clean loop bound |
| s_waitcnt | 23 | 37 | consequence of more LDS/cross-lane traffic (loads ARE pipelined — staggered `vmcnt(30…8)`) |
| global_load | 23 | 34 | more loads; check for reloads owned keeps in regs |
| VGPR / LDS / scratch | 64 / 8192 / 0 | 56 / 8192 / 0 | generated resource footprint is FINE (not the issue) |
| occupancy @ctx512 | ~4 wg/CU | **0.5 wg/CU (starved)** — `l_route=ceildiv(MAXC=4608,48)=96` from MAXC not ctx | ctx512 gap is PARTLY occupancy starvation |
| occupancy @ctx4096 | ~4 wg/CU | 3.58 wg/CU (~matched) | ctx4096 27× gap is NOT occupancy → the real dataflow wall |

Headline: comgr is already pipelining the global loads (staggered `s_waitcnt vmcnt`). The hot-loop cost is
the **per-token cross-lane reduce inside the serial online-softmax recurrence** — a true serial dependency
chain, plus per-element predication. This is a dataflow choice in the generated kernel, not a missing
scheduler pass.

## 2. Hypotheses, ranked by expected impact

- **H1 (dominant): per-token cross-lane reduce on the serial recurrence.** The e-shard q·k forces a
  `_warp_reduce_sum_staged` per token (`ds_bpermute` ladder, ~5 serial LDS-latency ops), and the
  online-softmax recurrence prevents overlapping tokens, so the latency is exposed. Owned avoids it
  (cross_lane=5). Explains the matched-occupancy ctx4096 gap.
- **H2 (cheap, separate): ctx512 occupancy starvation.** `l_route` is derived from `MAXC`, not the running
  ctx, so short ctx under-splits (0.5 wg/CU). Fixable independently; only affects ctx ≪ MAXC.
- **H3: per-element predication.** `v_cndmask` 15 vs 4 from `in_r.where` on t_safe/score/p applied every
  token; owned uses a clean per-block token count. Adds ALU + breaks some folding in the hot loop.
- **H4 (terminal): residual instruction-scheduling / latency-hiding.** If H1–H3 are addressed and a gap
  remains at matched occupancy, the residual is the same class as the prefill-GEMM wall (perf-state:
  "fine instruction-scheduling … needs an asm scheduler or vendored .co") — tinygrad's linearizer is a
  topo-sort with no scheduler, and comgr won't pipeline a serial recurrence.

## 3. Measurement per hypothesis (prove/refute)

- **H1:** isolated per-kernel timing (the DEBUG=2 eager-`custom_kernel` method, see
  `docs/decode-block-tile-codegen-result.md` Part A) of the generated tile vs owned at ctx4096 (matched
  occupancy). Then a microgate variant that **token-shards the q·k** (each lane computes a full token's dot,
  TK tokens processed in parallel across lanes, ONE reduce/merge per block instead of per token) — if the
  isolated tile time drops materially, H1 confirmed. Refute if removing per-token cross-lane doesn't move it.
- **H2:** set `l_route = ceildiv(Tc_u_concrete_per_ctx, 48)` (or a concrete `S=48` grid + symbolic `L`), re-run
  W==D@512 → expect 19 → materially higher; ISA-vec/route gates stay green. Refute if @512 doesn't move.
- **H3:** replace `in_r.where` element masks with a clean per-block valid-token count (loop bound
  `min(TK, Tc - (s*L + b*TK))`); re-diff `v_cndmask` (expect → ~owned) and re-time. Refute if cndmask
  drops but time doesn't.
- **H4:** after H1–H3, if matched-occupancy ctx4096 gen/owned isolated-time ratio is still ≫1 and the hot
  loop is a serial reduce→recurrence chain, inspect the disasm for unhidden `s_waitcnt lgkmcnt(0)` between
  the cross-lane ladder and the recurrence update → confirms latency-bound serial chain comgr can't pipeline.

## 4. Recommended first change (default-off, correctness-gated)

Two cheap steps before any big rewrite:
1. **H2 occupancy fix** — derive the split count from the actual ctx (concrete `S=48` grid), not from MAXC.
   Cheapest, isolates the ctx512 starvation from the dataflow wall, and makes the ctx4096-matched number the
   honest authority. Validate: route gate clean + W==D@512 moves.
2. Then the real lever, **H1 — token-shard the q·k dot** in a NEW microgate variant
   (`flash_block_tiled_xlane_score_pv_tile_*_tokshard`): lanes own tokens (not e-slices), each lane computes
   a full Hd dot from LDS-resident K, so there is **no per-token cross-lane reduce**; merge the TK partials
   once per block. Validate numerically in the microgate FIRST (fp16 ≤2e-5), then ISA-diff (expect
   `ds_bpermute` → toward owned's 5), then W==D@4096.
Keep everything behind `DECODE_ATTN_BLOCK_TILE` + a sub-flag; correctness-gated by the block-tile microgate.

## 5. Stop condition (scheduler wall vs kernel-authoring)

Conclude this is a **codegen-scheduler wall** (not an authoring issue) — and stop hand-iterating — when ALL hold:
- occupancy is matched at the measured ctx (H2 done),
- per-token cross-lane is eliminated (H1 token-shard done) and predication cleaned (H3),
- yet isolated per-kernel time at ctx4096 is still ≫ owned, and the disasm shows the residual is unhidden
  serial latency (lgkmcnt/vmcnt(0) stalls on a true dependency chain comgr cannot pipeline).
Then the label `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING` is CONFIRMED and is the **same wall as prefill GEMM**
(perf-state memory: needs an asm scheduler / vendored .co). The resolution is then a codegen instruction
scheduler (the dormant `extra/qk_asm_scheduler.py` / Track-B `Ops.INS` path), which is a separate, larger
project — and the honest terminal read: **competitive decode attention AND prefill GEMM are both blocked on
the one missing capability (a tinygrad instruction scheduler); machine-search purity should be proven on the
simpler generated GEMV/lane-map routes first.**

Conversely, if H1+H2(+H3) move W==D materially toward baseline, the wall was the per-token-cross-lane
dataflow, not the scheduler — keep going on the tile.

## Commands

```
DEV=AMD JIT=1 DECODE_ATTN_BLOCK_TILE=1 PYTHONPATH=. python3 extra/qk_decode_isa_vectorization_gate.py
DEV=AMD JIT=1 DECODE_ATTN_BLOCK_TILE=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py
DEV=AMD JIT=1 DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 DECODE_ATTN_BLOCK_TILE=1 V_DOT2_LOWERING=1 \
  PYTHONPATH=. python3 extra/qk_decode_runtime_overhead.py   # vs 82.4/103.5/101.8/94.6
# isolated per-kernel timing (H1): DEBUG=2 eager custom_kernel of the block tile vs owned at ctx 512 and 4096
```

Constraints: default-off; correctness-first; microgate before model; do not add another attention layout;
do not revive score-broadcast. Codex prompt: `docs/decode-block-tile-scheduling-codex-prompt.md` (to follow
if you greenlight H1/H2).
