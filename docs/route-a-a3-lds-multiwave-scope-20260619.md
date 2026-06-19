# CONTINUATION PROMPT — Route A / A3: LDS-staged, multi-wave RDNA3 WMMA GEMM (chase LLVM 42 → Tensile 66, dependency-free)

Paste this as the opening prompt of a fresh session. It is self-contained.

---

## Mission
Build a **dependency-free, LDS-staged, multi-wave RDNA3 (gfx1100) WMMA GEMM** that beats the single-wave A2
pipeline (~24–32 TFLOPS) and chases tinygrad's own LLVM-WMMA peak (**~42 TFLOPS**, the realistic target) and
rocBLAS/Tensile (**66–77**, the stretch). Built via tinygrad's `assemble→ELF` backend, **zero LLVM, zero
external `.co`**. If it clears ~42 (matching LLVM), route it in-model for prefill (A4). If it plateaus below the
single-wave kernel, or clearly below 42 with a named binding reason, that is the verdict — STOP and report (KILL).

Working dir: `/home/ubuntu/tinygrad-arkey`. Model: gfx1100 / RX 7900 XTX, wave32, LDS = **64 KB/workgroup**,
VGPR = **256/wave**. A parallel "Codex" agent edits the repo — **only touch new files and
`extra/gemm/rdna3_wmma_matmul.py`** (don't revert its uncommitted edits). Commits: `[test]` prefix, end with
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`, surface the short SHA. **No BEAM.**

## Why this is a PORT, not a new capability (the de-risking that justifies the work)
Every hard component already exists and is proven; A3 is assembling them:
1. **RDNA3 WMMA operand layout — SOLVED** (`extra/gemm/rdna3_wmma_matmul.py`, A1). A/B = 8 VGPR/lane (16 fp16),
   B stored transposed, C/D = 8 VGPR fp32, lane map `D[i] of lane l = C[row=i*2+(l>>4&1)][col=l&15]`. RMSE 2e-4.
2. **Hand-written software-pipelined K-loop in RDNA3 asm — PROVEN** (`build_gemm_pipe`, A2): double-buffer +
   targeted `s_waitcnt(vmcnt)`, correct, +32%. This is the *same pipelining* the LDS kernel needs across LDS.
3. **LDS + multi-wave on the asm path — PLUMBING CONFIRMED** (`extra/gemm/rdna4_asm_matmul.py`, the structural
   template). The exact mechanism (verified this session):
   - LDS: add `lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=BYTES, addrspace=AddrSpace.LOCAL), (), 'lds')`
     and include `lds` in `UOp.sink(A.base, B.base, C.base, lds, *gidxs, *lidxs, arg=…)`. Address LDS via raw
     `ds_load_b128`/`ds_store_b128` byte offsets. **Pad `size` to control occupancy** (rdna4 ref uses
     `max(LDS_SIZE, 65536//LIMIT_OCC)` — bigger LDS request = fewer resident workgroups).
   - Waves: `lidxs = [UOp.special(128, "lidx0")]` → 128 threads = **4 wave32**. (A1/A2 used `UOp.special(32,…)`
     = 1 wave — that single-wave-per-workgroup is THE binding reason A2 plateaued: no inter-wave latency hiding.)
   - Sync: RDNA3 has `s_barrier` (single instruction — simpler than RDNA4's `s_barrier_signal`/`s_barrier_wait`
     pair; just emit `s_barrier()`). DS waits use **combined `s_waitcnt` lgkmcnt** (RDNA3 has NO split
     `s_wait_dscnt`): lgkmcnt = bits[9:4] (encoder below). `ds_*` loads decrement lgkmcnt.
   - Confirmed present in `tinygrad.runtime.autogen.amd.rdna3.ins`: `ds_load_b128`, `ds_store_b128`,
     `ds_load_b64`, `ds_store_b64`, `s_barrier`, `s_waitcnt`. (Absent — RDNA4-only: `s_barrier_signal/wait`,
     `s_wait_dscnt`, `s_wait_loadcnt`. Don't use them.)

So the work is: re-derive the LDS↔fragment addressing for RDNA3's 8-VGPR/16-fp16 WMMA layout, map 4 waves to
output sub-tiles, and port the rdna4 ref's double-buffered LDS pipeline to RDNA3 encodings.

## State (read these first)
- `docs/route-a-a2-pipeline-result-20260619.md` — A2 result + the single-wave binding reason (READ FIRST).
- `docs/route-a-rdna3-wmma-result-20260619.md` + `…a2-continuation-prompt-20260619.md` — A1/A2 prior prompts.
- `extra/gemm/rdna4_asm_matmul.py` — the LDS-staged multi-wave **structural template** (RDNA4: 128×128 block,
  4 waves as 2×2, each computes a 64×64 sub-tile = 4×4 WMMA tiles; double-buffered LDS with `emit_iter_body`
  interleaving `ds_load`s among WMMAs; bank-conflict-free padded LDS strides). **Idea + structure transfer;
  its 4-VGPR-fragment layout, encodings, and `s_barrier_signal`/`s_wait_dscnt` do NOT — they hang gfx1100.**
- `extra/gemm/rdna3_wmma_matmul.py` — A1 `build_gemm` + A2 `build_gemm_pipe` + `waitcnt_vm` (RDNA3, correct).
- `extra/gemm/amd_uop_matmul.py`, `extra/gemm/amd_flash_attention.py` — UOp-level LDS references (not asm, but
  validate LDS layout/bank ideas).
- memory `amd-decode-next-step.md` (the BANKED arc) + MEMORY.md POWN/Route-A line.

## Numbers / ceilings (gfx1100, prefill ffn shape M=512 K=4096 N=12288 unless noted)
| kernel | TFLOPS | note |
|---|---:|---|
| A2 single-wave pipeline (best) | **24–32** | the bar to clear first |
| tinygrad LLVM-WMMA peak (POWN) | **~42** (35% of peak) | **the realistic A3 target** — LLVM *does* LDS+multi-wave |
| rocBLAS / hipBLASLt | **66–77** | stretch; needs full SW-pipelined K-loop on top of LDS |
| RDNA3 fp16 WMMA hardware peak | ~122 | 42=35%, 66=54% |

POWN verdict (banked): the `42→66` gap is the **software-pipelined K-loop** (a codegen capability tinygrad's
LLVM path can't express). A3's bet: LDS multi-wave gets us to ~42 (match LLVM); layering the A2 pipeline
technique (proven in asm) on top is the *only* dependency-free shot at >42. **Honest ceiling: matching ~42 is
plausible; 66 requires the full pipeline to land AND for the IC-served caveat (below) not to bite.**

## Technical design (RDNA3-specific — the parts that differ from the rdna4 ref)
- **Tile / wave mapping:** start with BLOCK_M=BLOCK_N=128, BLOCK_K=16, 4 waves as 2×2 (wave_m=lidx0>>5&1 mapped
  to rows, wave_n=lidx0>>6&1 to cols), each wave = 4×4 WMMA tiles → 64×64 sub-tile. (Same as rdna4 ref.)
- **VGPR budget — the RDNA3-specific squeeze:** RDNA3 fragments are **8 VGPR/16-fp16** (RDNA4 = 4 VGPR), so
  fragment pressure is **2× the rdna4 ref**. Per wave: acc = 4×4×8 = **128**; one K-slice of fragments =
  4 A-tiles×8 + 4 B-tiles×8 = **64**; double-buffering fragments = 128 → 128+128 = 256 = the whole budget, no
  room for addresses/temps. **Mitigations (decide by measurement):** (a) don't double-buffer *register*
  fragments — rely on LDS double-buffering + targeted lgkmcnt waits (keep 1 fragment buffer = 64, leaving 64
  for temps); (b) smaller per-wave tile (3×3 or 4×3 WMMA tiles); (c) 8 waves/workgroup (256 threads) so each
  wave owns a smaller sub-tile (fewer acc regs/wave → more VGPR headroom, more inter-wave latency hiding) — this
  is plausibly the RDNA3-correct answer since RDNA3 has more waves/SIMD than the 4-wave rdna4 config assumes.
- **LDS layout:** A block = BLOCK_M×BLOCK_K fp16, B block = BLOCK_K×BLOCK_N fp16 (store B transposed for
  contiguous WMMA column reads, as A1 does in global). Use **padded strides to avoid bank conflicts** (RDNA3 LDS
  = 32 banks × 4 B; the rdna4 ref's padded-stride comments are the model — re-derive offsets for the 8-VGPR
  read pattern: each lane reads 16 contiguous fp16 = one b128+one b128, or one b128 + b64×… match A1's frag
  shape). Budget: 128×16 + 16×128 fp16 = 8 KB single-buffer, 16 KB double-buffered — well under 64 KB, so LDS
  size is occupancy-tuned, not capacity-bound.
- **The K-loop:** waves cooperatively `global_load`→`ds_store` the next K-block to LDS, `s_barrier`, then each
  wave `ds_load`s its fragments and runs its 16 WMMAs. Double-buffer LDS (ping-pong A/B halves) + the A2
  pipeline idea: issue next-block global loads + ds_stores while WMMAs on the current block run, with targeted
  `s_waitcnt(lgkmcnt=…)` for the ds_loads and `s_waitcnt(vmcnt=…)` for the global loads. Guard the last
  prefetch (no over-read past K) — same discipline as A2.
- **waitcnt encoder (RDNA3, from the proven `extra/gemm/amd_asm_matmul.py`):**
  `simm16 = (expcnt&0x7) | ((lgkmcnt&0x3f)<<4) | ((vmcnt&0x3f)<<10)`; full wait = `s_waitcnt(simm16=0)`. A2's
  `waitcnt_vm(n)` already encodes vmcnt; add `waitcnt_lgkm(n)` for DS, and a combined form for the rare
  both-wait. Set unused counters to max (0x3f / 0x7) so you don't over-serialize.

## Phased plan (correctness-gated, hang-bounded)
- **P0 — LDS smoke (1 wave):** single workgroup, `global_load`→`ds_store`→`s_barrier`→`ds_load`→one WMMA→store.
  Proves DEFINE_LOCAL plumbing + `s_barrier` + lgkmcnt waits on RDNA3. Gate: RMSE<0.05 on a 16×16×16 tile.
- **P1 — multi-wave, no pipeline:** 4 (or 8) waves, 128×128 block, full barriers each K-iter, LDS single-buffer.
  This is the "match LLVM via occupancy + LDS reuse" milestone. Gate: RMSE<0.05 at N=2048 AND prefill shape;
  measure fair vs A2 (`GEMM=1 USEPIPE=1`) in the SAME process. **Decision point:** if P1 already ≥ A2 and
  approaching ~42, the occupancy/LDS thesis is validated → proceed to P2; if P1 ≤ A2, the IC-served caveat is
  biting (LDS staging tax not repaid) → likely KILL.
- **P2 — LDS double-buffer + SW-pipelined K-loop:** port the rdna4 `emit_iter_body` interleaving (issue next
  ds_loads / global loads among current WMMAs, targeted waitcnt). This is the only path >42 toward 66. Gate:
  beats P1, RMSE<0.05.
- **P3 — sweep:** block size {64,128,256}, waves {4,8}, BLOCK_K {16,32}, LDS pad / occupancy (`LIMIT_OCC`
  analog), per-wave tile shape. Verify VGPR/occupancy via `llvm-readobj --notes` on the ELF (vgpr_count,
  vgpr_spill_count — spills = kill that config).
- **A4 (only if P-phases clear ~42 isolated):** route in-model for prefill ffn shapes via the existing
  integration machinery (`extra/qk_tensile_inmodel.py` route_pf16/install — point it at our ELF kernel instead
  of rocBLAS's `.co`, flag-gated). Gate: warm pp512 ≥ PREFILL_V2 warmstart, dNLL ≤ 0.01
  (`extra/qk_prefill_v2_nll_eval.py`), **decode W==D untouched**.

## Gates / discipline
1. **Correctness first** every phase: RMSE<0.05 (fp16) at N=2048 AND prefill shape. Each wrong
   barrier/waitcnt/LDS-offset → GPU **Wait-timeout HANG ~30s** (recovers; verify with a tiny
   `(Tensor([1.,2])*2).realize()` after a hang). Build incrementally; commit correct checkpoints.
2. **Measure FAIR**: best/min over many warm runs, back-to-back vs A2 in the SAME process via the standalone
   `GEMM=1 USEPIPE=1` harness (NOT the interleaved `PIPE=1` one — its absolutes are cache-thrash-contaminated;
   ratio only). NEVER trust a single run (clock-ramp gave 13↔19↔24 this arc). DEBUG=2 `tm` for GPU time, never
   wall-clock ([[amd-decode-measurement-confounds]]).
3. **Bar:** P1 ≥ A2 (else KILL); ultimately beat ~42 isolated to justify A4. 66 is a stretch, not the gate.
4. **KILL (research mode — report, don't grind):** if P1 (multi-wave + LDS, the cheap structural win) does NOT
   beat the A2 single-wave pipeline, the IC-served caveat (CG-R1: LDS staging refuted as IC-served on gfx1100
   for the decode regime) extends to prefill GEMM too → dependency-free path is honestly closed; fall back to
   PREFILL_V2 (~80% llama, shipped) or the external Tensile `.co` (1.41× llama, dependency). Note: the IC-served
   refutation was measured for *decode attention*; prefill GEMM is large-M and POWN showed LLVM's own LDS path
   hits 42, so LDS *should* help here — P1 is the experiment that settles it.

## Honest caveats
- The `42→66` jump needs the full SW-pipelined K-loop (P2) to fully land AND the IC-served tax to be repaid by
  reuse. Realistic expected outcome: **match LLVM ~42** (good, dependency-free); 66 is a stretch goal.
- Multi-day expert-asm work: LDS addressing + bank-conflict avoidance + multi-wave barriers + the VGPR squeeze
  (worse on RDNA3 than the rdna4 ref due to 8-VGPR fragments). Bound GPU iterations (each hang ~30s).
- Even at ~42 dependency-free, weigh against PREFILL_V2 (already ~80% llama, shipped, zero new risk surface).

## First concrete step
P0: in a new `build_lds_tile()` (don't touch `build_gemm`/`build_gemm_pipe`), single wave32, one 16×16×16 tile
through global→LDS(ds_store)→s_barrier→LDS(ds_load)→WMMA→store. Add `waitcnt_lgkm(n)`. Verify RMSE<0.05. This
proves the RDNA3 LDS+barrier plumbing end-to-end before any multi-wave complexity. Decide P1 tile/wave config
from the VGPR budget once P0 is green.
