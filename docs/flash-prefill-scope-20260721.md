# Scope: Flash-Attention for Prefill (2026-07-21)

## ⭐ RESULT / CONCLUSION (2026-07-21) — reuse-decode-kernel path is NO-GO; read first
Spiked the make-or-break (reuse the decode flash kernel's LDS-staging + online-softmax, generalize to an M-query tile). Findings:
- **171× wall broken.** LDS reuse (the prior campaign's missing ingredient) works → correct, and from 171-756× slower down to ~3-7×.
- **Occupancy fixed (S-split), still not enough.** Adding KV live-split (S>1, reusing the decode split+combine) filled the 48 CUs, moved GFLOPS 3-9× off the floor (175→500-1650), but the best geometry is still **~1.5× (KV=512) to ~2.5× (KV=4096) slower than materialized SDPA.** Correctness held.
- **The real wall = compute, not occupancy: the kernel scores via scalar `fdot2`, never WMMA.** SDPA lowers to a tensor-core matmul (~10× more FLOPS-efficient). Peak ~1650 GFLOPS vs the ~15-20 TFLOP matmul regime.
- **Complete picture:** llama's `flash_attn_ext_f16` = fused AND WMMA. Materialized SDPA = WMMA but not fused. Our flash = fused but scalar (no WMMA). Matching llama needs BOTH — a fused online-softmax kernel whose score/PV run on WMMA tiles. That is the "genuinely new, zero in-repo precedent" work (closer to the abandoned Q4K-WMMA-tiled effort), a SEPARATE LARGE PROJECT — not a continuation of the S-split spike.
- **VERDICT: the reuse-decode-kernel flash path is NOT worth further spend.** The remaining llama gap at large context requires WMMA-in-fused-flash, scoped but big. Bonus win banked: found+fixed a latent WAR barrier bug in the decode kernel (`dd94fb03b`).
- **BubbleBeam note (correction):** the repo does NOT use upstream BEAM (it hangs gfx1100 — `model.py:261` "NO BEAM"). Its search = BubbleBeam+FutureSight, a static coalescing-scorer + offline-benchmarked candidate manifests today (static-first for troubleshootability; dynamic autotuner is the intended end-state once end-to-end is proven). So references below to "hand-tune, BEAM-hangs, high-risk" should read as "candidate-search via BubbleBeam" — but the NO-GO above is a *compute-ceiling* result, independent of the search tool.

---

Repo `/home/ubuntu/tinygrad-arkey`, `master @ be68175ba`. Models: Qwen3-8B (Hq=32,Hkv=8,G=4,Hd=128) and
14B (Hq=40,Hkv=8,G=5,Hd=128), gfx1100 (RX 7900 GRE). Read+plan only; no kernel edits here — a separate
worktree (`.claude/worktrees/agent-afe31d62421beb2b0`, untracked `extra/qk/flash_prefill_gate.py` +
`flash_prefill_tile_kernel.py`) is running a minimal-kernel feasibility spike in parallel; its result feeds
Part 3 below and this doc does not duplicate it.

## TL;DR

- **Prize**: at pp4096, llama's fused `flash_attn_ext_f16` keeps attention ~15% of the pass (~17-23ms/layer
  equivalent); tinygrad materializes the full `Hq×T×KV` score tensor (117MB/layer at KV=3584) and attention
  balloons to the dominant cost at long context. Turning flash off on llama costs +925ms at pp4096
  (~23ms/layer) — that's the number a working flash-prefill kernel needs to reclaim. GEMM/FFN is already
  at/ahead of llama (packed-WMMA ~1829 tok/s @ pp512 on 14B, ~parity with llama's ~1837); **attention fusion
  is the entire remaining structural gap**.
- **Reuse strategy**: the decode flash kernel (`extra/qk/flash_kernels.py`) already has the two hard
  ingredients — LDS-staged K/V tiles and online-softmax merge with d-sharded PV — but it is hard-wired to
  M=1 (one query row per workgroup-head). Turning it into flash-prefill means generalizing the *query*
  dimension from 1 to a tile of ~16-64 rows while keeping the same LDS-reuse structure, and adding causal
  masking + WMMA. This is a **new kernel that reuses the decode kernel's topology, not the decode kernel
  itself**.
- **Make-or-break gate**: a prior flash-prefill attempt (git history, refuted 2026-06-17) built a *score-free*
  fused kernel with **no LDS reuse** — it was byte-correct but 171-756× SLOWER than SDPA (0.19 TFLOP/s, ~129×
  redundant HBM re-reads of K/V per output-dim lane). The lesson isn't "flash is hard," it's specifically
  "score-free without shared-memory K/V reuse is a trap." Any new attempt's Step 0 gate must be: does the
  minimal single-head kernel with LDS reuse beat SDPA at all? If not, stop — don't repeat the 2026-06-17
  mistake at higher head-count.
- **Honest risk**: HIGH. This is a hand-written LDS/WMMA kernel project on a backend where BEAM (which would
  normally search tiling automatically) hangs on gfx1100 — every tiling/occupancy choice must be hand-tuned
  and hand-gated. The repo's track record on hand-kernel LDS/WMMA attention work is mixed: the *decode*
  block-tile kernel shipped but is "correct-not-fast" (~60-68% of the owned/oracle tile per
  `docs/pure-machine-search-roadmap.md:22`); the *prefill* attempt was refuted outright. Budget this as a
  multi-day-to-multi-week kernel-authoring effort with a real chance of ending in "banked, not shipped" again
  — same as 2026-06-17 and 2026-06-20.

---

## Part 1 — Self-review of existing code

### 1.1 The decode flash kernel — `extra/qk/flash_kernels.py`

One live builder remains: `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S,
Tc, staging, quant, rope)` (`extra/qk/flash_kernels.py:10-146`). A header comment (line 6-8) records that ~32
other builders (research/refuted score, PV, combine, lifecycle variants) were deleted as orphans on
2026-07-06 — this file is now a single generated candidate, not a library.

**Topology** (`flash_kernels.py:26-27`): `G = Hq//Hkv` warps per workgroup (one warp per GQA query head
sharing a kv-head), `LANES=32` lanes/warp (`THREADS = LANES*WARPS`), `TK=16` K-rows staged into LDS per block,
`R = Hd//LANES` output elements per lane, `NB = ceildiv(L, TK)` inner blocks. Grid: one workgroup per
`(kvh, split)` pair (`kvh = UOp.range(Hkv,...)`, `s = UOp.range(S,...)`, both `AxisType.GLOBAL`), `lane`/`warp`
are `AxisType.LOCAL` (not `UOp.special`, because `UOp.special` blocks `add_gpudims` — comment at line 42-44 —
needed for correct `ds_bpermute` lane addressing in the warp-reduce).

**LDS staging** (lines 49-90): `ksh`/`vsh` are `AddrSpace.LOCAL` placeholders sized `TK*Hd` half each (8KB
total for `KV_BOTH`, 4KB for `K_ONLY` where V is read straight from the global cache, relying on L2 warming).
An optional cooperative-staging LaneMap (`DECODE_STAGE_COALESCE`, default off) lets each thread own a
contiguous chunk so the global→LDS copy vectorizes to `global_load_dwordx4`; the fallback is
one-element-per-thread. A `UOp.barrier` closes the staging loop before any lane reads the tile.

**Score compute** (`_dot_reduce`, lines 91-101): per token, an `RP = Hd//64` reduce loop does an
`__builtin_amdgcn_fdot2` (packed fp16×fp16 dot, 2 elements/instr) against the LDS-staged K row, then a
warp-reduce-sum across the 32 lanes (`_warp_reduce_sum_staged` or an inline variant gated by
`DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE`). This is a **scalar dot product per (query-head, token)** — no WMMA
anywhere in this kernel; the "tensor core" work the task description alludes to for prefill does not exist
yet even in the decode reference.

**Online softmax** (lines 92-137): standard running max/sum/acc carried in `AddrSpace.REG` placeholders
(`acc[R]`, `den[1]`, `mx[1]`), updated per TK-block: `new_m = max(old_m, sc)`, `corr = exp(old_m-new_m)`,
`p = exp(sc-new_m)`, `acc = acc*corr + p*vd`, `den = den*corr + p`. An optional two-pass variant
(`DECODE_ATTN_TILE_SPLIT_SCORE`) separates the independent per-token score computation (pass 1, writes an LDS
score buffer) from the serial online-softmax merge (pass 2), so the `ds_bpermute` warp-reduces pipeline
back-to-back instead of stalling the merge chain.

**PV / d-sharding**: `d = lane*R + dd` — each of the 32 lanes owns `R = Hd/32` contiguous output elements, so
the P·V accumulation is embarrassingly lane-parallel with no cross-lane reduce needed for the output (only the
QK dot needs the warp-reduce). Output written to `pout[(h*S+s)*W + d]` where `W = Hd+2` (the extra 2 slots
hold `l` and `m` for the fused combine across splits, lines 138-144).

**M=1 baked in — where exactly**: nothing in the kernel body computes over a query tile; `q` is indexed once
per warp via `h*Hd + _e2` (line 96) with no query-row loop/range at all. Generalizing to M query rows means
adding a query-tile range and re-deriving (a) the LDS layout (would need either M copies of the K-tile-vs-M-Q
scores or a query-major restructuring), (b) M independent online-softmax states instead of one, (c) M-times
the register pressure for `acc`/`den`/`mx` unless further tiled. This is a nontrivial restructuring, not a
parameter bump.

**GQA**: handled by `G = Hq//Hkv` warps per kv-head-workgroup, each warp doing its own head's QK/softmax/PV
against the *same* staged K/V tile — this pattern (GQA-heads-share-staged-KV) is exactly what prefill needs
too, and transfers directly.

**Quant/rope**: both fused at the load site via `extra/qk/kv_load.py`'s `make_kv_element_loader` (int8
dequant: `val*scale[which,kvh,tok]`; rope-at-read: rotate un-roped K in-register from a `freqs` buffer,
`kv_load.py:32-40`). This load-site abstraction is reusable as-is by a prefill kernel with no changes.

**Reusability verdict for prefill**: LDS K/V staging mechanism (barrier, cooperative-stage option) —
**directly reusable**. Online-softmax algebra — **reusable in form, must be re-derived per-query-row (or
per-query-tile) instead of once**. d-sharded PV — **directly reusable** (independent of M). GQA
warp-per-query-head-sharing-KV-tile — **directly reusable**. `kv_load` fused dequant/rope — **directly
reusable**. Scalar `fdot2` scoring — **not what's wanted for prefill's compute-bound QK^T**; prefill needs
WMMA on the QK^T (and ideally PV) to get FLOP/s, which this kernel has zero precedent for.

### 1.2 Decode executor + spec + microgate — the custom_kernel bridge

- `extra/qk/flash_decode_attention_executor.py:10-25` (`flash_decode_live_split_block_tile`): flattens `q` to
  `[Hq*Hd]`, builds inputs `(q_f, cache_kv[, kv_scale][, freqs])`, calls
  `Tensor.empty(...).custom_kernel(*inputs, fxn=spec.emit_tile(Tc_u))[0]` for the tile kernel, then a second
  `custom_kernel` for `spec.emit_combine()` (the two-kernel split-combine; the old un-fused combine was removed
  2026-07-06, `fused_combine=False` now raises).
- `extra/qk/flash_decode_attention_spec.py`: pure dataclass descriptor layer — `LiveSplitGeometrySpec`
  (`split_count`, `token_block=16` fixed; `per_split_length`/`aligned_per_split_length`/`blocks` all operate on
  a possibly-symbolic `Tc:UOp` via `ceildiv_uop`, lines 26-40), `FlashDecodeTileSpec` (validates Hq/Hd/Hkv/MAXC,
  `staging∈{KV_BOTH,K_ONLY}`, `token_block` must currently be `16`; `.emit()` calls the kernel builder in
  `flash_kernels.py`), `FlashCombineSpec`, and the top-level `FlashDecodeAttentionSpec`/`describe_flash_decode_attention`.
  This descriptor/spec split (geometry math separate from UOp construction) is a clean reusable pattern for a
  prefill spec, but **every numeric field here (split_count, token_block=16 fixed, per-split alignment) is a
  decode-only geometry** built around "how many workgroups do we split one query row's KV range into,"
  not "how do we tile T query rows against KV." A prefill spec needs an analogous-but-distinct geometry
  (query-tile size, KV-tile size, causal skip range).
- **custom_kernel ABI**: `Tensor.custom_kernel(*lst, fxn, grad_fxn=None) -> list[Tensor]`
  (`tinygrad/tensor.py:194-200`) is a thin wrapper over `UOp.custom_kernel`; `fxn` is a Python callable
  `(pout:UOp, *inputs:UOp) -> UOp` that hand-builds the kernel body with `UOp.range`/`.store()`/`.sink(arg=KernelInfo(...))`
  — i.e. hand-authored UOp graphs bypass the normal scheduler/BEAM entirely (this is *why* this whole family
  of kernels exists: it's the only way to get hand-controlled LDS/warp-reduce/barrier structure past a backend
  where BEAM hangs).
- **Live-split geometry / gating**: `extra/qk/live_split_geometry.py` provides `ceildiv_uop` and
  `flash_fused_gmax_combine_kernel`. Route-level gating lives in `tinygrad/llm/decode_routes.py`
  (`flash_decode_attention_route`, line 130) and `tinygrad/llm/route_policy.py`
  (`should_use_flash_decode`, line 179) — see 1.4 below; there is no separate
  `decode_attention_block_tile_microgate.py` file in the current tree (the task's cited path does not exist at
  `be68175ba`; it may have been renamed/removed in the 2026-07-06/tier-3 cleanups — the live equivalent is the
  `_decode_attention_rolled_back` predicate in `extra/qk/pure_search_guard.py:48-51`, which reads
  `DECODE_LIVE_SPLIT` (default on) as the sole route selector).

### 1.3 The prior flash-prefill attempts — git history

Full arc, oldest to newest (all on `master`, all reachable by `git show`):

1. `2411fd8b0` **Stage-0 gate**: measured KV-tiled online-softmax *expressed in ordinary tinygrad ops*
   (no custom kernel) against SDPA at the real 8B shapes (Hq32/Hkv8/Hd128/T512, causal, GQA). Exact (err
   ≤0.004) but **0.15-0.52× (SLOWER)** — the online-softmax state traffic (`acc[Hq,T,Hd]` fp32 per tile) +
   GQA `repeat_interleave` cost more than SDPA's materialized-but-reused path. Verdict: no cheap
   ops-level win; the only path to a real win is a custom fused kernel with register/LDS-resident state.
2. `908cc8f96` **Phase 2 expressibility proof** (`extra/qk_flash_prefill_custom.py`): a single-head causal
   attention custom kernel proving the compute *can* be expressed without materializing `[T,KV]` scores.
   Key finding in the file header: formulation **A** (single coupled-accumulator online softmax in one
   kernel) is **REJECTED by the linearizer** (`assert y.src[1] not in x.backward_slice_with_self` in
   `codegen/late/linearizer.py` — the same coupled-multi-accumulator-reduce hazard flash-decode warns
   about). Formulation **B** (ship): two *sequential single-accumulator* reduces — pass 1 computes
   `m[i]=max_j(qk)`, pass 2 recomputes the dot and does a 1s-augmented weighted sum that folds the softmax
   denominator into an extra output column — plus a tiny combine kernel (divide). This works but **recomputes
   the QK dot twice** (no online fusion) and, critically, **every lane independently re-reads all of K (and V)
   from global memory** — there is no LDS tile shared across lanes anywhere in this formulation.
3. `1b4eb6200`/`dadd32051` **Phase 3/4**: measured this kernel (single-head, then GQA) at ~2.7-2.8× *faster*
   than SDPA — **this was wrong**.
4. `e3683ef9d` **Phase 5 correction** (superseding 3/4): the 2.7-2.8× numbers were wall-clock-around-`.realize()`
   in a warm loop, which measures host dispatch / cache-hit no-ops, not GPU execution. Honest `DEBUG=2`
   per-kernel `tm` (summed compute kernels, excluding one-time device-init copy, subprocess-isolated) gives
   the real numbers:

   | case | flash GPU ms | SDPA GPU ms | slowdown |
   |---|---:|---:|---|
   | single-head KV=512 | 45.2 | 0.3 | **171×** |
   | single-head KV=3584 | 332.4 | 1.0 | **338×** |
   | GQA KV=512 | 1374.3 | 1.8 | **756×** |
   | GQA KV=3584 | too slow/faulted to measure | 9.6 | ≫ |

   Root cause (both the commit message and the banked doc, `git show 44eaee4e9`, agree): the output dim `d`
   (`W=Hd+1=129`) is a **GLOBAL lane** in formulation B, so all 129 lanes independently stream the *entire*
   K (and V) from HBM to recompute the dot — ~129× redundant reads (≈178GB at KV=3584 vs SDPA's ≈1GB),
   running at ~0.19 TFLOP/s / ~367 GB/s regardless of shape. **Being "score-free" (no `[T,KV]` materialization)
   does not help if the alternative is re-reading K/V per lane instead of staging it once in LDS and
   reusing it.** SDPA materializes scores but *reuses* the K/V data efficiently; that wins by 2-3 orders of
   magnitude here. The doc explicitly states the fix is LDS tiling ("real flash-2… BEAM-territory, hangs
   gfx1100… hand-LDS is dangerous-power surface") and that Phase 6 (integration) was **cancelled**.
5. `5b933a047` (2026-06-20) **Follow-up scope**, `docs/prefill-flash-wmma-attention-scope-20260620.md`
   (archived/pruned from the working tree since; recovered via `git show`): reframes the prize after a
   separate "Branch B" fusion win cut *concrete*-regime attention to ~5% (near llama's 4.4%) — the remaining
   prize was said to live entirely in the **symbolic regime** (chunks 2+, i.e. `start_pos>0` continuation
   chunks, where `isinstance(start_pos,int)` gates the concrete TC-attn path off, line 583 today). Proposed
   3 increments: (0) force concrete KV so the shipped fusion fires on every chunk (no new kernel), (1) WMMA
   on the *explicit* (materializing) attention matmuls, (2) the real fused flash-prefill kernel (deferred,
   "multi-day build"). **Increment 0 is directly relevant to today's state**: `be68175ba` (2026-07-21, the
   commit this scope doc's `master` sits on) *is* Increment 0 — `prefill_concrete_kv_auto_decision` now
   defaults concrete-KV on for prefill-v2, fixing a bug where every continuation chunk silently fell to slow
   symbolic SDPA (`_workload_reuse` was hardcoded `False`). Verified real 14B `generate()` recovery: pp1024
   530→1664 (3.1×), pp2048 280→1515 (5.4×), pp4096 152→1258 (8.3×) tok/s. **This closes most of the "symbolic
   regime" prize the 2026-06-20 doc worried about — it does NOT touch the fundamental materialize-scores
   problem**, per the handoff doc's own words (`docs/14b-packed-wmma-warmstart-handoff-20260721.md:10`):
   "Both prefill attention branches materialize the full T×KV score matrix; the only flash kernel is
   decode-only (hard-gated T==1)… closing that residual needs a new flash-*prefill* kernel… deferred."

   **Methodology lesson** (repeated across 2/3/4/5 above, and this is the one fact the task explicitly asks
   to extract): **never time GPU kernels by wall-clock around a warm-loop `.realize()`** — it silently
   measures host dispatch and JIT-cache hits, not device execution, and can invert the verdict by 2-3 orders
   of magnitude. The only trustworthy signal here is `DEBUG=2` per-kernel `tm` (or `GlobalCounters.time_sum_s`
   with an explicit device sync), summed over the *compute* kernels only (exclude one-time device-init
   copies), ideally subprocess-isolated per case so a slow/faulting case can't corrupt later timings in the
   same process.

### 1.4 Current prefill attention path — `tinygrad/llm/model.py:570-603`, `tensor.py:1175-1204`

Three-way branch per forward call (`model.py:577,583,596`):

1. **Flash decode route** (line 577): `_should_use_flash_attention` → `ring_freqs is not None or
   _route_should_use_flash_decode(start_pos, T, use_flash)`. `should_use_flash_decode`
   (`tinygrad/llm/route_policy.py:179-187`) **hard-gates on `T==1` and `start_pos` being a `UOp`** (line 180:
   `if not (isinstance(start_pos, UOp) and isinstance(T, int) and T == 1): return False`) — this is the literal
   M=1 gate; prefill (T>1) can never enter this branch regardless of any env var. When it does fire (decode
   only), it calls `flash_decode_attention_route` (`tinygrad/llm/decode_routes.py:130`), which builds the
   live-split geometry and dispatches to the block-tile kernel described in 1.1/1.2.
2. **Concrete TC-attn path** (line 583): gated on `self.config.prefill_tc_attn and self._prefill_v2 and
   isinstance(start_pos, int) and T != 1` — i.e. prefill-v2 model, and `start_pos` is a Python int (concrete,
   not symbolic). Builds `qg[B,Hkv,G,T,Hd]`, `kg`/`vg[B,Hkv,1,KV,Hd]` (GQA via the `G` broadcast dim, no
   `repeat_interleave` — `kg`/`vg` broadcast against `qg`'s `G` axis in the matmul), computes
   `scores = (qg @ kg.transpose(-1,-2)).float() * scale` (**materializes the full `[B,Hkv,G,T,KV]` score
   tensor** — this is the explicit path the handoff doc calls out), adds the causal `mask`, softmax, then
   `s.cast(fp16) @ vg`. This is "TC attention" in name (the matmuls are eligible for the warmstart TC opt path
   the same way the FFN matmuls are — see 2.2/2.3 of `model.py`) but there is **no dedicated attention TC
   warmstart key today** (the `5b933a047` doc's "Increment 1" — add attention-shape keys to `_WARMSTART_OPTS`
   — was never built per that doc's own status; not reverified live here since building/measuring is out of
   scope for this doc).
3. **Symbolic SDPA fallback** (line 596-598): everything else (symbolic `start_pos`, non-prefill-v2 models,
   T==1 non-flash-eligible edge cases) calls `q.scaled_dot_product_attention(k, v, attn_mask=mask,
   enable_gqa=True)`. `Tensor.scaled_dot_product_attention` (`tensor.py:1175-1204`) is pure ordinary-op
   composition: GQA via `key.repeat_interleave(...)`/`value.repeat_interleave(...)` (line 1192-1193, **not**
   the branch-free broadcast the TC-attn path uses), `qk = q.matmul(key.T, dtype=upcast) / sqrt(Hd)`, mask add,
   `qk.softmax(-1) @ value`. This is a **plain materializing matmul→softmax→matmul chain with zero custom
   kernel involvement** — it is exactly what tinygrad's ordinary scheduler produces, and it is what "cratered"
   for continuation chunks before `be68175ba` (every `start_pos>0` chunk hit this path because `start_pos` was
   a symbolic `UOp` for the multi-chunk JIT).
4. **Causal mask**: built once at line 572, `Tensor.full((1,1,T,start_pos+T), -inf).triu(start_pos+1)` when
   `T!=1` — explicitly noted (line 570) as `causal_lower_right`, not the `causal_upper_left` semantics of
   `is_causal=True`. Both the TC-attn path (reshaped to broadcast over `Hkv,G`) and the SDPA fallback consume
   this same mask tensor.
5. **Symbolic vs concrete KV**: the branch selector at line 583 (`isinstance(start_pos, int)`) is literally
   the fork point between "gets the materializing-but-fused TC path" and "gets the symbolic ordinary-op SDPA
   path." Concrete-KV-by-default (`be68175ba`) means most prefill chunks now hit branch 2, but branch 2 still
   materializes the full score tensor — it's a better-fused materialize, not a flash (no-materialize) path.

### 1.5 Relevant infra

- **custom_kernel / UOp-kernel machinery**: `Tensor.custom_kernel` (`tensor.py:194-200`) → `UOp.custom_kernel`.
  Kernels are hand-built Python closures `(pout, *inputs) -> UOp` using `UOp.range`, `.after()` (ordering),
  `.store()`, `AddrSpace.{LOCAL,REG}` placeholders, `UOp.barrier`, and `.sink(arg=KernelInfo(name=..., 
  opts_to_apply=()))`. Because `opts_to_apply=()`, these kernels skip the scheduler's opt-search path entirely
  — this is the mechanism that lets a kernel author hand-place LDS staging and barriers on a backend where
  automatic search (BEAM) doesn't work.
- **Warmstart/candidate system**: `tinygrad/codegen/opt/postrange.py:530-607` — `_WARMSTART_OPTS` is a global
  dict keyed by `warmstart_key(out_dims, reduce, packed_dtype)` (shape-keyed, concrete dims only — a symbolic
  reduce dim can't key into it, per the 2026-06-20 doc's Increment-2 note), installed via the
  `warmstart_candidate_state` context manager, consumed in `apply_opts` (line 587) as a *forced* `Opt` list
  applied without search when the AST's shape key matches. `model.py:261-276` (`_prefill_v2_opts`) is the
  richest example: forces `Opt(TC,...)` + `UPCAST` + `UNROLL(reduce,8)` on the prefill-v2 fp16 FFN/projection
  matmul shapes — explicitly commented "NO BEAM — BEAM hangs gfx1100" (line 261). No attention-shape keys
  exist in this table today (confirmed by reading the full warmstart-opts construction site — only FFN/
  projection shapes are registered).
- **BEAM-hangs blocker — where documented**: authoritative live citation is `tinygrad/llm/model.py:261`
  ("gate-validated; NO BEAM — BEAM hangs gfx1100"). Corroborated by the archived-but-git-recoverable
  `docs/prefill-flash-wmma-attention-scope-20260620.md` ("No BEAM (hangs gfx1100)") and
  `docs/amd-decode-prefill-v2-increment2-phase5-correction-20260617.md` ("real flash-2 needs LDS tiling…
  that opt is what BEAM would find, but BEAM hangs gfx1100, and hand-LDS is dangerous-power surface"). No
  currently-committed doc explains the *root cause* of the hang (out of scope here); the practical consequence
  repeated everywhere is: **any new LDS/tiling kernel work on this backend must be hand-authored and
  hand-gated, not BEAM-searched.**
- **AMD:ISA renderer path**: `tinygrad/renderer/amd/elf.py:53` `assemble_linear(prg, lin, arch)` — the native
  UOp→AMD-ISA assembly path (`Ops.INS`) referenced by `docs/pure-machine-search-roadmap.md` as the
  "final-mile control" layer that lets generated kernels bypass LLVM/comgr. Relevant if a flash-prefill kernel
  needs custom instruction sequences (e.g. `v_wmma`, `ds_bpermute`) beyond what the UOp→HIP-C path emits
  cleanly; the decode kernel's `_warp_reduce_sum_staged`/`amd_warp_reduce.py` and `__builtin_amdgcn_fdot2`
  CUSTOMI usage (1.1 above) show the pattern for injecting AMD-specific intrinsics without going through the
  full ISA renderer.
- **Pure-machine-search guard**: `extra/qk/pure_search_guard.py` classifies every hot route's provenance
  (`machine_authored_generated` / `tinygrad_scheduler_generated` = pure; `compiler_primitive_spec_owned` /
  `external_handwritten_kernel` / `hand_authored_uop_template` / `rollback_oracle` = impure) and can hard-fail
  a run under `PURE_MACHINE_SEARCH_ONLY=1`. A hand-authored flash-prefill custom kernel would land in one of
  the impure buckets — worth noting for whoever wires it in: it will need either an explicit manifest entry
  (`extra/qk/route_manifest.py`) accepting it as `hand_authored_uop_template`, or it stays opt-in/env-gated
  (as the decode flash route already is via `DECODE_LIVE_SPLIT`) so the pure-search guard isn't tripped by
  default.

---

## Part 2 — The gap: decode kernel → working flash-prefill

Enumerated changes needed, each with what it reuses from 1.1/1.2 and its risk:

1. **Query-dim tiling (M: 1 → tile of ~16-64 rows sharing the staged K/V tile).** The decode kernel's whole
   point is one query row (one warp) reading a K/V tile staged once per block. Prefill needs a query-tile of
   `Mq` rows *also* sharing that same staged K/V tile — this is what actually captures the reuse win (K/V
   read once from HBM, reused against `Mq` queries × `TK` keys, not just against 1 query). Concretely: add a
   query-row range inside (or replacing) the current per-warp single-query structure; each lane or sub-group
   now needs to iterate `Mq` queries against the staged `TK` keys before advancing the K-tile.
   **Risk**: this is the single highest-value/highest-risk change — get the reuse factor wrong (e.g. structure
   it so `Mq` doesn't actually share one LDS load) and you reproduce the exact 2026-06-17 failure mode
   (correct math, no real reuse, memory-bound). The Step-0 gate in Part 3 exists specifically to catch this
   before scaling up.
2. **Online-softmax generalized across the query tile.** Today: one `(acc[R], den[1], mx[1])` triple per warp
   (=per query row). For `Mq` queries you need `Mq` independent softmax states — either `Mq`× the register
   footprint (fine for small `Mq`, e.g. 16, given `R=Hd/32=4` per state) or a further register-tiling scheme.
   **Risk**: register pressure. The decode kernel already uses `AddrSpace.REG` placeholders per warp; scaling
   to `Mq=16-64` states × `R=4` fp32 accumulators each could blow VGPR budget on RDNA3 and force occupancy
   down (this backend has already hit a VGPR wall on a *different* kernel — the "4x4 permanently excluded"
   note at `model.py:262-263` — so this is a documented real constraint on gfx1100, not hypothetical).
3. **GQA (G=4 for 8B / G=5 for 14B).** **Directly reusable** — the decode kernel's `G` warps-per-kv-head
   structure (1.1) is exactly the "GQA heads share one staged KV tile" pattern prefill also wants. The
   query-tiling change (item 1) needs to compose with this without blowing the thread/warp budget
   (`THREADS = LANES*WARPS` today; adding a query-tile dimension multiplies the per-workgroup work, not
   necessarily the warp count, if query-tiling is done via a register/lane loop rather than more warps).
4. **Causal masking for prefill.** Decode never masks (every staged key is valid for the single decode query
   at `start_pos`). Prefill must skip/zero score contributions for `key_pos > query_pos` per the
   `causal_lower_right` semantics already defined at `model.py:570-572`. Two options: (a) per-(query,key) mask
   check inside the online-softmax score step (simple, adds a branch per pair), or (b) skip whole KV-tiles that
   are fully above the causal diagonal for a given query-tile (real perf win at long context, matches how
   llama's flash implementation likely earns its speed — but adds tile-range bookkeeping tied to the
   query-tile's position). **Risk**: correctness (off-by-one on `causal_lower_right` vs the more common
   `causal_upper_left` bit the existing mask comment at line 570 already flags as a gotcha) and, if tile-skip
   is attempted, an extra layer of index arithmetic to get right under symbolic KV (item 5).
5. **Symbolic KV length for continuation chunks.** The decode kernel already threads a symbolic `Tc:UOp`
   through `LiveSplitGeometrySpec` (`flash_decode_attention_spec.py:26-40`, `ceildiv_uop`) — **directly
   reusable** as a pattern. But prefill's continuation chunks (post-`be68175ba`) are now mostly *concrete*
   KV by default (Increment-0 already shipped), which changes the priority here: symbolic-KV support is
   still needed for the cases concrete-KV opts out of (e.g. many distinct `start_pos` values without JIT
   caching, or explicitly disabled `prefill_concrete_kv`), but it is no longer the primary lever it looked
   like in the 2026-06-20 doc — most of that prize is already banked. Symbolic support should be a *later*
   increment on the flash-prefill kernel (item matches "wire into model.py" step in Part 3), not a blocker for
   proving the kernel works at all.
6. **Register/LDS budget for the query tile.** LDS: `TK*Hd` half elements for K (+ V if `KV_BOTH`) is already
   4-8KB per workgroup in decode; that's unchanged by query-tiling (LDS holds K/V, not Q). The new LDS/register
   pressure is entirely on the **query side**: either stage `Q`-tile in LDS too (adds `Mq*Hd` half more —
   e.g. `Mq=32, Hd=128` → 8KB) or keep Q in registers per-lane (multiplies the existing per-lane register
   footprint by `Mq`). **Risk**: RDNA3 LDS is 64KB/workgroup shared with other uses (register spill, other
   staged buffers) — need to budget K+V+Q tiles + softmax state + WMMA fragment storage against that ceiling
   for the chosen `Mq`/`TK`.
7. **WMMA on the tiled QK^T (K=128) and PV.** This is the item with **zero precedent in this codebase's flash
   kernels** — the decode kernel scores with a scalar `fdot2` warp-reduce (1.1), not WMMA. Getting FLOP/s out
   of a prefill kernel (where compute, not just memory traffic, starts to matter once queries are tiled)
   plausibly requires `v_wmma` fragments for the `Mq×TK` QK^T tile (reduce dim `Hd=128`, i.e. one clean 128-K
   WMMA tile) and for the `Mq×Hd` PV accumulation (reduce dim `TK=16`, an *awkward* small-K WMMA — RDNA3 WMMA
   native tile is typically 16×16×16, so `TK=16` PV might actually map cleanly, but this needs to be verified
   against actual RDNA3 WMMA instruction shapes, not assumed). **Risk**: this is genuinely new kernel-authoring
   work (fragment load/store layout, `v_wmma` intrinsic emission via `Ops.CUSTOMI` or the AMD ISA renderer
   path from 1.5) with no working reference in-repo to copy from for *attention* WMMA specifically (the
   existing WMMA references — `extra/gemm/rdna3_wmma_matmul.py`, `extra/qk/prefill/wmma.py:501-654`
   `build_gemm_lds2_q4k` — are GEMM-shaped, not flash-attention-shaped; adapting them is real work, not a
   drop-in).

**Net read on the gap**: items 3 (GQA) and 5 (symbolic-KV plumbing pattern) transfer almost for free. Items
1/2/6 (query tiling + its softmax/register consequences) are a substantial but well-understood restructuring
of a kernel that already works for M=1. Item 4 (causal) is a moderate, well-scoped addition. Item 7 (WMMA) is
the one genuinely open-ended, no-reference-in-repo piece of work, and is also the piece most likely to
determine whether the final kernel is compute- or memory-bound — i.e. whether it actually beats SDPA by a
meaningful margin or just barely, or not at all (echoing the 2026-06-17 lesson: LDS reuse fixes the
memory-bound catastrophe, but doesn't by itself guarantee a *large* win over an already-reasonably-efficient
SDPA path, especially post-`be68175ba` where SDPA-adjacent paths are less pathological than they were when the
2026-06-20 doc was written).

---

## Part 3 — The path to ship (ordered, gated)

Each step: build, measurable gate (correctness + honest `DEBUG=2` perf per the Part-1.3 methodology lesson —
**no wall-clock-around-realize, ever**), and stop/fallback condition.

**Perf targets to carry through every step**:
- SDPA baseline (today, post-`be68175ba`): whatever the concrete TC-attn / symbolic SDPA path currently
  measures per-layer at the target context (needs a fresh same-session `DEBUG=2` baseline at kernel-build
  time — do not reuse the pre-`be68175ba` numbers in this doc, they're stale for the concrete-KV-default
  regime).
- llama flash-on: ~17-23ms/layer equivalent (from the established-context prize framing; the +925ms at
  pp4096 flash-off delta ÷ ~36-40 layers ≈ 23-26ms/layer is the concrete anchor already measured).
- The materialization "prize": closing roughly the ~23ms/layer gap at pp4096-scale contexts. Below is a specific
  stated target: attention ~15-17% of the whole pass, matching llama's own regime, at large context.

### Step 0 — Minimal single-head LDS-reuse kernel: does the *concept* beat SDPA at all?
- **Build**: smallest possible query-tiled (e.g. `Mq=16`) single-head, non-causal (or trivially causal),
  concrete-shape kernel that stages one K/V tile in LDS and reuses it across the `Mq` queries. No GQA, no WMMA
  yet (scalar dot is fine here — the point is proving reuse, not proving FLOP/s). This is very likely close to
  what the parallel feasibility-spike worktree is already doing (`flash_prefill_tile_kernel.py`,
  `flash_prefill_gate.py`) — **do not duplicate; this step's result should come from that spike**, not a
  second implementation here.
- **Gate**: correctness (exact vs SDPA reference, small tolerance) AND honest `DEBUG=2` per-kernel `tm` beats
  SDPA at the same shape (even a modest win, e.g. 1.2-2×, is a legitimate pass — the point is proving positive
  ROI on LDS reuse, not hitting the final target yet).
- **Stop/fallback**: if this does NOT beat SDPA even with LDS reuse (i.e. the reuse factor achieved is still
  not enough, or some other overhead dominates at this small scale), **stop the whole line of work** and
  re-bank flash-prefill exactly as 2026-06-17 did — this is the step that would have caught that failure
  before it scaled to GQA. Do not proceed to any later step without a real, honestly-measured win here.

### Step 1 — Add GQA
- **Build**: apply the decode kernel's proven `G` warps-per-kv-head-sharing-one-staged-tile pattern (1.1,
  reusable almost verbatim) on top of the Step-0 query-tiled kernel.
- **Gate**: correctness vs SDPA with `enable_gqa=True` at G=4 (8B) and G=5 (14B); `DEBUG=2` perf still beats
  SDPA at the GQA shape (GQA changes the reuse arithmetic — more independent softmax states sharing the same
  K/V tile should only *improve* the reuse ratio, so a regression here is a red flag worth root-causing before
  continuing).
- **Stop/fallback**: if GQA erodes the Step-0 win to a wash or worse (e.g. register pressure from `G×Mq`
  independent softmax states forces occupancy down), back off `Mq` and re-gate before moving on; if no `Mq`
  recovers a win, stop here — single-head-only flash-prefill isn't useful for a real model.

### Step 2 — Real 40-head shape (14B) / 32-head (8B), realistic T
- **Build**: scale to the actual model shapes (Hq=32/40, Hkv=8, Hd=128, T≈512 per prefill chunk).
- **Gate**: correctness + `DEBUG=2` perf at realistic T; also check LDS/register budget (item 6, Part 2) isn't
  blowing occupancy at full head count.
- **Stop/fallback**: if scaling to real head-count reveals occupancy or LDS-capacity problems that Step 1
  didn't show at smaller scale, this is where `Mq`/`TK` need re-tuning (hand-tuned, no BEAM) before proceeding.

### Step 3 — Causal masking
- **Build**: item 4 from Part 2 — start with the simple per-pair mask check (correctness first); tile-skip
  optimization (skip whole above-diagonal KV tiles) is a follow-up perf pass, not required to pass this gate.
- **Gate**: correctness vs the existing `causal_lower_right` mask semantics (`model.py:570-572`) — this is
  the step most likely to have a silent off-by-one, so gate hard on exact match, not just "close." Perf:
  causal should, if anything, make the kernel *faster* per real key touched (once tile-skip is added) — a perf
  regression from adding masking (with tile-skip) would be a bug, not an expected tradeoff.
- **Stop/fallback**: correctness bugs here are expected on the first pass (this is fiddly indexing); budget
  real debugging time. Don't relax the tolerance to make it pass — a subtly-wrong causal mask silently degrades
  model quality without an obvious crash.

### Step 4 — Symbolic-KV length
- **Build**: item 5 from Part 2 — reuse the `LiveSplitGeometrySpec`/`ceildiv_uop` pattern from
  `flash_decode_attention_spec.py` to make the kernel's KV extent a `UOp` rather than a Python int.
- **Gate**: correctness across a few concrete `start_pos` values realized through the *same* symbolic-shaped
  kernel (proving the symbolic path isn't just re-specializing per shape); perf should be close to the
  concrete-KV numbers from Step 3 (symbolic codegen historically costs ~3× per the 2026-06-20 doc's finding
  for the *existing* SDPA path — verify the new kernel doesn't inherit that penalty, since a hand-authored
  custom kernel controls its own codegen and should not have the same symbolic-overhead structure as the
  ordinary-op SDPA path).
- **Stop/fallback**: given `be68175ba` already defaults concrete-KV on, this step's urgency is lower than the
  2026-06-20 doc assumed — if it proves difficult, it's reasonable to ship a concrete-KV-only flash-prefill
  kernel first (Step 5) and treat symbolic support as a later increment, not a blocker.

### Step 5 — Wire into `model.py` attention (route-agnostic: both fp16 and packed routes)
- **Build**: add a fourth branch (or replace branch 2, the concrete TC-attn path) at `model.py:577-598`,
  gated the same way the other hot routes are (env-flag default-off initially, promotable later; register
  with `extra/qk/route_manifest.py` per the pure-search-guard note in 1.5). Must not regress the
  already-working decode flash route (T==1 path, untouched) or the packed-WMMA FFN routes (orthogonal —
  attention and FFN are separate route families per `pure_search_guard.py`'s `HOT_FAMILIES`).
- **Gate**: whole-model `generate()` correctness (greedy-exact / rel RMSE / dNLL per the repo's standard
  gate discipline referenced throughout `docs/pure-machine-search-roadmap.md`), plus per-layer `DEBUG=2`
  attention-kernel time at realistic contexts (512/1024/2048/4096) compared against both the SDPA/TC-attn
  baseline (must beat it) and the llama flash-on target (~17-23ms/layer — the actual finish line).
- **Stop/fallback**: if wiring in produces a whole-model regression (e.g. via JIT capture interaction, or a
  route-selection conflict with the existing branches), don't force it live — keep it env-gated/default-off
  until the interaction is understood, same as the decode flash route's `DECODE_LIVE_SPLIT` gating pattern.

### Step 6 — Whole-model measurement vs llama
- **Build**: none — pure measurement, using the repo's existing prefill authority harnesses
  (`qk_prefill_whole_synced.py` per `pure-machine-search-roadmap.md:25`) for same-process synced,
  interleaved, clock-pinned comparison (the "iron law" measurement discipline cited throughout the archived
  docs — never cross-process clock comparison, never nosync realize loops).
- **Gate/finish line**: attention share of the whole pass at pp4096-scale drops from "dominant cost at long
  context" (current materializing-path behavior) to roughly llama's ~15-17% share, i.e. per-layer attention
  cost lands near llama's ~17-23ms/layer rather than growing unboundedly with KV via score materialization.
- **Decision point**: this is the step that answers whether the multi-week kernel investment paid off. If it
  lands close to llama parity → ship, promote, document. If it lands as a real-but-modest win (e.g. 1.3-1.5×
  over the current materializing path but still short of llama) → still bankable as a genuine improvement,
  worth shipping default-off-then-promoted like the other hot routes, just not the full "prize."

### Where the feasibility spike's result feeds in
The parallel worktree spike (`.claude/worktrees/agent-afe31d62421beb2b0`) is, by its stated purpose, most
likely attempting exactly Step 0 (or Step 0+1) above — a minimal LDS-reuse kernel and its beat-SDPA gate. Its
result should be read as the **actual Step-0/1 gate outcome** for this plan, not re-derived: if it passes,
this plan resumes at Step 2; if it fails, this plan should not proceed past Step 0 without first
understanding exactly why (was it a repeat of the 2026-06-17 no-reuse trap, or a different, more fixable
issue?).

### Honest effort/risk estimate
- **Steps 0-1** (prove reuse concept + GQA): if the spike's minimal kernel already exists and passes, this is
  largely already done; if it needs building from scratch, low-to-mid effort (days), **high uncertainty** —
  this is the step that has failed once already (2026-06-17) and is where "correct but not fast" (the decode
  kernel's own fate — 60-68% of oracle per the roadmap doc) is the *optimistic* historical outcome, not the
  pessimistic one.
- **Steps 2-4** (scale to real shapes, causal, symbolic): mid effort (days), moderate risk — mostly careful
  kernel-authoring and indexing correctness, following patterns that already exist and work in the decode
  kernel.
- **Step 5** (WMMA, folded into whichever step needs the FLOP/s): this is the wildcard — no in-repo reference
  for attention-shaped WMMA, hand-authored on a BEAM-hangs backend. Could be anywhere from "not needed at all"
  (if scalar dot + good LDS reuse already beats SDPA sufficiently, as decode's own history suggests might be
  the pragmatic ceiling) to "multi-week open-ended kernel-authoring project" if the FLOP/s really is required
  to hit the llama-parity target.
- **Step 5-wire-in + Step 6**: low-to-mid effort once the kernel itself works, but carries real integration
  risk given how many existing hot routes (`pure_search_guard.py`'s `HOT_FAMILIES`) already share this
  codepath.
- **Overall**: **HIGH risk, multi-day-to-multi-week effort**, with a real and repeat-precedented chance of
  ending in "banked, correct, not performant enough to ship" (the exact outcome twice already in this
  project's history — 2026-06-17 flash-prefill, and the decode block-tile's own "correct-not-fast" status).
  The single highest-leverage risk-reduction move is exactly what Step 0 is designed to be: get an honest
  `DEBUG=2` verdict on the *smallest possible* LDS-reuse kernel before investing in GQA/causal/WMMA/wiring —
  which is precisely what the parallel feasibility spike is already doing.
