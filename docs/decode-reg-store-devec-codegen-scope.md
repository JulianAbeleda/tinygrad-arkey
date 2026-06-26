# REG-store devectorize: env-gated pass so the decode tile can vectorize cache loads — scope (2026-06-26)

Continues `docs/decode-generated-tile-codegen-scope.md` Phase 1. The generated decode tile can be made to
coalesce its cache loads by marking the contiguous d/e axis `AxisType.UPCAST`, but that same UPCAST also
vectorizes the REG accumulator store, which the renderer emits as illegal C
(`make_float4(...) = make_float4(...)` → "expression is not assignable"). This scope defines a small,
env-gated codegen pass that lets the GLOBAL cache load coalesce while keeping the REG accumulator store
scalar. Do NOT implement until asked.

Diagnosis label being resolved: `SEARCH_BLOCKED_BY_CODEGEN__DYNAMIC_UPCAST_REG_STORE_AND_PTRCAT_PLACEMENT`.

## Root cause (verified, file:line)

1. The accumulator is a scalar-element REG buffer `DEFINE_REG<float.ptr(R, REG)>` (`UOp.placeholder`,
   `tinygrad/uop/ops.py:1057-1060`).
2. The expander special-cases a non-ptr `Ops.INDEX` over an `AddrSpace.REG` buffer
   (`tinygrad/codegen/late/expander.py:59-67`): it expands `acc[dd]` (dd UPCAST) into **one
   `Ops.STACK` of per-lane scalar `INDEX(acc, CONST_k)`** ("to avoid a VECTORIZE of REG pointers the
   devectorizer can't resolve"). The GLOBAL cache load keeps a **single vec `INDEX`** (`expander.py:50-57`)
   which `split_load_store` coalesces into a legal `LOAD(CAST<float.vec(4).ptr>)` — that is why
   `static_v_upcast_5d` and `k_upcast_lds_5d` PASS and only the REG-accumulator store FAILs.
3. `pm_add_loads` (`devectorizer.py:361-372`) wraps each STACK element as `LOAD(INDEX(acc,k))`. The
   load-strip-on-store rule (`devectorizer.py:370`) only matches `STORE` whose `src[0]` is *directly* a
   `LOAD`; here `src[0]` is `STACK(LOAD,…)`, so it is skipped. `devectorize_buf_and_index`
   (`devectorizer.py:271-278`) cannot touch it (`no_vectorized_buf` is a no-op on a scalar-element REG;
   `no_vectorized_index` needs a single INDEX on a CAST of the REG buf). `split_load_store` deliberately
   leaves REG unfolded (`devectorizer.py:171-172 elif buf.addrspace == AddrSpace.REG: pass`). The
   STACK-targeted REG store has **no rule** (there are rules for GEP/PTRCAT targets at
   `devectorizer.py:119-134`, none for a bare STACK target) → it survives to render.
4. Render: STORE rule `cstyle.py:71` emits `render_access(bidx) = var` with no lvalue check; for a REG
   `bidx` it falls through to `cstyle.py:210` and returns the STACK literal verbatim; the STACK rule
   `cstyle.py:63-65` makes it `make_float4(acc0,acc1,acc2,acc3)` → an rvalue on the LHS → compile error.

The separate `ptr_vec_v_5d` FAIL is **not this bug**: hand-authored `Ops.PTRCAT` is rejected by
`spec_tensor` (no PTRCAT rule; `tinygrad/uop/spec.py:43`, the only PTRCAT rule is in `spec_full`
`:256-257`, used only at SPEC>1). The pass must NOT author PTRCAT — it must let the existing devectorizer
build the legal `CAST(INDEX)`-of-vec load (`spec.py:104 .or_casted().load()`).

## The fix

### 1. Where it hooks
A new env-gated `graph_rewrite` in `tinygrad/codegen/__init__.py`, **immediately after the devectorize
stage** (`:110-111`), modeled exactly on the V_DOT2 hook (`:112-114`):
```python
if getenv("REG_STORE_DEVEC") and ren.target.device == "AMD":
  from extra.qk_reg_store_devec import pm_reg_store_devec
  sink = graph_rewrite(sink, pm_reg_store_devec, name="reg store devec")
```
Add `getenv("REG_STORE_DEVEC")` to the `to_program` cache key (`tinygrad/codegen/__init__.py:255`). It must
run AFTER line 110 (so the GLOBAL cache load is already coalesced and the REG store has settled into its
final `STORE(STACK(LOAD(INDEX(REG))))` form) and BEFORE the new-style lowering (`:142-144`), linearize, and
render. No interaction with fdot2 (which only matches `Ops.ADD`). New module `extra/qk_reg_store_devec.py`
mirrors `extra/qk_fdot2_lowering.py`.

### 2. Exact pattern + rewrite (recon-verified)
Bad node (post-devectorize): `STORE(src0=STACK[float.vec(R)] of LOAD(INDEX(DEFINE_REG, CONST_k)),
src1=STACK[float.vec(R)] of the R scalar update values)`. Rewrite the **target only** to per-lane scalar
stores:
```python
def devec_reg_store(store, tgt, val):
  ptrs = []
  for s in tgt.src:
    idx = s.src[0] if s.op is Ops.LOAD else s            # strip the pm_add_loads LOAD -> scalar ptr INDEX
    if idx.op is not Ops.INDEX or idx.src[0].addrspace != AddrSpace.REG: return None  # REG-only guard
    ptrs.append(idx)
  return UOp.group(*[p.store(val.gep(i)) for i, p in enumerate(ptrs)])  # per-lane SCALAR stores

pm_reg_store_devec = PatternMatcher([
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val")), name="store"), devec_reg_store),
])
```
`val.gep(i)` returns the STACK value's i-th scalar src (`ops.py` gep-on-STACK shortcut); each lane renders
as a legal scalar `acc[i] = val_i;`. `UOp.group(...)` as a store replacement is the established idiom
(`devectorizer.py:126,148`).

### 3. What it must avoid
- **Only the STORE target** (`src[0]` is `Ops.STACK` whose elements index `AddrSpace.REG`). Do not touch
  GLOBAL/LOCAL vec stores (their target is a `CAST<vec-ptr>`, not a STACK) or the coalesced GLOBAL cache
  load (the win — `cstyle.py:208`).
- **Strip the LOAD wrapper** to recover the INDEX (`s.src[0] if s.op is Ops.LOAD`); storing into the LOAD
  is wrong.
- **Leave the value (`src[1]`) vec**; slice per-lane with `.gep(i)`. Do not coalesce the REG store value.
- **Never author `Ops.PTRCAT`** or a hand STACK-of-ptr-INDEXes in the AST — `spec_tensor` rejects it
  (`spec.py:43`). Let the devectorizer produce the load.
- **Default-off**: env-gated + in the cache key; the shipped default route and the q4k GEMVs unchanged.

## Minimal proof gate

Reuse `extra/qk_decode_cache_identity_index_gate.py` (run:
`DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_cache_identity_index_gate.py`).
- **Target flip:** `dynamic_v_sum_upcast_5d` FAIL → **PASS** (the load coalesces, the REG accumulator store
  stays scalar, numeric matches the NumPy reference).
- **Must stay PASS:** `static_v_scalar_5d`, `static_v_upcast_5d`, `dynamic_v_sum_scalar_5d`,
  `k_upcast_lds_5d` (the pass is a no-op on them — no STACK REG-store target).
- **Out of scope (do not try to fix here):** `ptr_vec_v_5d` — it is a separate spec gap (PTRCAT not in
  `spec_tensor`); the correct path never authors PTRCAT, so the gate's verdict chain should be read by the
  per-row `dynamic_v_sum_upcast_5d` result, not the top-line verdict (which currently reports the ptr_vec
  row first). Consider adding a `REG_STORE_DEVEC`-on row-subset verdict, or simply assert
  `by_name["dynamic_v_sum_upcast_5d"]["pass"] is True`.

## End-to-end acceptance (after the cache gate flips)

Then apply `AxisType.UPCAST` to the V/PV (and K-stage) contiguous axis in
`flash_fused_xlane_score_pv_tile_whole_cache_kernel` (`extra/qk_flash_decode.py:841`) under the same env,
and run the full harness:
```
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_cache_identity_index_gate.py          # dynamic_v_sum_upcast_5d PASS
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_microgate.py   # FUSED_XLANE_SCORE_PV_MICROGATE_PASS
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py  # FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_attention_isa_diff_gate.py                    # ISA_DIFF_PINNED; generated global_load_d16>0 (or dwordx4>0)
DEV=AMD JIT=1 DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 V_DOT2_LOWERING=1 REG_STORE_DEVEC=1 \
  PYTHONPATH=. python3 extra/qk_decode_runtime_overhead.py                                                         # tok/s vs baseline 82.4/103.5/101.8/94.6
```
Acceptance: cache gate `dynamic_v_sum_upcast_5d` PASS; microgate PASS; route gate
`FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT`; ISA diff shows generated `global_load_d16 > 0` (or
`global_load_dwordx4 > 0`); no materialization; owned route absent. The payoff signal is the W==D toward
baseline (Phase 2 block-tiling may still be needed for the rest).

## Fallback terminal labels

- If the pass works but the existing coalescer still won't vectorize the *in-model* tile's loads (some
  shape the cache gate didn't cover): `SEARCH_BLOCKED_BY_CODEGEN__LOAD_COALESCER_CUSTOM_KERNEL_GAP` — record
  the exact uncoalesced INDEX shape.
- If vector loads land (ISA diff `global_load_d16 > 0`) but W==D is still far from baseline and GPU-bound:
  `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING` — proceed to Phase 2 (block-tile + multi-warp,
  `docs/decode-generated-tile-codegen-scope.md`) and, only if needed, the scheduler scope.

## Constraints

- Default-off (env-gated + cache key); shipped default route + q4k GEMVs byte-for-byte unchanged.
- Do not revive score-broadcast (economically refuted). Do not add another attention layout (correct).
- Do not hand-edit `tinygrad/runtime/autogen/**`. Do not author `Ops.PTRCAT` in the AST.
- Bracketed-prefix commit messages (repo hook): e.g. `[codegen] ...`.

Codex prompt: `docs/decode-reg-store-devec-codex-prompt.md`.
