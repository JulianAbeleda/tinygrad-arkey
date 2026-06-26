# Codex task prompt — REG-store devectorize pass (unblock vectorized cache loads in the generated decode tile)

Copy everything below the line into Codex. Full rationale + file:line map: `docs/decode-reg-store-devec-codegen-scope.md`.

---

You are in the tinygrad fork at `/home/ubuntu/tinygrad-arkey` (AMD gfx1100; hardware present, run real jobs
with `DEV=AMD JIT=1 PYTHONPATH=.`).

## Objective

Add a small, **env-gated, default-off** AMD codegen pass that lets a hand-built `custom_kernel` coalesce its
GLOBAL cache loads (by marking the contiguous lane-local axis `AxisType.UPCAST`) **while keeping the REG
accumulator store scalar**. Today, UPCAST on that axis correctly vectorizes the cache load but ALSO
vectorizes the REG accumulator store, which the renderer emits as illegal C
(`make_float4(...) = make_float4(...)` → "expression is not assignable"). Fix the REG store, not the load.
Do NOT change any attention algorithm/layout, and do NOT author `Ops.PTRCAT` (the UOp verifier rejects it).

## Verified root cause

The accumulator is a scalar-element `DEFINE_REG<float.ptr(R,REG)>` placeholder. The expander special-cases
a REG `Ops.INDEX` into one `Ops.STACK` of per-lane scalar `INDEX(acc,CONST_k)`
(`tinygrad/codegen/late/expander.py:59-67`). `pm_add_loads` wraps each as `LOAD(INDEX)`. No later pass
re-scalarizes a `STORE` whose target is that `STACK` (the load-strip rule `devectorizer.py:370` needs a
direct LOAD; `no_vectorized_*` `devectorizer.py:252-278` don't apply; `split_load_store` leaves REG unfolded
`devectorizer.py:171-172`). So it survives to the renderer: `cstyle.py:71` STORE rule + REG fall-through
`cstyle.py:210` + STACK literal rule `cstyle.py:63-65` → `make_float4(...) = make_float4(...)`. The GLOBAL
cache load is already coalesced correctly to `*((float4*)(...))` (`cstyle.py:208`) — leave it alone.

## Implement

1. New module `extra/qk_reg_store_devec.py` (mirror `extra/qk_fdot2_lowering.py` structure):
   ```python
   from tinygrad.uop.ops import Ops, UOp, AddrSpace, PatternMatcher, UPat
   def devec_reg_store(store, tgt, val):
     ptrs = []
     for s in tgt.src:
       idx = s.src[0] if s.op is Ops.LOAD else s            # strip the pm_add_loads LOAD -> scalar ptr INDEX
       if idx.op is not Ops.INDEX or idx.src[0].addrspace != AddrSpace.REG: return None  # REG-only guard
       ptrs.append(idx)
     return UOp.group(*[p.store(val.gep(i)) for i, p in enumerate(ptrs)])   # per-lane SCALAR stores
   pm_reg_store_devec = PatternMatcher([
     (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val")), name="store"), devec_reg_store),
   ])
   ```
   (Verify the exact import path for `PatternMatcher`/`UPat` against `extra/qk_fdot2_lowering.py`.)
2. Hook in `tinygrad/codegen/__init__.py`, immediately AFTER the devectorize `graph_rewrite` (`:110-111`),
   exactly like the V_DOT2 block at `:112-114`:
   ```python
   if getenv("REG_STORE_DEVEC") and ren.target.device == "AMD":
     from extra.qk_reg_store_devec import pm_reg_store_devec
     sink = graph_rewrite(sink, pm_reg_store_devec, name="reg store devec")
   ```
3. Add `getenv("REG_STORE_DEVEC")` to the `to_program` cache key (`tinygrad/codegen/__init__.py:255`).

## Constraints (hard)

- Match the STORE **target** only (`src[0]` is `Ops.STACK` whose elements index `AddrSpace.REG`). Never
  touch GLOBAL/LOCAL vec stores (target is a `CAST<vec-ptr>`, not a STACK) or the coalesced GLOBAL cache
  load. Keep `src[1]` (value) vec; slice per-lane with `.gep(i)`.
- Default-off: env-gated + in the cache key. Shipped default route + q4k GEMVs must be byte-for-byte
  unchanged (run a default-route sanity check with `REG_STORE_DEVEC` unset).
- Do NOT author `Ops.PTRCAT`. Do NOT edit `tinygrad/runtime/autogen/**`. Do NOT add an attention layout.

## Proof gate (do this first, before touching the in-model tile)

```
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_cache_identity_index_gate.py
```
Required: the `dynamic_v_sum_upcast_5d` row flips FAIL → **PASS** (assert
`by_name["dynamic_v_sum_upcast_5d"]["pass"] is True`); `static_v_scalar_5d`, `static_v_upcast_5d`,
`dynamic_v_sum_scalar_5d`, `k_upcast_lds_5d` stay PASS. `ptr_vec_v_5d` is OUT OF SCOPE (a separate
PTRCAT-spec gap; do not try to fix it, and do not let the top-line verdict mask the `dynamic_v_sum_upcast_5d`
result — read/assert the per-row pass). If the gate needs a `REG_STORE_DEVEC`-aware verdict, add one.

## End-to-end (after the cache gate flips)

Mark the V/PV (and K-stage) contiguous axis `AxisType.UPCAST` in
`flash_fused_xlane_score_pv_tile_whole_cache_kernel` (`extra/qk_flash_decode.py:841`) under the same env,
then run, asserting the pass-strings:
```
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_microgate.py   # FUSED_XLANE_SCORE_PV_MICROGATE_PASS
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py  # FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_attention_isa_diff_gate.py                    # ISA_DIFF_PINNED; generated global_load_d16>0 or dwordx4>0
DEV=AMD JIT=1 DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 V_DOT2_LOWERING=1 REG_STORE_DEVEC=1 \
  PYTHONPATH=. python3 extra/qk_decode_runtime_overhead.py                                                         # tok/s vs baseline 82.4/103.5/101.8/94.6
```

## Deliverable / labels

- Success: cache gate `dynamic_v_sum_upcast_5d` PASS, microgate + route gate PASS, ISA diff generated
  `global_load_d16 > 0`, no materialization, owned route absent; report W==D before/after and ISA markers
  before/after. (W==D may still need Phase 2 block-tiling — that is a separate scope.)
- If the pass works in the gate but the in-model tile still won't coalesce some load shape:
  `SEARCH_BLOCKED_BY_CODEGEN__LOAD_COALESCER_CUSTOM_KERNEL_GAP` (record the exact uncoalesced INDEX).
- If vec loads land but W==D stays slow + GPU-bound: `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING`.

Commit per step with gate verdicts in the message; bracketed-prefix required by the repo hook
(e.g. `[codegen] ...`).
