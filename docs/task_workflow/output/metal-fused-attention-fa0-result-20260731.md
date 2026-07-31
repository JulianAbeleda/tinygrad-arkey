# FA0 result — are the six attention Ops renderer-neutral, or welded to AMDISARenderer?

Date: 2026-07-31

Status: **FA0 complete. Verdict PARTIAL.** Compile-only, no GPU, read-only (no file changed except this
report). Answers §4 of `docs/task_workflow/input/metal-fused-attention-port-scope-20260731.md`. Does not
authorize promotion to `dev`/`master`; does not authorize starting FA2.

## 1. Verdict

**PARTIAL.** Of the six Ops, four lower to ordinary, renderer-neutral UOps as the module's docstring
claims. Two do not: one is structurally welded (physical-register aliasing an ISA-only concept), the
other is neutral in its arithmetic core but its *default* lowering path emits AMD-ISA-only markers that
require new per-renderer pattern matchers, not a registration.

| Op | verdict | why |
| --- | --- | --- |
| `AMD_PACKED_FRAGMENT_LOAD` | **NEUTRAL** | lowers to plain `STACK` of `INDEX`/`LOAD`; reused byte-for-byte by HIP |
| `AMD_ROW_SOFTMAX_SLOT` | **NEUTRAL** | pure tuple projection, no ISA content |
| `AMD_PV_C_LANE` | **NEUTRAL** | pure vector `GEP`, no ISA content |
| StateHandle phase publication | **NEUTRAL** | ordinary buffer `INDEX`/`LOAD`/`STORE`, no physical registers |
| `AMD_ROW_SOFTMAX_REPACK` | **PARTIAL** | row-softmax math is ordinary UOps; default path emits AMD-only markers + a hardcoded non-portable shuffle intrinsic name |
| `AMD_ATTENTION_LOOP_STATE` | **WELDED** | resolves to a physical-VGPR-tagged sentinel; only an ISA backend can interpret it |

This **contradicts the module's own docstring** (`tinygrad/renderer/isa/amd_attention_abi.py:14-15`):

> descriptor -> ordinary UOps, before instruction selection sees anything AMD-specific

That is true for four of six Ops but false for `AMD_ATTENTION_LOOP_STATE` outright, and false for
`AMD_ROW_SOFTMAX_REPACK` under the mode its own matcher list actually invokes by default. It also
**contradicts the scope packet's own §3.2 table**, which guessed `AMD_ATTENTION_LOOP_STATE` was "likely
neutral (flash state)" — it is the most welded of the six.

## 2. Method

Read `amd_attention_abi.py` in full (425 lines) and traced each Op's lowering function to its emitted
UOp form. Rather than building a synthetic probe, I used the strongest available oracle already in the
tree: `tinygrad/renderer/cstyle.py`'s `HIPRenderer`, a real, working, non-ISA C-style source renderer
that the module's own docstring says binds these same expansions
(`tinygrad/renderer/isa/amd_attention_abi.py:20-25`). Where HIP reuses a function unmodified, that is
direct proof of portability. Where HIP had to write its own separate wrapper or its own separate
implementation, that is direct proof the raw form is not renderer-neutral, and shows exactly what shape
of new code a second renderer needs. `MetalRenderer` (`tinygrad/renderer/cstyle.py:479-566`) was read to
confirm it currently binds **none** of `native_repack_matcher` / `native_state_lane_matcher` /
`native_loop_fragment_matcher` / `native_loop_state_matcher` — those attributes are set only on
`HIPRenderer` (`cstyle.py:618-633`, gated on `target.arch.split(":")[0] == "gfx1100"`) and on
`AMDISARenderer` (`tinygrad/renderer/isa/amd.py:2411-2415`). Metal today has no binding surface at all,
confirming the packet's claim in §3.2.

## 3. Per-Op evidence

### 3.1 `AMD_PACKED_FRAGMENT_LOAD` — NEUTRAL

`expand_loop_fragment` (`amd_attention_abi.py:98-187`) ends with

```python
return UOp(Ops.STACK,dtypes.half.vec(16),tuple(owner.index(off).load() for off in offs),
  tag=("amd_gfx1100_fragment_load_hd128_loop_v1",role,block,x.arg,*x.src))
```

(`amd_attention_abi.py:186-187`) — an ordinary vector-pack of ordinary `INDEX`/`LOAD` UOps. The `tag` is
inert provenance metadata for a *later, separate* AMD:ISA-only pass (`native_fragment_opaque_matcher`,
wired only onto `AMDISARenderer` at `amd.py:2411`); a renderer that ignores tags still gets a correct
`STACK` of loads.

Proof of portability: `tinygrad/renderer/cstyle.py:152-154`

```python
def _hip_expand_loop_fragment(x:UOp) -> UOp:
  from tinygrad.renderer.isa.amd import expand_loop_fragment
  return expand_loop_fragment(x)
```

HIP calls the identical function with zero modification. This is the strongest possible evidence: not
"the form looks portable," but "a second, non-ISA renderer already consumes this exact output."

### 3.2 `AMD_ROW_SOFTMAX_SLOT` — NEUTRAL

`amd_attention_abi.py:395`:

```python
(UPat(Ops.AMD_ROW_SOFTMAX_SLOT, src=(UPat(Ops.TUPLE, name="owner"),), name="x"), lambda x,owner: owner.src[x.arg.slot]),
```

Pure indexing into an `Ops.TUPLE`. No ISA content of any kind.

### 3.3 `AMD_PV_C_LANE` — NEUTRAL

`lower_native_pv_c_lane` (`amd_attention_abi.py:400-405`):

```python
def lower_native_pv_c_lane(x:UOp) -> UOp:
  x.arg.validate()
  e = x.arg.element
  if x.src[0].dtype != dtypes.float.vec(8) or not 0 <= e < 8:
    raise ValueError("invalid native PV-C lane projection")
  return x.src[0].gep(e)
```

`return x.src[0].gep(e)` is an ordinary vector-element `GEP`. Renderable by any C-style backend as
`x[e]`. No ISA content.

### 3.4 StateHandle phase publication — NEUTRAL

`lower_state_phase_transfer` (`amd_attention_abi.py:333-374`) produces only ordinary `INDEX`/`LOAD`/
`STORE` against a `storage` buffer UOp with `lane`/`element_offset`/`lane_stride` arithmetic, e.g.
(`amd_attention_abi.py:343-345`):

```python
base=handle.lane.alu(Ops.MUL,UOp.const(dtypes.weakint,handle.lane_stride)).alu(Ops.ADD,UOp.const(dtypes.weakint,handle.element_offset+element))
owner=handle.storage if len(x.src)==2 else handle.storage.after(x.src[2])
return owner.index(base).load()
```

No physical registers, no ISA-only Ops. This is the same mechanism a source renderer already uses for
any indexed buffer access.

**Note directly relevant to `AMD_ATTENTION_LOOP_STATE` below**: `amd_register_contracts.py:57-62`
records that this exact StateHandle path was tried, on AMD, as a replacement for the physical-register
loop state and *measured a negative result* (`docs/shared-attention-phase-lds-negative-result-20260724.md`,
"+2048 B LDS, 197 VGPRs, zero registers freed... this hard alias had no owner to transfer from"). That
is a performance rejection on AMD hardware, not a portability objection — the neutral mechanism exists,
compiles, and was measured; it was just worse than AMD's physical-register form *for AMD*.

### 3.5 `AMD_ROW_SOFTMAX_REPACK` — PARTIAL

The row-max/row-sum/softmax/LDS-publish/reload arithmetic in `expand_native_row_softmax_repack`
(`amd_attention_abi.py:189-326`) is ordinary UOps: `ALU`, `DEFINE_LOCAL`, `INDEX`, `LOAD`, `STORE`,
`BARRIER`. That part is portable in form — Metal has threadgroup memory and can express all of it.

Two things inside it are not:

**(a) A hardcoded, non-portable cross-lane-shuffle marker.** `amd_attention_abi.py:269-271` and
`:277-279`:

```python
for mask in x.arg.xor_masks:
  addr = lane_hw.alu(Ops.XOR, UOp.const(dtypes.int, mask)).alu(Ops.MUL, UOp.const(dtypes.int, 4))
  row_max = row_max.alu(Ops.MAX, UOp(Ops.CUSTOMI, dtypes.float, (addr, row_max), "bpermute"))
```

`"bpermute"` is a bare placeholder string, not renderable C/Metal syntax on its own — `_render_arg_format`
(`cstyle.py:12-21`) would emit the literal text `bpermute(...)`, which does not compile anywhere. It only
becomes real code via a **renderer-specific rewrite**: `hip_native_repack_pm` (`cstyle.py:192-198`)

```python
(UPat(Ops.CUSTOMI, name="x"), lambda x: x.replace(arg=_HIP_BPERMUTE_F32)
  if x.arg == "bpermute" and x.dtype == dtypes.float else None),
```

turns it into `__builtin_amdgcn_ds_bpermute(...)`. Crucially, the codebase already has a **portable**
abstraction for exactly this — `CStyleLanguage.warp_shfl_xor` (`cstyle.py:257`, documented as "None
means this target cannot express it") — and `MetalRenderer` already implements it, differently, at
`cstyle.py:520`:

```python
warp_shfl_xor = staticmethod(lambda val, offset, lane: UOp(Ops.CUSTOMI, val.dtype, (val,), arg=f"simd_shuffle_xor({{0}}, {offset})"))
```

`expand_native_row_softmax_repack` does not call `ctx.warp_shfl_xor` — it bypasses that hook and hand-rolls
an AMD-addressing-convention computation (`addr = lane XOR mask, ×4` — a byte address, matching
`ds_bpermute`'s calling convention) baked directly into the descriptor's expansion. Metal's own
`warp_shfl_xor` takes a constant XOR mask and ignores the source-lane address entirely (`simd_shuffle_xor`
resolves the source lane in hardware). These are two different calling conventions carried in the same
CUSTOMI src shape `(addr, value)`; a Metal binding cannot just add a rewrite rule keyed on the string
`"bpermute"` — it must either discard the AMD-style `addr` operand and substitute `simd_shuffle_xor(value,
mask)`, or (better) the descriptor's expansion should call `ctx.warp_shfl_xor` instead of hardcoding
`"bpermute"` so each renderer's own hook decides the addressing. Either way this is new pattern-matcher
work, not a registration.

**(b) The default lowering path emits AMD-ISA-only physical-register markers.** The module's own matcher
list invokes the function with its default argument:
`(UPat(Ops.AMD_ROW_SOFTMAX_REPACK, name="x"), expand_native_row_softmax_repack)` (`amd_attention_abi.py:394`),
i.e. `native_state=True` (`amd_attention_abi.py:189`, default). Under that mode
(`amd_attention_abi.py:291-296`):

```python
if stateful and native_state:
  mw = UOp(Ops.CUSTOMI, dtypes.void, (new_m,), arg=("amd_gfx1100_row_state_write_v1", state_owner, "m", e))
  lw = UOp(Ops.CUSTOMI, dtypes.void, (new_l,), arg=("amd_gfx1100_row_state_write_v1", state_owner, "l", e))
  aw = UOp(Ops.CUSTOMI, dtypes.void, (alpha,), arg=("amd_gfx1100_row_state_write_v1", state_owner, "alpha", e))
```

These `CUSTOMI` nodes carry a **tuple** `arg`, not a `str.format` template. Generic CUSTOMI rendering
(`_render_arg_format`, `cstyle.py:12-21`) calls `x.arg.format(...)` — a tuple has no `.format` method, so
this form cannot reach any source renderer's text output at all. It is resolved only by
`isel_customi` in `amd.py:480-489`, which turns it into a physical-register `_pin`
(`amd.py:487`: `tag=_pin(base, e)`) — i.e., real instruction selection for an ISA backend, contradicting
"before instruction selection sees anything AMD-specific" for this path.

HIP avoids this branch entirely: `_hip_expand_native_row_softmax` (`cstyle.py:139-141`)

```python
def _hip_expand_native_row_softmax(ctx, x:UOp) -> UOp:
  from tinygrad.renderer.isa.amd import expand_native_row_softmax_repack
  return expand_native_row_softmax_repack(ctx,x,native_state=False)
```

explicitly passes `native_state=False`, which routes through the *other*, ordinary-list branch
(`amd_attention_abi.py:284`, `:323-326`) — no register markers, just `STACK` of plain floats. This is
proof the neutral path exists and works, but it is **not** the path this file's own `native_repack_matcher`
wires by default — a second renderer must know to ask for it explicitly, exactly as HIP does.

**Net for this Op**: portable core, but binding it to a new renderer requires writing the HIP-shaped glue
(an explicit `native_state=False` call, plus a `warp_shfl_xor`-based rewrite of the bpermute marker) —
new pattern-matcher code following an existing precedent, not a bare registration.

### 3.6 `AMD_ATTENTION_LOOP_STATE` — WELDED

`lower_amd_attention_loop_state` (`amd_attention_abi.py:407-417`):

```python
def lower_amd_attention_loop_state(x:UOp) -> UOp:
  ...
  x.arg.validate(); base=AMD_ATTENTION_LOOP_STATE.base(x.arg.role, x.arg.block if x.arg.role=="acc" else 0)
  if x.arg.access in {"read","final_read"}:
    return _fixed_alias(base,x.arg.lane,dtypes.float)
  store=x.src[0]
  if store.op is not Ops.STORE or len(store.src)<2: raise ValueError("AMD attention loop-state write requires one STORE")
  return UOp(Ops.CUSTOMI,dtypes.void,(store.src[1],),arg=("amd_gfx1100_attention_loop_state_write_v1",x.arg.role,x.arg.block,x.arg.lane))
```

The read path returns `_fixed_alias(base, lane, dtypes.float)`, defined in
`amd_physical_regs.py:20-23`:

```python
def _fixed_alias(base:int, i:int, dtype, *deps:UOp) -> UOp:
  fixed = UOp(Ops.NOOP, dtype, tag=(FixedRegisterUse(f"v{base+i}", base+i),))
  return fixed if not deps else UOp(Ops.NOOP, dtype, src=(fixed,) + deps)
```

This is a `UOp` whose entire payload is a **physical VGPR number** (`FixedRegisterUse`, imported from
`tinygrad.renderer.isa`, `amd_physical_regs.py:16-17`). It has no source expression to render — nothing
to compute, nothing to index. A source (C-style) renderer has no way to say "this value already lives in
hardware register v72" from emitted source text; register pinning is an assembly/ISA-backend-only
concept, exactly the class of thing the task's method flagged ("register-file... assumptions a source
renderer cannot express"). The write path returns a tuple-arg `CUSTOMI`
(`arg=("amd_gfx1100_attention_loop_state_write_v1",...)`) with the same non-renderable-tuple problem as
§3.5(b) above, resolved only by `isel_customi` (`amd.py:474-479`), again to a physical-register `_pin`.

`amd_register_contracts.py:17-49` documents `AMDAttentionLoopStateMap` as "a privileged operation: it
hands out *physical* register numbers that bypass the register allocator entirely," and explicitly scopes
its callers to `amd_attention_abi.lower_amd_attention_loop_state`, `amd.py`'s isel, and `amd_wmma_residency`
— "Generic instruction selection must not call it, and nothing outside `DEV=AMD:ISA` may call it at all."
That is the module's own contract stating this Op's lowering is AMD:ISA-only.

**Structural confirmation**: `HIPRenderer` does not use `lower_amd_attention_loop_state` or
`native_loop_state_matcher` for this Op at all. It registers a wholly separate, independently written
expansion, `_hip_expand_attention_loop_state` (`cstyle.py:143-150`, wired at `cstyle.py:627`):

```python
def _hip_expand_attention_loop_state(x:UOp) -> UOp:
  from tinygrad.uop.ops import AMDLoopStateSpec
  if not isinstance(x.arg, AMDLoopStateSpec): raise ValueError("HIP attention loop state is missing its typed ABI")
  x.arg.validate()
  if x.arg.access in {"init","write"}: return x.src[0]
  reg=x.src[0]; offset=x.arg.block*8+x.arg.lane if x.arg.role=="acc" else x.arg.lane
  addr=reg.after(*x.src[1:]).index(UOp.const(dtypes.weakint,offset))
  return addr.load()
```

This is an ordinary buffer-indexed load/store, not a physical-register alias — but it is a **third**,
independent implementation of loop-carried state (distinct from both `lower_amd_attention_loop_state`'s
physical-register form and the StateHandle mechanism of §3.4). HIP needed a genuinely new emitter
function for this Op; `amd_attention_abi.py` supplies no renderer-neutral form of it at all. This is
direct proof, not inference: the one other non-ISA renderer already wired to this ABI could not reuse
this file's own lowering and had to write new code — precisely the "new Metal emitter" scenario
§4 of the scope packet asks FA0 to rule in or out, and for this Op it is ruled **in**.

## 4. What each side needs

**NEUTRAL (`AMD_PACKED_FRAGMENT_LOAD`, `AMD_ROW_SOFTMAX_SLOT`, `AMD_PV_C_LANE`, StateHandle phase
publication)**: a registration. `MetalRenderer` would set `self.native_loop_fragment_matcher =
native_loop_fragment_matcher`, and wire the `AMD_ROW_SOFTMAX_SLOT`/`AMD_PV_C_LANE`/StateHandle clauses of
`native_repack_matcher`/`native_state_lane_matcher` (`amd_attention_abi.py:391-397`, `:419-421`) the same
way `AMDISARenderer` does at `amd.py:2413-2414` — no new matcher logic required for these four.

**`AMD_ROW_SOFTMAX_REPACK`**: needs new pattern-matcher code, following the HIP precedent exactly:
(1) a Metal-side wrapper calling `expand_native_row_softmax_repack(ctx, x, native_state=False)`, mirroring
`_hip_expand_native_row_softmax` (`cstyle.py:139-141`); (2) a rewrite of the `"bpermute"` marker into a
call through Metal's own `warp_shfl_xor` (`cstyle.py:520`, `simd_shuffle_xor`) rather than HIP's
`__builtin_amdgcn_ds_bpermute` rewrite (`cstyle.py:195-197`), accounting for the differing addressing
convention noted in §3.5(a).

**`AMD_ATTENTION_LOOP_STATE`**: needs a new emitter, not a registration. The two candidates already in
the tree are (a) write a Metal-specific buffer-indexed expansion analogous to
`_hip_expand_attention_loop_state` (`cstyle.py:143-150`) — small, precedented, but a third independent
implementation of the same state; or (b) route this Op's semantics through the already-neutral StateHandle
mechanism (§3.4) instead — noting that path was tried for this exact state on AMD and rejected there on
*measured performance* grounds, not correctness (`amd_register_contracts.py:57-62`), so it is not
disqualified for Metal a priori.

## 5. What this changes about the port's shape

The scope packet's framing in §4 posed a binary: NEUTRAL (port is descriptor + registration, same shape
as the precontract work) or WELDED (new Metal emitter behind `_PREFILL_EMITTERS`). The real answer is
mixed at the Op level. Four of six Ops support the "registration" framing outright. `AMD_ATTENTION_LOOP_STATE`
requires the "new emitter" framing on its own — it is not avoidable by any registration, and HIP's own
history proves it. `AMD_ROW_SOFTMAX_REPACK` sits in between: its output is close to neutral, but the
module's default wiring does not produce that form, and reaching it needs new code of a kind and size
already demonstrated by HIP's existing wrappers (`cstyle.py:139-198`) — i.e., FA2's estimate should budget
for "HIP-sized" glue on this Op and a genuinely new emitter on loop state, not zero new lowering code.

## 6. What was not established

- Whether Metal's `simd_shuffle_xor` and AMD's `ds_bpermute` are numerically equivalent for the specific
  `xor_masks` reduction ladder used here (`x.arg.xor_masks`, `AMDRowSoftmaxRepackSpec`) was not checked —
  only that the calling conventions differ and a rewrite is required. FA2 must verify the reduction
  ladder's correctness on Metal's convention, not just its renderability.
- Whether option (b) in §4 (routing `AMD_ATTENTION_LOOP_STATE` through StateHandle) is performance-viable
  on Metal was not measured — no GPU was used, per the compile-only constraint. Only that it compiles and
  was previously measured (unfavorably) on AMD is established.
- `AMDRowSoftmaxRepackSpec`'s other descriptor fields (`validity_mode`, `dynamic_kv_v1`, `mode`) were read
  only far enough to confirm the control flow quoted above; a full enumeration of every mode combination's
  output was not performed.
