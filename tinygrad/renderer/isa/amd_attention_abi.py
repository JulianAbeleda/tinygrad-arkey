"""The AMD/rdna3 fused-attention typed ABI: lowering for the six bespoke AMD Ops.

The prefill fused-attention kernel does not go through generic instruction
selection alone. It carries six renderer-specific Ops whose meaning is fixed by
typed descriptors in tinygrad/uop/ops.py:

    Ops.PACKED_FRAGMENT_LOAD    (PackedFragmentLoopSpec) Q/K/V fragment addressing
    Ops.NATIVE_ROW_SOFTMAX_REPACK      (NativeRowSoftmaxRepackSpec)   QK-C -> P -> PV-A bridge
    Ops.ROW_SOFTMAX_SLOT                                    projection of the above
    Ops.AMD_PV_C_LANE               (AMDPVCLaneSpec)            PV accumulator lane view
    Ops.ATTENTION_LOOP_STATE    (LoopStateSpec)          loop-carried m/l/acc
    (plus StateHandle-based generic phase publication)

This module is the whole lowering surface for them: descriptor -> ordinary UOps,
before instruction selection sees anything AMD-specific. It exists as its own
module because these Ops are a *second system* living beside the generic RDNA3
renderer, and the honest thing is to give that system a visible boundary rather
than let it hide among the generic isel rules.

Consumers: AMDISARenderer binds the matchers below as its native_* pattern
matchers; tinygrad/renderer/cstyle.py (HIPRenderer) imports
expand_native_row_softmax_repack / expand_loop_fragment / native_repack_matcher /
native_state_lane_matcher so the HIP path lowers the same descriptors. Both
renderers are covered by the same compile-only byte-identity oracle, because they
share these expansions and can otherwise drift apart silently.

NOT here: the isel-side handling of these Ops (isel_packed_fragment,
isel_attention_output_drain, isel_attention_stats_drain, isel_attention_loop_state)
stays in amd.py, where it is entangled with the fragment allocator and the
instruction encoder.
"""
from __future__ import annotations
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Ops
from tinygrad.dtype import dtypes, AddrSpace, PtrDType
from tinygrad.helpers import getenv
from tinygrad.renderer.isa.amd_physical_regs import _fixed_alias
from tinygrad.renderer.isa.amd_register_contracts import AMD_ATTENTION_LOOP_STATE
from tinygrad.codegen.late.warp_reduce import warp_bpermute

def lower_cooperative_tile_load(x:UOp) -> UOp:
  from tinygrad.uop.ops import CooperativeTileLoadSpec
  if x.op is not Ops.COOPERATIVE_TILE_LOAD or not isinstance(x.arg, CooperativeTileLoadSpec): raise ValueError("invalid cooperative tile load")
  x.arg.validate(); owner, tile_base = x.src
  if owner.op is not Ops.PARAM or not isinstance(owner.dtype, PtrDType) or owner.ptrdtype.base is not dtypes.half: raise ValueError("cooperative tile owner must be fp16 global PARAM")
  thread=UOp.special(128, "lidx0"); tile_elements=16*128; shared=UOp(Ops.DEFINE_LOCAL,dtypes.half.ptr(tile_elements*x.arg.slots,AddrSpace.LOCAL),arg=("nv2a_shared",x.arg.phase_abi,x.arg.slots))
  pre=UOp(Ops.BARRIER,dtypes.void,(UOp.group(),),arg=("nv2a_pre_tile_barrier",x.arg.phase_abi)) if x.arg.pre_barrier else None
  stores=[]
  for i in range(16):
    idx=thread.alu(Ops.ADD,UOp.const(dtypes.weakint,i*128))
    src=owner.index(tile_base+idx).load(); src=src.after(pre) if pre is not None else src
    stores.append(shared.index(x.arg.slot_index*tile_elements+idx,ptr=True).store(src))
  barrier=UOp(Ops.BARRIER,dtypes.void,(UOp.group(*stores),),arg=("nv2a_tile_barrier",x.arg.phase_abi))
  from tinygrad.uop.ops import SharedTileOwnerSpec
  return shared.after(barrier).replace(tag=SharedTileOwnerSpec(phase_token=x.arg.phase_abi,
    loop_axis=x.arg.loop_axis, stage_generation=x.arg.stage_generation,
    end_barrier_token=x.arg.end_barrier_token, slots=x.arg.slots, slot_index=x.arg.slot_index))

def lower_cooperative_stage_begin(x:UOp) -> UOp:
  from tinygrad.uop.ops import CooperativeStageBeginSpec
  if x.op is not Ops.COOPERATIVE_STAGE_BEGIN or not isinstance(x.arg, CooperativeStageBeginSpec):
    raise ValueError("invalid cooperative stage begin")
  x.arg.validate()
  if len(x.src) != 2 or x.src[0] != x.arg.loop_axis: raise ValueError("stage begin axis mismatch")
  # The barrier is deliberately independent of lane/load predicates. Its arg is
  # the typed ordering token consumed by the staged shared-tile owner.
  return UOp(Ops.BARRIER, dtypes.void, (UOp.group(),), arg=("nv_sm120_cooperative_stage_begin_v1", x.arg.ordering_token, x.arg.loop_axis, x.arg.stage_generation))

def _shared_tile_owner(owner:UOp) -> tuple[UOp,UOp]:
  """Validate and return the single-buffer local tile with its publication edge."""
  from tinygrad.uop.ops import SharedTileOwnerSpec
  if owner.op is not Ops.AFTER or not isinstance(owner.tag, SharedTileOwnerSpec):
    raise ValueError("shared packed fragment requires tagged AFTER owner")
  owner.tag.validate()
  if len(owner.src) != 2 or owner.src[0].op is not Ops.DEFINE_LOCAL or owner.src[1].op is not Ops.BARRIER:
    raise ValueError("shared tile owner must be DEFINE_LOCAL AFTER matching BARRIER")
  local, barrier = owner.src
  if local.dtype != dtypes.half.ptr(2048*owner.tag.slots, AddrSpace.LOCAL) or barrier.arg != ("nv2a_tile_barrier", owner.tag.phase_token):
    raise ValueError("shared tile owner has invalid local tile or barrier")
  if len(barrier.src) != 1 or barrier.src[0].op not in {Ops.GROUP, Ops.STORE}:
    raise ValueError("shared tile owner barrier must publish one store group")
  return local, barrier


def drain_lane_encoding(head_dim:int, e:int, j:int, output_block_base:int) -> tuple[int, int, int]:
  """Register-level encoding of one C-fragment drain store, derived from the spec authority.

  The wave32 drain lane convention lives in exactly one place --
  ``AttentionOutputDrainSpec.drain_lane_coeffs`` -- and this function is how
  the AMD:ISA encoder consumes it, rather than restating the same 128/256/2048
  constants as shifts and immediates (which is how the HIP and ISA paths were
  free to drift apart from each other and from the declared ``address_expr``).

  Returns ``(halfwave_shift, group_row_stride, store_byte_offset)``:
    * the address VGPR is ``((halfwave << halfwave_shift) + col) * 2`` bytes,
    * plus ``gidx0 * group_row_stride`` for the grid form,
    * and the residual ``e``/``j`` terms ride in the store's byte immediate.

  ``c_col == 1`` and ``c_e == 2*c_halfwave`` are required because the encoder
  folds ``col`` in with no scale and encodes ``c_halfwave`` as a left shift.
  """
  from tinygrad.uop.ops import AttentionOutputDrainSpec
  c_e, c_half, c_j, c_col = AttentionOutputDrainSpec(head_dim=head_dim).drain_lane_coeffs
  if c_col != 1 or c_half & (c_half-1):
    raise ValueError("AMD:ISA attention drain encoder needs a unit column stride and a power-of-two halfwave stride")
  # NOTE (unresolved, pre-existing): group_row_stride is added to a BYTE address while the HIP path adds
  # 16*c_half in ELEMENTS. The value is the same integer, so this derivation is byte-identical to what the
  # encoder emitted before, but the two renderers do not agree on the grid term's unit. Nothing exercises
  # the AMD:ISA grid drain numerically, which is precisely why the divergence survived; it is recorded here
  # rather than silently "corrected" inside a knowledge-centralization change.
  return c_half.bit_length()-1, 16*c_half, (e*c_e + (j+output_block_base)*c_j) * 2

def _opaque_exact_fragment_inputs(x:UOp) -> UOp|None:
  if x.op is not Ops.WMMA or len(x.src) != 3: return None
  changed, src = False, list(x.src)
  for pos in (0,1):
    c=src[pos]
    if not (c.op is Ops.STACK and c.dtype == dtypes.half.vec(16) and len(c.src)==16 and isinstance(c.tag,tuple) and
            c.tag[:1] in {("amd_gfx1100_fragment_load_v1",),("amd_gfx1100_fragment_load_hd128_v1",),("amd_gfx1100_fragment_load_hd128_loop_v1",)} and
            all(v.op is Ops.LOAD and v.dtype==dtypes.half for v in c.src)): continue
    if c.tag[0] == "amd_gfx1100_fragment_load_hd128_loop_v1":
      from tinygrad.uop.ops import PackedFragmentLoopSpec
      _,role,hd_block,*payload=c.tag
      if payload and isinstance(payload[0], PackedFragmentLoopSpec): spec,*fragment_src=payload
      else:
        owner,lane,col,rng=payload
        spec,fragment_src=PackedFragmentLoopSpec(role=role,head_block=hd_block),[owner,lane,col,rng]
      src[pos]=UOp(Ops.PACKED_FRAGMENT_LOAD,dtypes.half.vec(spec.fragment_lanes),tuple(fragment_src),arg=spec)
      changed=True
      continue
    if c.tag[0] == "amd_gfx1100_fragment_load_hd128_v1": _,role,tile,hd_block,owner,lane,col=c.tag
    else: _,role,tile,owner,lane,col=c.tag; hd_block=None
    if role not in {"Q","K","V"} or not isinstance(tile,int) or tile not in {0,1}: raise ValueError("malformed gfx1100 fragment descriptor")
    if role == "Q" and pos != 0 or role == "K" and pos != 1 or role == "V" and pos != 1: raise ValueError("fragment role/WMMA operand mismatch")
    abi="amd_gfx1100_packed_fragment_hd128_v1" if hd_block is not None else "amd_gfx1100_packed_fragment_v1"
    arg=(abi,role,tile,hd_block) if hd_block is not None else (abi,role,tile)
    src[pos]=UOp(Ops.PACKED_FRAGMENT_LOAD,dtypes.half.vec(16),(owner,lane,col),arg=arg)
    changed=True
  return x.replace(src=tuple(src)) if changed else None

native_fragment_opaque_matcher=PatternMatcher([(UPat(Ops.WMMA,name="x"),_opaque_exact_fragment_inputs)])

def expand_loop_fragment(x:UOp) -> UOp:
  """Materialize the typed loop fragment before tensor/program verification.

  Its tag retains the owner and RANGE identity; the normal late opaque pass
  turns this back into a physical AMD carrier after index lowering.
  """
  from tinygrad.uop.ops import PackedFragmentLoopSpec, AMDMultiWaveAttentionGridSpec
  if not isinstance(x.arg, PackedFragmentLoopSpec): raise ValueError("loop fragment is malformed")
  x.arg.validate(); role,block=x.arg.role,x.arg.head_block
  shared_storage = x.arg.storage == "shared"
  if shared_storage:
    shared_local, shared_barrier = _shared_tile_owner(x.src[0])
    shared_owner = x.src[0]
  if isinstance(x.arg.grid, AMDMultiWaveAttentionGridSpec):
    if len(x.src) != 6: raise ValueError("multiwave loop fragment requires owner/lane/wave/column/range/group")
    owner,lane,wave_id,col,rng,*grid_src=x.src
  else:
    if len(x.src) not in {4,5}: raise ValueError("loop fragment is malformed")
    owner,lane,col,rng,*grid_src=x.src; wave_id=None
  if (x.arg.grid is None) != (len(grid_src)==0): raise ValueError("loop fragment grid ownership must be explicit")
  # Staged QK fragments carry a typed retirement token in their descriptor.
  # Make it a real load dependency here, before the opaque-fragment handoff;
  # retaining it only in the tag lets HIP materialize every fragment eagerly.
  stage_wait=getattr(x.arg, "stage_wait", None)
  owner=owner.after(stage_wait) if stage_wait is not None else owner
  # `128`/`2048` below were head_dim / 16*head_dim (per-token-row Hd stride / per-16-token-tile Hd
  # stride), hardcoded for Hd=128. Derived from the bound grid's own head_dim -- this is the residual
  # weld that blocked the Hd=64 numerics sweep (MMU fault: fragment addressing still assumed Hd=128
  # row strides against Hd=64-sized buffers). The grid-less branch (gbase=0, `not grid_src`) is the
  # fixed grid-less kv64_hd128_loop kernel (no head_dim kwarg exists there, per P-B2) -- hd stays the
  # literal 128 constant for that specific kernel family, not derived.
  hd = x.arg.grid.head_dim if x.arg.grid is not None else 128
  if shared_storage: grid_src=[]
  if not grid_src: gbase=UOp.const(dtypes.weakint,0)
  if shared_storage: gbase=x.src[0].tag.slot_index*2048
  elif isinstance(x.arg.grid, AMDMultiWaveAttentionGridSpec):
    grid,group=x.arg.grid,grid_src[0]
    kv_head,q_tile=group//grid.q_tiles,group%grid.q_tiles
    gbase=((kv_head*grid.waves_per_group+wave_id)*(grid.q_tokens*hd)+q_tile*16*hd) if role=="Q" else kv_head*(grid.kv_tokens*hd)
  elif role=="Q": gbase=grid_src[0]*(16*hd)
  else:
    grid=x.arg.grid
    gbase=(grid_src[0]//(grid.q_tiles*grid.group_ratio))*(grid.kv_tokens*hd)
  # OOB LOAD GUARD (correctness fix): K/V buffers carry no padding -- kernels.py's `sizes=` check on
  # amd_gfx1100_q16_grid_hd128_loop_attention requires the buffer to hold EXACTLY kv_heads*kv_tokens*hd
  # elements -- but `full_kv_tiles=(kv_tokens+15)//16` at that call site CEILS the KV-tile trip count to
  # cover a possibly-partial tail tile. On that tail tile some lanes' `token = rng*16+<lane offset>` is
  # >= kv_tokens: without a guard here those lanes still issue a REAL global load past the end of the
  # buffer. `expand_native_row_softmax_repack`'s `valid.where(value,-inf)`/`valid.where(weight,0)` (this
  # module, ~line 213/237) only zero the *softmax contribution* of an invalid KV column after the fact --
  # they never touch the load address, so an out-of-range V element that happens to be NaN/Inf (garbage
  # or genuinely unmapped memory) still contaminates the PV accumulation as `weight(=0) * garbage`, which
  # is not guaranteed finite (0*inf=NaN, 0*NaN=NaN). This guards the LOAD ADDRESS itself, the same idiom
  # postrange.py's PADTO opt uses for exactly this class of bug: `(valid & ...).where(idx, UOp.invalid())`
  # -- `UOp.valid` (tinygrad/uop/ops.py) is that exact helper, and codegen/late/gater.py's
  # `pm_move_gates_from_index` turns `buf.index(gate.where(idx, Invalid))` into a real masked/gated LOAD
  # (alt=0, exec-predicated), not a plain unconditional access.
  #
  # Only the grid-carrying construction (x.arg.grid is not None) has a `kv_tokens` to guard against; the
  # legacy grid-less branch above (`gbase=0`, no head_dim kwarg -- a different, always-fixed-kv64 kernel
  # family) is untouched.
  #
  # FOLD-AWAY (must not regress the aligned hot path): when kv_tokens is a compile-time multiple of 16,
  # `rng` (UOp.range(full_kv_tiles,...)) has a static bound of [0, full_kv_tiles-1], so every `token`
  # expression below has a provable static max of kv_tokens-1. tinygrad/uop/ops.py's CMPLT vmin/vmax rule
  # (`s0_vmax<s1_vmin, s0_vmin<s1_vmax`) then makes the comparison's own vmin==vmax==True, and
  # tinygrad/uop/symbolic.py:258-259 constant-folds any {CMPLT,...} UOp whose vmin==vmax to that constant;
  # symbolic.py:163 (`gate.where(c0,c1) -> c0 if gate.arg else c1`) then collapses `row_ok.where(idx,
  # Invalid)` straight back to `idx`. See test_amd_attention_kv_tile_oob_guard.py's
  # `*_guard_folds_away_when_aligned` tests, which assert the rendered ISA for an aligned geometry is
  # instruction-for-instruction identical to the pre-fix baseline.
  grid_kv_tokens=x.arg.grid.kv_tokens if x.arg.grid is not None else None
  def _row_ok(token): return None if grid_kv_tokens is None else token < grid_kv_tokens
  model=getattr(x.arg,"fragment_model",None)
  if shared_storage:
    if role not in {"K", "V"}: raise ValueError("shared packed fragments currently require K/V role")
    if model is not None:
      lanes=model.fragment_lanes(role)
      if role == "K":
        offs=tuple((model.operand_row(1,0,lane)*128 + block*16 + model.operand_k(1,i,lane)) for i in range(lanes))
      else:
        offs=tuple((model.operand_k(1,i,lane)*128 + block*16 + model.operand_row(1,0,lane)) for i in range(lanes))
    else:
      offs=tuple(col*128 + block*16+i for i in range(16))
    return UOp(Ops.STACK,dtypes.half.vec(model.fragment_lanes(role) if model is not None else 16),
      tuple(shared_owner.index(off).load() for off in offs),
      tag=("amd_gfx1100_fragment_load_hd128_loop_v1",role,block,x.arg,*x.src))
  if model is not None:
    # Fragment-model path: per-element load addresses derive from the target's own operand lane
    # layouts. The call offset is added only when this fragment belongs to a later WMMA call of a
    # multi-call tile (call 0 stays node-identical with the literal AMD tree).
    lanes=model.fragment_lanes(role)
    call_off=x.arg.call*model.tc.dims[0] if x.arg.call else 0
    if role=="Q":
      offs=tuple(gbase+model.operand_row(0,i,lane)*hd+block*16+model.operand_k(0,i,lane) for i in range(lanes))
    elif role=="K":
      row=model.operand_row(1,0,lane)
      if call_off: row=row.alu(Ops.ADD,UOp.const(dtypes.weakint,call_off))
      row_ok=_row_ok(rng*UOp.const(dtypes.weakint,16)+row)  # one KV row for the whole fragment
      offs=tuple(gbase+rng*16*hd+row*hd+block*16+model.operand_k(1,i,lane) for i in range(lanes))
      if row_ok is not None: offs=tuple(o.valid(row_ok) for o in offs)
    elif getenv("PREFILL_V_TRANSPOSED") and x.arg.grid is not None:
      row=model.operand_row(1,0,lane)
      if call_off: row=row.alu(Ops.ADD,UOp.const(dtypes.weakint,call_off))
      offs=tuple(gbase+(block*16+row)*x.arg.grid.kv_tokens+rng*16+model.operand_k(1,i,lane) for i in range(lanes))
      row_oks=tuple(_row_ok(rng*UOp.const(dtypes.weakint,16)+model.operand_k(1,i,lane)) for i in range(lanes))
      offs=tuple(o if g is None else o.valid(g) for o,g in zip(offs,row_oks))
    else:
      row=model.operand_row(1,0,lane)
      if call_off: row=row.alu(Ops.ADD,UOp.const(dtypes.weakint,call_off))
      offs=tuple(gbase+rng*16*hd+block*16+model.operand_k(1,i,lane)*hd+row for i in range(lanes))
      row_oks=tuple(_row_ok(rng*UOp.const(dtypes.weakint,16)+model.operand_k(1,i,lane)) for i in range(lanes))
      offs=tuple(o if g is None else o.valid(g) for o,g in zip(offs,row_oks))
  elif role=="Q": offs=tuple(gbase+col*hd+block*16+i for i in range(16))
  elif role=="K":
    row_ok=_row_ok(rng*UOp.const(dtypes.weakint,16)+col)  # same KV row for all 16 lanes of this fragment
    offs=tuple(gbase+rng*16*hd+col*hd+block*16+i for i in range(16))
    if row_ok is not None: offs=tuple(o.valid(row_ok) for o in offs)
  elif getenv("PREFILL_V_TRANSPOSED") and x.arg.grid is not None:
    # V VECTORIZATION (measured lever): row-major V is [kv][hd], so the PV WMMA B-fragment -- which
    # wants a fixed d=block*16+col and 16 VARYING kv -- reads with stride hd and lowers to 128
    # `global_load_d16_b16` 2-byte gathers per KV tile (8 blocks x 16). PMC: those are 89% of the
    # kernel's VMEM instructions and VMEM is ~63-65% of SQ busy cycles at every context (kv512->4096),
    # i.e. ~58% of the kernel. Reading a [hd][kv]-TRANSPOSED V makes `i` the contiguous index, so the
    # same 16 halves fold into 2 `global_load_b128` per block (16 total, matching K).
    # The caller must pass V pre-transposed (llm/fused_attention.py); gbase is unchanged because
    # hd*kv_tokens == kv_tokens*hd.
    offs=tuple(gbase+(block*16+col)*x.arg.grid.kv_tokens+rng*16+i for i in range(16))
    row_oks=tuple(_row_ok(rng*UOp.const(dtypes.weakint,16)+UOp.const(dtypes.weakint,i)) for i in range(16))
    offs=tuple(o if g is None else o.valid(g) for o,g in zip(offs,row_oks))
  else:
    offs=tuple(gbase+rng*16*hd+block*16+i*hd+col for i in range(16))
    row_oks=tuple(_row_ok(rng*UOp.const(dtypes.weakint,16)+UOp.const(dtypes.weakint,i)) for i in range(16))
    offs=tuple(o if g is None else o.valid(g) for o,g in zip(offs,row_oks))
  return UOp(Ops.STACK,dtypes.half.vec(model.fragment_lanes(role) if model is not None else 16),tuple(owner.index(off).load() for off in offs),
    tag=("amd_gfx1100_fragment_load_hd128_loop_v1",role,block,x.arg,*x.src))

def expand_native_row_softmax_repack(ctx, x:UOp, native_state:bool=True) -> UOp:
  """Expand the exact gfx1100-v1 QK-C -> PV-A bridge before isel."""
  from tinygrad.uop.ops import NativeRowSoftmaxRepackSpec, AMDMultiWaveAttentionGridSpec
  if not isinstance(x.arg, NativeRowSoftmaxRepackSpec): raise ValueError("AMD row-softmax repack is missing its native descriptor")
  x.arg.validate()
  initial_state = x.arg.mode == "initial_state_v1"
  if initial_state:
    if len(x.src) != 1: raise ValueError("initial-state repack must not carry old m/l state")
    score, m, l = x.src[0], None, None
  else:
    expected=(5 if x.arg.grid is not None else 4) if x.arg.dynamic_kv_v1 else 3
    if len(x.src) != expected: raise ValueError("row-softmax transition repack requires score/m/l and its declared tile source")
    score, m, l, *tile_src = x.src
    if x.arg.dynamic_kv_v1 and tile_src[0].op is not Ops.RANGE: raise ValueError("dynamic repack tile source must be RANGE")
  # A single-call tile carries the raw QK WMMA; a multi-call tile carries the
  # per-call C fragments concatenated into one STACK (both float.vec(8)).
  if not (score.op is Ops.WMMA and score.dtype == dtypes.float.vec(8)) and not \
      (score.op is Ops.STACK and score.dtype == dtypes.float.vec(8) and len(score.src) == 8 and
       all(s.dtype == dtypes.float for s in score.src)):
    raise ValueError("row-softmax repack requires one QK score carrier of float.vec(8)")
  stateful = x.arg.mode in {"initial_state_v1", "stateful_unnormalized_v1", "loop_state_v1"}
  native_state = native_state and x.arg.mode != "loop_state_v1"
  state_dt, state_shape = (dtypes.float.vec(8), (8,)) if stateful else (dtypes.float, ())
  if not initial_state and any(s.dtype != state_dt or s.shape != state_shape for s in (m, l)):
    raise ValueError("AMD row-softmax repack state dtype does not match descriptor mode")
  multiwave = isinstance(x.arg.grid, AMDMultiWaveAttentionGridSpec)
  nv_grouped = getattr(x.arg, "native_abi", "").startswith("nv_sm120_") and getattr(x.arg.grid, "local_size", 32) == 128
  tid = UOp.special(128 if nv_grouped else (x.arg.grid.local_size if multiwave else 32), "lidx0")
  lane = tid.alu(Ops.AND, UOp.const(dtypes.weakint, 31)) if multiwave else tid
  wave_id = tid.alu(Ops.SHR, UOp.const(dtypes.weakint, 5)) if (multiwave or nv_grouped) else UOp.const(dtypes.weakint, 0)
  wave_base = wave_id.alu(Ops.MUL, UOp.const(dtypes.weakint, 256))
  lane_hw = lane.cast(dtypes.int)
  halfwave, col = lane.alu(Ops.SHR, UOp.const(dtypes.weakint, 4)), lane.alu(Ops.AND, UOp.const(dtypes.weakint, 15))
  lds_size = (1024 if nv_grouped else 512) if multiwave or nv_grouped else 256
  lds = UOp(Ops.DEFINE_LOCAL, dtypes.half.ptr(lds_size, AddrSpace.LOCAL), arg=next(ctx))
  state_owner = next(ctx) if stateful and native_state else None
  state_writes_m, state_writes_l, state_writes_alpha = [], [], []
  stores, new_ms, new_ls, alphas, log2e = [], [], [], [], UOp.const(dtypes.float, 1.4426950408889634)
  model=getattr(x.arg,"fragment_model",None)
  def _score_value(e:int, row:UOp, col:UOp):
    """Masked, scaled score element at (row, col); validity is not published here."""
    valid = None
    fused_causal = False
    kv = qrow = None
    if x.arg.validity_mode in {"tail_v1", "causal_v1"}:
      fused_causal = x.arg.validity_mode == "causal_v1" and x.arg.grid is not None and \
        x.arg.query_start == x.arg.valid_kv-x.arg.grid.q_tokens
      if fused_causal:
        kv_base=tile_src[0].cast(dtypes.int).alu(Ops.MUL,UOp.const(dtypes.int,16))
        kv=col.cast(dtypes.int).alu(Ops.ADD,kv_base)
        qrow=row.cast(dtypes.int).alu(Ops.ADD,UOp.const(dtypes.int,x.arg.query_start))
        qrow=qrow.alu(Ops.ADD,(tile_src[1].cast(dtypes.int) % x.arg.grid.q_tiles)*16)
      else:
        kv_base = tile_src[0].cast(dtypes.weakint).alu(Ops.MUL,UOp.const(dtypes.weakint,16)) if x.arg.dynamic_kv_v1 \
          else UOp.const(dtypes.weakint,x.arg.kv_start)
        kv = col.alu(Ops.ADD, kv_base)
        qrow = row.alu(Ops.ADD, UOp.const(dtypes.weakint, x.arg.query_start))
        if x.arg.grid is not None:
          qtile=tile_src[1] % x.arg.grid.q_tiles
          qrow=qrow.alu(Ops.ADD,qtile*16)
      causal = kv.alu(Ops.CMPLT,qrow.alu(Ops.ADD,UOp.const(dtypes.weakint,1))) if x.arg.validity_mode == "causal_v1" else None
      if fused_causal: valid=None
      else:
        valid = kv.alu(Ops.CMPLT, UOp.const(dtypes.weakint, x.arg.valid_kv))
        if causal is not None: valid=valid.alu(Ops.AND,causal)
    value = score.gep(e).alu(Ops.MUL, UOp.const(dtypes.float, x.arg.score_scale))
    if x.arg.validity_mode == "causal_v1" and x.arg.grid is not None and \
       x.arg.query_start == x.arg.valid_kv-x.arg.grid.q_tokens:
      value=UOp(Ops.CUSTOMI,dtypes.float,(value,kv,qrow),"(({1}<={2})?{0}:-INFINITY)")
    if valid is not None: value = valid.where(value, UOp.const(dtypes.float, -float("inf")))
    return value, valid, fused_causal, kv, qrow
  if model is not None and model.reduction_within_lane:
    # DEFERRED reduction (NV-shaped tiles): element steps combine the lane's own
    # score elements pairwise, so no value may be reduced incrementally while its
    # pair is still live. Two passes over the ladder keep every step reading the
    # previous step's values: row-max first, then row-sum over the weights.
    n = model.score_elements
    raw = [_score_value(e, model.c_row_uop(e, lane), model.c_col_uop(e, lane)) for e in range(n)]
    row_maxs = [r[0] for r in raw]
    for kind, bit in model.col_reduction:
      if kind == "element":
        mask = 1 << bit
        for e in range(n):
          if e & mask == 0:
            partner = e | mask
            combined = row_maxs[e].alu(Ops.MAX, row_maxs[partner])
            row_maxs[e] = row_maxs[partner] = combined
      else:
        addr = lane_hw.alu(Ops.XOR, UOp.const(dtypes.int, 1 << bit)).alu(Ops.MUL, UOp.const(dtypes.int, 4))
        for e in range(n): row_maxs[e] = row_maxs[e].alu(Ops.MAX, warp_bpermute(addr, row_maxs[e]))
    weights = []
    for e in range(n):
      value, valid, fused_causal, kv, qrow = raw[e]
      new_m = row_maxs[e] if initial_state else m.gep(e).alu(Ops.MAX, row_maxs[e])
      weight = (value-new_m).alu(Ops.MUL, log2e).exp2()
      if fused_causal: weight=UOp(Ops.CUSTOMI,dtypes.float,(weight,kv,qrow),"(({1}<={2})?{0}:0.0f)")
      if valid is not None: weight = valid.where(weight, UOp.const(dtypes.float, 0))
      new_ms.append(new_m); weights.append(weight)
    row_sums = list(weights)
    for kind, bit in model.col_reduction:
      if kind == "element":
        mask = 1 << bit
        for e in range(n):
          if e & mask == 0:
            partner = e | mask
            combined = row_sums[e].alu(Ops.ADD, row_sums[partner])
            row_sums[e] = row_sums[partner] = combined
      else:
        addr = lane_hw.alu(Ops.XOR, UOp.const(dtypes.int, 1 << bit)).alu(Ops.MUL, UOp.const(dtypes.int, 4))
        for e in range(n): row_sums[e] = row_sums[e].alu(Ops.ADD, warp_bpermute(addr, row_sums[e]))
    state_rows = []
    for e in range(n):
      old_m, old_l = (m.gep(e), l.gep(e)) if stateful and not initial_state else (m, l)
      new_m = new_ms[e]
      raw_alpha = UOp.const(dtypes.float, 1) if initial_state else row_sums[e].ne(UOp.const(dtypes.float, 0)).where(
        (old_m-new_m).alu(Ops.MUL, log2e).exp2(), UOp.const(dtypes.float, 1))
      alpha = raw_alpha
      new_l = row_sums[e] if initial_state else old_l.alu(Ops.MUL, alpha).alu(Ops.ADD, row_sums[e])
      if not stateful or not native_state: new_ls.append(new_l)
      alphas.append(alpha)
      state_rows.append((new_m, new_l, alpha))
    # Publish every element at its C-fragment position; the PV-A reload below
    # transposes through LDS exactly as the interleaved path does.
    for e in range(n):
      new_m, new_l, alpha = state_rows[e]
      normalized = (weights[e] if stateful else weights[e] / new_l).cast(dtypes.half)
      row, col_e = model.c_row_uop(e, lane), model.c_col_uop(e, lane)
      published_row = lds.index(wave_base.alu(Ops.ADD,
        row.alu(Ops.MUL, UOp.const(dtypes.weakint, 16)).alu(Ops.ADD, col_e))).store(normalized)
      if stateful and native_state:
        mw = UOp(Ops.CUSTOMI, dtypes.void, (new_m,), arg=("amd_gfx1100_row_state_write_v1", state_owner, "m", e))
        lw = UOp(Ops.CUSTOMI, dtypes.void, (new_l,), arg=("amd_gfx1100_row_state_write_v1", state_owner, "l", e))
        aw = UOp(Ops.CUSTOMI, dtypes.void, (alpha,), arg=("amd_gfx1100_row_state_write_v1", state_owner, "alpha", e))
        state_writes_m.append(mw); state_writes_l.append(lw); state_writes_alpha.append(aw)
        stores.append(UOp.group(published_row, mw, lw, aw))
      else: stores.append(published_row)
  else:
    for e in range(model.score_elements if model is not None else 8):
      old_m, old_l = (m.gep(e), l.gep(e)) if stateful and not initial_state else (m, l)
      row = model.c_row_uop(e, lane) if model is not None else UOp.const(dtypes.weakint, 2*e).alu(Ops.ADD, halfwave)
      col_e = model.c_col_uop(0, lane) if model is not None else col
      value, valid, fused_causal, kv, qrow = _score_value(e, row, col_e)
      if stores: value = value.bitcast(dtypes.uint).after(UOp.group(stores[-1])).bitcast(dtypes.float)
      # THEORY 6 (measured, 2026-07-24) -- the two butterflies below are exactly the two the algorithm
      # needs, but on the SHIPPED HIP path they used to cost THREE cross-lane traversals per row. Neither
      # extra traversal is emitted here; both are artifacts of how this expression tree is RENDERED, and
      # both are addressed by PREFILL_SOFTMAX_REDUCE_FUSE in tinygrad/renderer/cstyle.py:
      #   (a) Ops.CUSTOMI is inlined unconditionally by the C renderer, ignoring child_count. Every rung of
      #       these ladders has two consumers (the next fmaxf AND the next bpermute), so the emitted C grows
      #       as 2^n: a 4-step ladder renders as 15 textual bpermutes, 272 across the 8-row repack where
      #       only 64 are distinct.
      #   (b) `new_m = max(old_m, row_max)` below is not a native HIP op, so decompositions.py rewrites it
      #       to (a<b).where(b,a) -- inlining the whole ladder twice more, and lowering to an exec-masked
      #       v_cmpx_lt_f32/s_cbranch_execz region that LLVM's CSE will not cross, so a third ladder is
      #       REMATERIALIZED inside the guard.
      # Result was 96 ds_bpermute_b32 + 97 mandatory s_waitcnt lgkmcnt(0) + 135 v_max_f32 per KV tile,
      # against 64 bpermute for the two real reductions. Do not "simplify" this by hoisting row_max into a
      # Python temp -- it already is one; the duplication is in the renderer, not here.
      row_max = value
      for mask in x.arg.xor_masks:
        addr = lane_hw.alu(Ops.XOR, UOp.const(dtypes.int, mask)).alu(Ops.MUL, UOp.const(dtypes.int, 4))
        row_max = row_max.alu(Ops.MAX, warp_bpermute(addr, row_max))
      new_m = row_max if initial_state else old_m.alu(Ops.MAX, row_max)
      weight = (value-new_m).alu(Ops.MUL, log2e).exp2()
      if fused_causal: weight=UOp(Ops.CUSTOMI,dtypes.float,(weight,kv,qrow),"(({1}<={2})?{0}:0.0f)")
      if valid is not None: weight = valid.where(weight, UOp.const(dtypes.float, 0))
      row_sum = weight
      for mask in x.arg.xor_masks:
        addr = lane_hw.alu(Ops.XOR, UOp.const(dtypes.int, mask)).alu(Ops.MUL, UOp.const(dtypes.int, 4))
        row_sum = row_sum.alu(Ops.ADD, warp_bpermute(addr, row_sum))
      raw_alpha = UOp.const(dtypes.float, 1) if initial_state else row_sum.ne(UOp.const(dtypes.float, 0)).where(
        (old_m-new_m).alu(Ops.MUL, log2e).exp2(), UOp.const(dtypes.float, 1))
      alpha = raw_alpha
      new_l = row_sum if initial_state else old_l.alu(Ops.MUL, alpha).alu(Ops.ADD, row_sum)
      if not stateful or not native_state: new_ms.append(new_m); new_ls.append(new_l)
      alphas.append(alpha)
      normalized = (weight if stateful else weight / new_l).cast(dtypes.half)
      published_row = lds.index(wave_base.alu(Ops.ADD,
        row.alu(Ops.MUL, UOp.const(dtypes.weakint, 16)).alu(Ops.ADD, col))).store(normalized)
      # Serialize row publication so eight independent butterfly/exp/CVT trees
      # do not become simultaneously live before the barrier.
      if stateful and native_state:
        mw = UOp(Ops.CUSTOMI, dtypes.void, (new_m,), arg=("amd_gfx1100_row_state_write_v1", state_owner, "m", e))
        lw = UOp(Ops.CUSTOMI, dtypes.void, (new_l,), arg=("amd_gfx1100_row_state_write_v1", state_owner, "l", e))
        aw = UOp(Ops.CUSTOMI, dtypes.void, (alpha,), arg=("amd_gfx1100_row_state_write_v1", state_owner, "alpha", e))
        state_writes_m.append(mw); state_writes_l.append(lw); state_writes_alpha.append(aw)
        stores.append(UOp.group(published_row, mw, lw, aw))
      else: stores.append(published_row)
  # A workgroup barrier is necessary unless the launch descriptor proves that
  # the complete workgroup is one gfx1100 wave.  In that exact case wave issue
  # order plus lgkmcnt(0) publishes all P stores before any PV-A reload without
  # paying for an inter-wave rendezvous that cannot have participants.
  if multiwave:
    from tinygrad.codegen.opt.compiler_policies import WaveLDSFence
    ready = UOp(Ops.BARRIER, dtypes.void, (UOp.group(*stores),), arg=WaveLDSFence(
      wave_size=x.arg.grid.wave_size, workgroup_size=x.arg.grid.local_size, wave_slices=x.arg.grid.p_wave_slices))
  elif x.arg.grid is not None and x.arg.grid.single_wave_workgroup:
    from tinygrad.codegen.opt.compiler_policies import WaveLDSFence
    ready = UOp(Ops.BARRIER, dtypes.void, (UOp.group(*stores),), arg=WaveLDSFence(
      wave_size=x.arg.grid.wave_size, workgroup_size=x.arg.grid.local_size))
  else: ready = UOp.barrier(UOp.group(*stores))
  reload_row = wave_base.alu(Ops.ADD, col.alu(Ops.MUL, UOp.const(dtypes.weakint, 16)))
  published = lds.after(ready)
  if model is not None:
    # PV-A reload reads the A-operand positions of the target's own layout; the
    # LDS transpose from the C positions published above is the same move the
    # literal AMD path makes (there, A row == col and A k == element index).
    vals = [published.index(wave_base.alu(Ops.ADD,
      model.operand_row(0, i, lane).alu(Ops.MUL, UOp.const(dtypes.weakint, 16)).alu(Ops.ADD,
      model.operand_k(0, i, lane)))).load() for i in range(model.pv_a_lanes)]
  else: vals = [published.index(reload_row.alu(Ops.ADD, UOp.const(dtypes.weakint, i))).load() for i in range(16)]
  if stateful and native_state:
    new_ms = [UOp(Ops.CUSTOMI, dtypes.float, (state_writes_m[i], ready), arg=("amd_gfx1100_row_state_read_v1", state_owner, "m", i)) for i in range(8)]
    new_ls = [UOp(Ops.CUSTOMI, dtypes.float, (state_writes_l[i], ready), arg=("amd_gfx1100_row_state_read_v1", state_owner, "l", i)) for i in range(8)]
    alphas = [UOp(Ops.CUSTOMI, dtypes.float, (state_writes_alpha[i], ready), arg=("amd_gfx1100_row_state_read_v1", state_owner, "alpha", i)) for i in range(8)]
  p = UOp(Ops.STACK, dtypes.half.vec(model.pv_a_lanes if model is not None else 16), tuple(vals),
    tag=("amd_gfx1100_pv_a_reload_v1",))
  # Each physical accumulator element owns one row. These vector states stay
  # replicated in the native C layout until a descriptor-owned final store.
  alpha_owner = x
  tagged_alphas = tuple(a.replace(tag=("amd_gfx1100_online_softmax_alpha_v1", alpha_owner)) for a in alphas)
  return UOp(Ops.TUPLE, dtypes.void, (p,
    UOp(Ops.STACK, dtypes.float.vec(8), tuple(new_ms), tag=("amd_gfx1100_row_state_v1", "m")),
    UOp(Ops.STACK, dtypes.float.vec(8), tuple(new_ls), tag=("amd_gfx1100_row_state_v1", "l")),
    UOp(Ops.STACK, dtypes.float.vec(8), tagged_alphas, tag=("amd_gfx1100_row_state_v1", "alpha"))))

def lower_native_row_state_gep(x:UOp, s:UOp) -> UOp|None:
  if not isinstance(s.tag, tuple) or s.tag[:1] != ("amd_gfx1100_row_state_v1",): return None
  if len(x.arg) != 1 or not 0 <= x.arg[0] < 8: raise ValueError("invalid native row-state projection")
  return s.src[x.arg[0]]

def lower_state_phase_transfer(x:UOp) -> UOp|None:
  """Lower generic explicit state publication to its caller-owned LDS region.

  Storage-free handles deliberately remain opaque: they are semantic phase
  carriers and must not acquire an implicit allocation in a renderer.
  """
  from tinygrad.uop.ops import StateHandle
  if isinstance(x.arg,tuple) and len(x.arg)==3 and x.arg[0] == "state_loop_read_v1" and isinstance(x.arg[1],StateHandle):
    handle,element=x.arg[1:]; handle.validate()
    if handle.storage is None or len(x.src) not in (2,3) or x.src[:2] != (handle.storage,handle.lane): return None
    base=handle.lane.alu(Ops.MUL,UOp.const(dtypes.weakint,handle.lane_stride)).alu(Ops.ADD,UOp.const(dtypes.weakint,handle.element_offset+element))
    owner=handle.storage if len(x.src)==2 else handle.storage.after(x.src[2])
    return owner.index(base).load()
  if isinstance(x.arg,tuple) and len(x.arg)==3 and x.arg[0] == "state_loop_write_v1" and isinstance(x.arg[1],StateHandle):
    handle,element=x.arg[1:]; handle.validate()
    if handle.storage is None or len(x.src) not in (3,4) or x.src[1:3] != (handle.storage,handle.lane): return None
    value,_,lane,*deps=x.src
    base=lane.alu(Ops.MUL,UOp.const(dtypes.weakint,handle.lane_stride)).alu(Ops.ADD,UOp.const(dtypes.weakint,handle.element_offset+element))
    return handle.storage.after(*deps).index(base).store(value)
  if not (isinstance(x.arg,tuple) and len(x.arg)==2 and x.arg[0] in {"state_publish_v1","state_reload_v1"} and isinstance(x.arg[1],StateHandle)): return None
  op,handle=x.arg; handle.validate()
  if handle.storage is None: return None
  if op=="state_publish_v1":
    value,storage,lane=x.src
    base=lane.alu(Ops.MUL,UOp.const(dtypes.weakint,handle.lane_stride)).alu(Ops.ADD,UOp.const(dtypes.weakint,handle.element_offset))
    stores=tuple(storage.index(base.alu(Ops.ADD,UOp.const(dtypes.weakint,i))).store(value.gep(i)) for i in range(handle.region.lanes))
    return value.after(UOp.group(*stores))
  # Generic construction may already have materialized the lane-major reload
  # as a provenance-preserving one-source carrier before renderer rewrites.
  if len(x.src)==1:
    lanes=x.src[0]
    if handle.region.lanes > 1 and lanes.op is Ops.STACK and lanes.dtype == handle.dtype and \
       lanes.tag == ("state_reload_lanes_v1",handle) and len(lanes.src) == handle.region.lanes and \
       all(source.dtype == handle.region.dtype for source in lanes.src):
      # Keep the typed carrier until a scalar consumer projects a lane.  A raw
      # STACK is simplified by UOp GEP construction and loses this provenance.
      return None
    return lanes
  published,storage,lane,*deps=x.src
  base=lane.alu(Ops.MUL,UOp.const(dtypes.weakint,handle.lane_stride)).alu(Ops.ADD,UOp.const(dtypes.weakint,handle.element_offset))
  owner=storage.after(published,*deps)
  return UOp(Ops.STACK,handle.dtype,tuple(owner.index(base.alu(Ops.ADD,UOp.const(dtypes.weakint,i))).load() for i in range(handle.region.lanes)))

def lower_state_phase_reload_gep(x:UOp, carrier:UOp) -> UOp|None:
  """Consume one typed reload lane without exposing a raw vector stack."""
  if not (isinstance(carrier.arg,tuple) and carrier.arg[:1] == ("state_reload_v1",) and len(carrier.arg) == 2): return None
  handle=carrier.arg[1]
  from tinygrad.uop.ops import StateHandle
  if not isinstance(handle,StateHandle) or handle.storage is None or handle.region.lanes <= 1 or len(x.arg) != 1 or not 0 <= x.arg[0] < handle.region.lanes:
    return None
  try: handle.validate()
  except (TypeError,ValueError): return None
  if len(carrier.src) != 1: return None
  lanes=carrier.src[0]
  if lanes.op is not Ops.STACK or lanes.tag != ("state_reload_lanes_v1",handle) or len(lanes.src) != handle.region.lanes:
    return None
  return lanes.src[x.arg[0]]

native_repack_matcher = PatternMatcher([
  (UPat(Ops.GEP, src=(UPat(Ops.CUSTOMI,name="carrier"),), name="x"), lower_state_phase_reload_gep),
  (UPat((Ops.CUSTOMI,Ops.CUSTOM),name="x"), lower_state_phase_transfer),
  (UPat(Ops.NATIVE_ROW_SOFTMAX_REPACK, name="x"), expand_native_row_softmax_repack),
  (UPat(Ops.ROW_SOFTMAX_SLOT, src=(UPat(Ops.TUPLE, name="owner"),), name="x"), lambda x,owner: owner.src[x.arg.slot]),
  (UPat(Ops.GEP, src=(UPat(Ops.STACK, name="s"),), name="x"), lower_native_row_state_gep),
])
native_loop_fragment_matcher=PatternMatcher([(UPat(Ops.PACKED_FRAGMENT_LOAD,name="x"),expand_loop_fragment)])
native_stage_begin_matcher=PatternMatcher([(UPat(Ops.COOPERATIVE_STAGE_BEGIN,name="x"),lower_cooperative_stage_begin)])

def lower_native_pv_c_lane(x:UOp) -> UOp:
  x.arg.validate()
  e = x.arg.element
  if x.src[0].dtype != dtypes.float.vec(8) or not 0 <= e < 8:
    raise ValueError("invalid native PV-C lane projection")
  return x.src[0].gep(e)

def lower_amd_attention_loop_state(x:UOp) -> UOp:
  from tinygrad.uop.ops import LoopStateSpec
  if not isinstance(x.arg, LoopStateSpec): raise ValueError("AMD attention loop state is missing its typed ABI")
  # The physical VGPR map is contained in amd_register_contracts.AMD_ATTENTION_LOOP_STATE; see its
  # docstring for the invariant, the caller list, and the negative result that came of it being a bare dict.
  x.arg.validate(); base=AMD_ATTENTION_LOOP_STATE.base(x.arg.role, x.arg.block if x.arg.role=="acc" else 0)
  if x.arg.access in {"read","final_read"}:
    return _fixed_alias(base,x.arg.lane,dtypes.float)
  store=x.src[0]
  if store.op is not Ops.STORE or len(store.src)<2: raise ValueError("AMD attention loop-state write requires one STORE")
  return UOp(Ops.CUSTOMI,dtypes.void,(store.src[1],),arg=("amd_gfx1100_attention_loop_state_write_v1",x.arg.role,x.arg.block,x.arg.lane))

native_state_lane_matcher = PatternMatcher([
  (UPat(Ops.AMD_PV_C_LANE, name="x"), lower_native_pv_c_lane),
])

native_loop_state_matcher = PatternMatcher([
  (UPat(Ops.ATTENTION_LOOP_STATE, name="x"), lower_amd_attention_loop_state),
])
