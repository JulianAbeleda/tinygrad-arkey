"""The AMD/rdna3 fused-attention typed ABI: lowering for the six bespoke AMD Ops.

The prefill fused-attention kernel does not go through generic instruction
selection alone. It carries six renderer-specific Ops whose meaning is fixed by
typed descriptors in tinygrad/uop/ops.py:

    Ops.AMD_PACKED_FRAGMENT_LOAD    (AMDPackedFragmentLoopSpec) Q/K/V fragment addressing
    Ops.AMD_ROW_SOFTMAX_REPACK      (AMDRowSoftmaxRepackSpec)   QK-C -> P -> PV-A bridge
    Ops.AMD_ROW_SOFTMAX_SLOT                                    projection of the above
    Ops.AMD_PV_C_LANE               (AMDPVCLaneSpec)            PV accumulator lane view
    Ops.AMD_ATTENTION_LOOP_STATE    (AMDLoopStateSpec)          loop-carried m/l/acc
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
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.renderer.isa.amd_physical_regs import _fixed_alias


def _opaque_exact_fragment_inputs(x:UOp) -> UOp|None:
  if x.op is not Ops.WMMA or len(x.src) != 3: return None
  changed, src = False, list(x.src)
  for pos in (0,1):
    c=src[pos]
    if not (c.op is Ops.STACK and c.dtype == dtypes.half.vec(16) and len(c.src)==16 and isinstance(c.tag,tuple) and
            c.tag[:1] in {("amd_gfx1100_fragment_load_v1",),("amd_gfx1100_fragment_load_hd128_v1",),("amd_gfx1100_fragment_load_hd128_loop_v1",)} and
            all(v.op is Ops.LOAD and v.dtype==dtypes.half for v in c.src)): continue
    if c.tag[0] == "amd_gfx1100_fragment_load_hd128_loop_v1":
      from tinygrad.uop.ops import AMDPackedFragmentLoopSpec
      _,role,hd_block,*payload=c.tag
      if payload and isinstance(payload[0], AMDPackedFragmentLoopSpec): spec,*fragment_src=payload
      else:
        owner,lane,col,rng=payload
        spec,fragment_src=AMDPackedFragmentLoopSpec(role=role,head_block=hd_block),[owner,lane,col,rng]
      src[pos]=UOp(Ops.AMD_PACKED_FRAGMENT_LOAD,dtypes.half.vec(16),tuple(fragment_src),arg=spec)
      changed=True
      continue
    if c.tag[0] == "amd_gfx1100_fragment_load_hd128_v1": _,role,tile,hd_block,owner,lane,col=c.tag
    else: _,role,tile,owner,lane,col=c.tag; hd_block=None
    if role not in {"Q","K","V"} or not isinstance(tile,int) or tile not in {0,1}: raise ValueError("malformed gfx1100 fragment descriptor")
    if role == "Q" and pos != 0 or role == "K" and pos != 1 or role == "V" and pos != 1: raise ValueError("fragment role/WMMA operand mismatch")
    abi="amd_gfx1100_packed_fragment_hd128_v1" if hd_block is not None else "amd_gfx1100_packed_fragment_v1"
    arg=(abi,role,tile,hd_block) if hd_block is not None else (abi,role,tile)
    src[pos]=UOp(Ops.AMD_PACKED_FRAGMENT_LOAD,dtypes.half.vec(16),(owner,lane,col),arg=arg)
    changed=True
  return x.replace(src=tuple(src)) if changed else None

native_fragment_opaque_matcher=PatternMatcher([(UPat(Ops.WMMA,name="x"),_opaque_exact_fragment_inputs)])

def expand_loop_fragment(x:UOp) -> UOp:
  """Materialize the typed loop fragment before tensor/program verification.

  Its tag retains the owner and RANGE identity; the normal late opaque pass
  turns this back into a physical AMD carrier after index lowering.
  """
  from tinygrad.uop.ops import AMDPackedFragmentLoopSpec, AMDMultiWaveAttentionGridSpec
  if not isinstance(x.arg, AMDPackedFragmentLoopSpec): raise ValueError("loop fragment is malformed")
  x.arg.validate(); role,block=x.arg.role,x.arg.head_block
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
  if not grid_src: gbase=UOp.const(dtypes.weakint,0)
  elif isinstance(x.arg.grid, AMDMultiWaveAttentionGridSpec):
    grid,group=x.arg.grid,grid_src[0]
    kv_head,q_tile=group//grid.q_tiles,group%grid.q_tiles
    gbase=((kv_head*grid.waves_per_group+wave_id)*(grid.q_tokens*hd)+q_tile*16*hd) if role=="Q" else kv_head*(grid.kv_tokens*hd)
  elif role=="Q": gbase=grid_src[0]*(16*hd)
  else:
    grid=x.arg.grid
    gbase=(grid_src[0]//(grid.q_tiles*grid.group_ratio))*(grid.kv_tokens*hd)
  if role=="Q": offs=tuple(gbase+col*hd+block*16+i for i in range(16))
  elif role=="K": offs=tuple(gbase+rng*16*hd+col*hd+block*16+i for i in range(16))
  else: offs=tuple(gbase+rng*16*hd+block*16+i*hd+col for i in range(16))
  return UOp(Ops.STACK,dtypes.half.vec(16),tuple(owner.index(off).load() for off in offs),
    tag=("amd_gfx1100_fragment_load_hd128_loop_v1",role,block,x.arg,*x.src))

def expand_native_row_softmax_repack(ctx, x:UOp, native_state:bool=True) -> UOp:
  """Expand the exact gfx1100-v1 QK-C -> PV-A bridge before isel."""
  from tinygrad.uop.ops import AMDRowSoftmaxRepackSpec, AMDMultiWaveAttentionGridSpec
  if not isinstance(x.arg, AMDRowSoftmaxRepackSpec): raise ValueError("AMD row-softmax repack is missing its native descriptor")
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
  if score.op is not Ops.WMMA or score.dtype != dtypes.float.vec(8):
    raise ValueError("AMD row-softmax repack requires one raw QK WMMA float.vec(8)")
  stateful = x.arg.mode in {"initial_state_v1", "stateful_unnormalized_v1", "loop_state_v1"}
  native_state = native_state and x.arg.mode != "loop_state_v1"
  state_dt, state_shape = (dtypes.float.vec(8), (8,)) if stateful else (dtypes.float, ())
  if not initial_state and any(s.dtype != state_dt or s.shape != state_shape for s in (m, l)):
    raise ValueError("AMD row-softmax repack state dtype does not match descriptor mode")
  multiwave = isinstance(x.arg.grid, AMDMultiWaveAttentionGridSpec)
  tid = UOp.special(x.arg.grid.local_size if multiwave else 32, "lidx0")
  lane = tid.alu(Ops.AND, UOp.const(dtypes.weakint, 31)) if multiwave else tid
  wave_id = tid.alu(Ops.SHR, UOp.const(dtypes.weakint, 5)) if multiwave else UOp.const(dtypes.weakint, 0)
  wave_base = wave_id.alu(Ops.MUL, UOp.const(dtypes.weakint, 256))
  lane_hw = lane.cast(dtypes.int)
  halfwave, col = lane.alu(Ops.SHR, UOp.const(dtypes.weakint, 4)), lane.alu(Ops.AND, UOp.const(dtypes.weakint, 15))
  lds = UOp(Ops.DEFINE_LOCAL, dtypes.half.ptr(512 if multiwave else 256, AddrSpace.LOCAL), arg=next(ctx))
  state_owner = next(ctx) if stateful and native_state else None
  state_writes_m, state_writes_l, state_writes_alpha = [], [], []
  stores, new_ms, new_ls, alphas, log2e = [], [], [], [], UOp.const(dtypes.float, 1.4426950408889634)
  for e in range(8):
    old_m, old_l = (m.gep(e), l.gep(e)) if stateful and not initial_state else (m, l)
    row = UOp.const(dtypes.weakint, 2*e).alu(Ops.ADD, halfwave)
    valid = None
    fused_causal = False
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
    if stores: value = value.bitcast(dtypes.uint).after(UOp.group(stores[-1])).bitcast(dtypes.float)
    row_max = value
    for mask in x.arg.xor_masks:
      addr = lane_hw.alu(Ops.XOR, UOp.const(dtypes.int, mask)).alu(Ops.MUL, UOp.const(dtypes.int, 4))
      row_max = row_max.alu(Ops.MAX, UOp(Ops.CUSTOMI, dtypes.float, (addr, row_max), "bpermute"))
    new_m = row_max if initial_state else old_m.alu(Ops.MAX, row_max)
    weight = (value-new_m).alu(Ops.MUL, log2e).exp2()
    if fused_causal: weight=UOp(Ops.CUSTOMI,dtypes.float,(weight,kv,qrow),"(({1}<={2})?{0}:0.0f)")
    if valid is not None: weight = valid.where(weight, UOp.const(dtypes.float, 0))
    row_sum = weight
    for mask in x.arg.xor_masks:
      addr = lane_hw.alu(Ops.XOR, UOp.const(dtypes.int, mask)).alu(Ops.MUL, UOp.const(dtypes.int, 4))
      row_sum = row_sum.alu(Ops.ADD, UOp(Ops.CUSTOMI, dtypes.float, (addr, row_sum), "bpermute"))
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
  vals = [published.index(reload_row.alu(Ops.ADD, UOp.const(dtypes.weakint, i))).load() for i in range(16)]
  if stateful and native_state:
    new_ms = [UOp(Ops.CUSTOMI, dtypes.float, (state_writes_m[i], ready), arg=("amd_gfx1100_row_state_read_v1", state_owner, "m", i)) for i in range(8)]
    new_ls = [UOp(Ops.CUSTOMI, dtypes.float, (state_writes_l[i], ready), arg=("amd_gfx1100_row_state_read_v1", state_owner, "l", i)) for i in range(8)]
    alphas = [UOp(Ops.CUSTOMI, dtypes.float, (state_writes_alpha[i], ready), arg=("amd_gfx1100_row_state_read_v1", state_owner, "alpha", i)) for i in range(8)]
  p = UOp(Ops.STACK, dtypes.half.vec(16), tuple(vals), tag=("amd_gfx1100_pv_a_reload_v1",))
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
  (UPat(Ops.AMD_ROW_SOFTMAX_REPACK, name="x"), expand_native_row_softmax_repack),
  (UPat(Ops.AMD_ROW_SOFTMAX_SLOT, src=(UPat(Ops.TUPLE, name="owner"),), name="x"), lambda x,owner: owner.src[x.arg.slot]),
  (UPat(Ops.GEP, src=(UPat(Ops.STACK, name="s"),), name="x"), lower_native_row_state_gep),
])
native_loop_fragment_matcher=PatternMatcher([(UPat(Ops.AMD_PACKED_FRAGMENT_LOAD,name="x"),expand_loop_fragment)])

def lower_native_pv_c_lane(x:UOp) -> UOp:
  x.arg.validate()
  e = x.arg.element
  if x.src[0].dtype != dtypes.float.vec(8) or not 0 <= e < 8:
    raise ValueError("invalid native PV-C lane projection")
  return x.src[0].gep(e)

def lower_amd_attention_loop_state(x:UOp) -> UOp:
  from tinygrad.uop.ops import AMDLoopStateSpec
  if not isinstance(x.arg, AMDLoopStateSpec): raise ValueError("AMD attention loop state is missing its typed ABI")
  x.arg.validate(); base={"m":72,"l":80,"acc":8}[x.arg.role] + (x.arg.block*8 if x.arg.role=="acc" else 0)
  if x.arg.access in {"read","final_read"}:
    return _fixed_alias(base,x.arg.lane,dtypes.float)
  store=x.src[0]
  if store.op is not Ops.STORE or len(store.src)<2: raise ValueError("AMD attention loop-state write requires one STORE")
  return UOp(Ops.CUSTOMI,dtypes.void,(store.src[1],),arg=("amd_gfx1100_attention_loop_state_write_v1",x.arg.role,x.arg.block,x.arg.lane))

native_state_lane_matcher = PatternMatcher([
  (UPat(Ops.AMD_PV_C_LANE, name="x"), lower_native_pv_c_lane),
])

native_loop_state_matcher = PatternMatcher([
  (UPat(Ops.AMD_ATTENTION_LOOP_STATE, name="x"), lower_amd_attention_loop_state),
])
