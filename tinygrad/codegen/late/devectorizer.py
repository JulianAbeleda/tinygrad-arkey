from typing import Any, cast
import functools, itertools
from collections import defaultdict
from tinygrad.dtype import dtypes, ImageDType, AddrSpace, Invalid, PtrDType
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, GroupOp, RegisterResidentAccumulator
from tinygrad.uop.symbolic import uop_given_valid, parse_valid, invalid_gate
from tinygrad.helpers import getenv, flatten, prod
from tinygrad.renderer import Renderer

# ***** image load valid simplification *****

@functools.cache
def _drop_valid_stmts(valid:UOp, idx:UOp, height:int, width:int) -> list[UOp]:
  # can drop valid if idx is out of bound when valid is False
  drop_stmt = []
  for stmt in valid.split_uop(Ops.AND):
    if (res:=parse_valid(stmt)) is None: continue
    X, is_upper_bound, c = res

    # for X0 + X1 + ... >= 1, check if it's out of bound when Xi = 0 for all i
    if not is_upper_bound and c == 1 and all(u.op in GroupOp.Irreducible and u.vmin == 0 for u in X.split_uop(Ops.ADD)):
      testidx = functools.reduce(lambda nowidx,u: nowidx.substitute({u:u.const_like(0)}), X.split_uop(Ops.ADD), idx)
      if testidx.gep(0).vmax < 0 or testidx.gep(1).vmax < 0:
        drop_stmt.append(stmt)
        continue

    # if X <= c, check if it's out of bound when X = c+1
    # if X >= c, check if it's out of bound when X = c-1
    test_value = c + 1 if is_upper_bound else c - 1
    for i,b in zip(idx.src, (width, height)):
      if i.is_increasing():
        rw = i.substitute({X:X.const_like(test_value)})
        if rw.vmin >= b or rw.vmax < 0:
          drop_stmt.append(stmt)
          break
  return drop_stmt

def simplify_valid_load(buf:UOp, start_idx:UOp, valid:UOp) -> UOp|None:
  idx = uop_given_valid(valid, start_idx)
  return None if idx is start_idx else buf.index(idx.valid(valid), ptr=True)

def simplify_valid_image_load(buf:UOp, idx_y:UOp, idx_x:UOp, valid:UOp) -> UOp|None:
  if not isinstance(buf.dtype, ImageDType): return None
  start_idx = UOp.vectorize(idx_x, idx_y)
  idx = uop_given_valid(valid, start_idx)
  drop_stmt = _drop_valid_stmts(valid, idx, buf.dtype.shape[0], buf.dtype.shape[1])

  if not drop_stmt and idx is start_idx: return None
  new_valid = UOp.uprod(*ss) if (ss:=[s for s in valid.split_uop(Ops.AND) if s not in drop_stmt]) else None
  idx_y, idx_x = idx.gep(1), idx.gep(0)
  return buf.index(idx_y.valid(new_valid), idx_x.valid(new_valid), ptr=True) if new_valid is not None else buf.index(idx_y, idx_x, ptr=True)

load_store_indexing = PatternMatcher([
  # image load valid idx simplification
  (UPat(Ops.INDEX, src=(UPat.var("buf"), invalid_gate)), lambda buf,x,i,cond: simplify_valid_load(buf, x, cond)),
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("valid").where(UPat.var("idx_y"), UPat(arg=Invalid)),
                                         UPat.var("valid").where(UPat.var("idx_x"), UPat(arg=Invalid)))), simplify_valid_image_load),
])

# ***** load/store grouping *****

def expand_index(ctx, buf:UOp, vec:UOp):
  # determine optimal image shapes
  if isinstance(dt:=buf.dtype, ImageDType):
    x, valid = vec.get_idx().gep(0), vec.get_valid().gep(0)
    # search for dims that drop the most valid statements
    best_drop, cands = -1, []
    for ch, cw in ImageDType.valid_dims(dt, ctx.target.arch):
      if (dropped:=len(_drop_valid_stmts(valid, cidx:=uop_given_valid(valid, UOp.vectorize((x//4)%cw, x//(4*cw))), ch, cw))) > best_drop:
        best_drop, cands = dropped, [(ch, cw, cidx)]
      elif dropped == best_drop: cands.append((ch, cw, cidx))
    # and tiebreak with indexing complexity (ie. number of nodes)
    h, w, _ = cands[0] if len(cands) == 1 else min(cands, key=lambda cand: len(cand[2].gep(1).simplify().backward_slice))
    assert buf.op is Ops.RESHAPE
    buf = buf.src[0].replace(dtype=(dtypes.imageh if dt.itemsize == 2 else dtypes.imagef)((h, w, 4))).flatten()
  if getenv("UNSAFE_DISABLE_MASK", 0): vec = vec.get_idx()
  # generate the individual indexes
  return UOp(Ops.STACK, buf.dtype, tuple(buf.index(vec.gep(i), ptr=True) for i in range(vec.dtype.count)))

def fold_expanded_index(midx:UOp):
  buf = midx.src[0].src[0]
  if not isinstance(buf.dtype, PtrDType): return None
  buf_size = buf.ptrdtype.size if buf.ptrdtype.size != -1 else buf.max_numel()
  if not all(s.src[0] is buf for s in midx.src): return None
  if not all(isinstance(s.dtype, PtrDType) for s in midx.src): return None

  # extract all the relevant offsets
  offsets_rootsrc: defaultdict[Any, dict[int, list[int]]] = defaultdict(dict)
  for i in range(len(midx.src)):
    idx: Any = midx.src[i].src[1].get_idx()
    if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: root_src, arg = idx.src[0], idx.src[1].arg
    elif idx.op is Ops.ADD and idx.src[0].op is Ops.CONST: root_src, arg = idx.src[1], idx.src[0].arg
    elif idx.op is Ops.CONST and idx.arg is Invalid: root_src, arg = "INVALID", 0
    elif idx.op is Ops.CONST: root_src, arg = "CONST", idx.arg
    else: root_src, arg = idx, 0
    root_src = (midx.src[i].src[1].get_valid(), root_src)
    offsets_rootsrc[root_src].setdefault(arg, []).append(i)

  # then rewrite everything we can into groups
  ret = []
  idxs: list[int|None] = [None]*len(midx.src)
  global_offset = 0
  no_group = getenv("DEVECTORIZE_NO_PTR_GROUP", 0)
  for offsets in offsets_rootsrc.values():
    grouped_offsets = [[x] for x in sorted(offsets.keys())] if no_group else \
      [[x for _,x in group] for _,group in itertools.groupby(enumerate(sorted(offsets.keys())), lambda x: x[1]-x[0])]
    for grp in grouped_offsets:
      # get the index offset for this element. using [0] is okay, because they are the same
      lidx = midx.src[offsets[grp[0]][0]]
      if len(grp) > 1: lidx = lidx.cast(buf.ptrdtype.base.vec(len(grp)).ptr(size=buf_size, addrspace=buf.addrspace))
      # set the idxs of the output
      for i,g in enumerate(grp):
        for oo in offsets[g]: idxs[oo] = global_offset+i
      # add this lidx to the CAT
      ret.append(lidx)
      global_offset += len(grp)
  assert None not in idxs, f"some idxs are missing {idxs}"
  # this base thing is for image, we want the CAT to be a normal pointer
  post_cat = UOp(Ops.PTRCAT, buf.ptrdtype.base.ptr(size=buf_size, addrspace=buf.addrspace).vec(global_offset), tuple(ret))
  return post_cat.gep(tuple(cast(list[int], idxs)))

def _gep_local_ptrcat(g:UOp, cat:UOp):
  if not cat.src or not all(isinstance(s.dtype, PtrDType) and s.addrspace == AddrSpace.LOCAL for s in cat.src): return None
  idx = g.arg
  if isinstance(idx, int): idx = (idx,)
  if not isinstance(idx, tuple) or len(idx) == 0 or not all(isinstance(i, int) for i in idx): return None
  if len(idx) == 1:
    off = idx[0]
    for s in cat.src:
      if off < s.dtype.base.count: return s.gep(off)
      off -= s.dtype.base.count
  return None

def cat_after_store(cat:UOp, data:UOp):
  # TODO: this is written in many places
  offset = 0
  ret: list[UOp] = []
  for s in cat.src:
    ret.append(s.store(data.gep(tuple(range(offset, offset+s.dtype.count)))))
    offset += s.dtype.count
  return UOp.group(*ret)

def stack_load(tgt:UOp, ld:UOp) -> UOp|None:
  if ld.dtype.count != len(tgt.src): return None
  if not all(isinstance(p.dtype, PtrDType) for p in tgt.src): return None
  return UOp(Ops.STACK, ld.dtype, tuple(p.load(dtype=ld.dtype.scalar()) for p in tgt.src))

def gep_on_store(gep:UOp, st:UOp, gate:UOp|None=None):
  # NOTE: we need to invert the gep here, but it may be an expanding gep
  # A duplicate destination needs an operation-aware combine. Decline it here:
  # the projection reducer below owns proven ADD partials, while every other
  # duplicate shape must remain visible instead of being silently last-wins.
  # This inversion is valid only for a full permutation of the base lanes.
  # Sparse/subset maps would lose their destination keys when the GEP is
  # removed, just as duplicate maps lose their combine semantics.
  if tuple(sorted(gep.arg)) != tuple(range(gep.src[0].dtype.vcount)): return None
  a = {}
  for i,x in enumerate(gep.arg): a[x] = i
  new_arg = tuple(x[1] for x in sorted(a.items()))
  return gep.src[0].store(st.gep(new_arg), gate)

def _reg_index(u:UOp) -> tuple[UOp, UOp]|None:
  """Return the DEFINE_REG and slot for an INDEX through an AFTER chain."""
  if u.op is not Ops.INDEX or len(u.src) < 2 or not isinstance(u.src[0].dtype, PtrDType) or u.src[0].dtype.addrspace is not AddrSpace.REG:
    return None
  base = u.src[0]
  while base.op is Ops.AFTER: base = base.src[0]
  return (base, u.src[1]) if base.op is Ops.DEFINE_REG else None

def _reg_value_info(u:UOp) -> tuple[UOp, UOp]|None:
  """Identify a REG value, retaining harmless lane/cast wrappers."""
  if (ri:=_reg_index(u)) is not None: return ri
  if u.op is Ops.LOAD and u.src and (ri:=_reg_index(u.src[0])) is not None: return ri
  if u.op in {Ops.CAST, Ops.BITCAST, Ops.GEP, Ops.RESHAPE, Ops.EXPAND} and u.src:
    infos = [x for s in u.src if (x:=_reg_value_info(s)) is not None]
    if len(infos) == 1: return infos[0]
  return None

def _reg_index_node(u:UOp) -> UOp|None:
  if _reg_index(u) is not None: return u
  if u.op is Ops.LOAD and u.src and _reg_index(u.src[0]) is not None: return u.src[0]
  if u.op in {Ops.CAST, Ops.BITCAST, Ops.GEP, Ops.RESHAPE, Ops.EXPAND}:
    nodes = [x for s in u.src if (x:=_reg_index_node(s)) is not None]
    if len(nodes) == 1: return nodes[0]
  return None

def _is_additive_reg_value(u:UOp) -> bool:
  """Prove that a REG output read follows one or more ADD accumulator updates."""
  if (idx:=_reg_index_node(u)) is None or (ri:=_reg_index(idx)) is None: return False
  reg, _ = ri
  updates = []
  for dep in idx.src[0].backward_slice:
    if dep.op is not Ops.STORE or len(dep.src) < 2 or (target:=_reg_index(dep.src[0])) is None or target[0] is not reg: continue
    if reg in dep.src[1].backward_slice_with_self: updates.append(dep.src[1].op)
  return bool(updates) and all(op is Ops.ADD for op in updates)

def _reg_value_leaves(u:UOp) -> list[tuple[UOp, UOp, UOp]]:
  """Find maximal scalar REG-value leaves in an arbitrary scalar expression."""
  if u.dtype.count == 1 and (ri:=_reg_value_info(u)) is not None: return [(u, *ri)]
  return [leaf for s in u.src for leaf in _reg_value_leaves(s)]

def _destination_groups(arg:tuple[int, ...]) -> list[tuple[int, list[int]]]:
  groups:dict[int, list[int]] = defaultdict(list)
  for pos, dst in enumerate(arg): groups[dst].append(pos)
  return sorted(groups.items())

def _store_target_for_keys(base:UOp, keys:tuple[int, ...]) -> UOp|None:
  # The downstream STORE-GEP inversion can remove only a full permutation.
  # Keeping a sparse GEP temporarily would not be fail-closed: a later folding
  # pass would erase the selected keys and store the compact value to the base.
  return base if keys == tuple(range(base.dtype.vcount)) else None

def _reduce_scalar_reg_group(lanes:list[UOp]) -> tuple[UOp, UOp]|None:
  """Normalize and ADD-reduce one group of scalar additive-REG expressions."""
  lane_info = []
  for lane in lanes:
    leaves = list(dict.fromkeys(_reg_value_leaves(lane)))
    if len(leaves) != 1 or not _is_additive_reg_value(leaves[0][0]): return None
    lane_info.append(leaves[0])
  # Scalarized REG loads carry distinct INDEX slots; vector REG reads carry a
  # shared slot plus distinct GEP lanes. The maximal value leaf covers both.
  if len({leaf for leaf,_,_ in lane_info}) != len(lane_info): return None
  regs = {reg for _,reg,_ in lane_info}
  if len(regs) != 1: return None
  common_leaf = lane_info[0][0]
  normalized = [lane.substitute({leaf:common_leaf}, walk=True) for lane,(leaf,_,_) in zip(lanes, lane_info)]
  if len(set(normalized)) != 1: return None
  summed = functools.reduce(lambda a,b: a+b, (leaf for leaf,_,_ in lane_info))
  return normalized[0].substitute({common_leaf:summed}, walk=True), next(iter(regs))

def _reduce_scalarized_reg_partials(base:UOp, arg:tuple[int, ...], st:UOp) -> UOp|None:
  """Reduce a post-devectorization STACK by normalizing one REG leaf per lane.

  Replacing each lane's accumulator leaf with a common leaf makes every other
  dependency part of the proof: residuals/biases must be group-uniform, while
  nonlinear expressions derived from that same accumulator remain intact.
  """
  if st.op is not Ops.STACK or len(st.src) != len(arg): return None
  groups = _destination_groups(arg)
  if not groups or min(len(pos) for _,pos in groups) < 2 or len({len(pos) for _,pos in groups}) != 1: return None
  values:list[UOp] = []
  common_reg = None
  for _, pos in groups:
    if (reduced:=_reduce_scalar_reg_group([st.src[p] for p in pos])) is None: return None
    value, reg = reduced
    if common_reg is not None and reg is not common_reg: return None
    common_reg = reg
    values.append(value)
  target = _store_target_for_keys(base, tuple(key for key,_ in groups))
  if target is None: return None
  value = values[0] if len(values) == 1 else UOp(Ops.STACK, st.dtype.scalar().vec(len(values)), tuple(values))
  return target.store(value)

def reduce_duplicate_output_store(gep:UOp, st:UOp) -> UOp|None:
  """Preserve additive REG projection partials before generic GEP-store inversion drops duplicates."""
  base = gep.src[0]
  global_index = base.op is Ops.INDEX and bool(base.src) and isinstance(base.src[0].dtype, PtrDType) and base.src[0].dtype.addrspace is AddrSpace.GLOBAL
  global_ptrcat = base.op is Ops.PTRCAT and bool(base.src) and all(isinstance(x.dtype, PtrDType) and x.dtype.addrspace is AddrSpace.GLOBAL for x in base.src)
  if not (global_index or global_ptrcat): return None
  if not isinstance(gep.arg, tuple) or st.dtype.count != len(gep.arg): return None
  if (scalarized:=_reduce_scalarized_reg_partials(base, gep.arg, st)) is not None: return scalarized
  keyed_groups = _destination_groups(gep.arg)
  groups = [pos for _,pos in keyed_groups]
  if not groups or len(groups[0]) < 2 or not all(len(pos) == len(groups[0]) for pos in groups): return None
  def is_reg_value(x:UOp) -> bool:
    return x.dtype.count == st.dtype.count and _reg_value_info(x) is not None and _is_additive_reg_value(x)
  reg, other = (st, None) if is_reg_value(st) else (None, None)
  if reg is None and st.op in {Ops.ADD, Ops.MUL} and len(st.src) == 2:
    candidates = [(candidate, other) for candidate,other in ((st.src[0], st.src[1]), (st.src[1], st.src[0])) if is_reg_value(candidate)]
    if len(candidates) == 1: reg, other = candidates[0]
  if reg is None: return None

  # Every non-REG lane must either be a proven broadcast within each output
  # group, or the SiLU reciprocal derived from the same projection value.
  if other is not None and reg not in other.backward_slice_with_self:
    for pos in groups:
      projected = [other.src[0].gep((other.arg[p],)) if other.op is Ops.GEP and isinstance(other.arg, tuple) else other.gep((p,)) for p in pos]
      if len(set(projected)) != 1: return None
  elif other is not None and not (st.op is Ops.MUL and other.op is Ops.RECIPROCAL): return None

  replacement_lanes:list[UOp|None] = [None] * st.dtype.count
  for pos in groups:
    partials = [reg.gep((p,)) for p in pos]
    if len(set(partials)) != len(partials): return None
    value = functools.reduce(lambda a,b: a+b, partials)
    for p in pos: replacement_lanes[p] = value
  replacement = UOp(Ops.STACK, reg.dtype, tuple(cast(UOp, x) for x in replacement_lanes))
  reduced = st.substitute({reg: replacement}, walk=True)
  target = _store_target_for_keys(base, tuple(key for key,_ in keyed_groups))
  return target.store(reduced.gep(tuple(pos[0] for pos in groups))) if target is not None else None

pm_reduce_duplicate_output_store = PatternMatcher([
  (UPat(Ops.STORE, src=(UPat(Ops.GEP, name="gep"), UPat.var("st"))), reduce_duplicate_output_store),
])

load_store_folding = PatternMatcher([
  (UPat(Ops.PTRCAT, name="cat"), lambda cat: cat.src[0] if len(cat.src) == 1 and cat.dtype == cat.src[0].dtype else None),
  (UPat(Ops.INDEX, src=(UPat(Ops.STACK, src=UPat(name="buf")), UPat.var("vec"))), expand_index),
  (UPat(Ops.STACK, src=UPat(Ops.INDEX), name="midx"), fold_expanded_index),
  (UPat(Ops.GEP, src=(UPat(Ops.PTRCAT, name="cat"),), name="g"), _gep_local_ptrcat),
  # GEP after LOAD
  (UPat(Ops.LOAD, src=(UPat(Ops.GEP, name="gep"),), name="ld", allow_any_len=True),
   lambda gep, ld: ld.replace(dtype=ld.dtype.scalar().vec(gep.dtype.count), src=(gep.src[0],)+ld.src[1:]).gep(gep.arg)),
  (UPat(Ops.LOAD, src=(UPat(Ops.STACK, name="tgt"),), name="ld"), stack_load),
  # GEP on data of STORE
  (UPat(Ops.STORE, src=(UPat(Ops.GEP, name="gep"), UPat.var("st"))), reduce_duplicate_output_store),
  (UPat(Ops.STORE, src=(UPat(Ops.GEP, name="gep"), UPat.var("st"), UPat.var("gate"))), gep_on_store),
  (UPat(Ops.STORE, src=(UPat(Ops.GEP, name="gep"), UPat.var("st"))), gep_on_store),
  # put PTRCAT after LOAD
  (UPat(Ops.LOAD, src=(UPat(Ops.PTRCAT, name="cat"),), name="ld", allow_any_len=True),
   lambda cat,ld: UOp(Ops.VCAT, cat.dtype.base.vec(cat.dtype.vcount), tuple(ld.replace(dtype=x.dtype.base, src=(x,)+ld.src[1:]) for x in cat.src))),
  # put PTRCAT after STORE
  (UPat(Ops.STORE, src=(UPat(Ops.PTRCAT, name="cat"), UPat(name="data"))), cat_after_store),
])

# *** correct load/store ***

def split_load_store(ctx:Renderer|None, ls:UOp, idx:UOp):
  # this splits loads and stores into multiple chunks

  # if there's only one element to load/store, no splitting needed
  sz = max(ls.src[0].dtype.count, ls.dtype.count if ls.op is Ops.LOAD else ls.src[1].dtype.count)
  if sz == 1: return None
  buf = idx.src[0]

  # determine fold lengths
  lengths = []
  must_divide = True
  local_widths = () if ctx is None or ls.op is not Ops.STORE or buf.addrspace != AddrSpace.LOCAL else \
    ctx.local_store_vector_widths.get(buf.dtype.base, ())
  if local_widths:
    lengths = list(local_widths)
    must_divide = ctx.local_store_requires_static_alignment
  elif ctx is not None and ctx.target.device == "DSP":
    lengths = [128,64,32,16,8,4]
    must_divide = False
  elif buf.addrspace == AddrSpace.GLOBAL and buf.dtype.base in (dtypes.uint32, dtypes.uint16) and ctx is not None and ctx.supports_float4:
    # Native packed storage uses the same generic b128/b64 memory carriers.
    lengths = [16//buf.dtype.base.itemsize, 8//buf.dtype.base.itemsize]
  elif buf.dtype.base not in (dtypes.float, dtypes.half, *dtypes.fp8s) and not isinstance(buf.dtype, ImageDType):
    pass
  elif buf.addrspace == AddrSpace.REG:
    pass
  elif isinstance(buf.dtype, ImageDType):
    lengths = [4]
  elif ctx is not None and ctx.supports_float4:
    # TODO: a better way to get this than ctx
    lengths = [8,4,2] if buf.dtype.base == dtypes.half and getenv("ALLOW_HALF8") else [4,2]
  lengths.append(1)  # worst case, it's not folded

  # filter fold lengths that don't divide
  offset, mask = idx.src[1].get_idx(), idx.src[1].get_valid()
  if must_divide: lengths = [x for x in lengths if offset.divides(x) is not None]

  # split based on the fold lengths
  global_offset = 0
  # Packed LDS stages use a byte-addressed LOCAL arena (uchar pointer), while
  # the value being split is expressed in its scalar dtype.  Advancing a half
  # lane therefore advances two bytes, not one.  Typed pointers already use
  # their element units, so keep this correction narrowly scoped to byte LOCAL
  # storage.
  elem_bytes = (ls.src[1].dtype.scalar().itemsize if ls.op is Ops.STORE else ls.dtype.scalar().itemsize)
  logical_count = ls.src[1].dtype.count if ls.op is Ops.STORE else ls.dtype.count
  # The affected producer fields are exactly half2 metadata records.  Keep
  # larger fragment carriers on their existing packed paths; scalarizing them
  # here would needlessly multiply the full WMMA kernel.
  byte_local = buf.addrspace == AddrSpace.LOCAL and buf.dtype.base.itemsize == 1 and elem_bytes > 1 and logical_count == 2
  ret = []
  buf_size = buf.ptrdtype.size if isinstance(buf.dtype, PtrDType) and buf.ptrdtype.size != -1 else buf.max_numel()
  while global_offset < sz:
    # with 1 at the end of the lengths list, this will always hit
    for fold_length in lengths:
      if global_offset+fold_length > sz: continue
      # A byte-backed pointer cannot represent a typed multi-element pointer
      # cast without changing the carrier width as well.  Scalarize these
      # mixed-width LOCAL stores; the byte stride correction above then gives
      # each scalar lane its true address.
      if byte_local and fold_length > 1: continue
      chunk_offset = offset + global_offset * elem_bytes if byte_local else offset + global_offset
      if fold_length > 1 and (chunk_offset.vmin < 0 or chunk_offset.vmax + fold_length > buf_size): continue
      lidx = buf.index(chunk_offset.valid(mask), ptr=True)
      if fold_length > 1: lidx = lidx.cast(buf.ptrdtype.base.vec(fold_length).ptr(size=buf_size, addrspace=buf.addrspace))
      if ls.op is Ops.STORE:
        ret.append(ls.replace(src=(lidx,ls.src[1].gep(tuple(range(global_offset, global_offset+fold_length))))+ls.src[2:]))
      else: ret.append(ls.replace(src=(lidx,)+ls.src[1:], dtype=ls.dtype.scalar().vec(fold_length)))
      global_offset += fold_length
      break

  # if it wasn't split, we return None. otherwise we CAT them
  if len(ret) == 1: return ret[0] if ls.src[0].dtype.count == 1 and ret[0] is not ls else None
  return UOp(Ops.VCAT, ls.dtype, tuple(ret)) if ls.op is Ops.LOAD else UOp.group(*ret)

def get_image_idx(idx:UOp, width:int):
  x, valid = idx.src[1].get_idx(), idx.src[1].get_valid()
  idx_x, idx_y = (x // 4) % width, x // (4*width)
  assert idx.src[0].op is Ops.RESHAPE, "image idx must be on reshape"
  return idx.replace(src=(idx.src[0].src[0], idx_y.valid(valid), idx_x.valid(valid)))

def image_fixup(ls:UOp):
  # normal image load or store, with the CAST from expand_index
  if isinstance(dt:=ls.src[0].src[0].dtype, ImageDType) and ls.src[0].op is Ops.CAST:
    assert ls.src[0].dtype.count == 4, "image must be casted to 4"
    return ls.replace(src=(get_image_idx(ls.src[0].src[0], dt.shape[1]),)+ls.src[1:])

  # this is an unprocessed image without a cast, we should just make it a buffer
  if isinstance(dt, ImageDType) and len(ls.src[0].src) == 2:
    off = ls.src[0].src[1]
    assert ls.src[0].src[0].op is Ops.RESHAPE, "image idx must be on reshape"
    idx = ls.src[0].src[0].src[0].replace(dtype=(new_dt:=dtypes.half if dt.itemsize == 2 else dtypes.float).ptr(dt.size)).index(off)
    return ls.replace(src=(idx,), dtype=new_dt).cast(dtypes.float) if ls.op is Ops.LOAD else ls.replace(src=(idx, ls.src[1].cast(new_dt)))

def split_indexed_load_store(ctx:Renderer|None, ls:UOp, idx:UOp):
  return split_load_store(ctx, ls, idx) if idx.op is Ops.INDEX else None

correct_load_store = PatternMatcher([
  # split LOAD/STORE
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat.var("idx"),), name="ls", allow_any_len=True), split_indexed_load_store),
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat(Ops.INDEX, name="idx").cast(),), name="ls", allow_any_len=True), split_load_store),
  # image indexing, including unfoldable images
  (UPat((Ops.LOAD, Ops.STORE), name="ls"), image_fixup),
])

# *** uop expander ***

# TODO: there's a lot shared with gep_through_wmma here
def no_vectorized_wmma(wmma:UOp):
  out_sz = prod(x[1] for x in wmma.arg[6][-1])
  if wmma.dtype.count == out_sz: return None
  tsrcs = []
  for s,sz in zip(wmma.src, wmma.arg[6]):
    ssz = prod(x[1] for x in sz)
    tsrcs.append([s.gep(tuple(range(grp, grp+ssz))) for grp in range(0, s.dtype.count, ssz)])
  wmmas = [UOp(Ops.WMMA, wmma.dtype.scalar().vec(out_sz), tsrc, wmma.arg) for tsrc in zip(*tsrcs)]
  wmma_ex = flatten([[e.gep(i) for i in range(out_sz)] for e in wmmas])
  return UOp(Ops.STACK, wmma.dtype, tuple(wmma_ex))

def no_vectorized_alu(alu:UOp):
  if alu.dtype.vcount == 1: return None
  # Native fragments are vector-valued register files, not ordinary ALUs.
  # Keep their consumer projections intact until native-fragment lowering.
  if any(getattr(s, "tag", None) and s.tag[0] == "native_fragment_carrier_v1" for s in alu.src): return None
  if alu.op is Ops.WHERE and alu.src[2].arg is Invalid: return None  # image load/store has cond.where(idx.vec(2), Invalid) as the index
  alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg) for i in range(alu.dtype.vcount))
  return UOp(Ops.STACK, alu.dtype, alus)

def _output_load_lane(u:UOp) -> tuple[UOp, int]|None:
  """Recover (GLOBAL INDEX, vector lane) from an output address consumed as a LOAD."""
  if u.op is not Ops.GEP or not isinstance(u.arg, tuple) or len(u.arg) != 1: return None
  ld = u.src[0]
  if ld.op is not Ops.LOAD: return None
  idx = ld.src[0]
  if idx.op is Ops.CAST: idx = idx.src[0]
  if idx.op is not Ops.INDEX or len(idx.src) < 2: return None
  if getattr(idx.src[0], "addrspace", None) is not AddrSpace.GLOBAL: return None
  return (idx, u.arg[0])

def _uniform_contiguous_groups(items:list[UOp]) -> list[list[int]]|None:
  groups, i = [], 0
  while i < len(items):
    j = i
    while j < len(items) and items[j] is items[i]: j += 1
    groups.append(list(range(i, j))); i = j
  return groups if groups and all(len(pos) == len(groups[0]) for pos in groups) else None

def _sum_distinct_lanes(val:UOp, pos:list[int]) -> UOp|None:
  lanes = [val.gep((p,)) for p in pos]
  return functools.reduce(lambda a,b: a+b, lanes) if len(set(lanes)) == len(lanes) else None

def devectorize_bare_output_store(tgt:UOp, val:UOp, gate:UOp|None=None) -> UOp|None:
  """Restore GLOBAL addresses that add_loads consumed as scalar LOAD targets."""
  if val.dtype.count != len(tgt.src) or gate is not None and gate.dtype.count != len(tgt.src): return None
  addresses = []
  for lane in tgt.src:
    if lane.op is not Ops.LOAD or not lane.src or lane.src[0].op is not Ops.INDEX: return None
    idx = lane.src[0]
    if getattr(idx.src[0], "addrspace", None) is not AddrSpace.GLOBAL: return None
    addresses.append(idx)
  groups = _uniform_contiguous_groups(list(tgt.src))
  if groups is None: return None
  stores = []
  for pos in groups:
    if len(pos) == 1: value = val.gep(pos[0])
    elif gate is not None: return None
    else:
      lanes = [val.gep((p,)) for p in pos]
      if len(set(lanes)) == 1: value = lanes[0]
      elif val.op is Ops.STACK and (reduced:=_reduce_scalar_reg_group([val.src[p] for p in pos])) is not None: value = reduced[0]
      elif _is_additive_reg_value(val) and len(set(lanes)) == len(lanes): value = functools.reduce(lambda a,b: a+b, lanes)
      else: return None
    stores.append(addresses[pos[0]].store(value, gate.gep(pos[0]) if gate is not None else None))
  return UOp.group(*stores)

def devectorize_output_projection_store(tgt:UOp, val:UOp) -> UOp|None:
  """Restore GLOBAL output addresses and ADD-reduce repeated UPCAST partial groups.

  This is intentionally ADD-only and GLOBAL-only: its validated producer is an
  additive matmul epilogue. LOCAL/REG lanes may carry MAX or MUL reductions, so
  they fail closed. Uniform repeated groups and distinct partial values are
  required; nonuniform groups and broadcasts are declined.
  """
  if val.dtype.count != len(tgt.src): return None
  info = [_output_load_lane(p) for p in tgt.src]
  if any(x is None for x in info): return None
  groups = _uniform_contiguous_groups(list(tgt.src))
  if groups is None: return None
  g = len(groups[0])
  if g < 2: return None
  stores = []
  for pos in groups:
    idx, lane = info[pos[0]]
    addr = idx.src[0].index(idx.src[1] + UOp.const(idx.src[1].dtype, lane))
    if (value:=_sum_distinct_lanes(val, pos)) is None: return None
    stores.append(addr.store(value))
  return UOp.group(*stores)

pm_output_projection_store = PatternMatcher([
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"), UPat.var("gate"))), devectorize_bare_output_store),
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"))), devectorize_bare_output_store),
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"))), devectorize_output_projection_store),
])

def scalarize_shaped_store(store:UOp) -> UOp|None:
  """Devectorize shaped scatter stores while preserving exact address/value lanes."""
  # EXP's older lowering already handles ordinary contiguous vector stores.
  # The missing upstream behavior is specifically a shaped scatter destination;
  # widening this to every shaped STORE perturbs the established GEP pipeline.
  if store.src[0].op is not Ops.STACK or store.shape == (): return None
  # Broadcasting must already be unpacked. Invalid is a scalar sentinel that
  # intentionally applies to every lane, matching upstream do_devectorize.
  if not all(source.shape == store.shape or source.base.arg is Invalid for source in store.src): return None
  # The older movement pipeline represents this scatter as flat STACKs.
  if len(store.shape) != 1 or store.src[1].op is not Ops.STACK: return None
  lanes = int(store.shape[0])
  if len(store.src[0].src) != lanes or len(store.src[1].src) != lanes: return None
  stores = []
  for i in range(lanes):
    src = (store.src[0].src[i], store.src[1].src[i])
    if len(store.src) == 3:
      gate = store.src[2].base if store.src[2].base.arg is Invalid else \
        store.src[2].src[i] if store.src[2].op is Ops.STACK else store.src[2].index(UOp.const(dtypes.weakint, i))
      src += (gate,)
    stores.append(store.replace(src=src))
  return UOp.group(*stores)

def _keep_register_tag(tag) -> bool: return isinstance(tag, RegisterResidentAccumulator) or isinstance(tag, tuple) and tag and tag[0] in ("wmma_frag_buffer_proof", "register_pipe_stage_buffer")

def no_vectorized_buf(buf:UOp):
  # TODO: this fails on regs
  #assert buf.max_numel() == buf.ptrdtype.size
  out = buf.replace(dtype=buf.ptrdtype.base.scalar().ptr(buf.ptrdtype.size*buf.ptrdtype.count, buf.addrspace)).cast(buf.dtype)
  return out.replace(tag=buf.tag) if _keep_register_tag(buf.tag) else out

def no_vectorized_index(buf:UOp, cast:UOp, idx:UOp, bcast:UOp|None=None):
  cnt = cast.dtype.count
  if bcast is not None and bcast.op is Ops.GEP:
    # GEP selects specific lanes; bcast.arg[k] is the offset for lane k, iterate groups × selected lanes
    pairs = [(k, g + bcast.arg[k]) for g, k in itertools.product(range(cast.dtype.vcount), range(len(bcast.arg)))]
  elif bcast is not None:
    # BROADCAST: cross product of components × lanes
    pairs = [(j, c) for c, j in itertools.product(range(cnt), range(bcast.dtype.vcount))]
  else:
    # simple scalar index: one lane, all components
    pairs = [(0, c) for c in range(cnt)]
  idx_lanes, offsets = (tuple(x) for x in zip(*pairs))
  out = buf.broadcast(len(pairs)).index(idx.gep(idx_lanes)*cnt + UOp.const(dtypes.weakint.vec(len(pairs)), offsets), ptr=True)
  return out.replace(tag=buf.tag) if _keep_register_tag(buf.tag) else out

devectorize_buf_and_index = PatternMatcher([
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG), name="buf"), no_vectorized_buf),
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG)).or_after(name="buf").cast(name="cast").index(UPat.var("idx")), no_vectorized_index),
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG)).or_after(name="buf").cast(name="cast").broadcast(name="bcast").index(UPat.var("idx")),
   no_vectorized_index),
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG)).or_after(name="buf").cast(name="cast").gep(name="bcast").index(UPat.var("idx")),
   no_vectorized_index),
])

devectorize_alu = PatternMatcher([
  # CAST after AFTER
  (UPat(Ops.CAST, name="c").f(Ops.AFTER, allow_any_len=True, name="a"),
   lambda c,a: c.src[0].after(*a.src[1:]).cast(c.dtype)),
  # no ALU on vectorized dtypes
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name="alu"), no_vectorized_alu),
  (UPat(Ops.WMMA, name="wmma"), no_vectorized_wmma),
])

# Keep shaped STORE ownership separate from value devectorization: output-projection
# reductions must recover repeated destinations before the fallback scalarizes them.
devectorize_store = PatternMatcher([
  (UPat(Ops.STORE, name="store"), scalarize_shaped_store),
])

pm_render = PatternMatcher([
  # preserve AFTER ordering while scalarizing a vector value for rendering
  (UPat(Ops.AFTER, name="a").f(Ops.GEP, name="gep"), lambda gep,a:
   a.replace(dtype=gep.dtype, src=(a.src[0].gep(gep.arg),)+a.src[1:])),
  # for rendering, we use explicit VECTORIZE
  (UPat(Ops.CONST, name='c'),
   lambda c: UOp(Ops.STACK, c.dtype, (UOp.const(c.dtype.scalar(), c.arg),)*c.dtype.vcount) if c.dtype.vcount > 1 else None),
  (UPat(Ops.GEP, name='gep'), lambda gep: UOp(Ops.STACK, gep.dtype, tuple(gep.src[0].gep(x) for x in gep.arg)) if len(gep.arg) > 1 else None),
  (UPat(Ops.GEP, name='gep'), lambda gep: gep.src[0] if gep.src[0].dtype.vcount == 1 and gep.arg == (0,) else None),
  (UPat(Ops.STACK, src=(UPat(name='x'),)), lambda x: x),
])

# REDUCE->ACC lowering moved to reduce_lowering.py
# manual-acc upcast fix + REG-store devec moved to reg_store.py

# add loads

def add_load(idx:UOp):
  if isinstance(idx.dtype, PtrDType): return None
  if not isinstance(idx.src[0].dtype, PtrDType):
    raise RuntimeError(f"invalid composite reduction slot: INDEX owner is not pointer-typed ({idx.src[0].dtype})")
  return idx.replace(dtype=idx.src[0].dtype).load(dtype=idx.dtype.base)

pm_add_loads = PatternMatcher([
  # add loads to non ptr index
  (UPat(Ops.INDEX, name="idx"), add_load),
  # remove loads from stores
  (UPat(Ops.STORE, src=(UPat(Ops.LOAD),), allow_any_len=True, name="s"), lambda s: s.replace(src=(s.src[0].src[0],)+s.src[1:])),
  (UPat(Ops.LOAD, src=(UPat(Ops.LOAD),), allow_any_len=True, name="l"), lambda l: l.replace(src=(l.src[0].src[0],)+l.src[1:])),
])

# make images

pm_imageh_store = PatternMatcher([
  # store<imageh>(idx, x) is actually store(idx, x.cast(half)) so we can pull the cast into the store
  (UPat.var("x", dtypes.float).cast(dtypes.half), lambda x: x),
  # store(imageh, a.where(b.half(), c).float()) -> store(imageh, a.where(b, c.float()))
  (UPat(Ops.WHERE, src=(UPat.var("a"), UPat.var("b", dtypes.float).cast(dtypes.half), UPat.var("c"))), lambda a,b,c: a.where(b,c.cast(dtypes.float))),
  # otherwise, we cast to float
  (UPat(GroupOp.All, name="x"), lambda x: x.cast(dtypes.float))
])

def make_image(ctx, ls, buf, off):
  if (vcount:=buf.dtype.vcount) != 1: buf = buf.src[0]
  if buf.op == Ops.PARAM and not isinstance(dt:=buf.dtype, ImageDType) and (dims:=ImageDType.valid_dims(dt, ctx)):
    buf = buf.replace(dtype=(dtypes.imageh if dt.base == dtypes.half else dtypes.imagef)((*dims[0], 4))).flatten()
    if vcount != 1: buf = UOp.vectorize(*([buf] * vcount))
    if ls.op is Ops.LOAD: return ls.replace(src=(buf.index(off, ptr=True),), dtype=dtypes.float.vec(ls.dtype.vcount)).cast(dt.base)
    return buf.index(off, ptr=True).store(pm_imageh_store.rewrite(ls.src[1]) if dt.base == dtypes.half else ls.src[1])

pm_make_images = PatternMatcher([
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("off"))),), allow_any_len=True, name="ls"), make_image),
  # load<imageh> is actually load<half>.cast(float), so load<imageh>.half().float() -> load<half>.float().half().float() -> load<half>.float()
  (UPat(Ops.LOAD, name="li").cast(dtypes.half).cast(dtypes.float), lambda li: li if isinstance(li.src[0].dtype, ImageDType) else None),
])
