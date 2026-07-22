from typing import Any, cast
import functools, itertools
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.dtype import dtypes, ImageDType, DType, AddrSpace, Invalid, PtrDType
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, GroupOp, RegisterResidentAccumulator, identity_element, AxisType
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
  # fake argsort. TODO: handle duplicates
  a = {}
  for i,x in enumerate(gep.arg): a[x] = i
  new_arg = tuple(x[1] for x in sorted(a.items()))
  return gep.src[0].store(st.gep(new_arg), gate)

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
  if alu.op is Ops.WHERE and alu.src[2].arg is Invalid: return None  # image load/store has cond.where(idx.vec(2), Invalid) as the index
  alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg) for i in range(alu.dtype.vcount))
  return UOp(Ops.STACK, alu.dtype, alus)

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

pm_render = PatternMatcher([
  # preserve AFTER ordering while scalarizing a vector value for rendering
  (UPat(Ops.AFTER, name="a").f(Ops.GEP, name="gep"), lambda gep,a: a.src[0].gep(gep.arg).after(*a.src[1:])),
  # for rendering, we use explicit VECTORIZE
  (UPat(Ops.CONST, name='c'),
   lambda c: UOp(Ops.STACK, c.dtype, (UOp.const(c.dtype.scalar(), c.arg),)*c.dtype.vcount) if c.dtype.vcount > 1 else None),
  (UPat(Ops.GEP, name='gep'), lambda gep: UOp(Ops.STACK, gep.dtype, tuple(gep.src[0].gep(x) for x in gep.arg)) if len(gep.arg) > 1 else None),
  (UPat(Ops.GEP, name='gep'), lambda gep: gep.src[0] if gep.src[0].dtype.vcount == 1 and gep.arg == (0,) else None),
  (UPat(Ops.STACK, src=(UPat(name='x'),)), lambda x: x),
])

# *** Ops.REDUCE -> Ops.DEFINE_ACC ***

@dataclass
class ReduceContext:
  acc_num: int = 0

def horizontal_reduce(inp:UOp, out_dtype:DType) -> list[UOp]:
  # if this has a horizontal reduction component, do that first
  if inp.dtype != out_dtype:
    # NOTE: [0 1 2 3 4 5 6 7] -> [0+4, 1+5, 2+6, 3+7]
    horizontal_amount = inp.dtype.count//out_dtype.count
    return [inp.gep(tuple(range(i, inp.dtype.count, horizontal_amount))) for i in range(0, horizontal_amount)]
  return [inp]

def reduce_to_acc(ctx:ReduceContext, red:UOp):
  inp, reduce_range = red.src[0], red.src[1:]

  # Composite reduce with no ranges yet (pre-rangeify): pass through unchanged
  from tinygrad.uop.ops import CompositeReduce
  if isinstance(red.arg[0], CompositeReduce) and len(reduce_range) == 0:
    return None  # let rangeify expand it first

  lst = horizontal_reduce(inp, red.dtype)
  assert all(x.dtype == red.dtype for x in lst), f"horizontal reduction mismatch {lst[0].dtype} != {red.dtype}"
  # if we have a range
  if len(reduce_range) != 0:
    topo = inp.toposort()
    ended_ranges = flatten([x.ended_ranges for x in topo if x.op is Ops.END])
    input_ranges = tuple([x for x in topo if x.op is Ops.RANGE and x not in reduce_range and x not in ended_ranges])

    # Check for composite reduce (multi-accumulator)
    from tinygrad.uop.ops import CompositeReduce
    if isinstance(red.arg[0], CompositeReduce):
      composite = red.arg[0]
      
      # Coupled combine: online-softmax (m,l,acc) with correction
      if composite.combine_fn == "online_softmax":
        # Slots: m (MAX), l (ADD), acc (ADD). Input: vec2(score, v)
        assert len(composite.slots) == 3
        LOG2E = red.const(red.dtype, 1.4426950408889634)
        NEG1 = red.const(red.dtype, -1.0)
        
        inp_score = inp if inp.dtype.count == 1 else inp.gep(0)
        inp_v = inp if inp.dtype.count == 1 else inp.gep(1)
        
        accs = []
        acc_reads = []
        for i, slot in enumerate(composite.slots):
          ident = red.const(slot.dtype, slot.identity if slot.identity is not None else identity_element(slot.op, slot.dtype.scalar()))
          acc = UOp.placeholder((1,), slot.dtype, ctx.acc_num, AddrSpace.REG)
          ctx.acc_num += 1
          acc_init = acc.after(*input_ranges).index(UOp.const(dtypes.weakint, 0)).store(ident)
          acc_read = acc.after(acc_init, *reduce_range).index(UOp.const(dtypes.weakint, 0))
          accs.append(acc)
          acc_reads.append(acc_read)
        
        m_old, l_old, acc_old = acc_reads
        
        # m_new = max(m_old, score)
        m_new = m_old.alu(Ops.MAX, inp_score)
        # diff = m_old - m_new (= m_old + (-1)*m_new)
        diff = m_old.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
        # correction = exp(diff) = exp2(diff * log2(e))
        corr = diff.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
        # score_shifted = score - m_new
        score_shifted = inp_score.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
        # exp_score = exp(score_shifted)
        exp_score = score_shifted.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
        # l_new = l_old * corr + exp_score
        l_new = l_old.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score)
        # acc_new = acc_old * corr + exp_score * v
        acc_new = acc_old.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score.alu(Ops.MUL, inp_v))
        
        results = []
        for acc, new_val in zip(accs, [m_new, l_new, acc_new]):
          acc_end = acc.index(UOp.const(dtypes.weakint, 0)).store(new_val).end(*reduce_range).rtag("mergeable")
          results.append(acc.after(acc_end).index(UOp.const(dtypes.weakint, 0)))
        
        # Return acc / l as the attention output (multi-output solved via division)
        rcp_l = results[1].alu(Ops.RECIPROCAL)
        return results[2].alu(Ops.MUL, rcp_l)
      
      # Coupled combine: online-softmax, l-only. Slots: m (MAX), l (ADD). Input: score scalar.
      if composite.combine_fn == "online_softmax_l":
        assert len(composite.slots) == 2
        LOG2E = red.const_like(1.4426950408889634, dtype=red.dtype.scalar())
        NEG1 = red.const_like(-1.0, dtype=red.dtype.scalar())
        
        inp_score = inp  # scalar per-element input from REDUCE loop
        
        accs = []
        acc_reads = []
        for i, slot in enumerate(composite.slots):
          ident = red.const(slot.dtype, slot.identity if slot.identity is not None else identity_element(slot.op, slot.dtype.scalar()))
          acc = UOp.placeholder((1,), slot.dtype, ctx.acc_num, AddrSpace.REG)
          ctx.acc_num += 1
          acc_init = acc.after(*input_ranges).index(UOp.const(dtypes.weakint, 0)).store(ident)
          acc_read = acc.after(acc_init, *reduce_range).index(UOp.const(dtypes.weakint, 0))
          accs.append(acc)
          acc_reads.append(acc_read)
        
        m_old, l_old = acc_reads
        
        # m_new = max(m_old, score)
        m_new = m_old.alu(Ops.MAX, inp_score)
        # diff = m_old - m_new
        diff = m_old.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
        # correction = exp(diff) = exp2(diff * log2(e))
        corr = diff.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
        # score_shifted = score - m_new
        score_shifted = inp_score.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
        # exp_score = exp(score_shifted)
        exp_score = score_shifted.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
        # l_new = l_old * corr + exp_score
        l_new = l_old.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score)
        
        # NOTE: each slot's END must stay reachable from the returned value, or DCE drops the non-surfaced slot's
        # store (its final read is never consumed) and the loop-carried accumulator (here, m) never advances --
        # silently freezing at its identity every iteration. Anchor the return on ALL ends via .after(*ends), not
        # just the surfaced slot's own end, so merge_reduce_ends (which walks sink.backward_slice) sees every END
        # sharing this reduce_range and merges them into one real loop. (Verified: without this, l == 1.0 for
        # every input because m stays at -inf forever and the correction term is always exp(-inf) == 0.)
        ends = [acc.index(UOp.const(dtypes.weakint, 0)).store(new_val).end(*reduce_range).rtag("mergeable")
                for acc, new_val in zip(accs, [m_new, l_new])]
        results = [acc.after(end).index(UOp.const(dtypes.weakint, 0)) for acc, end in zip(accs, ends)]

        # Return l (last slot), anchored on every slot's end so m's store isn't DCE'd.
        return accs[-1].after(*ends).index(UOp.const(dtypes.weakint, 0))

      # Coupled combine: online-softmax, acc-only. Slots: m (MAX), acc (ADD, vec-Hd). Input: vec(score, v...).
      # Surfaces acc (last slot), via the same accs[-1] convention as the default independent-slots path.
      if composite.combine_fn == "online_softmax_acc":
        assert len(composite.slots) == 2
        # m/score-space math is always scalar (slot m's dtype), even though red.dtype (the surfaced acc) is vec-Hd.
        m_dtype = composite.slots[0].dtype
        LOG2E = red.const(m_dtype, 1.4426950408889634)
        NEG1 = red.const(m_dtype, -1.0)

        inp_score = inp.gep(0)
        inp_v = inp.gep(tuple(range(1, inp.dtype.count)))

        accs = []
        acc_reads = []
        for i, slot in enumerate(composite.slots):
          ident = red.const(slot.dtype, slot.identity if slot.identity is not None else identity_element(slot.op, slot.dtype.scalar()))
          acc = UOp.placeholder((1,), slot.dtype, ctx.acc_num, AddrSpace.REG)
          ctx.acc_num += 1
          acc_init = acc.after(*input_ranges).index(UOp.const(dtypes.weakint, 0)).store(ident)
          acc_read = acc.after(acc_init, *reduce_range).index(UOp.const(dtypes.weakint, 0))
          accs.append(acc)
          acc_reads.append(acc_read)

        m_old, acc_old = acc_reads

        # m_new = max(m_old, score)
        m_new = m_old.alu(Ops.MAX, inp_score)
        # diff = m_old - m_new (= m_old + (-1)*m_new)
        diff = m_old.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
        # correction = exp(diff) = exp2(diff * log2(e))
        corr = diff.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
        # score_shifted = score - m_new
        score_shifted = inp_score.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
        # exp_score = exp(score_shifted)
        exp_score = score_shifted.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
        # acc_new = acc_old * corr + exp_score * v  (exp_score broadcast across the v vector)
        exp_score_v = exp_score if inp_v.dtype.count == 1 else exp_score.broadcast(inp_v.dtype.count)
        acc_new = acc_old.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score_v.alu(Ops.MUL, inp_v))

        # See the online_softmax_l branch above for why every slot's END must stay reachable from the return value.
        ends = [acc.index(UOp.const(dtypes.weakint, 0)).store(new_val).end(*reduce_range).rtag("mergeable")
                for acc, new_val in zip(accs, [m_new, acc_new])]
        results = [acc.after(end).index(UOp.const(dtypes.weakint, 0)) for acc, end in zip(accs, ends)]

        return accs[-1].after(*ends).index(UOp.const(dtypes.weakint, 0))

      # Default: independent slots
      accs = []
      for i, slot in enumerate(composite.slots):
        ident = red.const(slot.dtype, slot.identity if slot.identity is not None else identity_element(slot.op, slot.dtype.scalar()))
        acc = UOp.placeholder((1,), slot.dtype, ctx.acc_num, AddrSpace.REG)
        ctx.acc_num += 1
        acc_init = acc.after(*input_ranges).index(UOp.const(dtypes.weakint, 0)).store(ident)
        inp_lst = horizontal_reduce(inp, slot.dtype)
        acc_read = acc.after(acc_init, *reduce_range).index(UOp.const(dtypes.weakint, 0))
        lst_slot = [acc_read] + inp_lst
        ret_slot = functools.reduce(lambda x,y: x.alu(slot.op, y), lst_slot)
        acc_end = acc.index(UOp.const(dtypes.weakint, 0)).store(ret_slot).end(*reduce_range).rtag("mergeable")
        accs.append(acc.after(acc_end).index(UOp.const(dtypes.weakint, 0)))
      return accs[-1]

    identity = red.const(red.dtype, identity_element(red.arg[0], red.dtype.scalar()))
    acc = UOp.placeholder((1,), red.dtype, ctx.acc_num, AddrSpace.REG).replace(tag=red.tag if isinstance(red.tag, RegisterResidentAccumulator) else None)
    acc_init = acc.after(*input_ranges).index(UOp.const(dtypes.weakint, 0)).store(identity)
    lst = [acc.after(acc_init, *reduce_range).index(UOp.const(dtypes.weakint, 0))] + lst  # put acc as the first element
    ctx.acc_num += 1
  ret = functools.reduce(lambda x,y: x.alu(red.arg[0], y), lst)
  if len(reduce_range) == 0: return ret
  end = acc.index(UOp.const(dtypes.weakint, 0)).store(ret).end(*reduce_range).rtag("mergeable")
  return acc.after(end).index(UOp.const(dtypes.weakint, 0))

def merge_reduce_ends(ctx:ReduceContext, sink:UOp):
  # merge ENDs that share the same range and nesting context (only those created by reduce_to_acc)
  # ENDs at different nesting depths get cloned RANGEs so each RANGE maps to one END
  range_to_ends: dict[tuple[UOp, ...], list[UOp]] = {}
  for u in sink.backward_slice:
    if u.op is Ops.END and u.tag == "mergeable": range_to_ends.setdefault(u.src[1:], []).append(u)
  subs: dict[UOp, UOp] = {}
  next_axis = max((u.arg[0] for u in sink.backward_slice if u.op is Ops.RANGE), default=-1) + 1
  for r, ends in range_to_ends.items():
    if len(ends) <= 1: continue
    by_ctx: dict[frozenset[UOp], list[UOp]] = {}
    for e in ends: by_ctx.setdefault(frozenset(e.ranges), []).append(e)
    for i, group in enumerate(by_ctx.values()):
      tr = r if i == 0 else tuple(rr.replace(arg=(next_axis + j, *rr.arg[1:])) for j, rr in enumerate(r))
      if i > 0: next_axis += len(r)
      mapped = [e.substitute(dict(zip(r, tr))) if i > 0 else e for e in group]
      merged = mapped[0] if len(mapped) == 1 else UOp.group(*(e.src[0] for e in mapped)).end(*tr)
      for e in group: subs[e] = merged
  return sink.substitute(subs) if subs else None

pm_reduce = PatternMatcher([
  # REDUCE -> DEFINE_ACC+ASSIGN, then merge ENDs with same range
  (UPat(Ops.REDUCE, name="red"), reduce_to_acc),
  (UPat(Ops.SINK, name="sink"), merge_reduce_ends),
  # tensor core built in accumulate
  (UPat(Ops.WMMA, name="wmma") + UPat.var("add"),
    lambda add, wmma: UOp(wmma.op, wmma.dtype, (wmma.src[0], wmma.src[1], wmma.src[2]+add), wmma.arg)),
])

# Manual END/AFTER scalar-REG accumulator widening for AMD generated reductions.
#
# Hand-written reductions (flash/gemv kernels) use a manual loop-carried accumulator
# `acc.index(0).store(op(acc.after(reduce_range).index(0), contrib)).end(reduce_range)` rather than Ops.REDUCE, so
# reduce_to_acc/horizontal_reduce never runs on them. When the optimizer UPCAST/UNROLLs the reduce or an output axis,
# the reduce body becomes a vector and this idiom broadcasts the size-1 scalar slot: the store target becomes
# `make_floatN(acc,...,acc) = <N partials>`, which is not assignable (and REG_STORE_DEVEC aliases the lanes -> NaN).
#
# This rewrite gives the manual accumulator the same treatment Ops.REDUCE gets: it sizes the REG to the true
# output width W (= the init-store width; how many distinct output lanes the accumulator feeds) and horizontally
# reduces the N/W reduce-axis lanes with the accumulator's own op before the (now genuine) width-W store. The rebuilt
# accumulator mirrors reduce_to_acc's SSA form exactly (input ranges on the init, single after on the read, bare store
# target, one mergeable END per accumulator merged by merge_reduce_ends). It is exact and fail-closed: it only touches
# stores whose target is a broadcast of a scalar-REG slot-0 index fed by `op(broadcast(acc), contrib)` for a supported
# op, and leaves everything else unchanged.
_reduce_acc_ops = {Ops.ADD, Ops.MAX, Ops.MUL}

def _reg_index(u:UOp) -> tuple[UOp, UOp]|None:
  # the DEFINE_REG and slot index that INDEX(after-chain(DEFINE_REG in REG space), idx) targets, else None
  if u.op is not Ops.INDEX or not isinstance(u.src[0].dtype, PtrDType) or u.src[0].dtype.addrspace != AddrSpace.REG: return None
  if len(u.src) < 2: return None
  b = u.src[0]
  while b.op is Ops.AFTER: b = b.src[0]
  return (b, u.src[1]) if b.op is Ops.DEFINE_REG else None

def _is_const_zero(u:UOp) -> bool:
  return u.op is Ops.CONST and u.arg == 0

def _reg_slot0(u:UOp) -> UOp|None:
  # the DEFINE_REG that INDEX(after-chain(DEFINE_REG in REG space), CONST 0) targets, else None
  ri = _reg_index(u)
  return ri[0] if ri is not None and _is_const_zero(ri[1]) else None

def _broadcast_elem(u:UOp) -> UOp|None:
  # the repeated element of a same-lane broadcast STACK, else None
  return u.src[0] if u.op is Ops.STACK and len(u.src) > 1 and len(set(u.src)) == 1 else None

def _manual_acc_store(store:UOp):
  # STORE(broadcast(reg slot0), op(broadcast(acc_read), contrib)) -> (reg, op, target_idx, acc_read_idx, contrib, N)
  # else None. target_idx can carry ordering deps, for example den.after(num_update)[0].
  data = store.src[1]
  te = _broadcast_elem(store.src[0])
  if te is None: te = store.src[0]
  if (tri:=_reg_index(te)) is None: return None
  reg, target_idx = tri
  target_count = len(store.src[0].src) if store.src[0].op is Ops.STACK else data.dtype.count
  if data.dtype.count != target_count: return None

  def _split_acc_contrib(u:UOp) -> tuple[UOp, UOp]|None:
    acc = contrib = None
    for s in u.src:
      se = _broadcast_elem(s)
      if se is None: se = s
      sri = _reg_index(se)
      if sri is not None and sri[0] is reg and sri[1] is target_idx and s.dtype.count in {1, data.dtype.count}: acc = se
      else: contrib = s
    return (acc, contrib) if acc is not None and contrib is not None else None

  if data.op in _reduce_acc_ops and len(data.src) == 2:
    if (sp:=_split_acc_contrib(data)) is None: return None
    acc, contrib = sp
  elif data.op is Ops.STACK and len(data.src) > 1 and data.src[0].op in _reduce_acc_ops and all(x.op is data.src[0].op and len(x.src) == 2 for x in data.src):
    accs, contribs = [], []
    for x in data.src:
      if (sp:=_split_acc_contrib(x)) is None: return None
      accs.append(sp[0]); contribs.append(sp[1])
    acc, contrib = accs[0], UOp(Ops.STACK, data.dtype, tuple(contribs))
    data = data.replace(op=data.src[0].op, src=(acc.broadcast(len(contribs)), contrib))
  else: return None
  if acc is None or contrib is None or contrib.dtype.count not in {1, data.dtype.count}: return None
  return reg, data.op, te, acc, contrib, data.dtype.count

def _acc_after_chain(idx:UOp):
  # (all after-srcs above the DEFINE_REG, the AFTER node holding the reg's input ranges)
  b, reg_gpu, extra = idx.src[0], None, []
  while b.op is Ops.AFTER:
    extra += list(b.src[1:])
    if b.src[0].op is Ops.DEFINE_REG: reg_gpu = b
    b = b.src[0]
  return extra, reg_gpu

def _is_manual_acc_init(reg:UOp, store:UOp) -> bool:
  if store.op is not Ops.STORE or len(store.src) < 2: return False
  if store.src[0].op is Ops.STACK: tgts = store.src[0].src
  else:
    te = _broadcast_elem(store.src[0])
    tgts = (te if te is not None else store.src[0],)
  return any((ri:=_reg_index(t)) is not None and ri[0] is reg for t in tgts) and reg not in store.src[1].backward_slice

def _manual_acc_init_width(reg:UOp, sink:UOp) -> int|None:
  # width of the accumulator's init store (a store to reg whose data does not depend on reg)
  for u in sink.backward_slice:
    if _is_manual_acc_init(reg, u): return u.src[1].dtype.count
  return None

def _reg_lane_stack(base:UOp, dtype:DType) -> UOp:
  return UOp(Ops.STACK, dtype, tuple(base.index(UOp.const(dtypes.weakint, i)) for i in range(dtype.count)))

def _manual_reduce_lanes(contrib:UOp, op:Ops, width:int) -> list[UOp]:
  # Manual accumulators are output-major: each output owns one contiguous group of reduction lanes.
  reduce_width = contrib.dtype.count // width
  return [functools.reduce(lambda a,b: a.alu(op, b),
                           [contrib.gep((w*reduce_width+r,)) for r in range(reduce_width)]) for w in range(width)]

def reduce_acc_upcast_fix(sink:UOp) -> UOp|None:
  subs: dict[UOp, UOp] = {}
  wide: dict[UOp, UOp] = {}
  reduce_by_reg: dict[UOp, tuple[UOp, ...]] = {}
  matches = [(store, sp) for store in sink.backward_slice if store.op is Ops.STORE and (sp:=_manual_acc_store(store)) is not None]
  match_stores = {store for store,_ in matches}
  # Process producers first. Mixed manual accumulators often encode `den.after(num_update)`, and replacements are not
  # recursively substituted inside later replacement UOps.
  matches.sort(key=lambda x: len([s for s in match_stores if s is not x[0] and s in x[0].backward_slice]))
  for store, sp in matches:
    reg, op, target, acc, contrib, N = sp
    acc_extra, reg_gpu = _acc_after_chain(acc)
    target_extra, _ = _acc_after_chain(target)
    reduce_range = tuple(r for r in acc_extra if r.op is Ops.RANGE and r.arg[1] is AxisType.REDUCE)
    _, target_idx = _reg_index(target) or (None, None)
    if target_idx is None: continue
    if not reduce_range: continue                              # fail closed: can't identify the reduce axis
    if (W:=_manual_acc_init_width(reg, sink)) is None or W < 1: continue
    if not _is_const_zero(target_idx): W = 1                   # dynamic REG slot: reduce lanes into that slot
    if N % W != 0 or (W > 1 and not _is_const_zero(target_idx)): continue
    sdt = reg.dtype.base
    if sdt.count != 1: continue                                # fail closed: only widen genuine scalar-REG accumulators
    elem_dt = sdt.vec(W) if W > 1 else sdt
    reg_wide = reg if W == 1 else wide.setdefault(reg, reg.replace(dtype=sdt.ptr(W, addrspace=AddrSpace.REG)))
    czero = UOp.const(dtypes.weakint, 0)
    def _wide_read(*deps:UOp) -> UOp:
      base = reg_wide.after(*deps) if deps else reg_wide
      return base.index(target_idx) if W == 1 else _reg_lane_stack(base, elem_dt)
    reduce_by_reg[reg] = reduce_range
    # canonical accumulator, matching reduce_to_acc: input ranges on init, single after on read, bare store target.
    # Preserve non-reduce ordering deps from the original after-chain (for example den.after(num_update) in mixed
    # accumulators), but replace any already-rewritten deps with their wide equivalents.
    init = None
    if W > 1:
      init_deps = tuple(reg_gpu.src[1:] if reg_gpu is not None else ())
      init_base = reg_wide.after(*init_deps) if init_deps else reg_wide
      ident = identity_element(op, sdt)
      init = UOp.group(*(init_base.index(UOp.const(dtypes.weakint, i)).store(UOp.const(sdt, ident)) for i in range(W)))
      for u in sink.backward_slice:
        if _is_manual_acc_init(reg, u): subs[u] = init
    dep_srcs = tuple(dict.fromkeys(target_extra + acc_extra))
    deps = tuple((subs.get(x, x) if subs else x) for x in dep_srcs
                 if x.op in {Ops.STORE, Ops.END} and x not in reduce_range and not (W > 1 and _is_manual_acc_init(reg, x)))
    read_deps = ((init,) if init is not None else ()) + deps + reduce_range
    read = _wide_read(*read_deps)
    lanes = [contrib] if contrib.dtype.count == 1 else _manual_reduce_lanes(contrib, op, W)
    hred = lanes[0] if W == 1 else UOp(Ops.STACK, elem_dt, tuple(lanes))
    upd = read.alu(op, hred)
    store_base = reg_wide.after(*deps) if deps else reg_wide
    new_store = store_base.index(target_idx).store(upd) if W == 1 else \
      UOp.group(*(store_base.index(UOp.const(dtypes.weakint, i)).store(upd.gep(i)) for i in range(W)))
    subs[store] = new_store
    # If the original update is already wrapped by END(reduce_range), rewrite that END in-place. Creating a second END
    # over the same range makes CFGContext see a nested same-range cycle (TG-P12 failure).
    for e in sink.backward_slice:
      if e.op is not Ops.END or e.src[0] is not store: continue
      if tuple(e.src[1:]) == reduce_range:
        ended_stores = [subs[m] for m,_ in matches if m in e.src[0].backward_slice_with_self and m in subs]
        end_src = UOp.group(*ended_stores) if len(ended_stores) > 1 else new_store
      else:
        end_src = new_store
      subs[e] = e.replace(src=(end_src,)+e.src[1:])
  # redirect accumulator output reads (reads after a STORE/END, not in-loop reads under the reduce range) to the wide reg.
  for u in sink.backward_slice:
    tgt = _broadcast_elem(u)
    if tgt is None: tgt = u
    reg = _reg_slot0(tgt)
    if reg not in wide: continue
    after_srcs, _ = _acc_after_chain(tgt)
    if any(r in after_srcs for r in reduce_by_reg.get(reg, ())): continue
    if not any(s.op in {Ops.STORE, Ops.END} for s in after_srcs): continue
    new_after = tuple((subs.get(s, s) if subs else s) for s in after_srcs)
    nr_base = wide[reg].after(*new_after) if new_after else wide[reg]
    nr = nr_base.index(UOp.const(dtypes.weakint, 0)) if u.dtype.count == 1 else _reg_lane_stack(nr_base, u.dtype)
    if nr.dtype == u.dtype: subs[u] = nr
  if not subs: return None
  return sink.substitute(subs, walk=True)

pm_reduce_acc_upcast_fix = PatternMatcher([(UPat(Ops.SINK, name="sink"), reduce_acc_upcast_fix)])

def _distinct_reg_store_indexes(tgt:UOp) -> list[UOp]|None:
  ptrs = [s.src[0] if s.op is Ops.LOAD else s for s in tgt.src]
  if not all(p.op is Ops.INDEX and isinstance(p.src[0].dtype, PtrDType) and p.src[0].dtype.addrspace == AddrSpace.REG for p in ptrs): return None
  return ptrs if len(set(ptrs)) == len(ptrs) else None

def _group_wmma_reg_store(tgt:UOp, val:UOp) -> UOp|None:
  """Recover WMMA output-contract groups from an expanded distinct REG store."""
  wmma = val if val.op is Ops.WMMA else val.src[0] if val.op is Ops.GEP and val.src[0].op is Ops.WMMA else None
  if wmma is None or tgt.op is not Ops.STACK or len(tgt.src) != val.dtype.count: return None
  try: width = prod(sz for _axis,sz in wmma.arg[6][2])
  except (IndexError,TypeError,ValueError): return None
  if width <= 1 or len(tgt.src) % width: return None
  if (ptrs:=_distinct_reg_store_indexes(tgt)) is None or not all(p.src[1].op is Ops.CONST for p in ptrs): return None
  base=ptrs[0].src[0]
  if any(p.src[0] is not base for p in ptrs): return None
  ordered=sorted(((p.src[1].arg,lane,p) for lane,p in enumerate(ptrs)),key=lambda x:x[0])
  if [x[0] for x in ordered] != list(range(ordered[0][0],ordered[0][0]+len(ordered))): return None
  stores=[]
  for start in range(0,len(ordered),width):
    group=ordered[start:start+width]; off=group[0][0]
    dst=base.index(UOp.const(dtypes.weakint,off),dtype=val.dtype.scalar().vec(width))
    stores.append(dst.store(val.gep(tuple(x[1] for x in group))))
  return UOp.group(*stores)

def _devec_distinct_reg_store(tgt:UOp, val:UOp) -> UOp|None:
  if (ptrs:=_distinct_reg_store_indexes(tgt)) is None: return None
  return UOp.group(*[p.store(val.gep(i)) for i,p in enumerate(ptrs)])

def _devec_stack_store(tgt:UOp, val:UOp, gate:UOp|None=None) -> UOp|None:
  if val.dtype.count != len(tgt.src): return None
  if gate is not None and gate.dtype.count != len(tgt.src): return None
  stores = []
  for i,p in enumerate(tgt.src):
    if not isinstance(p.dtype, PtrDType): return None
    ptr = p.gep(0) if p.dtype.base.count != 1 else p
    stores.append(ptr.store(val.gep(i), gate.gep(i) if gate is not None else None))
  return UOp.group(*stores)

pm_distinct_reg_store_devec = PatternMatcher([
  (UPat(Ops.GEP, src=(UPat(Ops.PTRCAT, name="cat"),), name="g"), _gep_local_ptrcat),
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"))), _devec_distinct_reg_store),
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"), UPat.var("gate"))), _devec_stack_store),
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"))), _devec_stack_store),
])
pm_group_wmma_reg_store = PatternMatcher([
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"))), _group_wmma_reg_store),
])

# add loads

def add_load(idx:UOp):
  if isinstance(idx.dtype, PtrDType): return None
  assert isinstance(idx.src[0].dtype, PtrDType), f"param is not PtrDType {idx.src[0].dtype}"
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
