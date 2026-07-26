import functools
from tinygrad.dtype import dtypes, DType, AddrSpace, PtrDType
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, identity_element, AxisType
from tinygrad.helpers import prod
from tinygrad.codegen.late.devectorizer import _gep_local_ptrcat

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

def _output_load_lane(u:UOp) -> tuple[UOp, int]|None:
  # (GLOBAL INDEX, load-lane) if u is a scalar LOAD(INDEX(...)) or
  # GEP(LOAD([CAST] INDEX(...)), lane), else None.
  # This is an output address that add_loads turned into a wide vector LOAD, whose lanes were then read
  # back as the (unassignable) STORE target instead of staying an addressable INDEX.
  # GLOBAL-only ON PURPOSE (see _devec_output_projection_store): the only validated producer is the GLOBAL
  # matmul-epilogue (LM-head) whose reduction is ADD.  A LOCAL/REG lane would be an online-softmax/composite
  # combine intermediate whose reduction may be MAX (gmax) or MUL -- ADD-combining that would be silently
  # wrong (the reduce op is unrecoverable at this stage), so those are excluded here and owned by
  # reduce_acc_upcast_fix / a future op-aware combine lowering.
  if u.op is Ops.LOAD: ld, lane = u, 0
  elif u.op is Ops.GEP and isinstance(u.arg, tuple) and len(u.arg) == 1: ld, lane = u.src[0], u.arg[0]
  else: return None
  if ld.op is not Ops.LOAD: return None
  idx = ld.src[0]
  if idx.op is Ops.CAST: idx = idx.src[0]
  if idx.op is not Ops.INDEX or len(idx.src) < 2: return None
  if getattr(idx.src[0], "addrspace", None) is not AddrSpace.GLOBAL: return None
  return (idx, lane)

def _devec_output_projection_store(tgt:UOp, val:UOp) -> UOp|None:
  # Sibling of the bare-LOAD(INDEX) output-projection restoration in codegen/__init__.py:235-244
  # (the had_deferred_reduce_projection block): that owner only matches lanes that are bare LOAD(INDEX)
  # and assumes one distinct address per lane.  This handles the wide-load/UPCAST'd variant it misses:
  # deferred-reduce output projection where lanes are GEP(LOAD(INDEX(out,addr))) and an UPCAST'd inner
  # reduce axis left each distinct output address duplicated in contiguous same-size groups (the
  # 32-value -> 16-address make_floatN(...) lvalue on gfx1100).  Restore addressable per-address global
  # stores, horizontally ADD-reducing each group's partials (the sum the UPCAST'd reduce axis represents).
  #
  # ADD-ONLY, by construction.  The reduce op is unrecoverable at this codegen stage (it is baked into the
  # ALU chain by lower_deferred_reduce_slot; the store carries no op), so this pass cannot combine with the
  # true op -- it always sums.  That is safe ONLY because _output_load_lane restricts to GLOBAL output-buffer
  # lanes, whose sole producer is the additive matmul epilogue (LM-head).  A non-additive combine (the
  # online-softmax gmax MAX-reduce -- exactly TG-P9.4's split-preserving combine) lives in REG/LOCAL and is
  # excluded, so it can never be silently ADD-mis-combined here.  Making this op-aware (recover/propagate the
  # reduce op so a MAX/MUL combine lowers correctly) is deferred to the fused-combine work, where a MAX
  # producer actually exists to test against; adding untested MAX handling here now would be speculative.
  # Fail-closed: only fires when every lane is a GLOBAL output-load lane read, groups are contiguous and
  # uniform with size>1, and each group's values are distinct (a genuine many->one reduction, not a broadcast).
  if val.dtype.count != len(tgt.src): return None
  info = [_output_load_lane(p) for p in tgt.src]
  if any(x is None for x in info): return None
  groups, i = [], 0
  while i < len(tgt.src):
    j = i
    while j < len(tgt.src) and tgt.src[j] is tgt.src[i]: j += 1
    groups.append(list(range(i, j))); i = j
  g = len(groups[0])
  if g < 2 or any(len(pos) != g for pos in groups): return None
  for pos in groups:
    if len({val.gep((p,)) for p in pos}) != g: return None      # identical values -> broadcast, leave it
  stores = []
  for pos in groups:
    idx, lane = info[pos[0]]
    addr = idx.src[0].index(idx.src[1] + UOp.const(idx.src[1].dtype, lane))
    stores.append(addr.store(functools.reduce(lambda a,b: a+b, [val.gep((p,)) for p in pos])))
  return UOp.group(*stores)

pm_distinct_reg_store_devec = PatternMatcher([
  (UPat(Ops.GEP, src=(UPat(Ops.PTRCAT, name="cat"),), name="g"), _gep_local_ptrcat),
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"))), _devec_output_projection_store),
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"))), _devec_distinct_reg_store),
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"), UPat.var("gate"))), _devec_stack_store),
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"))), _devec_stack_store),
])
pm_group_wmma_reg_store = PatternMatcher([
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"))), _group_wmma_reg_store),
])
