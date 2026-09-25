import itertools
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.helpers import getenv, DEBUG, prod, NOLOCALS, TC_OPT, TC_SELECT, USE_TC, IMAGE
from tinygrad.dtype import PtrDType, ImageDType, dtypes
from tinygrad.uop.ops import Ops, resolve, AxisType, GroupOp
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt.kernel_pipeline import validate_scheduler_tile_loop_pressure


def _buf_idx(b):
  """Index expression of a buffer for index-structure heuristics. Non-weakint indexes are
  legal for rendering (e.g. custom-kernel q6k byte reads build int32 index arithmetic) but
  do not participate in the weakint index-structure heuristics, so they return None."""
  return b.src[1].get_idx() if b.src[1].dtype.scalar() is dtypes.weakint else None

def _is_composite_landing(k:Scheduler) -> bool:
  """Scheduler-boundary landing kernels (multi-CALL composites with dozens of output dims
  and hundreds of buffers) arrive fully scheduled; the local/upcast heuristics assume
  small elementwise/reduce shapes and corrupt them. Leave those schedules untouched."""
  return len(k.bufs) > 128 or len(k.full_shape) > 16

# Expanded accumulator lanes are not the whole live set: indexing, input
# fragments, masks and writeback need short-lived carriers too.  Keep that
# headroom in the scheduler admission decision so an otherwise legal output
# tile cannot consume the complete spill-free pool.  This is deliberately a
# target-independent pressure unit estimate; renderers remain authoritative
# for final physical resources.
SCHEDULER_TRANSIENT_VGPR_RESERVE = 128

def _pressure_admits(*, accumulators:int, fragments:int=0, transient_reserve:int=SCHEDULER_TRANSIENT_VGPR_RESERVE) -> bool:
  try:
    validate_scheduler_tile_loop_pressure(resident_accumulator_vgprs=accumulators,
      resident_fragment_vgprs=fragments, transient_vgpr_reserve=transient_reserve)
    return True
  except ValueError:
    return False

def _epilogue_transient_reserve(k:Scheduler) -> int:
  """Pressure reserve from fused ALU consumers of a reduction result."""
  if k.reduceop is None: return SCHEDULER_TRANSIENT_VGPR_RESERVE
  fused_alu = sum(u is not k.reduceop and u.op in GroupOp.ALU and k.reduceop in u.backward_slice for u in k.ast.toposort())
  return min(PINNED_SCHEDULE_RESERVE_MAX, SCHEDULER_TRANSIENT_VGPR_RESERVE + fused_alu*8)

PINNED_SCHEDULE_RESERVE_MAX = 160

def bounded_reduction_unroll(upcast_lanes:int, reduction_size:int, choices:tuple[int, ...]) -> int|None:
  """Largest requested reduction split admitted with the resident output tile.

  A zero return is represented by ``reduction_size``: UNROLL(..., 0) fully
  expands the axis.  The estimate is intentionally based only on expansion
  residency, so symbolic and concrete kernels take the same path.
  """
  if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in (upcast_lanes, reduction_size)):
    raise ValueError("schedule pressure dimensions must be positive ints")
  for choice in choices:
    if choice <= 0 or reduction_size % choice: continue
    if _pressure_admits(accumulators=upcast_lanes*choice): return choice
  return None

def _loaded(u):
  """The load under a widening CAST (and the BITCAST that reads bf16 as 16-bit words), or the value itself."""
  inner = u
  while inner.op in (Ops.CAST, Ops.BITCAST) and inner.src:
    inner = inner.src[0]
  return inner if inner.op is Ops.INDEX and inner is not u else u


def _widened(u):
  """The product under a widening CAST of the reduce input (a bf16 product summed in fp32), or the value itself."""
  return u.src[0] if u.op is Ops.CAST and u.src and u.src[0].op is Ops.MUL else u

def _matvec(k:Scheduler, max_batch:int):
  """(mulop, activation idx, weight idx, first reduce range, batch ranges) of a (small-M batched) matvec, else None.

  A small-M batched matvec (M activation rows against one weight) carries batch ranges in the activation index that the
  weight index lacks; they must be GLOBAL and M <= max_batch. A widening CAST of a load (a bf16 weight read as fp32) or
  of the product (a bf16 product summed in fp32) is still a matvec."""
  if not (k.reduceop is not None and k.reduceop.arg[0] is Ops.ADD and len(k.full_shape) >= 2 and k.ranges_of(AxisType.REDUCE)): return None
  if (mulop:=_widened(k.reduceop.src[0])).op is not Ops.MUL or not all(_loaded(x).op is Ops.INDEX for x in mulop.src): return None
  idx0, idx1 = _buf_idx(_loaded(mulop.src[0])), _buf_idx(_loaded(mulop.src[1]))
  if idx0 is None or idx1 is None: return None
  first_reduce_rng = k.ranges_of(AxisType.REDUCE)[0]
  batch = [r for r in idx0.ranges if r not in idx1.ranges]
  if not all(r.arg[-1] is AxisType.GLOBAL and isinstance(r.vmax, int) for r in batch) or prod(r.vmax+1 for r in batch) > max_batch: return None
  if not any(u is first_reduce_rng for u in idx0.split_uop(Ops.ADD)): return None
  return mulop, idx0, idx1, first_reduce_rng, batch

def _wide_bf16(k:Scheduler, mulop) -> bool:
  """bf16 weights on a target that folds 16-byte bf16 loads (the measured sm_120 schedule applies)."""
  return bool(getenv("MV_WIDE", 1)) and bool(k.ren.global_bf16_vector_widths) and _loaded(mulop.src[1]).src[0].dtype.base == dtypes.bfloat16

def hand_coded_optimizations(k:Scheduler) -> Scheduler:
  if _is_composite_landing(k): return k

  # first try the tensor cores
  """ Attempts to apply a tensor core optimization to the kernel. If one exists and applies properly, return true, otherwise return false.
  Tensor cores are optimized instructions that matrix multiply-accumulate across a wave of threads: D(M, N) = A(M, K) * B(K, N) + C(M, N).

  Keyword arguments:
  use_tensor_cores -- controls how tensor cores are applied (default 1)
    0: will disable any tensor core matching
    1: enable tensor cores
    2: apply tensor core shape but don't use UOp.WMMA
  extra_opts -- additional Opt's to apply after the tensor core instead of the hand-coded additional Opt's (default None)
  tc_select -- specifies which tensor core(s) to use for optimization (default -1)
    -1: iterates through all available tensor cores in order and uses the first one that matches the requirements (dims and dtypes)
    [0-N]: uses only the n'th tensor core available; useful for search
  tc_opt -- controls which kinds of kernels may be eligible for tensor cores application
    0: applies to only kernels with a single reduce axis and direct Ops.LOAD into Ops.MUL
    1: allows kernels with multiple reduce axes and also multiplication of Ops.CAST'd buffers
    2: allows kernels with M, N, K axes that are not multiples of the tensor core dimensions by applying padding those axes as needed
  """
  # NOTE: unless TC_OPT is > 0, we only trigger tensor cores if there's only one reduce axis
  # A small-M (M<=16) bf16 matvec streams its weight faster through the matvec schedule below than through a
  # 16-row tensor-core tile (sm_120, 3136x12544 at M=16: TC 342us, batched matvec 126us).
  small_mv = (mv:=_matvec(k, getenv("MV_MAX_BATCH", 16))) is not None and bool(mv[4]) and k.ren.has_local and _wide_bf16(k, mv[0])
  if USE_TC > 0 and not small_mv and (len(k.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE)) == 1 or (TC_OPT.value >= 1)):
    good_tc_opt = False
    tk = k.copy()
    try: # check TC first and apply hand-coded opts if successful
      rngs = tk.apply_opt(Opt(OptOps.TC, 0, (TC_SELECT.value, TC_OPT.value, USE_TC.value)))
      good_tc_opt = True
    except KernelOptError:
      pass
    if good_tc_opt:
      if rngs is not None:
        tc_output_lanes = tk.tensor_core.elements_per_thread[2]
        tc_fragment_lanes = sum(tk.tensor_core.elements_per_thread[:2])
        transient_reserve = _epilogue_transient_reserve(k)
        try:
          # The intrinsic's minimum carrier set is itself optional.  If a fused
          # epilogue leaves no bounded transient headroom, skip the optional
          # upcast and local but keep the TC schedule (Piece 2: REDUCE-preserving
          # fusion kernel pressure may exceed the conservative heuristic estimate).
          pressure_ok = _pressure_admits(accumulators=tc_output_lanes, fragments=tc_fragment_lanes,
                                         transient_reserve=transient_reserve)
          if pressure_ok:
            for tc_dim in [1,0]: # attempt to upcast M and N
              szs = [sz for sz in [5,4,3,2] if rngs[tc_dim].src[0].divides(sz) is not None]
              szs = [sz for sz in szs if _pressure_admits(accumulators=tc_output_lanes*sz,
                fragments=tc_fragment_lanes, transient_reserve=transient_reserve)]
              if szs:
                # set it to the replaced range
                rngs[tc_dim] = tk.apply_opt(Opt(OptOps.UPCAST, tk.rngs.index(rngs[tc_dim]), szs[0]))[0]
                tc_output_lanes *= szs[0]
            if (szs := [sz for sz in [4,2] if rngs[0].src[0].divides(sz) is not None]): # attempt to local N
              tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(rngs[0]), szs[0]))
        except KernelOptError:
          pass
      if good_tc_opt: return tk

  # make a copy so it does not mutate the input
  k = k.copy()

  # upcast float4 images, this must be early so we don't accidentally add locals before the upcast
  if IMAGE:
    for buf_index,buf in enumerate(k.bufs):
      if isinstance(buf.src[0].dtype, PtrDType) and ImageDType.valid_dims(buf.src[0].dtype, k.ren.target.arch):
        # part of is_expanded
        buf_idx = _buf_idx(k.bufs[buf_index])
        unit_stride_axes_mul_4 = [k.rngs.index(c) for c in buf_idx.split_uop(Ops.ADD) if
          c.op is Ops.RANGE and (c.vmax+1)%4 == 0] if buf_idx is not None else []
        if len(unit_stride_axes_mul_4):
          if (axis:=unit_stride_axes_mul_4[0]) in k.upcastable_dims:
            k.apply_opt(Opt(OptOps.UPCAST, axis, 4))
          elif axis in k.unrollable_dims:
            k.apply_opt(Opt(OptOps.UNROLL, k.unrollable_dims.index(axis), 4))

  # should use matvec - TODO: adjust/tune based on the wide vs tall/large vs small mat
  MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)
  # MV_VEC: reduction elements per lane per load; MV_WIDE (in _wide_bf16): the measured bf16 schedule
  MV_MAX_BATCH, MV_VEC = getenv("MV_MAX_BATCH", 16), getenv("MV_VEC", 1)
  if k.ren.has_local and getenv("MV",1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and \
    k.ren.has_shared and (mv:=_matvec(k, MV_MAX_BATCH)) is not None:
    mulop, idx0, idx1, first_reduce_rng, batch = mv
    nbatch = prod(r.vmax+1 for r in batch)
    tpr, blocksize, vec = MV_THREADS_PER_ROW, MV_BLOCKSIZE, MV_VEC
    # keep the accumulator tile (rows per thread x batch) near the unbatched one
    rows_per_thread = min(MV_ROWS_PER_THREAD, max(1, MV_ROWS_PER_THREAD*4 // nbatch)) if batch else MV_ROWS_PER_THREAD
    if _wide_bf16(k, mulop):
      # bf16 weights on a target that folds 16-byte bf16 loads (measured sm_120): a warp-wide row split with one
      # row per lane and 128-thread blocks keeps enough loads in flight to stream at ~80-95% of DRAM bandwidth;
      # the 8/4/4 default launches too few threads (ffn_down 3136x12544: 558 -> 1335 GB/s). The reduction
      # UNROLL feeds each weight load to every batch row (M=8: 546-634 -> 1057-1132 GB/s).
      tpr, vec = next(((t, v) for t, v in ((32, 8), (16, 4), (8, 8), (8, 4), (8, 1))
                       if first_reduce_rng.src[0].divides(t*v) is not None), (MV_THREADS_PER_ROW, 1))
      blocksize, rows_per_thread = max(1, 128 // tpr), 1
    for global_idx in k.axes_of(AxisType.GLOBAL):
      if batch and (k.rngs[global_idx] in batch or k.rngs[global_idx] not in idx1.ranges): continue
      if first_reduce_rng.src[0].divides(tpr) is not None and k.full_shape[global_idx]%(blocksize*rows_per_thread) == 0:
        if DEBUG >= 3:
          print(f"MATVEC: {k.full_shape=} {first_reduce_rng.render()} {blocksize=} {tpr=} {rows_per_thread=} {vec=}")
        if vec > 1 and first_reduce_rng.src[0].divides(vec*tpr) is not None:
          k.apply_opt(Opt(OptOps.UNROLL, 0, vec))
        try:
          if tpr > 1: k.apply_opt(Opt(OptOps.GROUP, 0, tpr))
        except KernelOptError: pass
        if blocksize > 1: k.apply_opt(Opt(OptOps.LOCAL, global_idx, blocksize))
        if rows_per_thread > 1: k.apply_opt(Opt(OptOps.UPCAST, global_idx, rows_per_thread))
        # up to 8 batch rows share each weight load in registers; more rows spill, so the rest become blocks
        budget = 8
        for rng in batch:
          size = rng.vmax+1
          amt = size if size <= budget else next((a for a in (8, 4, 2) if a <= budget and size % a == 0), 1)
          if amt > 1: k.apply_opt(Opt(OptOps.UPCAST, k.rngs.index(rng), 0 if amt == size else amt))
          budget //= amt
        return k

  # MV_DEQUANT (opt-in, research): the strict matvec check above requires reduceop.src[0] == MUL(INDEX, INDEX)
  # (two DIRECT loads). A fused-dequant matvec is MUL(dequant(INDEX(words)), INDEX(x)) -- the weight operand is a
  # MUL/SUB/CAST chain, not a bare INDEX -- so the detector misses it and the GEMV falls to GROUPTOP/output-parallel
  # (uncoalesced). This branch "sees through" the dequant: apply the same GROUP+LOCAL+UPCAST matvec opts to any
  # ADD-reduce of matvec shape. Tests whether GROUP (coalescing) was a navigation/recognition gap. Default-off.
  if getenv("MV_DEQUANT") and k.ren.has_local and k.ren.has_shared and getenv("MV", 1) != 0 and k.reduceop is not None \
     and k.reduceop.arg[0] is Ops.ADD and len(k.full_shape) >= 2 and k.ranges_of(AxisType.REDUCE):
    first_reduce_rng = k.ranges_of(AxisType.REDUCE)[0]
    for global_idx in k.axes_of(AxisType.GLOBAL):
      if first_reduce_rng.src[0].divides(MV_THREADS_PER_ROW) is not None and k.full_shape[global_idx] % (MV_BLOCKSIZE*MV_ROWS_PER_THREAD) == 0:
        if DEBUG >= 3: print(f"MV_DEQUANT MATVEC: {k.full_shape=} {MV_THREADS_PER_ROW=} {MV_BLOCKSIZE=} {MV_ROWS_PER_THREAD=}")
        # A fused-dequant matvec whose weights are packed on a small trailing reduce axis (e.g. a 32-code trellis
        # period, or Q4_K's nibble/word axis) needs that axis UNROLL'd so the packed group-word is loaded ONCE and
        # the per-code decode ALU is register-resident; else it stays a REDUCE loop that re-loads and does not hide
        # under the weight stream. GROUP+LOCAL alone leaves it a loop -> ~2x slower. Unroll the packed period FIRST
        # (before GROUP): unroll-then-group beat group-then-unroll 335 vs 434us (trellis 17408x5120, gfx1100; the
        # default heuristic without MV_DEQUANT was 641us). Unroll ONLY the largest small genuine REDUCE axis (the
        # period), not the K-block reduce (unrolling that too measured worse).
        if getenv("MV_UNROLL_REDUCE", 1):
          cap = getenv("MV_UNROLL_MAX", 32)
          small = [i for i,t in enumerate(k.axis_types) if t is AxisType.REDUCE
                   and isinstance(sz:=k.full_shape[i], int) and 1 < sz <= cap]
          # keep >=1 bare REDUCE axis for GROUP to split (GROUP uses axes_of(REDUCE)[0])
          if small and (len(k.axes_of(AxisType.REDUCE)) > 1 or MV_THREADS_PER_ROW <= 1):
            try: k.apply_opt(Opt(OptOps.UNROLL, k.unrollable_dims.index(max(small, key=lambda i: k.full_shape[i])), 0))
            except (KernelOptError, ValueError): pass
        try:
          if MV_THREADS_PER_ROW > 1: k.apply_opt(Opt(OptOps.GROUP, 0, MV_THREADS_PER_ROW))
        except KernelOptError: pass
        if MV_BLOCKSIZE > 1: k.apply_opt(Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE))
        if MV_ROWS_PER_THREAD > 1: k.apply_opt(Opt(OptOps.UPCAST, global_idx, MV_ROWS_PER_THREAD))
        return k

  # are we grouping? (requires local shape support)
  if resolve(prod(k.output_shape[i] for i in k.upcastable_dims) <= (240 if NOLOCALS else 2048), False):
    for axis, sz in itertools.product((0, 1, 2), (16,)):
      try:
        k.apply_opt(Opt(OptOps.GROUPTOP, axis, sz))
        break
      except KernelOptError: pass

  # no more opt if we are grouping
  if k.group_for_reduces: return k

  # **** below this line need to be optional and benchmarked ****

  # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
  to_upcast: list[int] = []
  where_gate_rngs = {r for u in k.ast.backward_slice if u.op is Ops.WHERE for r in u.src[0].ranges}
  # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
  for axis in k.upcastable_dims:
    # for Schedule, we check if the range is used in INDEX gates or WHERE gates
    is_masked = k.rngs[axis] in where_gate_rngs
    if k.full_shape[axis] <= 7 and is_masked and prod(k.full_shape[j] for j in to_upcast) * k.full_shape[axis] <= 7 * 7:
      if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
      to_upcast.append(axis)
  for axis in to_upcast[::-1]:
    try: k.apply_opt(Opt(OptOps.UPCAST, axis, 0))
    except KernelOptError: pass

  # potentially do more upcasts of non reduce axes based on a heuristic
  is_dsp = k.ren is not None and k.ren.target.device == "DSP"
  upcasted_axis: set[int] = set()
  while len(k.bufs) <= 64 and resolve(prod(k.output_shape[i] for i in k.upcastable_dims) >= 1024) and (k.upcast_size() < 32):
    xb_choices = []
    # consider all upcastable axes with 3 or 4 upcast (128 on the DSP)
    for axis, upcast_amount in itertools.product(k.upcastable_dims, ([128] if not len(upcasted_axis) else []) if is_dsp else [3,4]):
      # if we haven't upcasted it, it mods, and buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
      if axis in upcasted_axis or k.full_shape[axis]%upcast_amount != 0: continue
      rng = k.rngs[axis]
      if any((idx:=_buf_idx(b)) is not None and rng not in idx.backward_slice and all(r2 in idx.backward_slice
          for r2 in k.ranges_of(AxisType.UPCAST, AxisType.UNROLL)) for b in k.bufs):
        num_strides, sum_strides = 0, 0
        for b in k.bufs:
          idx = _buf_idx(b)
          if idx is None: continue
          if rng in idx.backward_slice: num_strides += 1
          for c in idx.split_uop(Ops.ADD):
            if c is rng: sum_strides += 1
            if c.op is Ops.MUL and c.src[0] is rng and c.src[1].op is Ops.CONST: sum_strides += c.src[1].arg
            if c.op is Ops.MUL and c.src[1] is rng and c.src[0].op is Ops.CONST: sum_strides += c.src[0].arg
        xb_choices.append((num_strides, sum_strides, axis, upcast_amount))
    if xb_choices:
      xb_choices = sorted(xb_choices)
      if DEBUG >= 4: print(f"more upcast axis : {xb_choices}")
      try: k.apply_opt(Opt(OptOps.UPCAST, xb_choices[0][2], xb_choices[0][3]))
      except KernelOptError: break
      upcasted_axis.add(xb_choices[0][2])
    else: break

  # if last reduce dim is small(ish), loop unroll the reduce
  # NOTE: this can fail on multireduce with mismatching dimensions, this is okay
  try:
    if k.unrollable_dims and (k.upcast_size() <= 4 or not k.axes_of(AxisType.UNROLL)) and (k.upcast_size() < 64):
      if (s:=k.full_shape[k.unrollable_dims[-1]]) <= 32:
        unroll = bounded_reduction_unroll(k.upcast_size(), s, tuple(x for x in (s,16,8,4,2) if x <= s))
        if unroll is not None: k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 0 if unroll == s else unroll))
        # if it's small, upcast a second reduce dimension too
        if unroll is not None and k.unrollable_dims and s <= 3 and k.full_shape[k.unrollable_dims[-1]] <= 3 and \
           _pressure_admits(accumulators=k.upcast_size()*k.full_shape[k.unrollable_dims[-1]]):
          k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 0))
      else:
        for splits in [4]:
          if k.full_shape[axis:=k.unrollable_dims[-1]]%splits == 0:
            k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, splits))
            break
  except KernelOptError: pass

  # if nothing at all is upcasted and it's easy to, do an upcast
  for splits in [4]:
    if not k.upcasted and k.upcastable_dims and k.full_shape[k.upcastable_dims[-1]] % splits == 0:
      # Composite reductions own their output lanes as scalar LOOP state.  The
      # scheduler rejects packing those ranges; this opportunistic fallback
      # must fail closed and leave the valid scalar schedule intact.
      try: k.apply_opt(Opt(OptOps.UPCAST, k.upcastable_dims[-1], splits))
      except KernelOptError: pass

  # **** local groups ****

  if k.ren.has_local:
    if NOLOCALS:
      k.apply_opt(Opt(OptOps.NOLOCALS))
    else:
      # prioritize making expand axes local
      local_axis_ranking = [(any((idx:=_buf_idx(b)) is not None and k.rngs[axis] not in idx.backward_slice for b in k.bufs), axis) \
                              for axis in k.axes_of(AxisType.GLOBAL, AxisType.LOOP) if k.rngs[axis].src[0].op is Ops.CONST]
      to_local: list[tuple[int, int]] = []
      for _, axis in sorted(local_axis_ranking, key=lambda x: (-x[0], -x[1])):
        local_size = prod(sz for _, sz in to_local)
        local_sz: int|None = next((x for x in ([32] * (axis == 0) + [16,8,4,3,2]) if k.full_shape[axis] % x == 0 and local_size * x <= 128), None)
        if local_sz is not None: to_local.append((axis, local_sz))
      deleted_shape = 0
      for axis, local_sz in sorted(to_local[:3]):
        axis = axis - deleted_shape
        will_delete_shape = local_sz == k.full_shape[axis]
        k.apply_opt(Opt(OptOps.LOCAL, axis, local_sz))
        if will_delete_shape: deleted_shape += 1

  # **** threading ****

  if k.ren.has_threads and k.ren.global_max is not None:
    for threads in [32,16,12,8,6,5,4,3,2]:
      # Skip if too many threads. Heuristic: use about 128K ops per thread
      if threads > k.ren.global_max[0] or resolve(prod(k.full_shape) // (128 << 10) < threads): continue
      for axis in k.axes_of(AxisType.LOOP):
        if k.full_shape[axis] % threads == 0:
          try: k.apply_opt(Opt(OptOps.THREAD, axis, threads))
          except KernelOptError: pass
          break
      if k.applied_opts and k.applied_opts[-1].op is OptOps.THREAD: break

  return k
