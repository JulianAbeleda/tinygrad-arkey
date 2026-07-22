import itertools
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.helpers import getenv, DEBUG, prod, NOLOCALS, TC_OPT, TC_SELECT, USE_TC, IMAGE
from tinygrad.dtype import PtrDType, ImageDType
from tinygrad.uop.ops import Ops, resolve, AxisType, GroupOp
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt.kernel_pipeline import validate_scheduler_tile_loop_pressure

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

def hand_coded_optimizations(k:Scheduler) -> Scheduler:
  # Composite reduces (online-softmax) need the loop structure preserved.
  # The scheduler's UPCAST/UNROLL vectorizes the body, which the composite
  # lowering in reduce_to_acc can't handle. Skip all opts for composite kernels.
  from tinygrad.uop.ops import CompositeReduce
  for u in k.ast.toposort():
    if u.op is Ops.REDUCE and isinstance(u.arg[0], CompositeReduce):
      return k

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
  if USE_TC > 0 and (len(k.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE)) == 1 or (TC_OPT.value >= 1)):
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
        unit_stride_axes_mul_4 = [k.rngs.index(c) for c in k.bufs[buf_index].src[1].get_idx().split_uop(Ops.ADD) if
          c.op is Ops.RANGE and (c.vmax+1)%4 == 0]
        if len(unit_stride_axes_mul_4):
          if (axis:=unit_stride_axes_mul_4[0]) in k.upcastable_dims:
            k.apply_opt(Opt(OptOps.UPCAST, axis, 4))
          elif axis in k.unrollable_dims:
            k.apply_opt(Opt(OptOps.UNROLL, k.unrollable_dims.index(axis), 4))

  # should use matvec - TODO: adjust/tune based on the wide vs tall/large vs small mat
  MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)
  if k.ren.has_local and getenv("MV",1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and  \
    k.reduceop is not None and k.reduceop.arg[0] is Ops.ADD and len(k.full_shape) >= 2 and k.ren.has_shared and \
    (mulop:=k.reduceop.src[0]).op is Ops.MUL and mulop.src[0].op is Ops.INDEX and mulop.src[1].op is Ops.INDEX:
    idx0, idx1 = mulop.src[0].src[1].get_idx(), mulop.src[1].src[1].get_idx()
    if k.ranges_of(AxisType.REDUCE):
      first_reduce_rng = k.ranges_of(AxisType.REDUCE)[0]
      if any(u is first_reduce_rng for u in idx0.split_uop(Ops.ADD)) and all(r in idx1.ranges for r in idx0.ranges):
        for global_idx in k.axes_of(AxisType.GLOBAL):
          if first_reduce_rng.src[0].divides(MV_THREADS_PER_ROW) is not None and k.full_shape[global_idx]%(MV_BLOCKSIZE*MV_ROWS_PER_THREAD) == 0:
            if DEBUG >= 3:
              print(f"MATVEC: {k.full_shape=} {first_reduce_rng.render()} {MV_BLOCKSIZE=} {MV_THREADS_PER_ROW=} {MV_ROWS_PER_THREAD=}")
            try:
              if MV_THREADS_PER_ROW > 1: k.apply_opt(Opt(OptOps.GROUP, 0, MV_THREADS_PER_ROW))
            except KernelOptError: pass
            if MV_BLOCKSIZE > 1: k.apply_opt(Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE))
            if MV_ROWS_PER_THREAD > 1: k.apply_opt(Opt(OptOps.UPCAST, global_idx, MV_ROWS_PER_THREAD))
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
  for axis in to_upcast[::-1]: k.apply_opt(Opt(OptOps.UPCAST, axis, 0))

  # potentially do more upcasts of non reduce axes based on a heuristic
  is_dsp = k.ren is not None and k.ren.target.device == "DSP"
  upcasted_axis: set[int] = set()
  while resolve(prod(k.output_shape[i] for i in k.upcastable_dims) >= 1024) and (k.upcast_size() < 32):
    xb_choices = []
    # consider all upcastable axes with 3 or 4 upcast (128 on the DSP)
    for axis, upcast_amount in itertools.product(k.upcastable_dims, ([128] if not len(upcasted_axis) else []) if is_dsp else [3,4]):
      # if we haven't upcasted it, it mods, and buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
      if axis in upcasted_axis or k.full_shape[axis]%upcast_amount != 0: continue
      rng = k.rngs[axis]
      if any(rng not in b.src[1].get_idx().backward_slice and all(r2 in b.src[1].get_idx().backward_slice
          for r2 in k.ranges_of(AxisType.UPCAST, AxisType.UNROLL)) for b in k.bufs):
        num_strides, sum_strides = 0, 0
        for b in k.bufs:
          idx = b.src[1].get_idx()
          if rng in idx.backward_slice: num_strides += 1
          for c in idx.split_uop(Ops.ADD):
            if c is rng: sum_strides += 1
            if c.op is Ops.MUL and c.src[0] is rng and c.src[1].op is Ops.CONST: sum_strides += c.src[1].arg
            if c.op is Ops.MUL and c.src[1] is rng and c.src[0].op is Ops.CONST: sum_strides += c.src[0].arg
        xb_choices.append((num_strides, sum_strides, axis, upcast_amount))
    if xb_choices:
      xb_choices = sorted(xb_choices)
      if DEBUG >= 4: print(f"more upcast axis : {xb_choices}")
      k.apply_opt(Opt(OptOps.UPCAST, xb_choices[0][2], xb_choices[0][3]))
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
      k.apply_opt(Opt(OptOps.UPCAST, k.upcastable_dims[-1], splits))

  # **** local groups ****

  if k.ren.has_local:
    if NOLOCALS:
      k.apply_opt(Opt(OptOps.NOLOCALS))
    else:
      # prioritize making expand axes local
      local_axis_ranking = [(any(k.rngs[axis] not in b.src[1].get_idx().backward_slice for b in k.bufs), axis) \
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
