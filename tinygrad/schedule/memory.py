from collections import defaultdict
from contextvars import ContextVar
from tinygrad.device import Device
from tinygrad.helpers import NO_MEMORY_PLANNER, DEBUG, round_up
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from tinygrad.runtime.support.memory import TLSFAllocator

def _collect_bufs(u:UOp) -> list[UOp]:
  if u.op is Ops.BUFFER: return [u]
  if u.op is Ops.MEMORY_SEMANTIC and len(u.src) == 1: return _collect_bufs(u.src[0])
  if u.op in {Ops.MSELECT, Ops.MSTACK}: return [b for s in u.src for b in _collect_bufs(s)]
  return []

def _can_plan(b:UOp, held_bufs:set[UOp]) -> bool:
  if b in held_bufs: return False
  devs = (b.device,) if isinstance(b.device, str) else b.device
  return all(not d.startswith(("DISK", "TINYFS")) and hasattr(Device[d].allocator, "_offset") for d in devs)

LaneKey = tuple[str, int]
_memory_manifest_collectors:ContextVar[tuple] = ContextVar("memory_manifest_collectors", default=())

def memory_plan_rewrite(linear:UOp, held_bufs:set[UOp]|None=None) -> UOp:
  collectors = _memory_manifest_collectors.get()
  if NO_MEMORY_PLANNER and not collectors: return linear
  if held_bufs is None: held_bufs = set()
  first:dict[UOp, int] = {}; last:dict[UOp, int] = {}; copy_bufs:set[UOp] = set()
  for i, si in enumerate(linear.src):
    bufs = [b for src in si.src[1:] for b in _collect_bufs(src) if _can_plan(b, held_bufs)]
    for b in bufs: first.setdefault(b, i); last[b] = i
    if si.src[0].op is Ops.COPY: copy_bufs.update(bufs)
  key = lambda b: (b.device, 1 if b in copy_bufs else 0)
  hold = {b:last[b]-first[b]+1 for b in first if b in copy_bufs}
  block_size = 256
  nbytes = {b:round_up(b.arg*b.dtype.itemsize, block_size) for b in first}
  events = sorted([(first[b], True, b) for b in first] + [(last[b]+1+hold.get(b, 0), False, b) for b in first], key=lambda x:(x[0], x[1]))
  total_memory = sum(nbytes.values()) * 2
  offsets:dict[UOp, int] = {}
  peaks:dict[LaneKey, tuple[int, TLSFAllocator]] = defaultdict(lambda:(0, TLSFAllocator(total_memory, block_size=block_size, lv2_cnt=32)))
  for _, opening, buf in events:
    if opening: offsets[buf] = peaks[key(buf)][1].alloc(nbytes[buf])
    else: peaks[key(buf)][1].free(offsets[buf])
    peaks[key(buf)] = (max(peaks[key(buf)][0], offsets[buf] + buf.arg*buf.dtype.itemsize), peaks[key(buf)][1])
  arena_sizes = {k:round_up(peak, block_size) for k,(peak,_) in peaks.items()}
  arenas = {} if NO_MEMORY_PLANNER else {k:UOp.new_buffer(k[0], sz, dtypes.int8) for k,sz in arena_sizes.items()}
  for collect in collectors: collect(linear, held_bufs, arenas if not NO_MEMORY_PLANNER else None)
  if NO_MEMORY_PLANNER or not offsets: return linear
  replace_map = {b:UOp(Ops.SLICE, b.dtype, (arenas[key(b)], UOp.const(dtypes.weakint, offset)), b.arg) for b,offset in offsets.items()}
  if DEBUG >= 1 and (omem:=sum(nbytes.values())/1e6) != (nmem:=sum(arena_sizes.values())/1e6):
    print(f"memory reduced from {omem:.2f} MB -> {nmem:.2f} MB, {len(first)} -> {len(arenas)} bufs")
  return linear.substitute(replace_map, name="memory plan", walk=True)
