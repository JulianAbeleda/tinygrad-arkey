from __future__ import annotations
from dataclasses import dataclass, replace
from collections import defaultdict, deque
from typing import Any, ClassVar, Generic, TypeVar, Iterator, Generator, TYPE_CHECKING
import importlib, inspect, functools, pathlib, os, contextlib, re, atexit, pickle, decimal, ctypes, json, time
from tinygrad.helpers import LRU, getenv, diskcache_get, diskcache_put, DEBUG, GlobalCounters, flat_mv, PROFILE, temp, colored, ContextVar
from tinygrad.helpers import Context, CCACHE, ALLOW_DEVICE_USAGE, MAX_BUFFER_SIZE, cpu_events, ProfileEvent, ProfilePointEvent, suppress_finalizing
from tinygrad.helpers import select_by_name, select_first_inited, DEV, TracingKey, size_to_str, pluralize
from tinygrad.dtype import DType, PtrDType, _to_np_dtype
if TYPE_CHECKING: from tinygrad.renderer import Renderer

# **************** Device ****************

ALL_DEVICES = ["AMD", "NV", "CPU"]
class _Device:
  def __init__(self) -> None:
    self._devices = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
    self._opened_devices:set[str] = set()
  @functools.cache  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def _canonicalize(self, device:str) -> str: return re.sub(r":0$", "", (d:=device.split(":", 1)[0].upper()) + device[len(d):])
  # NOTE: you can't cache canonicalize in case Device.DEFAULT changes
  def canonicalize(self, device:str|None) -> str: return self._canonicalize(device if device is not None else Device.DEFAULT)
  def __getitem__(self, ix:str) -> Compiled:
    ix = self.canonicalize(ix)
    assert ALLOW_DEVICE_USAGE or ix.split(":")[0] in ["DISK", "TINYFS", "NPY", "PYTHON"], f"usage of device {ix} disallowed"
    return self.__get_canonicalized_item(ix)
  @functools.cache  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __get_canonicalized_item(self, ix:str) -> Compiled:
    base = (__package__ or __name__).split('.')[0]  # tinygrad
    x = ix.split(":")[0].lower()
    ret = [cls for cname, cls in inspect.getmembers(importlib.import_module(f'{base}.runtime.ops_{x}')) \
           if (cname.lower() == x + "device")][0](ix)
    if DEBUG >= 1: print(f"opened device {ix} from pid:{os.getpid()}")
    self._opened_devices.add(ix)
    return ret
  @property
  def default(self) -> Compiled: return self[self.DEFAULT]
  def get_available_devices(self) -> Iterator[str]:
    for device in ALL_DEVICES:
      with contextlib.suppress(Exception): yield self[device].device
  @property
  def DEFAULT(self) -> str: return DEV.device or self._select_device
  @DEFAULT.setter
  def DEFAULT(self, v): raise AttributeError(f'setting Device.DEFAULT is deprecated, use "with Context(DEV={v!r})" or "DEV.value = {v!r}"')
  @functools.cached_property
  def _select_device(self) -> str:
    assert (dev:=next((d for d in self._devices if d not in ["DISK", "TINYFS", "NPY"] and getenv(d) == 1), None)) is None, \
      f"{dev}=1 is deprecated, use DEV={dev} instead"
    try:
      device = next(self.get_available_devices())
      os.environ["DEV"] = device   # we set this in environment for spawned children
      return device
    except StopIteration as exc: raise RuntimeError("no usable devices") from exc
Device: _Device = _Device()
atexit.register(lambda: [Device[dn].finalize() for dn in Device._opened_devices])

def canonicalize_device(device:str|tuple|list|None) -> str|tuple[str, ...]:
  if not isinstance(device, (tuple, list)): return Device.canonicalize(device)
  return canonical[0] if len(canonical:=tuple(Device.canonicalize(d) for d in device)) == 1 else canonical

# **************** Profile ****************

@dataclass(frozen=True)
class ProfileDeviceEvent(ProfileEvent): device:str; tdiff:decimal.Decimal=decimal.Decimal(0); props:dict[str,Any]|None=None # noqa: E702

@dataclass(frozen=True)
class ProfileProgramEvent(ProfileEvent): device:str; name:str; lib:bytes|None; base:int|None; tag:int|None=None # noqa: E702

@dataclass(frozen=True)
class ProfileGraphEntry: device:str; name:str|TracingKey; st_id:int; en_id:int; metadata:dict[str,Any]|None=None # noqa: E702

@dataclass(frozen=True)
class ProfileGraphEvent(ProfileEvent): ents:list[ProfileGraphEntry]; deps:list[list[int]]; sigs:list[decimal.Decimal] # noqa: E702

# **************** Buffer + Allocators ****************

@dataclass(frozen=True, eq=True)
class BufferSpec:
  # TODO: move device, size, dtype here?
  uncached: bool = False
  cpu_access: bool = False
  host: bool = False
  nolru: bool = False
  external_ptr: int|None = None

class MultiBuffer:
  def __init__(self, device:tuple[str, ...], size:int, dtype:DType):
    self.bufs = [Buffer(d, size, dtype) for d in device]
  @property
  def size(self): return self.bufs[0].size
  @property
  def dtype(self): return self.bufs[0].dtype
  def ref(self, cnt):
    for b in self.bufs: b.ref(cnt)
    return self
  def is_allocated(self): return all(x.is_allocated() for x in self.bufs)
  def __repr__(self): return f"<multibuf real:{self.is_allocated()} device:{tuple(x.device for x in self.bufs)} size:{self.size} dtype:{self.dtype}>"

class Buffer:
  profile_events:list[ProfileEvent] = []
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:BufferSpec|None=None, initial_value:bytes|None=None,
               uop_refcount=0, base:Buffer|None=None, offset:int=0, preallocate=False):
    assert isinstance(dtype, DType) and not isinstance(dtype, PtrDType)
    self.device, self.size, self.dtype, self.options, self.offset, self.allocated_views = device, size, dtype, options, offset, 0
    self._bufs: dict[str, Any] = {}
    if base is None:
      assert offset == 0, "base buffers can't have offset"
      self._base = None
      self._uop_refcount = uop_refcount
      if opaque is not None: self.allocate(opaque)
      if initial_value is not None:
        self.allocate()
        self.copyin(memoryview(initial_value))
    else:
      assert base._base is None, "base can't have a base"
      assert device == base.device, "base must have the same device"
      self._base = base
    if preallocate: self.allocate()
  @property
  def base(self) -> Buffer: return self._base if self._base is not None else self
  @property
  def uop_refcount(self): return self.base._uop_refcount
  @property
  def _buf(self) -> Any: return self._bufs[self.device]
  def ref(self, cnt):
    self.base._uop_refcount += cnt
    return self
  # check if the underlying buffer is allocated and the current buffer/view is initialized
  def is_initialized(self) -> bool: return self.is_allocated() and self.device in self._bufs
  # check if the underlying buffer is allocated, possibly from the base object
  def is_allocated(self) -> bool: return self.base.is_allocated() if self._base is not None else self.device in self._bufs
  def get_buf(self, device: str) -> Any:
    if device not in self._bufs:
      allocator = Device[device].allocator
      if device == self.device: self.ensure_allocated()
      elif self._base is not None:
        assert hasattr(allocator, "_offset"), "offset function required for view"
        self._bufs[device] = allocator._offset(self._base.get_buf(device), self.nbytes, self.offset)
      else:
        self._bufs[device] = allocator._map(self.ensure_allocated()._buf)
    return self._bufs[device]
  def ensure_allocated(self) -> Buffer: return self.allocate() if not self.is_initialized() else self
  def allocate(self, opaque=None, external_ptr=None) -> Buffer:
    assert not self.is_initialized(), "can't allocate already allocated buffer"
    if DEBUG >= 7: print(f"buffer: allocate {self.nbytes} bytes on {self.device}")
    if not self.device.startswith("NULL") and self.size > MAX_BUFFER_SIZE > 0 and (self.options is None or self.options.external_ptr is None):
      raise RuntimeError(f"buffer of size {self.size/1e6:.2f}M is too large")
    self.allocator:Allocator = Device[self.device].allocator
    if external_ptr is not None:
      self.options = replace(self.options, external_ptr=external_ptr) if self.options else BufferSpec(external_ptr=external_ptr)
    if self._base is not None:
      self._base.ensure_allocated()
      self._base.allocated_views += 1
      assert hasattr(self.allocator, "_offset"), "offset function required for view"
      self._bufs[self.device] = self.allocator._offset(self.base._buf, self.nbytes, self.offset)
    else:
      self._bufs[self.device] = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
      if not self.device.startswith("DISK") and (self.options is None or self.options.external_ptr is None):
        GlobalCounters.mem_used += self.nbytes
        GlobalCounters.mem_used_per_device[self.device] += self.nbytes
      if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "alloc", self.trace_num, {"dtype":self.dtype, "sz":self.size}))
      # ALLOC_TRACE fault-to-allocation ring (tinygrad/device.py, near KERNARGS_AUDIT/DISPATCH_TRACE): only
      # buffers with a real GPU VA (HCQBuffer.va_addr, e.g. AMD via KFDIface.alloc) are worth recording --
      # everything else is a no-op via getattr. `buf.size` is the KFD-reported *mapped* size, which can be
      # page-rounded above `self.nbytes` (the requested size); both are recorded so the difference is visible.
      if ALLOC_TRACE and (va:=getattr((buf:=self._bufs[self.device]), 'va_addr', None)) is not None:
        self._at_alloc_id = alloc_trace_record_alloc(self.device, va, getattr(buf, 'size', self.nbytes), self.nbytes)
    return self
  def deallocate(self):
    assert self.device in self._bufs, "buffer must be allocated to deallocate"
    if DEBUG is not None and DEBUG >= 7: print(f"buffer: deallocate {self.nbytes} bytes on {self.device}")
    if self._base is None:
      if GlobalCounters is not None and not self.device.startswith("DISK") and (self.options is None or self.options.external_ptr is None):
        GlobalCounters.mem_used -= self.nbytes
        GlobalCounters.mem_used_per_device[self.device] -= self.nbytes
      if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "free", self.trace_num))
      for dev, mb in self._bufs.items():
        if dev != self.device:
          Device[dev].allocator._unmap(mb)
      self.allocator.free(self._buf, self.nbytes, self.options)
      if ALLOC_TRACE and (aid:=getattr(self, '_at_alloc_id', -1)) >= 0: alloc_trace_record_free(aid)
    elif self._base is not None:
      self._base.allocated_views -= 1
    self._bufs.clear()
  def __reduce__(self):
    buf = None
    if self._base is not None:
      return self.__class__, (self.device, self.size, self.dtype, None, None, None, 0, self.base, self.offset, self.is_allocated())
    if self.device == "NPY": return self.__class__, (self.device, self.size, self.dtype, self._buf, self.options, None, self.uop_refcount)
    if self.is_allocated():
      buf = bytearray(self.nbytes)
      self.copyout(memoryview(buf))
    return self.__class__, (self.device, self.size, self.dtype, None, self.options, buf, self.uop_refcount)
  @property
  def trace_num(self) -> int:
    if not hasattr(self, '_trace_num'): self._trace_num = len(Buffer.profile_events)
    return self._trace_num
  @property
  def nbytes(self): return self.size*self.dtype.itemsize
  @suppress_finalizing
  def __del__(self): (self.device not in self._bufs) or self.deallocate()
  def __repr__(self):
    return f"<buf real:{self.is_allocated()} device:{self.device} size:{self.size} dtype:{self.dtype}" + \
           (f" offset:{self.offset}" if self._base is not None else "") + (f" {self.options=}" if self.options is not None else "") + ">"
  def as_memoryview(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    # zero copy with as_memoryview (disabled by default due to use after free)
    if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, '_as_buffer') and self.options is None:
      return self.allocator._as_buffer(self._buf)
    assert not force_zero_copy, "force zero copy was passed, but copy is required"
    return self.copyout(memoryview(bytearray(self.nbytes)))
  def numpy(self) -> 'np.ndarray': # type: ignore [name-defined] # noqa: F821
    import numpy as np
    assert _to_np_dtype(self.dtype.base) is not None, f"no np dtype for {self.dtype.base}"
    return np.frombuffer(self.as_memoryview(), dtype=_to_np_dtype(self.dtype.base))
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_initialized(), "can't copyin to unallocated buffer"
    self.allocator._copyin(self._buf, mv)
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_initialized(), "can't copyout unallocated buffer"
    self.allocator._copyout(mv, self._buf)
    return mv
  def view(self, size:int, dtype:DType, offset:int) -> Buffer:
    assert offset < self.nbytes, "offset must be less than nbytes"
    return Buffer(self.device, size, dtype, base=self.base, offset=self.offset+offset)

KERNARGS_AUDIT = ContextVar("KERNARGS_AUDIT", 0)        # 1 = record and report at exit, 2 = raise on the first hit
KERNARGS_WRAP_DRAIN = ContextVar("KERNARGS_WRAP_DRAIN", 1)  # THE FIX. 0 rolls back to the unguarded wrap.

# Same defect class as KERNARGS_WRAP_DRAIN above, at the PM4 indirect-buffer allocator (ops_amd.py pm4_ib_alloc):
# the PM4 command stream the CP fetches asynchronously lives inside that allocation, so recycling it under an
# in-flight submission corrupts commands rather than one field. 0 rolls back to the unguarded wrap.
PM4_IB_WRAP_DRAIN = ContextVar("PM4_IB_WRAP_DRAIN", 1)

# Same defect class again, at NV's command-queue allocator (ops_nv.py cmdq_allocator): the command buffer a
# GPFIFO entry points GPU-side execution at lives inside that allocation. UNTESTED on this machine -- no NVIDIA
# GPU available to reproduce or A/B; ships on by default on the strength of the analytic match to the other two
# sites, not a measured reproduction. 0 rolls back to the unguarded wrap.
NV_CMDQ_WRAP_DRAIN = ContextVar("NV_CMDQ_WRAP_DRAIN", 1)

# DETECTOR for the live gfx1100 fault signature. dmesg classifies it precisely: every fault at
# 0x0000ffffffbfe000 (56), 0x100000000 (22) and 0x0 (27) reports `Faulty UTCL2 client ID: SQC (inst)` --
# the INSTRUCTION cache. Those addresses are program counters, not data pointers; a wave was launched at a
# bogus PC. (The separate 0x00007xxx... faults all report TCP and are ordinary data OOB -- a different bug.)
#
# A wave's entry PC comes from hsa_kernel_dispatch_packet_t.kernel_object, and ops_amd.py:359 places that
# packet INSIDE the kernargs allocation (args_state.buf.offset(kernargs_segment_size)). Those allocations
# come from a BumpAllocator over a 16MiB buffer with wrap=True (hcq.py:438) that recycles from offset 0 with
# NO check that the memory it is reusing belongs to a dispatch still in flight. Overwrite a live dispatch
# packet and kernel_object becomes whatever lands there -- which is exactly a wild PC.
#
# So this records wraps that happen while the device timeline has not drained. It ONLY OBSERVES: no
# synchronization, no ordering change, and it costs one comparison per kernargs allocation (unlike the
# earlier data-pointer audit, which cost 41-71% and was aimed at the wrong client entirely).
#
# The fix this is designed to justify already exists in-tree for copy-staging buffers: HCQAllocatorBase
# .b_timeline tags each buffer with the timeline value at which it becomes safe. The portable fix is to
# extend that defer-until-drained discipline to the kernargs wrap.
_kernargs_wrap_hits: list[tuple[str,int,int]] = []
_kernargs_wrap_total: list[int] = [0]

def _audit_kernargs_wrap(dev, wrapped:bool) -> None:
  """Record a kernargs wrap that REUSED memory while the device still had un-drained work.

  Called after the guard, so a hit means the invariant was actually violated -- with KERNARGS_WRAP_DRAIN on
  this must stay 0, and with it off it reproduces the hazard.
  """
  if not KERNARGS_AUDIT or not wrapped: return
  _kernargs_wrap_total[0] += 1
  sig = getattr(dev, "timeline_signal", None)
  if sig is None: return                                  # non-HCQ backend: nothing to observe
  try: observed, pending = sig.value, getattr(dev, "timeline_value", 0) - 1
  except Exception: return                                # never let the detector break a run
  if observed < pending:
    _kernargs_wrap_hits.append((getattr(dev, "device", "?"), observed, pending))
    if KERNARGS_AUDIT >= 2:
      raise RuntimeError(f"KERNARGS_AUDIT: kernargs wrapped on {getattr(dev,'device','?')} with work still in "
                         f"flight (timeline {observed} < {pending}). A live dispatch packet may be overwritten, "
                         f"which launches a wave at a wild PC.")

def _dump_kernargs_audit() -> None:
  if not KERNARGS_AUDIT: return
  print(f"\n=== KERNARGS_AUDIT: {_kernargs_wrap_total[0]} kernargs wrap(s), "
        f"{len(_kernargs_wrap_hits)} with work still in flight ===")
  if _kernargs_wrap_hits:
    worst = max(_kernargs_wrap_hits, key=lambda h: h[2]-h[1])
    print(f"  widest gap: {worst[2]-worst[1]} timeline values behind on {worst[0]}")
atexit.register(_dump_kernargs_audit)

DISPATCH_TRACE = ContextVar("DISPATCH_TRACE", 0)  # 1 = serialize every dispatch and record it; dump on device error.

# FAULT-TO-DISPATCH CORRELATION PROBE. dmesg names the faulting VA and
# the faulting pid, but never the kernel that was running -- every probe that tried to infer it indirectly came
# back ambiguous. This is "the one probe that cannot come back ambiguous": force at most one dispatch to ever
# be in flight (synchronize after every single dispatch instead of only when wait=True), record its name+pid
# right before that synchronize, and if the synchronize raises -- a fault, hang, or timeout -- the last
# recorded dispatch IS the one that was executing when it happened. No inference, no ambiguity.
#
# DIAGNOSTIC ONLY, matching KERNARGS_AUDIT's shape: default-off ContextVar, and it is not something you would
# ever leave on -- serializing every dispatch is orders of magnitude slower than normal execution. That cost
# is deliberate and acceptable because this only runs while deliberately hunting a fault.
#
# Backend-agnostic by construction: gated on `getattr(dev, 'timeline_signal', None)`, the exact guard
# KERNARGS_AUDIT uses. Backends with no HCQ timeline signal (METAL, CUDA, CPU) never take the branch, so they
# are bit-identical whether this flag is 0 or 1.
_dispatch_trace_ring: deque = deque(maxlen=64)      # recent dispatches, oldest first, for context around a fault
_dispatch_trace_inflight: list[tuple|None] = [None]  # the one dispatch currently between submit and synchronize

def _dispatch_trace_before(dev, name:str, global_size, local_size) -> None:
  """Record the dispatch about to be waited on. Call AFTER submit, BEFORE the serializing dev.synchronize()."""
  if not DISPATCH_TRACE or getattr(dev, "timeline_signal", None) is None: return  # off, or non-HCQ: no-op
  rec = (getattr(dev, "device", "?"), name, os.getpid(), getattr(dev, "timeline_value", 0) - 1,
         tuple(global_size), tuple(local_size))
  _dispatch_trace_inflight[0] = rec
  _dispatch_trace_ring.append(rec)

def _dispatch_trace_after() -> None:
  """The just-recorded dispatch drained cleanly with no error. Call AFTER a successful dev.synchronize()."""
  if DISPATCH_TRACE: _dispatch_trace_inflight[0] = None

def _dispatch_trace_dump(dev, exc:BaseException) -> None:
  """Called from the synchronize() except-block, before any re-raise. Names the dispatch that was in flight."""
  if not DISPATCH_TRACE: return
  rec = _dispatch_trace_inflight[0]
  if rec is None:
    print(f"\n=== DISPATCH_TRACE: {getattr(dev,'device','?')} errored with no traced dispatch in flight "
          "(error predates the first traced dispatch) ===")
    return
  device, name, pid, tv, gs, ls = rec
  print(f"\n=== DISPATCH_TRACE: device error while '{name}' was in flight on {device} "
        f"(pid={pid}, timeline={tv}, global_size={gs}, local_size={ls}) ===\n    exception: {exc!r}")
  history = list(_dispatch_trace_ring)[:-1]
  if history:
    print("  preceding dispatches (most recent last):")
    for d, n, p, t, _, _ in history[-8:]: print(f"    {d} {n} pid={p} timeline={t}")

ALLOC_TRACE = ContextVar("ALLOC_TRACE", 0)  # 1 = record every allocation and dispatch to a fixed ring; dump at exit or via alloc_trace_dump().

# FAULT-TO-ALLOCATION ATTRIBUTION RING. Observed real-VA faults land in
# 0x00007xxx_xxxxx000, which is exactly where KFDIface.alloc's anon_mmap(0, ...) puts tinygrad's own buffers
# (ops_amd.py:795-823: tinygrad never chooses the GPU VA, the host mmap does, and the KFD ioctl echoes it
# back -- `assert addr == buf == mem.va_addr`, ops_amd.py:819). DISPATCH_TRACE (above) answers "what kernel
# was running" by serializing every dispatch, which is far too expensive to leave on for a full run and to
# use as a *allocation* probe (allocations are not the thing DISPATCH_TRACE watches). This is a SEPARATE,
# deliberately much cheaper mechanism: no serialization, no synchronize() added anywhere, records into a
# fixed-size preallocated ring (ctypes structures, no Python object churn in steady state) and only touches
# disk when the process exits or alloc_trace_dump() is called explicitly.
#
# Same default-off ContextVar shape as KERNARGS_AUDIT / DISPATCH_TRACE. Ring capacities are read once at
# import (functools.cache'd getenv) since they size a preallocated ctypes array -- changing them after the
# ring is created has no effect, which the self-test pins.
ALLOC_TRACE_ALLOCS      = getenv("ALLOC_TRACE_ALLOCS", 1 << 16)      # allocation-lifetime ring capacity
ALLOC_TRACE_DISPATCHES  = getenv("ALLOC_TRACE_DISPATCHES", 1 << 16)  # dispatch ring capacity
ALLOC_TRACE_MAX_ARGS    = getenv("ALLOC_TRACE_MAX_ARGS", 8)          # kernarg pointers recorded per dispatch
ALLOC_TRACE_FILE        = getenv("ALLOC_TRACE_FILE", "")             # dump path; "" = temp("alloc_trace.json") at dump time

# ts_ns fields use time.time_ns() (wall-clock, CLOCK_REALTIME via vDSO -- no syscall, ~20-40ns), not
# perf_counter_ns() (monotonic but epoch-less). Wall-clock is deliberate: it is the only clock a `dmesg` /
# `journalctl -k` line can be compared against, and that comparison is the entire point of this ring.
class _ATAllocRec(ctypes.Structure):
  _fields_ = [("alloc_id", ctypes.c_int64), ("device_id", ctypes.c_int32), ("_pad", ctypes.c_int32),
              ("va_start", ctypes.c_uint64), ("va_end", ctypes.c_uint64),
              ("req_size", ctypes.c_uint64), ("mapped_size", ctypes.c_uint64),
              ("alloc_seq", ctypes.c_int64), ("free_seq", ctypes.c_int64),  # free_seq == -1: not freed (as of dump)
              ("alloc_ts_ns", ctypes.c_int64), ("free_ts_ns", ctypes.c_int64)]  # free_ts_ns == -1: not freed

class _ATDispatchRec(ctypes.Structure):
  _fields_ = [("dispatch_id", ctypes.c_int64), ("device_id", ctypes.c_int32), ("kernel_id", ctypes.c_int32),
              ("gx", ctypes.c_uint32), ("gy", ctypes.c_uint32), ("gz", ctypes.c_uint32),
              ("lx", ctypes.c_uint32), ("ly", ctypes.c_uint32), ("lz", ctypes.c_uint32),
              ("submit_seq", ctypes.c_int64), ("signal_target", ctypes.c_int64),  # completion sequence, resolved at dump time
              ("submit_ts_ns", ctypes.c_int64),
              ("nargs", ctypes.c_uint32), ("_pad2", ctypes.c_uint32),
              ("arg_va", ctypes.c_uint64 * ALLOC_TRACE_MAX_ARGS), ("arg_size", ctypes.c_uint64 * ALLOC_TRACE_MAX_ARGS)]

_at_alloc_ring: ctypes.Array|None = None
_at_dispatch_ring: ctypes.Array|None = None
_at_alloc_count  = [0]  # next allocation id to hand out == total allocations recorded
_at_dispatch_count = [0]
_at_seq = [0]           # single monotonic Lamport-style clock shared by allocs, frees, and dispatch submits
_at_device_ids: dict[str,int] = {}
_at_device_names: list[str] = []
_at_kernel_ids: dict[str,int] = {}
_at_kernel_names: list[str] = []

def _at_init() -> None:
  global _at_alloc_ring, _at_dispatch_ring
  if _at_alloc_ring is None: _at_alloc_ring = (_ATAllocRec * ALLOC_TRACE_ALLOCS)()
  if _at_dispatch_ring is None: _at_dispatch_ring = (_ATDispatchRec * ALLOC_TRACE_DISPATCHES)()

def _at_id_for(name:str, ids:dict[str,int], names:list[str]) -> int:
  i = ids.get(name)
  if i is None: i = ids[name] = len(names); names.append(name)  # first-seen-only: no cost on the steady-state path
  return i

def alloc_trace_record_alloc(device:str, va_start:int, mapped_size:int, req_size:int) -> int:
  """Record a logical buffer allocation. O(1), no I/O, no heap allocation once (device) is already known.
  Returns the allocation id to hand back to alloc_trace_record_free, or -1 if tracing is off."""
  if not ALLOC_TRACE: return -1
  _at_init()
  aid, seq = _at_alloc_count[0], _at_seq[0]
  _at_alloc_count[0] += 1; _at_seq[0] += 1
  rec = _at_alloc_ring[aid % ALLOC_TRACE_ALLOCS]
  rec.alloc_id, rec.device_id = aid, _at_id_for(device, _at_device_ids, _at_device_names)
  rec.va_start, rec.va_end = va_start, va_start + mapped_size
  rec.req_size, rec.mapped_size = req_size, mapped_size
  rec.alloc_seq, rec.free_seq = seq, -1
  rec.alloc_ts_ns, rec.free_ts_ns = time.time_ns(), -1
  return aid

def alloc_trace_record_free(alloc_id:int) -> None:
  """Record the free of a previously-recorded allocation. No-op if tracing is off or the id was never issued
  (-1) or has already fallen out of the ring (wrapped past ALLOC_TRACE_ALLOCS allocations ago)."""
  if not ALLOC_TRACE or alloc_id < 0: return
  _at_init()
  ts = time.time_ns()
  if _at_alloc_count[0] - alloc_id <= ALLOC_TRACE_ALLOCS:
    rec = _at_alloc_ring[alloc_id % ALLOC_TRACE_ALLOCS]
    if rec.alloc_id == alloc_id: rec.free_seq, rec.free_ts_ns = _at_seq[0], ts
  _at_seq[0] += 1

def alloc_trace_record_dispatch(device:str, name:str, global_size, local_size, bufs, signal_target:int) -> None:
  """Record one dispatch: kernel identity, grid/local dims, and the VA+size of up to ALLOC_TRACE_MAX_ARGS
  pointer args (from `bufs`, objects with `.va_addr`/`.size` -- HCQBuffer; non-HCQ args are silently 0/0).
  `signal_target` is the timeline value this dispatch's completion signal will reach; completion is resolved
  against the live signal value once, at dump time -- no polling, no per-dispatch synchronize()."""
  if not ALLOC_TRACE: return
  _at_init()
  did, seq = _at_dispatch_count[0], _at_seq[0]
  _at_dispatch_count[0] += 1; _at_seq[0] += 1
  rec = _at_dispatch_ring[did % ALLOC_TRACE_DISPATCHES]
  rec.dispatch_id, rec.device_id = did, _at_id_for(device, _at_device_ids, _at_device_names)
  rec.kernel_id = _at_id_for(name, _at_kernel_ids, _at_kernel_names)
  gs, ls = (tuple(global_size) + (1,1,1))[:3], (tuple(local_size) + (1,1,1))[:3]
  rec.gx, rec.gy, rec.gz = gs
  rec.lx, rec.ly, rec.lz = ls
  rec.submit_seq, rec.signal_target, rec.submit_ts_ns = seq, signal_target, time.time_ns()
  n = min(len(bufs), ALLOC_TRACE_MAX_ARGS)
  rec.nargs = n
  for i in range(n):
    rec.arg_va[i], rec.arg_size[i] = getattr(bufs[i], 'va_addr', 0), getattr(bufs[i], 'size', 0)

def alloc_trace_dump(path:str|None=None) -> str|None:
  """Write the ring contents (and the id->name tables) to a JSON file. Safe to call more than once (e.g. once
  explicitly mid-run, then again at exit); each call is a full re-dump of current ring state. Returns the
  path written, or None if tracing was never turned on (nothing to dump)."""
  if not ALLOC_TRACE and _at_alloc_ring is None and _at_dispatch_ring is None: return None
  _at_init()
  # Resolve dispatch completion against the *current* signal value of every device seen, once, here -- not
  # per-dispatch. Best-effort: devices are looked up by canonicalized name; a device that was closed or never
  # opened in this process (e.g. re-analyzing a dump offline) just leaves completion unresolved (-1).
  signal_now: dict[int,int] = {}
  for dname, did in _at_device_ids.items():
    try:
      dev = Device[dname] if dname in Device._opened_devices else None
      sig = getattr(dev, "timeline_signal", None)
      if sig is not None: signal_now[did] = sig.value
    except Exception: pass  # never let the dump crash a real run

  n_alloc = min(_at_alloc_count[0], ALLOC_TRACE_ALLOCS)
  n_disp  = min(_at_dispatch_count[0], ALLOC_TRACE_DISPATCHES)
  allocs = []
  for aid in range(max(0, _at_alloc_count[0]-n_alloc), _at_alloc_count[0]):
    r = _at_alloc_ring[aid % ALLOC_TRACE_ALLOCS]
    if r.alloc_id != aid: continue  # slot was overwritten by a later id sharing the same modulus (shouldn't happen given n_alloc, but be safe)
    allocs.append({"alloc_id": r.alloc_id, "device": _at_device_names[r.device_id], "va_start": r.va_start, "va_end": r.va_end,
                    "req_size": r.req_size, "mapped_size": r.mapped_size, "alloc_seq": r.alloc_seq,
                    "free_seq": (None if r.free_seq < 0 else r.free_seq),
                    "alloc_ts_ns": r.alloc_ts_ns, "free_ts_ns": (None if r.free_ts_ns < 0 else r.free_ts_ns)})
  dispatches = []
  for did in range(max(0, _at_dispatch_count[0]-n_disp), _at_dispatch_count[0]):
    r = _at_dispatch_ring[did % ALLOC_TRACE_DISPATCHES]
    if r.dispatch_id != did: continue
    dname = _at_device_names[r.device_id]
    completed = signal_now.get(r.device_id) is not None and signal_now[r.device_id] >= r.signal_target
    dispatches.append({"dispatch_id": r.dispatch_id, "device": dname, "kernel": _at_kernel_names[r.kernel_id],
                        "global_size": [r.gx, r.gy, r.gz], "local_size": [r.lx, r.ly, r.lz],
                        "submit_seq": r.submit_seq, "signal_target": r.signal_target, "submit_ts_ns": r.submit_ts_ns,
                        "completed": completed if signal_now.get(r.device_id) is not None else None,
                        "args": [{"va": r.arg_va[i], "size": r.arg_size[i]} for i in range(r.nargs)]})

  out = {"format": "tinygrad-alloc-trace-v1", "dumped_at_unix": time.time(),
         "alloc_ring_capacity": ALLOC_TRACE_ALLOCS, "dispatch_ring_capacity": ALLOC_TRACE_DISPATCHES,
         "total_allocs_recorded": _at_alloc_count[0], "total_dispatches_recorded": _at_dispatch_count[0],
         "allocs": allocs, "dispatches": dispatches}
  p = path or ALLOC_TRACE_FILE or temp("alloc_trace.json")
  with open(p, "w") as f: json.dump(out, f)
  return p

def _dump_alloc_trace() -> None:
  if not ALLOC_TRACE: return
  p = alloc_trace_dump()
  if p is not None:
    print(f"\n=== ALLOC_TRACE: {_at_alloc_count[0]} allocation(s), {_at_dispatch_count[0]} dispatch(es) -> {p} ===")
atexit.register(_dump_alloc_trace)

DeviceType = TypeVar('DeviceType', bound='Compiled')

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator(Generic[DeviceType]):
  def __init__(self, dev:DeviceType, supports_copy_from_disk:bool=True, supports_transfer:bool=True):
    self.dev: DeviceType = dev
    self.default_buffer_spec: BufferSpec = BufferSpec()
    self.supports_copy_from_disk, self.supports_transfer = supports_copy_from_disk, supports_transfer
  # overridden in LRUAllocator
  def alloc(self, size:int, options:BufferSpec|None=None):
    assert size > 0, f"alloc size must be positive, getting {size}"
    try: return self._alloc(size, options if options is not None else self.default_buffer_spec)
    except (RuntimeError, MemoryError) as e: raise MemoryError(f"Allocation of {size_to_str(size)} failed on {self.dev.device}. "
                                                               f"Used: {size_to_str(GlobalCounters.mem_used_per_device[self.dev.device])}") from e
  def free(self, opaque, size:int, options:BufferSpec|None=None):
    self._free(opaque, options if options is not None else self.default_buffer_spec)

  # implemented by the runtime
  def _alloc(self, size:int, options:BufferSpec): raise NotImplementedError("need alloc")
  def _free(self, opaque, options:BufferSpec): pass  # if opaque is a Python object, you don't need a free
  def _copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def _copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")
  def _map(self, buf): raise NotImplementedError("need map")
  def _unmap(self, mb): pass  # default no-op; override if _map allocates iface-side state
  # def _as_buffer(self, src) -> memoryview:
  # def _offset(self, buf, size:int, offset:int):
  # def _transfer(self, dest, src, sz:int, src_dev, dest_dev):
  def _encode_decode(self, bufout, bufin, desc, hist:list, shape:tuple[int,...], frame_pos:int): raise NotImplementedError("need encdec") # optional

class LRUAllocator(Allocator, Generic[DeviceType]):
  """
  The LRU Allocator is responsible for caching buffers.
  It ensures that buffers are not freed until it is absolutely necessary, optimizing performance.
  """
  def __init__(self, dev:DeviceType, **kwargs):
    self.cache: dict[tuple[int, BufferSpec|None], Any] = defaultdict(list)
    super().__init__(dev, **kwargs)
  def alloc(self, size:int, options:BufferSpec|None=None):
    if len(c := self.cache[(size, options)]): return c.pop()
    try: return super().alloc(size, options)
    except (RuntimeError, MemoryError):
      self.free_cache()
      return super().alloc(size, options)
  def free_cache(self):
    for (sz,options),opaques in self.cache.items():
      for opaque in opaques: super().free(opaque, sz, options)
      opaques.clear()
  def free(self, opaque:Any, size:int, options:BufferSpec|None=None):
    if LRU and (options is None or (not options.nolru and options.external_ptr is None)): self.cache[(size, options)].append(opaque)
    else: super().free(opaque, size, options)

# **************** for Compiled Devices ****************

class CompileError(Exception): pass

class Compiler:
  # process-wide compiled-kernel cache counters (a cache hit reused a compiled lib; a miss actually compiled one)
  cache_hits: ClassVar[int] = 0
  cache_misses: ClassVar[int] = 0
  def __init__(self, cachekey:str|None=None): self.cachekey = cachekey if CCACHE else None
  def compile(self, src:str) -> bytes: return src.encode()   # NOTE: empty compiler is the default
  def compile_cached(self, src:str, cache_context:tuple[str, str]|None=None) -> bytes:
    # Keep the historical one-column string cache schema. Candidate context namespaces the source key without
    # migrating or invalidating existing compiler cache tables; the legacy key remains exactly `src`.
    key = src if cache_context is None else f"candidate:{cache_context[0]}:{cache_context[1]}\n{src}"
    if self.cachekey is not None and (lib := diskcache_get(self.cachekey, key)) is not None:
      Compiler.cache_hits += 1
      return lib
    assert not getenv("ASSERT_COMPILE"), f"tried to compile with ASSERT_COMPILE set\n{src}"
    lib = self.compile(src)
    if self.cachekey is not None: diskcache_put(self.cachekey, key, lib)
    Compiler.cache_misses += 1
    return lib
  def disassemble(self, lib:bytes): pass

class Compiled:
  profile_events:list[ProfileEvent] = [ProfileDeviceEvent("CPU")] # NOTE: CPU is the default device.

  def __init__(self, device:str, allocator:Allocator, renderers:list[type[Renderer]], runtime, graph=None, arch=None):
    from tinygrad.renderer import Renderer
    self.device, self.allocator, self.runtime, self.graph, self.renderers = device, allocator, runtime, graph, renderers or [Renderer]
    self.arch = arch
    self.cached_renderer:dict[Any, Renderer] = {}

  @property
  def renderer(self) -> Renderer: return self._select_renderer()

  @property
  def compiler(self) -> Compiler:
    if (ret:=self.renderer.compiler) is None: raise RuntimeError(f"no compiler for {self.device}")
    return ret

  def _renderer_name(self, r:type[Renderer]) -> str:
    return r.__name__.upper().removesuffix("RENDERER").removeprefix(devname:=self.device.split(':')[0].upper()) or devname

  def _select_renderer(self) -> Renderer:
    assert (rn:=next((self._renderer_name(r) for r in self.renderers if getenv(f"{self.device}_{self._renderer_name(r)}")), None)) is None, \
      f"{self.device}_{rn}=1 is deprecated, use DEV={self.device}:{rn} or {self.device}_CC={rn} instead"
    t = DEV.target(self.device.split(':')[0], **({"arch":self.arch} if self.arch else {}))
    return select_first_inited(select_by_name(self.renderers, self._renderer_name, t.renderer, f"{self.device} has no renderer {t.renderer!r}"),
                               f"No renderer for {self.device} is available", self.cached_renderer, t)

  def count(self) -> int:
    """
    Returns the number of physical accelerators available to the runtime.
    """
    return 1

  def synchronize(self):
    """
    Synchronize all pending operations on the device.

    This method ensures that all previously queued operations on the device have been completed before proceeding.
    """
    # override this in your device implementation
  def _at_profile_finalize(self):
    """
    Called at the end of profiling to allow the device to finalize any profiling.
    """
    # override this in your device implementation
  def finalize(self):
    """
    Called at the end of process lifetime to allow the device to finalize.
    """
    # override this in your device implementation

if PROFILE:
  @atexit.register
  def finalize_profile():
    devs = [Device[d] for d in Device._opened_devices]
    for dev in devs: dev.synchronize()
    for dev in devs: dev._at_profile_finalize()

    with open(fn:=temp("profile.pkl", append_user=True), "wb") as f: pickle.dump(cpu_events+Compiled.profile_events+Buffer.profile_events, f)

    PROFILE.value = 0
    from tinygrad.uop.ops import launch_viz
    launch_viz("PROFILE", fn)

def enumerate_devices_str() -> Generator[str, None, None]:
  from tinygrad import Tensor, Device

  for device in ALL_DEVICES:
    ren_results, iface_results = [], []
    try:
      d = Device[device]
      for iface in [i for i in getattr(d, 'ifaces', []) if not i.__name__.startswith("MOCK")]:
        try:
          name = iface.__name__[:-5]
          default_text, count = ("(default)", d.count()) if type(d.iface) is iface else (f"(DEV={name}+{device} to make default)", iface(d, 0).count) # type: ignore
          iface_results.append(f"{colored('+', 'green')} {name}: {pluralize('device', count)} {default_text}")
        except Exception as e: iface_results.append(f"{colored('-', 'red')} {iface.__name__[:-5]}: {e}")
      for r in d.renderers:
        try:
          with Context(CACHELEVEL=0, DEV=f"{device}:{d._renderer_name(r)}"): test = (Tensor([1,2,3], device=device) * 2).tolist()
          if test != [2,4,6]: raise ValueError(f"got {test} instead of [2, 4, 6]")
          default_text = '(default)' if type(d.renderer) is r else f'(DEV={device}:{d._renderer_name(r)} to make default)'
          ren_results.append(f"{colored('+', 'green')} {d._renderer_name(r)} {default_text}")
        except Exception as e: ren_results.append(f"{colored('-', 'red')} {d._renderer_name(r)}: {e}")
      result = (colored('PASS', 'green') + ("\n"+" "*12+"interfaces:\n" if iface_results else "") + '\n'.join([" "*13+x for x in iface_results]) +
                (("\n"+" "*12+"renderers:\n") + '\n'.join([" "*13+x for x in ren_results]) if len(ren_results) > 1 else ""))
    except Exception as e: result = f"{colored('FAIL', 'red')} {e}"
    yield f"{'*' if device == Device.DEFAULT else ' '} {device:8s}: {result}"

if __name__ == "__main__":
  for s in enumerate_devices_str(): print(s)
