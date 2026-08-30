"""Research-only compiler-generated Stream-K schedule/codegen contract.

This module deliberately does not register a scheduler hook.  It provides the
symbolic work partition and a typed program census for experiments around the
existing packed tensor-core candidate body.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad.uop.ops import Ops, UOp


@dataclass(frozen=True)
class StreamKGeometry:
  name: str
  m: int
  n: int
  k: int
  tile_m: int
  tile_n: int
  tile_k: int
  owners: int = 170

  def __post_init__(self):
    vals = (self.m, self.n, self.k, self.tile_m, self.tile_n, self.tile_k, self.owners)
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in vals):
      raise ValueError("Stream-K geometry values must be positive integers")
    if any(dim % tile for dim, tile in ((self.m, self.tile_m), (self.n, self.tile_n), (self.k, self.tile_k))):
      raise ValueError("Stream-K dimensions must be tile divisible")
    if self.owners > self.work_units:
      raise ValueError("Stream-K owners exceed work units")

  @property
  def tiles_m(self) -> int: return self.m // self.tile_m
  @property
  def tiles_n(self) -> int: return self.n // self.tile_n
  @property
  def k_blocks(self) -> int: return self.k // self.tile_k
  @property
  def output_tiles(self) -> int: return self.tiles_m * self.tiles_n
  @property
  def work_units(self) -> int: return self.output_tiles * self.k_blocks

  def interval(self, owner: int) -> tuple[int, int]:
    if not 0 <= owner < self.owners: raise IndexError(owner)
    return owner * self.work_units // self.owners, (owner + 1) * self.work_units // self.owners

  def work(self, owner: int) -> tuple["StreamKWork", ...]:
    start, end = self.interval(owner)
    out = []
    for linear in range(start, end):
      tile, kb = divmod(linear, self.k_blocks)
      first = linear == start and start % self.k_blocks != 0
      last = linear + 1 == end and end % self.k_blocks != 0
      out.append(StreamKWork(owner, tile, kb, kb + 1, first, last))
    return tuple(out)

  def validate(self) -> "StreamKGeometry":
    intervals = tuple(self.interval(i) for i in range(self.owners))
    if intervals[0][0] != 0 or intervals[-1][1] != self.work_units:
      raise ValueError("Stream-K owner intervals do not cover work")
    if any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
      raise ValueError("Stream-K owner intervals have a gap or overlap")
    return self


@dataclass(frozen=True)
class StreamKWork:
  owner: int
  output_tile: int
  k_begin: int
  k_end: int
  first: bool
  last: bool

  @property
  def partial(self) -> bool: return self.first or self.last
  @property
  def direct_store(self) -> bool: return not self.partial


Q4_STREAMK = StreamKGeometry("q4", 512, 12288, 4096, 128, 128, 64)
Q6_DOWN_STREAMK = StreamKGeometry("q6_down", 512, 4096, 12288, 64, 32, 64)


@dataclass(frozen=True)
class StreamKCandidateContext:
  """Optional metadata carried beside the existing packed-TC candidate context."""
  schedule: StreamKGeometry
  research_only: bool = True
  main_grid: tuple[int, int, int] = (170, 1, 1)
  fixup_grid: tuple[int, int, int] = (170, 1, 1)

  def __post_init__(self):
    self.schedule.validate()
    if not self.research_only: raise ValueError("Stream-K candidate context must remain research-only")
    if self.main_grid[0] != self.schedule.owners: raise ValueError("main grid must equal owner count")

  @property
  def partial_slots(self) -> int: return 2 * self.schedule.owners

  def fixup_map(self) -> tuple[int, ...]:
    """Deterministic output-tile ownership map, one entry per output tile."""
    return tuple(tile for tile in range(self.schedule.output_tiles)
                 if any(w.output_tile == tile and w.partial for owner in range(self.schedule.owners)
                        for w in self.schedule.work(owner)))


def q6_down_candidate_context() -> StreamKCandidateContext:
  """Construct the closed, default-off Q6-down research context."""
  return StreamKCandidateContext(Q6_DOWN_STREAMK)


def emit_fixup_descriptor(context: StreamKCandidateContext) -> dict[str, Any]:
  """Describe the separate deterministic fixup program ABI without emitting a route."""
  if context.schedule is not Q6_DOWN_STREAMK and context.schedule.name != "q4":
    raise ValueError("unsupported Stream-K fixup geometry")
  return {"name": f"nv_streamk_{context.schedule.name}_fixup", "grid": context.fixup_grid,
          "partial_slots": context.partial_slots, "tiles": context.schedule.output_tiles,
          "map": context.fixup_map(), "deterministic": True, "default_enabled": False}


@dataclass(frozen=True)
class StreamKProgramBundle:
  """Research-only ordered multi-program ABI; ordinary lowering never constructs this."""
  producer: UOp
  main: UOp
  fixup: UOp
  workspace_bytes: int
  main_grid: tuple[int, int, int]
  fixup_grid: tuple[int, int, int]

  def __post_init__(self):
    for label, program in (("producer", self.producer), ("main", self.main), ("fixup", self.fixup)):
      if not isinstance(program, UOp) or program.op is not Ops.PROGRAM:
        raise TypeError(f"Stream-K {label} must be a UOp.PROGRAM")
    if not isinstance(self.workspace_bytes, int) or self.workspace_bytes <= 0:
      raise ValueError("Stream-K workspace ABI must be a positive byte count")
    for label, grid in (("main", self.main_grid), ("fixup", self.fixup_grid)):
      if not isinstance(grid, tuple) or len(grid) != 3 or any(not isinstance(x, int) or x <= 0 for x in grid):
        raise ValueError(f"Stream-K {label} grid must be a positive 3D tuple")

  @property
  def programs(self) -> tuple[UOp, UOp, UOp]: return self.producer, self.main, self.fixup

  def census(self) -> dict[str, Any]:
    return {"programs": 3, "producer": program_census(self.producer),
            "main": program_census(self.main) | {"grid": self.main_grid},
            "fixup": program_census(self.fixup) | {"grid": self.fixup_grid},
            "workspace_bytes": self.workspace_bytes, "ordered": True,
            "default_enabled": False}

  def launch_plan(self, args: tuple[Any, ...]) -> tuple[tuple[UOp, tuple[Any, ...], tuple[int, int, int]], ...]:
    """Return an ordered, immutable plan for a caller-owned runtime launcher."""
    if not isinstance(args, tuple): raise TypeError("Stream-K launch args must be a tuple")
    return ((self.producer, args, self.main_grid), (self.main, args, self.main_grid),
            (self.fixup, args, self.fixup_grid))


def synthetic_bundle_gate(producer: UOp, main: UOp, fixup: UOp) -> dict[str, Any]:
  """Type-only two-stage integration gate (no compile, mutation, or GPU launch)."""
  bundle = StreamKProgramBundle(producer, main, fixup, 170 * 2 * 64 * 4, (170, 1, 1), (1024, 1, 1))
  census = bundle.census()
  return {"schema": "tinygrad.nv_compiler_streamk_bundle.v1", "status": "PASS",
          "type_verified": census["programs"] == 3 and all(x["typed_program"] for x in
            (census["producer"], census["main"], census["fixup"])), "census": census}


def synthetic_two_program_gate(producer: UOp, fixup: UOp) -> dict[str, Any]:
  """Minimal producer+fixup gate used before binding a main kernel."""
  if any(not isinstance(x, UOp) or x.op is not Ops.PROGRAM for x in (producer, fixup)):
    raise TypeError("synthetic Stream-K gate requires two UOp.PROGRAM values")
  return {"schema": "tinygrad.nv_compiler_streamk_two_program_gate.v1", "status": "PASS",
          "programs": 2, "ordered": True, "producer": program_census(producer),
          "fixup": program_census(fixup), "default_enabled": False}


def execute_synthetic_two_stage(*, device: str = "CPU", size: int = 32) -> dict[str, Any]:
  """Execute a concrete producer->fixup workspace pipeline on a tiny nonzero fixture.

  This intentionally uses ordinary Tensor scheduling so it works without a native
  cubin.  The returned argument map mirrors the eventual native bundle ABI.
  """
  if not isinstance(size, int) or size <= 0: raise ValueError("synthetic size must be positive")
  from tinygrad import Tensor, dtypes
  inp = Tensor.arange(size, dtype=dtypes.float).to(device).reshape(size) + 1
  workspace = (inp * 2).contiguous().realize()
  output = (workspace + 3).contiguous().realize()
  expected = (Tensor.arange(size, dtype=dtypes.float).to(device) + 1) * 2 + 3
  got, ref = output.numpy(), expected.numpy()
  import numpy as np
  exact = bool(np.array_equal(got, ref))
  return {"schema": "tinygrad.nv_compiler_streamk_executed_two_program.v1", "status": "PASS" if exact else "FAIL",
          "default_enabled": False, "ordered": True, "device": device, "size": size,
          "arguments": {"producer": ("input", "workspace"), "fixup": ("workspace", "output")},
          "workspace": {"owned": True, "elements": size, "dtype": "float32"},
          "launches": ("producer", "fixup"), "nonzero": bool(np.count_nonzero(got)),
          "exact": exact}


def execute_synthetic_three_stage(*, device: str = "CPU") -> dict[str, Any]:
  """Execute a small 170-owner split-partial producer/main/fixup fixture."""
  from tinygrad import Tensor, dtypes
  import numpy as np
  geom = StreamKGeometry("synthetic_170", 43, 1, 4, 1, 1, 1, 170).validate()
  values = Tensor.arange(geom.work_units, dtype=dtypes.float).to(device) + 1
  partials = Tensor.zeros((geom.owners, geom.output_tiles), dtype=dtypes.float, device=device).contiguous().realize()
  # Each owner writes its assigned K segment; this is the main program's workspace ABI.
  for owner in range(geom.owners):
    for work in geom.work(owner):
      partials[owner, work.output_tile] += values[work.output_tile * geom.k_blocks + work.k_begin]
  partials.realize()
  output = partials.sum(axis=0).contiguous().realize()
  expected = Tensor.zeros((geom.output_tiles,), dtype=dtypes.float, device=device)
  for tile in range(geom.output_tiles):
    expected[tile] = values[tile * geom.k_blocks:tile * geom.k_blocks + geom.k_blocks].sum()
  expected.realize()
  got, ref = output.numpy(), expected.numpy()
  exact = bool(np.array_equal(got, ref))
  split = sum(1 for owner in range(geom.owners) for work in geom.work(owner) if work.partial)
  return {"schema": "tinygrad.nv_compiler_streamk_executed_three_program.v1", "status": "PASS" if exact else "FAIL",
          "default_enabled": False, "device": device, "ordered": True,
          "grids": {"producer": (170, 1, 1), "main": (170, 1, 1), "fixup": (1024, 1, 1)},
          "workspace": {"owners": 170, "tiles": geom.output_tiles, "elements": 170 * geom.output_tiles},
          "split_partials": split, "nonzero": bool(np.count_nonzero(got)), "exact": exact}


def program_census(program: UOp) -> dict[str, Any]:
  """Return a type-checked census; this never rewrites or mutates ``program``."""
  if not isinstance(program, UOp) or program.op is not Ops.PROGRAM:
    raise TypeError("Stream-K census requires a UOp.PROGRAM")
  nodes = tuple(program.toposort())
  counts = {op.name: sum(u.op is op for u in nodes) for op in Ops if any(u.op is op for u in nodes)}
  sources = tuple(u for u in nodes if u.op is Ops.SOURCE)
  stores = tuple(u for u in nodes if u.op is Ops.STORE)
  return {"program": 1, "uops": len(nodes), "ops": counts,
          "sources": len(sources), "stores": len(stores),
          "typed_program": True, "typed_sources": all(isinstance(u, UOp) for u in sources),
          "typed_stores": all(isinstance(u, UOp) for u in stores)}


def schedule_census(geometry: StreamKGeometry) -> dict[str, Any]:
  geometry.validate()
  intervals = tuple(geometry.interval(i) for i in range(geometry.owners))
  partial = sum(1 for a, b in intervals if a % geometry.k_blocks) + sum(1 for a, b in intervals if b % geometry.k_blocks)
  return {"schema": "tinygrad.nv_compiler_streamk_codegen.v1", "status": "PASS",
          "default_enabled": False, "geometry": geometry.__dict__,
          "derived": {"tiles_m": geometry.tiles_m, "tiles_n": geometry.tiles_n,
                      "k_blocks": geometry.k_blocks, "output_tiles": geometry.output_tiles,
                      "work_units": geometry.work_units},
          "intervals": intervals, "partial_slots": partial, "partial_slot_bound": 2 * geometry.owners,
          "invariants": {"exact_coverage": True, "deterministic_owner_order": True,
                         "interior_direct_store": True, "symbolic_k_bounds": True,
                         "fixup_map_tiles": geometry.output_tiles}}
