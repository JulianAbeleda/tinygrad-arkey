"""CPU-only C3 memory certificate for frozen five-buffer epoch PROGRAMs.

The retained pre-lowering sink proves the logical element addresses.  The
selected native UOp graph proves which kernarg base and byte offset each AMD
global-memory instruction actually uses.  This module evaluates both over the
entire declared launch grid and fails closed on unsupported pointer/address
forms.  It creates no Device, runtime, allocator, or queue.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import struct
from typing import Any, Iterable, Mapping

from tinygrad.dtype import PtrDType
from tinygrad.renderer.isa.amd import AMDOps
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_exact_role_spec import ExactRoleSpec
from extra.qk.mmq_llama_five_buffer_graph import five_buffer_parameters


SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_epoch_memory_certificate.v1"
ABI_NAMES = ("output", "q4", "q8_values", "q8_scales", "q8_original_sums")
_NATIVE_GLOBAL_LOADS = frozenset((
  AMDOps.GLOBAL_LOAD, AMDOps.GLOBAL_LOAD_B64,
  AMDOps.GLOBAL_LOAD_B128, AMDOps.GLOBAL_LOAD_B128_GENERIC,
))
_NATIVE_GLOBAL_MEMORY = _NATIVE_GLOBAL_LOADS | frozenset((AMDOps.GLOBAL_STORE,))
_NATIVE_BINARY = {
  AMDOps.V_IADD: lambda a, b: a + b,
  AMDOps.V_IMUL: lambda a, b: a * b,
  AMDOps.V_OFFSET: lambda a, b: a << b,
  AMDOps.V_LSHR: lambda a, b: (a & 0xffffffff) >> b,
  AMDOps.V_AND: lambda a, b: a & b,
}
_NATIVE_LEAF = frozenset((AMDOps.WG_ID, AMDOps.WI_ID, AMDOps.MOV))


def _sha256_json(value: Any) -> str:
  return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _counter_digest(counter: Mapping[int, int], *, unique: bool = False) -> str:
  digest = hashlib.sha256(b"tinygrad.c3.unique.v1\0" if unique else b"tinygrad.c3.counter.v1\0")
  for element, count in sorted(counter.items()):
    digest.update(struct.pack("<Q", element))
    if not unique: digest.update(struct.pack("<Q", count))
  return digest.hexdigest()


class _ExactOnceCoverage:
  def __init__(self, elements: int):
    self.seen, self.accesses = bytearray(elements), 0

  def add(self, element: int, *, label: str, count: int = 1) -> None:
    if not 0 <= element < len(self.seen): raise ValueError(f"C3 {label} element {element} is outside its allocation")
    if count != 1 or self.seen[element]:
      raise ValueError(f"C3 {label} writes or reads output element {element} more than once")
    self.seen[element], self.accesses = 1, self.accesses + 1

  def complete(self) -> bool: return self.accesses == len(self.seen) and all(self.seen)

  def row(self) -> dict[str, Any]:
    digest = hashlib.sha256(b"tinygrad.c3.exact_once.v1\0" + self.seen).hexdigest()
    return {
      "accesses": self.accesses, "unique_elements": self.accesses,
      "min_element": 0 if self.seen else None, "max_element": len(self.seen)-1 if self.seen else None,
      "allocation_elements": len(self.seen), "coverage_sha256": digest, "unique_coverage_sha256": digest,
    }


def _peel_pointer_address(address: UOp) -> tuple[UOp, UOp | None] | None:
  offsets, cursor = [], address
  while True:
    if cursor.op is Ops.INDEX:
      if len(cursor.src) < 2: return None
      offsets.append(cursor.src[1])
      cursor = cursor.src[0]
      continue
    if cursor.op is Ops.AFTER and isinstance(cursor.dtype, PtrDType):
      if not cursor.src: return cursor, None
      cursor = cursor.src[0]
      continue
    break
  if not offsets: return cursor, None
  total = offsets[0]
  for offset in offsets[1:]: total = total + offset
  return cursor, total


def _effective_param_index(address: UOp) -> tuple[int, UOp] | None:
  peeled = _peel_pointer_address(address)
  if peeled is None: return None
  cursor, total = peeled
  if cursor.op is not Ops.PARAM or total is None: return None
  return int(cursor.arg.slot), total


def _global_param_slots(address: UOp) -> set[int]:
  return {
    int(node.arg.slot) for node in address.toposort()
    if node.op is Ops.PARAM and isinstance(node.dtype, PtrDType) and node.dtype.addrspace.name == "GLOBAL"
  }


def _special_expression(value: UOp) -> UOp:
  replacements = {
    node: UOp.variable(str(node.arg), int(node.vmin), int(node.vmax))
    for node in value.toposort() if node.op is Ops.SPECIAL
  }
  return value.substitute(replacements).simplify()


def _full_coordinates(role_spec: ExactRoleSpec) -> Iterable[dict[str, int]]:
  for gidx0 in range(role_spec.n // 128):
    for gidx1 in range(role_spec.m // 128):
      for lidx0 in range(256):
        yield {"gidx0": gidx0, "gidx1": gidx1, "lidx0": lidx0}


def _source_memory_expressions(sink: UOp) -> dict[tuple[str, int], tuple[UOp, ...]]:
  expressions: dict[tuple[str, int], list[UOp]] = {}
  for node in sink.toposort():
    if node.op not in (Ops.LOAD, Ops.STORE): continue
    peeled = _peel_pointer_address(node.src[0])
    terminal = node.src[0] if peeled is None else peeled[0]
    flattened, slots = _effective_param_index(node.src[0]), _global_param_slots(terminal)
    if slots and flattened is None:
      raise ValueError("C3 source memory address has an unsupported global pointer chain")
    if flattened is None: continue
    slot, address = flattened
    if slots != {slot} or slot not in range(5):
      raise ValueError("C3 source memory address does not resolve to exactly one five-buffer ABI slot")
    value = node if node.op is Ops.LOAD else node.src[1]
    if value.dtype.count != 1:
      raise ValueError("C3 source global vector memory must be scalarized before certification")
    expressions.setdefault((node.op.name.lower(), slot), []).append(_special_expression(address))
  return {key: tuple(values) for key, values in expressions.items()}


def _expected_source_counters(role_spec: ExactRoleSpec, epoch: int) -> dict[tuple[str, int], Counter[int] | None]:
  if not 0 <= epoch < role_spec.epochs: raise ValueError("C3 epoch is outside the admitted role")
  expected: dict[tuple[str, int], Counter[int] | None] = {
    ("load", 0): None, ("store", 0): None,
  }

  q4: Counter[int] = Counter()
  word_multiplicity = (8, 32, 32, 16) + (2,) * 32
  for tile_n in range(role_spec.n // 128):
    base = (tile_n * 128 * role_spec.epochs + epoch) * 36
    for row in range(128):
      for word, multiplicity in enumerate(word_multiplicity):
        q4[base + row * role_spec.epochs * 36 + word] = multiplicity * (role_spec.m // 128)
  expected[("load", 1)] = q4

  for slot, width in ((2, 128), (3, 4), (4, 4)):
    counter: Counter[int] = Counter()
    for tile_m in range(role_spec.m // 128):
      base = (epoch * 2 * role_spec.m + tile_m * 128) * width
      for phase in range(2):
        for row in range(128):
          for element in range(width):
            counter[base + (phase * role_spec.m + row) * width + element] = role_spec.n // 128
    expected[("load", slot)] = counter
  return expected


def _counter_row(counter: Mapping[int, int], *, elements: int) -> dict[str, Any]:
  if not counter: return {"accesses": 0, "unique_elements": 0, "min_element": None, "max_element": None,
                          "allocation_elements": elements, "coverage_sha256": _counter_digest({}),
                          "unique_coverage_sha256": _counter_digest({}, unique=True)}
  ordered = sorted(counter.items())
  return {
    "accesses": sum(counter.values()), "unique_elements": len(counter),
    "min_element": ordered[0][0], "max_element": ordered[-1][0],
    "allocation_elements": elements, "coverage_sha256": _counter_digest(counter),
    "unique_coverage_sha256": _counter_digest(counter, unique=True),
  }


def certify_source_sink_memory(role_spec: ExactRoleSpec, sink: UOp, epoch: int) -> dict[str, Any]:
  """Exhaustively prove logical source LOAD and output RMW STORE coverage."""
  if not isinstance(sink, UOp) or sink.op is not Ops.SINK:
    raise ValueError("C3 source authority must be a retained pre-lowering SINK")
  expressions = _source_memory_expressions(sink)
  expected = _expected_source_counters(role_spec, epoch)
  if set(expressions) != set(expected):
    raise ValueError(f"C3 source memory operation/slot census differs: {sorted(expressions)} != {sorted(expected)}")
  actual: dict[tuple[str, int], Counter[int] | _ExactOnceCoverage] = {
    key: _ExactOnceCoverage(role_spec.m * role_spec.n) if key[1] == 0 else Counter()
    for key in expressions
  }
  try:
    for coordinate in _full_coordinates(role_spec):
      for key, values in expressions.items():
        for value in values:
          element = int(value.sym_infer(coordinate))
          if isinstance(actual[key], _ExactOnceCoverage):
            actual[key].add(element, label=f"source {key[0]} output")
          else:
            actual[key][element] += 1
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(f"C3 source address cannot be evaluated over the full grid: {exc}") from exc

  parameters = five_buffer_parameters(*role_spec.shape)
  rows = []
  for key in sorted(expected):
    kind, slot = key
    if isinstance(actual[key], _ExactOnceCoverage):
      if expected[key] is not None: raise AssertionError("C3 internal output expectation mismatch")
      if not actual[key].complete(): raise ValueError(f"C3 source {kind} output coverage is incomplete")
      row = actual[key].row()
    else:
      assert expected[key] is not None
      if actual[key] != expected[key]:
        missing = expected[key] - actual[key]
        excess = actual[key] - expected[key]
        raise ValueError(
          f"C3 source {kind} {ABI_NAMES[slot]} coverage mismatch "
          f"(missing={sum(missing.values())}, excess={sum(excess.values())})")
      if actual[key] and (min(actual[key]) < 0 or max(actual[key]) >= parameters[slot].size):
        raise ValueError(f"C3 source {kind} {ABI_NAMES[slot]} address is outside its allocation")
      row = _counter_row(actual[key], elements=parameters[slot].size)
    rows.append({"kind": kind, "slot": slot, "name": ABI_NAMES[slot], **row})
  return {
    "authority": "retained_pre_lowering_sink", "epoch": epoch, "sink_key": sink.key.hex(),
    "full_grid": list(role_spec.program.grid), "local_size": [256, 1, 1],
    "rows": rows, "output_read_modify_write_complete_once": True,
  }


def _native_pointer_slot(pointer: UOp) -> int:
  while pointer.op is Ops.AFTER and pointer.src: pointer = pointer.src[0]
  if pointer.op is not Ops.INS or pointer.arg is not AMDOps.S_LOAD_PTR or len(pointer.src) < 2:
    raise ValueError("C3 native global memory base is not an S_LOAD_PTR")
  offset, parameter = pointer.src[:2]
  if offset.op is not Ops.CONST or type(offset.arg) is not int or parameter.op is not Ops.PARAM:
    raise ValueError("C3 native S_LOAD_PTR lost its constant kernarg slot")
  slot = int(parameter.arg.slot)
  if slot not in range(5) or int(offset.arg) != slot * 8:
    raise ValueError("C3 native S_LOAD_PTR kernarg byte offset differs from its ABI slot")
  return slot


def _native_specials(value: UOp, found: set[str] | None = None) -> set[str]:
  found = set() if found is None else found
  if value.op is Ops.SPECIAL:
    found.add(str(value.arg))
  elif value.op is Ops.AFTER and value.src:
    _native_specials(value.src[0], found)
  elif value.op in (Ops.CAST, Ops.BITCAST) and value.src:
    _native_specials(value.src[0], found)
  elif value.op is Ops.INS and value.arg in _NATIVE_BINARY:
    _native_specials(value.src[0], found); _native_specials(value.src[1], found)
  elif value.op is Ops.INS and value.arg in _NATIVE_LEAF and value.src:
    _native_specials(value.src[0], found)
  elif value.op is Ops.CONST:
    pass
  else:
    raise ValueError(f"C3 native address uses unsupported node {value.op.name}:{getattr(value.arg, 'name', value.arg)!r}")
  return found


def _eval_native_address(value: UOp, coordinate: Mapping[str, int], memo: dict[UOp, int]) -> int:
  if value in memo: return memo[value]
  if value.op is Ops.CONST:
    result = int(value.arg)
  elif value.op is Ops.SPECIAL:
    try: result = int(coordinate[str(value.arg)])
    except KeyError as exc: raise ValueError(f"C3 native address uses unknown SPECIAL {value.arg!r}") from exc
  elif value.op is Ops.AFTER and value.src:
    result = _eval_native_address(value.src[0], coordinate, memo)
  elif value.op in (Ops.CAST, Ops.BITCAST) and value.src:
    result = _eval_native_address(value.src[0], coordinate, memo)
  elif value.op is Ops.INS and value.arg in _NATIVE_LEAF and value.src:
    result = _eval_native_address(value.src[0], coordinate, memo)
  elif value.op is Ops.INS and value.arg in _NATIVE_BINARY and len(value.src) >= 2:
    lhs = _eval_native_address(value.src[0], coordinate, memo)
    rhs = _eval_native_address(value.src[1], coordinate, memo)
    result = _NATIVE_BINARY[value.arg](lhs, rhs)
  else:
    raise ValueError(f"C3 native address uses unsupported node {value.op.name}:{getattr(value.arg, 'name', value.arg)!r}")
  memo[value] = result & 0xffffffff
  return memo[value]


def _coordinate_projection(role_spec: ExactRoleSpec, names: set[str]) -> Iterable[dict[str, int]]:
  allowed = {"gidx0": role_spec.n // 128, "gidx1": role_spec.m // 128, "lidx0": 256}
  if not names <= set(allowed): raise ValueError(f"C3 native address uses unsupported launch axes {sorted(names - set(allowed))}")
  axes = tuple(name for name in ("gidx0", "gidx1", "lidx0") if name in names)
  if not axes:
    yield {}
    return
  def descend(index: int, row: dict[str, int]):
    if index == len(axes):
      yield dict(row)
      return
    name = axes[index]
    for value in range(allowed[name]):
      row[name] = value
      yield from descend(index + 1, row)
  yield from descend(0, {})


def _coordinate_repetition(role_spec: ExactRoleSpec, names: set[str]) -> int:
  extents = {"gidx0": role_spec.n // 128, "gidx1": role_spec.m // 128, "lidx0": 256}
  repetition = 1
  for name, extent in extents.items():
    if name not in names: repetition *= extent
  return repetition


def _native_memory_rows(program: UOp) -> tuple[tuple[str, int, UOp, int, int], ...]:
  rows = []
  all_ins = [node for node in program.src[0].toposort() if node.op is Ops.INS]
  pointer_slots = [_native_pointer_slot(node) for node in all_ins if node.arg is AMDOps.S_LOAD_PTR]
  if sorted(pointer_slots) != list(range(5)):
    raise ValueError("C3 native PROGRAM must load each five-buffer kernarg pointer exactly once")
  unsupported = [
    node for node in all_ins if getattr(node.arg, "name", "") in
    ("GATED_STORE",) and len(node.src) > 3 and node.src[3].op is Ops.CONST and node.src[3].arg == 0
  ]
  if unsupported: raise ValueError("C3 native global gated stores are not yet certifiable")
  for node in all_ins:
    if node.arg not in _NATIVE_GLOBAL_MEMORY: continue
    slot = _native_pointer_slot(node.src[1])
    if node.arg in _NATIVE_GLOBAL_LOADS:
      width = 8 if node.arg is AMDOps.GLOBAL_LOAD_B64 else 16 if node.arg in (
        AMDOps.GLOBAL_LOAD_B128, AMDOps.GLOBAL_LOAD_B128_GENERIC) else node.dtype.itemsize
      if len(node.src) < 3 or node.src[2].op is not Ops.CONST:
        raise ValueError("C3 native global load lost its constant immediate")
      rows.append(("load", slot, node.src[0], int(node.src[2].arg), width))
    else:
      if len(node.src) < 4 or node.src[-1].op is not Ops.CONST:
        raise ValueError("C3 native global store lost its scalarized item width")
      width = int(node.src[-1].arg)
      if width not in (1, 2, 4, 8): raise ValueError("C3 native global store has an unsupported item width")
      for lane in range(len(node.src) - 3):
        rows.append(("store", slot, node.src[0], lane * width, width))
  return tuple(rows)


def _native_address_fingerprint(value: UOp) -> tuple[Any, ...]:
  if value.op is Ops.CONST: return ("const", int(value.arg))
  if value.op is Ops.SPECIAL: return ("special", str(value.arg))
  if value.op is Ops.AFTER and value.src: return _native_address_fingerprint(value.src[0])
  if value.op in (Ops.CAST, Ops.BITCAST) and value.src:
    return (value.op.name, _native_address_fingerprint(value.src[0]))
  if value.op is Ops.INS and value.arg in _NATIVE_LEAF and value.src:
    return (value.arg.name, _native_address_fingerprint(value.src[0]))
  if value.op is Ops.INS and value.arg in _NATIVE_BINARY and len(value.src) >= 2:
    return (value.arg.name, _native_address_fingerprint(value.src[0]), _native_address_fingerprint(value.src[1]))
  raise ValueError(f"C3 native address uses unsupported node {value.op.name}:{getattr(value.arg, 'name', value.arg)!r}")


def certify_native_program_memory(role_spec: ExactRoleSpec, program: UOp, epoch: int,
                                  source_certificate: Mapping[str, Any] | None = None, *,
                                  _evaluation_cache: dict[tuple[Any, ...], tuple[int, ...]] | None = None
                                  ) -> dict[str, Any]:
  """Map every final native global instruction to an ABI base and byte bound."""
  if not isinstance(program, UOp) or program.op is not Ops.PROGRAM:
    raise ValueError("C3 native authority must be an Ops.PROGRAM")
  if not 0 <= epoch < role_spec.epochs: raise ValueError("C3 epoch is outside the admitted role")
  if not program.src or program.src[0].op is not Ops.SINK or tuple(program.arg.globals) != tuple(range(5)):
    raise ValueError("C3 native PROGRAM lost its closed five-buffer ABI")
  if tuple(program.arg.global_size) != role_spec.program.grid or tuple(program.arg.local_size or ()) != (256, 1, 1):
    raise ValueError("C3 native PROGRAM geometry differs from the role")
  parameters = five_buffer_parameters(*role_spec.shape)
  native_params = sorted(
    {node for node in program.src[0].toposort() if node.op is Ops.PARAM},
    key=lambda node: int(node.arg.slot))
  if len(native_params) != 5 or any(
      int(node.arg.slot) != parameter.slot or node.dtype != parameter.dtype.ptr(parameter.size)
      for node, parameter in zip(native_params, parameters)):
    raise ValueError("C3 native PROGRAM parameter dtypes/extents differ from the five-buffer ABI")
  memory = _native_memory_rows(program)
  if not memory: raise ValueError("C3 native PROGRAM contains no certifiable global memory instructions")
  counters: dict[tuple[str, int], Counter[int] | _ExactOnceCoverage] = {}
  instruction_census = Counter((kind, slot) for kind, slot, *_ in memory)
  for kind, slot, address, immediate, width in memory:
    element_width = parameters[slot].dtype.itemsize
    if width % element_width:
      raise ValueError(f"C3 native {kind} width splits a {ABI_NAMES[slot]} element")
    key = (kind, slot)
    counter = counters.setdefault(
      key, _ExactOnceCoverage(parameters[slot].size) if slot == 0 else Counter())
    names = _native_specials(address)
    repetition = _coordinate_repetition(role_spec, names)
    fingerprint = ("native", role_spec.shape, tuple(sorted(names)), _native_address_fingerprint(address))
    byte_bases = None if _evaluation_cache is None else _evaluation_cache.get(fingerprint)
    if byte_bases is None:
      byte_bases = tuple(
        _eval_native_address(address, coordinate, {})
        for coordinate in _coordinate_projection(role_spec, names))
      if _evaluation_cache is not None: _evaluation_cache[fingerprint] = byte_bases
    for byte_base in byte_bases:
      byte_start = byte_base + immediate
      byte_end = byte_start + width
      if byte_start < 0 or byte_end > parameters[slot].size * element_width:
        raise ValueError(
          f"C3 native {kind} {ABI_NAMES[slot]} byte range [{byte_start},{byte_end}) "
          f"exceeds allocation {parameters[slot].size * element_width}")
      if byte_start % element_width:
        raise ValueError(f"C3 native {kind} {ABI_NAMES[slot]} address is not element-aligned")
      for element in range(byte_start // element_width, byte_end // element_width):
        if isinstance(counter, _ExactOnceCoverage):
          counter.add(element, label=f"native {kind} output", count=repetition)
        else:
          counter[element] += repetition

  required = {("load", slot) for slot in range(5)} | {("store", 0)}
  if set(counters) != required:
    raise ValueError(f"C3 native memory operation/slot census differs: {sorted(counters)} != {sorted(required)}")
  if not all(isinstance(counters[key], _ExactOnceCoverage) and counters[key].complete()
             for key in (("load", 0), ("store", 0))):
    raise ValueError("C3 native output read/modify/write coverage is not complete exactly once")

  if source_certificate is not None:
    source_rows = {(str(row["kind"]), int(row["slot"])): row for row in source_certificate.get("rows", ())}
    for key, counter in counters.items():
      row = source_rows.get(key)
      native_unique_digest = counter.row()["unique_coverage_sha256"] if isinstance(counter, _ExactOnceCoverage) \
        else _counter_digest(counter, unique=True)
      if row is None or row.get("unique_coverage_sha256") != native_unique_digest:
        raise ValueError(f"C3 lowering changed the unique {key[0]} {ABI_NAMES[key[1]]} address set")

  rows = []
  for key in sorted(counters):
    kind, slot = key
    row = counters[key].row() if isinstance(counters[key], _ExactOnceCoverage) \
      else _counter_row(counters[key], elements=parameters[slot].size)
    rows.append({
      "kind": kind, "slot": slot, "name": ABI_NAMES[slot],
      "selected_instruction_lanes": instruction_census[key], **row,
    })
  return {
    "authority": "retained_final_selected_native_uop_graph", "epoch": epoch,
    "program_key": program.key.hex(), "global_memory_instruction_lanes": len(memory),
    "rows": rows, "all_native_global_bases_resolve_to_five_buffer_kernarg_slots": True,
    "all_native_effective_addresses_within_declared_allocations": True,
    "output_read_modify_write_complete_once": True,
  }


def certify_frozen_epoch_memory(role_spec: ExactRoleSpec, sink: UOp, program: UOp, epoch: int, *,
                                _evaluation_cache: dict[tuple[Any, ...], tuple[int, ...]] | None = None
                                ) -> dict[str, Any]:
  source = certify_source_sink_memory(role_spec, sink, epoch)
  native = certify_native_program_memory(role_spec, program, epoch, source, _evaluation_cache=_evaluation_cache)
  return {"epoch": epoch, "source_sink": source, "final_native": native}


def certify_frozen_epoch_program_family(role_spec: ExactRoleSpec, sinks: tuple[UOp, ...],
                                        programs: tuple[UOp, ...]) -> dict[str, Any]:
  """Certify an exact ordered family without loading or creating GPU state."""
  if len(sinks) != role_spec.epochs or len(programs) != role_spec.epochs:
    raise ValueError("C3 family must contain exactly one sink and PROGRAM per epoch")
  sink_keys, program_keys = tuple(sink.key.hex() for sink in sinks), tuple(program.key.hex() for program in programs)
  if len(set(sink_keys)) != role_spec.epochs or len(set(program_keys)) != role_spec.epochs:
    raise ValueError("C3 family sink and PROGRAM identities must be unique per epoch")
  evaluation_cache: dict[tuple[Any, ...], tuple[int, ...]] = {}
  variants = [
    certify_frozen_epoch_memory(role_spec, sink, program, epoch, _evaluation_cache=evaluation_cache)
    for epoch, (sink, program) in enumerate(zip(sinks, programs))
  ]
  body = {
    "schema": SCHEMA, "state": "PASS", "cpu_only": True,
    "role": {
      "name": role_spec.role, "shape": list(role_spec.shape), "epochs": role_spec.epochs,
      "candidate_identity": role_spec.candidate_canonical_identity,
    },
    "ordered_sink_keys": list(sink_keys), "ordered_program_keys": list(program_keys),
    "variants": variants,
  }
  return {**body, "certificate_sha256": _sha256_json(body)}


__all__ = [
  "SCHEMA", "certify_frozen_epoch_memory", "certify_frozen_epoch_program_family",
  "certify_native_program_memory", "certify_source_sink_memory",
]
