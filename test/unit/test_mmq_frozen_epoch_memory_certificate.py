from __future__ import annotations

import pytest

from tinygrad import dtypes
from tinygrad.renderer.isa.amd import AMDOps
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

from extra.qk.mmq_exact_role_spec import ExactRoleSpec
from extra.qk.mmq_frozen_epoch_memory_certificate import (
  SCHEMA, _eval_native_address, certify_frozen_epoch_memory, certify_frozen_epoch_program_family, certify_native_program_memory,
  certify_source_sink_memory,
)
from extra.qk.mmq_llama_five_buffer_full_kernel import build_llama_five_buffer_epoch_offset_family
from extra.qk.mmq_llama_five_buffer_graph import five_buffer_parameters


ROLE = ExactRoleSpec("unit_c3", 128, 128, 256, "0" * 64)


@pytest.fixture(scope="module")
def source_sink() -> UOp:
  return build_llama_five_buffer_epoch_offset_family(*ROLE.shape).variants[0].sink


def _ins(op: AMDOps, a: UOp, b: int) -> UOp:
  return UOp(Ops.INS, dtypes.int32, (a, UOp.const(dtypes.int32, b).rtag()), op)


def _synthetic_native_program(*, bad_q4_offset: bool = False,
                              bad_q4_pointer_offset: bool = False,
                              unsupported_q4_address: bool = False,
                              wrapped_q4_address: bool = False) -> UOp:
  parameters = five_buffer_parameters(*ROLE.shape)
  params = tuple(UOp.param(parameter.slot, parameter.dtype.ptr(parameter.size)) for parameter in parameters)
  pointers = tuple(UOp(
    Ops.INS, params[parameter.slot].dtype,
    (UOp.const(dtypes.int32, parameter.slot * 8 + (4 if bad_q4_pointer_offset and parameter.slot == 1 else 0)).rtag(),
     params[parameter.slot]),
    AMDOps.S_LOAD_PTR,
  ) for parameter in parameters)
  local = UOp(Ops.INS, dtypes.int32,
    (UOp.special(256, "lidx0"), UOp.const(dtypes.int32, 0).rtag()), AMDOps.WI_ID)
  memory = []

  # One lane owns 64 consecutive 256-element strips: exactly all 128x128 output elements.
  for iteration in range(64):
    output_element = _ins(AMDOps.V_IADD, local, iteration * 256)
    output_byte = _ins(AMDOps.V_OFFSET, output_element, 2)
    load = UOp(Ops.INS, dtypes.float32, (output_byte, pointers[0], UOp.const(dtypes.int32, 0).rtag()),
               AMDOps.GLOBAL_LOAD)
    memory.append(load)
    memory.append(UOp(Ops.INS, dtypes.void,
      (output_byte, pointers[0].after(load), load, UOp.const(dtypes.int32, 4).rtag()),
      AMDOps.GLOBAL_STORE))

  # Q4 has 128 rows x 36 words. Two adjacent lanes intentionally share a row;
  # native CSE/multiplicity is irrelevant to the required unique-address proof.
  q4_row = _ins(AMDOps.V_LSHR, local, 1)
  for word in range(36):
    element = _ins(AMDOps.V_IADD, _ins(AMDOps.V_IMUL, q4_row, 36), word)
    byte = _ins(AMDOps.V_OFFSET, element, 2)
    if bad_q4_offset and word == 35: byte = UOp.const(dtypes.int32, parameters[1].size * 4)
    if unsupported_q4_address and word == 35:
      byte = UOp(Ops.INS, dtypes.int32, (byte, UOp.const(dtypes.int32, 0)), AMDOps.V_OR)
    if wrapped_q4_address and word == 0:
      # The hardware result is zero after uint32 wrap and would therefore look
      # in-bounds unless the certificate audits arithmetic intermediates.
      byte = UOp(Ops.INS, dtypes.int32, (
        UOp.const(dtypes.uint32, 0xffffffff), UOp.const(dtypes.uint32, 1)), AMDOps.V_IADD)
    memory.append(UOp(Ops.INS, dtypes.uint32, (byte, pointers[1], UOp.const(dtypes.int32, 0).rtag()),
                      AMDOps.GLOBAL_LOAD))

  # Two Q8 records: 128 native lanes each cover one element in 128 consecutive strips.
  for iteration in range(128):
    byte = _ins(AMDOps.V_IADD, local, iteration * 256)
    memory.append(UOp(Ops.INS, dtypes.int8, (byte, pointers[2], UOp.const(dtypes.int32, 0).rtag()),
                      AMDOps.GLOBAL_LOAD))
  for slot in (3, 4):
    for iteration in range(4):
      element = _ins(AMDOps.V_IADD, local, iteration * 256)
      byte = _ins(AMDOps.V_OFFSET, element, 2)
      memory.append(UOp(Ops.INS, dtypes.float32, (byte, pointers[slot], UOp.const(dtypes.int32, 0).rtag()),
                        AMDOps.GLOBAL_LOAD))

  return UOp(Ops.PROGRAM, src=(UOp(Ops.SINK, src=tuple(memory)),), arg=ProgramInfo(
    name="unit_c3", globals=tuple(range(5)), global_size=ROLE.program.grid, local_size=(256, 1, 1)))


def test_c3_source_and_final_native_cover_full_grid_and_match_unique_sets(source_sink: UOp):
  certificate = certify_frozen_epoch_memory(ROLE, source_sink, _synthetic_native_program(), 0)
  assert certificate["source_sink"]["output_read_modify_write_complete_once"] is True
  assert certificate["final_native"]["all_native_global_bases_resolve_to_five_buffer_kernarg_slots"] is True
  assert certificate["final_native"]["all_native_effective_addresses_within_declared_allocations"] is True
  arithmetic = certificate["final_native"]["native_address_arithmetic"]
  assert arithmetic["evaluated_binary_operations"] > 0
  assert arithmetic["projected_address_evaluations"] > 0
  assert arithmetic["all_leaves_passthroughs_operands_and_results_range_checked_before_memoization"] is True
  rows = {(row["kind"], row["name"]): row for row in certificate["final_native"]["rows"]}
  assert rows[("load", "output")]["unique_elements"] == rows[("store", "output")]["unique_elements"] == 128 * 128
  assert rows[("load", "q4")]["unique_elements"] == 128 * 36
  assert rows[("load", "q8_values")]["unique_elements"] == 2 * 128 * 128


def test_c3_family_certificate_is_content_addressed_and_cpu_only(source_sink: UOp):
  certificate = certify_frozen_epoch_program_family(ROLE, (source_sink,), (_synthetic_native_program(),))
  assert certificate["schema"] == SCHEMA and certificate["state"] == "PASS"
  assert certificate["cpu_only"] is True and len(certificate["certificate_sha256"]) == 64
  assert len(certificate["variants"]) == 1


def test_c3_source_rejects_missing_output_store(source_sink: UOp):
  stores = [node for node in source_sink.toposort() if node.op is Ops.STORE and
            any(param.op is Ops.PARAM and param.arg.slot == 0 for param in node.src[0].toposort())]
  assert stores
  damaged = source_sink.substitute({stores[0]: stores[0].src[1]}, walk=True)
  with pytest.raises(ValueError, match="source memory operation/slot census|output coverage mismatch|source store output"):
    certify_source_sink_memory(ROLE, damaged, 0)


@pytest.mark.parametrize(("kwargs", "message"), (
  ({"bad_q4_offset": True}, "exceeds allocation"),
  ({"bad_q4_pointer_offset": True}, "kernarg byte offset"),
  ({"unsupported_q4_address": True}, "unsupported node"),
  ({"wrapped_q4_address": True}, "out-of-range uint32 value"),
))
def test_c3_native_rejects_out_of_bounds_wrong_base_and_opaque_address(kwargs, message):
  with pytest.raises(ValueError, match=message):
    certify_native_program_memory(ROLE, _synthetic_native_program(**kwargs), 0)


@pytest.mark.parametrize("op", (Ops.CAST, Ops.BITCAST))
def test_c3_native_rejects_uint64_value_that_casts_or_bitcasts_outside_uint32(op):
  wide = UOp.const(dtypes.uint64, 0x1_0000_0000)
  narrowed = UOp(op, dtypes.uint32, (wide,))
  with pytest.raises(ValueError, match="out-of-range uint32 value|BITCAST changes scalar storage width"):
    _eval_native_address(narrowed, {}, {})


@pytest.mark.parametrize(("value", "message"), (
  (UOp(Ops.INS, dtypes.int32, (
    UOp.const(dtypes.uint32, 1), UOp.const(dtypes.uint32, 32)), AMDOps.V_OFFSET), "invalid uint32 shift"),
  (UOp(Ops.INS, dtypes.int32, (
    UOp.const(dtypes.uint32, 1), UOp.const(dtypes.uint32, 32)), AMDOps.V_LSHR), "invalid uint32 shift"),
  (UOp(Ops.INS, dtypes.int32, (
    UOp.const(dtypes.uint32, 0x80000000), UOp.const(dtypes.uint32, 1)), AMDOps.V_OFFSET), "out-of-range uint32 value"),
  (UOp(Ops.INS, dtypes.int32, (
    UOp.const(dtypes.uint32, 0xffffffff), UOp.const(dtypes.uint32, 1)), AMDOps.V_IADD), "out-of-range uint32 value"),
  (UOp(Ops.INS, dtypes.int32, (
    UOp.const(dtypes.uint64, 0x1_0000_0000), UOp.const(dtypes.uint32, 0)), AMDOps.V_AND), "out-of-range uint32 value"),
  (UOp(Ops.INS, dtypes.int32, (UOp.const(dtypes.uint64, 0x1_0000_0000),), AMDOps.MOV), "out-of-range uint32 value"),
))
def test_c3_native_rejects_invalid_shift_binary_operand_result_and_mov_leaf(value, message):
  with pytest.raises(ValueError, match=message):
    _eval_native_address(value, {}, {})


def test_c3_native_rejects_out_of_range_special_leaf_before_memoization():
  special = UOp.special(1, "gidx0")
  with pytest.raises(ValueError, match="out-of-range uint32 value"):
    _eval_native_address(special, {"gidx0": 0x1_0000_0000}, {})


def test_c3_native_rejects_noninteger_const_and_wrong_workgroup_axis_semantics():
  with pytest.raises(ValueError, match="CONST must carry an exact integer"):
    _eval_native_address(UOp.const(dtypes.float32, 1.5), {}, {})
  wrong_axis = UOp(Ops.INS, dtypes.int32, (
    UOp.special(1, "gidx0"), UOp.const(dtypes.int32, 3)), AMDOps.WG_ID)
  with pytest.raises(ValueError, match="SPECIAL/axis semantics differ"):
    _eval_native_address(wrong_axis, {"gidx0": 0}, {})
