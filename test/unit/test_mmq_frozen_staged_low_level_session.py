from __future__ import annotations

import gc
import hashlib
from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest

from extra.qk.mmq_frozen_staged_low_level_session import (
  ABI_NAMES, ATTESTATION_SCHEMA, CANDIDATE_TRACE_SCHEMA,
  DIAGNOSTIC_GLOBAL_SIZE_ALLOWLIST,
  DIAGNOSTIC_PENDING_OBSERVATION_SCHEMA, DIAGNOSTIC_RECEIPT_SCHEMA,
  FrozenStagedAbiSlot, FrozenStagedLowLevelDependencies,
  FrozenStagedLowLevelInvocation, FrozenStagedLowLevelSession,
  FrozenStagedProgramAuthority, INVOCATION_FAILURE_ATTR,
  INVOCATION_FAILURE_SCHEMA, PM4_NO_DOORBELL_RECEIPT_SCHEMA,
  PM4_NO_DOORBELL_CHECK_KEYS,
  PM4_PRE_SUBMIT_CAPTURE_POINT, PM4_PRE_SUBMIT_CHECK_KEYS,
  PM4_PRE_SUBMIT_SCHEMA, _dispatch_production_runtime,
  production_frozen_staged_low_level_dependencies,
)


def _sid(label: str) -> str:
  return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _raw(label: str) -> str:
  return hashlib.sha256(label.encode()).hexdigest()


def _authority() -> FrozenStagedProgramAuthority:
  elements = (512 * 17408, 17408 * 36, 2 * 512 * 128,
              2 * 512 * 4, 2 * 512 * 4)
  itemsize = (4, 4, 1, 4, 4)
  dtype = ("float", "uint", "char", "float", "float")
  abi = tuple(FrozenStagedAbiSlot(
    slot, name, f"dtypes.{dtype[slot]}.ptr({elements[slot]})",
    elements[slot], elements[slot] * itemsize[slot],
    "inout" if slot == 0 else "in")
    for slot, name in enumerate(ABI_NAMES))
  return FrozenStagedProgramAuthority(
    family_identity=_sid("family"),
    candidate_executable_identity=_sid("candidate"),
    input_identity=_sid("input"),
    program_key=_raw("program"), binary_sha256=_raw("binary"),
    source_sha256=_raw("source"),
    serialized_program_sha256=_raw("serialized"),
    function_name="mmq_llama_five_buffer_full_grid_accumulate",
    compile_target="AMD:ISA:gfx1100", role="ffn_gate_up",
    full_shape=(512, 17408, 5120), program_shape=(512, 17408, 256),
    dispatch_count=20, global_size=(136, 4, 1),
    local_size=(256, 1, 1), globals=(0, 1, 2, 3, 4),
    abi=abi, requires_recompile=False).validate()


class FakeBuffer:
  def __init__(
      self, label: str, va: int, nbytes: int, *, device: str = "AMD",
      dtype: str | None = None, shape: tuple[int, ...] | None = None):
    self.label, self.va, self.nbytes = label, va, nbytes
    self.device, self.dtype, self.shape = device, dtype, shape


class FakeDependencies:
  def __init__(self, authority: FrozenStagedProgramAuthority):
    self.authority = authority
    self.events: list[str] = []
    self.next_va = 0x100000
    self.clock = 100
    self.runtime = object()
    self.runtime_row = {
      "queue_mode": "PM4", "runtime_class": "fake.AMDProgram",
      "runtime_name": authority.function_name, "runtime_device": "AMD",
      "runtime_object_identity": id(self.runtime),
      "runtime_device_identity_exact": True,
      "runtime_cache_binding_exact": True,
      "program_key": authority.program_key,
      "binary_sha256": authority.binary_sha256,
      "library_va": 0x80000000, "library_nbytes": 0x20000,
      "entry_va": 0x80000100,
    }
    self.binding_row = authority.binding_observation()
    self.q8_refs: list[weakref.ReferenceType[FakeBuffer]] = []
    self.launch_drift: str | None = None
    self.q8_alias_va: int | None = None
    self.q8_dtype_drift = False
    self.q8_shape_drift = False
    self.view_drift = 0
    self.fail_transfer = False
    self.cleanup_sync_failure = False
    self.cleanup_sync_armed = False
    self.dispatch_failure = False
    self.pm4_submit_policy = "execute"
    self.pm4_no_doorbell_receipt_sink: list[dict] = []
    self.pm4_snapshot_receipt_mode = "valid"
    self.pm4_snapshot_policy_calls: list[str] = []
    self.diagnostic_global_size: tuple[int, int, int] | None = None
    self.diagnostic_dispatch_receipt_sink: list[dict] | None = None

  def _allocate_bytes(self, label: str, nbytes: int) -> FakeBuffer:
    value = FakeBuffer(label, self.next_va, nbytes)
    self.next_va += ((nbytes + 0xffff) // 0x10000) * 0x10000
    return value

  def observe_binding(self, _binding):
    self.events.append("observe_binding")
    return dict(self.binding_row)

  def create_runtime(self, _binding):
    self.events.append("create_runtime")
    return self.runtime

  def observe_runtime(self, runtime):
    assert runtime is self.runtime
    self.events.append("observe_runtime")
    return dict(self.runtime_row)

  def allocate(self, slot):
    self.events.append(f"allocate:{slot.name}")
    return self._allocate_bytes(slot.name, slot.nbytes)

  def realize_many(self, values):
    self.events.append(f"realize:{len(values)}")

  def observe_buffer(self, value):
    return {
      "va": value.va, "nbytes": value.nbytes, "device": value.device,
      "dtype": value.dtype, "shape": value.shape}

  def zero_output(self, output):
    assert output.label == "output"
    self.events.append("zero_output")

  def produce_q8(self, common, authority):
    assert common is self.common
    assert authority is self.authority
    self.events.append(f"produce_q8:{id(common)}")
    q8_dtypes = ("dtypes.char", "dtypes.float", "dtypes.float")
    q8_shapes = ((40, 512, 128), (40, 512, 4), (40, 512, 4))
    values = tuple(
      FakeBuffer(
        f"full-{slot.name}", self.q8_alias_va,
        slot.nbytes * authority.dispatch_count,
        dtype=q8_dtypes[index], shape=q8_shapes[index])
      if self.q8_alias_va is not None else self._allocate_bytes(
        f"full-{slot.name}", slot.nbytes * authority.dispatch_count)
      for index, slot in enumerate(authority.abi[2:]))
    for index, value in enumerate(values):
      value.dtype, value.shape = q8_dtypes[index], q8_shapes[index]
    if self.q8_dtype_drift:
      values[0].dtype = "dtypes.uchar"
    if self.q8_shape_drift:
      values[0].shape = (1,)
    self.q8_refs.extend(weakref.ref(value) for value in values)
    return values

  def epoch_view(self, source, slot, epoch):
    self.events.append(f"view:{epoch}:{slot.name}")
    value = FakeBuffer(
      f"{source.label}[{epoch}]", source.va + epoch * slot.nbytes,
      slot.nbytes)
    value.va += self.view_drift
    return value

  def transfer(self, destination, source, nbytes):
    assert destination.nbytes == source.nbytes == nbytes
    self.events.append(
      f"transfer:{source.label.split('[')[0]}->{destination.label}")
    if self.fail_transfer:
      self.cleanup_sync_armed = self.cleanup_sync_failure
      raise RuntimeError("injected transfer failure")

  def synchronize(self):
    self.events.append("sync")
    if self.cleanup_sync_armed:
      raise RuntimeError("injected cleanup synchronization failure")

  def diagnostic_pre_submit(self, runtime, buffers):
    kernarg_va = 0x123456789000
    return {
      "schema": PM4_PRE_SUBMIT_SCHEMA,
      "capture_point": PM4_PRE_SUBMIT_CAPTURE_POINT,
      "runtime_object_identity": id(runtime),
      "runtime_class": self.runtime_row["runtime_class"],
      "runtime_name": self.runtime_row["runtime_name"],
      "runtime_device": self.runtime_row["runtime_device"],
      "kernarg_va": kernarg_va, "kernarg_nbytes": 40,
      "kernarg_qwords": [value.va for value in buffers],
      "pm4_kernarg_user_data": {
        "packet_dword_offset": 11, "register_index": 0,
        "low_dword": kernarg_va & 0xffffffff,
        "high_dword": kernarg_va >> 32, "pointer": kernarg_va,
      },
      "pm4_dispatch_direct": {
        "packet_dword_offset": 13,
        "group_counts": list(self.diagnostic_global_size),
        "dispatch_initiator": 1,
      },
      "pm4_workgroup_size": {
        "packet_dword_offset": 7, "register_index": 0,
        "size": list(self.authority.local_size),
      },
      "pm4_program_entry": {
        "packet_dword_offset": 3, "register_index": 0,
        "low_dword": self.runtime_row["entry_va"] >> 8,
        "high_dword": 0, "entry_va": self.runtime_row["entry_va"],
      },
      "argument_buffers": [
        {"slot": slot, "va": value.va, "size": value.nbytes}
        for slot, value in enumerate(buffers)],
      "pm4_dword_count": 17, "pm4_sha256": "3" * 64,
      "checks": {key: True for key in PM4_PRE_SUBMIT_CHECK_KEYS},
      "all_checks_pass": True,
    }

  def dispatch(self, runtime, buffers, authority, epoch):
    assert runtime is self.runtime
    self.events.append(f"dispatch:{epoch}")
    effective_grid = self.diagnostic_global_size or authority.global_size
    row = {
      "epoch": epoch, "program_key": authority.program_key,
      "binary_sha256": authority.binary_sha256,
      "global_size": list(effective_grid),
      "local_size": list(authority.local_size),
      "argument_vas": [value.va for value in buffers],
      "kernarg_pointer_words": [value.va for value in buffers],
      "kernarg_pointer_words_match_bound": True,
    }
    if self.diagnostic_global_size is not None:
      row.update({
        "frozen_global_size": list(authority.global_size),
        "effective_global_size": list(effective_grid),
        "pre_submit": self.diagnostic_pre_submit(runtime, buffers),
        "submit_evidence": {
          "native_submit_entered_count": 1,
          "native_submit_returned_count": 1,
          "target_submit_entered": True,
          "target_submit_returned": True,
          "target_dispatch_submitted": True,
        },
      })
    if self.launch_drift is not None:
      row[self.launch_drift] = \
        _raw("wrong") if self.launch_drift == "binary_sha256" else \
        False if self.launch_drift == "kernarg_pointer_words_match_bound" \
        else [1, 2, 3]
    if self.dispatch_failure:
      launch = {
        "epoch": epoch, "global_size": list(effective_grid),
        "local_size": list(authority.local_size),
        "arguments": [
          {
            "call_index": slot, "slot": slot, "name": ABI_NAMES[slot],
            "va": value.va, "base_va": value.va, "offset_bytes": 0,
            "nbytes": value.nbytes, "base_nbytes": value.nbytes,
            "va_matches_base_offset": True,
          }
          for slot, value in enumerate(buffers)],
        "kernarg": {
          "va": 0x123456789000, "size": 40,
          "bound_pointer_words": [value.va for value in buffers],
          "pointer_words": None,
          "pointer_words_read_error":
            "RuntimeError: simulated post-fault kernarg mapping loss",
        },
      }
      failure = RuntimeError("injected target dispatch fault")
      pre_submit_checks = {
        key: True for key in PM4_PRE_SUBMIT_CHECK_KEYS}
      failure.mmq_runtime_dispatch_failure = {
        "schema": "tinygrad.mmq_q4k_q8_1.runtime_dispatch_failure.v2",
        "failure_boundary":
          "runtime_call_raised_after_kernarg_capture_before_return",
        "wait": True, "launch": launch,
        "authoritative_qword_snapshot": "pre_submit",
        "pre_submit": {
          "schema":
            "tinygrad.mmq_q4k_q8_1.pm4_pre_submit_snapshot.v2",
          "capture_point":
            "AMDComputeQueue._submit_after_complete_command_construction_"
            "before_ring_copy_and_doorbell",
          "runtime_object_identity": id(runtime),
          "runtime_class": self.runtime_row["runtime_class"],
          "runtime_name": self.runtime_row["runtime_name"],
          "runtime_device": self.runtime_row["runtime_device"],
          "kernarg_va": launch["kernarg"]["va"],
          "kernarg_nbytes": launch["kernarg"]["size"],
          "kernarg_qwords": [value.va for value in buffers],
          "pm4_kernarg_user_data": {
            "packet_dword_offset": 11, "register_index": 0,
            "low_dword": launch["kernarg"]["va"] & 0xffffffff,
            "high_dword": launch["kernarg"]["va"] >> 32,
            "pointer": launch["kernarg"]["va"],
          },
          "pm4_dispatch_direct": {
            "packet_dword_offset": 13,
            "group_counts": list(effective_grid),
            "dispatch_initiator": 1,
          },
          "pm4_workgroup_size": {
            "packet_dword_offset": 7, "register_index": 0,
            "size": list(authority.local_size),
          },
          "pm4_program_entry": {
            "packet_dword_offset": 3, "register_index": 0,
            "low_dword": self.runtime_row["entry_va"] >> 8,
            "high_dword": 0, "entry_va": self.runtime_row["entry_va"],
          },
          "argument_buffers": [
            {"slot": slot, "va": value.va, "size": value.nbytes}
            for slot, value in enumerate(buffers)],
          "pm4_dword_count": 17, "pm4_sha256": "3" * 64,
          "checks": pre_submit_checks, "all_checks_pass": True,
        },
        "submit_evidence": {
          "native_submit_entered_count": 1,
          "native_submit_returned_count": 1,
          "target_submit_entered": True,
          "target_submit_returned": True,
          "target_dispatch_submitted": True,
        },
      }
      raise failure
    if self.pm4_submit_policy == "snapshot_only":
      def snapshot_dispatch(
          _runtime, _buffers, _globals, **kwargs):
        self.pm4_snapshot_policy_calls.append(kwargs["pm4_submit_policy"])
        launch = {
          "epoch": epoch,
          "arguments": [
            {"slot": slot, "va": value.va, "size": value.nbytes}
            for slot, value in enumerate(buffers)],
          "kernarg": {
            "pointer_words": [value.va for value in buffers],
            "pointer_words_match_bound": True,
          },
        }
        if self.pm4_snapshot_receipt_mode != "zero":
          checks = {key: True for key in PM4_PRE_SUBMIT_CHECK_KEYS}
          kernarg_va = 0x123456789000
          receipt = {
            "schema": PM4_NO_DOORBELL_RECEIPT_SCHEMA,
            "status": "CAPTURED_NO_SUBMIT",
            "submit_policy": "snapshot_only",
            "target_dispatch_submitted": False,
            "native_submit_call_count": 0,
            "ring_copy_performed": False,
            "doorbell_rung": False,
            "timeline_value_before": 40,
            "timeline_value_after_runtime_unwind": 41,
            "timeline_value_after_rollback": 40,
            "timeline_rollback_applied": True,
            "prof_exec_counter_before": 70,
            "prof_exec_counter_after_runtime_unwind": 71,
            "prof_exec_counter_after_rollback": 70,
            "timeline_signal_value_before": 39,
            "timeline_signal_value_after": 39,
            "terminal_child_required": True,
            "promotion_evidence_eligible": False,
            "checks": {
              key: True for key in PM4_NO_DOORBELL_CHECK_KEYS},
            "all_checks_pass": True,
            "pre_submit": {
              "schema": PM4_PRE_SUBMIT_SCHEMA,
              "capture_point": PM4_PRE_SUBMIT_CAPTURE_POINT,
              "runtime_object_identity": id(runtime),
              "runtime_class": self.runtime_row["runtime_class"],
              "runtime_name": self.runtime_row["runtime_name"],
              "runtime_device": self.runtime_row["runtime_device"],
              "kernarg_va": kernarg_va, "kernarg_nbytes": 40,
              "kernarg_qwords": [value.va for value in buffers],
              "pm4_kernarg_user_data": {
                "packet_dword_offset": 11, "register_index": 0,
                "low_dword": kernarg_va & 0xffffffff,
                "high_dword": kernarg_va >> 32,
                "pointer": kernarg_va,
              },
              "argument_buffers": [
                {"slot": slot, "va": value.va, "size": value.nbytes}
                for slot, value in enumerate(buffers)],
              "pm4_dword_count": 17, "pm4_sha256": "3" * 64,
              "checks": checks, "all_checks_pass": True,
            },
          }
          if self.pm4_snapshot_receipt_mode == "bad":
            receipt["target_dispatch_submitted"] = True
          launch["pm4_no_doorbell_receipt"] = receipt
        kwargs["runtime_evidence"]["launch_count"] = 1
        kwargs["runtime_evidence"]["launches"].append(launch)

      return _dispatch_production_runtime(
        runtime, buffers, authority, epoch,
        runtime_observation=self.runtime_row,
        dispatch_with_runtime_evidence=snapshot_dispatch,
        pm4_submit_policy=self.pm4_submit_policy,
        pm4_no_doorbell_receipt_sink=
          self.pm4_no_doorbell_receipt_sink,
        fixed_five_vas=tuple(value.va for value in buffers))
    return row

  def clock_ns(self):
    self.clock += 10
    return self.clock

  def build(self):
    return FrozenStagedLowLevelDependencies(
      observe_binding=self.observe_binding,
      create_runtime=self.create_runtime,
      observe_runtime=self.observe_runtime,
      allocate=self.allocate, realize_many=self.realize_many,
      observe_buffer=self.observe_buffer, zero_output=self.zero_output,
      produce_q8=self.produce_q8, epoch_view=self.epoch_view,
      transfer=self.transfer, synchronize=self.synchronize,
      dispatch=self.dispatch, clock_ns=self.clock_ns,
      diagnostic_global_size=self.diagnostic_global_size,
      diagnostic_dispatch_receipt_sink=
        self.diagnostic_dispatch_receipt_sink)


def _prepared():
  authority = _authority()
  fake = FakeDependencies(authority)
  fake.common = FakeBuffer(
    "resident-fp16", 0x90000000,
    authority.full_shape[0] * authority.full_shape[2] * 2)
  q4 = FakeBuffer(
    "epoch-major-q4", 0x40000000,
    authority.abi[1].nbytes * authority.dispatch_count)
  binding = object()
  session = FrozenStagedLowLevelSession.prepare(
    binding=binding, authority=authority,
    common_resident_fp16=fake.common, q4_epoch_major=q4,
    dependencies=fake.build())
  return authority, fake, binding, session


def _prepared_diagnostic(grid=(1, 1, 1)):
  authority = _authority()
  fake = FakeDependencies(authority)
  fake.diagnostic_global_size = grid
  fake.diagnostic_dispatch_receipt_sink = []
  fake.common = FakeBuffer(
    "resident-fp16", 0x90000000,
    authority.full_shape[0] * authority.full_shape[2] * 2)
  q4 = FakeBuffer(
    "epoch-major-q4", 0x40000000,
    authority.abi[1].nbytes * authority.dispatch_count)
  session = FrozenStagedLowLevelSession.prepare(
    binding=object(), authority=authority,
    common_resident_fp16=fake.common, q4_epoch_major=q4,
    dependencies=fake.build())
  return authority, fake, session


def _flatten_trace(trace):
  rows = [
    trace["activation_producer"], trace["route_setup"],
    trace["output_initialization"]]
  for epoch in trace["epochs"]:
    rows.extend(epoch[name] for name in (
      "gather", "q4_transfer", "q8_values_transfer",
      "q8_scales_transfer", "q8_sums_transfer", "staging_sync",
      "dispatch", "dispatch_sync"))
  return rows


@pytest.mark.parametrize("prefix", (1, 3, 20))
def test_invoke_has_exact_order_and_gap_free_trace(prefix):
  authority, fake, _, session = _prepared()
  invocation = session.invoke(prefix)
  assert invocation.output is session.output
  assert invocation.candidate_phase_trace["schema"] == CANDIDATE_TRACE_SCHEMA
  assert len(invocation.candidate_phase_trace["epochs"]) == prefix
  flattened = _flatten_trace(invocation.candidate_phase_trace)
  assert all(row["end_ns"] > row["start_ns"] for row in flattened)
  assert all(
    prior["end_ns"] == current["start_ns"]
    for prior, current in zip(flattened, flattened[1:]))

  relevant = [
    event for event in fake.events
    if event == "zero_output" or event == "sync" or
       event.startswith(("produce_q8", "view:", "transfer:", "dispatch:"))
  ]
  # Ignore the one preparation sync, then require the complete invocation
  # recurrence: producer sync, zero sync, 4 views, 4 ordered transfers,
  # staging sync, dispatch, dispatch sync.
  assert relevant[1:4] == [
    f"produce_q8:{id(fake.common)}", "sync", "zero_output"]
  assert relevant[4] == "sync"
  cursor = 5
  for epoch in range(prefix):
    assert relevant[cursor:cursor+4] == [
      f"view:{epoch}:q4", f"view:{epoch}:q8_values",
      f"view:{epoch}:q8_scales",
      f"view:{epoch}:q8_original_sums"]
    assert [row.split("->")[-1] for row in relevant[cursor+4:cursor+8]] == [
      "q4", "q8_values", "q8_scales", "q8_original_sums"]
    assert relevant[cursor+8:cursor+11] == [
      "sync", f"dispatch:{epoch}", "sync"]
    cursor += 11
  assert cursor == len(relevant)
  assert invocation.pending_observation["launch_count"] == prefix
  assert invocation.pending_observation["fixed_five_vas"] == \
    list(session.fixed_five_vas)
  assert session.common_resident_fp16 is fake.common
  assert authority.input_identity == \
    invocation.pending_observation["input_identity"]


def test_q8_lifetime_extends_through_post_sync_attestation_then_releases():
  _, fake, _, session = _prepared()
  invocation = session.invoke(3)
  gc.collect()
  assert session.has_pending_invocation
  assert len(fake.q8_refs) == 3
  assert all(ref() is not None for ref in fake.q8_refs)
  attestation = session.attest_post_sync(invocation, "PM4")
  assert attestation.schema == ATTESTATION_SCHEMA
  assert attestation.status == "PASS"
  assert attestation.launch_count == 3
  assert attestation.fixed_five_vas == session.fixed_five_vas
  assert attestation.runtime_object_identity == id(fake.runtime)
  assert attestation.runtime_cache_binding_exact is True
  assert attestation.observation_identity.startswith("sha256:")
  assert session.has_pending_invocation is False
  gc.collect()
  assert all(ref() is None for ref in fake.q8_refs)


def test_pending_invocation_blocks_reuse_and_wrong_invocation_fails_closed():
  _, _, _, session = _prepared()
  invocation = session.invoke(1)
  with pytest.raises(RuntimeError, match="not been attested"):
    session.invoke(1)
  impostor = FrozenStagedLowLevelInvocation(
    invocation.output, invocation.candidate_phase_trace,
    invocation.pending_observation)
  with pytest.raises(ValueError, match="identity differs"):
    session.attest_post_sync(impostor, "PM4")
  assert session.has_pending_invocation


def test_binding_drift_is_rejected_before_runtime_or_allocation():
  authority = _authority()
  fake = FakeDependencies(authority)
  fake.binding_row["binary_sha256"] = _raw("wrong")
  fake.common = FakeBuffer(
    "resident-fp16", 0x90000000,
    authority.full_shape[0] * authority.full_shape[2] * 2)
  q4 = FakeBuffer(
    "epoch-major-q4", 0x40000000,
    authority.abi[1].nbytes * authority.dispatch_count)
  with pytest.raises(ValueError, match="binding differs"):
    FrozenStagedLowLevelSession.prepare(
      binding=object(), authority=authority,
      common_resident_fp16=fake.common, q4_epoch_major=q4,
      dependencies=fake.build())
  assert fake.events == ["observe_binding"]


def test_prepare_rejects_buffer_extent_overlapping_runtime_code():
  authority = _authority()
  fake = FakeDependencies(authority)
  fake.next_va = fake.runtime_row["library_va"] - authority.abi[0].nbytes // 2
  fake.common = FakeBuffer(
    "resident-fp16", 0x90000000,
    authority.full_shape[0] * authority.full_shape[2] * 2)
  q4 = FakeBuffer(
    "epoch-major-q4", 0x40000000,
    authority.abi[1].nbytes * authority.dispatch_count)
  with pytest.raises(ValueError, match="overlap"):
    FrozenStagedLowLevelSession.prepare(
      binding=object(), authority=authority,
      common_resident_fp16=fake.common, q4_epoch_major=q4,
      dependencies=fake.build())


def test_runtime_code_range_drift_fails_closed_before_dispatch():
  _, fake, _, session = _prepared()
  fake.runtime_row["entry_va"] += 0x100
  with pytest.raises(ValueError, match="runtime/code range drifted"):
    session.invoke(1)
  assert not any(event.startswith("dispatch:") for event in fake.events)
  with pytest.raises(RuntimeError, match="failed closed"):
    session.invoke(1)


def test_runtime_cache_drift_fails_closed_before_dispatch():
  _, fake, _, session = _prepared()
  fake.runtime_row["runtime_cache_binding_exact"] = False
  with pytest.raises(ValueError, match="runtime cache"):
    session.invoke(1)
  assert not any(event.startswith("dispatch:") for event in fake.events)


def test_fixed_va_drift_is_rejected_at_attestation():
  _, fake, _, session = _prepared()
  invocation = session.invoke(1)
  session.stages[0].va = session.output.va + session.output.nbytes - 32
  with pytest.raises(ValueError, match="extents overlap"):
    session.attest_post_sync(invocation, "PM4")
  assert session.has_pending_invocation is False



@pytest.mark.parametrize("field", (
  "binary_sha256", "global_size", "local_size", "argument_vas",
  "kernarg_pointer_words", "kernarg_pointer_words_match_bound",
))
def test_every_launch_authority_field_drift_is_rejected(field):
  _, fake, _, session = _prepared()
  fake.launch_drift = field
  invocation = session.invoke(1)
  with pytest.raises(ValueError, match="launch observation"):
    session.attest_post_sync(invocation, "PM4")


def test_queue_drift_fails_closed_and_retains_no_pending_state():
  _, _, _, session = _prepared()
  invocation = session.invoke(1)
  with pytest.raises(ValueError, match="queue mode differs"):
    session.attest_post_sync(invocation, "AQL")
  assert session.has_pending_invocation is False


def _binding_for_authority(
    authority: FrozenStagedProgramAuthority, manifest_abi=None):
  manifest_abi = manifest_abi or [
    {"slot": slot.slot, "name": slot.name,
     "dtype": slot.dtype,
     "elements": slot.elements}
    for slot in authority.abi]
  role = SimpleNamespace(
    role=authority.role, shape=authority.full_shape,
    epochs=authority.dispatch_count,
    program=SimpleNamespace(
      shape=authority.program_shape,
      abi_elements=tuple(slot.elements for slot in authority.abi)))
  manifest = {
    "program": {
      "abi": manifest_abi, "function": authority.function_name,
      "compile_target": authority.compile_target,
      "global_size": list(authority.global_size),
      "local_size": list(authority.local_size),
      "globals": list(authority.globals),
    },
    "artifacts": {
      "serialized_program_sha256": authority.serialized_program_sha256},
    "consumer": {"requires_recompile": False},
  }
  arg = SimpleNamespace(
    vals=lambda _values: (), function_name=authority.function_name,
    global_size=authority.global_size, local_size=authority.local_size,
    globals=authority.globals,
    outs=(0,), ins=tuple(range(5)))
  program = SimpleNamespace(
    arg=arg, key=SimpleNamespace(hex=lambda: authority.program_key),
    src=(
      SimpleNamespace(op=SimpleNamespace(name="SOURCE"), arg="source"),
      SimpleNamespace(op=SimpleNamespace(name="BINARY"), arg=b"binary"),
      SimpleNamespace(op=SimpleNamespace(name="DEVICE"), arg="AMD")))
  return SimpleNamespace(
    role_spec=role,
    artifact=SimpleNamespace(
      manifest=manifest, program=program, source="source", binary=b"binary"),
    program_key=authority.program_key,
    binary_sha256=authority.binary_sha256,
    source_sha256=authority.source_sha256)


def test_program_authority_from_binding_derives_exact_bundle_fields():
  authority = _authority()
  binding = _binding_for_authority(authority)
  rebuilt = FrozenStagedProgramAuthority.from_binding(
    binding, family_identity=authority.family_identity,
    candidate_executable_identity=authority.candidate_executable_identity,
    input_identity=authority.input_identity)
  assert rebuilt == authority


@pytest.mark.parametrize("change", (
  {"role": "not_ffn_gate_up"},
  {"full_shape": (1, 1, 5120), "program_shape": (1, 1, 256)},
  {"global_size": (1, 1, 1)},
  {"local_size": (1, 1, 1)},
  {"function_name": "wrong"},
  {"compile_target": "AMD:ISA:gfx9999"},
))
def test_program_authority_rejects_nonexact_role_geometry(change):
  with pytest.raises(ValueError):
    replace(_authority(), **change).validate()


@pytest.mark.parametrize("field,value", (
  ("slot", 99), ("elements", 1), ("dtype", "bogus.pointer"),
))
def test_program_authority_from_binding_rejects_manifest_abi_drift(
    field, value):
  authority = _authority()
  manifest_abi = [
    {"slot": slot.slot, "name": slot.name, "dtype": slot.dtype,
     "elements": slot.elements} for slot in authority.abi]
  manifest_abi[0][field] = value
  binding = _binding_for_authority(authority, manifest_abi)
  with pytest.raises(ValueError, match="manifest ABI"):
    FrozenStagedProgramAuthority.from_binding(
      binding, family_identity=authority.family_identity,
      candidate_executable_identity=authority.candidate_executable_identity,
      input_identity=authority.input_identity)


@pytest.mark.parametrize("mutation", (
  "program_key", "function", "global_size", "outs", "device",
  "source_payload", "requires_recompile",
))
def test_program_authority_from_binding_rejects_actual_program_drift(mutation):
  authority = _authority()
  binding = _binding_for_authority(authority)
  if mutation == "program_key":
    binding.artifact.program.key = SimpleNamespace(hex=lambda: _raw("wrong"))
  elif mutation == "function":
    binding.artifact.program.arg.function_name = "wrong"
  elif mutation == "global_size":
    binding.artifact.program.arg.global_size = (1, 1, 1)
  elif mutation == "outs":
    binding.artifact.program.arg.outs = ()
  elif mutation == "device":
    binding.artifact.program.src[-1].arg = "CPU"
  elif mutation == "source_payload":
    binding.artifact.program.src[0].arg = "wrong"
  else:
    binding.artifact.manifest["consumer"]["requires_recompile"] = True
  with pytest.raises(ValueError):
    FrozenStagedProgramAuthority.from_binding(
      binding, family_identity=authority.family_identity,
      candidate_executable_identity=authority.candidate_executable_identity,
      input_identity=authority.input_identity)


def test_prepare_rejects_resident_or_q4_alias_with_persistent_buffers():
  authority = _authority()
  for alias in ("resident", "q4"):
    fake = FakeDependencies(authority)
    common = FakeBuffer(
      "resident-fp16",
      fake.next_va if alias == "resident" else 0x90000000,
      authority.full_shape[0] * authority.full_shape[2] * 2)
    q4 = FakeBuffer(
      "epoch-major-q4",
      fake.next_va if alias == "q4" else 0x40000000,
      authority.abi[1].nbytes * authority.dispatch_count)
    with pytest.raises(ValueError, match="overlap"):
      FrozenStagedLowLevelSession.prepare(
        binding=object(), authority=authority,
        common_resident_fp16=common, q4_epoch_major=q4,
        dependencies=fake.build())


def test_invoke_rejects_q8_aliases_and_epoch_view_drift():
  _, fake, _, session = _prepared()
  fake.q8_alias_va = session.output.va
  with pytest.raises(ValueError, match="overlap"):
    session.invoke(1)

  _, fake, _, session = _prepared()
  fake.view_drift = 4
  with pytest.raises(ValueError, match="source view VA differs"):
    session.invoke(1)


@pytest.mark.parametrize("kind", ("resident_device", "q4_device"))
def test_prepare_rejects_foreign_device_inputs(kind):
  authority = _authority()
  fake = FakeDependencies(authority)
  fake.common = FakeBuffer(
    "resident-fp16", 0x90000000,
    authority.full_shape[0] * authority.full_shape[2] * 2,
    device="CPU" if kind == "resident_device" else "AMD")
  q4 = FakeBuffer(
    "epoch-major-q4", 0x40000000,
    authority.abi[1].nbytes * authority.dispatch_count,
    device="CPU" if kind == "q4_device" else "AMD")
  with pytest.raises(ValueError, match="resident on AMD"):
    FrozenStagedLowLevelSession.prepare(
      binding=object(), authority=authority,
      common_resident_fp16=fake.common, q4_epoch_major=q4,
      dependencies=fake.build())


@pytest.mark.parametrize("kind", ("dtype", "shape"))
def test_invoke_rejects_wrong_q8_source_metadata(kind):
  _, fake, _, session = _prepared()
  setattr(fake, f"q8_{kind}_drift", True)
  error = TypeError if kind == "dtype" else ValueError
  with pytest.raises(error, match=f"q8_values source {kind} differs"):
    session.invoke(1)


def test_q8_failure_cleanup_drains_before_release_or_retains_on_sync_failure():
  _, fake, _, session = _prepared()
  fake.fail_transfer = True
  with pytest.raises(RuntimeError, match="transfer failure"):
    session.invoke(1)
  gc.collect()
  assert all(ref() is None for ref in fake.q8_refs)

  _, fake, _, session = _prepared()
  fake.fail_transfer = True
  fake.cleanup_sync_failure = True
  with pytest.raises(RuntimeError, match="transfer failure"):
    session.invoke(1)
  gc.collect()
  assert session._failed_q8_sources is not None
  assert all(ref() is not None for ref in fake.q8_refs)


def test_dispatch_fault_retains_exact_low_level_phase_runtime_and_launch():
  authority, fake, _, session = _prepared()
  fake.dispatch_failure = True
  with pytest.raises(
      RuntimeError, match="injected target dispatch fault") as caught:
    session.invoke(1)
  failure = getattr(caught.value, INVOCATION_FAILURE_ATTR)
  assert failure["schema"] == INVOCATION_FAILURE_SCHEMA
  assert failure["phase"] == "epoch_dispatch"
  assert failure["subphase"] == \
    "runtime_call_raised_after_kernarg_capture_before_return"
  assert failure["epoch"] == 0 and failure["queue_mode"] == "PM4"
  assert failure["family_identity"] == authority.family_identity
  assert failure["program_key"] == authority.program_key
  assert failure["binary_sha256"] == authority.binary_sha256
  assert failure["runtime_observation"] == \
    session.prepared_runtime_observation
  assert failure["runtime_observation"]["runtime_object_identity"] == \
    id(fake.runtime)
  assert failure["fixed_five_vas"] == list(session.fixed_five_vas)
  dispatch = failure["dispatch_failure"]
  assert dispatch["all_authority_checks_pass"] is True
  assert all(dispatch["checks"].values())
  assert dispatch["authoritative_qword_snapshot"] == "pre_submit"
  assert dispatch["pre_submit"]["kernarg_qwords"] == \
    list(session.fixed_five_vas)
  assert [row["va"] for row in
          dispatch["pre_submit"]["argument_buffers"]] == \
    list(session.fixed_five_vas)
  assert dispatch["pre_submit"]["runtime_object_identity"] == id(fake.runtime)
  assert dispatch["pre_submit"]["pm4_kernarg_user_data"]["pointer"] == \
    dispatch["pre_submit"]["kernarg_va"]
  assert not any(dispatch["post_failure_checks"].values())
  assert dispatch["launch"]["global_size"] == \
    list(authority.global_size)
  assert dispatch["launch"]["local_size"] == \
    list(authority.local_size)
  assert [row["va"] for row in dispatch["launch"]["arguments"]] == \
    list(session.fixed_five_vas)
  assert dispatch["launch"]["kernarg"]["pointer_words"] is None
  assert "simulated post-fault kernarg mapping loss" in \
    dispatch["launch"]["kernarg"]["pointer_words_read_error"]
  assert session.has_pending_invocation is False
  with pytest.raises(RuntimeError, match="failed closed"):
    session.invoke(1)


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_dispatch_fault_rejects_mutated_pre_submit_internal_check_keys(
    mutation):
  _, fake, _, session = _prepared()
  fake.dispatch_failure = True
  original_dispatch = session.dependencies.dispatch

  def mutate_failure(*args, **kwargs):
    try:
      return original_dispatch(*args, **kwargs)
    except RuntimeError as exc:
      checks = exc.mmq_runtime_dispatch_failure["pre_submit"]["checks"]
      if mutation == "missing":
        checks.pop("pm4_packet_stream_decoded")
      else:
        checks["fabricated_only_check"] = True
      raise

  object.__setattr__(session.dependencies, "dispatch", mutate_failure)
  with pytest.raises(
      RuntimeError, match="injected target dispatch fault") as caught:
    session.invoke(1)
  dispatch = getattr(
    caught.value, INVOCATION_FAILURE_ATTR)["dispatch_failure"]
  assert dispatch["checks"]["pre_submit_internal_checks_exact"] is False
  assert dispatch["all_authority_checks_pass"] is False


@pytest.mark.parametrize("field,value", (
  ("runtime_name", "wrong"),
  ("runtime_device", "AMD:1"),
  ("runtime_device_identity_exact", False),
  ("binary_sha256", "0" * 64),
))
def test_poison_runtime_identity_fails_closed(field, value):
  _, fake, _, session = _prepared()
  fake.runtime_row[field] = value
  with pytest.raises(ValueError):
    session.invoke(1)


def test_snapshot_only_receipt_is_exactly_once_copied_and_syncs_after_rollback():
  _, fake, _, session = _prepared()
  fake.pm4_submit_policy = "snapshot_only"
  invocation = session.invoke(1)

  assert fake.pm4_snapshot_policy_calls == ["snapshot_only"]
  assert len(fake.pm4_no_doorbell_receipt_sink) == 1
  receipt = fake.pm4_no_doorbell_receipt_sink[0]
  assert receipt["status"] == "CAPTURED_NO_SUBMIT"
  assert receipt["target_dispatch_submitted"] is False
  assert receipt["native_submit_call_count"] == 0
  assert receipt["timeline_rollback_applied"] is True
  assert fake.events[-2:] == ["dispatch:0", "sync"]

  launch_receipt = \
    session._pending.launch_observations[0]["pm4_no_doorbell_receipt"]
  assert launch_receipt == receipt
  assert launch_receipt is not receipt
  launch_receipt["pre_submit"]["kernarg_qwords"][0] = 1
  assert receipt["pre_submit"]["kernarg_qwords"] == \
    list(session.fixed_five_vas)
  launch_receipt["pre_submit"]["kernarg_qwords"][0] = \
    session.fixed_five_vas[0]

  attestation = session.attest_post_sync(invocation, "PM4")
  assert attestation.status == "PASS"


@pytest.mark.parametrize("receipt_mode,error_match", (
  ("zero", "produced no PM4 no-doorbell receipt"),
  ("bad", "receipt differs from authority"),
))
def test_snapshot_only_bad_or_zero_receipt_fails_closed(
    receipt_mode, error_match):
  _, fake, _, session = _prepared()
  fake.pm4_submit_policy = "snapshot_only"
  fake.pm4_snapshot_receipt_mode = receipt_mode
  with pytest.raises(ValueError, match=error_match):
    session.invoke(1)
  assert fake.pm4_no_doorbell_receipt_sink == []
  assert session.has_pending_invocation is False
  with pytest.raises(RuntimeError, match="failed closed"):
    session.invoke(1)


def test_snapshot_only_requires_exactly_one_launch_receipt():
  authority = _authority()
  runtime = object()
  buffers = tuple(
    FakeBuffer(slot.name, 0x10000000 + slot.slot * 0x4000000, slot.nbytes)
    for slot in authority.abi)
  runtime_row = {
    "runtime_object_identity": id(runtime),
    "runtime_class": "fake.AMDProgram",
    "runtime_name": authority.function_name,
    "runtime_device": "AMD",
    "entry_va": 0x80000100,
  }

  def no_launch(_runtime, _buffers, _globals, **_kwargs):
    pass

  with pytest.raises(ValueError, match="no launch evidence"):
    _dispatch_production_runtime(
      runtime, buffers, authority, 0, runtime_observation=runtime_row,
      dispatch_with_runtime_evidence=no_launch,
      pm4_submit_policy="snapshot_only",
      pm4_no_doorbell_receipt_sink=[],
      fixed_five_vas=tuple(value.va for value in buffers))

  def two_launches(_runtime, _buffers, _globals, **kwargs):
    kwargs["runtime_evidence"]["launch_count"] = 2
    kwargs["runtime_evidence"]["launches"].extend(({}, {}))

  with pytest.raises(ValueError, match="exactly one launch receipt"):
    _dispatch_production_runtime(
      runtime, buffers, authority, 0, runtime_observation=runtime_row,
      dispatch_with_runtime_evidence=two_launches,
      pm4_submit_policy="snapshot_only",
      pm4_no_doorbell_receipt_sink=[],
      fixed_five_vas=tuple(value.va for value in buffers))


def test_production_dispatch_uses_effective_grid_and_retains_exact_pm4_state():
  authority = _authority()
  runtime = object()
  buffers = tuple(
    FakeBuffer(
      slot.name, 0x10000000 + slot.slot * 0x4000000, slot.nbytes)
    for slot in authority.abi)
  fixed_vas = tuple(value.va for value in buffers)
  runtime_row = {
    "runtime_object_identity": id(runtime),
    "runtime_class": "fake.AMDProgram",
    "runtime_name": authority.function_name,
    "runtime_device": "AMD",
    "entry_va": 0x80000100,
  }
  calls = []

  def retained_dispatch(
      observed_runtime, observed_buffers, observed_globals, **kwargs):
    calls.append((
      observed_runtime, observed_buffers, observed_globals, dict(kwargs)))
    assert kwargs["global_size"] == (40, 4, 1)
    assert kwargs["local_size"] == authority.local_size
    assert kwargs["retain_pm4_pre_submit"] is True
    assert kwargs["pm4_submit_policy"] == "execute"
    kernarg_va = 0x123456789000
    launch = {
      "epoch": 0,
      "arguments": [
        {"slot": slot, "va": value.va, "size": value.nbytes}
        for slot, value in enumerate(buffers)],
      "kernarg": {
        "pointer_words": list(fixed_vas),
        "pointer_words_match_bound": True,
      },
      "pre_submit": {
        "schema": PM4_PRE_SUBMIT_SCHEMA,
        "capture_point": PM4_PRE_SUBMIT_CAPTURE_POINT,
        **runtime_row,
        "kernarg_va": kernarg_va, "kernarg_nbytes": 40,
        "kernarg_qwords": list(fixed_vas),
        "pm4_kernarg_user_data": {
          "packet_dword_offset": 11, "register_index": 0,
          "low_dword": kernarg_va & 0xffffffff,
          "high_dword": kernarg_va >> 32, "pointer": kernarg_va,
        },
        "pm4_dispatch_direct": {
          "packet_dword_offset": 13,
          "group_counts": [40, 4, 1], "dispatch_initiator": 1,
        },
        "pm4_workgroup_size": {
          "packet_dword_offset": 7, "register_index": 0,
          "size": list(authority.local_size),
        },
        "pm4_program_entry": {
          "packet_dword_offset": 3, "register_index": 0,
          "low_dword": runtime_row["entry_va"] >> 8,
          "high_dword": 0, "entry_va": runtime_row["entry_va"],
        },
        "argument_buffers": [
          {"slot": slot, "va": value.va, "size": value.nbytes}
          for slot, value in enumerate(buffers)],
        "pm4_dword_count": 17, "pm4_sha256": "3" * 64,
        "checks": {key: True for key in PM4_PRE_SUBMIT_CHECK_KEYS},
        "all_checks_pass": True,
      },
      "submit_evidence": {
        "native_submit_entered_count": 1,
        "native_submit_returned_count": 1,
        "target_submit_entered": True,
        "target_submit_returned": True,
        "target_dispatch_submitted": True,
      },
    }
    kwargs["runtime_evidence"]["launch_count"] = 1
    kwargs["runtime_evidence"]["launches"].append(launch)

  row = _dispatch_production_runtime(
    runtime, buffers, authority, 0, runtime_observation=runtime_row,
    dispatch_with_runtime_evidence=retained_dispatch,
    pm4_submit_policy="execute", pm4_no_doorbell_receipt_sink=None,
    fixed_five_vas=fixed_vas, effective_global_size=(40, 4, 1))
  assert len(calls) == 1
  assert calls[0][:3] == (runtime, buffers, authority.globals)
  assert row["program_key"] == authority.program_key
  assert row["binary_sha256"] == authority.binary_sha256
  assert row["frozen_global_size"] == list(authority.global_size)
  assert row["effective_global_size"] == [40, 4, 1]
  assert row["argument_vas"] == list(fixed_vas)
  assert row["pre_submit"]["kernarg_qwords"] == list(fixed_vas)


@pytest.mark.parametrize("policy,sink,error,error_match", (
  ("bogus", None, ValueError, "pm4_submit_policy"),
  ("snapshot_only", None, TypeError, "requires a list receipt sink"),
  ("execute", [], ValueError, "does not accept a receipt sink"),
))
def test_production_policy_and_sink_fail_before_device_import(
    policy, sink, error, error_match):
  with pytest.raises(error, match=error_match):
    production_frozen_staged_low_level_dependencies(
      _authority(), pm4_submit_policy=policy,
      pm4_no_doorbell_receipt_sink=sink)


@pytest.mark.parametrize("grid", sorted(DIAGNOSTIC_GLOBAL_SIZE_ALLOWLIST))
def test_allowlisted_diagnostic_grid_is_one_epoch_and_promotion_ineligible(
    grid):
  authority, fake, session = _prepared_diagnostic(grid)
  invocation = session.invoke_diagnostic_one_epoch(grid)
  pending = invocation.pending_observation
  assert pending["schema"] == DIAGNOSTIC_PENDING_OBSERVATION_SCHEMA
  assert pending["promotion_evidence_eligible"] is False
  assert pending["frozen_global_size"] == list(authority.global_size)
  assert pending["effective_global_size"] == list(grid)
  assert pending["local_size"] == list(authority.local_size)
  assert pending["launch_count"] == 1
  assert len(invocation.candidate_phase_trace["epochs"]) == 1
  launch = session._pending.launch_observations[0]
  assert launch["program_key"] == authority.program_key
  assert launch["binary_sha256"] == authority.binary_sha256
  assert launch["frozen_global_size"] == list(authority.global_size)
  assert launch["effective_global_size"] == list(grid)
  assert launch["global_size"] == list(grid)
  assert launch["local_size"] == list(authority.local_size)
  assert launch["argument_vas"] == list(session.fixed_five_vas)
  assert launch["kernarg_pointer_words"] == list(session.fixed_five_vas)
  assert launch["pre_submit"]["kernarg_qwords"] == \
    list(session.fixed_five_vas)
  assert launch["pre_submit"]["pm4_kernarg_user_data"]["pointer"] == \
    launch["pre_submit"]["kernarg_va"]
  assert fake.diagnostic_dispatch_receipt_sink == []

  receipt = session.complete_diagnostic_post_sync(invocation, "PM4")
  assert receipt.schema == DIAGNOSTIC_RECEIPT_SCHEMA
  assert receipt.status == "PASS"
  assert receipt.promotion_evidence_eligible is False
  assert receipt.target_dispatch_submitted is True
  assert receipt.post_dispatch_sync_completed is True
  assert receipt.frozen_global_size == authority.global_size
  assert receipt.effective_global_size == grid
  assert receipt.local_size == authority.local_size
  assert receipt.fixed_five_vas == session.fixed_five_vas
  assert receipt.program_key == authority.program_key
  assert receipt.binary_sha256 == authority.binary_sha256
  assert receipt.runtime_object_identity == id(fake.runtime)
  assert receipt.pre_submit["kernarg_qwords"] == \
    list(session.fixed_five_vas)
  assert receipt.launch_count == 1
  assert fake.events[-1] == "observe_runtime"
  assert len(fake.diagnostic_dispatch_receipt_sink) == 1
  sink_receipt = fake.diagnostic_dispatch_receipt_sink[0]
  assert sink_receipt["schema"] == DIAGNOSTIC_RECEIPT_SCHEMA
  assert sink_receipt["effective_global_size"] == list(grid)
  assert sink_receipt["fixed_five_vas"] == list(session.fixed_five_vas)
  assert sink_receipt["pre_submit"]["pm4_sha256"] == "3" * 64
  assert sink_receipt["observation_identity"] == receipt.observation_identity
  assert session.has_pending_invocation is False


@pytest.mark.parametrize("grid", (
  (136, 4, 1), (137, 1, 1), (1, 1, 2), (0, 1, 1), (1, 1),
))
def test_diagnostic_grid_rejected_before_device_import(grid):
  with pytest.raises(ValueError, match="diagnostic global size"):
    production_frozen_staged_low_level_dependencies(
      _authority(), diagnostic_global_size=grid,
      diagnostic_dispatch_receipt_sink=[])


@pytest.mark.parametrize("kwargs,error,error_match", (
  (
    {"diagnostic_global_size": (1, 1, 1)},
    ValueError, "provided together"),
  (
    {"diagnostic_dispatch_receipt_sink": []},
    ValueError, "provided together"),
  (
    {
      "diagnostic_global_size": (1, 1, 1),
      "diagnostic_dispatch_receipt_sink": [],
      "pm4_submit_policy": "snapshot_only",
      "pm4_no_doorbell_receipt_sink": [],
    },
    ValueError, "requires execute PM4"),
))
def test_diagnostic_dependency_contract_rejected_before_device_import(
    kwargs, error, error_match):
  with pytest.raises(error, match=error_match):
    production_frozen_staged_low_level_dependencies(_authority(), **kwargs)


def test_ordinary_and_diagnostic_terminal_states_are_strictly_separate():
  _, _, _, ordinary = _prepared()
  ordinary_invocation = ordinary.invoke(1)
  with pytest.raises(
      ValueError, match="rejects ordinary invocation"):
    ordinary.complete_diagnostic_post_sync(ordinary_invocation, "PM4")
  assert ordinary.has_pending_invocation
  ordinary.attest_post_sync(ordinary_invocation, "PM4")

  _, _, diagnostic = _prepared_diagnostic((1, 1, 1))
  with pytest.raises(
      RuntimeError, match="rejects diagnostic-configured"):
    diagnostic.invoke(1)
  invocation = diagnostic.invoke_diagnostic_one_epoch((1, 1, 1))
  with pytest.raises(
      ValueError, match="rejects diagnostic invocation"):
    diagnostic.attest_post_sync(invocation, "PM4")
  assert diagnostic.has_pending_invocation
  diagnostic.complete_diagnostic_post_sync(invocation, "PM4")


def test_diagnostic_fault_preserves_effective_grid_and_frozen_authority():
  authority, fake, session = _prepared_diagnostic((40, 4, 1))
  fake.dispatch_failure = True
  with pytest.raises(
      RuntimeError, match="injected target dispatch fault") as caught:
    session.invoke_diagnostic_one_epoch((40, 4, 1))
  failure = getattr(caught.value, INVOCATION_FAILURE_ATTR)
  assert failure["diagnostic"] is True
  assert failure["promotion_evidence_eligible"] is False
  assert failure["frozen_global_size"] == list(authority.global_size)
  assert failure["effective_global_size"] == [40, 4, 1]
  dispatch = failure["dispatch_failure"]
  assert dispatch["frozen_global_size"] == list(authority.global_size)
  assert dispatch["effective_global_size"] == [40, 4, 1]
  assert dispatch["launch"]["global_size"] == [40, 4, 1]
  assert dispatch["checks"]["global_size_exact"] is True
  assert dispatch["submit_evidence"] == {
    "native_submit_entered_count": 1,
    "native_submit_returned_count": 1,
    "target_submit_entered": True,
    "target_submit_returned": True,
    "target_dispatch_submitted": True,
  }
  assert dispatch["all_authority_checks_pass"] is True


def test_diagnostic_invocation_rejects_grid_different_from_dependencies():
  _, _, session = _prepared_diagnostic((1, 1, 1))
  with pytest.raises(ValueError, match="differs from configured"):
    session.invoke_diagnostic_one_epoch((2, 1, 1))
  assert session.has_pending_invocation is False


def test_diagnostic_aql_route_is_rejected_before_target_dispatch():
  _, fake, session = _prepared_diagnostic((1, 1, 1))
  fake.runtime_row["queue_mode"] = "AQL"
  session.prepared_runtime_observation["queue_mode"] = "AQL"
  with pytest.raises(ValueError, match="requires PM4"):
    session.invoke_diagnostic_one_epoch((1, 1, 1))
  assert not any(event.startswith("dispatch:") for event in fake.events)
  assert session.has_pending_invocation is False


@pytest.mark.parametrize("mutation", (
  "qword", "runtime", "user_data", "command_hash",
  "dispatch_grid", "local_size", "program_entry",
))
def test_diagnostic_completion_rejects_pre_submit_authority_drift(mutation):
  _, _, session = _prepared_diagnostic((1, 1, 1))
  invocation = session.invoke_diagnostic_one_epoch((1, 1, 1))
  snapshot = session._pending.launch_observations[0]["pre_submit"]
  if mutation == "qword":
    snapshot["kernarg_qwords"][0] += 8
  elif mutation == "runtime":
    snapshot["runtime_object_identity"] += 1
  elif mutation == "user_data":
    snapshot["pm4_kernarg_user_data"]["pointer"] += 8
  elif mutation == "dispatch_grid":
    snapshot["pm4_dispatch_direct"]["group_counts"][0] += 1
  elif mutation == "local_size":
    snapshot["pm4_workgroup_size"]["size"][0] //= 2
  elif mutation == "program_entry":
    snapshot["pm4_program_entry"]["entry_va"] += 0x100
  else:
    snapshot["pm4_sha256"] = None
  with pytest.raises(ValueError, match="pre-submit differs"):
    session.complete_diagnostic_post_sync(invocation, "PM4")
  assert session.has_pending_invocation is False
