from __future__ import annotations

import gc
import hashlib
from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest

from extra.qk.mmq_frozen_staged_low_level_session import (
  ABI_NAMES, ATTESTATION_SCHEMA, CANDIDATE_TRACE_SCHEMA,
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

  def dispatch(self, runtime, buffers, authority, epoch):
    assert runtime is self.runtime
    self.events.append(f"dispatch:{epoch}")
    row = {
      "epoch": epoch, "program_key": authority.program_key,
      "binary_sha256": authority.binary_sha256,
      "global_size": list(authority.global_size),
      "local_size": list(authority.local_size),
      "argument_vas": [value.va for value in buffers],
      "kernarg_pointer_words": [value.va for value in buffers],
      "kernarg_pointer_words_match_bound": True,
    }
    if self.launch_drift is not None:
      row[self.launch_drift] = \
        _raw("wrong") if self.launch_drift == "binary_sha256" else \
        False if self.launch_drift == "kernarg_pointer_words_match_bound" \
        else [1, 2, 3]
    if self.dispatch_failure:
      launch = {
        "epoch": epoch, "global_size": list(authority.global_size),
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
        "queue_device_matches_submit_device": True,
        "runtime_device_matches_submit_device": True,
        "args_state_program_matches_runtime": True,
        "exact_five_argument_buffers": True,
        "exact_five_kernarg_qwords": True,
        "five_qwords_match_constructed_buffers": True,
        "pm4_command_words_concrete": True,
        "pm4_command_stream_nonempty": True,
        "pm4_packet_stream_decoded": True,
        "pm4_kernarg_user_data_found_once": True,
        "pm4_kernarg_uses_user_data_0": True,
        "pm4_kernarg_user_data_matches_kernarg_va": True,
      }
      failure.mmq_runtime_dispatch_failure = {
        "schema": "tinygrad.mmq_q4k_q8_1.runtime_dispatch_failure.v1",
        "failure_boundary":
          "runtime_call_raised_after_kernarg_capture_before_return",
        "wait": True, "launch": launch,
        "authoritative_qword_snapshot": "pre_submit",
        "pre_submit": {
          "schema":
            "tinygrad.mmq_q4k_q8_1.pm4_pre_submit_snapshot.v1",
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
          "argument_buffers": [
            {"slot": slot, "va": value.va, "size": value.nbytes}
            for slot, value in enumerate(buffers)],
          "pm4_dword_count": 17, "pm4_sha256": "3" * 64,
          "checks": pre_submit_checks, "all_checks_pass": True,
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
      dispatch=self.dispatch, clock_ns=self.clock_ns)


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
