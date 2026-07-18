import hashlib
import json
import numpy as np
import pytest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_exact_role_spec import exact_role_spec
from extra.qk.mmq_llama_five_buffer_full_kernel import build_llama_five_buffer_full_kernel
from extra.qk.mmq_llama_five_buffer_gpu_harness import (FrozenRuntimePreconstructionError,
  TARGET_IN_PLACE_ACCUMULATION, _accumulate_target_role_epoch,
  _aql_packet_census_from_exception,
  _pm4_dispatch_census_from_exception,
  _aql_target_program_identity, _audit_target_aql_kernargs,
  _bind_sink, _dispatch_with_runtime_evidence, _numeric_comparison, _pack_q4_epochs_contiguous,
  _decode_aql_kernel_dispatch_packet, _load_frozen_execution_binding, _random_q4_words, _runtime_identity_evidence,
  _dispatch_error_runtime_reuse_evidence,
  _fixed_base_ordinal_reference_operands, _fixed_base_ordinal_sequence_reference_operands,
  _fixed_base_prefix_reference_operands,
  _frozen_program_set_ordinal_sequence_target_identities, _frozen_program_set_ordinal_target_identity,
  _frozen_program_set_target_identities,
  _producer_oracle_diagnostic, _producer_probe_status,
  _crosscheck_preconstructed_dispatch_runtimes,
  _preconstruct_frozen_program_runtimes,
  _realize_outputs_together, _realize_with_aql_packet_census, _realize_with_pm4_dispatch_census,
  _retained_producer_tensors,
  _scheduler_prefix_two_launches,
  _validate_v2_fixed_base_ordinal, _validate_v2_fixed_base_ordinal_sequence,
  _validate_v2_fixed_base_prefix_epochs,
  _validate_frozen_execution_fixture, _validate_frozen_fixture, _validated_child_env_overrides,
  _zero_persistent_target_output,
  main, run_amd_validation, run_frozen_scheduler_prefix_two_probe_isolated,
  run_frozen_epoch_program_set_ordinal_probe_isolated,
  run_frozen_epoch_program_set_ordinal_sequence_probe_isolated,
  run_frozen_epoch_program_set_prefix_probe_isolated,
  run_frozen_scheduler_producer_prefix_probe_isolated,
  run_full_grid_target_role_probe, run_full_grid_target_role_probe_isolated)


def test_gpu_harness_random_q4_fixture_has_independent_abi_shape():
  words = _random_q4_words(128, 256, 20260717)
  assert words.dtype == np.uint32 and words.shape == (128 * 36,)
  assert np.isfinite(words.view(np.uint8)).all()


def test_gpu_harness_preloaded_q4_pack_makes_each_epoch_a_contiguous_n_slice():
  blocks = np.arange(4 * 3 * 144, dtype=np.uint8).reshape(4, 3, 144)
  packed = _pack_q4_epochs_contiguous(blocks).view(np.uint8)
  epoch_bytes = blocks.shape[0] * blocks.shape[2]
  for epoch in range(blocks.shape[1]):
    expected = np.ascontiguousarray(blocks[:, epoch, :]).reshape(-1)
    assert np.array_equal(packed[epoch*epoch_bytes:(epoch+1)*epoch_bytes], expected)
  # The original N-major flattening cannot be consumed by one base-offset view.
  assert not np.array_equal(blocks.reshape(-1)[:epoch_bytes], np.ascontiguousarray(blocks[:, 0, :]).reshape(-1))


def test_gpu_harness_preloaded_q4_pack_rejects_layout_drift():
  for bad in (np.zeros((2, 3, 143), dtype=np.uint8), np.zeros((2, 3, 144), dtype=np.uint32),
              np.zeros((2, 144), dtype=np.uint8)):
    try: _pack_q4_epochs_contiguous(bad)
    except ValueError: pass
    else: raise AssertionError("invalid Q4 preload layout must fail closed")


def test_gpu_harness_binds_exact_five_buffer_slots_without_reauthoring_graph():
  sink = build_llama_five_buffer_full_kernel(128, 128, 256).sink
  args = tuple(UOp.placeholder((size,), dtype, slot) for slot, (size, dtype) in enumerate(
    ((128 * 128, dtypes.float32), (128 * 36, dtypes.uint32),
     (2 * 128 * 128, dtypes.int8), (2 * 128 * 4, dtypes.float32),
     (2 * 128 * 4, dtypes.float32))))
  bound = _bind_sink(sink, args)
  params = {u.arg.slot for u in bound.toposort() if u.op is Ops.PARAM}
  assert params == set(range(5))
  assert bound.arg.name == "mmq_llama_five_buffer_full_grid"


def test_gpu_harness_timeout_path_fails_closed_without_gpu_access():
  row = run_amd_validation(timeout_seconds=0)
  assert row["passed"] is False
  assert row["verdict"] == "MMQ_LLAMA_FIVE_BUFFER_GPU_BLOCKED"
  assert row["blocker"] == "timeout_seconds must be positive"


def test_target_role_stable_metadata_staging_requires_preloaded_sources():
  # This guard executes before runtime construction and keeps the fixed-VA
  # SDMA path from silently falling back to per-launch host allocations.
  with pytest.raises(ValueError, match="requires preloaded_epochs"):
    run_full_grid_target_role_probe(stable_metadata_staging=True, preloaded_epochs=False)


def test_target_role_stable_epoch_staging_requires_stable_metadata_before_gpu():
  with pytest.raises(ValueError, match="requires stable_metadata_staging"):
    run_full_grid_target_role_probe(stable_epoch_staging=True, preloaded_epochs=True)

def test_target_role_async_epochs_require_safe_fixed_va_contract_before_gpu():
  with pytest.raises(ValueError, match="asynchronous epoch dispatch requires"):
    run_full_grid_target_role_probe(wait_each_dispatch=False)


def test_scheduler_prefix_two_address_modes_are_exact_and_producer_free():
  epoch0, epoch1 = tuple(object() for _ in range(4)), tuple(object() for _ in range(4))
  same = _scheduler_prefix_two_launches("same", (epoch0, epoch1))
  changed = _scheduler_prefix_two_launches("changed", (epoch0, epoch1))
  assert same == (epoch0, epoch0) and same[0] is same[1]
  assert changed == (epoch0, epoch1) and changed[0] is not changed[1]
  assert all(left is not right for left, right in zip(*changed))
  with pytest.raises(ValueError, match="must be 'same' or 'changed'"):
    _scheduler_prefix_two_launches("mixed", (epoch0, epoch1))
  with pytest.raises(ValueError, match="distinct input tensors"):
    _scheduler_prefix_two_launches("changed", (epoch0, epoch0))


@pytest.mark.parametrize("change_slot,changed_index", [
  ("q4", 0), ("q8_values", 1), ("q8_scales", 2), ("q8_sums", 3)])
def test_scheduler_prefix_two_changes_exactly_one_selected_input_slot(change_slot, changed_index):
  epoch0, epoch1 = tuple(object() for _ in range(4)), tuple(object() for _ in range(4))
  first, second = _scheduler_prefix_two_launches("changed", (epoch0, epoch1), change_slot)
  assert first == epoch0
  assert [left is right for left, right in zip(first, second)] == [
    index != changed_index for index in range(4)]
  assert second[changed_index] is epoch1[changed_index]


def test_scheduler_prefix_two_slot_selector_fails_closed():
  epoch0, epoch1 = tuple(object() for _ in range(4)), tuple(object() for _ in range(4))
  with pytest.raises(ValueError, match="change_slot must be one of"):
    _scheduler_prefix_two_launches("changed", (epoch0, epoch1), "output")
  with pytest.raises(ValueError, match="same mode does not accept"):
    _scheduler_prefix_two_launches("same", (epoch0, epoch1), "q4")


def test_scheduler_prefix_two_can_hold_only_scale_va_fixed():
  epoch0, epoch1 = tuple(object() for _ in range(4)), tuple(object() for _ in range(4))
  first, second = _scheduler_prefix_two_launches(
    "changed", (epoch0, epoch1), "all_except_q8_scales")
  assert first == epoch0
  assert [left is right for left, right in zip(first, second)] == [False, False, True, False]


def test_scheduler_prefix_two_aql_packet_decoder_reports_exact_dispatch_safety_fields():
  import ctypes
  from tinygrad.runtime.autogen import hsa
  from tinygrad.runtime.ops_amd import AQL_HDR
  packet = hsa.hsa_kernel_dispatch_packet_t(
    header=AQL_HDR | (hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE),
    kernel_object=0x1234, kernarg_address=0x5678)
  row = _decode_aql_kernel_dispatch_packet(bytes(packet))
  assert row["kernel_dispatch"] is True and row["barrier"] is True
  assert row["acquire_fence_scope"] == row["release_fence_scope"] == hsa.HSA_FENCE_SCOPE_SYSTEM
  assert row["kernel_object"] == 0x1234 and row["kernarg_address"] == 0x5678
  invalid = bytearray(bytes(packet))
  invalid[:2] = int(hsa.HSA_PACKET_TYPE_INVALID << hsa.HSA_PACKET_HEADER_TYPE).to_bytes(2, "little")
  assert _decode_aql_kernel_dispatch_packet(bytes(invalid)) == {
    "packet_type": hsa.HSA_PACKET_TYPE_INVALID, "kernel_dispatch": False}
  with pytest.raises(ValueError, match="exactly 64 bytes"):
    _decode_aql_kernel_dispatch_packet(bytes(ctypes.sizeof(packet) - 1))


def test_aql_target_census_identity_and_five_qword_scale_contract_are_cpu_testable():
  program = SimpleNamespace(name="target", lib=b"exact frozen binary")
  identity = _aql_target_program_identity(program)
  assert identity["function_name"] == "target" and len(identity["binary_sha256"]) == 64
  first = [0x1000, 0x2000, 0x3000, 0x4000, 0x5000]
  second = [0x1000, 0x2100, 0x3100, 0x4000, 0x5100]
  checks = _audit_target_aql_kernargs(
    second, [first], expected_vas=None, require_fixed_scale_va=True)
  assert checks == {
    "five_qwords_nonzero": True, "five_qwords_match_expected_vas": True,
    "output_va_fixed": True, "q8_scale_va_fixed": True, "all_five_vas_fixed": True,
    "all_five_vas_distinct": True}
  zero = _audit_target_aql_kernargs(
    [0x1000, 0, 0x3100, 0x4000, 0x5100], [first],
    expected_vas=None, require_fixed_scale_va=True)
  assert zero["five_qwords_nonzero"] is False
  moved_scale = _audit_target_aql_kernargs(
    [0x1000, 0x2100, 0x3100, 0x4100, 0x5100], [first],
    expected_vas=None, require_fixed_scale_va=True)
  assert moved_scale["q8_scale_va_fixed"] is False
  moved_input = _audit_target_aql_kernargs(
    [0x1000, 0x2100, 0x3000, 0x4000, 0x5000], [first],
    expected_vas=None, require_fixed_scale_va=False, require_all_five_vas_fixed=True)
  assert moved_input["all_five_vas_fixed"] is False
  for prior_call_count in (2, 19, 67):
    full_role_fixed = _audit_target_aql_kernargs(
      first.copy(), [first.copy() for _ in range(prior_call_count)],
      expected_vas=None, require_fixed_scale_va=False, require_all_five_vas_fixed=True)
    assert full_role_fixed["five_qwords_nonzero"] is True
    assert full_role_fixed["all_five_vas_fixed"] is True
  aliased = _audit_target_aql_kernargs(
    [0x1000, 0x2000, 0x3000, 0x4000, 0x4000], [],
    expected_vas=None, require_fixed_scale_va=False,
    require_all_five_vas_distinct=True)
  assert aliased["all_five_vas_distinct"] is False
  with pytest.raises(ValueError, match="exactly five"):
    _audit_target_aql_kernargs([1, 2], [], expected_vas=None, require_fixed_scale_va=True)


def test_aql_packet_census_retains_accepted_packet_and_kernargs_when_doorbell_faults(monkeypatch):
  import ctypes
  from tinygrad.device import Device
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.autogen import hsa

  descriptor_va, kernarg_va = 0x810000, 0x910000
  argument_vas = [0x100000 + slot*0x20000 for slot in range(5)]
  argument_sizes = [0x10000 + slot*0x1000 for slot in range(5)]
  desc, queue = object(), object()

  class FakeDevice:
    is_aql = True
    def compute_queue_desc(self, index):
      assert index == 0
      return desc
  dev = FakeDevice()
  monkeypatch.setattr(type(Device), "__getitem__", lambda self, key: dev)

  class View:
    def __init__(self, values): self.values = values
    def view(self, **kwargs): return self.values
  class Kernarg:
    va_addr, size = kernarg_va, 40
    def cpu_view(self): return View(argument_vas)
  args_state = SimpleNamespace(
    buf=Kernarg(),
    bufs=tuple(SimpleNamespace(va_addr=va, size=size)
               for va, size in zip(argument_vas, argument_sizes)))
  program = SimpleNamespace(
    name="target", lib=b"target binary",
    lib_gpu=SimpleNamespace(va_addr=0x800000, size=0x20000),
    prog_addr=0x804000, aql_prog_addr=descriptor_va,
    kernargs_segment_size=40)
  identity = _aql_target_program_identity(program)
  program_key = "ab" * 32
  from tinygrad.engine.realize import runtime_cache
  monkeypatch.setitem(runtime_cache, (bytes.fromhex(program_key), "AMD"), program)

  class Slot:
    def __init__(self): self.data = bytearray(64)
    def view(self, **kwargs): return memoryview(self.data)
  slot = Slot()
  packet = bytes(hsa.hsa_kernel_dispatch_packet_t(
    header=ops_amd.AQL_HDR | (hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE),
    kernel_object=descriptor_va, kernarg_address=ctypes.c_void_p(kernarg_va)))

  monkeypatch.setattr(ops_amd.AMDComputeAQLQueue, "exec",
                      lambda self, prg, state, global_size, local_size: self)
  monkeypatch.setattr(ops_amd, "_publish_aql_packet",
                      lambda target, payload: target.data.__setitem__(slice(None), payload))
  def faulting_doorbell(self, doorbell_dev, doorbell_value=None):
    raise RuntimeError("simulated post-audit MMU fault")
  monkeypatch.setattr(ops_amd.AMDQueueDesc, "signal_doorbell", faulting_doorbell)

  class Output:
    def realize(self, *retained):
      ops_amd.AMDComputeAQLQueue.exec(queue, program, args_state, (1, 1, 1), (1, 1, 1))
      ops_amd._publish_aql_packet(slot, packet)
      ops_amd.AMDQueueDesc.signal_doorbell(desc, dev)

  with pytest.raises(RuntimeError, match="simulated post-audit MMU fault") as raised:
    _realize_with_aql_packet_census(
      Output(), target_program_identities=(identity,),
      target_program_keys=(program_key,), require_all_five_vas_fixed=True,
      require_runtime_lifecycle=True)
  census = _aql_packet_census_from_exception(raised.value)
  assert census is not None
  assert census["status"] == "REALIZATION_ERROR"
  assert census["realization_exception"] == "RuntimeError"
  assert census["realization_error"] == "simulated post-audit MMU fault"
  assert census["compute_doorbell_count"] == 1
  assert census["accepted_target_call_count"] == census["call_count"] == 1
  assert census["pending_constructed_dispatch_count"] == 0
  assert census["pending_published_packet_count"] == 0
  call = census["calls"][0]
  assert call["accepted_before_doorbell"] is True and call["all_checks_pass"] is True
  assert call["program_identity"] == call["expected_program_identity"] == identity
  assert call["kernel_object"] == descriptor_va and call["kernarg_address"] == kernarg_va
  assert call["kernarg_qwords"] == argument_vas
  lifecycle = call["runtime_lifecycle"]
  assert lifecycle["program_library_va"] == 0x800000
  assert lifecycle["program_library_nbytes"] == 0x20000
  assert lifecycle["program_entry_va"] == 0x804000
  assert lifecycle["program_entry_offset"] == 0x4000
  assert lifecycle["program_descriptor_va"] == descriptor_va
  assert lifecycle["program_descriptor_offset"] == descriptor_va - 0x800000
  assert lifecycle["runtime_cache_bindings"] == [{"program_key": program_key, "device": "AMD"}]
  assert lifecycle["kernarg_va"] == kernarg_va
  assert lifecycle["kernarg_payload_nbytes"] == lifecycle["kernarg_allocation_nbytes"] == 40
  assert lifecycle["all_checks_pass"] is True
  assert all(lifecycle["checks"].values())
  assert call["argument_buffers"] == [
    {"slot": slot, "va": va, "size": size}
    for slot, (va, size) in enumerate(zip(argument_vas, argument_sizes))]
  assert call["checks"]["five_qwords_match_constructed_buffers"] is True
  assert call["checks"]["five_constructed_buffer_vas_distinct"] is True
  assert census["all_accepted_target_calls_pass"] is True


def test_pm4_dispatch_census_retains_accepted_submit_and_kernargs_when_doorbell_faults(monkeypatch):
  from tinygrad.device import Device
  from tinygrad.runtime import ops_amd

  kernarg_va = 0x910000
  argument_vas = [0x100000 + slot*0x20000 for slot in range(5)]
  argument_sizes = [0x10000 + slot*0x1000 for slot in range(5)]

  class FakeDevice:
    is_aql = False
  dev = FakeDevice()
  monkeypatch.setattr(type(Device), "__getitem__", lambda self, key: dev)

  class View:
    def __init__(self, values): self.values = values
    def view(self, **kwargs): return self.values
  class Kernarg:
    va_addr, size = kernarg_va, 40
    def cpu_view(self): return View(argument_vas)
  args_state = SimpleNamespace(
    buf=Kernarg(),
    bufs=tuple(SimpleNamespace(va_addr=va, size=size)
               for va, size in zip(argument_vas, argument_sizes)))
  program = SimpleNamespace(
    name="target", lib=b"target binary",
    lib_gpu=SimpleNamespace(va_addr=0x800000, size=0x20000),
    prog_addr=0x804000, aql_prog_addr=0x810000,
    kernargs_segment_size=40)
  identity = _aql_target_program_identity(program)
  program_key = "cd" * 32
  from tinygrad.engine.realize import runtime_cache
  monkeypatch.setitem(runtime_cache, (bytes.fromhex(program_key), "AMD"), program)

  queue = object.__new__(ops_amd.AMDComputeQueue)
  queue.dev, queue.binded_device, queue._q = dev, None, [0xc0001000, 0xc0001001]

  def fake_exec(self, prg, state, global_size, local_size):
    self._q.extend([0xc0002000, 0xc0002001])
    return self
  def faulting_submit(self, submit_dev):
    raise RuntimeError("simulated PM4 post-audit MMU fault")
  monkeypatch.setattr(ops_amd.AMDComputeQueue, "exec", fake_exec)
  monkeypatch.setattr(ops_amd.AMDComputeQueue, "_submit", faulting_submit)

  class Output:
    def realize(self, *retained):
      ops_amd.AMDComputeQueue.exec(
        queue, program, args_state, (8, 4, 1), (256, 1, 1))
      queue._q.append(0xc0003000)
      ops_amd.AMDComputeQueue._submit(queue, dev)

  with pytest.raises(RuntimeError, match="simulated PM4 post-audit MMU fault") as raised:
    _realize_with_pm4_dispatch_census(
      Output(), target_program_identities=(identity,),
      target_program_keys=(program_key,),
      target_launch_dims=(((8, 4, 1), (256, 1, 1)),),
      require_all_five_vas_fixed=True, require_all_five_vas_distinct=True)
  census = _pm4_dispatch_census_from_exception(raised.value)
  assert census is not None and census["status"] == "REALIZATION_ERROR"
  assert census["queue_mode"] == "PM4"
  assert census["accepted_target_call_count"] == census["call_count"] == 1
  assert census["pending_target_queue_count"] == 0
  call = census["calls"][0]
  assert call["accepted_before_doorbell"] is True and call["all_checks_pass"] is True
  assert call["program_identity"] == call["expected_program_identity"] == identity
  assert call["kernarg_qwords"] == argument_vas
  assert call["global_size"] == call["expected_global_size"] == [8, 4, 1]
  assert call["local_size"] == call["expected_local_size"] == [256, 1, 1]
  assert call["pm4_dword_count"] == 5 and len(call["pm4_sha256"]) == 64
  assert call["before_exec_dword_count"] == 2 and call["after_exec_dword_count"] == 4
  assert all(call["checks"].values())
  assert census["all_accepted_target_calls_pass"] is True

  # A same-name runtime with the wrong binary identity must be stopped before
  # the native _submit can copy commands into the ring or ring the doorbell.
  queue._q = [0xc0001000, 0xc0001001]
  submitted = []
  def must_not_submit(self, submit_dev):
    submitted.append(True)
    return self
  monkeypatch.setattr(ops_amd.AMDComputeQueue, "exec", fake_exec)
  monkeypatch.setattr(ops_amd.AMDComputeQueue, "_submit", must_not_submit)
  wrong_identity = {**identity, "binary_sha256": "0" * 64}
  with pytest.raises(RuntimeError, match="rejected submit before doorbell") as rejected:
    _realize_with_pm4_dispatch_census(
      Output(), target_program_identities=(wrong_identity,),
      target_program_keys=(program_key,),
      target_launch_dims=(((8, 4, 1), (256, 1, 1)),),
      require_all_five_vas_fixed=True, require_all_five_vas_distinct=True)
  rejected_census = _pm4_dispatch_census_from_exception(rejected.value)
  assert submitted == []
  assert rejected_census is not None
  assert rejected_census["accepted_target_call_count"] == 0
  assert rejected_census["calls"][0]["checks"]["ordered_program_identity_matches"] is False


def test_scheduler_producer_diagnostic_reports_qvalues_metadata_and_target_half_rounding():
  values = np.array([[[1, -2, 3]]], dtype=np.int8)
  scales = np.array([[[0.125]]], dtype=np.float32)
  sums = np.array([[[1.5]]], dtype=np.float32)
  exact = _producer_oracle_diagnostic(values, scales, sums, values.copy(), scales.copy(), sums.copy())
  assert exact["status"] == "PASS" and exact["qvalue_mismatch_count"] == 0
  assert exact["max_scale_abs_error"] == exact["max_sum_abs_error"] == 0.0

  actual_values = values.copy(); actual_values[0, 0, 1] = -1
  actual_scales = scales + np.float32(1e-6)
  actual_sums = sums + np.float32(1e-5)
  drift = _producer_oracle_diagnostic(
    actual_values, actual_scales, actual_sums, values, scales, sums)
  assert drift["status"] == "PRODUCER_ORACLE_ROUNDING_DRIFT"
  assert drift["qvalue_mismatch_count"] == 1
  assert drift["max_scale_abs_error"] > 0 and drift["max_sum_abs_error"] > 0
  assert drift["target_half_scale_mismatch_count"] == 0
  assert drift["target_half_sum_mismatch_count"] == 0


def test_scheduler_producer_probe_status_keeps_consumer_mismatch_distinct_from_rounding_drift():
  assert _producer_probe_status("pass", "PASS") == ("PASS", None)
  assert _producer_probe_status("pass", "PRODUCER_ORACLE_ROUNDING_DRIFT") == (
    "PRODUCER_ORACLE_ROUNDING_DRIFT", None)
  status, blocker = _producer_probe_status("mismatch", "PASS")
  assert status == "CONSUMER_MISMATCH" and "actual producer bytes" in blocker


def test_scheduler_producer_diagnostic_tensors_are_companion_outputs_of_one_realize():
  realized = []
  class Output:
    def realize(self, *companions): realized.append(companions)
  tiles = [
    SimpleNamespace(values=object(), scales=object(), sums=object()),
    SimpleNamespace(values=object(), scales=object(), sums=object()),
  ]
  retained = _retained_producer_tensors(tiles)
  assert retained == (
    tiles[0].values, tiles[0].scales, tiles[0].sums,
    tiles[1].values, tiles[1].scales, tiles[1].sums)
  _realize_outputs_together(Output(), retained)
  assert realized == [retained]
  reused = object()
  with pytest.raises(RuntimeError, match="distinct retained tensors"):
    _retained_producer_tensors([SimpleNamespace(values=reused, scales=reused, sums=object())])


def test_companion_realize_keeps_intermediate_allocations_live_for_post_readback():
  source = Tensor(list(range(8)), device="CPU")
  first = (source + 1).contiguous()
  second = (first * 3).contiguous()
  output = second.sum()
  _realize_outputs_together(output, (first, second))
  assert first.uop.has_buffer_identity() and second.uop.has_buffer_identity()
  assert first.uop.buffer is not second.uop.buffer
  np.testing.assert_array_equal(first.numpy(), np.arange(1, 9))
  np.testing.assert_array_equal(second.numpy(), np.arange(1, 9) * 3)


def test_scheduler_prefix_two_isolated_wrapper_reuses_health_guard_and_narrow_aql(tmp_path):
  class _Proc:
    returncode = 0
    stdout = '{"schema":"tinygrad.mmq_frozen_scheduler_prefix_two_probe.v1","status":"PASS"}\n'
    stderr = ""
  bundle = tmp_path / "frozen"
  with patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run", return_value=_Proc()) as run, \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value=""), \
       patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=True):
    result = run_frozen_scheduler_prefix_two_probe_isolated(
      role_spec=exact_role_spec("attn_kv"), frozen_bundle=bundle, address_mode="changed",
      change_slot="q8_scales", timeout_seconds=1, child_env_overrides={"AMD_AQL": "1"})
  assert result["status"] == "PASS" and result["health_before"] is result["health_after"] is True
  assert result["child_env_overrides"] == {"AMD_AQL": "1"}
  assert run.call_args.kwargs["env"]["AMD_AQL"] == "1"
  code = run.call_args.args[0][2]
  assert "run_frozen_scheduler_prefix_two_probe" in code
  assert "address_mode='changed'" in code and "exact_role_spec('attn_kv'" in code
  assert "change_slot='q8_scales'" in code


def test_scheduler_producer_prefix_isolated_reuses_health_guard_and_exact_epoch_limit(tmp_path):
  class _Proc:
    returncode = 0
    stdout = '{"schema":"tinygrad.mmq_frozen_scheduler_producer_prefix_probe.v1","status":"PASS"}\n'
    stderr = ""
  bundle = tmp_path / "frozen"
  with patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run", return_value=_Proc()) as run, \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value=""), \
       patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=True):
    result = run_frozen_scheduler_producer_prefix_probe_isolated(
      role_spec=exact_role_spec("attn_kv"), frozen_bundle=bundle, epoch_limit=2,
      timeout_seconds=1, child_env_overrides={"AMD_AQL": "1"})
  assert result["status"] == "PASS" and result["health_before"] is result["health_after"] is True
  assert result["child_env_overrides"] == {"AMD_AQL": "1"}
  assert run.call_args.kwargs["env"]["AMD_AQL"] == "1"
  code = run.call_args.args[0][2]
  assert "run_frozen_scheduler_producer_prefix_probe" in code
  assert "epoch_limit=2" in code and "exact_role_spec('attn_kv'" in code


def test_scheduler_producer_prefix_rejects_bad_limit_before_health_or_gpu(tmp_path):
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe") as health, \
       patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run") as run:
    result = run_frozen_scheduler_producer_prefix_probe_isolated(
      frozen_bundle=tmp_path / "frozen", epoch_limit=3)
  health.assert_not_called()
  run.assert_not_called()
  assert result["status"] == "BLOCKED" and "must be 1 or 2" in result["exact_blocker"]


@pytest.mark.parametrize("prefix_epochs", [3, 4])
def test_v2_fixed_base_prefix_reference_slices_static_offsets_from_full_buffers(prefix_epochs):
  q4 = np.arange(3 * 4 * 144, dtype=np.uint8).reshape(3, 4, 144)
  values = np.arange(8 * 2 * 128, dtype=np.int16).astype(np.int8).reshape(8, 2, 128)
  scales = np.arange(8 * 2 * 4, dtype=np.float32).reshape(8, 2, 4)
  sums = scales + 1000
  q4_prefix, values_prefix, scales_prefix, sums_prefix = \
    _fixed_base_prefix_reference_operands(q4, values, scales, sums, prefix_epochs)
  records = prefix_epochs * 2
  np.testing.assert_array_equal(q4_prefix, np.ascontiguousarray(q4[:, :prefix_epochs, :]).reshape(-1))
  np.testing.assert_array_equal(values_prefix, values[:records])
  np.testing.assert_array_equal(scales_prefix, scales[:records])
  np.testing.assert_array_equal(sums_prefix, sums[:records])
  assert all(value.flags.c_contiguous for value in (q4_prefix, values_prefix, scales_prefix, sums_prefix))


def test_v2_fixed_base_ordinal_reference_slices_one_exact_static_offset():
  q4 = np.arange(3 * 4 * 144, dtype=np.uint8).reshape(3, 4, 144)
  values = np.arange(8 * 2 * 128, dtype=np.int16).astype(np.int8).reshape(8, 2, 128)
  scales = np.arange(8 * 2 * 4, dtype=np.float32).reshape(8, 2, 4)
  sums = scales + 1000
  q4_epoch, values_epoch, scales_epoch, sums_epoch = \
    _fixed_base_ordinal_reference_operands(q4, values, scales, sums, 2)
  np.testing.assert_array_equal(q4_epoch, np.ascontiguousarray(q4[:, 2:3, :]).reshape(-1))
  np.testing.assert_array_equal(values_epoch, values[4:6])
  np.testing.assert_array_equal(scales_epoch, scales[4:6])
  np.testing.assert_array_equal(sums_epoch, sums[4:6])
  assert all(value.flags.c_contiguous for value in (q4_epoch, values_epoch, scales_epoch, sums_epoch))
  for invalid in (-1, 4, True):
    with pytest.raises(ValueError, match="outside the Q4 epoch extent"):
      _fixed_base_ordinal_reference_operands(q4, values, scales, sums, invalid)


def test_v2_fixed_base_ordinal_sequence_reference_concatenates_exact_selected_epochs():
  q4 = np.arange(3 * 5 * 144, dtype=np.uint8).reshape(3, 5, 144)
  values = np.arange(10 * 2 * 128, dtype=np.int16).astype(np.int8).reshape(10, 2, 128)
  scales = np.arange(10 * 2 * 4, dtype=np.float32).reshape(10, 2, 4)
  sums = scales + 1000
  q4_selected, values_selected, scales_selected, sums_selected = \
    _fixed_base_ordinal_sequence_reference_operands(q4, values, scales, sums, (1, 3))
  np.testing.assert_array_equal(
    q4_selected, np.concatenate([q4[:, 1:2, :], q4[:, 3:4, :]], axis=1).reshape(-1))
  np.testing.assert_array_equal(values_selected, values[[2, 3, 6, 7]])
  np.testing.assert_array_equal(scales_selected, scales[[2, 3, 6, 7]])
  np.testing.assert_array_equal(sums_selected, sums[[2, 3, 6, 7]])
  assert all(value.flags.c_contiguous
             for value in (q4_selected, values_selected, scales_selected, sums_selected))
  for invalid in ((1, 1), (3, 1), (-1, 1), (1, 5)):
    with pytest.raises(ValueError, match="strictly increasing ordinals"):
      _fixed_base_ordinal_sequence_reference_operands(q4, values, scales, sums, invalid)


@pytest.mark.parametrize("prefix_epochs", [3, 20, 68])
def test_v2_fixed_base_target_identities_are_binary_exact_ordered_and_distinct(prefix_epochs):
  programs = tuple(SimpleNamespace(arg=SimpleNamespace(function_name="target")) for _ in range(prefix_epochs))
  binaries = tuple(f"epoch-{epoch}".encode() for epoch in range(prefix_epochs))
  binding = SimpleNamespace(artifact=SimpleNamespace(
    programs=programs, binaries=binaries))
  identities = _frozen_program_set_target_identities(binding, prefix_epochs)
  assert [row["function_name"] for row in identities] == ["target"] * prefix_epochs
  assert [row["binary_sha256"] for row in identities] == [
    hashlib.sha256(binary).hexdigest() for binary in binaries]
  assert len({row["binary_sha256"] for row in identities[:3]}) == 3
  assert len({row["binary_sha256"] for row in identities}) == prefix_epochs
  duplicate = SimpleNamespace(artifact=SimpleNamespace(
    programs=programs, binaries=binaries[:-1] + (binaries[-2],)))
  with pytest.raises(ValueError, match="not distinct"):
    _frozen_program_set_target_identities(duplicate, prefix_epochs)


def _fake_runtime_preconstruction_family(count=3, corrupt=None):
  from tinygrad.engine.realize import runtime_cache
  programs = tuple(SimpleNamespace(key=bytes([epoch+1])*32) for epoch in range(count))
  binaries = tuple(f"binary-{epoch}".encode() for epoch in range(count))
  identities = tuple({
    "function_name": "target",
    "binary_sha256": hashlib.sha256(binary).hexdigest(),
  } for binary in binaries)
  keys = tuple(program.key.hex() for program in programs)
  dev = SimpleNamespace(
    timeline_value=10, timeline_signal=SimpleNamespace(value=9),
    prof_exec_counter=7)
  seen = []

  def resolve(device, program):
    epoch = programs.index(program)
    seen.append(epoch)
    base = 0x100000 + epoch*0x4000
    if corrupt == "overlap" and epoch == 1: base = 0x100800
    runtime = SimpleNamespace(
      dev=dev, name="target",
      lib=b"wrong" if corrupt == "binary" and epoch == 1 else binaries[epoch],
      lib_gpu=SimpleNamespace(va_addr=base, size=0x2000),
      prog_addr=(base-4 if corrupt == "entry" and epoch == 1 else base+0x100),
      aql_prog_addr=base+0x180)
    runtime_cache[(program.key, device)] = \
      object() if corrupt == "cache" and epoch == 1 else runtime
    dev.timeline_value += 1
    dev.timeline_signal.value = dev.timeline_value - 1
    return runtime
  return programs, keys, identities, dev, seen, resolve


def test_v2_runtime_preconstruction_uses_exact_get_runtime_cache_and_code_lifetimes():
  from tinygrad.engine.realize import runtime_cache
  programs, keys, identities, dev, seen, resolve = _fake_runtime_preconstruction_family()
  with patch.dict(runtime_cache, {}, clear=True), \
       patch("tinygrad.device.Device", {"AMD": dev}), \
       patch("tinygrad.engine.realize.get_runtime", side_effect=resolve) as get_runtime:
    evidence = _preconstruct_frozen_program_runtimes(programs, keys, identities)
  assert seen == [0, 1, 2]
  assert [call.args for call in get_runtime.call_args_list] == [
    ("AMD", program) for program in programs]
  assert evidence["status"] == "PASS" and evidence["all_checks_pass"] is True
  assert evidence["no_compute_dispatch_during_preconstruction"] is True
  assert evidence["runtime_cache_retains_code_allocations"] is True
  assert evidence["prof_exec_counter_before"] == evidence["prof_exec_counter_after"] == 7
  assert evidence["timeline_before"]["timeline_value"] == 10
  assert evidence["timeline_after"]["timeline_value"] == 13
  assert [row["epoch"] for row in evidence["runtimes"]] == [0, 1, 2]
  assert all(row["all_checks_pass"] for row in evidence["runtimes"])
  assert len({row["runtime_object_id"] for row in evidence["runtimes"]}) == 3


@pytest.mark.parametrize(("corrupt", "failed_check"), [
  ("binary", "runtime_identity_matches_retained_binary"),
  ("cache", "runtime_cache_exact_program_binding"),
  ("overlap", "program_library_disjoint_from_prior"),
  ("entry", "program_entry_in_library_range"),
])
def test_v2_runtime_preconstruction_fails_closed_with_partial_evidence(corrupt, failed_check):
  from tinygrad.engine.realize import runtime_cache
  programs, keys, identities, dev, seen, resolve = \
    _fake_runtime_preconstruction_family(corrupt=corrupt)
  with patch.dict(runtime_cache, {}, clear=True), \
       patch("tinygrad.device.Device", {"AMD": dev}), \
       patch("tinygrad.engine.realize.get_runtime", side_effect=resolve), \
       pytest.raises(FrozenRuntimePreconstructionError) as caught:
    _preconstruct_frozen_program_runtimes(programs, keys, identities)
  evidence = caught.value.runtime_preconstruction
  assert evidence["status"] == "POST_GET_RUNTIME_AUDIT_ERROR"
  assert evidence["failure_boundary"] == "lifecycle_audit_after_get_runtime_return"
  assert evidence["all_checks_pass"] is False and seen == [0, 1]
  assert evidence["runtimes"][-1]["checks"][failed_check] is False
  assert evidence["runtimes"][-1]["all_checks_pass"] is False


def test_v2_runtime_preconstruction_rejects_polluted_cache_before_get_runtime():
  from tinygrad.engine.realize import runtime_cache
  programs, keys, identities, dev, seen, resolve = _fake_runtime_preconstruction_family()
  with patch.dict(runtime_cache, {(programs[0].key, "AMD"): object()}, clear=True), \
       patch("tinygrad.device.Device", {"AMD": dev}), \
       patch("tinygrad.engine.realize.get_runtime", side_effect=resolve) as get_runtime, \
       pytest.raises(FrozenRuntimePreconstructionError) as caught:
    _preconstruct_frozen_program_runtimes(programs, keys, identities)
  get_runtime.assert_not_called()
  assert seen == []
  assert caught.value.runtime_preconstruction["status"] == "REJECTED_PREEXISTING_CACHE"
  assert caught.value.runtime_preconstruction["preexisting_program_keys"] == [keys[0]]


def test_v2_runtime_preconstruction_records_third_attempt_before_get_runtime_raises():
  from tinygrad.engine.realize import runtime_cache
  programs, keys, identities, dev, seen, resolve = _fake_runtime_preconstruction_family()

  def fail_third(device, program):
    if program is programs[2]:
      seen.append(2)
      raise RuntimeError("third runtime construction failed")
    return resolve(device, program)

  with patch.dict(runtime_cache, {}, clear=True), \
       patch("tinygrad.device.Device", {"AMD": dev}), \
       patch("tinygrad.engine.realize.get_runtime", side_effect=fail_third), \
       pytest.raises(FrozenRuntimePreconstructionError) as caught:
    _preconstruct_frozen_program_runtimes(programs, keys, identities)
  evidence = caught.value.runtime_preconstruction
  assert evidence["status"] == "GET_RUNTIME_ERROR"
  assert evidence["failure_boundary"] == "get_runtime_call_raised_before_return"
  assert [attempt["epoch"] for attempt in evidence["attempts"]] == [0, 1, 2]
  failed = evidence["failed_attempt"]
  assert failed["epoch"] == 2 and failed["program_key"] == keys[2]
  assert failed["expected_program_identity"] == identities[2]
  assert failed["get_runtime_begin"]["timeline_value"] == 12
  assert failed["get_runtime_returned"] is False
  assert len(evidence["runtimes"]) == 2


def test_v2_dispatch_runtime_crosscheck_rejects_ordered_object_identity_drift():
  preconstruction = {
    "enabled": True,
    "runtimes": [
      {"program_key": "1"*64, "runtime_object_id": 101},
      {"program_key": "2"*64, "runtime_object_id": 202},
      {"program_key": "3"*64, "runtime_object_id": 303},
    ],
  }
  exact = {"calls": [
    {"program_key": "1"*64, "runtime_lifecycle": {"runtime_object_id": 101}},
    {"program_key": "2"*64, "runtime_lifecycle": {"runtime_object_id": 202}},
    {"program_key": "3"*64, "runtime_lifecycle": {"runtime_object_id": 303}},
  ]}
  passed = _crosscheck_preconstructed_dispatch_runtimes(preconstruction, exact)
  assert passed["status"] == "PASS" and passed["all_checks_pass"] is True

  drifted = {"calls": [*exact["calls"][:2], {
    "program_key": "3"*64, "runtime_lifecycle": {"runtime_object_id": 404}}]}
  rejected = _crosscheck_preconstructed_dispatch_runtimes(preconstruction, drifted)
  assert rejected["status"] == "MISMATCH" and rejected["all_checks_pass"] is False
  assert rejected["checks"]["ordered_program_keys_match"] is True
  assert rejected["checks"]["ordered_runtime_object_ids_match"] is False


def test_v2_dispatch_runtime_crosscheck_preserves_matching_partial_dispatch_evidence():
  preconstruction = {
    "enabled": True,
    "runtimes": [
      {"program_key": "1"*64, "runtime_object_id": 101},
      {"program_key": "2"*64, "runtime_object_id": 202},
      {"program_key": "3"*64, "runtime_object_id": 303},
    ],
  }
  partial = {"calls": [
    {"program_key": "1"*64, "runtime_lifecycle": {"runtime_object_id": 101}},
    {"program_key": "2"*64, "runtime_lifecycle": {"runtime_object_id": 202}},
  ]}
  evidence = _crosscheck_preconstructed_dispatch_runtimes(preconstruction, partial)
  assert evidence["status"] == "INCOMPLETE" and evidence["all_checks_pass"] is False
  assert evidence["checks"]["observed_dispatch_prefix_reuses_preconstructed_runtimes"] is True
  assert evidence["dispatch_runtime_object_ids"] == [101, 202]


def test_v2_dispatch_exception_recovers_partial_runtime_reuse_without_unbound_state():
  preconstruction = {
    "enabled": True,
    "runtimes": [
      {"program_key": "1"*64, "runtime_object_id": 101},
      {"program_key": "2"*64, "runtime_object_id": 202},
      {"program_key": "3"*64, "runtime_object_id": 303},
    ],
  }
  partial = {"calls": [
    {"program_key": "1"*64, "runtime_lifecycle": {"runtime_object_id": 101}},
    {"program_key": "2"*64, "runtime_lifecycle": {"runtime_object_id": 202}},
  ]}
  exc = RuntimeError("realization failed")
  exc.pm4_dispatch_census = partial
  recovered, crosscheck = _dispatch_error_runtime_reuse_evidence(preconstruction, exc)
  assert recovered == partial
  assert crosscheck["status"] == "INCOMPLETE"
  assert crosscheck["dispatch_runtime_object_ids"] == [101, 202]


def test_v2_fixed_base_ordinal_identity_selects_only_exact_retained_binary():
  programs = tuple(SimpleNamespace(arg=SimpleNamespace(function_name="target")) for _ in range(4))
  binaries = tuple(f"epoch-{epoch}".encode() for epoch in range(4))
  binding = SimpleNamespace(artifact=SimpleNamespace(programs=programs, binaries=binaries))
  assert _frozen_program_set_ordinal_target_identity(binding, 2) == {
    "function_name": "target",
    "binary_sha256": hashlib.sha256(b"epoch-2").hexdigest(),
  }
  for invalid in (-1, 4, True):
    with pytest.raises(ValueError, match="outside the complete retained PROGRAM family"):
      _frozen_program_set_ordinal_target_identity(binding, invalid)


def test_v2_fixed_base_ordinal_sequence_identities_preserve_selected_order():
  programs = tuple(SimpleNamespace(arg=SimpleNamespace(function_name="target")) for _ in range(4))
  binaries = tuple(f"epoch-{epoch}".encode() for epoch in range(4))
  binding = SimpleNamespace(artifact=SimpleNamespace(programs=programs, binaries=binaries))
  identities = _frozen_program_set_ordinal_sequence_target_identities(binding, (1, 2))
  assert [row["binary_sha256"] for row in identities] == [
    hashlib.sha256(b"epoch-1").hexdigest(), hashlib.sha256(b"epoch-2").hexdigest()]
  duplicate = SimpleNamespace(artifact=SimpleNamespace(
    programs=programs, binaries=(b"epoch-0", b"same", b"same", b"epoch-3")))
  with pytest.raises(ValueError, match="not distinct"):
    _frozen_program_set_ordinal_sequence_target_identities(duplicate, (1, 2))


@pytest.mark.parametrize(("role", "prefix_epochs"), [
  ("attn_kv", 3), ("attn_kv", 20), ("ffn_down", 68),
])
def test_v2_fixed_base_isolated_reuses_health_aql_and_exact_prefix(tmp_path, role, prefix_epochs):
  class _Proc:
    returncode = 0
    stdout = '{"schema":"tinygrad.mmq_frozen_epoch_program_set_prefix_probe.v2","status":"PASS"}\n'
    stderr = ""
  bundle = tmp_path / "frozen-v2"
  with patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run", return_value=_Proc()) as run, \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value=""), \
       patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=True) as health:
    result = run_frozen_epoch_program_set_prefix_probe_isolated(
      role_spec=exact_role_spec(role), frozen_bundle=bundle,
      prefix_epochs=prefix_epochs, timeout_seconds=1)
  assert result["status"] == "PASS"
  assert result["health_before"] is result["health_after"] is True
  assert result["child_env_overrides"] == {"AMD_AQL": "1"}
  assert run.call_args.kwargs["env"]["AMD_AQL"] == "1"
  assert health.call_args_list[0].args[0] == {"AMD_AQL": "1"}
  code = run.call_args.args[0][2]
  assert "run_frozen_epoch_program_set_prefix_probe" in code
  assert f"prefix_epochs={prefix_epochs}" in code and f"exact_role_spec({role!r}" in code
  assert "preconstruct_runtimes=False" in code


def test_v2_fixed_base_isolated_forwards_runtime_preconstruction_opt_in(tmp_path):
  class _Proc:
    returncode = 0
    stdout = '{"schema":"tinygrad.mmq_frozen_epoch_program_set_prefix_probe.v2","status":"PASS","runtime_preconstruction":{"enabled":true,"status":"PASS"}}\n'
    stderr = ""
  with patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run", return_value=_Proc()) as run, \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value=""), \
       patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=True):
    result = run_frozen_epoch_program_set_prefix_probe_isolated(
      frozen_bundle=tmp_path / "frozen-v2", prefix_epochs=3,
      preconstruct_runtimes=True, timeout_seconds=1)
  assert result["status"] == "PASS"
  assert result["runtime_preconstruction"] == {"enabled": True, "status": "PASS"}
  assert "preconstruct_runtimes=True" in run.call_args.args[0][2]


@pytest.mark.parametrize(("role", "epoch"), [("attn_kv", 2), ("ffn_down", 67)])
def test_v2_fixed_base_ordinal_isolated_reuses_health_aql_and_exact_epoch(tmp_path, role, epoch):
  class _Proc:
    returncode = 0
    stdout = '{"schema":"tinygrad.mmq_frozen_epoch_program_set_ordinal_probe.v2","status":"PASS","scheduler_prefix_semantics_changed":false}\n'
    stderr = ""
  bundle = tmp_path / "frozen-v2"
  with patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run", return_value=_Proc()) as run, \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value=""), \
       patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=True) as health:
    result = run_frozen_epoch_program_set_ordinal_probe_isolated(
      role_spec=exact_role_spec(role), frozen_bundle=bundle,
      epoch=epoch, timeout_seconds=1)
  assert result["status"] == "PASS" and result["scheduler_prefix_semantics_changed"] is False
  assert result["health_before"] is result["health_after"] is True
  assert result["child_env_overrides"] == {"AMD_AQL": "1"}
  assert run.call_args.kwargs["env"]["AMD_AQL"] == "1"
  assert health.call_args_list[0].args[0] == {"AMD_AQL": "1"}
  code = run.call_args.args[0][2]
  assert "run_frozen_epoch_program_set_ordinal_probe" in code
  assert f"epoch={epoch}" in code and f"exact_role_spec({role!r}" in code


def test_v2_fixed_base_ordinal_sequence_isolated_reuses_health_aql_and_exact_order(tmp_path):
  class _Proc:
    returncode = 0
    stdout = '{"schema":"tinygrad.mmq_frozen_epoch_program_set_ordinal_sequence_probe.v2","status":"PASS","scheduler_prefix_semantics_changed":false}\n'
    stderr = ""
  bundle = tmp_path / "frozen-v2"
  with patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run", return_value=_Proc()) as run, \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value=""), \
       patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=True) as health:
    result = run_frozen_epoch_program_set_ordinal_sequence_probe_isolated(
      role_spec=exact_role_spec("attn_kv"), frozen_bundle=bundle,
      epochs=[1, 2], timeout_seconds=1)
  assert result["status"] == "PASS" and result["scheduler_prefix_semantics_changed"] is False
  assert result["health_before"] is result["health_after"] is True
  assert result["child_env_overrides"] == {"AMD_AQL": "1"}
  assert run.call_args.kwargs["env"]["AMD_AQL"] == "1"
  assert health.call_args_list[0].args[0] == {"AMD_AQL": "1"}
  code = run.call_args.args[0][2]
  assert "run_frozen_epoch_program_set_ordinal_sequence_probe" in code
  assert "epochs=(1, 2)" in code and "exact_role_spec('attn_kv'" in code


def test_v2_fixed_base_rejects_bad_prefix_before_health_or_gpu(tmp_path):
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe") as health, \
       patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run") as run:
    bad_prefix = run_frozen_epoch_program_set_prefix_probe_isolated(
      frozen_bundle=tmp_path / "frozen", prefix_epochs=4)
    not_full_for_role = run_frozen_epoch_program_set_prefix_probe_isolated(
      role_spec=exact_role_spec("ffn_down"),
      frozen_bundle=tmp_path / "frozen", prefix_epochs=20)
  health.assert_not_called()
  run.assert_not_called()
  assert bad_prefix["status"] == "BLOCKED" and "(1, 2, 3, 20)" in bad_prefix["exact_blocker"]
  assert not_full_for_role["status"] == "BLOCKED" and "(1, 2, 3, 68)" in not_full_for_role["exact_blocker"]


def test_v2_fixed_base_rejects_non_bool_runtime_preconstruction_before_health_or_gpu(tmp_path):
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe") as health, \
       patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run") as run:
    result = run_frozen_epoch_program_set_prefix_probe_isolated(
      frozen_bundle=tmp_path / "frozen", prefix_epochs=3,
      preconstruct_runtimes=1)
  health.assert_not_called()
  run.assert_not_called()
  assert result["status"] == "BLOCKED"
  assert result["exact_blocker"] == "preconstruct_runtimes must be a bool"


def test_v2_fixed_base_ordinal_rejects_bad_epoch_before_health_or_gpu(tmp_path):
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe") as health, \
       patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run") as run:
    negative = run_frozen_epoch_program_set_ordinal_probe_isolated(
      frozen_bundle=tmp_path / "frozen", epoch=-1)
    at_extent = run_frozen_epoch_program_set_ordinal_probe_isolated(
      frozen_bundle=tmp_path / "frozen", epoch=20)
  health.assert_not_called()
  run.assert_not_called()
  assert negative["status"] == "BLOCKED" and "must be in [0,20)" in negative["exact_blocker"]
  assert at_extent["status"] == "BLOCKED" and "must be in [0,20)" in at_extent["exact_blocker"]


def test_v2_fixed_base_ordinal_sequence_rejects_bad_selection_before_gpu(tmp_path):
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe") as health, \
       patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run") as run:
    duplicate = run_frozen_epoch_program_set_ordinal_sequence_probe_isolated(
      frozen_bundle=tmp_path / "frozen", epochs=(1, 1))
    reversed_order = run_frozen_epoch_program_set_ordinal_sequence_probe_isolated(
      frozen_bundle=tmp_path / "frozen", epochs=(2, 1))
  health.assert_not_called()
  run.assert_not_called()
  assert duplicate["status"] == "BLOCKED" and "strictly increasing" in duplicate["exact_blocker"]
  assert reversed_order["status"] == "BLOCKED" and "strictly increasing" in reversed_order["exact_blocker"]


def test_v2_fixed_base_isolated_wrappers_admit_explicit_pm4_with_same_health_boundary(tmp_path):
  class _Proc:
    returncode = 0
    stdout = '{"status":"PASS","scheduler_prefix_semantics_changed":false}\n'
    stderr = ""
  bundle = tmp_path / "frozen-v2"
  with patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run", return_value=_Proc()) as run, \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value=""), \
       patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=True) as health:
    results = (
      run_frozen_epoch_program_set_prefix_probe_isolated(
        frozen_bundle=bundle, prefix_epochs=3, timeout_seconds=1,
        child_env_overrides={"AMD_AQL": "0"}),
      run_frozen_epoch_program_set_ordinal_probe_isolated(
        frozen_bundle=bundle, epoch=2, timeout_seconds=1,
        child_env_overrides={"AMD_AQL": "0"}),
      run_frozen_epoch_program_set_ordinal_sequence_probe_isolated(
        frozen_bundle=bundle, epochs=(1, 2), timeout_seconds=1,
        child_env_overrides={"AMD_AQL": "0"}),
    )
  assert all(result["status"] == "PASS" for result in results)
  assert all(result["child_env_overrides"] == {"AMD_AQL": "0"} for result in results)
  assert all(call.kwargs["env"]["AMD_AQL"] == "0" for call in run.call_args_list)
  assert all(call.args[0] == {"AMD_AQL": "0"} for call in health.call_args_list)


def test_v2_fixed_base_prefix_admission_uses_dynamic_full_role_epoch_count():
  attn, down = exact_role_spec("attn_kv"), exact_role_spec("ffn_down")
  assert [_validate_v2_fixed_base_prefix_epochs(attn, value) for value in (1, 2, 3, 20)] == [1, 2, 3, 20]
  assert [_validate_v2_fixed_base_prefix_epochs(down, value) for value in (1, 2, 3, 68)] == [1, 2, 3, 68]
  for role_spec, invalid in ((attn, 4), (attn, 68), (down, 4), (down, 20), (down, True)):
    with pytest.raises(ValueError, match="prefix_epochs must be one of"):
      _validate_v2_fixed_base_prefix_epochs(role_spec, invalid)


def test_v2_fixed_base_ordinal_admission_uses_dynamic_full_role_epoch_count():
  attn, down = exact_role_spec("attn_kv"), exact_role_spec("ffn_down")
  assert [_validate_v2_fixed_base_ordinal(attn, value) for value in (0, 2, 19)] == [0, 2, 19]
  assert [_validate_v2_fixed_base_ordinal(down, value) for value in (0, 2, 67)] == [0, 2, 67]
  for role_spec, invalid in ((attn, -1), (attn, 20), (down, 68), (down, True)):
    with pytest.raises(ValueError, match="epoch must be in"):
      _validate_v2_fixed_base_ordinal(role_spec, invalid)


def test_v2_fixed_base_ordinal_sequence_admission_is_two_and_strictly_increasing():
  attn, down = exact_role_spec("attn_kv"), exact_role_spec("ffn_down")
  assert _validate_v2_fixed_base_ordinal_sequence(attn, [1, 2]) == (1, 2)
  assert _validate_v2_fixed_base_ordinal_sequence(down, (1, 67)) == (1, 67)
  for role_spec, invalid in (
      (attn, ()), (attn, (1,)), (attn, (1, 2, 3)), (attn, (1, True))):
    with pytest.raises(ValueError, match="exactly two integer ordinals"):
      _validate_v2_fixed_base_ordinal_sequence(role_spec, invalid)
  for invalid in ((-1, 1), (1, 1), (2, 1), (1, 20)):
    with pytest.raises(ValueError, match="strictly increasing"):
      _validate_v2_fixed_base_ordinal_sequence(attn, invalid)


def test_v2_fixed_base_cli_accepts_dynamic_full_role_epoch_count(monkeypatch, capsys, tmp_path):
  bundle = tmp_path / "frozen-v2"
  monkeypatch.setattr("sys.argv", [
    "mmq_llama_five_buffer_gpu_harness",
    "--scheduler-v2-fixed-base-prefix-epochs", "68",
    "--scheduler-v2-fixed-base-preconstruct-runtimes",
    "--target-role-name", "ffn_down",
    "--target-role-frozen-bundle", str(bundle),
  ])
  with patch(
      "extra.qk.mmq_llama_five_buffer_gpu_harness.run_frozen_epoch_program_set_prefix_probe_isolated",
      return_value={"status": "PASS"}) as probe:
    assert main() == 0
  assert probe.call_args.kwargs["role_spec"].role == "ffn_down"
  assert probe.call_args.kwargs["prefix_epochs"] == 68
  assert probe.call_args.kwargs["preconstruct_runtimes"] is True
  assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_v2_fixed_base_ordinal_cli_dispatches_research_only_probe(monkeypatch, capsys, tmp_path):
  bundle = tmp_path / "frozen-v2"
  monkeypatch.setattr("sys.argv", [
    "mmq_llama_five_buffer_gpu_harness",
    "--scheduler-v2-fixed-base-ordinal", "2",
    "--target-role-name", "attn_kv",
    "--target-role-frozen-bundle", str(bundle),
  ])
  with patch(
      "extra.qk.mmq_llama_five_buffer_gpu_harness.run_frozen_epoch_program_set_ordinal_probe_isolated",
      return_value={"status": "PASS", "scheduler_prefix_semantics_changed": False}) as probe:
    assert main() == 0
  assert probe.call_args.kwargs["role_spec"].role == "attn_kv"
  assert probe.call_args.kwargs["epoch"] == 2
  assert json.loads(capsys.readouterr().out)["scheduler_prefix_semantics_changed"] is False


def test_v2_fixed_base_ordinal_sequence_cli_dispatches_exact_selection(monkeypatch, capsys, tmp_path):
  bundle = tmp_path / "frozen-v2"
  monkeypatch.setattr("sys.argv", [
    "mmq_llama_five_buffer_gpu_harness",
    "--scheduler-v2-fixed-base-ordinal-sequence", "1", "2",
    "--target-role-name", "attn_kv",
    "--target-role-frozen-bundle", str(bundle),
  ])
  with patch(
      "extra.qk.mmq_llama_five_buffer_gpu_harness.run_frozen_epoch_program_set_ordinal_sequence_probe_isolated",
      return_value={"status": "PASS", "scheduler_prefix_semantics_changed": False}) as probe:
    assert main() == 0
  assert probe.call_args.kwargs["role_spec"].role == "attn_kv"
  assert probe.call_args.kwargs["epochs"] == [1, 2]
  assert json.loads(capsys.readouterr().out)["scheduler_prefix_semantics_changed"] is False


def test_target_role_in_place_mode_fails_closed_before_gpu_for_unsafe_options():
  with pytest.raises(ValueError, match="requires persistent_buffers"):
    run_full_grid_target_role_probe(in_kernel_accumulate=True)
  with pytest.raises(ValueError, match="intermediate readback"):
    run_full_grid_target_role_probe(in_kernel_accumulate=True, persistent_buffers=True, per_epoch_check=True)
  with pytest.raises(ValueError, match="mutually exclusive"):
    run_full_grid_target_role_probe(in_kernel_accumulate=True, persistent_buffers=True, host_accumulate=True)


def test_target_role_in_place_mode_compiles_accumulating_sink_without_gpu(monkeypatch):
  from extra.qk import mmq_llama_five_buffer_full_kernel as full_kernel
  built = []
  sentinel = object()
  def fake_build(m, n, k, *, accumulate=False):
    built.append((m, n, k, accumulate))
    return sentinel
  monkeypatch.setattr(full_kernel, "build_llama_five_buffer_full_kernel", fake_build)
  monkeypatch.setattr(full_kernel, "compile_llama_five_buffer_full_kernel",
                      lambda kernel: SimpleNamespace(emitted=False, program=None, blocker="cpu-test-stop"))
  row = run_full_grid_target_role_probe(in_kernel_accumulate=True, persistent_buffers=True)
  assert built == [(512, 17408, 256, True)]
  assert row["status"] == "BLOCKED" and row["exact_blocker"] == "cpu-test-stop"
  assert row["accumulation"] == TARGET_IN_PLACE_ACCUMULATION


@pytest.mark.parametrize("role,program_shape", [
  ("attn_kv", (512, 1024, 256)), ("attn_qo", (512, 5120, 256)), ("ffn_down", (512, 5120, 256))])
def test_target_role_probe_derives_program_geometry_from_admitted_role_without_gpu(monkeypatch, role, program_shape):
  from extra.qk import mmq_llama_five_buffer_full_kernel as full_kernel
  role_spec, built = exact_role_spec(role), []
  monkeypatch.setattr(full_kernel, "build_llama_five_buffer_full_kernel",
                      lambda m, n, k, *, accumulate=False:
                      built.append((m, n, k, accumulate)) or object())
  monkeypatch.setattr(full_kernel, "compile_llama_five_buffer_full_kernel",
                      lambda kernel: SimpleNamespace(emitted=False, program=None, blocker="cpu-test-stop"))
  row = run_full_grid_target_role_probe(role_spec=role_spec, in_kernel_accumulate=True, persistent_buffers=True)
  assert built == [(*program_shape, True)]
  assert row["shape"] == list(role_spec.shape) and row["exact_blocker"] == "cpu-test-stop"


def test_target_role_probe_rejects_noncanonical_role_spec_before_health_or_compile():
  kv = exact_role_spec("attn_kv")
  forged = replace(kv, candidate_canonical_identity="0" * 64)
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe") as health, \
       patch("extra.qk.mmq_llama_five_buffer_full_kernel.compile_llama_five_buffer_full_kernel") as compile_program:
    row = run_full_grid_target_role_probe_isolated(role_spec=forged, timeout_seconds=1)
  health.assert_not_called(); compile_program.assert_not_called()
  assert row["status"] == "BLOCKED" and "exact role admission failed" in row["exact_blocker"]


def test_target_role_frozen_bundle_replaces_compile_and_fails_closed_on_identity(monkeypatch):
  from extra.qk import mmq_frozen_target_artifact as frozen
  from extra.qk import mmq_llama_five_buffer_full_kernel as full_kernel
  compile_program = Mock(side_effect=AssertionError("must not compile"))
  monkeypatch.setattr(full_kernel, "compile_llama_five_buffer_full_kernel", compile_program)
  monkeypatch.setattr(frozen, "load_frozen_target_artifact", lambda path: SimpleNamespace(
    manifest={"schema": frozen.SCHEMA, "state": "FROZEN", "accumulation": "wrong", "accumulate": True},
    program=object(), fixture={}))
  row = run_full_grid_target_role_probe(
    in_kernel_accumulate=True, persistent_buffers=True, frozen_bundle="/cpu-only/frozen.tar")
  compile_program.assert_not_called()
  assert row["status"] == "BLOCKED"
  assert row["exact_blocker"] == "frozen target bundle validation failed"
  assert row["compile_performed"] is False and row["requires_recompile"] is False


def test_target_role_frozen_bundle_separates_qo_donor_from_down_execution_fixture_without_gpu():
  qo, down = exact_role_spec("attn_qo"), exact_role_spec("ffn_down")
  donor_fixture = {"schema": "fixture.v1", "role": qo.role, "shape": list(qo.shape)}
  artifact = SimpleNamespace(
    manifest={"schema": "frozen.v1", "state": "FROZEN",
              "artifacts": {"serialized_program_sha256": "program-sha"},
              "files": {"fixture.json": {"sha256": "donor-fixture-sha"}}},
    fixture=donor_fixture)
  binding = SimpleNamespace(
    artifact=artifact, artifact_role_spec=qo, role_spec=down,
    program_key="shared-program-key", shared_program_geometry=True)
  calls = []
  loaded, identity = _load_frozen_execution_binding(
    down, "/cpu-only/qo.tar",
    binding_loader=lambda role, path: calls.append((role, path)) or binding)
  assert loaded is binding and calls == [(down, "/cpu-only/qo.tar")]
  assert identity["artifact_role"] == "attn_qo"
  assert identity["artifact_full_role_shape"] == list(qo.shape)
  assert identity["execution_role"] == "ffn_down"
  assert identity["execution_full_role_shape"] == list(down.shape)
  assert identity["fixture_sha256"] == identity["artifact_fixture_sha256"] == "donor-fixture-sha"
  assert identity["fixture_relationship"] == "distinct_full_role_shared_program_geometry"

  execution_fixture = {"schema": "fixture.v1", "role": down.role, "shape": list(down.shape),
                       "total_epochs": down.epochs}
  roles = _validate_frozen_execution_fixture(binding, execution_fixture, dict(execution_fixture))
  assert roles["artifact_fixture_equals_execution_fixture"] is False
  assert roles["artifact_role"] == "attn_qo" and roles["execution_role"] == "ffn_down"
  assert roles["relationship"] == "distinct_full_role_shared_program_geometry"
  with pytest.raises(ValueError, match="differs from frozen bundle"):
    _validate_frozen_execution_fixture(binding, execution_fixture, {**execution_fixture, "total_epochs": 20})


def test_target_role_runtime_evidence_captures_views_kernarg_words_and_launch_count():
  class Handle:
    def __init__(self, va, size): self.va_addr, self.size = va, size
  class Buffer:
    def __init__(self, va, size, *, base=None, offset=0):
      self._handle, self.nbytes, self.offset = Handle(va, size), size, offset
      self._base = self if base is None else base
    @property
    def base(self): return self._base
    def get_buf(self, device): return self._handle
  bases = [Buffer(0x1000 + i*0x1000, 0x800) for i in range(5)]
  buffers = tuple(Buffer(base._handle.va_addr + 0x40, 0x100, base=base, offset=0x40) for base in bases)
  words = [buf.get_buf("AMD").va_addr for buf in buffers]
  class View:
    def view(self, **kwargs): return words
  class Kernarg:
    va_addr, size = 0x9000, 40
    def cpu_view(self): return View()
  class State:
    buf, bufs = Kernarg(), tuple(buf.get_buf("AMD") for buf in buffers)
  class Runtime:
    def fill_kernargs(self, bufs, vals=(), kernargs=None): return State()
    def __call__(self, *args, global_size, local_size, vals, wait):
      self.fill_kernargs(args, vals)
      return 1.0
  evidence = {"launches": [], "launch_count": 0}
  _dispatch_with_runtime_evidence(
    Runtime(), buffers, tuple(range(5)), global_size=(136, 4, 1), local_size=(256, 1, 1),
    vals=(), runtime_evidence=evidence, context={"epoch": 2})
  assert evidence["launch_count"] == 1
  launch = evidence["launches"][0]
  assert launch["epoch"] == 2 and len(launch["arguments"]) == 5
  assert all(row["va_matches_base_offset"] for row in launch["arguments"])
  assert launch["kernarg"]["va"] == 0x9000
  assert launch["kernarg"]["pointer_words"] == words
  assert launch["kernarg"]["pointer_words_match_bound"] is True


def test_target_role_runtime_identity_distinguishes_pm4_from_aql(monkeypatch):
  class Queue: pass
  class Device:
    is_aql, hw_compute_queue_t = True, Queue
  runtime = SimpleNamespace(lib_gpu=SimpleNamespace(va_addr=0x100000, size=0x2000),
                            prog_addr=0x100400, aql_prog_addr=0x100100)
  monkeypatch.setenv("AMD_AQL", "1")
  row = _runtime_identity_evidence(Device(), runtime, "a" * 64)
  assert row["amd_aql_env"] == "1" and row["amd_aql_effective"] is True
  assert row["queue_mode"] == "AQL" and row["queue_class"].endswith(".Queue")
  assert row["lib_va"] == 0x100000 and row["entry_va"] == 0x100400
  assert row["descriptor_va"] == 0x100100 and row["binary_sha256"] == "a" * 64


def test_target_role_frozen_fixture_validation_requires_exact_complete_identity():
  fixture = {"schema": "fixture.v1", "repack": {"q4_sha256": "a" * 64},
             "seeds": {"q4": 1}, "total_epochs": 20}
  _validate_frozen_fixture(fixture, json.loads(json.dumps(fixture)))
  changed = json.loads(json.dumps(fixture))
  changed["repack"]["q4_sha256"] = "b" * 64
  with pytest.raises(ValueError, match="differs from frozen bundle"):
    _validate_frozen_fixture(fixture, changed)


def test_target_role_in_place_sequence_zeros_same_output_and_epoch_step_has_no_hidden_op():
  output, copied = object(), []
  zeros = np.zeros(8, dtype=np.float32)
  for _ in range(2):
    assert _zero_persistent_target_output(output, zeros, lambda dst, src: copied.append((dst, src.copy()))) is output
  assert len(copied) == 2
  assert all(dst is output and src.dtype == np.float32 and not np.any(src) for dst, src in copied)

  class NoReadOrAdd:
    def numpy(self): raise AssertionError("in-place epoch must not read back")
    def __add__(self, other): raise AssertionError("in-place epoch must not launch an external add")
  partial = NoReadOrAdd()
  accum, accum_host = _accumulate_target_role_epoch(
    partial, NoReadOrAdd(), None, None, mode=TARGET_IN_PLACE_ACCUMULATION)
  assert accum is partial and accum_host is None


def test_target_role_isolated_wrapper_propagates_stable_metadata_flag():
  class _Proc:
    returncode = 0
    stdout = '{"status":"BLOCKED"}\nlate shutdown diagnostic\n'
    stderr = ""
  with patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run", return_value=_Proc()) as run, \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value=""), \
       patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=True):
    result = run_full_grid_target_role_probe_isolated(timeout_seconds=1, preloaded_epochs=True,
                                                       stable_metadata_staging=True,
                                                       stable_epoch_staging=True,
                                                       persistent_buffers=True,
                                                       in_kernel_accumulate=True,
                                                       wait_each_dispatch=False)
  assert result["status"] == "BLOCKED"
  assert result["kernel_faults"] == [] and result["health_after"] is True
  code = run.call_args.args[0][2]
  assert "stable_metadata_staging=True" in code
  assert "stable_epoch_staging=True" in code
  assert "in_kernel_accumulate=True" in code
  assert "wait_each_dispatch=False" in code


def test_target_role_isolated_wrapper_propagates_admitted_role_to_child():
  class _Proc:
    returncode = 0
    stdout = '{"status":"BLOCKED"}\n'
    stderr = ""
  role_spec = exact_role_spec("attn_kv")
  with patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run", return_value=_Proc()) as run, \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value=""), \
       patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=True):
    run_full_grid_target_role_probe_isolated(role_spec=role_spec, timeout_seconds=1)
  code = run.call_args.args[0][2]
  assert "exact_role_spec('attn_kv', shape=(512, 1024, 5120))" in code


def test_target_role_isolated_wrapper_propagates_frozen_bundle_and_narrow_aql_env(tmp_path):
  class _Proc:
    returncode = 0
    stdout = '{"status":"BLOCKED","compile_performed":false,"requires_recompile":false}\n'
    stderr = ""
  bundle = tmp_path / "frozen target.tar"
  with patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run", return_value=_Proc()) as run, \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value=""), \
       patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=True):
    result = run_full_grid_target_role_probe_isolated(
      timeout_seconds=1, in_kernel_accumulate=True, persistent_buffers=True,
      frozen_bundle=bundle, child_env_overrides={"AMD_AQL": "0"})
  assert result["child_env_overrides"] == {"AMD_AQL": "0"}
  assert result["mode_health_before"] is True and result["mode_health_after"] is True
  assert result["health_mode"] == {"amd_aql_env": "0", "before": True, "after": True}
  assert run.call_args.kwargs["env"]["AMD_AQL"] == "0"
  code = run.call_args.args[0][2]
  assert f"frozen_bundle={str(bundle.resolve())!r}" in code


def test_target_role_isolated_wrapper_rejects_broad_or_invalid_env_overrides():
  assert _validated_child_env_overrides({"AMD_AQL": "0"}) == {"AMD_AQL": "0"}
  with pytest.raises(ValueError, match="only permits AMD_AQL"):
    _validated_child_env_overrides({"PATH": "/tmp"})
  with pytest.raises(ValueError, match="must be '0' or '1'"):
    _validated_child_env_overrides({"AMD_AQL": "yes"})


def test_target_role_isolated_wrapper_blocks_before_target_when_preflight_is_unhealthy():
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=False), \
       patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run") as run:
    result = run_full_grid_target_role_probe_isolated(timeout_seconds=1)
  run.assert_not_called()
  assert result["status"] == "BLOCKED"
  assert result["exact_blocker"] == "pre-run GPU health probe failed"
  assert result["health_before"] is False


def test_target_role_isolated_wrapper_rejects_unsafe_in_kernel_readback_before_health_or_target():
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe") as health, \
       patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run") as run:
    result = run_full_grid_target_role_probe_isolated(
      timeout_seconds=1, in_kernel_accumulate=True, persistent_buffers=True, per_epoch_check=True)
  health.assert_not_called()
  run.assert_not_called()
  assert result["status"] == "BLOCKED"
  assert "intermediate readback" in result["exact_blocker"]


def test_target_role_isolated_timeout_captures_journal_and_post_health():
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", side_effect=[True, False]), \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value="journal"), \
       patch("extra.qk.mmq_target_epoch_orchestrator.parse_kernel_faults", return_value=["gpu-reset"]), \
       patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run",
             side_effect=__import__("subprocess").TimeoutExpired(["python"], 1)):
    result = run_full_grid_target_role_probe_isolated(timeout_seconds=1, epoch_limit=1)
  assert result["status"] == "BLOCKED" and result["timeout"] is True
  assert result["health_before"] is True and result["health_after"] is False
  assert result["kernel_faults"] == ["gpu-reset"]


def test_gpu_harness_numeric_mismatch_is_structured_and_json_safe():
  got = np.array([[np.nan, 2.0, np.inf], [4.0, 8.0, 0.0]], dtype=np.float32)
  reference = np.array([[1.0, 2.0, 3.0], [4.0, 7.0, 0.0]], dtype=np.float32)
  result = _numeric_comparison(got, reference)
  assert result["status"] == "mismatch"
  assert result["mismatch_count"] == 3
  assert result["first_mismatch_index"] == [0, 0]
  assert result["first_mismatch_got"] == "nan"
  assert result["first_mismatch_reference"] == 1.0
  assert result["nan_got"] == 1 and result["inf_got"] == 1
  assert result["joint_finite"] == 4
  assert result["max_abs_error"] == 1.0 and result["mean_abs_error"] == 0.25
  json.dumps(result, allow_nan=False)


def test_gpu_harness_numeric_match_reports_comparator_pass():
  result = _numeric_comparison(np.array([1.0, 2.0], dtype=np.float32),
                               np.array([1.0, 2.001], dtype=np.float32))
  assert result["status"] == "pass"
  assert result["mismatch_count"] == 0
  assert result["first_mismatch_index"] is None
  assert result["nan_got"] == result["nan_reference"] == 0
