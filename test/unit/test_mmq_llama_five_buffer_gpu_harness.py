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
from extra.qk.mmq_llama_five_buffer_gpu_harness import (TARGET_IN_PLACE_ACCUMULATION, _accumulate_target_role_epoch,
  _aql_target_program_identity, _audit_target_aql_kernargs,
  _bind_sink, _dispatch_with_runtime_evidence, _numeric_comparison, _pack_q4_epochs_contiguous,
  _decode_aql_kernel_dispatch_packet, _load_frozen_execution_binding, _random_q4_words, _runtime_identity_evidence,
  _fixed_base_prefix_reference_operands, _frozen_program_set_target_identities,
  _producer_oracle_diagnostic, _producer_probe_status,
  _realize_outputs_together, _retained_producer_tensors,
  _scheduler_prefix_two_launches,
  _validate_v2_fixed_base_prefix_epochs,
  _validate_frozen_execution_fixture, _validate_frozen_fixture, _validated_child_env_overrides,
  _zero_persistent_target_output,
  main, run_amd_validation, run_frozen_scheduler_prefix_two_probe_isolated,
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
    "output_va_fixed": True, "q8_scale_va_fixed": True, "all_five_vas_fixed": True}
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
  with pytest.raises(ValueError, match="exactly five"):
    _audit_target_aql_kernargs([1, 2], [], expected_vas=None, require_fixed_scale_va=True)


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


def test_v2_fixed_base_rejects_bad_prefix_or_pm4_before_health_or_gpu(tmp_path):
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe") as health, \
       patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run") as run:
    bad_prefix = run_frozen_epoch_program_set_prefix_probe_isolated(
      frozen_bundle=tmp_path / "frozen", prefix_epochs=4)
    not_full_for_role = run_frozen_epoch_program_set_prefix_probe_isolated(
      role_spec=exact_role_spec("ffn_down"),
      frozen_bundle=tmp_path / "frozen", prefix_epochs=20)
    pm4 = run_frozen_epoch_program_set_prefix_probe_isolated(
      frozen_bundle=tmp_path / "frozen", prefix_epochs=1,
      child_env_overrides={"AMD_AQL": "0"})
  health.assert_not_called()
  run.assert_not_called()
  assert bad_prefix["status"] == "BLOCKED" and "(1, 2, 3, 20)" in bad_prefix["exact_blocker"]
  assert not_full_for_role["status"] == "BLOCKED" and "(1, 2, 3, 68)" in not_full_for_role["exact_blocker"]
  assert pm4["status"] == "BLOCKED" and "requires AMD_AQL=1" in pm4["exact_blocker"]


def test_v2_fixed_base_prefix_admission_uses_dynamic_full_role_epoch_count():
  attn, down = exact_role_spec("attn_kv"), exact_role_spec("ffn_down")
  assert [_validate_v2_fixed_base_prefix_epochs(attn, value) for value in (1, 2, 3, 20)] == [1, 2, 3, 20]
  assert [_validate_v2_fixed_base_prefix_epochs(down, value) for value in (1, 2, 3, 68)] == [1, 2, 3, 68]
  for role_spec, invalid in ((attn, 4), (attn, 68), (down, 4), (down, 20), (down, True)):
    with pytest.raises(ValueError, match="prefix_epochs must be one of"):
      _validate_v2_fixed_base_prefix_epochs(role_spec, invalid)


def test_v2_fixed_base_cli_accepts_dynamic_full_role_epoch_count(monkeypatch, capsys, tmp_path):
  bundle = tmp_path / "frozen-v2"
  monkeypatch.setattr("sys.argv", [
    "mmq_llama_five_buffer_gpu_harness",
    "--scheduler-v2-fixed-base-prefix-epochs", "68",
    "--target-role-name", "ffn_down",
    "--target-role-frozen-bundle", str(bundle),
  ])
  with patch(
      "extra.qk.mmq_llama_five_buffer_gpu_harness.run_frozen_epoch_program_set_prefix_probe_isolated",
      return_value={"status": "PASS"}) as probe:
    assert main() == 0
  assert probe.call_args.kwargs["role_spec"].role == "ffn_down"
  assert probe.call_args.kwargs["prefix_epochs"] == 68
  assert json.loads(capsys.readouterr().out)["status"] == "PASS"


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
