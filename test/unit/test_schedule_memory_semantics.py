import pytest
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.memory_semantics import (MemorySemanticClass, PREFILL_ACTIVATION, PREFILL_OUTPUT, PREFILL_SCRATCH,
  KV_CACHE, MODEL_PARAMETER, RUNTIME_ACTIVATION, RUNTIME_INPUT, RUNTIME_OUTPUT, RUNTIME_PERSISTENT, RUNTIME_SCRATCH,
  candidate_workspace, kv_cache, mark_memory_semantic, materialize_runtime_input, model_parameter, prefill_activation, prefill_output,
  prefill_scratch, runtime_activation, runtime_input, runtime_output,
  runtime_persistent, runtime_scratch)
from extra.qk.schedule_memory_manifest import collect_memory_plan_manifests
from extra.qk.schedule_memory_evidence import schedule_memory_evidence
from tinygrad.uop.ops import Ops, UOp
from tinygrad.uop.spec import spec_program, spec_tensor, type_verify

@pytest.mark.parametrize("mark,owner,size", [
  (prefill_activation, PREFILL_ACTIVATION, 7),
  (prefill_output, PREFILL_OUTPUT, 8),
  (prefill_scratch, PREFILL_SCRATCH, 9),
])
def test_lazy_tensor_semantic_reaches_concrete_schedule_buffer(mark, owner, size):
  out = mark((Tensor.rand(size, device="CPU") + 1).contiguous())
  with collect_memory_plan_manifests() as manifests: out.realize()
  assert any(row.semantic_owner == owner for manifest in manifests for row in manifest.buffers)

@pytest.mark.parametrize("mark,owner", [
  (model_parameter, MODEL_PARAMETER), (kv_cache, KV_CACHE),
  (runtime_persistent, RUNTIME_PERSISTENT), (runtime_input, RUNTIME_INPUT),
  (runtime_activation, RUNTIME_ACTIVATION), (runtime_output, RUNTIME_OUTPUT), (runtime_scratch, RUNTIME_SCRATCH),
])
def test_persistent_and_external_semantic_markers(mark, owner):
  out = mark((Tensor.rand(7, device="CPU") + 1).contiguous())
  with collect_memory_plan_manifests() as manifests: out.realize()
  assert any(row.semantic_owner == owner for manifest in manifests for row in manifest.buffers)

def test_candidate_id_is_rejected_for_non_candidate_classes():
  for semantic_class in MemorySemanticClass:
    if semantic_class is not MemorySemanticClass.CANDIDATE_WORKSPACE:
      with pytest.raises(ValueError):
        mark_memory_semantic(Tensor.empty(1, device="CPU"), type(MODEL_PARAMETER)(semantic_class, "candidate"))

def test_candidate_workspace_preserves_candidate_id():
  owner = candidate_workspace("candidate-a")
  out = mark_memory_semantic((Tensor.rand(8, device="CPU") * 2).contiguous(), owner)
  with collect_memory_plan_manifests() as manifests: out.realize()
  row = next(row for manifest in manifests for row in manifest.buffers if row.semantic_owner != "unknown")
  assert row.semantic_owner == owner
  assert row.semantic_owner.semantic_class is MemorySemanticClass.CANDIDATE_WORKSPACE
  assert row.semantic_owner.candidate_id == "candidate-a"


def test_typed_owner_is_accepted_by_fail_closed_schedule_evidence():
  source = prefill_activation(Tensor.empty(8, device="CPU"))
  out = prefill_output((source + 1).contiguous())
  with collect_memory_plan_manifests() as manifests: out.realize()
  assert manifests
  evidence = [schedule_memory_evidence(x) for x in manifests]
  assert all(x.complete for x in evidence)
  assert {row.semantic_class for item in evidence for row in item.peak_by_semantic_class} >= {
    "prefill_activation", "prefill_output"}

def test_unknown_is_explicit_and_conflicting_remark_fails_closed():
  out = (Tensor.rand(8, device="CPU") + 1).contiguous()
  mark_memory_semantic(out, PREFILL_OUTPUT)
  with pytest.raises(ValueError, match="already has semantic owner"): mark_memory_semantic(out, PREFILL_SCRATCH)
  unmarked = (Tensor.rand(9, device="CPU") + 2).contiguous()
  with collect_memory_plan_manifests() as manifests: unmarked.realize()
  assert all(row.semantic_owner == "unknown" for manifest in manifests for row in manifest.buffers)


def test_fresh_contiguous_allocation_may_have_a_different_owner_than_its_source_view():
  source = prefill_scratch((Tensor.rand(8, device="CPU") + 1).contiguous()).realize()
  fresh = prefill_activation(source.clone().contiguous())
  assert fresh.uop.op is Ops.MEMORY_SEMANTIC and fresh.uop.arg == PREFILL_ACTIVATION
  with collect_memory_plan_manifests() as manifests: fresh.realize()
  owners = {row.semantic_owner for manifest in manifests for row in manifest.buffers}
  assert {PREFILL_SCRATCH, PREFILL_ACTIVATION} <= owners

def test_owned_contiguous_materialization_binds_the_written_destination_slot():
  source = model_parameter(Tensor.rand(8, device="CPU").realize())
  output = model_parameter(source.contiguous())
  source_id = f"buffer:{source.uop.buf_uop.key.hex()}"
  with collect_memory_plan_manifests() as manifests: output.realize()
  output_id = f"buffer:{output.uop.buf_uop.key.hex()}"
  rows = [row for manifest in manifests for row in manifest.buffers]
  assert source_id != output_id
  assert all(row.semantic_owner == MODEL_PARAMETER for row in rows if row.identity in (source_id, output_id))
  assert {row.identity for row in rows if row.semantic_owner == MODEL_PARAMETER} >= {source_id, output_id}

def test_owned_gguf_backed_copy_marks_destination_slot_without_wrapping_executable_argument(tmp_path):
  # Faithful packed-loader topology: a slice of DISK storage is copied to the
  # default compute device, made contiguous, and only the selected result is
  # declared to be model storage.
  path = tmp_path/"packed.bin"
  path.write_bytes(bytes(256))
  raw = Tensor(path, dtype=dtypes.uint32)
  output = model_parameter(raw[4:20].to(None).contiguous())
  assert any(u.op is Ops.COPY for u in output.uop.toposort())

  with collect_memory_plan_manifests() as manifests: linear, _ = Tensor.linear_with_vars(output)
  copy_call = next(call for call in linear.src if call.op is Ops.CALL and call.src[0].op is Ops.COPY)
  destination = copy_call.src[1]
  assert destination.op is Ops.BUFFER
  assert dict(copy_call.arg.memory_semantic_slots)[0] == MODEL_PARAMETER
  assert destination.buf_uop is output.uop.buf_uop
  destination_id = f"buffer:{destination.buf_uop.key.hex()}"
  row = next(row for manifest in manifests for row in manifest.buffers if row.identity == destination_id)
  assert row.device == output.device
  assert row.semantic_owner == MODEL_PARAMETER

def test_tinyjit_capture_preserves_explicit_owner():
  @TinyJit
  def run(x):
    return prefill_output((x + 1).contiguous())

  x = Tensor.rand(8, device="CPU").realize()
  run(x).realize()  # warmup
  with collect_memory_plan_manifests() as manifests: run(x).realize()  # capture and combined memory plan
  assert any(row.semantic_owner == PREFILL_OUTPUT for manifest in manifests for row in manifest.buffers)


def test_tinyjit_input_signature_ignores_per_invocation_semantic_owner():
  @TinyJit
  def run(x): return (x + 1).contiguous()

  plain = Tensor.rand(8, device="CPU").realize()
  output = prefill_output(Tensor.rand(8, device="CPU").realize())
  decode = runtime_input(Tensor.rand(8, device="CPU").realize())
  run(plain).realize()   # warmup
  run(output).realize()  # capture with one logical owner
  got = run(decode).realize()  # replay the same physical contract with another
  assert got.shape == (8,)


def test_runtime_input_materialization_owns_host_and_device_staging_buffers():
  with collect_memory_plan_manifests() as manifests:
    value = materialize_runtime_input(Tensor(list(range(2048)), dtype=dtypes.int32).reshape(1, 2048).contiguous())
  rows = [row for manifest in manifests for row in manifest.buffers]
  assert rows and all(row.semantic_owner == RUNTIME_INPUT for row in rows)
  assert {row.device for row in rows} >= {"PYTHON", value.device}


def test_structural_owner_is_part_of_cse_identity_and_tensor_only_vocabulary():
  source = UOp.new_buffer("CPU", 8, Tensor.empty(1).dtype, num=9101)
  activation = mark_memory_semantic(source, PREFILL_ACTIVATION)
  scratch = mark_memory_semantic(source, PREFILL_SCRATCH)
  assert activation is not scratch and activation.key != scratch.key
  type_verify(activation, spec_tensor)
  with pytest.raises(RuntimeError, match="verification failed"): type_verify(activation, spec_program)


def test_scheduler_consumes_semantic_carrier_before_backend_kernel():
  out = prefill_output((Tensor.rand(8, device="CPU") + 1).contiguous())
  linear, _ = Tensor.linear_with_vars(out)
  kernel_asts = [call.src[0] for call in linear.src if call.op is Ops.CALL and call.src[0].op is Ops.SINK]
  assert kernel_asts
  assert all(all(node.op is not Ops.MEMORY_SEMANTIC for node in ast.toposort()) for ast in kernel_asts)


def test_top_level_output_owner_does_not_change_dispatch_topology():
  plain_source = Tensor.rand(8, device="CPU").realize()
  owned_source = Tensor.rand(8, device="CPU").realize()
  plain = (plain_source + 1).contiguous()
  owned = prefill_output((owned_source + 1).contiguous())
  plain_linear, _ = Tensor.linear_with_vars(plain)
  owned_linear, _ = Tensor.linear_with_vars(owned)
  assert [(call.src[0].op, len(call.src)) for call in owned_linear.src] == \
         [(call.src[0].op, len(call.src)) for call in plain_linear.src]
  assert all(arg.op is not Ops.MEMORY_SEMANTIC for call in owned_linear.src for arg in call.src[1:])
  assert any(PREFILL_OUTPUT in dict(call.arg.memory_semantic_slots).values() for call in owned_linear.src)


def test_multi_kernel_owned_intermediate_and_output_are_both_concrete():
  intermediate = prefill_activation((Tensor.rand(8, device="CPU") + 1).contiguous())
  out = prefill_output((intermediate * 2).contiguous())
  with collect_memory_plan_manifests() as manifests: out.realize()
  owners = {row.semantic_owner for manifest in manifests for row in manifest.buffers}
  assert {PREFILL_ACTIVATION, PREFILL_OUTPUT} <= owners


def test_execution_output_classifies_only_planned_written_unknowns_as_scratch():
  @TinyJit
  def run(x):
    intermediate = (x + 1).contiguous()
    return prefill_output((intermediate * 2).contiguous())

  source = Tensor.rand(8, device="CPU").realize()
  run(source).realize()
  with collect_memory_plan_manifests() as manifests: run(source).realize()
  rows = [row for manifest in manifests for row in manifest.buffers]
  source_id = f"buffer:{source.uop.buf_uop.key.hex()}"
  assert any(row.semantic_owner == PREFILL_SCRATCH and row.arena_identity.startswith("arena:") for row in rows)
  assert all(row.semantic_owner == "unknown" for row in rows if row.identity == source_id)


def test_runtime_output_classifies_internal_intermediate_as_runtime_scratch():
  @TinyJit
  def run(feedback):
    intermediate = (feedback + 1).contiguous()
    return runtime_output((intermediate * 2).contiguous())

  feedback = prefill_output(Tensor.rand(8, device="CPU").realize())
  run(feedback).realize()
  with collect_memory_plan_manifests() as manifests: run(feedback).realize()
  rows = [row for manifest in manifests for row in manifest.buffers]
  assert any(row.semantic_owner == RUNTIME_SCRATCH for row in rows)
  assert any(row.semantic_owner == RUNTIME_OUTPUT for row in rows)
