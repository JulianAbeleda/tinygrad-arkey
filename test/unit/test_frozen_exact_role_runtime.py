from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from tinygrad import dtypes
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

from extra.qk.mmq_exact_role_spec import ExactRoleSpec, exact_role_spec
from extra.qk.mmq_frozen_target_artifact import (
  ACCUMULATION, BACKEND_ID, FUNCTION_NAME, SCHEMA, FrozenTargetArtifact,
)
from extra.qk.prefill import frozen_exact_role_runtime as adapter
from extra.qk.q4k_q8_activation_producer import Q4KQ8ActivationTile


class FakeTensor:
  def __init__(self, shape, dtype, label="tensor"):
    self.shape, self.dtype, self.label = tuple(shape), dtype, label

  def reshape(self, *shape):
    shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)
    if -1 in shape:
      assert shape.count(-1) == 1
      known = 1
      for value in shape:
        if value != -1: known *= value
      total = 1
      for value in self.shape: total *= value
      shape = tuple(total // known if value == -1 else value for value in shape)
    return FakeTensor(shape, self.dtype, self.label)

  def __getitem__(self, index):
    if len(self.shape) == 3 and isinstance(index, tuple) and len(index) == 3 and isinstance(index[1], int):
      return FakeTensor((self.shape[0], self.shape[2]), self.dtype, self.label)
    if len(self.shape) == 2 and isinstance(index, tuple) and len(index) == 2:
      rows, cols = index
      assert isinstance(rows, slice) and isinstance(cols, slice)
      return FakeTensor((self.shape[0], cols.stop - cols.start), self.dtype, self.label)
    raise AssertionError(f"unsupported fake slice {self.shape!r} {index!r}")

  def cast(self, dtype): return FakeTensor(self.shape, dtype, self.label)
  def contiguous(self): return self


class FakeBoundary:
  device = "AMD"
  synchronized_epoch_dispatch = True

  def __init__(self):
    self.allocations, self.stages, self.dispatches, self.zeroed, self.runtime_programs = [], [], [], [], []

  def allocate(self, spec):
    buffer = FakeTensor((spec.elements,), spec.dtype, f"buffer:{spec.name}")
    self.allocations.append((spec, buffer))
    return buffer

  def zero(self, output): self.zeroed.append(output)

  def stage(self, destination, source, *, name, epoch):
    assert adapter._elements(destination) == adapter._elements(source)
    assert destination.dtype == source.dtype
    self.stages.append((name, epoch, destination, source))

  def create_runtime(self, program):
    self.runtime_programs.append(program)
    return object()

  def dispatch(self, runtime, buffers, *, program, epoch):
    self.dispatches.append((runtime, buffers, program, epoch))

  def finish(self, output, shape): return output.reshape(shape)


def _program(role_spec: ExactRoleSpec, *, source="exact source", binary=b"exact binary"):
  return UOp(Ops.PROGRAM, src=(
    UOp(Ops.SINK), UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR),
    UOp(Ops.SOURCE, arg=source), UOp(Ops.BINARY, arg=binary),
  ), arg=ProgramInfo(name=FUNCTION_NAME, globals=tuple(range(5)),
                     global_size=role_spec.program.grid, local_size=(256, 1, 1)))


def _artifact(role_spec: ExactRoleSpec, *, program=None, source="exact source", binary=b"exact binary"):
  program = _program(role_spec, source=source, binary=binary) if program is None else program
  abi = tuple({"slot": slot, "name": name, "dtype": f"{dtype}.ptr({elements})", "elements": elements}
              for slot, (name, dtype, elements) in enumerate(
                zip(adapter.ABI_NAMES, adapter.ABI_DTYPES, role_spec.program.abi_elements)))
  manifest = {
    "schema": SCHEMA, "state": "FROZEN", "backend_id": BACKEND_ID,
    "accumulation": ACCUMULATION, "accumulate": True,
    "gpu_runtime_initialized": False, "gpu_dispatch_performed": False,
    "shape": list(role_spec.program.shape), "full_role_shape": list(role_spec.shape),
    "consumer": {"requires_recompile": False},
    "program": {"key": program.key.hex(), "globals": list(range(5)),
                "global_size": list(role_spec.program.grid), "local_size": [256, 1, 1],
                "abi": list(abi)},
    "artifacts": {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "binary_sha256": hashlib.sha256(binary).hexdigest()},
  }
  fixture = {"role": role_spec.role, "shape": list(role_spec.shape)}
  return FrozenTargetArtifact(manifest, program, binary, source, "disassembly", fixture)


def _producer(calls):
  def produce(source, spec):
    calls.append((source, spec))
    assert source.shape == (spec.m, spec.k) and source.dtype == dtypes.float32 and spec.k == 256
    return Q4KQ8ActivationTile(
      FakeTensor(spec.values_shape, dtypes.int8, "q8-values"),
      FakeTensor(spec.metadata_shape, dtypes.float32, "q8-scales"),
      FakeTensor(spec.metadata_shape, dtypes.float32, "q8-sums"),
    )
  return produce


def _linear(role_spec, calls):
  packed = FakeTensor((role_spec.n * role_spec.epochs * 36,), dtypes.uint32, "exact-packed-weight")
  def packed_weight():
    calls.append("packed-weight")
    return packed
  return SimpleNamespace(bias=None, out_features=role_spec.n, in_features=role_spec.k,
                         q4k_storage=object(), prefill_packed_weight=packed_weight)


def _run(role_spec, artifact, *, boundary=None):
  producer_calls, weight_calls = [], []
  boundary = FakeBoundary() if boundary is None else boundary
  result = adapter.run_frozen_exact_q4k_research(
    _linear(role_spec, weight_calls), FakeTensor((role_spec.m, role_spec.k), dtypes.float16, "activation"),
    role_spec=role_spec, frozen_bundle="/frozen/exact.tar", enabled=True,
    artifact_loader=lambda path: artifact, boundary=boundary, activation_producer=_producer(producer_calls))
  return result, boundary, producer_calls, weight_calls


def test_frozen_exact_runtime_is_default_off_before_loading_or_touching_weight():
  role_spec = exact_role_spec("attn_kv")
  lin = SimpleNamespace(prefill_packed_weight=lambda: (_ for _ in ()).throw(AssertionError("weight touched")))
  assert adapter.run_frozen_exact_q4k_research(
    lin, object(), role_spec=role_spec, frozen_bundle="/missing", enabled=False,
    artifact_loader=lambda path: (_ for _ in ()).throw(AssertionError("artifact loaded"))) is None


def test_frozen_binding_rejects_forged_inventory_candidate_identity():
  admitted = exact_role_spec("attn_kv")
  forged = ExactRoleSpec(admitted.role, *admitted.shape, "0" * 64)
  with pytest.raises(ValueError, match="differs from its inventory-admitted candidate identity"):
    adapter.load_frozen_exact_role_binding(
      forged, "/frozen/exact.tar", artifact_loader=lambda path: _artifact(admitted))


def test_exact_k256_dispatch_count_grid_abi_and_stable_five_buffers():
  role_spec = exact_role_spec("attn_kv")
  result, boundary, producer_calls, weight_calls = _run(role_spec, _artifact(role_spec))
  assert result is not None and result.output.shape == (1, role_spec.m, role_spec.n)
  assert weight_calls == ["packed-weight"] and len(boundary.runtime_programs) == 1
  assert len(boundary.allocations) == 5 and len({id(row[1]) for row in boundary.allocations}) == 5
  assert boundary.zeroed == [boundary.allocations[0][1]]
  assert len(boundary.dispatches) == role_spec.epochs == 20
  assert [row[3] for row in boundary.dispatches] == list(range(role_spec.epochs))
  assert all(row[1] == tuple(x[1] for x in boundary.allocations) for row in boundary.dispatches)
  assert all(dispatch["global_size"] == list(role_spec.program.grid) for dispatch in result.evidence["dispatches"])
  assert [row["elements"] for row in result.evidence["abi"]] == list(role_spec.program.abi_elements)
  assert len(producer_calls) == role_spec.epochs
  assert all(call[0].shape == (role_spec.m, 256) for call in producer_calls)
  for name in adapter.ABI_NAMES[1:]:
    destinations = [id(dst) for staged_name, _, dst, _ in boundary.stages if staged_name == name]
    assert len(destinations) == role_spec.epochs and len(set(destinations)) == 1


def test_shared_n5120_frozen_program_is_accepted_for_qo_and_down_with_k_bounded_staging():
  qo, down = exact_role_spec("attn_qo"), exact_role_spec("ffn_down")
  qo_artifact = _artifact(qo)
  qo_result, _, _, _ = _run(qo, qo_artifact)
  down_result, _, _, _ = _run(down, qo_artifact)
  assert qo.program == down.program and qo.epochs == 20 and down.epochs == 68
  assert qo_result.binding.artifact_role_spec.role == "attn_qo"
  assert down_result.binding.role_spec.role == "ffn_down"
  assert down_result.binding.artifact_role_spec.role == "attn_qo"
  assert down_result.evidence["shared_program_geometry"] is True
  assert qo_result.evidence["dispatch_count"] == qo.epochs
  assert down_result.evidence["dispatch_count"] == down.epochs
  assert qo_result.evidence["staging"] == down_result.evidence["staging"]
  assert down_result.evidence["staging"]["depends_on_epoch_count"] is False


def test_runtime_adapter_never_calls_old_emitter_or_recompiler(monkeypatch):
  import extra.qk.prefill_route_adapter as old_adapter
  import extra.qk.mmq_target_epoch_orchestrator as compiler
  monkeypatch.setattr(old_adapter, "run_cooperative_q4k",
                      lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("old emitter called")))
  monkeypatch.setattr(compiler, "compile_target_program",
                      lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("recompile called")))
  role_spec = exact_role_spec("attn_kv")
  result, _, _, _ = _run(role_spec, _artifact(role_spec))
  assert result.evidence["mmq_compile_performed"] is False
  assert result.evidence["mmq_requires_recompile"] is False
  assert result.evidence["q8_producer_and_staging_use_tinygrad_runtime_lowering"] is True
  assert result.evidence["q4_gather_and_staging_use_tinygrad_runtime_lowering"] is True
  assert result.evidence["synchronized_epoch_dispatch"] is True
  assert result.evidence["hip_used"] is False


def test_fixed_epoch_staging_rejects_unsynchronized_runtime_boundary():
  role_spec, boundary = exact_role_spec("attn_kv"), FakeBoundary()
  boundary.synchronized_epoch_dispatch = False
  with pytest.raises(ValueError, match="requires a synchronized epoch dispatch"):
    _run(role_spec, _artifact(role_spec), boundary=boundary)


def test_native_tinygrad_boundary_waits_for_epoch_before_staging_can_be_overwritten():
  role_spec, calls = exact_role_spec("attn_kv"), []
  handles = tuple(object() for _ in range(5))
  buffers = tuple(SimpleNamespace(
    uop=SimpleNamespace(buffer=SimpleNamespace(get_buf=lambda device, handle=handle: handle))
  ) for handle in handles)
  class Runtime:
    def __call__(self, *args, **kwargs): calls.append((args, kwargs))
  boundary = adapter.TinygradFrozenRuntimeBoundary()
  boundary.dispatch(Runtime(), buffers, program=_program(role_spec), epoch=0)
  assert calls == [(handles, {
    "global_size": role_spec.program.grid, "local_size": (256, 1, 1), "vals": (), "wait": True,
  })]


@pytest.mark.parametrize("drift, message", [
  ("source", "source or binary payload"),
  ("binary", "source or binary payload"),
  ("grid", "launch geometry"),
  ("abi", "key, grid, or ABI"),
])
def test_frozen_runtime_fails_loud_on_program_source_binary_grid_or_abi_drift(drift, message):
  role_spec = exact_role_spec("attn_kv")
  artifact = _artifact(role_spec)
  if drift == "source":
    artifact = FrozenTargetArtifact(artifact.manifest, artifact.program, artifact.binary, "drift", artifact.disassembly, artifact.fixture)
  elif drift == "binary":
    artifact = FrozenTargetArtifact(artifact.manifest, artifact.program, b"drift", artifact.source, artifact.disassembly, artifact.fixture)
  elif drift == "grid":
    program = artifact.program.replace(arg=ProgramInfo(
      name=FUNCTION_NAME, globals=tuple(range(5)), global_size=(1, 1, 1), local_size=(256, 1, 1)))
    artifact = FrozenTargetArtifact(artifact.manifest, program, artifact.binary, artifact.source, artifact.disassembly, artifact.fixture)
  else:
    manifest = dict(artifact.manifest)
    manifest["program"] = dict(manifest["program"])
    manifest["program"]["abi"] = []
    artifact = FrozenTargetArtifact(manifest, artifact.program, artifact.binary, artifact.source, artifact.disassembly, artifact.fixture)
  with pytest.raises(ValueError, match=message):
    adapter.load_frozen_exact_role_binding(
      role_spec, "/frozen/exact.tar", artifact_loader=lambda path: artifact)
