import hashlib, json, sys, pytest

from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import KernelCandidateContext, KernelInfo, Ops, ProgramInfo, UOp

from extra.qk.prefill import single_buffer_execution_authority as auth


def _program(identity="1" * 64, *, ops_ins=False):
  local = UOp(Ops.DEFINE_LOCAL, dtypes.half.ptr(size=64, addrspace=AddrSpace.LOCAL), arg="lds")
  idx = local.index(UOp.const(dtypes.int, 0))
  value = idx.load()
  sink = value.store(idx).sink(arg=KernelInfo(candidate_context=KernelCandidateContext(
    "boltbeam.full_kernel_candidate.v1", identity)))
  linear = UOp(Ops.LINEAR, src=((UOp(Ops.INS, arg="raw"),) if ops_ins else (UOp(Ops.NOOP),)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), linear,
    UOp(Ops.SOURCE, arg="ds_store_b128\nds_load_b128\ns_barrier"), UOp(Ops.BINARY, arg=b"binary")))


def test_exact_candidate_program_selection_and_hashes():
  identity = "1" * 64
  program = _program(identity)
  compiled = UOp(Ops.LINEAR, src=(program.call(),))
  assert auth._candidate_programs(compiled, identity) == [program]
  hashes = auth._program_identity(program)
  assert hashes["source_sha256"] == hashlib.sha256(b"ds_store_b128\nds_load_b128\ns_barrier").hexdigest()
  assert hashes["binary_sha256"] == hashlib.sha256(b"binary").hexdigest()


def test_compiler_lds_truth_requires_transport_and_barrier():
  row = auth._compiler_lds_truth(_program())
  assert row["actual_compiler_lds_staging"] is True
  program = _program().replace(src=(*_program().src[:3], UOp(Ops.SOURCE, arg="ds_store_b128\nds_load_b128"),
                                    UOp(Ops.BINARY, arg=b"binary")))
  assert auth._compiler_lds_truth(program)["actual_compiler_lds_staging"] is False


def test_compiler_lds_truth_accepts_compiler_rendered_shared_transport():
  program = _program().replace(src=(*_program().src[:3], UOp(Ops.SOURCE, arg=(
    "__attribute__((shared, aligned(16)))half buf0[2048];\n"
    "(*(buf0+0)) = data0[0]; half x = (*(buf0+0));\n__builtin_amdgcn_s_barrier();")),
    UOp(Ops.BINARY, arg=b"binary")))
  row = auth._compiler_lds_truth(program)
  assert row["actual_compiler_lds_staging"] is True
  assert row["shared_declaration_count"] == 1 and row["lds_bytes"] == 4096


def test_raw_ops_ins_surface_is_not_pure():
  from extra.qk.prefill.anchor_isa_resource_capture import _program_surface
  assert _program_surface(_program(ops_ins=True))["strict_pure"] is False


def test_structural_binding_rejects_context_bound_but_different_program():
  payload = {"workload": {"target": {"wave_size": 32}}, "schedule": {"tile": {"m": 128, "n": 128, "k": 32},
    "threads": 256, "waves": {"m": 4, "n": 2},
    "lds": {"windows": {"a": [0, 10240], "b": [10240, 20480]}, "strides": {"a": 80, "b": 80}}}}
  program = _program().replace(arg=ProgramInfo(local_size=(32, 1, 1)))
  binding = auth._structural_binding(payload, program, {"lds_bytes": 12288})
  assert binding["matches_payload"] is False
  assert "threads: expected 256, emitted 32" in binding["errors"]
  assert "LDS bytes: expected 20480, emitted 12288" in binding["errors"]
  assert "wave count: expected 8, emitted 1" in binding["errors"]
  assert "tile: emitted structure is unproven" in binding["errors"]
  assert binding["pre_gpu_eligible"] is False
  with pytest.raises(RuntimeError, match="refusing GPU execution"):
    auth._require_pre_gpu_structure(binding)


def test_structural_binding_still_rejects_matching_resource_totals_without_semantic_proof():
  payload = {"workload": {"target": {"wave_size": 32}}, "schedule": {"tile": {"m": 128, "n": 128, "k": 32},
    "threads": 256, "waves": {"m": 4, "n": 2},
    "lds": {"windows": {"a": [0, 10240], "b": [10240, 20480]}, "strides": {"a": 80, "b": 80}}}}
  program = _program().replace(arg=ProgramInfo(local_size=(256, 1, 1)))
  binding = auth._structural_binding(payload, program, {"lds_bytes": 20480})
  assert binding["actual"]["wave_count"] == 8
  assert binding["evidence"]["wave_count"] == "launch threads divided by target wave size"
  assert binding["pre_gpu_eligible"] is False
  assert binding["errors"] == ["tile: emitted structure is unproven", "waves: emitted structure is unproven",
                               "lds_windows: emitted structure is unproven", "lds_strides: emitted structure is unproven"]


def test_structural_binding_proves_only_exact_ordered_4x2_local_axes():
  payload = {"workload": {"target": {"wave_size": 32}}, "schedule": {"tile": {"m": 128, "n": 128, "k": 32},
    "threads": 256, "waves": {"m": 4, "n": 2},
    "lds": {"windows": {"a": [0, 10240], "b": [10240, 20480]}, "strides": {"a": 80, "b": 80}}}}
  binding = auth._structural_binding(payload, _program().replace(arg=ProgramInfo(local_size=(32, 4, 2))), {"lds_bytes": 20480})
  assert binding["actual"]["waves"] == {"m": 4, "n": 2}
  assert binding["evidence"]["waves"] == "ordered PROGRAM local axes (wave_size, waves_m, waves_n)"
  assert "waves: emitted structure is unproven" not in binding["errors"]
  assert binding["pre_gpu_eligible"] is False
  swapped = auth._structural_binding(payload, _program().replace(arg=ProgramInfo(local_size=(32, 2, 4))), {"lds_bytes": 20480})
  assert swapped["actual"]["waves"] is None
  assert "waves: emitted structure is unproven" in swapped["errors"]


def test_cli_writes_blocked_artifact_and_returns_nonzero(monkeypatch, tmp_path):
  payload = {"schema_version": "boltbeam.full_kernel_candidate.v1"}
  report = {"schema": auth.SCHEMA, "structural_binding": {"pre_gpu_eligible": False},
            "runtime": {"status": "not_run", "binary_equal": None},
            "correctness": {"status": "not_run", "passed": False},
            "runtime_binary_matches_candidate": None, "passed": False}
  output = tmp_path / "blocked.json"
  monkeypatch.setattr(auth, "_load_payload", lambda _: payload)
  monkeypatch.setattr(auth, "run", lambda *_args, **_kwargs: report)
  monkeypatch.setattr(sys, "argv", ["authority", "--candidate", str(tmp_path / "candidate.json"),
                                    "--candidate-hash", "1" * 64, "--output", str(output)])
  assert auth.main() == 1
  assert json.loads(output.read_text()) == report
