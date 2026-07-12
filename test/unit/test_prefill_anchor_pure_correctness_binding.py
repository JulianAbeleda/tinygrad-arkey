import hashlib
import numpy as np
import pytest

from extra.qk.prefill import anchor_pure_correctness_binding as gate


def test_anchor_cases_are_exact_shape_and_independently_defined():
  for name in gate.DEFAULT_CASES:
    a, b, ref = gate._case_arrays(name)
    assert a.shape == (gate.M, gate.K) and b.shape == (gate.N, gate.K)
    assert ref.shape == (gate.M, gate.N)
  assert np.all(gate._case_arrays("constant")[2] == -8.0)
  assert np.count_nonzero(gate._case_arrays("alternating")[2]) == 0


def test_execution_config_hash_is_canonical():
  assert gate._canonical_hash({"b": 2, "a": 1}) == gate._canonical_hash({"a": 1, "b": 2})
  assert len(gate._canonical_hash({"a": 1})) == 64


def test_pure_environment_rejects_graph_gemm(monkeypatch):
  monkeypatch.setenv("PURE_MACHINE_SEARCH_ONLY", "1")
  monkeypatch.setenv("PREFILL_GRAPH_GEMM", "1")
  with pytest.raises(RuntimeError, match="must be disabled"):
    gate._assert_pure_environment()


def test_program_identity_hashes_bound_payload():
  from tinygrad.dtype import dtypes
  from tinygrad.uop.ops import KernelInfo, Ops, UOp
  program = UOp(Ops.PROGRAM, src=(UOp.sink(), UOp(Ops.DEVICE, arg="AMD"),
    UOp(Ops.LINEAR, src=(UOp(Ops.NOOP, dtypes.void),)), UOp(Ops.SOURCE, arg="source"),
    UOp(Ops.BINARY, arg=b"binary")), arg=KernelInfo(name="k"))
  row = gate._program_identity(program)
  assert row["source_sha256"] == hashlib.sha256(b"source").hexdigest()
  assert row["binary_sha256"] == hashlib.sha256(b"binary").hexdigest()
