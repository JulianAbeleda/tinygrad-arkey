import pytest

from extra.qk.decode import current_decode_execution_adapter as adapter
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops


def _request(**changes):
  values = {"adapter_id": adapter.ADAPTER_ID, "route_id": adapter.ROUTE_ID,
            "role": "ffn_gate_up", "rows": 12288, "k": 4096}
  values.update(changes)
  return adapter.CurrentDecodeCompileRequest(**values)


def test_request_is_explicit_and_reuses_current_promoted_manifest_authority():
  request = _request()
  assert request.route_id == "decode_q4k_g3_generated"
  with pytest.raises(ValueError, match="adapter_id must be"):
    _request(adapter_id="inferred-from-model-name")
  with pytest.raises(ValueError, match="route_id must be"):
    _request(route_id="retired_owned_kernel")
  with pytest.raises(ValueError, match="not admitted"):
    _request(role="lm_head")


def test_sink_binds_exact_promoted_semantic_abi_without_allocating_or_dispatching():
  request = _request()
  sink = adapter.build_current_decode_sink(request)
  assert sink.op is Ops.SINK
  buffers = [u for u in sink.toposort() if u.op is Ops.PARAM]
  assert [u.arg.slot for u in buffers] == [0, 1, 2]
  assert [u.dtype.size for u in buffers] == [12288, 12288 * (4096 // 256) * 36, 4096]
  assert [u.dtype.base for u in buffers] == [dtypes.float32, dtypes.uint32, dtypes.float16]


def test_compile_only_collects_final_artifacts_and_truthfully_blocks_execution():
  program, evidence = adapter.prepare_current_decode_compile(_request())
  assert program.op is Ops.PROGRAM
  assert program.arg.name == "q4k_g3_lanemap_gemv_12288_4096"
  assert evidence["passed"] is True and evidence["classification"] == "compile_only"
  assert evidence["capture"] == {"mode": "compile_only", "dispatch_permitted": False}
  assert len(evidence["source_sha256"]) == len(evidence["binary_sha256"]) == 64
  assert evidence["final_isa"]["text"]
  assert evidence["resource_summary"]["vgpr"] > 0
  assert [x["semantic_role"] for x in evidence["semantic_operands"]] == [
    "decode_output", "packed_weight", "decode_activation"]
  assert evidence["execution"]["dispatch_state"] == "not_attempted"
  assert evidence["execution"]["blocker"]["code"] == "exact_decode_input_reference_authority_unavailable"
  assert evidence["counter_evidence"]["status"] == "not_collected"


def test_adapter_refuses_to_fabricate_prepared_execution(monkeypatch):
  blocker = adapter.DecodeExecutionBlocker("exact_decode_input_reference_authority_unavailable", "execution", True,
    {"reason": "missing authority"})
  evidence = {"execution": {"blocker": blocker.to_dict()}}
  monkeypatch.setattr(adapter.CurrentDecodeExecutionAdapter, "classify", lambda self, _request: (object(), evidence))
  instance = adapter.CurrentDecodeExecutionAdapter()
  with pytest.raises(adapter.DecodeExecutionBlocked) as exc:
    instance.prepare(_request())
  assert exc.value.blocker == blocker


def _build_immutable_artifact(tmp_path, rows=32, k=1024, seed=99):
  import numpy as np
  from tinygrad import Tensor, dtypes
  from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, q4_k_reference
  rng = np.random.default_rng(seed)
  nblocks = (rows * k) // Q4_K_BLOCK_ELEMS
  raw = rng.integers(0, 256, size=nblocks * Q4_K_BLOCK_BYTES, dtype=np.uint8).reshape(nblocks, Q4_K_BLOCK_BYTES)
  d = (rng.standard_normal(nblocks).astype(np.float32) * 5e-4).astype(np.float16)
  dmin = (rng.standard_normal(nblocks).astype(np.float32) * 5e-4).astype(np.float16)
  raw[:, 0:2] = d.view(np.uint8).reshape(nblocks, 2); raw[:, 2:4] = dmin.view(np.uint8).reshape(nblocks, 2)
  byte_t = Tensor(raw.reshape(-1).copy()).realize()
  words = byte_t.bitcast(dtypes.uint32).numpy().astype(np.uint32)
  W = q4_k_reference(byte_t, rows * k).reshape(rows, k).cast(dtypes.float32).realize()
  x = (rng.standard_normal(k).astype(np.float32)).astype(np.float16)
  reference = (W @ Tensor(x.copy()).cast(dtypes.float32)).numpy().astype(np.float32)
  p = tmp_path / "decode_artifact.npz"
  np.savez(p, packed_words=words, activation=x, reference=reference)
  return str(p), rows, k


def test_verify_full_output_correctness_against_immutable_artifact(tmp_path):
  path, rows, k = _build_immutable_artifact(tmp_path)
  req = adapter.CurrentDecodeCompileRequest(adapter_id=adapter.ADAPTER_ID, route_id=adapter.ROUTE_ID,
                                            role="ffn_gate_up", rows=rows, k=k)
  result = adapter.CurrentDecodeExecutionAdapter().verify(req, path)
  assert result["status"] == "pass"
  assert result["element_count"] == rows and result["finite_output"] and result["inputs_unchanged"]
  assert result["relative_error"] < 1e-2
  assert result["reference_basis"] == "independent_q4k_dequant_gemv"
  assert result["identities"]["packed_words_sha256"] and result["identities"]["reference_sha256"]


def test_load_artifact_rejects_wrong_shape(tmp_path):
  import numpy as np, pytest
  p = tmp_path / "bad.npz"
  np.savez(p, packed_words=np.zeros(10, np.uint32), activation=np.zeros(1024, np.float16),
           reference=np.zeros(32, np.float32))
  req = adapter.CurrentDecodeCompileRequest(adapter_id=adapter.ADAPTER_ID, route_id=adapter.ROUTE_ID,
                                            role="ffn_gate_up", rows=32, k=1024)
  with pytest.raises(ValueError):
    adapter.load_immutable_decode_artifact(str(p), req)


def test_prepare_without_artifact_still_blocks():
  import pytest
  req = adapter.CurrentDecodeCompileRequest(adapter_id=adapter.ADAPTER_ID, route_id=adapter.ROUTE_ID,
                                            role="ffn_gate_up", rows=12288, k=4096)
  with pytest.raises(adapter.DecodeExecutionBlocked):
    adapter.CurrentDecodeExecutionAdapter().prepare(req)
