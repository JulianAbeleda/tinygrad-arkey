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
