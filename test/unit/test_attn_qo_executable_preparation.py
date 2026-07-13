import pytest

from extra.qk.prefill.attn_qo_executable_preparation import compile_attn_qo_pair, compile_attn_qo_program
from extra.qk.prefill.attn_qo_direct_l2_adapter_20260712 import prepare_exact_pair
from tinygrad.runtime.bridge import prepare_executable
from tinygrad.uop.ops import Ops


def test_exact_direct_attn_qo_compiles_to_real_program_and_passes_evidence():
  prepared = compile_attn_qo_program()
  assert prepared["transport"] == "direct_l2"
  assert prepared["compile_evidence"]["passed"] is True
  assert prepared["compile_evidence"]["capture"]["dispatch_permitted"] is False
  assert prepared["dispatch_performed"] is False


def test_exact_lds_attn_qo_compiles_to_real_program_and_preserves_transport_identity():
  prepared = compile_attn_qo_program(transport="lds")
  assert prepared["transport"] == "lds"
  assert prepared["compile_evidence"]["passed"] is True
  assert prepared["compile_evidence"]["transport"] == "lds"
  assert prepared["compile_evidence"]["schedule"]["lds_bytes"] == 20480
  assert prepared["compile_evidence"]["capture"]["dispatch_permitted"] is False
  assert prepared["dispatch_performed"] is False


def test_unknown_attn_qo_transport_fails_closed():
  with pytest.raises(ValueError, match="unsupported"):
    compile_attn_qo_program(transport="unknown")


@pytest.mark.parametrize("transport", ("direct_l2", "lds"))
def test_both_transports_join_to_the_shared_non_dispatching_bridge(transport):
  prepared = compile_attn_qo_program(transport=transport)
  binary = next(u.arg for u in prepared["program"].src if u.op is Ops.BINARY)

  class Runtime:
    lib = binary
    def __call__(self, *args, **kwargs):
      raise AssertionError("dispatch must remain explicit")

  from unittest.mock import patch
  with patch("tinygrad.runtime.bridge.get_runtime", return_value=Runtime()):
    handle = prepare_executable(prepared["program"], prepared["compile_evidence"], device="AMD")
  assert handle.artifact.binary == binary


def test_exact_pair_is_admitted_by_the_existing_pair_authority():
  prepared = compile_attn_qo_pair()
  direct, lds = prepared["transports"]["direct_l2"], prepared["transports"]["lds"]
  result = prepare_exact_pair(
    direct_payload=direct["candidate"], lds_payload=lds["candidate"],
    direct_binary_sha256=direct["compile_evidence"]["binary_sha256"],
    lds_binary_sha256=lds["compile_evidence"]["binary_sha256"], pair_key=prepared["pair_key"])
  assert result["status"] == "prepared"
  assert result["candidates"]["direct_l2"]["binary_sha256"] != result["candidates"]["lds"]["binary_sha256"]
