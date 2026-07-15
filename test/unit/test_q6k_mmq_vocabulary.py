import struct
import pytest

from extra.qk.mmq_logical_vocabulary import Axis, EdgePredicate, Stage, Staging
from extra.qk.q6k_mmq_vocabulary import Q6K_BLOCK_BYTES, Q6KDecode, Q6KMMQDescriptor, q6k_block_dot, q6k_weight


def descriptor(**kw):
  axes = tuple(Axis(n, e, tile=16 if n in ("m", "n") else None)
               for n, e in (("m", 1), ("n", 1), ("k", 256), ("group", 16), ("activation_block", 8)))
  return Q6KMMQDescriptor(axes, edge_predicates=(EdgePredicate("m"), EdgePredicate("n"), EdgePredicate("k")), **kw)


def test_q6_reference_decodes_signed_scale_and_zero_point_without_tensor_fallback():
  block = bytearray(Q6K_BLOCK_BYTES)
  block[0] = 0x01; block[192] = 2
  block[208:210] = struct.pack("<e", 0.5)
  activation = [0.0] * 256; activation[0] = 1.0
  assert q6k_weight(block, 0, 0) == pytest.approx(-31.0)
  assert q6k_block_dot(block, activation) == pytest.approx(-31.0)


def test_q6_grammar_is_distinct_and_common_contract_is_reused():
  d = descriptor()
  assert d.quant == "Q6_K" and d.decode.byte_layout.startswith("ql_")
  assert d.staging.weights is Stage.DIRECT
  assert d.ownership.writeback == "exactly_one_owner"
  assert "q4k" not in d.canonical()


@pytest.mark.parametrize("bad", [Q6KDecode(block_elements=128), Q6KDecode(d_offset=206)])
def test_q6_decode_rejects_non_q6_grammar(bad):
  with pytest.raises(ValueError, match="Q6_K"):
    Q6KMMQDescriptor(descriptor().axes, decode=bad, edge_predicates=descriptor().edge_predicates)


def test_q6_reference_fails_closed_on_full_block_and_activation_contracts():
  with pytest.raises(ValueError): q6k_weight(b"", 0, 0)
  with pytest.raises(ValueError): q6k_block_dot(bytes(Q6K_BLOCK_BYTES), [1.0])
