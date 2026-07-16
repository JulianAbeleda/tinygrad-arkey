import pytest

from extra.qk.q4k_symbolic_loop_validation import PackedQ4Case, validate


def test_packed_q4_32x32x512_symbolic_loop_has_unique_addresses_and_stores():
  report = validate()
  assert report["verdict"] == "PACKED_Q4_SYMBOLIC_LOOP_PASS"
  assert report["output_tiles"] == 4
  assert report["unique_weight_addresses"] == 32 * 16
  assert report["unique_activation_addresses"] == 32 * 16
  assert report["duplicate_store_count"] == 0


@pytest.mark.parametrize("kwargs", [
  {"m": 31}, {"n": 31}, {"k": 500}, {"tile": 8, "k": 250},
])
def test_symbolic_witness_rejects_non_divisible_geometry(kwargs):
  with pytest.raises(ValueError): validate(PackedQ4Case(**kwargs))
