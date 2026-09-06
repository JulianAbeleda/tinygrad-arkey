from types import SimpleNamespace
import pytest

from tinygrad.engine.realize import _validated_program_global_slots


def test_program_global_slots_accept_valid_indices():
  info = SimpleNamespace(name="valid", globals=(2, 0))
  assert _validated_program_global_slots(info, 3) == (2, 0)


@pytest.mark.parametrize("slots", ((3,), (-1,), (True,)))
def test_program_global_slots_reject_invalid_indices(slots):
  info = SimpleNamespace(name="bad_abi", globals=slots)
  with pytest.raises(RuntimeError, match=r"PROGRAM 'bad_abi'.*buffer count 3"):
    _validated_program_global_slots(info, 3)
