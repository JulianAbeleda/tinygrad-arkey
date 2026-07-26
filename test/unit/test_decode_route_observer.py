import pytest

from tinygrad.llm.decode_route_observer import (decode_route_scope, notify_decode_route_execution,
  observe_decode_route_executions)


class _Output:
  uop = "output-uop"


def _notify():
  notify_decode_route_execution(route_id="route", candidate_id="candidate", model_identity="model",
    shape=(1, 1, 4096, 4096), tile_name="tile", combine_name="combine", output_path="path", output=_Output())


def test_decode_observer_requires_scope_and_observer():
  seen = []
  _notify()
  with observe_decode_route_executions(seen.append): _notify()
  with decode_route_scope(): _notify()
  assert seen == []


def test_decode_observer_records_selected_route_and_output_uop():
  seen = []
  with observe_decode_route_executions(seen.append), decode_route_scope(): _notify()
  assert len(seen) == 1
  assert seen[0].route_id == "route"
  assert seen[0].shape == (1, 1, 4096, 4096)
  assert seen[0].output_uop == "output-uop"


def test_decode_observer_rejects_non_callable():
  with pytest.raises(TypeError):
    with observe_decode_route_executions(None): pass
