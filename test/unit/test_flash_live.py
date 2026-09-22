"""Live flash hooks: the wiring and the request contract, without a GPU. The GPU run is the provider itself
(BoltBeam exp's flash worker against `search_provider.py --live-flash`)."""
import pytest

from extra.llm_research import flash_live, search_provider as provider


def test_context_tokens_is_required_and_positive():
  assert flash_live.context_tokens({"execution": {"context_tokens": 513}}) == 513
  for payload in ({}, {"execution": {}}, {"execution": {"context_tokens": 0}}, {"execution": {"context_tokens": True}},
                  {"execution": {"context_tokens": "513"}}):
    with pytest.raises(provider.ProtocolError) as err: flash_live.context_tokens(payload)
    assert err.value.code == "admission_rejected"


def test_the_control_is_the_production_tile_not_the_candidates_geometry():
  """The check oracle is split 48 with the emitter's defaults; a candidate's token block or lane width never leaks in."""
  tile = {"Hq": 32, "Hd": 128, "Hkv": 8, "MAXC": 4608, "staging": "KV_BOTH", "quant": False, "rope": False,
          "split_count": 32, "token_block": 32, "lane_width": 16}
  control = flash_live._control_spec(tile)
  assert control.tile.split_count == flash_live.CONTROL_SPLIT
  assert (control.tile.token_block, control.tile.lane_width) == (16, 32)
  assert control.combine is not None


def test_live_flash_hooks_are_bound_to_one_device():
  live = flash_live.LiveFlash("METAL")
  adapter = provider.FlashAdapter("METAL", live_backend=True, compile_fn=live.compile, check_fn=live.check, measure_fn=live.measure)
  assert adapter.compile_fn.__self__ is live and adapter.measure_fn.__self__ is live
  assert live.device == "METAL" and not live._tiles


def test_measure_refuses_warmups_before_touching_a_gpu():
  live = flash_live.LiveFlash("NONE")
  with pytest.raises(provider.ProtocolError) as err:
    live.measure({"candidate_hash": "a" * 64, "execution": {"context_tokens": 513, "samples": 5, "warmups": 2}}, {"tile": {}, "combine": None})
  assert err.value.code == "admission_rejected" and "warmups" in str(err.value)
