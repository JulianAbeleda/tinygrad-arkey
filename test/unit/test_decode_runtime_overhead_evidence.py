from extra.llm_research.decode.decode_runtime_overhead import _token_evidence


def test_complete_token_ids_are_opt_in_for_cross_runtime_correctness():
  compact = _token_evidence([1, 2, 3])
  complete = _token_evidence([1, 2, 3], include_token_ids=True)
  assert "token_ids" not in compact
  assert complete["token_ids"] == [1, 2, 3]
  assert complete["sha256"] == compact["sha256"]


def test_observed_jits_include_both_slots_and_flash_variants():
  from types import SimpleNamespace as NS
  from extra.llm_research.decode.decode_runtime_overhead import _decode_jits, _used_decode_jits
  pair = [NS(cnt=2, captured=object()), NS(cnt=2, captured=object())]
  model = NS(rollout_jit=NS(cnt=0, captured=None), rollout_greedy_pingpong_jits_flash_live={64:pair},
             prefill_jit=NS(cnt=9, captured=object()))
  before = {name:jit.cnt for name,jit in _decode_jits(model).items()}
  pair[0].cnt += 1
  pair[1].cnt += 1
  assert list(_used_decode_jits(model, before)) == [
    'rollout_greedy_pingpong_jits_flash_live[64][0]', 'rollout_greedy_pingpong_jits_flash_live[64][1]']


def test_capture_warms_and_observes_both_production_slots(monkeypatch):
  from types import SimpleNamespace as NS
  from extra.llm_research.decode import decode_runtime_overhead as mod
  pair = [NS(cnt=0, captured=None), NS(cnt=0, captured=None)]
  model = NS(_decode_feedback_pingpong_promoted=True, rollout_greedy_pingpong_jits=pair,
             reset_generation_state=lambda:None)
  def generation():
    for index in range(6):
      jit = pair[index % 2]
      jit.cnt += 1
      if jit.cnt >= 2: jit.captured = object()
      yield 1
  monkeypatch.setattr(mod, '_prefill', lambda *args:(generation(), 1))
  _, selected, count = mod._capture_decode_graph(model, [1]*128, 32, 3, False)
  assert count == 6
  assert len(selected) == 2
  assert all(jit.cnt == 3 and jit.captured is not None for jit in selected.values())
