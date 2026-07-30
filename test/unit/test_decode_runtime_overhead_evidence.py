from extra.llm_research.decode.decode_runtime_overhead import _token_evidence


def test_complete_token_ids_are_opt_in_for_cross_runtime_correctness():
  compact = _token_evidence([1, 2, 3])
  complete = _token_evidence([1, 2, 3], include_token_ids=True)
  assert "token_ids" not in compact
  assert complete["token_ids"] == [1, 2, 3]
  assert complete["sha256"] == compact["sha256"]
