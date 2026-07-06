import pytest

from extra.qk import prefill_graph_gemm_route as route


def test_prefill_pipe_role_selective_zero_is_retired(monkeypatch):
  route._resolve_schedule.cache_clear()
  monkeypatch.setenv("PREFILL_PIPE_ROLE_SELECTIVE", "0")
  with pytest.raises(RuntimeError, match="PREFILL_PIPE_ROLE_SELECTIVE=0 global-pipe rollback was retired"):
    route._resolve_schedule(4096, 4096)
  route._resolve_schedule.cache_clear()
