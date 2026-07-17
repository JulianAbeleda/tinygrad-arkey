from extra.qk.mmq_llama_wmma_correction_gpu_harness import BLOCKED, PROTOCOL, run_wmma_consumer_gpu


def test_gpu_adapter_is_structured_and_does_not_dispatch_during_cpu_tests(monkeypatch):
  # The adapter owns runtime invocation; this unit test only checks its public
  # protocol and that an unavailable backend becomes an auditable blocker.
  monkeypatch.setattr("tinygrad.engine.realize.get_runtime", lambda *_args: (_ for _ in ()).throw(RuntimeError("no GPU")))
  result = run_wmma_consumer_gpu()
  assert result["protocol"] == PROTOCOL and result["verdict"] == BLOCKED
  assert result["passed"] is False and "exception" in result
