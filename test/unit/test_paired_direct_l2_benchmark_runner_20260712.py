from extra.qk.prefill.paired_direct_l2_benchmark_runner_20260712 import run_paired_direct_l2_benchmark

IDENTITY = "a" * 64


def _callbacks(speedup=True):
  calls = []
  def artifact(s): return {"artifact": {"status": "pass"}, "binary_sha256": ("b" if s == "direct_l2" else "c") * 64}
  def binding(s): return {"route_binding": {"status": "pass", "storage": s}}
  def correct(s): return {"correctness": {"status": "pass"}}
  def bench(s, phase, index):
    calls.append((s, phase, index))
    if phase == "warmup": return {"samples_ms": [], "counters": {}}
    ms = 8.0 if s == "direct_l2" and speedup else 10.0
    return {"samples_ms": [ms], "counters": {g: {"status": "live"} for g in ("l2", "memory", "compute")}}
  return artifact, binding, correct, bench, calls


def test_paired_direct_l2_runner_randomizes_and_promotes_from_fake_callbacks():
  callbacks = _callbacks()
  report = run_paired_direct_l2_benchmark(role="attn_qo", shape={"m": 1},
      canonical_identity=IDENTITY, environment={"target": "fixture"},
      artifact=callbacks[0], route_binding=callbacks[1], correctness=callbacks[2],
      benchmark=callbacks[3], rounds=12, warmups=1, seed=7)
  assert report["decision"] == "promote_direct_l2"
  assert len(report["protocol"]["randomized_interleaved_order"]) == 13
  assert len(callbacks[4]) == 26
  assert report["protocol"]["dispatch"] == "external-callback-only"


def test_paired_direct_l2_runner_blocks_failed_route_prerequisite():
  artifact, binding, correct, bench, _ = _callbacks()
  def bad_binding(storage): return {"route_binding": {"status": "blocked"}}
  report = run_paired_direct_l2_benchmark(role="attn_qo", shape={"m": 1},
      canonical_identity=IDENTITY, environment={}, artifact=artifact,
      route_binding=bad_binding, correctness=correct, benchmark=bench, rounds=12)
  assert report["decision"] == "blocked"
  assert report["rows"]["direct_l2"]["route_binding"]["status"] == "blocked"

