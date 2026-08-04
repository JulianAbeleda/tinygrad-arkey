"""Hermetic CPU-only tests for the B2 capture-based multi-stream CUDAGraph lowerer.

These tests pin the pure scheduling contract of tinygrad.runtime.graph.cuda
(plan_multi_stream / cross_stream_edges) plus the no-hardware import contract:
the scheduler must be deterministic, respect the range-aware dependency DAG
(call indices are topological, producers strictly precede consumers), spread
independent work across streams, keep per-stream ordinals monotonic on the
canonical DAGs, and emit exactly the cross-stream edges the capture encoder
turns into event waits. Nothing here touches a GPU or the driver; whether
captured multi-stream graphs co-schedule is answered by the B1 probe record.
"""
import heapq, random

from tinygrad.runtime.graph import cuda

# Canonical DAGs: (preds, costs). Node order is topological by construction.
CHAIN = ([[], [0], [1], [2]], [1, 1, 1, 1])
TWO_CHAINS = ([[], [], [0], [1]], [1, 1, 1, 1])
DIAMOND = ([[], [0], [0], [1, 2]], [1, 1, 1, 1])
FAN_OUT = ([[], [0], [0], [0]], [1, 1, 1, 1])


def test_module_imports_without_hardware():
  assert hasattr(cuda, "plan_multi_stream") and hasattr(cuda, "cross_stream_edges")
  assert hasattr(cuda, "CUDAGraph")


def test_n1_plan_is_all_zeros():
  assert cuda.plan_multi_stream(4, CHAIN[0], CHAIN[1], 1) == [0, 0, 0, 0]
  assert cuda.plan_multi_stream(4, DIAMOND[0], DIAMOND[1], 1) == [0, 0, 0, 0]


def test_planner_determinism():
  preds, costs = TWO_CHAINS
  assert cuda.plan_multi_stream(4, preds, costs, 2) == cuda.plan_multi_stream(4, preds, costs, 2)


def test_planner_determinism_on_random_dag():
  rng = random.Random(42)
  n, n_streams = 12, 3
  preds = [[] for _ in range(n)]
  for j in range(n):
    preds[j] = sorted(rng.sample(range(j), rng.randint(0, min(3, j))))
  costs = [rng.randint(1, 100) for _ in range(n)]
  assert cuda.plan_multi_stream(n, preds, costs, n_streams) == \
    cuda.plan_multi_stream(n, preds, costs, n_streams)


def test_streams_are_in_range():
  streams = cuda.plan_multi_stream(4, DIAMOND[0], DIAMOND[1], 2)
  assert all(0 <= s < 2 for s in streams)


def test_chain_stays_ordered_on_one_stream():
  # Deps force a single chain; the scheduler keeps it together (tie-break lowest stream).
  assert cuda.plan_multi_stream(4, CHAIN[0], CHAIN[1], 2) == [0, 0, 0, 0]


def test_two_independent_chains_spread():
  streams = cuda.plan_multi_stream(4, TWO_CHAINS[0], TWO_CHAINS[1], 2)
  assert streams == [0, 1, 0, 1]
  independent = [(0, 1), (0, 3), (2, 1), (2, 3)]
  assert any(streams[a] != streams[b] for a, b in independent), "independent nodes must spread"


def test_diamond_streams():
  assert cuda.plan_multi_stream(4, DIAMOND[0], DIAMOND[1], 2) == [0, 0, 1, 0]


def test_fan_out_streams():
  assert cuda.plan_multi_stream(4, FAN_OUT[0], FAN_OUT[1], 2) == [0, 0, 1, 0]


def test_tie_break_prefers_lowest_stream_index():
  # All independent, equal costs: nodes 2 ties between streams and must pick 0.
  preds = [[], [], [], []]
  assert cuda.plan_multi_stream(4, preds, [1, 1, 1, 1], 2) == [0, 1, 0, 1]


def test_dependency_preservation_via_replay():
  # Replay the ready-set discipline; every call must be popped only after all
  # of its producers, and the replay must reproduce the planner's streams.
  for preds, costs in (CHAIN, TWO_CHAINS, DIAMOND, FAN_OUT):
    n = len(preds)
    streams = cuda.plan_multi_stream(n, preds, costs, 2)
    consumers = [[] for _ in range(n)]
    for j in range(n):
      for p in preds[j]: consumers[p].append(j)
    tail = [0] * n
    for j in range(n - 1, -1, -1): tail[j] = costs[j] + max((tail[c] for c in consumers[j]), default=0)
    ready = [(-tail[j], j) for j in range(n) if not preds[j]]
    heapq.heapify(ready)
    left = [len(p) for p in preds]
    assigned: set[int] = set()
    replay: list[int] = []
    while ready:
      _, j = heapq.heappop(ready)
      assert all(p in assigned for p in preds[j]), f"call {j} popped before its producers"
      assigned.add(j)
      replay.append(streams[j])
      for c in consumers[j]:
        left[c] -= 1
        if left[c] == 0: heapq.heappush(ready, (-tail[c], c))
    assert len(assigned) == n and replay == streams


def test_edges_preserved_on_canonical_dags():
  for preds, costs in (CHAIN, TWO_CHAINS, DIAMOND, FAN_OUT):
    streams = cuda.plan_multi_stream(len(preds), preds, costs, 2)
    edges = cuda.cross_stream_edges(preds, streams)
    for j, ps in enumerate(preds):
      for p in ps:
        assert p < j, "preds must be topological"
        assert 0 <= streams[p] < 2 and 0 <= streams[j] < 2
        if streams[p] != streams[j]:
          assert (p, j) in edges, f"cross edge ({p},{j}) missing"
        else:
          assert (p, j) not in edges, f"same-stream edge ({p},{j}) must not be emitted"


def test_cross_stream_edges_known_example():
  preds = [[], [0], [0], [1, 2]]
  streams = [0, 0, 1, 0]
  assert cuda.cross_stream_edges(preds, streams) == [(0, 2), (2, 3)]


def test_cross_stream_edges_dedupe_and_order():
  # Duplicate producer lists dedupe; edges are ordered by j then p.
  preds = [[], [0, 0], [0], [1, 2]]
  streams = [0, 1, 0, 0]
  assert cuda.cross_stream_edges(preds, streams) == [(0, 1), (1, 3)]


def test_per_stream_ordinals_strictly_increasing():
  for preds, costs in (CHAIN, TWO_CHAINS, DIAMOND):
    streams = cuda.plan_multi_stream(len(preds), preds, costs, 2)
    for s in range(2):
      seq = [j for j in range(len(streams)) if streams[j] == s]
      assert seq == sorted(seq), f"stream {s} ordinals not increasing: {seq}"
