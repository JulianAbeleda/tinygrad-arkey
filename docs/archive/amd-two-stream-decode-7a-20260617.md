# Phase 7a — two-stream decode throughput: BLOCKED by the dispatch path (report the missing API)

Phase 6 recommended concurrent-stream throughput as the multi-ring primitive's best payoff. Phase 7a set out to
measure two same-process decode streams on two rings. **Per the plan's explicit instruction — "if routing a
decode stream to a ring is blocked by the global-timeline serialization, stop and report the exact missing API,
do not silently measure two streams that still serialize" — this is that report. It is blocked, by the dispatch
path, NOT the hardware.** No fake throughput number was produced. `extra/amd_two_stream_decode_probe.py`,
`bench/amd-two-stream-decode-probe/`.

## Evidence

1. **No per-stream ring-routing API (clean, primary).** A real Qwen3-8B decode run with `AMD_COMPUTE_RINGS=2`
   leaves `dev.compute_queues == [0]` — **ring 1 is never created or used.** The dispatch path
   (`HCQProgram.__call__`, hcq.py:374; profile path 307/313) hardcodes `unwrap(dev.hw_compute_queue_t)()`, which
   builds a ring-0 queue. Nothing in the model / TinyJit / GraphRunner path can pin a stream's kernels to ring 1.
2. **Shared global timeline serializes dispatch (directional).** Every dispatched kernel does
   `.wait(dev.timeline_signal, v-1)` then `.signal(dev.timeline_signal, next)` on the device's **single**
   `timeline_signal` (hcq.py:417). Two streams share it, so stream B's kernels wait on stream A's timeline value
   — serial even if routed. The probe shows HCQProgram-style shared-timeline dispatch is ~1.46× slower than
   independent per-stream signals even across two rings (directional only — the shared mode also issues more
   host submits; the clean fact is #1).
3. **Hardware is willing.** Phases 3–5 already proved two rings overlap 2.00× and order correctly via a *custom*
   submission path that bypasses the global timeline and selects rings explicitly. So the limit is purely the
   normal dispatch path, not the GPU.

## The exact missing APIs (what Phase 7 would require)

1. **Per-stream / per-graph compute-ring selection.** A device-level "active compute ring" (e.g. a `ContextVar`
   the dispatch reads) or a TinyJit/stream ring binding, so an entire stream's kernels target ring N. Today the
   queue builder is hardcoded to ring 0 in `HCQProgram.__call__` and the profile/graph paths.
2. **Independent per-stream timelines (sync domains).** The device has one global `timeline_signal` that every
   op waits/signals; two concurrent streams need independent timeline signals so they don't serialize through a
   shared monotonic value. This is the deeper change — even with ring routing, the shared timeline serializes.
3. **(Secondary) Concurrent model state.** One `cache_kv` per model; two streams need two state/cache sets (two
   model instances or a batched cache) — a model-layer concern, not runtime.

## Decision

**Stop here; do not build the dispatch rewrite.** Same-process two-stream overlap requires a new concurrent-
dispatch path (per-stream ring + per-stream timeline) — a substantial `[runtime]` change to tinygrad's core HCQ
dispatch, well beyond the contained queue-selection primitive. That is exactly the kind of deep core-runtime
surgery to gate behind a proven, funded need.

The primitive's overlap is **already realized cross-process** (separate processes = separate devices/timelines/
rings — the original +32% finding). For a server/throughput use case, running N inference processes is the
low-risk path that needs zero core-dispatch changes. In-process two-stream overlap is a real but separate arc
(the two APIs above), to open only if cross-process isolation is unacceptable for the target deployment.

## Where the multi-ring arc lands (Phases 0–7a)

- **Primitive: proven & shipped, default off.** P1 NFC queue-idx, P2 opt-in 2nd ring (KFD+non-AQL gated), P3
  2.00× independent overlap, P4 cross-ring deps (both directions + copy queue), P5 DAG scheduler 1.59×.
- **Decode payoff: scoped honestly.** Single-stream latency is Amdahl-bounded (P6: host 55% + bandwidth-bound
  GEMVs, ~1.05–1.15× ceiling). Two-stream throughput is the better use but is **blocked in-process** by the
  dispatch path (this phase) — realizable cross-process today, or via the two missing APIs.
- **Net:** a real, de-risked, tested runtime primitive sitting behind `AMD_COMPUTE_RINGS`, with its decode
  applications precisely mapped and gated. No model/default behavior changed anywhere in the arc.

Anchors: `amd-multiring-compute-primitive-20260617.md`, `amd-decode-overlap-scheduler-plan-20260617.md`,
`bench/amd-two-stream-decode-probe/`. Memory: `amd-multiring-compute`.
