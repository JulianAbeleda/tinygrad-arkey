# SOLUTION SCOPE — close the host-bound prefill gap (tinygrad 1511 -> llama 3086)

## Diagnosis (established, decisive)
Prefill is HOST-BOUND (GPU-clock-invariant: 324->2331 MHz = 0 speedup). GPU idle most of the time; wall dominated
by the host. cProfile of the warm replay: the dominant cost is `_HCQSignal.wait` spin-polling `self.value`.

## Root cause (grounded in code)
`hcq.py:260  value = self.base_buf.cpu_view().view(0,8,'Q')[0]` -- `.view()` constructs a FRESH `MMIOInterface`
(`to_mv(addr,nbytes).cast(fmt)`, line 16/20) on EVERY read. The `wait` loop (line 296) calls `.value` ~483K
times/forward -> ~170ms/forward (>50% of the ~333ms wall) in pure memoryview-creation (cProfile: value+to_mv+
view+__init__). The signal address is FIXED -> the view should be created once and reused.

## Fix (candidate #1, ~5 lines, runtime)
In `_HCQSignal.wait`: hoist `view = self.base_buf.cpu_view().view(0,8,'Q')` before the loop; read `view[0]` in the
loop instead of `self.value`. Eliminates per-poll allocation.

## Validation plan
- P0 (decisive, cheap): apply the fix, measure prefill pp512 before/after (best-of). Prefill is clock-invariant ->
  before/after is NOT clock-confounded. Expect: if poll-churn is on the critical path (host-bound), wall drops ~170ms
  -> ~1511 -> toward ~3000 tok/s. If ~0 change -> the churn was overlapped with GPU work and the host bottleneck is
  elsewhere (_sleep granularity / Python / scheduling) -> iterate.
- P1 (correctness): signal semantics identical (same reads, no allocation change) -> byte-identical generation.
- P2 (decode): the wait is SHARED HCQ runtime -> decode (also busy-waits) benefits too. BUT decode is Codex's track
  -> coordinate; verify decode unaffected/improved, don't regress W==D.
- P3 (if #1 insufficient): pipeline prefill chunks (submit chunk N+1 before waiting on N -> overlap host/GPU; uses
  the 2nd compute ring AMD_COMPUTE_RINGS=2), and/or faster _sleep poll granularity.

## Risks
- The poll-churn may be OVERLAPPED with GPU (then no wall win) -- clock-invariance argues against this, but P0
  settles it empirically.
- SHARED runtime (all HCQ: decode, every model) -> must not break correctness; coordinate with Codex (decode).
- Win may be partial (other host overhead remains) -> P3 fallback.

## Why this is the right lever
It's the measured dominant host cost (~170ms/forward), a tiny runtime fix, clock-invariance says it's on the
critical path, and it's HIGH-LEVERAGE (helps decode + every HCQ model, not just prefill). Closing it could bring
prefill to ~llama parity AND help decode -- the first cross-cutting win of the campaign.
