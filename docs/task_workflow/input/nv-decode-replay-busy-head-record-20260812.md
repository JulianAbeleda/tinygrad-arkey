# NV decode replay busy - HEAD steady replay measurement record (P2.4)

Date: 2026-08-12
Branch: `nvidia-bringup-20260731`
Status: measurement record. Executes P2 step 4 of
`nv-decode-overlap-reopen-resource-join-scope-20260812.md` (section 5):
re-measure the tinygrad steady replay busy at HEAD with the dedicated replay
probe, giving the host-gap row a HEAD number in place of the campaign 0.883
factor estimate. This record's probe-fix + evidence commits land on top of
`713b2a020`; the measurement basis is `dcbf9932b` (the task-specified HEAD).
Runtime is byte-identical across the basis (`git diff a8b560457..HEAD --
tinygrad/` empty), so the same-session production wall remains valid at HEAD.

## Why

The gap attribution (`nv-decode-gap-attribution-same-session-20260812.md`)
priced tinygrad replay busy as ~4790 us via the campaign 0.883 factor times
the DEBUG=2 prime kernel-sum (5424.9 us), leaving a ~413 us host/gap share.
0.883 was a campaign estimate, not a HEAD measurement. The reopen scope's
P2.4 asks for a dedicated replay-probe number so the host-gap row carries a
measured HEAD value.

## Audit

- Probe: `extra/llm_research/decode/replay_overlap_probe.py`, the dedicated
  replay busy probe (PROFILE=1 + HCQ_GRAPH_PROFILE_JSON; per-graph HCQ
  timestamps vs node-sum; measured token T3, flush token T4; fused prefill
  attention OFF per house convention). Semantics unchanged by the fix below.
- Session: this workspace, RTX 5090 (driver 595.84), d512,
  Qwen3-8B-Q4_K_M, 5 HCQ groups, 596 kernel instances per token.
- Runs (both `flock /tmp/gpu-bench.lock`, 0% GPU util at lock):
  - Primary at `dcbf9932b` -> `docs/task_workflow/evidence/nv-replay-busy-head-20260812.json`
  - Rep at `713b2a020` (HEAD moved mid-session; user commit on top of the
    basis; tooling-only, runtime untouched) -> `docs/task_workflow/evidence/nv-replay-busy-head-20260812-rep.json`
- Probe fix (minimal, protocol preserved): the 08-03-era probe required one
  jsonl line per measured launch, but at HEAD the prefill replays also write
  HCQ profile lines. Fix: `capture["on"]` gates the measured token only, and
  each measured launch is aligned to its profile line by member-name prefix +
  entry count, exactly one match or fail closed.
- Probe wall (46.19 s / 48.10 s) is compile-inflated: decode graphs are
  captured during the measured token at HEAD (same artifact as the 08-03
  run). It is not a steady-token wall; busy% uses the same-session production
  wall 5.2031 ms/token, valid at HEAD per the byte-identical runtime.

## Arithmetic

Primary (dcbf9932b): node-sum 4973.2 us, span 4973.0 us, 0.0% overlap.
Rep (713b2a020): node-sum 4972.4 us, span 4972.5 us (within 0.1%).

| group | kernels | node-sum us | span us |
| --- | ---: | ---: | ---: |
| 32k | 32 | 213.9 | 213.8 |
| 64k | 64 | 488.8 | 488.8 |
| 128k | 128 | 853.4 | 853.5 |
| 256k | 256 | 2004.2 | 2004.2 |
| 116k | 116 | 1412.9 | 1412.8 |
| total | 596 | 4973.2 | 4973.0 |

- busy% = span / production wall = 4973.0 / 5203.1 = **95.6%** (95.58).
- host gap = wall - span = 5203.1 - 4973.0 = **230.1 us**.
- HEAD replay factor vs prime kernel-sum = 4973.2 / 5424.9 = **0.917**;
  the campaign 0.883 x 5424.9 = 4790.2 us busy, 412.9 us gap, 92.1% was
  pessimistic.
- llama pair (gap attribution): span 3835.2 us, median inter-replay host gap
  212.5 us, wall 4074.1 us, 94.1% busy. Same-basis host gap (wall - span):
  llama 238.9 us vs HEAD 230.1 us.
- span/node-sum = 1.0 (0.0% overlap): HCQ replay members execute strictly
  serially, no in-graph co-schedule (consistent with the closed 08-05 arms).

## Gates

- P2.4 result: HEAD replay busy **95.6%**, host gap **230.1 us**, both OBSERVED.
- vs llama: busy 95.6% vs 94.1%; host gap 230.1 us vs 238.9 us (same basis;
  +17.6 us vs the 212.5 us median). The ~413 us estimate was an artifact of
  the 0.883 factor; the measured host gap is at/below llama parity.
- Attribution impact: the 1.129 ms/token gap remains kernel-work-bound (the
  652 us class table), not host-bound; host-gap compression ceiling shrinks
  from ~174-413 us to ~230 us.
- `python3 sz.py`: PASS, 42845 / 100000 authored budgeted lines (baseline).

## Evidence

- `docs/task_workflow/evidence/nv-replay-busy-head-20260812.json` (primary, dcbf9932b)
- `docs/task_workflow/evidence/nv-replay-busy-head-20260812-rep.json` (rep, 713b2a020)
- raw runs: `/tmp/nv_replay_busy_head_20260812.json`, `/tmp/nv_replay_busy_head_20260812_rep.json`
- probe: `extra/llm_research/decode/replay_overlap_probe.py` (fix in this commit)
