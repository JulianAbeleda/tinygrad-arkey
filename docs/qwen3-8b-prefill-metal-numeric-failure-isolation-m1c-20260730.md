# M1c — isolating the cause of the Metal numeric failure

Repo `exp` @ `c9e3b9bd1`. Same dispatch M1b already qualified as real and reproducible:
Q4_K, `ffn_gate_up`, shape `(512,12288,4096)`, geometry `(256,64,32,8,1,1)`
(`tm,tn,tk,wm,wn,bc`), device METAL. Driver: `scratchpad/m1c_isolate_cause.py`, a
line-for-line reuse of `scratchpad/m1b_metal_qualification_run.py`'s payload
construction / compile / admit / guarded-execution machinery, with one addition:
the guarded-execution `readback` hook is wrapped (`dataclasses.replace`, no edits to
`extra/llm_research/prefill/guarded_execution.py`) to also stash a copy of the full
output array per round, and `run_guarded_execution` is called directly (instead of via
the `run_tinygrad_executable_guarded` convenience wrapper) so the wrapped hooks can be
passed through. `Device["METAL"].synchronize()` is called before and after every round,
exactly as in M1b.

Ran once, 1 warmup + 3 measured rounds, full arrays for `reference` (R) and each round's
`output` (A) saved to `/tmp/m1c_metal_RA_arrays.npz`. This run's `max_abs_error` was
29168.0 / 29024.0 / 29008.0 across the three rounds — consistent with M1b's originally
recorded 29120.0 / 29088.0 / 29104.0 (same order of magnitude, not bit-identical; see the
non-determinism finding below for why).

## Verdict: NOT a permutation

The multiset test is decisive on cardinality alone, before even comparing values:

- `R` (reference): 6,291,456 elements, 6,081,976 nonzero, range **[-0.1875, 52.5]**.
- `A` (actual Metal output, round 0): 6,291,456 elements, only **1,178,860** nonzero
  (18.7% of all elements), range **[-0.09375, 29168.0]**.
- `sorted(A.ravel()) == sorted(R.ravel())` → **False**. `np.count_nonzero(R) -
  np.count_nonzero(A)` = 4,903,116. A permutation preserves the multiset of values
  exactly, so it must preserve the nonzero count exactly; it does not, by a factor of
  ~5x. This alone rules out a pure lane/fragment permutation.
- Exact elementwise matches `A0 == R`: 163,611 / 6,291,456 (2.6%). Within-tolerance
  (`atol=0.02, rtol=0.02`) matches: 204,296 / 6,291,456 (3.2%).
- Value range: R is O(1-50), A's written elements span both a plausible O(1-100) band
  and a disjoint O(10,000-30,000) band (see below) — magnitude evidence against a pure
  permutation and toward a wrong/uninitialized memory read, as anticipated by the task
  brief.

## What the arrays actually show: incomplete tile coverage + garbage reads, not a coordinate relabeling

The output buffer is zero-initialized by `guarded_execution.allocate` before dispatch
(`np.zeros(reference.shape, dtype=output_dtype)`), so every zero in `A` means the kernel
never wrote that location at all — it isn't a computed value that happens to equal zero.
At the 5,112,596 positions where `A0 == 0`, `R` is essentially never zero (only 162,730
of those positions have `R == 0`); the reference is mostly nonzero everywhere by
construction (the fixture's `_decode_selected_q4` term rarely cancels exactly).

Per-tile (`tm=256, tn=64`) structure within one row of two M-tiles (rows are
0-indexed, 512 rows = 2 M-tiles of 256):

- **Rows 0-127 of each 256-row M-tile** (2 tiles → rows 0-127, 256-383): only **even**
  rows are written (64 of 128 rows), and only the **first half of every 64-wide N-tile**
  is written (columns `[0,32)` of each `[0,64)` block; columns `[32,64)` are always
  zero). Values here are plausible-magnitude and match `R`'s O(1-50) range, though most
  don't match `R` exactly at the same position.
- **Rows 128-159 of each M-tile** (rows 128-159, 384-415 globally) — exactly **one**
  32-row block, which lines up exactly with wave index 4 of 8 under this geometry's
  `wm=8` (`tm/wm = 256/8 = 32` rows per wave; wave 4 spans rows `[128,160)`) — are
  written **contiguously** (all 32 rows, not just even ones), same column pattern
  (first half of each 64-wide N-tile), but every value here is a large-magnitude
  outlier: exactly 393,216 elements (= 6144 cols/active-row × 64 rows), all with
  `|value| >= 10,000` and none in `[100, 10,000)` — a clean gap separates this cluster
  from the plausible-magnitude cluster.
- **Rows 160-255 of each M-tile** (rows 160-255, 416-511 globally): entirely
  unwritten (all zero).

So: 64 (even-row, plausible) + 32 (wave-4, huge) = 96 written rows per 256-row tile,
192 total — matches the observed count of 192 nonzero rows out of 512.

## Non-determinism: the strongest evidence against any fixed mapping

The same dispatch, same inputs, run three times back-to-back (`Device[METAL].synchronize()`
before and after each round, no intervening host mutation — `inputs_unchanged: true` in
every round's guard record):

- The three rounds are **not bit-identical**: `max(|A0-A1|) = 3904.0`,
  `max(|A0-A2|) = 3568.0`.
- Even restricting to positions that are nonzero in *both* A0 and A1 (1,178,240
  positions), 426,476 of them (36%) have different values between rounds — mean abs
  diff 126.4, max 3904.0.
- The coarse write-coverage pattern (which 192 rows, which columns) is stable across
  rounds, but even that isn't perfectly stable: 1,086 individual positions (all inside
  the "plausible" row group, e.g. rows 10, 14, 26, 30, 58, 62...) flip between written
  and unwritten across rounds.

A fixed lane/fragment permutation of a correctly computed result is deterministic by
construction — same inputs, same (wrong) index map, same output, every time. What was
measured instead is not deterministic. That, combined with the missing writes (fewer
nonzero elements than the reference) and the disjoint huge-magnitude value cluster
concentrated in exactly one wave's row range, points to a memory-safety-shaped bug —
most likely uninitialized or out-of-bounds LDS/accumulator state being read back for
part of the tile — not a lane-index bug that shuffles otherwise-correct values into the
wrong slots.

## What I could not establish

- I did not trace *which* specific LDS offset or accumulator register produces the
  wave-4 garbage, or why waves other than 4 instead leave 96 of their 256/8=32-row
  allocation entirely unwritten and only stripe every other row within their written
  half. That would require reading the rendered Metal source / disassembly for this
  exact kernel (`kernel_name` differs run to run since a fresh candidate identity isn't
  pinned here the way M1b's `docs/...json` pins one instance) and matching indices to
  thread coordinates, which is out of scope for a numeric-array diagnosis.
- I did not determine whether the missing-write region (rows 160-255, and the odd rows
  in 0-127) is a store-loop that simply never issues those iterations (undercoverage of
  `elements_per_thread`) versus a store-loop that issues them but computes an
  out-of-bounds/aliased destination address that lands outside the output buffer (and
  is silently absorbed) — both are consistent with what's observable from the host
  side.
- I did not re-run enough rounds to determine whether the coarse row/column coverage
  pattern (which positions are written at all) is fully deterministic given fixed
  geometry, or whether it too can drift given enough repetitions — only 3 rounds were
  captured, and 1,086 positions already flipped between round 0 and round 1.

## Files

- `scratchpad/m1c_isolate_cause.py` — the driver used for this run.
- `/tmp/m1c_metal_RA_arrays.npz` — full `reference`, `output_round0..2` arrays from this
  run (not committed; local scratch output, reproducible by re-running the script).
- `/tmp/m1c_metal_child_result.json` — full guard/round metadata from this run (not
  committed; local scratch output).
