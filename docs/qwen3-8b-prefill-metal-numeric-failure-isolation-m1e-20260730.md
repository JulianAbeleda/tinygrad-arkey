# M1e -- a reusable Metal precontract test lane, then a configuration sweep

Repo `exp` @ `5e21b6f80` (clean start). Same dispatch family M1b/M1c/M1d qualified: Q4_K,
`ffn_gate_up`, geometry family `(tm,tn,tk,wm,wn,bc)`, device METAL, single GPU lane held alone
throughout (no concurrent GPU work).

## Part 1 -- the lane

`extra/llm_research/prefill/metal_precontract_lane.py`. One entry point:

```python
from extra.llm_research.prefill.metal_precontract_lane import ProbeConfig, run_precontract_probe

config = ProbeConfig(quant="Q4_K", role="ffn_gate_up", shape=(512, 12288, 4096),
                      geometry=(256, 64, 32, 8, 1, 1), device="METAL", rounds=3, warmups=1)
result = run_precontract_probe(config)   # -> ProbeResult; result.to_json() for a plain dict
```

`ProbeResult` carries `status` (`"measured"` / `"skipped"` / `"error"`), `max_abs_error`,
`coverage` (written/never-written counts and fractions, value range of written cells),
`determinism` (bit-identical across `rounds`, and the max inter-round difference if not),
`compile` (`active_lds_bytes`, `threads`, `admitted`, kernel name, local/global launch size), and
`passed` (`True` only if `max_abs_error <= 0.02`, rounds are bit-identical, and coverage is 100%).

It is a generalization, not a reimplementation, of `scratchpad/m1c_isolate_cause.py`'s proven
machinery (itself M1b's driver plus one addition):

- **Payload construction** (`_payload_for_config`) is line-for-line M1c's
  `_payload_for_local_row`'s schedule-field injection (`schedule.tile`, `schedule.waves`,
  `schedule.threads`, `schedule.lds.windows/strides`, `schedule.pipeline.buffer_count`),
  generalized off of a fixed `PackedWmmaRoute` so any `(shape, geometry)` pair can be probed, not
  only a route already frozen into `tinygrad.llm.packed_wmma_prefill.PACKED_WMMA_ROUTES` (no row
  was added there by this task).
- **Compile + guarded execute** (`_child_run`, run inside a `run_isolated(...,
  start_method="spawn")` child) is M1c's exact sequence: `compile_current_prefill_program` ->
  minimal `{"passed": True, "binary_sha256": ...}` evidence -> `prepare_executable` -> a
  `readback` hook wrapped via `dataclasses.replace` to capture every round's full output array ->
  `run_guarded_execution`, with `Device[device].synchronize()` before and after every round.
- **Never patches production code.** `current_prefill_execution_adapter.prepare_current_prefill_compile`
  unconditionally AMD-ELF-disassembles the compiled binary regardless of `device`, which requires a
  ROCm `llvm-objdump`/`llvm-readelf` toolchain this machine does not have. The lane works around
  it exactly as M1b/M1c did -- calling `compile_current_prefill_program` directly and building its
  own minimal evidence -- without editing that file.
- **Admission is pure Python.** `admit_probe_config` calls `admit_current_prefill`/
  `derive_packed_weight_candidate` in the caller's own process, no `Device[...]` touched. A config
  that fails admission is returned as `status="skipped"` before `run_isolated` is ever called
  (verified directly: `test_run_precontract_probe_skips_admission_rejected_config_without_touching_gpu`
  monkeypatches `run_isolated` to fail the test if it is ever invoked for a rejected config).

`run_canary` (`packed_wmma_correctness_canary.py`) was already `device`-parameterized as of M1b
(`c9e3b9bd1`); this lane builds on that rather than forking it.

**Unit test** (no GPU): `test/unit/test_metal_precontract_lane.py`, 10 tests -- config validation,
exact schedule-field injection, admission on both the established M1c dispatch (recovers
`active_lds_bytes=25600`, matching PG2/M1b) and a deliberately tile-indivisible geometry
(`FullKernelAdmissionError: geometry_divisibility`), the admission-reject skip path end-to-end
with `run_isolated` monkeypatched to fail the test if called, and `_summarize`'s coverage/
determinism math against synthetic captured arrays (both a partial-coverage/non-deterministic
case and a full-coverage/bit-identical "passed" case). All 10 pass. Full `test/unit` suite:
114 failed / 1816 passed / 25 skipped both before and after this task's changes -- same
pre-existing failing-test-id set, no new failures (the task's own new test file contributed 0
failures).

## Part 2 -- the sweep

Sweep driver: `scratchpad/m1e_metal_precontract_sweep.py` (calls `run_precontract_probe` in a
loop; not a new compile/admit/execute driver). Calibration: before trusting the lane for new
numbers, it was run once against the exact M1c dispatch and reproduced M1c's finding closely
(`max_abs_error` 29248.0/29008.0/28960.0 vs M1c's 29168.0/29024.0/29008.0; `written_fraction`
18.7% vs M1c's 18.7%; non-deterministic, `max_inter_round_diff=3744.0` vs M1c's 3904.0/3568.0 --
same signature, not bit-identical between runs as expected of a non-deterministic fault).

All shapes/geometries below: Q4_K, `ffn_gate_up`, `qwen3_8b_q4k_m_gfx1100` profile, device METAL,
1 warmup + 3 measured rounds.

| # | Group | tm,tn,tk,wm,wn,bc | shape (m,n,k) | status | max_abs_error | coverage (written) | determinism | pass |
|---|---|---|---|---|---:|---:|---|---|
| 1 | wave_count | 256,64,32,**8**,1,1 | 512,12288,4096 | measured | 29056.0 | 18.733% | not bit-identical, max diff 4480.0 | FAIL |
| 2 | wave_count | 256,64,32,**4**,1,1 | 512,12288,4096 | measured | 28432.0 | 18.745% | not bit-identical, max diff 1664.0 | FAIL |
| 3 | wave_count | 256,64,32,**2**,1,1 | 512,12288,4096 | measured | 28352.0 | 18.745% | not bit-identical, max diff 768.0 | FAIL |
| 4 | wave_count | 256,64,32,**1**,1,1 | 512,12288,4096 | **error** | -- | -- | -- | -- |
| 5 | scale | 256,64,32,8,1,1 | 256,256,256 | measured | 2188.0 | 18.747% | not bit-identical, max diff 283.0 | FAIL |
| 6 | tile_shape | 64,32,32,8,1,1 | 512,12288,4096 | **skipped** | -- | -- | -- | -- |
| 7 | tile_shape | 128,128,32,16,1,1 | 512,12288,4096 | **skipped** | -- | -- | -- | -- |
| 8 | tile_shape | 64,128,32,8,1,2 | 512,12288,4096 | **skipped** | -- | -- | -- | -- |

Every row's `active_lds_bytes` (rows 1-3, 5): 25600. All measured rows used `rtol=atol=0.02`; a
"pass" requires `max_abs_error<=0.02` AND bit-identical rounds AND 100% coverage -- none of the
five measured configs cleared any of the three bars, let alone all three.

### No config was measured correct

**No config in this sweep was correct.** Every config the lane could actually dispatch (5 of 8)
failed the same way M1c documented: large `max_abs_error` (thousands, not the O(0.02) tolerance),
partial write coverage (~18.7%, not 100%), and non-determinism between rounds. The
correct/incorrect boundary this sweep set out to find was **not located** -- it was not crossed
by any config that could be measured.

### 1. Wave count -- inconclusive on the sharpest probe (wm=1 could not be measured)

`wm=1,2,4,8` all admitted cleanly (all four use `tm=256`, which is divisible by `wm*16` for every
`wm` in `{1,2,4,8}` under the admission gate's own AMD tensor-core dims -- no `tm`/`wm`
co-adjustment was needed, contrary to what the task brief anticipated might be necessary).

`wm=8,4,2` all measured and all failed, with two real trends across the 3 measured points:
- `max_abs_error` decreases mildly as `wm` decreases: 29056.0 -> 28432.0 -> 28352.0.
- Non-determinism shrinks sharply and monotonically as `wm` decreases:
  `max_inter_round_diff` 4480.0 -> 1664.0 -> 768.0 (roughly halving each step down).
- Write coverage stays essentially flat at ~18.7% across all three (`0.187332`, `0.187453`,
  `0.187452`) -- if the missing-write pattern scaled with wave count the way "only one wave's
  worth of writes lands" would predict, coverage would change with `wm`; it does not, to within
  the ~0.01% noise M1c already documented as inherent instability of the coverage pattern itself.

**`wm=1` -- the config that would have decided whether the fault is in multi-wave decomposition --
could not be measured.** It failed at compile time, reproducibly, 3/3 attempts (the sweep run, plus
two isolated retries with an intervening independent health-probe check that confirmed
`Device["METAL"]` was otherwise healthy in between):
```
RuntimeError: Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED.
This error occurred after multiple retries.
```
This is Apple's Metal compiler service (`XPC`) itself crashing/disconnecting while compiling this
one specific kernel source, not a numeric result and not a `run_guarded_execution` failure --
`compile_current_prefill_program` never returns a PROGRAM for this config. Because of this, **the
task's sharpest, most decisive probe is inconclusive**: the wave-count trend above is suggestive
(shrinking non-determinism as `wm` drops), but the search space does not collapse, because
whether `wm=1` computes correctly could not be determined either way.

### 2. Scale -- the failure survives at ~1% of the size

Shape `(256,256,256)` (`k=256` kept as a multiple of 256 because the canary's Q4_K fixture indexes
blocks as `n*(k//256)+k_position//256`, which requires `k//256` to be exact) at the identical
geometry `(256,64,32,8,1,1)`: `max_abs_error=2188.0` (down from ~29,000 at the full shape, but
still ~100,000x the 0.02 tolerance), coverage `18.747%` (statistically the same fraction as the
full-size shape), non-deterministic (`max_inter_round_diff=283.0`). **The failure survives scale
reduction essentially unchanged in character** -- same ~18.7% coverage signature, same
non-determinism, only the absolute error magnitude shrinks with the smaller value range.

### 3. Tile shape -- all three M1a-population tuples were rejected at admission, for a reason distinct from the numeric bug

Three tuples from M1a's 20-identity population (`docs/task_workflow/output/m1a-readiness-and-geometry-population-result-20260730.md`)
-- `(64,32,32,8,1,1)`, `(128,128,32,16,1,1)`, `(64,128,32,8,1,2)` -- were all **skipped at
admission**, all with the identical reason:
```
capability_geometry: tile must divide into whole per-wave tensor-core subtiles and K steps
```
This is a real, load-bearing finding distinct from the numeric bug under investigation. M1a's
population was legality-checked against **Metal's own** tensor-core dims (`tc.dims=(8,8,8)`,
confirmed directly in M1d's trace). But `admit_current_prefill`'s pure-Python admission path
(`extra/llm_research/runtime_specs.py:452-458`, `derive_precontract_factors`) always resolves its
tensor-core facts from `tinygrad.codegen.opt.tc.amd_rdna3` (`dims=(16,16,16)`) **regardless of the
`device` the caller intends to compile for** -- exactly the same AMD-facts-independent-of-device
pattern M1d already found for `candidate_pipeline`/`register_mode`. Under AMD's `(16,16,16)` tc,
`subtiles_m = tm/(wm*16)` and `subtiles_n = tn/(wn*16)` must both be whole; the M1c/M1b geometry
`(256,64,32,8,1,1)` satisfies this by coincidence (`256/(8*16)=2`, `64/(1*16)=4`), and so does
every `wm` in the wave-count sweep (`tm=256` is divisible by `16*wm` for `wm in {1,2,4,8}`), but
**none of M1a's 23 tuples do** (verified by admitting all 23, including both `(wm,wn)` splits
where the population lists two) -- every one of them was sized against `wm | tm/8`, not
`wm | tm/16`. So this sweep could not obtain a real GPU measurement of a differently-shaped tile
at all; every geometry the admission gate would accept for this role/shape, at this repo's current
state, is a `tm=256` variant of the one family M1b/M1c already established fails. **I did not find
or search for an alternate tile geometry outside M1a's population that both admits and differs in
tile shape from `256,64,32`** -- the task named that specific population as the probe, and I
report its outcome (uniform admission rejection) rather than substituting a different search.

## What this sweep establishes and what it does not

Established:
- The Metal numeric failure is not confined to the one exact M1b/M1c dispatch: it reproduces,
  with the same ~18.7% coverage / non-determinism signature, across three wave counts (`wm=8,4,2`)
  and at a ~1%-of-size shape. It is a property of this geometry *family*, not one specific
  `(shape,geometry)` pair.
- Non-determinism magnitude shrinks monotonically as `wm` decreases across the three points that
  could be measured (4480.0 -> 1664.0 -> 768.0), while write coverage stays flat (~18.7%) and
  `max_abs_error` decreases only mildly. This is consistent with -- but does not prove -- a
  multi-wave interaction, since the trend does not reach a clean zero/pass at any measured point.

Not established:
- **Whether `wm=1` computes correctly.** The one config that would have answered this directly
  could not be compiled, 3/3 attempts, due to a Metal compiler-service (XPC) crash unrelated (as
  far as this task could determine) to the numeric bug under investigation. Reproducing or
  isolating *that* crash (e.g. against a non-packed dense kernel, or on different macOS/Metal
  toolchain versions) was out of scope here.
- **Whether the failure changes character at a genuinely different tile shape** (`tm != 256`).
  Every legal tile shape this admission gate accepts, given the role/shape available, reduces to
  `tm=256`; M1a's differently-shaped population is real but not admissible through this repo's
  own pure-Python admission gate as it exists today. Whether that gate's hardcoded AMD tensor-core
  dims are themselves a bug (should resolve target-specific tc facts when `device="METAL"`) or
  intentional (target/capability admission is deliberately AMD-shaped independent of runtime
  device, as M1a Q1/M1d already found for other facts) was not investigated further here; no gate
  or admission criterion was weakened to work around it.
- No config in this sweep reached `max_abs_error<=0.02`; consequently there is no correct/
  incorrect boundary to report as a shape/geometry threshold -- only "every measurable point on
  this side of an untestable wall (`wm=1`) fails."

## Files

- `extra/llm_research/prefill/metal_precontract_lane.py` -- the committed, reusable lane (Part 1).
- `test/unit/test_metal_precontract_lane.py` -- its no-GPU unit tests (10 tests, all pass).
- `scratchpad/m1e_metal_precontract_sweep.py` -- the Part 2 sweep driver (calls the lane 8 times).
- `scratchpad/m1e_calibration_check.py` -- one-off calibration run confirming the lane reproduces
  M1c's result for the identical dispatch, before trusting it for new configs.
- `scratchpad/m1e_wm1_retry.py` -- the isolated wm=1 retry driver (run twice more after the sweep;
  identical `XPC_ERROR_CONNECTION_INTERRUPTED` both times).
- `/tmp/m1e_metal_precontract_sweep_results.json` -- full structured results from the sweep run
  (not committed; local scratch output, reproducible by re-running the script).
