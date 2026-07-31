# Metal precontract numeric failure — exhaustive scope

Date: 2026-07-31

Status: scoped, not implemented. Branch boundary: tinygrad `exp`. Does not authorize promotion to
`dev`/`master`.

Supersedes the single-cause framing in `docs/qwen3-8b-prefill-metal-numeric-failure-isolation-m1c/m1d/m1e-20260730.md`.
**There are two independent bugs.** Five hypotheses died because each explained one and left the other
standing.

---

## 1. The failure

Q4_K, `ffn_gate_up`, `(512, 12288, 4096)`, geometry `(256, 64, 32, 8, 1, 1)`, Metal, M4 10-core. The
kernel compiles, admits, dispatches, and passes every host-safety guard. It computes wrong answers.

Reference (independent numpy Q4_K decode, `packed_wmma_correctness_canary.py:41-132`): 6,081,976 of
6,291,456 output cells nonzero — **96.7%**.

Measured output: **18.74% nonzero**, `max_abs_error` 29,072, not bit-identical across identical
dispatches. AMD runs the same source correctly.

---

## 2. What is established, with citations

### 2.1 Two bugs, separated by the K-tile sweep

`docs/qwen3-8b-prefill-metal-m1g-k-sweep-20260731.json`, fixed geometry, K swept, 3 rounds each:

| K | iterations | `max_abs_error` | coverage | max inter-round diff |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 8 | 2390 | 18.747% | 732 |
| 512 | 16 | 4248 | 18.743% | 992 |
| 1024 | 32 | 7272 | 18.734% | 1208 |
| 2048 | 64 | 14632 | 18.746% | 2480 |
| 4096 | 128 | 29072 | 18.735% | 4224 |

- **Error scales linearly with iteration count.** error/iters = 299, 266, 227, 229, 227.
- **Non-determinism scales with it**: 732 → 4224.
- **Coverage does not move.** 18.734–18.747% across a 16× range.

Coverage is also invariant to wave count (M1e: `wm` = 8/4/2) and to problem size (M1e: 256³).

### 2.2 BUG A — loop-carried write-after-read. Confirmed, located.

`build_precontract_lds_stage` (`tinygrad/codegen/opt/kernel_lds.py:598-670`) is called with
`pipeline_plan=None` when `bc=1`, which forces `slot_base=0` for every K-tile iteration: **one physical
LDS window, unbuffered, reused across all iterations.** Its only synchronisation is
`UOp.barrier(producer)` (`:657`), which orders *this* iteration's producer stores → *this* iteration's
fragment reads. **Nothing orders this iteration's reads → next iteration's stores.**

Linear error growth with iteration count is exactly this hazard's signature.

Two attempts to insert a second barrier both failed to compile, on **both** targets:
`Ops.CONTRACT` is not a legal `Ops.AFTER` src-0 type and not a legal `Ops.GROUP` member
(`uop/spec.py:181-193`); wrapping the `Ops.WMMA` instead passes that check but the generic expander
cannot expand an `AFTER`-wrapped WMMA per-lane — only `renderer/isa/amd.py` has hand-written logic for
it. So **this is not a one-line fix.**

### 2.3 BUG B — 81% of outputs are exactly zero. Open. Two locations eliminated.

- **Not the store addresses.** `scratchpad/m1f_store_address_diff.py` transcribed both targets' address
  formulas and evaluated them over the entire 98,304-thread grid with NumPy: every one of 6,291,456
  output cells is hit by exactly one `(thread, slot)` pair on **both** targets. `max_hit_count =
  min_hit_count = 1`, zero gaps, zero collisions.
- **Not the accumulator.** Measured directly in the rendered sources (2026-07-31): **all 64 `buf0`
  slots are read into WMMA as accumulator initialisers and all 64 are written back from WMMA results,
  on both targets.** Metal reads at even offsets 0,2,…,62 (32 fragments × 2 elements) and writes all 64
  scalar positions; AMD does the same at its own decomposition. An earlier inference that "only 12 of 64
  accumulator slots receive WMMA output" is **false** and is retracted here.

Since every cell is stored exactly once, and every accumulator slot is filled from a WMMA result, the
cells reading as zero **were stored — with a value the WMMA produced as zero.**

**Therefore the zeros originate upstream of the accumulator: in the fragment operands the WMMA
consumes, i.e. in the LDS staging.** That is the only remaining location, and it is unexamined.

### 2.4 Why coverage cannot be explained by BUG A

A race produces *wrong* values, and its severity scales with contention and with the number of
opportunities to collide. Coverage is flat across a 16× iteration range, across wave counts, and across
problem size. **A structural gap does not scale; a race does.** They are different bugs.

---

## 3. The question BUG B reduces to

`18.735%` of 6,291,456 is 1,178,716 cells. `0.1875 = 12/64` exactly.

**Does the producer write every LDS byte that the fragment loads read?**

The same check M1f ran for global stores, run for the LDS window: enumerate the producer's write
addresses and the fragment loads' read addresses over the full thread grid, and compare the sets. If the
reads cover positions the writes never touch, those positions hold whatever the window held before —
and every fragment built from them is garbage or zero.

This is answerable **compile-only**, from the rendered sources, with no GPU.

---

## 4. Architectural boundaries

| Concern | Authority |
| --- | --- |
| LDS staging construction | `tinygrad/codegen/opt/kernel_lds.py::build_precontract_lds_stage` |
| double-buffered pipeline | `tinygrad/codegen/opt/kernel_pipeline.py::build_stage1_uop_graph` |
| rendering without execution | `Target.parse` + `to_program` (`scratchpad/m1d_confirm_c_fragment.py`) |
| correctness / coverage / determinism measurement | `extra/llm_research/prefill/metal_precontract_lane.py` |
| address-set enumeration | `scratchpad/m1f_store_address_diff.py` (the template) |

**Required reuse.** The lane is the instrument; do not write a sixth bespoke driver. M1f's brute-force
address enumeration is the template for the LDS check — same shape, different buffer.

---

## 5. Work packages

### MB0 — LDS write/read coverage (BUG B)

Prerequisite: none. **Compile-only.**

Enumerate, for both targets, the producer's LDS write addresses and the fragment loads' LDS read
addresses across the full thread grid. Report whether reads ⊆ writes, and if not, exactly which
positions are read-but-never-written and what fraction of the fragment data that represents.

Stop condition: if reads ⊆ writes on Metal, BUG B is not an LDS coverage gap. Report that; the next
candidate is the dequant values themselves (§7 item 2).

### MB1 — Fix BUG B

Prerequisite: MB0. Scope depends on MB0's answer; do not pre-commit to a fix shape.

### MB2 — Fix BUG A

Prerequisite: independent of MB0/MB1, but **measure after MB1** so the two are not conflated.

The architecturally sound fix named by the failed barrier attempt: route this branch onto the existing
software-pipelined `KernelStage1PipelinePlan` machinery, which already expresses the loop-carried
dependency correctly (a next-iteration prefetch grouped with the current accumulator update under one
closing barrier).

`bc=2` selects that path. **It is blocked by two AMD literals in that branch**
(`tinygrad/codegen/opt/postrange.py:498-504`):

```python
if len(c_axes) != 3: raise KernelOptError("buffer2 accumulator contract does not have three binary axes")
accumulator_total = factors.subtiles_m*factors.subtiles_n*8
```

Three binary C axes = 8 elements/lane = RDNA3. Metal has 2 elements/lane = one binary axis. M1d proved
this code is dead for `bc=1`; `bc=2` walks into it. Parameterise both from `tc.elements_per_thread[2]`
exactly as PG0/PG1a/PG1 did for the sibling branch — **derive, never branch on backend.**

Then measure a `bc=2` geometry. M1a's population has five legal ones under `bc*(tm+tn)*80 <= 32768`:
`(64,32)`, `(64,64)`, `(64,128)`, `(128,32)`, `(128,64)`.

### MB3 — Re-measure and qualify

Prerequisite: MB1 and MB2. Full §6 contract, plus the `wm` and K sweeps so results are directly
comparable to M1e's and M1g's tables.

---

## 6. Evidence contract

1. **Three axes, always, reported separately**: `max_abs_error`, write coverage, determinism across ≥3
   rounds. This failure was misdiagnosed for a day because coverage and error were collapsed into one
   verdict; they move independently.
2. **AMD non-regression is mandatory and structural.** `scratchpad/pg0_amd_rendered_source_equality.py`
   piped to `shasum -a 256` must begin `ce03d94bb58a`, 17 `__WMMA`. If a change legitimately moves it,
   show the diff — never silently accept a new hash. There is no AMD hardware here; execution
   non-regression cannot be shown.
3. **No `if backend == "METAL"`.** Derive from declared target facts or parameterise. Nine AMD couplings
   have been removed this way; do not add a tenth in the other direction.
4. `test/unit` carries ~114 pre-existing failures. Diff failing-test-id **sets** (111 unique ids), never
   counts.
5. **One variable per experiment.** If a second change is needed to get a pass, stop and report.
6. Every number from a command actually run. Two conclusions have been retracted in this campaign over
   fabricated figures.
7. `Device["METAL"].synchronize()` before reading any result or stopping any clock.
8. GPU work is serialised. Concurrent measurement produced 12.57 / 3.13 / 9.91 tok/s for one identical
   configuration.

---

## 7. Ranked candidates for BUG B, after MB0

1. **LDS read/write coverage gap** — MB0 tests this directly. Highest prior: it is the only untested
   location on the data path, and a structural gap is the only thing that produces a constant fraction.
2. **Dequant produces zeros for most tiles.** `dequant_tile`'s block/group addressing could resolve to
   the wrong packed bytes on Metal's decomposition, yielding zeros rather than garbage. Testable by
   dumping a staged tile and comparing against the numpy decode.
3. **Fragment lane mapping** reads positions the producer wrote under a different lane→address mapping.
   Distinguishable from 1 because the *set* of written bytes would be complete while the per-lane
   correspondence is wrong.

`0.1875 = 12/64` should be readable off whichever mechanism is responsible, and any proposed cause that
cannot produce exactly that fraction is not the cause.

---

## 8. Non-goals

- The generic-TC-opt route (`postrange.py` nested-split recovery). Independent, separately sequenced,
  and currently a *correct* but unstaged path (T4: 0.0 error, 100% coverage, deterministic, 544 GFLOPS).
- Whole-model prefill wiring or tok/s claims. This scope ends at a correct kernel.
- Promotion to `dev`/`master`, or adding rows to `PACKED_WMMA_ROUTES`.
- Chasing a matrix unit. Measured 2026-07-31: M4 has none — plain FMA reaches 3909 GFLOPS against
  `simdgroup_multiply_accumulate`'s 3781 (`docs/what-makes-a-token-fast-20260731.md` §5, §10). This work
  is worth doing because tile-granularity staging keeps dequant out of the inner loop, which on a
  single-unit machine is the entire lever — not because it reaches a faster pipe.

---

## 9. Known limitations

- **No AMD hardware.** AMD correctness is asserted from prior campaign evidence and its byte-identical
  rendered source, not re-measured here.
- **AIR is pre-register-allocation.** Disassembly shows which instructions the frontend emits; it cannot
  show register allocation, scheduling, or occupancy — those live in Apple's private backend.
- **`R = 3781 GFLOPS` was measured before the `iters` calibration bug was found** in the sibling FMA
  harness. The one-unit conclusion survives (all three FMA variants land within ±10%), but any
  "% of peak" figure resting on it should be re-derived after `R` is re-run calibrated.
- The `18.75%` coverage metric counts nonzero as written, so a cell that legitimately computes to
  exactly 0.0 is undercounted. It is a lower bound, and the reference is 96.7% nonzero, so the gap is
  real — but the metric itself is not exact.
