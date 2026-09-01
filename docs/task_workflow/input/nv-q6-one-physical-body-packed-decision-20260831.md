# NV Q6 one-physical-body packed decision (2026-08-31)

## Decision

`PROMOTE_ONE_PHYSICAL_BODY_PACKED`

The genuine 170-CTA Stream-K route is admitted. It keeps the packed trusted-FP16 weight-scale contract and executes all one-or-two owned segments through one physical IMMA body under nested runtime ranges. It is not source-spliced and does not depend on the llama cubin.

## Gate configuration

- Fixture: the pinned real-model Q6/Q8 fixture used by the K-prefix and packed-contract gates.
- Ownership: 170 CTAs, deterministic owner-to-segment mapping, plane-major scratch slots.
- Candidate: one global kernel, one outer segment range and one inner K256-epoch range whose extent depends on the segment.
- Anchor: the accepted packed trusted-FP16 route with two source-level segment bodies.
- Reduction: unchanged deterministic standalone fixup for this gate.
- Timing: same-process alternating R31 under `flock /tmp/nv-q6-oracle-gpu.lock`.
- Correctness: trusted direct-wide reference plus bitwise candidate/anchor comparison of partials and final output.

## Correctness

| arm | max abs error | mean abs error | failing count | GPU fixup vs CPU | trusted reference |
|---|---:|---:|---:|---|---|
| duplicated packed anchor | 0.00067138671875 | 0.00002147154009435326 | 0 | bit-exact | pass |
| one physical body packed | 0.00067138671875 | 0.00002147154009435326 | 0 | bit-exact | pass |

- Candidate partial scratch is bit-exact to the anchor.
- Candidate final output is bit-exact to the anchor.
- The dynamic ownership table covers the same segment records and scratch slots as the anchor.

## Structural proof

- Exactly one segment `RANGE` and one epoch `RANGE` exist in the candidate AST.
- The epoch extent depends on the active segment.
- Segment-only stores are retained.
- Segment and epoch have distinct `END` nodes.
- The candidate lowers through `pm_final_rewrite` as one global kernel with no spliced helper bodies.
- Normalized SASS contains 256 IMMA instructions instead of the anchor's duplicated 512 and 5 barriers instead of 11.

## Normalized SASS and resources

| metric | duplicated packed anchor | one physical body packed |
|---|---:|---:|
| cubin SHA-256 | `16cffa9c89fda30d1022d346b8814f27374e72dbaff9f90b2a299d032f382625` | `1df61553f7ebb9904108c2ed14b0c256abdce067a2ae3a1bfe45fcc86a243e1f` |
| instructions | 10376 | 5192 |
| registers | 255 | 255 |
| stack bytes | 64 | 48 |
| local static bytes | 0 | 0 |
| shared static bytes | 1024 | 1024 |
| LDL / STL | 24 / 24 | 12 / 12 |
| IMMA | 512 | 256 |
| LDSM | 64 | 32 |
| LDG | 218 | 109 |
| STS | 146 | 73 |
| BAR | 11 | 5 |

The candidate has no resource regression and removes the duplicated physical body exactly at the static-instruction level.

## Locked timing

R31 medians in microseconds:

| arm | main | fixup | end-to-end pair |
|---|---:|---:|---:|
| duplicated packed anchor | 308.640 | 25.696 | 334.560 |
| one physical body packed | 246.912 | 25.088 | 271.840 |

Paired candidate-minus-anchor R31:

- Main median: -61.696 us; candidate wins 31/31.
- Fixup median: -0.736 us; candidate wins 29/31.
- End-to-end median: -62.592 us; candidate wins 31/31.
- End-to-end range: -73.504 to -60.672 us.
- Median end-to-end reduction: 18.71%.

## Compiler substrate admitted with this result

The earlier simplify change preserved nested runtime range lifecycles, but this real kernel exposed a second lexical-lifecycle error in `do_split_ends`: deriving ended ranges from `SINK(epoch).ranges` pulled the dynamic outer segment range into the inner epoch split and created a self-referential `END`/`DEFINE_REG` graph. The bounded repair uses `flatten_range` and only splits explicit lexical ranges ended by that `END`. Focused and existing regression coverage reports `19 passed, 2 skipped`, and this full candidate proves the repaired graph renders and executes through the normal rewrite pipeline.

## Commands

Focused compiler and kernel tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  test/unit/test_nested_runtime_range_lifecycle.py \
  test/unit/test_generic_tc_split_range_axis.py \
  test/unit/test_rangeify_multireduce.py \
  test/unit/test_nv_q6_oracle_streamk_single_body_packed.py
```

Result: `19 passed, 2 skipped`.

Locked full qualification:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock env PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_streamk_single_body_packed.py \
  --rounds 31 \
  --out docs/task_workflow/evidence/nv-q6-oracle-streamk-single-body-packed-20260831/result.json \
  --artifacts docs/task_workflow/evidence/nv-q6-oracle-streamk-single-body-packed-20260831/artifacts
```

Result: `PROMOTE_ONE_PHYSICAL_BODY_PACKED`.

## Evidence

- Machine-readable result: `docs/task_workflow/evidence/nv-q6-oracle-streamk-single-body-packed-20260831/result.json`
- Normalized cubin/SASS artifacts: `docs/task_workflow/evidence/nv-q6-oracle-streamk-single-body-packed-20260831/artifacts/`

## Proven, inferred, unknown

- Proven: one physical body preserves the accepted packed-contract numerics, all ownership outputs, and partial scratch bits on this fixture.
- Proven: the nested range compiler repair lowers the real dynamic outer-segment/inner-epoch graph without source splicing or bypassing final rewrite.
- Proven: one-body wins the duplicated anchor by 62.592 us median end-to-end under the locked R31 protocol.
- Inferred: most of the gain is removal of the duplicated static body and its associated instruction-cache/register-lifecycle pressure; timing alone does not attribute hardware stalls.
- Unknown: whether independent Q8 publication/prefetch or reduction changes improve this newly admitted anchor. Those remain separate gates.

## Next gate

Use the one-physical-body packed route as the sole anchor. Test Q8 late-prefetch, Q6 `d` publication, combined publication, and reduction variants independently; require trusted-reference exactness, no material resource regression, and a positive locked paired R31 result before promotion.
