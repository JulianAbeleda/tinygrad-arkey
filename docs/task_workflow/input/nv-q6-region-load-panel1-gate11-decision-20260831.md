# Gate11 RegionLoad Q8 panel1 decision

## Decision

`REJECT_SOURCE_SASS`. Do not run trusted correctness or R31 timing, and do not promote the candidate.

## Frozen contract

- Main anchor SHA256: `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137`.
- All-partials fixup SHA256: `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514`.
- Timing reference: `256.256 us`.
- Default builder path rebuilt byte-identically to the frozen main anchor.

## Bound API use

The isolated candidate opens one region with `before_overwrite.post_barrier_region(TRUE, workgroup_uniform=True)`, applies `q8_record[index].load().load_in_region(panel1_region)` to exactly 18 immutable-for-region global `uint32` panel1 values, includes all 18 shared publications directly in `panel1_region.end_region(*panel1_publish)`, and closes with the admitted overwrite barrier. The RegionLoad marker is outside every INDEX expression.

## Focused result

`5 passed in 1.16s` for `test/unit/test_lexical_load_region.py` and `test/unit/test_nv_q6_region_load_panel1.py` under the oracle flock.

## Exact binary chain

The existing overwrite barrier is `0x9820`. The 18 `LDG.E` instructions occupy `0x9830..0x9940` in `0x10` steps and load, in order, `R126,R127,R128,R129,R130,R131,R141,R142,R145,R146,R147,R148,R149,R150,R117,R118,R119,R124`. Their global offsets are `0x4800..0x8c00` in `0x400` steps.

The frozen phase-tail `FADD` is later at `0x9f80`. The matching 18 `STS` instructions are `0xa7e0,0xa800,0xa820..0xa910`; they consume the same registers and publish to shared offsets `0x9800..0xdc00` in `0x400` steps. The closing barrier is `0xa930`.

First LDG ordinal `2435` to first STS ordinal `2686` is `251` instructions, failing the hard `<=160` gate.

## Static gate table

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Panel1 physical LDG / logical load | 1 | 1 | pass |
| Panel1 LDG / STS | 18 / 18 | 18 / 18 | pass |
| First LDG to first STS | <=160 | 251 | fail |
| IMMA / LDSM | 256 / 32 | 256 / 32 | pass |
| LDG / STS / STG / BAR | 109 / 73 / 64 / 4 | 109 / 73 / 64 / 4 | pass |
| LDS | 176 | 184 | fail |
| I2FP / FMUL / FADD / FFMA | 1024 / 1544 / 1024 / 0 | 1024 / 1544 / 1024 / 0 | pass |
| Scheduling LOP3 delta | 0 | +1 (`211` to `212`) | fail |
| MEMBAR / ATOM | 0 / 0 | 0 / 0 | pass |
| Stack / LDL / STL | 0 / 0 / 0 | 32 / 8 / 8 | fail |
| Registers | <=255 | 255 | pass |

Candidate instructions are `5168`, versus `5136` for the anchor. Candidate cubin SHA256 is `ae5ce52319802c7c2d475ce2672ea5359f0cc7e86fca45497d26af795fbbb89a`.

## Bounded repair

The one allowed integration repair changed candidate construction from all 18 LOAD UOps followed by all 18 STORE UOps to pairwise LOAD/STORE root construction. It preserved the region contract and produced the identical candidate cubin SHA and identical SASS. UOp creation order is therefore not a scheduling lever for this gap.

## Causal finding and next implication

RegionLoad enforces the existing post-barrier source region but supplies no late dependency within the remaining phase0 accumulator fold. Ptxas emits all 18 loads immediately after `BAR 0x9820`, before the frozen `FADD 0x9f80`, and keeps them live until publication. Any next primitive must express the region's earliest scheduling boundary at the exact scalar phase-tail operation without putting the token in INDEX and without extending all 18 value lifetimes. Retrying lexical or UOp creation-order variants is not admissible.

## Commands

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock env PYTHONPATH=. \
  .venv/bin/python -m pytest -q \
  test/unit/test_lexical_load_region.py \
  test/unit/test_nv_q6_region_load_panel1.py
```

```bash
timeout --signal=INT --kill-after=10s 520s \
  flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env NV_Q6_GPU_LOCK_HELD=1 PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_region_load_panel1.py \
  --out docs/task_workflow/evidence/nv-q6-region-load-panel1-gate11-20260831/pre-sass.json \
  --artifacts docs/task_workflow/evidence/nv-q6-region-load-panel1-gate11-20260831/artifacts \
  --compile-bound 240
```

The lock was acquired for both runs and released. `flock -n /tmp/nv-q6-oracle-gpu.lock true` returned `0`. No GPU workload, correctness suite, or timing run started.
