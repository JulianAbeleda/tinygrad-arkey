# NV full-population QKV producer substrate campaign

Controlling scope:
`docs/task_workflow/input/nv-full-qkv-producer-substrate-campaign-scope-20260825.md`

Starting commit: `5e7f36945215ebc4ed2efbaf887ef241dafce7fd`

## Current status

| population | blocks | status | controlling result | booked wall |
| --- | ---: | --- | --- | ---: |
| S44 shared-Q8 Q4/Q4/Q4 | 9 | `WALL_CLOSED` | installed-policy isolated rollback, reps-7 | 0; loses 22.397 us/token |
| O44 ordinary Q4/Q4/Q4 | 9 | `WALL_CLOSED` | installed-policy isolated rollback, reps-7 | 0; flat/loses 0.862 us/token |
| S46 shared-Q8 Q4/Q4/Q6 | 8 | `WALL_CLOSED` | split-region full-grid reps-9 | 0; repaired wall loses 5.527 us/token |
| O46 ordinary Q4/Q4/Q6 | 10 | `WALL_CLOSED` | one-warp virtual-Q6 full-grid reps-7 | 0; wall loses 9.200 us/token |

`WALL_PASS` is not `INSTALLED`: S44 still requires policy/composition gates.
The campaign currently books no installed endpoint movement from this matrix.

## Phase tracker

| phase | state | exit artifact / decision |
| --- | --- | --- |
| 0. committed uniform-region substrate | `COMPLETE` | `01f63ed6e`; 19 focused tests pass |
| 1A. O44 boundary-matched discriminator | `COMPLETE` | full producer advances flash/output readiness; bare wall -3.388 us/token at reps-9 |
| 1B. O44 direct V-cache composition | `COMPLETE` | existing producer-owned cache completion composes at +6.867 us/token, reps-9 |
| 2. reusable producer specification | `COMPLETE` | typed uniform grid plus selected producer-owned cache completion |
| 3. S46 full-grid qualification | `COMPLETE` | exact/device pass; repaired reps-9 wall closed at -5.527 us/token |
| 4. O46 full-grid qualification | `COMPLETE` | exact/device pass; reps-7 wall closed at -9.200 us/token |
| 5. independent policy landing | `COMPLETE` | both candidate policies closed after load-time rollback qualification |
| 6. composed wall and fresh ledger | `COMPLETE` | no winners; clean reps-9 endpoint and fresh ledger recorded |

## Running arithmetic

Reference clean endpoint: approximately `4250.6 us/token = 235.26 tok/s`.
Reference variable endpoint: approximately `4282-4292 us/token = 233-234 tok/s`.

| quantity | value |
| --- | ---: |
| clean-regime latency needed for 240 | about 84 us/token |
| S44 booked installed recovery | 0 us/token |
| O44 booked installed recovery | 0 us/token |
| remaining after both, if O44 is unlocked | about 36.4 us/token |
| required average across 18 mixed blocks | about 2.02 us/block |

Only installed composed wall recovery may update the endpoint. Isolated and
device-ledger columns remain ceilings until then.

## Next action

Campaign complete. No full-grid recovery is booked; the retained installed
endpoint uses the previously promoted pair/cache topology.

## Phase 1 controlling evidence

- `phase1-o44-boundary/completion-readiness-analysis.json`: 72 matched block
  samples per arm.  The full producer delays Q-only readiness by 2.0 us/block,
  but advances cache readiness and flash start by 2.5-2.75 us/block and advances
  attention-output completion by 2.5 us/block.  Downstream readiness is not the
  cause of the bare wall loss.
- `phase1-o44-boundary/production-wall-rebracket-r9.json`: repaired bare
  full-producer bracket, exact hash, all nine blocks, `-3.388 us/token`.
- `phase1-o44-boundary/composed-producer-sink-profile.json`: exact hash, nine
  fewer nodes and `-32.750 us/token` GPU union for the full producer when both
  arms use producer-owned K/V cache completion.
- `phase1-o44-boundary/composed-producer-sink-wall-r9.json`: exact hash and
  confirmed `+6.867 us/token`; candidate beats both controls.  This selects the
  reusable output/completion contract.  It is a research wall pass, not yet an
  installed/default route.

## Phase 5 installed-policy adjudication

- `phase5-policy/s44-isolated-rollback-r7.json`: all nine S44 full producers
  execute with exact hashes and equal packed allocations, but lose 22.397
  us/token to the installed S44 K/V pair when O44 full-grid is disabled.
- `phase5-policy/o44-isolated-rollback-r7.json`: all nine O44 full producers
  execute with exact hashes, but lose 0.862 us/token at midpoint and fail to
  beat both installed-pair controls when S44 full-grid is disabled.
- The reciprocal composed rollback arms also fail: S44 loses 18.983 us/token
  with O44 active; O44 sits between controls with S44 active.  Both generated
  full-grid policy records therefore retain empty `promoted_targets`.

## Phase 6 final endpoint and wall ledger

- `phase6-composed/clean-installed-endpoint-r9.json`: 4.350776 ms/token =
  229.844 tok/s in the current slower session, with exact hashes, zero S44/O44
  full-grid admissions, and all 36 producer cache sinks.  The earlier fast
  demonstrated regime around 235 tok/s remains separately valid; this run does
  not overwrite it.
- `phase6-composed/final-installed-ledger.json`: 4.107250 ms GPU union,
  4.110720 ms node sum, and 3.470 us overlap.  Its 5.530389 ms profiled wall
  includes 1.423139 ms of profiling/host gap and is not a tok/s authority.
- `campaign-ledger.json` records all four population closures and zero booked
  full-grid recovery.  At the current endpoint, 240 is 184.109 us away.
- The wall-oriented reference is not 240.  Normalizing the exact 4.670534 GB
  weight stream to the demonstrated 1.627 TB/s large-body rate while retaining
  current non-weight device mass gives a 3.697 ms / 270.5 tok/s model ceiling.
  The current distance to that modeled wall is about 653 us/token.  This is a
  measured-rate ceiling, not a hard hardware lower bound: it says the remaining
  lever is broad issue/dequant service rate or numerical byte reduction, not
  another launch-only full-grid composition.

## Phase 3 S46 controlling evidence

- `phase3-s46/production-profile-split-regions-v3.json`: exact token hash and
  all eight blocks; 16 nodes removed and GPU union improves by 25.500 us/token.
- The first mixed emitter computed both Q4-K and Q6-V in every auxiliary CTA;
  its negative wall was promoted into the disjoint-region repair.  Two compiler
  walls (sibling terminal regions and a region-scoped register declaration)
  were also repaired before final judgment.
- `phase3-s46/production-wall-split-regions-r9.json`: candidate 4.355902 ms,
  controls 4.344939 and 4.355810 ms, exact hash.  It does not beat either
  control consistently and loses 5.527 us/token at the midpoint.  With
  population, output, arithmetic, topology, and cache composition accounted,
  S46 is `WALL_CLOSED` for this full-grid geometry.

## Phase 4 O46 controlling evidence

- `phase4-o46/production-profile.json`: exact token hash and all ten blocks;
  20 nodes removed and GPU union improves by 48.250 us/token.
- The one-warp adapter preserves the installed Q4 geometry and represents the
  four installed Q6 warps as four independent accumulators and shuffle trees,
  merging their totals left-to-right.  This avoids redundant Q reads but
  serializes the four Q6 partials on the per-block critical path.
- `phase4-o46/production-wall-r7.json`: candidate 4.416224 ms, controls
  4.400842 and 4.413207 ms, exact hash.  The candidate loses both controls and
  loses 9.200 us/token at the midpoint.  With the aggregate-work win and
  critical-span loss both measured, O46 is `WALL_CLOSED` for this geometry.
