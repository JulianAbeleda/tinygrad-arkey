# NV full-population QKV producer substrate campaign

Controlling scope:
`docs/task_workflow/input/nv-full-qkv-producer-substrate-campaign-scope-20260825.md`

Starting commit: `5e7f36945215ebc4ed2efbaf887ef241dafce7fd`

## Current status

| population | blocks | status | controlling result | booked wall |
| --- | ---: | --- | --- | ---: |
| S44 shared-Q8 Q4/Q4/Q4 | 9 | `WALL_PASS` | `nv-shared-q8-qkv-producer-result-20260825.md` | +16.119 us/token |
| O44 ordinary Q4/Q4/Q4 | 9 | `WALL_PASS` | composed producer/cache confirmation, reps-9 | +6.867 us/token; research composition |
| S46 shared-Q8 Q4/Q4/Q6 | 8 | `NOT_STARTED` | — | 0 |
| O46 ordinary Q4/Q4/Q6 | 10 | `NOT_STARTED` | — | 0 |

`WALL_PASS` is not `INSTALLED`: S44 still requires policy/composition gates.
The campaign currently books no installed endpoint movement from this matrix.

## Phase tracker

| phase | state | exit artifact / decision |
| --- | --- | --- |
| 0. committed uniform-region substrate | `COMPLETE` | `01f63ed6e`; 19 focused tests pass |
| 1A. O44 boundary-matched discriminator | `COMPLETE` | full producer advances flash/output readiness; bare wall -3.388 us/token at reps-9 |
| 1B. O44 direct V-cache composition | `COMPLETE` | existing producer-owned cache completion composes at +6.867 us/token, reps-9 |
| 2. reusable producer specification | `IN_PROGRESS` | selected contract: full producer plus producer-owned cache completion |
| 3. S46 full-grid qualification | `NOT_STARTED` | wall pass or closure |
| 4. O46 full-grid qualification | `NOT_STARTED` | wall pass or closure |
| 5. independent policy landing | `NOT_STARTED` | rollback-qualified installed rows |
| 6. composed wall and fresh ledger | `NOT_STARTED` | reps>=9 composition and ledger |

## Running arithmetic

Reference clean endpoint: approximately `4250.6 us/token = 235.26 tok/s`.
Reference variable endpoint: approximately `4282-4292 us/token = 233-234 tok/s`.

| quantity | value |
| --- | ---: |
| clean-regime latency needed for 240 | about 84 us/token |
| S44 measured wall recovery | 16.119 us/token |
| O44 composed wall recovery | 6.867 us/token |
| remaining after both, if O44 is unlocked | about 36.4 us/token |
| required average across 18 mixed blocks | about 2.02 us/block |

Only installed composed wall recovery may update the endpoint. Isolated and
device-ledger columns remain ceilings until then.

## Next action

Factor the selected full-producer plus producer-owned cache-completion contract,
then implement and microgate the S46 shared-Q8 Q4/Q4/Q6 population.

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
