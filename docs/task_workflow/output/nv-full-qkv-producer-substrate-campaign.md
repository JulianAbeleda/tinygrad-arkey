# NV full-population QKV producer substrate campaign

Controlling scope:
`docs/task_workflow/input/nv-full-qkv-producer-substrate-campaign-scope-20260825.md`

Starting commit: `5e7f36945215ebc4ed2efbaf887ef241dafce7fd`

## Current status

| population | blocks | status | controlling result | booked wall |
| --- | ---: | --- | --- | ---: |
| S44 shared-Q8 Q4/Q4/Q4 | 9 | `WALL_PASS` | `nv-shared-q8-qkv-producer-result-20260825.md` | +16.119 us/token |
| O44 ordinary Q4/Q4/Q4 | 9 | `ACCOUNTING_REPAIR` | `nv-ordinary-q4-qkv-full-result-20260825.md` | 0; current contract loses 27.746 us/token |
| S46 shared-Q8 Q4/Q4/Q6 | 8 | `NOT_STARTED` | — | 0 |
| O46 ordinary Q4/Q4/Q6 | 10 | `NOT_STARTED` | — | 0 |

`WALL_PASS` is not `INSTALLED`: S44 still requires policy/composition gates.
The campaign currently books no installed endpoint movement from this matrix.

## Phase tracker

| phase | state | exit artifact / decision |
| --- | --- | --- |
| 0. committed uniform-region substrate | `NOT_STARTED` | clean-checkout render/tests |
| 1A. O44 boundary-matched discriminator | `NOT_STARTED` | boundary vs readiness partition |
| 1B. O44 direct V-cache composition | `NOT_STARTED` | O44 wall pass or closure |
| 2. reusable producer specification | `NOT_STARTED` | pre/post source and topology gate |
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
| O44 trapped device-union recovery | 31.500 us/token |
| remaining after both, if O44 is unlocked | about 36.4 us/token |
| required average across 18 mixed blocks | about 2.02 us/block |

Only installed composed wall recovery may update the endpoint. Isolated and
device-ledger columns remain ceilings until then.

## Next action

Audit and commit the workgroup-uniform compiler contract, then execute the O44
boundary-matched discriminator defined in scope Phase 1A.
