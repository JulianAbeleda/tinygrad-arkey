# Gate12 hardened renderer direct-copy decision

## Verdict

`REJECT_SOURCE_SASS`. Do not run trusted correctness, fixup exactness, or R31 timing.

## Frozen artifacts

- Main anchor SHA256: `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137`.
- All-partials fixup SHA256: `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514`.
- Timing reference: `256.256 us`.
- Fresh anchor rebuild SHA256: `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137`.
- Fresh candidate cubin SHA256: `dbecf56c280a10016ea73c4406d68dcb094c22691dc6e235b216c2e831347a24`.
- Fresh candidate source SHA256: `206ebe0ea6214fccfa6c389c19e6b4e6f1d9e0fcc38557495552710555e90017`.

## Hardened focused source gate

The focused suite passed `14/14` in `16.60s`. The real Q6 source contains exactly 18 paired direct assignments at candidate source lines `1231..1248`, zero named panel-load temporaries, and the same four existing `__syncthreads()` calls as the default source.

```c
*(buf0+alu8) = *(data2_1769472+(alu244+4608));
// ...16 corresponding direct copies...
*(buf0+alu25) = *(data2_1769472+(alu244+8960));
```

The fresh build did not reuse either earlier diagnostic cubin.

## Static result

| Gate | Required | Fresh observation | Result |
|---|---:|---:|---|
| Panel1 LDG / STS | 18 / 18 | 18 / 18 | pass |
| First LDG to first STS | <=160 | 251 | fail |
| IMMA / LDSM | 256 / 32 | 256 / 32 | pass |
| LDS | 176 | 184 | fail |
| LDG / STS / STG / BAR | 109 / 73 / 64 / 4 | 109 / 73 / 64 / 4 | pass |
| I2FP / FMUL / FADD / FFMA | 1024 / 1544 / 1024 / 0 | 1024 / 1544 / 1024 / 0 | pass |
| Stack / local / LDL / STL | 0 / 0 / 0 / 0 | 32 / 0 / 8 / 8 | fail |
| Scheduling LOP3 delta | 0 | +1 (`211` to `212`) | fail |
| MEMBAR / ATOM | 0 / 0 | 0 / 0 | pass |
| Registers | <=255 | 255 | pass |

Candidate instruction count is `5168`.

The exact ordering remains `BAR 0x9820`, 18 panel loads at `0x9830..0x9940`, frozen phase-tail `FADD 0x9f80`, first store `0xa7e0`, final store `0xa910`, and publication `BAR 0xa930`. First-load ordinal `2435` to first-store ordinal `2686` is `251` instructions.

## Causal conclusion

The hardened renderer successfully changed the CUDA source to the requested direct-copy form, but the full candidate retained the rejected binary ordering, eight extra LDS, one extra LOP3, and eight local spill load/store pairs. Removing named C temporaries is not a sufficient machine scheduling or allocation control in this kernel.

## Commands and lock

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock env PYTHONPATH=. \
  .venv/bin/python -m pytest -q \
  test/unit/test_lexical_load_region.py \
  test/unit/test_nv_q6_region_load_panel1.py \
  test/unit/test_nv_q6_region_copy_panel1.py
```

```bash
timeout --signal=INT --kill-after=10s 520s \
  flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  env NV_Q6_GPU_LOCK_HELD=1 PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_region_load_panel1.py \
  --out docs/task_workflow/evidence/nv-q6-region-copy-panel1-gate12-20260831/fresh-pre-sass.json \
  --artifacts docs/task_workflow/evidence/nv-q6-region-copy-panel1-gate12-20260831/fresh-artifacts \
  --compile-bound 240
```

The flock was held for both runs and released afterward. `flock -n /tmp/nv-q6-oracle-gpu.lock true` returned `0`. No GPU workload started.
