# NV installed-island Phase 0 provenance and no-change gate

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`
GPU: RTX 5090, UUID `GPU-c800ade9-21ea-2e55-f75c-6d7a458fb186`

Evidence: `docs/task_workflow/evidence/nv-installed-islands-20260822/phase0/`

## Verdict

`MEASURED` Phase 0 passes. Every Section 3 locked artifact reconstructs
byte-for-byte or value-for-value, and no prohibited source path changed.
GPU work may proceed to Phase 1.

## Pass gate

| gate | result |
| --- | --- |
| role census byte-identical | `PASS` (regen SHA equals retained SHA) |
| Q/K cubin hashes match | `PASS` (`sha256sum -c` exit 0, 10 files OK) |
| HCQ slope hashes match | `PASS` (`sha256sum -c` exit 0, 2 files OK) |
| plain no-op slope ~0.649 us | `PASS` (0.649032, R2 0.999807) |
| Q/K slope ~1.698 us | `PASS` (1.697644, R2 0.999989) |
| no unauthorized source path changed | `PASS` |

## Reconstructed calibration facts

`MEASURED` from retained raw artifacts, recomputed this session:

```text
Q exact body               1.190 us mean / 1.184 us median
K exact body               1.1962 us mean / 1.184 us median
clean chained Q/K HCQ      1.697644 us/kernel
faithful per-kernel HCQ    1.696 us/kernel median
plain no-op HCQ floor      0.649032 us/kernel
production Q/K interval    2.5 us/kernel median (retained capture)
```

The accepted prior conclusions are preserved as-is:

```text
Q/K compiler-body hypothesis: rejected
global ~1.4 us fixed-HCQ-tax hypothesis: rejected
remaining Q/K installation mechanism: unmeasured
```

## Reconstruction detail

`MEASURED` The role census script regenerated
`role-census.json` with SHA
`0326f0d21e10059a92196a439431f5bd58fb04353a6b20d972e94b3cece494cf`,
byte-identical to the retained artifact. Census closure reproduces
tinygrad node_sum 4677.92 us, llama node_sum 3878.25 us, delta 799.67 us,
named-nine sum 646.84 us, remainder 152.83 us.

`MEASURED` The HCQ OLS slopes were recomputed from the raw
`nv_hcq_dispatch_slope.json` rows (n vs median drain), not trusted from the
reported fit. All four retained slopes match to six decimals; R-squared is
computed and stored.

`MEASURED` Q/K exact-cubin artifacts all pass their retained SHA. The
retained `exact-cubin.json` body figures (Q 1.190 / K 1.1962 us) are the
values used below and were not re-derived from scratch in Phase 0.

## Clock lock

`MEASURED` `2850 MHz` is a supported graphics clock and `14001 MHz` is the
supported memory clock on this device. `sudo` is passwordless. The
established clock-lock protocol is therefore:

```text
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 2850
sudo nvidia-smi -lmc 14001
```

It is applied at the start of Phase 1 and reset after each measurement
bracket. Persistence mode is currently disabled and will be re-enabled only
for the duration of locked measurement.

## Pre-existing dirty state

`MEASURED` Seven tracked files are modified before this scope began and are
not this scope's work: `tinygrad/engine/jit.py`, `tinygrad/llm/decode_routes.py`,
`tinygrad/renderer/cuda.py`, `tinygrad/runtime/graph/hcq.py`,
`tinygrad/runtime/ops_nv.py`,
`extra/llm_research/decode/full_token_dag_capture.py`, and
`extra/llm_research/decode/nv_norm_native_wall_ab.py`. `git diff --check`
returns exit 0.

## Ledger snapshot

```text
node_sum   = 4677.920 us (tinygrad) / 3878.254 us (llama PDL-off)
union      = 4671.500 us (tinygrad, retained)
overlap    = 6.420 us (tinygrad, retained)
wall       = 4747.530 us (prior fresh control; replaced by Phase 1)
host_gap   = 76.030 us (prior, replaced by Phase 1)
useful_body = unmeasured
booked_recovery = 0.000 us
remaining_to_240 = 580.833 us (prior control) -> updated in Phase 1
```
