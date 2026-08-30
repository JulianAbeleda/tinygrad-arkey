# NVIDIA pp512 current HCQ ledger (2026-08-29)

## Verdict

**PASS for current-route low-perturbation accounting.** This packet measures
the exact composed unroll-4 + Q4-V candidate and does not replace the
unprofiled R9 wall authority.

## Authority and protocol

- Workload: Qwen3-8B Q4_K_M, prompt 512, token 198, RTX 5090, driver 595.84.
- Route: 198 canonical producers/mains: 72 gate/up, 36 K, 72 Q/O, 18 Q4-V;
  54 FP16 overlays remain for Q6-V/down.
- Both brackets used fresh processes, `HCQ_NUM_COMPUTE=2`, the existing
  dependency cut, one warmup, three timed rounds, and `--replay-cycles 1`.
- Observer-on uses HCQ graph timestamps. Observer-off is the synchronized
  no-profile control.

## R9 observer gate

| bracket | samples (ms) | minimum (ms) | median (ms) |
|---|---|---:|---:|
| observer off | 72.118733, 72.115267, 72.104098 | 72.104098 | 72.115267 |
| observer on | 72.526412, 72.557147, 72.532443 | 72.526412 | 72.532443 |

Observer-on median overhead is `0.417176 ms` / `72.115267 ms` = **0.578%**,
below the required 2% ceiling.

## Current device ledger

The selected observer-on invocation contains **1,467/1,467 classified
intervals; unknown count 0**. Classification counts are exact: q 36, k 36,
v 18, o 36, gate 36, up 36, down 36, Flash 36, plus support/setup/output.

| accounting quantity | us |
|---|---:|
| device span | 71,962.720 |
| interval union | 71,948.672 |
| device idle | 14.048 |
| shared overlap | 0.000 |
| duplicate active charge | 0.000 |

The identity closes exactly: `interval union + device idle = device span`.
There are no overlap sets to attribute. The observer-on wall is 72,532.443 us,
so the host/graph boundary residual is 569.723 us after the device span.

| region | exclusive device time (us) |
|---|---:|
| down | 19,076.352 |
| up | 13,035.552 |
| gate | 12,796.256 |
| Q | 4,480.512 |
| O | 4,492.096 |
| V | 3,199.936 |
| Flash | 3,341.056 |
| vocabulary | 2,911.360 |
| K | 2,205.536 |
| norm/conversion | 1,424.128 |
| activation/multiply | 745.696 |
| residual/RoPE/KV support | 4,224.608 |

Historical traced rows in `nv-prefill-exact-cross-runtime-trace` predate the
current 18-role Q4-V route and are not mixed into this ledger.

## Evidence

- `docs/task_workflow/evidence/nv-prefill-current-hcq-ledger-20260829/observer-on.accounting.json`
- `docs/task_workflow/evidence/nv-prefill-current-hcq-ledger-20260829/observer-on.json`
- `docs/task_workflow/evidence/nv-prefill-current-hcq-ledger-20260829/observer-on.profile.jsonl`
- `docs/task_workflow/evidence/nv-prefill-current-hcq-ledger-20260829/observer-on.census.jsonl`
- `docs/task_workflow/evidence/nv-prefill-current-hcq-ledger-20260829/observer-off.json`
- `extra/llm_research/prefill/nv_prefill_current_hcq_ledger.py`

The route remains default-off. No B0/C0/D0/E0/F0 work was performed.
