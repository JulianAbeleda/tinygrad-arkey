# NV K/V overlap placement result (2026-08-22)

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`, HEAD `6570abc02`
Target: RTX 5090, `DEV=NV`, sm_120, Qwen3-8B-Q4_K_M, fixed depth 512.

Status: measurement record. No production runtime, renderer, scheduler, or
model file was changed. Every GPU arm ran as a fresh process under
`/tmp/gpu-bench.lock`. Token stream SHA-256 is identical across all arms.

## Verdict

The alternate K/V ready-placement candidate is **wall-negative**. It regresses
the landed reference, so the current `ready_placement=1` (the K/V overlap pin)
is already the better placement. Do not re-chase this placement axis.

| arm | ready_placement | tok/s median | vs landed |
| --- | ---: | ---: | ---: |
| landed reference | 1 | 210.938 | -- |
| candidate | 0 | 207.920 | -3.018 tok/s (-1.43%) |
| control A | 0 | 206.590 | -4.348 tok/s |
| control C | 0 | 205.115 | -5.823 tok/s |

The candidate and both controls hold `ready_placement=0`, and all three land
below the `ready_placement=1` reference. The regression is consistent across
arms, not a single noisy sample. Token stream SHA-256 is
`1d299b89d95fdd2667c855eea2531284007ed39435a58265ef933dad58a4771b` in every
arm, so the delta is placement, not output.

## What this means

The landed graph already carries the K/V overlap pin. Removing or moving it
serializes work that the pin overlaps, costing roughly 3-6 tok/s. This is the
complement of the overlap-ledger finding: tinygrad's overlap mass is ~6 us and
llama's is ~1128 us, but llama's overlap is 91.9-95.4% shadow. The landed pin
captures the small real overlap that is available; a different placement does
not create more real concurrency.

## Evidence

- `docs/task_workflow/evidence/nv-kv-overlap-20260822/kv_overlap_landed_ref.json`
- `docs/task_workflow/evidence/nv-kv-overlap-20260822/kv_overlap_candidate.json`
- `docs/task_workflow/evidence/nv-kv-overlap-20260822/kv_overlap_control_a.json`
- `docs/task_workflow/evidence/nv-kv-overlap-20260822/kv_overlap_control_c.json`
- `docs/task_workflow/evidence/nv-kv-overlap-20260822/kv_overlap_candidate_census.jsonl`
