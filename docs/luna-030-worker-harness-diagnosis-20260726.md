# LUNA-030 fixed-depth worker harness diagnosis (2026-07-26)

## Verdict

`TOOL_FAILURE` is **not** supported for the retained unprofiled 8B ctx128
worker invocation. The fixed-depth worker completed successfully; the
contrary `contract-manifest.json` and `contract-finding.md` are stale summary
artifacts and must not gate subsequent work.

This is a static/evidence diagnosis. It does not establish a new GPU runtime
fix, nor does it replace the required fresh positive controls for any later
locked/profiled row.

## Evidence order

The retained directory is:

`bench/luna-rocprofv3/20260726/LUNA-030-contract/unprofiled-8b-ctx128/`

Its primary artifacts show the following sequence (local timestamps):

| Artifact | Timestamp | Meaning |
|---|---:|---|
| `worker-authority.json` | `17:45:06.839` | Atomically published complete v2 authority artifact. |
| `stdout.log` | `17:45:06.843` | Contains the worker completion sentinel `@@DONE@@`. |
| `stderr.log` | `17:45:06.840` | Reports the ctx128 row and `artifact: .../worker-authority.json`. |
| `exit-status.txt` | `17:45:07.620` | Contains `0`; the outer wrapper did resume. |
| `contract-finding.md` | `17:45:13.309` | Incorrectly says the authority and exit-status files are missing. |
| `contract-manifest.json` | `17:45:13.310` | Repeats that incorrect earlier state. |

The primary artifact also records `route: "flash"`, the model identity, and
the exact argv. Therefore the apparent condition "model admission stdout but
no authority/exit artifact" is not a worker startup or argument-parsing
failure. It is a stale post-run classification, likely retained from the
pre-`PYTHONPATH` attempt or another incomplete observation, that was not
recomputed after the successful invocation.

## Static contract check

With the repository root on `PYTHONPATH`, the runner's `--help` exits zero and
accepts both `--warmup-decode` and required `--out`. Without that environment,
the direct-script import fails before argument parsing with
`ModuleNotFoundError: No module named 'extra'`.

The runner writes its JSON only after all measurements finish, using an
atomic temporary-file replacement, then writes `@@DONE@@` and returns zero.
Consequently, a genuinely missing authority file is a completion failure (or
an artifact-collection problem), not evidence that model admission alone
completed the runner.

## Exact known-good worker invocation

Run from any directory; paths are absolute deliberately:

```bash
env PYTHONPATH=/home/ubuntu/worktrees/luna-tinygrad-trace-8b128 \
  FLASH_DECODE_THRESHOLD=128 \
  /usr/bin/python3 /home/ubuntu/worktrees/luna-tinygrad-trace-8b128/extra/qk/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --ckpts 128 --max-context 256 --nmeas 1 --reps 1 \
  --warmup-decode 2 --chunk-size 32 \
  --out /absolute/fresh/output/worker-authority.json
```

Do not reuse an output directory. After a future GPU-authorized run, classify
worker completion only from the primary set: child exit status `0`, the
`@@DONE@@` sentinel, the JSON authority artifact, and its route fields. A
derived manifest/finding must be regenerated from that set rather than copied
from an earlier failed attempt.

## Smallest deterministic repair

No worker code change is justified by this evidence. Repair the artifact
producer/collector that generated the stale `contract-manifest.json` and
`contract-finding.md`: generate those derived files only after the worker has
returned, and derive `authority_file_observed`, `route_identity_observed`, and
the exit-status claim from the final on-disk primary artifacts. Preserve the
pre-`PYTHONPATH` failure separately instead of carrying it into the corrected
run's summary.

The next GPU action remains subject to the project lock and positive-control
requirements; this diagnosis is not that action.
