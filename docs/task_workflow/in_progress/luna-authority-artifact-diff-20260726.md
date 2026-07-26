# Tinygrad 8B authority artifact/lifecycle comparison

## Scope and method

This is a CPU/static comparison of commits `ec770de2504fe520db4ba23eed4f2b974a958f4b`,
`cf42ca8bc4f402f10cff53c946307495fedcca9c`, and
`12482ea356c21858cf3e599355e0038d21ffbf01`.  No GPU command was run.

## Source attribution

| Item | Evidence |
| --- | --- |
| Successful ctx128 source commit | `923f6ff0237f9a46aa656a27efc778f7ac6412fc`, recorded in `8b-ctx128/commit.txt` and in `authority.json` tool path/worktree. |
| Source blob used by ctx128 | `e4ba422e7d381a0f7985deed6ea8b1925d103662` for `extra/qk/decode/decode_runtime_overhead.py`.  `ec770de25` retains this identical blob. |
| Lifecycle change | `cf42ca8b` changes only `extra/qk/decode/decode_runtime_overhead.py`, to blob `4222ffe4d2d342e254ba074201e17c58078bdf73`.  It adds atomic `*.status.json` reports at `started`, `loading_model`, `warming`, `measuring`, `writing_artifact`, and completion/exception handling. |
| ctx512 evidence commit | `12482ea35`, whose parent is `cf42ca8b`; it adds only the ctx512 lifecycle artifacts. |

The source control flow before `cf42ca8b` is the same benchmark workload.  The lifecycle change does not alter model loading, prompt construction, `_warm_depth`, W/D measurement, route selection, or benchmark flags.  Therefore it cannot by itself explain a new warmup failure; it makes the pre-existing stop observable.

## Reproduction identity comparison

| Field | ctx128 retained completion | ctx512 lifecycle stop |
| --- | --- | --- |
| Model path | `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf` | Same argv path |
| Model size/mtime | `5027783488` bytes; `1781138912862342896` ns | Not reached far enough to retain identity; same argv path |
| Model hashes | Content SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`; metadata identity `b8ef0be84bfa0588efae9fb84a3b3e5b7beb53f5620ada7d8c48bd3a26633605` | Not produced |
| Environment | `DEV=AMD`, `PYTHONPATH=.`, `TINYGRAD_PREFILL_PACKED_WMMA` unset | Identical |
| Lock/worktree | `flock -x /tmp/gpu-bench.lock`; `/home/ubuntu/worktrees/luna-tinygrad-worker-smoke` | Identical |
| Common flags | `--max-context 4608 --nmeas 1 --reps 1 --warmup-decode 2 --chunk-size 32` | Identical |
| Checkpoint | `--ckpts 128` | `--ckpts 512` |
| Completion/route | exit `0`; `@@DONE@@`; one measured row, route `sdpa`, 1021 programs/token | no exit file or authority JSON; lifecycle phase `warming`, `ctx: 512` |

The ctx128 result is expected to select `sdpa`: its retained runtime setting has a flash threshold of `512`, and route selection occurs later in `_measure_d`.  The ctx512 stop occurs in `_warm_depth`, before `_measure_d` calls `_route`; consequently there is no evidence that ctx512 selected flash, SDPA, or any failing decode kernel.

## The advertised ctx128 worker failure is not the retained completion

`authority.json` says it was created at `2026-07-26T17:49:06-0400` and records a full successful invocation.  Its stdout includes `@@DONE@@`, stderr includes the ctx128 measurement and artifact path, and `authority.exit-status.txt` is `0`.

Seven seconds later, `run-manifest.json` was created at `2026-07-26T17:49:13-0400`, but asserts the opposite: only admission stdout, no authority JSON, no exit status, and a non-completing locked shell.  `FINDINGS.md` repeats that assertion.  Both records name the same worktree and the same output path.  A single invocation cannot both atomically create the recorded authority artifact/exit status and fail to create them.  The only defensible classification is that the retained ctx128 completion files are stale from a distinct successful invocation (albeit with the same recorded source, model, and argv), while the manifest/findings describe a later failed invocation.  Thus `ec770de25` is not a valid single-run authority bundle and must not be used as a positive control.

The ctx512 lifecycle status was written at `2026-07-26T17:54:30-0400`, after the lifecycle source commit.  It proves only that the new process passed model loading and entered warmup; it does not reconcile the ctx128 bundle.

## Concrete next diagnostic

Run one locked, unprofiled fresh process only after replacing the shared output directory with a newly created run-id directory.  Keep the exact 8B model, environment, and benchmark flags, but add lifecycle reports within `_warm_depth` immediately before and after `_reset`, `_prefill`, each `next(gen)`, and `gen.close()`.  Retain the child exit status, status JSON, stdout/stderr, model content hash, source commit/blob, and route report in that one directory.  This distinguishes prompt prefill, first/second warmup decode, and generator teardown without claiming a decode route before the run reaches `_measure_d`.
