# AMD ATT decoder solution result - 2026-06-19

Purpose: execute the reopen scope from `amd-att-decoder-blocker-scope-20260619.md`.

Artifacts:

- `extra/amd_att_decoder_d0d1_binary_probe.py`
- `bench/amd-scheduler-tooling-backend/att_decoder_binary_probe.json`

## Verdict

**ATT_DECODER_BINARY_PASS.**

The external ATT oracle is now runnable for a HIP control kernel.

## What Fixed It

There were two blockers, not one:

1. `librocprof-trace-decoder.so` was missing from the local ROCm install.
2. The HIP control probes were being compiled/linked through Ubuntu HIP 5.7 (`/usr/bin/hipcc`, `/usr/include/hip`,
   `/lib/x86_64-linux-gnu/libamdhip64.so.5`) while `rocprofv3` and most runtime packages were ROCm 7.2.4.

That mixed stack caused `rocprofv3 --att` and even `--kernel-trace` to crash at `hipSetDevice`.

The working solution:

- install/repair versioned ROCm compiler package `hipcc7.2.4`;
- build the HIP control with `/opt/rocm-7.2.4/bin/hipcc`;
- force ROCm 7.2 HIP headers via a repo-local copy because `/usr/include/hip` still shadows the include search path;
- force ROCm 7.2 runtime libraries with `-L/opt/rocm-7.2.4/lib` and `-Wl,-rpath,/opt/rocm-7.2.4/lib`;
- download and use ROCprof Trace Decoder release `0.1.6`:
  - `rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.tar.gz`;
  - decoder library:
    `bench/amd-scheduler-tooling-backend/att_decoder_binary_probe_work/decoder_install_0_0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib/librocprof-trace-decoder.so`.

## Gates

| gate | result |
|---|---|
| GitHub release audit finds Linux decoder | pass |
| decoder exports `rocprof_trace_decoder_*` ABI | pass |
| HIP control links to ROCm 7.2 `libamdhip64.so.7` | pass |
| `rocprofv3 --att` exits 0 | pass |
| ATT payloads produced | pass |
| decoded UI/wave files produced | pass |

Representative output files:

- `att_control_8118_shader_engine_0_1.att`
- `att_control_8118_shader_engine_0_3.att`
- `att_control_results.json`
- `ui_output_agent_8118_dispatch_1/code.json`
- `ui_output_agent_8118_dispatch_1/se0_sm1_sl0_wv0.json`
- `ui_output_agent_8118_dispatch_3/occupancy.json`

## Source Build Check

The source path was also tested enough to de-risk it:

- `rocm-7.2.4` tag does **not** include `projects/rocprof-trace-decoder`;
- current `ROCm/rocm-systems` does include it;
- a repo-local source build with `DISABLE_COMGR=ON` succeeds and installs
  `librocprof-trace-decoder.so.0.2.0`;
- with the fixed ROCm 7.2 HIP control, the source-built decoder also produces ATT payloads.

The binary release remains preferred because it is simpler and now passes.

## Packaging Notes

Installing `hipcc7.2.4` initially hit package conflicts because versioned and unversioned ROCm packages both own files
under `/opt/rocm-7.2.4`. The package state was repaired with `--force-overwrite`, then `rocm-llvm7.2.4` postinst was
unblocked by removing pre-existing compiler symlinks so the versioned package could recreate them.

Current relevant facts:

- `/opt/rocm-7.2.4/bin/hipcc` exists;
- `hipcc7.2.4` is configured;
- `rocm-llvm7.2.4` is configured;
- `/usr/bin/hipcc` still exists and is Ubuntu HIP 5.7, so probes must call `/opt/rocm-7.2.4/bin/hipcc` explicitly.

## Meaning

T1c's "external ATT blocked" verdict is superseded by this result.

The external ROCprofiler ATT oracle is now available for HIP controls. The remaining work is no longer decoder
installation; it is using this oracle to compare ROCprofiler's working command/service path against tinygrad HCQ SQTT.

## Next Gate

Run the full oracle-to-HCQ diff:

1. archive the working ATT control output in a stable artifact;
2. parse the decoded wave/instruction files enough to prove instruction records are present;
3. compare ROCprofiler/AQLprofile setup against tinygrad HCQ setup beyond `MASK/TOKEN/CTRL`;
4. implement one env-gated command-sequence patch if the diff names a precise missing command;
5. rerun `PYTHONPATH=. python3 extra/amd_sqtt_t1_body_mapping_proof.py`.

Pass condition remains unchanged: tinygrad HCQ SQTT must emit q8 body instruction packets, not only lifecycle packets.
