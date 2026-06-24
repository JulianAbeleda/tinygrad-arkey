# AMD scheduler tooling T1b ATT/AQLprofile result - 2026-06-19

Purpose: do both requested T1b paths after `NO_LOCAL_REGISTER_KNOB_BODY_MAPPING`:

1. use ROCm's mature ATT path as an external body-trace oracle;
2. recover AQLprofile's low-level SQTT command setup and transplant the plausible register values into tinygrad HCQ.

Artifacts:

- `extra/amd_sqtt_t1b_att_aqlprofile.py`
- `bench/amd-scheduler-tooling-backend/t1b_att_aqlprofile.json`
- updated `extra/amd_sqtt_t1_body_mapping_proof.py`
- updated `bench/amd-scheduler-tooling-backend/t1_body_mapping_proof.json`

## Verdict

**PARTIAL TOOLING PASS, TINYGRAD BODY MAPPING STILL BLOCKED.**

AQLprofile is installed and linkable, and it emits nonzero SQTT command buffers. But transplanting the recovered
`MASK/TOKEN/CTRL` register values into tinygrad's HCQ SQTT setup still produces zero body instruction packets.

The external `rocprofv3 --att` oracle remains blocked by this ROCm install's ATT decoder packaging/runtime path.

## SDK install status

The requested ROCprofiler SDK dev pieces were already installed under `/opt/rocm-7.2.4`:

- `rocprofiler-sdk`
- `rocprofiler-dev`
- `hsa-amd-aqlprofile`
- SDK CMake config: `/opt/rocm-7.2.4/lib/cmake/rocprofiler-sdk/rocprofiler-sdk-config.cmake`
- AQLprofile library: `/opt/rocm-7.2.4/lib/libhsa-amd-aqlprofile64.so`

The earlier "missing SDK config" read was a path issue: `/opt/rocm` does not expose all versioned package files, while
`/opt/rocm-7.2.4` does.

## Track 1 - external ATT oracle

The HIP control kernel builds and runs. `rocprofv3 --att` does not produce a usable trace:

- default path: fatal error, `rocprof-trace-decoder library path not found`;
- local alias experiment: symlinked `librocprof-trace-decoder.so -> librocprofiler-sdk.so` because the SDK library exports
  decoder symbols; this gets past lookup but segfaults during HIP initialization;
- official SDK sample build is also blocked by packaging: CMake's HIP config requires
  `/opt/rocm-7.2.4/bin/hipcc`, but this image only has `/usr/bin/hipcc`, and `/opt/rocm-7.2.4/bin` is not writable by the
  current user.

Result: **BLOCKED_EXTERNAL_ATT_ORACLE**. This is a ROCm packaging/toolchain issue, not evidence about tinygrad's trace.

## Track 2 - AQLprofile PM4/register recovery

`hsa-amd-aqlprofile` links and runs. The probe sweeps TRACE parameter sets.

Working minimal contracts:

| parameter set | `start` | `stop` | nonzero command buffer |
|---|---:|---:|---:|
| `cu_se_only` | 0 | 0 | yes |
| `cu_only` | 0 | 0 | yes |
| `no_params` | 0 | 0 | yes |

Contracts with explicit token masks fail `start` with status `4096`.

The useful recovered command stream writes the same SQTT register family tinygrad already touches:

- `SQ_THREAD_TRACE_BUF0_SIZE`
- `SQ_THREAD_TRACE_BUF0_BASE`
- `SQ_THREAD_TRACE_MASK`
- `SQ_THREAD_TRACE_TOKEN_MASK`
- `SQ_THREAD_TRACE_CTRL`
- GRBM SE selection

Representative AQLprofile start values:

| register | recovered value |
|---|---:|
| `SQ_THREAD_TRACE_MASK` | `0x30013` |
| `SQ_THREAD_TRACE_TOKEN_MASK` | `0xc080683` |
| `SQ_THREAD_TRACE_CTRL` | `0xa0423941` |

A stop-like token value also appears:

| register | recovered value |
|---|---:|
| `SQ_THREAD_TRACE_TOKEN_MASK` | `0xf0007ff` |

The legacy PM4 converter returns status `4096`, but the command buffer itself is populated and usable for register
diffing. Result: **PASS_AQLPROFILE_COMMAND_RECOVERY**.

## Transplant proof

Added env-gated raw register overrides in `tinygrad/runtime/ops_amd.py`:

- `SQTT_RAW_MASK`
- `SQTT_RAW_TOKEN_MASK`
- `SQTT_RAW_CTRL`

Default runtime behavior is unchanged when unset.

Then reran the existing body-mapping proof with two new AQLprofile-derived configs:

| config | SQTT bytes | raw body packets | mapped body instructions | pass |
|---|---:|---:|---:|---|
| baseline | 1.78 MB | 0 | 0 | no |
| detail mode | 1.76 MB | 0 | 0 | no |
| ttrace exec | 1.78 MB | 0 | 0 | no |
| detail + ttrace exec | 1.89 MB | 0 | 0 | no |
| AQLprofile start regs | 3.23 MB | 0 | 0 | no |
| AQLprofile stop-like token | 8.8 KB | 0 | 0 | no |

The AQLprofile start register set changes trace volume substantially, proving the raw values are active, but it still
only yields lifecycle-style packets (`NOP`, `WAVESTART`, `WAVEEND`, time deltas) and maps only `S_ENDPGM`. It does not
emit `INST`, `VALUINST`, `IMMEDIATE`, `VMEMEXEC`, or `ALUEXEC` body packet classes.

## Meaning

This narrows the missing piece.

It is not:

- the local decoder filter;
- the simple `SQTT_MODE` / `TTRACE_EXEC` knobs;
- the three obvious AQLprofile SQTT registers (`MASK`, `TOKEN_MASK`, `CTRL`);
- a failure to write raw register values through HCQ.

What remains is outside this bounded register transplant:

- a missing command/event sequence around trace start/stop;
- a command-buffer ordering/GRBM/predicate difference;
- a ROCprofiler SDK thread-trace service detail not represented by the simple AQLprofile TRACE setup;
- or the missing external decoder/runtime package needed to observe ROCm's full ATT setup directly.

## Decision

Do not start Track B scheduler/codegen work from SQTT body evidence yet.

The next bounded tooling step is:

1. fix the external ATT oracle packaging path, ideally with a complete ROCm ATT decoder install or a writable ROCm path
   for the official SDK sample;
2. capture/diff the full ROCprofiler command sequence against tinygrad's HCQ setup, not only the three raw registers;
3. rerun `extra/amd_sqtt_t1_body_mapping_proof.py`.

Until then, tinygrad-native HCQ SQTT remains lifecycle-only for this q8 attribution use case.
