# AMD scheduler tooling T1c ATT decoder repair result - 2026-06-19

Purpose: exhaust local repair candidates for the external `rocprofv3 --att` oracle after T1b found that
`librocprof-trace-decoder.so` is missing from this ROCm install.

Artifacts:

- `extra/amd_sqtt_t1c_att_decoder_repair.py`
- `bench/amd-scheduler-tooling-backend/t1c_att_decoder_repair.json`

## Verdict

**ATT_DECODER_REPAIR_BLOCKED.**

No installed package or local alias gives a usable `rocprofv3 --att` decoder path.

## What Was Tested

The probe built a standalone HIP control kernel and ran `rocprofv3 --att` with three decoder-library conditions:

| variant | result |
|---|---|
| no decoder | fast fail: `rocprof-trace-decoder library path not found` |
| `librocprofiler-sdk.so` aliased as `librocprof-trace-decoder.so` | aborts in HIP initialization |
| legacy `libatt_plugin.so` aliased as `librocprof-trace-decoder.so` | hangs until timeout |

No variant produced ATT output files.

## Package Inventory

Downloaded and inspected the available ROCm packages:

- `rocprofiler-sdk7.2.4`
- `rocprofiler-sdk-rpath7.2.4`
- `rocprofiler7.2.4`
- `rocprofiler-plugins7.2.4`

None contains `librocprof-trace-decoder.so`.

The SDK library itself contains strings showing it expects a separate decoder ABI:

- `librocprof-trace-decoder.so`
- `rocprof_trace_decoder_parse_data`
- `rocprof_trace_decoder_get_info_string`
- `rocprof_trace_decoder_get_status_string`

The legacy ATT plugin does not export that ABI. It only exports the older rocprofiler plugin entry points, so the timeout
is expected and not a viable bridge.

## Meaning

The external ATT oracle is now exhausted for the local installation.

This is not a tinygrad SQTT-register issue. It is a ROCm packaging/toolchain issue: the installed `rocprofv3` has ATT
front-end support, but the required decoder shared object is absent from the available packages.

Combined with T1b:

- AQLprofile command recovery works;
- simple AQLprofile register transplant into HCQ does not emit body instruction packets;
- external `rocprofv3 --att` cannot currently be used to capture the full mature command sequence.

## Decision

Stop spending bounded decode time on local ATT repair.

To reopen external ATT, one of these must change:

1. install a ROCm build/package that actually ships `librocprof-trace-decoder.so`;
2. obtain/build the decoder from AMD source with the expected `rocprof_trace_decoder_*` ABI;
3. run the same probe on a machine whose ROCm install has working ATT.

Until then, the decode path should not depend on SQTT body attribution. Use the existing PMC/lifecycle evidence and keep
native AMD scheduler/codegen as a project-level route, not a bounded primitive patch.
