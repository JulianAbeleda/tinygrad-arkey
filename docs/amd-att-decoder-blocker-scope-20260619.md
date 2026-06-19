# AMD ATT decoder blocker scope - 2026-06-19

Purpose: scope the known blocker after T1b/T1c:

```text
tinygrad HCQ SQTT body attribution needs an external mature ATT oracle, but this ROCm install lacks a working
librocprof-trace-decoder.so.
```

This is now a tooling dependency project, not a tinygrad register sweep.

References:

- AMD ROCprofiler-SDK thread-trace docs state the flow is tracing, decoding, then visualization; `rocprofv3` handles
  tracing/decoding, and ROCprof Trace Decoder is a prerequisite installed in `/opt/rocm/lib` or passed via
  `--att-library-path`: <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.2.0/how-to/using-thread-trace.html>
- `ROCm/rocprof-trace-decoder` says the decoder transforms `.att` thread-trace data into tool-consumable output, and
  notes that source has moved to `ROCm/rocm-systems`; users on ROCm `>= 7.13` should not need a separate install:
  <https://github.com/ROCm/rocprof-trace-decoder>
- The release page confirms standalone decoder binaries exist historically, with gfx11/gfx12 fixes in earlier releases:
  <https://github.com/ROCm/rocprof-trace-decoder/releases>

## Current Facts

Local proofs:

- T1b: AQLprofile command recovery passes, but transplanting recovered `SQ_THREAD_TRACE_MASK`,
  `SQ_THREAD_TRACE_TOKEN_MASK`, and `SQ_THREAD_TRACE_CTRL` into tinygrad HCQ still produces `0` body instruction packets.
- T1c: local `rocprofv3 --att` repair is exhausted:
  - no decoder: fast fail, missing `librocprof-trace-decoder.so`;
  - `librocprofiler-sdk.so` alias: aborts during HIP initialization;
  - legacy `libatt_plugin.so` alias: hangs until timeout;
  - available apt packages do not include `librocprof-trace-decoder.so`.

External facts:

- The decoder is a separate component for ROCm 7.2-era installs.
- Upstream says the decoder source now lives under `ROCm/rocm-systems`, not the deprecated standalone repo.
- Upstream says ROCm `>= 7.13` includes it by default; this machine is ROCm `7.2.4`.

## Non-Goals

- Do not continue random SQTT register sweeps.
- Do not treat a decoder build as a decode-performance win; it is attribution tooling only.
- Do not modify `/opt/rocm-7.2.4` in place unless the install path is explicitly chosen and reversible.
- Do not require this tooling for shipped/default tinygrad behavior.
- Do not restart Track B scheduler/codegen from this scope alone; the output only decides whether Track T can name a
  precise scheduler feature.

## Success Definition

The blocker is cleared only when this command class works on a HIP control kernel:

```bash
rocprofv3 --att --att-library-path <decoder-lib-dir> -d <out> -- <hip-control>
```

Required output:

- `rocprofv3` exits `0`;
- output directory contains ATT/thread-trace payloads or decoded thread-trace files;
- payloads include wave/instruction data for the HIP control kernel;
- rerunning `extra/amd_sqtt_t1c_att_decoder_repair.py` changes verdict to `ATT_DECODER_REPAIR_PASS`.

Only after that do we proceed to the tinygrad-specific gate:

```bash
PYTHONPATH=. python3 extra/amd_sqtt_t1_body_mapping_proof.py
```

Required tinygrad gate:

- `raw_body_packet_events_top20 > 0`;
- `body_instruction_events > 0`;
- mapped instructions include q8 body instructions, not only `S_ENDPGM`.

## Track D1 - Binary Decoder Acquisition

Goal: install or unpack a known ROCprof Trace Decoder binary without rebuilding ROCm.

Steps:

1. Query the GitHub release assets for `ROCm/rocprof-trace-decoder`.
2. Select a Linux x86_64 asset compatible with ROCm 7.2-era `rocprofv3`, starting from the newest asset that advertises
   gfx11/gfx12 support.
3. Unpack into a repo-local or user-local path, for example:

   ```text
   bench/amd-scheduler-tooling-backend/att-decoder-bin/<version>/lib/librocprof-trace-decoder.so
   ```

4. Run `extra/amd_sqtt_t1c_att_decoder_repair.py` with an override for that path.
5. If pass, run a minimal `rocprofv3 --att` HIP control and archive output metadata.

Gate:

- decoder `.so` exists and exports:
  - `rocprof_trace_decoder_parse_data`
  - `rocprof_trace_decoder_get_info_string`
  - `rocprof_trace_decoder_get_status_string`
- `rocprofv3 --att` produces payloads for the HIP control.

Kill:

- no compatible release asset is available;
- decoder loads but reproduces the same no-payload/no-body behavior;
- decoder ABI is incompatible with ROCm 7.2.4 `rocprofv3`.

Expected cost: low, if an asset is available.

## Track D2 - Source Build From ROCm/rocm-systems

Goal: build `librocprof-trace-decoder.so` from the current upstream source without changing the system ROCm install.

Steps:

1. Clone or fetch `ROCm/rocm-systems`.
2. Locate `projects/rocprof-trace-decoder`.
3. Build out-of-tree with the local ROCm prefix:

   ```bash
   cmake -S projects/rocprof-trace-decoder \
     -B build-rocprof-trace-decoder \
     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.4 \
     -DCMAKE_INSTALL_PREFIX=<repo-local-install>
   cmake --build build-rocprof-trace-decoder --parallel
   cmake --install build-rocprof-trace-decoder
   ```

4. Run the D1 gate with the locally installed library path.

Gate:

- source builds without requiring a full ROCm rebuild;
- exported ABI matches `rocprofv3`'s expected symbols;
- `rocprofv3 --att` produces payloads on the HIP control.

Kill:

- build requires unreleased ROCm 7.13+ SDK headers/libraries not present locally;
- build succeeds but ABI is incompatible with ROCm 7.2.4;
- HIP control still produces no ATT payloads after decoder load.

Expected cost: medium. This is the most principled local path if binary acquisition fails.

## Track D3 - Known-Good ROCm Environment

Goal: avoid local package skew by running the same probes in a ROCm environment that ships the decoder.

Options:

- ROCm `>= 7.13` install, per upstream note that the decoder is included by default;
- AMD container or machine where `find /opt/rocm -name 'librocprof-trace-decoder*'` succeeds;
- separate W7800/gfx1100-only device exposure if using containers, to avoid unsupported-agent noise.

Gate:

- `extra/amd_sqtt_t1c_att_decoder_repair.py` passes unmodified except for path/env differences;
- `rocprofv3 --att` emits HIP control payloads;
- optional: run tinygrad HCQ proof if tinygrad can initialize in that environment.

Kill:

- environment sees multiple unsupported agents and cannot isolate gfx1100;
- ROCm ATT still emits code-object metadata but no `.att`/thread-trace payload;
- tinygrad HCQ cannot run in the environment and the oracle cannot be used for command-sequence diffing.

Expected cost: medium to high depending on access.

## Track D4 - Decoder ABI Shim Is Not Recommended

Goal: write our own `librocprof-trace-decoder.so` shim.

Verdict: do not fund as a first path.

Reason:

- The missing piece is not merely symbol names; the decoder must parse `.att` shader data and return ROCprofiler's
  expected records.
- T1c already showed aliasing incompatible libraries can abort or hang the profiled process.
- A shim that only satisfies symbol lookup would not solve body attribution.

Reopen only if upstream source build exposes a tiny, stable ABI wrapper that can be copied exactly.

## Phase Plan

### D0 - Release Asset Audit

Deliverable:

- `bench/amd-scheduler-tooling-backend/att_decoder_asset_audit.json`

Actions:

- query GitHub releases;
- list Linux assets, versions, sizes, and download URLs;
- classify compatibility with ROCm 7.2.4/gfx1100.

Gate:

- at least one candidate asset exists.

### D1 - Binary Install Probe

Deliverable:

- `bench/amd-scheduler-tooling-backend/att_decoder_binary_probe.json`

Actions:

- download/unpack selected asset into a repo-local directory;
- validate symbols with `nm -D`;
- run T1c with `--att-library-path` pointing to the asset directory.

Gate:

- `ATT_DECODER_REPAIR_PASS`.

### D2 - Source Build Probe

Deliverable:

- `bench/amd-scheduler-tooling-backend/att_decoder_source_build.json`

Actions:

- clone/fetch `ROCm/rocm-systems`;
- build only `projects/rocprof-trace-decoder`;
- install to repo-local path;
- run the same D1 gate.

Gate:

- source-built decoder passes T1c.

### D3 - Full Oracle Capture

Deliverable:

- `bench/amd-scheduler-tooling-backend/att_oracle_capture.json`

Actions:

- run HIP control with working decoder;
- parse output file inventory;
- identify raw `.att`, decoded wave records, code-object mapping, and instruction timeline availability.

Gate:

- instruction-level records exist for the control kernel.

### D4 - Tinygrad Diff Attempt

Deliverable:

- `bench/amd-scheduler-tooling-backend/att_tinygrad_diff.json`

Actions:

- compare ROCprofiler command setup to tinygrad HCQ setup:
  - AQLprofile command buffer;
  - SQTT register writes;
  - GRBM selection;
  - start/stop event ordering;
  - serialization and target CU/SIMD settings.
- implement one env-gated command-sequence patch if the diff names a precise missing command.
- rerun `t1_body_mapping_proof`.

Gate:

- q8 body packets appear in tinygrad HCQ SQTT.

Kill:

- ROCprofiler oracle works but the command sequence cannot be observed or applied to HCQ;
- patch still produces lifecycle-only traces.

## Final Decision Matrix

| outcome | decision |
|---|---|
| D1 binary passes | use binary decoder for Track T oracle; no source build needed |
| D1 fails, D2 passes | use source-built decoder; archive build recipe |
| D1/D2 fail, D3 external environment passes | run oracle externally; do not block local decode work |
| all fail | close ATT oracle as external tooling unavailable; continue with PMC/lifecycle evidence only |
| oracle passes but tinygrad body mapping still fails | Track T remains blocked; Track B is a project-level investment, not a primitive patch |

## Recommendation

Run D0-D1 first. The upstream docs explicitly point users to decoder release binaries, and the standalone repo release
page exists. If a compatible binary is available, that is the shortest path to a real external ATT oracle.

If D1 fails, run D2 from `ROCm/rocm-systems`. Do not spend more time on local aliasing or SQTT register sweeps; T1b/T1c
already closed those.
