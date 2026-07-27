# Tinygrad launch observer

`tinygrad.runtime.launch_observer.LaunchObserver` is an explicit, child-process
producer seam for `HCQProgram` launches. It is inert unless
`TINYGRAD_LAUNCH_SIDECAR` is set, so ordinary runtime paths do not allocate a
writer or change synchronization.

Required environment when enabled:

- `TINYGRAD_LAUNCH_SIDECAR`: output JSON path;
- `TINYGRAD_OBSERVATION_CANDIDATE_ID`: immutable candidate identity;
- `TINYGRAD_OBSERVATION_SOURCE_SHA256`: 64-character lowercase digest of the
  pinned source identity;
- `TINYGRAD_OBSERVATION_SOURCE_COMMIT` and
  `TINYGRAD_OBSERVATION_SOURCE_TREE_SHA256`: source provenance;
- `TINYGRAD_OBSERVATION_TARGET_ID`: observed target identity;
- `TINYGRAD_OBSERVATION_RUNTIME_ID`: runtime identity;
- `TINYGRAD_OBSERVATION_RUN_ID` and `TINYGRAD_OBSERVATION_ATTEMPT_ID`: run
  identities.

Set `TINYGRAD_OBSERVATION_SYNC=1` for a capture pass that requires completion
timestamps. This intentionally synchronizes after submission and must not be
used as an authoritative performance timing pass. Without the flag, the
observer remains enqueue-only and its completion timestamp is not suitable for
counter attribution.

The writer hashes the exact `HCQProgram.lib` bytes, records grid/workgroup and
dispatch identity at the real `HCQProgram.__call__` boundary, and atomically
rewrites a `tinygrad.kfd_launch_sidecar.v1` document after each completed
record. Missing source identity fails closed. No KFD device, ioctl, profiler,
or counter backend is opened by this module.

The first GPU positive control must prove that the sidecar contains real
launches from the isolated child process and that its code-object digests match
the separately captured resource artifact. Until that control passes, sidecars
remain producer observations rather than a GPU-causality claim.
