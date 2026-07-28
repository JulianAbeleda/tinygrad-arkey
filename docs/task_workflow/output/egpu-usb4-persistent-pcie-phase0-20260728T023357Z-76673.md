# USB4 eGPU persistent PCIe service phase 0

Collected: 2026-07-28T02:33:31Z through 2026-07-28T02:33:57Z

Status: P0 complete for CPU/native implementation; hardware qualification is blocked until the target PCI endpoint enumerates.

Task owner: Codex `/root`, with low-effort agents assigned bounded native, protocol, build, and harness files.

Branch: `feature/egpu-usb4-keeper`

Worktree: `/Users/julianabeleda/worktrees/tinygrad-egpu-usb4-keeper`

Base commit: `498a640fa125ce9cade655d5cf9ed502b0a2de18`

Production worktree: `/Users/julianabeleda/env/tinygrad-arkey`, `master` at the same commit. Its recovered `extra/usbgpu/` remains untracked and untouched.

## Lock proof

- Runner: `extra/usbgpu/tools/with_gpu_lock.py`
- Advisory lock: `/tmp/gpu-bench.lock` (resolved by macOS to `/private/tmp/gpu-bench.lock`)
- Schema: `tinygrad.gpu.lock.v1`
- Recorded owner PID: `76673`
- Recorded nonce: `c6704814b0f648008cf42b3a5a6422aa`
- Focused CPU tests: `8 passed` across the lock and initial wire-fixture suites.
- All PCI, USB4, extension-state, and TinyGPU CLI observations below ran as descendants of the lock runner. No reset, install, activation, workload, sleep, or power action ran.

## Host and toolchain

| Field | Value |
|---|---|
| Host | Julian's Mac mini, Apple Silicon `arm64` |
| macOS | 26.5, build `25F71` |
| Xcode | 26.5, build `17F42` |
| DriverKit SDK | 25.5 |
| SIP | disabled |
| AC policy | system sleep 0, display sleep 0, low-power mode 0 |

## Source state

- Historical restoration commit: `a0250a41d` (`4c5e67cff^`).
- The feature worktree restores 26 bounded installer files from that commit.
- All restored behavior files match the historical commit. The only restoration-only change adds `build/` to the installer `.gitignore`.
- The production recovery differs from history only in `TinyGPUDriver.cpp` and `TinyGPUDriver.iig`, where an unreviewed keeper prototype was added. Those prototype changes were not copied into the feature worktree.
- Native source is staged but not yet committed; P1 must finish wire conformance and a clean build before it becomes implementation authority.

## Installed provenance

Installed app: `/Applications/TinyGPU.app`

| Component | Identifier | SHA-256 | Signing |
|---|---|---|---|
| App executable | `org.tinygrad.arkey.tinygpu.installer` | `35a9d85097896b73355cf025ef3617feb3aa51cb5125a54072cfd6e35ba9d8b2` | ad-hoc, no TeamIdentifier, CDHash `8a146bafe4c72faf7f5f14cc5a648a471cbbb151` |
| Driver extension executable | `org.tinygrad.arkey.tinygpu.driver2` | `d9cf53f60e1b6b969d84305e5549fe0f3904dbed72fffd1eb15df875f8dc4524` | ad-hoc, no TeamIdentifier, CDHash `3f6446cf9ec54491a2628069ed43b6088fc91188` |

Both bundles pass `codesign --verify --deep --strict`. The active arkey extension reports version `1.0.0/3` and `[activated enabled]`. A legacy `org.tinygrad.tinygpu.driver2` extension is active but disabled.

The installed binaries cannot be traced to a source commit. Historical release provisioning profiles and the Developer ID signing path are unavailable. This installation is development evidence only and cannot satisfy P5 provenance.

## Capability and topology

- No TinyGPU server process was running during collection.
- `TinyGPU status` returned success and only reported extension activation.
- `TinyGPU keepalive status` returned exit 2 with `Unknown command: keepalive`; installed native keeper status is unsupported.
- USB4 bus 0 reports the ADTLINK UT4G connected in USB4 mode at 40 Gb/s, firmware `5c.9`.
- `system_profiler SPPCIDataType` returned no target PCI device, and no `1002:744c` target was found in the locked IORegistry search.
- The RX 7900 XTX is therefore not enumerated at this baseline. This is a signal/enumeration precondition, not evidence for or against keeper behavior.
- External PSU model, rail capacity, GPU lead topology, and riser power wiring were not available from software and remain operator-supplied P6 metadata.

## Gate result

P0 passes for continued CPU-only implementation because the owner, branch, worktree, lock discipline, recovered-source state, toolchain, installed hashes, signing limitations, current capability result, and topology are recorded. Do not begin hardware qualification or install a new extension until P1-P5 pass and the target endpoint enumerates under the lock.
