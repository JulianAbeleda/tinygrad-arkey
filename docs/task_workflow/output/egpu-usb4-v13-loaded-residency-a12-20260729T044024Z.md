# TinyGPU v13 A12 single-process loaded-residency result

Collected: 2026-07-29T04:35:14Z through 2026-07-29T04:40:31Z

Status: **failed before the loaded interval**. Qwen3-8B memory admission passed
after two fail-closed host-fact corrections, but the third attempt lost the
USB4 PCIe tunnel while transferring the model at 4.68 GB used. No token was
produced and the required immediate post-residency A2 was not run.

## Authority and provenance

- Scope: `docs/task_workflow/input/egpu-usb4-persistent-pcie-service-scope-20260727.md`.
- A12 runner: `d6b36baf7bee3c21f7908679604346bb90bc1aad`.
- Large-allocation granularity correction: `37b6e759c3bd04cf39a0185819cb43d722aa61b5`.
- Live allocator-memory correction and final run source:
  `bef584e280f335729dd138d0ae685dc8029edd29`.
- Installed v13 source: `8f7afc45f274f8c2a4ffbeee286684a2a1013c42`.
- Installed/live app SHA-256:
  `e4ba9c1413afa87039d9d306f6c84540f05639022e3849c90da9dc2cb6643798`.
- Installed/live DEXT SHA-256:
  `0e343ec11652a426dcbfde8825ceb3581eae9ff789ab425b881ae17da0f461dc`.
- Model: `/Users/julianabeleda/Models/Qwen3-8B-Q4_K_M.gguf`,
  `5027783488` bytes, SHA-256
  `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- The operator reported performing the approved reset before admission. The
  agent performed no reset, replug, install, provider termination, or power
  action.

The installed build inputs remained git-equivalent descendants and the live
binary hashes matched the audited install transcript on every attempt.

## Attempts

1. `04:35:14Z`: failed before model load because direct TinyGPU AMD did not
   publish its existing 2 MiB large-allocation granularity. Zero tokens, no
   loaded sample, same healthy provider, clean resource teardown.
2. `04:38:03Z`: the granularity check passed; admission then failed because
   macOS has no `rocm-smi` total/free VRAM source. Zero tokens, no loaded sample,
   same healthy provider, clean resource teardown.
3. `04:40:13Z`: the direct AMD allocator's live TLSF heap supplied exact
   allocatable total/free bytes. Admission passed:

   ```text
   max_context=explicit -> 1024
   free 25.6GB
   budget 25.5GB
   weights 5.0GB
   KV 0.15GB/1k
   prefill-peak 0.07GB/1k
   ```

   TinyGPU accepted repeated 2 MiB DMA preparations. Before first-token
   realization completed, the process returned:

   ```text
   MemoryError: Allocation of 128 B failed on AMD. Used: 4.68 GB
   ... finalizer ... BrokenPipeError: [Errno 32] Broken pipe
   ```

The wrapper's allocation message is not the root cause. The exact kernel
timeline below establishes an upstream tunnel collapse.

## Fatal kernel timeline

The third child ran from `00:40:13.167843-0400` through
`00:40:23.644677-0400`.

- `00:40:20.503039-0400`: both ACIO lanes began a 32-event Gen2/3 burst.
  Counts within the child interval were:

  | Lane | 82 | 83 | 84 | 87 | 88 |
  |---:|---:|---:|---:|---:|---:|
  | 0 | 1 | 1 | 3 | 3 | 8 |
  | 1 | 1 | 1 | 11 | 3 | 0 |

- `00:40:20.529381-0400`: the upstream bridge reported
  `linkStatus 0x0000`.
- `00:40:20.529481-0400`: IOPCIFamily marked AMD `1002:744c`,
  `1002:ab30`, `1002:7446`, and `1002:7444` dead, along with the downstream
  bridge chain.
- `00:40:20.530190-0400`: macOS force-closed old TinyGPU service
  `0x100001f9b`.
- `00:40:20.604289-0400`: IOThunderboltFamily requested port power-down.
- `00:40:20.655059-0400`: the tunneled PCIe transport was removed.
- `00:40:22.206439-0400`: `AppleTunneledPCIE::setPowerState` reached 0.
- `00:40:22.775015-0400`: tunnel retraining requested power state 2.
- `00:40:24.935542-0400`: a fresh v13 DEXT server launched.
- `00:40:24.939549-0400`: fresh TinyGPU service `0x1000020c6` published.

The pre-run sample on the old service was healthy with keeper `301/301`, zero
failures, BAR5 active, full power confirmed, PCI command `0 -> 7`, and
`publishable=true`. The first read after macOS recovery showed a fresh keeper
at `7/7` and new power/PCI timestamps. Although the public provider-generation
field restarted at numeric value 1, the service IDs, counter reset, and power
timestamps prove a provider replacement.

## Classification

This arm does not show whether a fully loaded, lightly active model could
stabilize an otherwise healthy tunnel, because the tunnel failed before the
model became resident. It does establish a stronger immediate boundary:

- historical minimal AMD compute still works;
- large Qwen DMA population reached approximately 4.68 GB;
- the both-lane ACIO error signature then preceded zero link, endpoint removal,
  TinyGPU force-close, and PCIe tunnel power-down;
- full DriverKit power residency, BAR5 retention, keeper traffic, and PCI
  command mask 7 did not prevent that sequence.

This materially raises the physical USB4 path. A known-good certified cable
and alternate host port A/B is now more discriminating than another software
keeper change. This result alone cannot distinguish cable, connector/port,
UT4G/ASM2464 bridge, enclosure power integrity, or combined signal margin.

## Artifacts

- `egpu-usb4-persistent-pcie-A12-20260729T043529Z-10496.json`
- `egpu-usb4-persistent-model-residency-20260729T043514Z-10496.json`
- `egpu-usb4-persistent-pcie-A12-20260729T043810Z-10770.json`
- `egpu-usb4-persistent-model-residency-20260729T043803Z-10770.json`
- `egpu-usb4-persistent-pcie-A12-20260729T044024Z-10930.json`
- `egpu-usb4-persistent-model-residency-20260729T044009Z-10930.json`

Verification for the runner and both host-fact corrections: 96 targeted tests
passed. A broader pre-existing source-policy test remains failed because old
`model.py` branches contain literal `8B/14B` profile names; that failure is
unrelated to these changes and was not suppressed.
