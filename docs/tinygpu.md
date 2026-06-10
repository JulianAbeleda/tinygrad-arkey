# TinyGPU

TinyGPU app lets you use AMD and NVIDIA GPUs on macOS over USB4/Thunderbolt with tinygrad-arkey.

## Requirements

- macOS (13.0+)
- USB4/Thunderbolt port
- A supported GPU (AMD RDNA3+ or NVIDIA Ampere+)

## Setup

### 1. Connect your GPU

Plug the supported GPU into your Mac over USB4/Thunderbolt.

### 2. Initiate the driver install

> **Note:** If tinygrad is cloned but not installed, run commands with `PYTHONPATH=.`

```bash
curl -fsSL https://raw.githubusercontent.com/JulianAbeleda/tinygrad-arkey/master/extra/setup_tinygpu_osx.sh | sh
```

This downloads TinyGPU.app and triggers a system prompt to install the driver extension.

### 3. Enable the driver

You should see a system prompt: **"TinyGPU" would like to use a new driver extension**. Click **Open System Settings** and toggle TinyGPU on.

If you missed the prompt, go to **System Settings > General > Login Items & Extensions > Driver Extensions** and toggle TinyGPU on.

### 4. Compiler Setup

#### AMD

```bash
curl -fsSL https://raw.githubusercontent.com/JulianAbeleda/tinygrad-arkey/master/extra/setup_hipcomgr_osx.sh | sh
```

#### NV

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) if you don't have it.

```bash
curl -fsSL https://raw.githubusercontent.com/JulianAbeleda/tinygrad-arkey/master/extra/setup_nvcc_osx.sh | sh
```

Make sure `~/.local/bin` is on your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 5. Use it!

```bash
DEV={AMD|NV} python3 -m tinygrad.llm
```

## Troubleshooting (AMD / RX 7900 XTX over UT4G)

Hard-won operational rules from the 2026-06 bring-up (details in
`docs/amd-kdb-root-cause.md`):

- **RX 7900 XTX env:** boot the remote path with
  `AM_REMOTE_SKIP_RESIZE_BAR=1 AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c`.
- **After any bridge death, power-cycle the GPU — actually remove power.**
  A cable replug does not reset the card's gated state; MMHUB register reads
  then hang the fabric and drop the whole PCIe tree.
- **Verify /Applications/TinyGPU.app is the arkey build** before debugging
  anything: `codesign -dv /Applications/TinyGPU.app` should show
  `org.tinygrad.arkey...`. A version-mismatched app/dext pair fails in
  bizarre ways (e.g. single BAR0 writes closing the connection).
- **Don't reuse a long-running serve.py bridge.** Check its age with
  `ps -o etime -p $(pgrep -f serve.py)`; a stale bridge runs stale protocol
  code against a new client and hangs. Kill and restart it each session.
- Bulk MMIO transfer knobs: `REMOTE_MMIO_CHUNK` (default 4-byte chunking on
  macOS, 0 = bulk) and `REMOTE_MMIO_FENCE_EVERY` (serializing readback
  cadence, 0 = off) in `extra/remote/serve.py`.

**Note:** Use `JITBEAM=2` to search for faster kernels (one-time search cost, results cached).
