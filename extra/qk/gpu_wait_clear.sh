#!/bin/bash
# gpu_wait_clear <min_free_gb> [timeout_s] [poll_s]
# General GPU-clear gate for ANY workload (not model-specific): scans AMD VRAM and
# WAITS (polling) until at least <min_free_gb> is free, up to <timeout_s>.
# - min_free_gb: required free VRAM the CALLER passes for whatever it's about to run
#   (e.g. weights + KV + overlay headroom). Nothing here is tied to a specific model.
# - total VRAM is read live from the device, so it adapts to any GPU.
# Exit 0 when clear; 1 on timeout.
min_free_gb=${1:?usage: gpu_wait_clear <min_free_gb> [timeout_s] [poll_s]}
timeout_s=${2:-1200}; poll_s=${3:-15}
_vram() { timeout 15 rocm-smi --showmeminfo vram 2>/dev/null | grep -iE "$1 Memory \(B\)" | grep -oE '[0-9]+$' | head -1; }
waited=0
while true; do
  total_b=$(_vram "Total"); used_b=$(_vram "Used")
  if [ -z "$total_b" ] || [ -z "$used_b" ]; then echo "GPU SCAN FAILED (rocm-smi); proceeding cautiously"; exit 0; fi
  free_gb=$(( (total_b - used_b) / 1000000000 ))
  if [ "$free_gb" -ge "$min_free_gb" ]; then echo "GPU CLEAR: ${free_gb}GB free (need ${min_free_gb}) after ${waited}s"; exit 0; fi
  if [ "$waited" -ge "$timeout_s" ]; then
    echo "GPU WAIT TIMEOUT: ${free_gb}GB free (need ${min_free_gb}) after ${waited}s; holders:"
    timeout 15 rocm-smi --showpidgpus 2>/dev/null | grep -iE "PID [0-9]"; exit 1; fi
  echo "GPU BUSY: ${free_gb}GB free, need ${min_free_gb} — waiting ${poll_s}s (elapsed ${waited}s)"
  sleep "$poll_s"; waited=$((waited+poll_s))
done
