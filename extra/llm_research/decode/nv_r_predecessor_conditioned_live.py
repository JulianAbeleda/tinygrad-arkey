#!/usr/bin/env python3
"""Predecessor-conditioned C0/C2 replay against cloned live decode buffers.

The corrected harness proved all six retained R cubins are cache-sensitive
under a full 128 MiB eviction, but the production cache state is set by the
installed predecessor, not by a stress flush.  This tool closes that gap for
the Q branch of one decode layer:

    provider norm -> Q projection

It monkeypatches ``NVProgram.__call__`` during a real decode, forces ``wait``
for the selected provider and Q-projection calls so their buffers are final,
clones the live buffers, validates that replaying the exact production cubin on
the clones reproduces the production output, and then measures three intervals
on the same GPU process before it exits:

  C0   isolated Q projection on cloned (pre-warmed) input
  C2   provider norm -> Q projection on the chain's fresh intermediate buffer
  C3   FFN gate/up + down -> provider -> Q projection (full local prefix)

``C2 - C0`` is the predecessor-conditioned residual: the part of the
production command interval that an isolated replay does not reproduce.
``C3 - C2`` isolates the cache-eviction effect of the ~98 MB FFN weight stream
that immediately precedes the provider in the production schedule.

Measurement tooling only; no production model, renderer, scheduler, runtime,
or route-policy code is changed.  The hook is restored after the decode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nv_r_residual_cache_dispatch_probe import _alloc, _make_queue  # noqa: E402

SCHEMA = "tinygrad.nv_r_predecessor_conditioned_live.v1"
PROVIDER = "rmsnorm_q8_1_llama_provider_4096"
TARGET = "q4k_warp_coop_q8_dp4a_partial_4096_4096"
GATEUP = "q4k_g3_lanemap_gemv_w1w3fused16_12288_4096"
FFNDOWN = "q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd"
SELECTED = (PROVIDER, TARGET, GATEUP, FFNDOWN)
# Only the chain whose data we replay for correctness needs live-buffer clones.
# The FFN prefix only has to stream its real-sized weights to set cache state.
CLONE_KEYS = {PROVIDER, TARGET}


def _sha(b: bytes) -> str:
  return hashlib.sha256(b).hexdigest()


def _va(b) -> int:
  va = getattr(b, "va_addr", None)
  if va is None:
    va = getattr(getattr(b, "_buf", None), "va_addr", None)
  return int(va)


def _size(b) -> int:
  sz = getattr(b, "size", None)
  if sz is None:
    sz = getattr(getattr(b, "_buf", None), "size", None)
  return int(sz)


def _clone(dev, b):
  """Copy a live buffer's current contents into a fresh allocation."""
  host = memoryview(bytearray(_size(b)))
  dev.allocator._copyout(host, b)
  dst = _alloc(dev, _size(b))
  dev.allocator._copyin(dst, host)
  return dst


def _fresh_bufs(dev, sizes):
  """Zeroed buffers of the given sizes, for cache-traffic-only prefix replay."""
  bufs = tuple(_alloc(dev, sz) for sz in sizes)
  for b in bufs:
    dev.allocator._copyin(b, memoryview(bytearray(b.size)))
  return bufs


def _checksum(dev, b) -> str:
  host = memoryview(bytearray(_size(b)))
  dev.allocator._copyout(host, b)
  return _sha(bytes(host))


def install_hook(cap: dict, layer: int):
  import tinygrad.runtime.ops_nv as ops_nv
  orig_call = ops_nv.NVProgram.__call__
  seen: dict[str, int] = {}

  def patched_call(self, *bufs, global_size=(1, 1, 1), local_size=(1, 1, 1),
                   vals=(), wait=False, timeout=None):
    name = self.name
    is_sel = False
    if name in SELECTED:
      seen[name] = seen.get(name, 0) + 1
      is_sel = seen[name] == layer + 1

    # Force completion for the selected calls so every buffer is final before
    # it is cloned; the decode is slower but its outputs are unchanged.
    ret = orig_call(self, *bufs, global_size=global_size, local_size=local_size,
                    vals=vals, wait=(wait or is_sel), timeout=timeout)

    if is_sel:
      rec = {
        "program": self,
        "cubin": bytes(self.lib),
        "cubin_sha": _sha(bytes(self.lib)),
        "buf_va": [_va(b) for b in bufs],
        "buf_size": [_size(b) for b in bufs],
        "vals": list(vals),
        "global_size": list(global_size),
        "local_size": list(local_size),
      }
      if name in CLONE_KEYS:
        rec["clones"] = [_clone(self.dev, b) for b in bufs]
      cap[name] = rec

    return ret

  ops_nv.NVProgram.__call__ = patched_call
  return orig_call


def restore_hook(orig_call):
  import tinygrad.runtime.ops_nv as ops_nv
  ops_nv.NVProgram.__call__ = orig_call


def run_decode(cap: dict, layer: int, model: str):
  from tinygrad import Device
  from tinygrad.llm.generate import load_model_and_tokenizer

  orig = install_hook(cap, layer)
  try:
    dev = Device[Device.DEFAULT]
    mdl, tok = load_model_and_tokenizer(model, 4608, seed=20260617)
    base = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
    prompt = base[:512]
    gen = mdl.generate(prompt.copy(), chunk_size=32, temperature=0.0)
    try:
      next(gen)
      for _ in range(6):
        next(gen)
      dev.synchronize()
    finally:
      gen.close()
  finally:
    restore_hook(orig)


def _measure(dev, pred_launches, target_prg, target_bufs, target_vals, target_grid,
             target_block, n, timeout_s):
  """Time target over n samples, optionally preceded by provider launches.

  ``pred_launches`` is a list of ``(prg, bufs, vals, grid, block)``.  A fresh
  ``fill_kernargs`` QMD is built for every dispatch so no mutable packet is
  aliased across in-flight launches; the buffers themselves are reused, which
  is safe because each sample is submitted and synchronized serially.
  """
  durs = []
  for _ in range(n):
    q = _make_queue(dev)
    st = dev.new_signal()
    en = dev.new_signal()
    for (pp, pbufs, pvals, pg, pb) in pred_launches:
      pargs = pp.fill_kernargs(tuple(pbufs), vals=tuple(pvals))
      q.exec(pp, pargs, pg, pb)
    q.timestamp(st)
    targs = target_prg.fill_kernargs(tuple(target_bufs), vals=tuple(target_vals))
    q.exec(target_prg, targs, target_grid, target_block)
    q.timestamp(en)
    q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
    dev.synchronize(timeout=int(timeout_s * 1000))
    durs.append(float(en.timestamp - st.timestamp))
  return durs


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--layer", type=int, default=0)
  ap.add_argument("--n", type=int, default=24)
  ap.add_argument("--warmup", type=int, default=4)
  ap.add_argument("--timeout-s", type=float, default=180.0)
  ap.add_argument("--capture-only", action="store_true")
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  from tinygrad import Device

  cap: dict = {}
  run_decode(cap, args.layer, args.model)

  if any(k not in cap for k in (TARGET, PROVIDER, GATEUP, FFNDOWN)):
    raise SystemExit(f"capture failed: have={sorted(cap)}")

  dev = Device["NV"]
  t = cap[TARGET]
  p = cap[PROVIDER]
  grid = tuple(t["global_size"])
  block = tuple(t["local_size"])
  tprg = t["program"]

  # The production cubin receives its buffers as (output, weight, input):
  #   [65536, 9437184, 8192] = [Q8/Q output, Q4_K weight, bf16 input].
  # Guard against a live-layout change silently reordering the replay.
  TARGET_OUT_IDX, TARGET_W_IDX, TARGET_IN_IDX = 0, 1, 2
  assert t["buf_size"] == [65536, 9437184, 8192], f"unexpected target layout: {t['buf_size']}"

  # Identify the provider's output as the buffer whose VA equals the target's
  # input VA. If the ordinals did not align this reports a mismatch.
  t_in_va = t["buf_va"][TARGET_IN_IDX]
  p_out_idx = next((i for i, va in enumerate(p["buf_va"]) if va == t_in_va), None)

  result: dict = {
    "schema": SCHEMA,
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "layer": args.layer,
    "target": TARGET,
    "provider": PROVIDER,
    "target_cubin_sha": t["cubin_sha"],
    "provider_cubin_sha": p["cubin_sha"],
    "target_buf_size": t["buf_size"],
    "provider_buf_size": p["buf_size"],
    "provider_output_idx": p_out_idx,
    "gateup_cubin_sha": cap[GATEUP]["cubin_sha"],
    "gateup_buf_size": cap[GATEUP]["buf_size"],
    "gateup_vals": cap[GATEUP]["vals"],
    "gateup_global_size": cap[GATEUP]["global_size"],
    "gateup_local_size": cap[GATEUP]["local_size"],
    "ffndown_cubin_sha": cap[FFNDOWN]["cubin_sha"],
    "ffndown_buf_size": cap[FFNDOWN]["buf_size"],
    "ffndown_vals": cap[FFNDOWN]["vals"],
    "ffndown_global_size": cap[FFNDOWN]["global_size"],
    "ffndown_local_size": cap[FFNDOWN]["local_size"],
    "target_vals": t["vals"],
    "provider_vals": p["vals"],
    "target_global_size": t["global_size"],
    "target_local_size": t["local_size"],
    "provider_global_size": p["global_size"],
    "provider_local_size": p["local_size"],
  }

  if p_out_idx is None:
    result["verdict"] = "PROVIDER_MISMATCH"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2

  # Validation: replay the target on cloned input+weight and compare its output
  # to the production output cloned from the live target buffers.
  out_fresh = _alloc(dev, t["buf_size"][TARGET_OUT_IDX])
  dev.allocator._copyin(out_fresh, memoryview(bytearray(out_fresh.size)))
  args_val = tprg.fill_kernargs(
      (out_fresh, t["clones"][TARGET_W_IDX], t["clones"][TARGET_IN_IDX]),
      vals=tuple(t["vals"]))
  q = _make_queue(dev)
  q.exec(tprg, args_val, grid, block)
  q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize(timeout=int(args.timeout_s * 1000))
  replay_checksum = _checksum(dev, out_fresh)
  prod_checksum = _checksum(dev, t["clones"][TARGET_OUT_IDX])
  output_match = replay_checksum == prod_checksum
  result.update({
    "replay_output_checksum": replay_checksum,
    "production_output_checksum": prod_checksum,
    "replay_matches_production": output_match,
  })

  if args.capture_only or not output_match:
    result["verdict"] = "CAPTURE_VALIDATED" if output_match else "CAPTURE_MISMATCH"
  else:
    # C0: isolated target on cloned input (pre-warmed by repeated launch).
    c0_bufs = (out_fresh, t["clones"][TARGET_W_IDX], t["clones"][TARGET_IN_IDX])
    c0 = _measure(dev, [], tprg, c0_bufs, tuple(t["vals"]), grid, block,
                  args.n, args.timeout_s)[args.warmup:]

    # C2: provider -> target. Provider writes a fresh intermediate buffer that
    # becomes the target input, reproducing the production chain's cache state.
    pprg = p["program"]
    pgrid = tuple(p["global_size"])
    pblock = tuple(p["local_size"])
    prov_bufs = list(p["clones"])
    prov_out = _alloc(dev, p["buf_size"][p_out_idx])
    dev.allocator._copyin(prov_out, memoryview(bytearray(prov_out.size)))
    prov_bufs[p_out_idx] = prov_out
    c2_bufs = (out_fresh, t["clones"][TARGET_W_IDX], prov_out)
    c2 = _measure(dev, [(pprg, tuple(prov_bufs), tuple(p["vals"]), pgrid, pblock)],
                  tprg, c2_bufs, tuple(t["vals"]), grid, block,
                  args.n, args.timeout_s)[args.warmup:]

    # C3: full local prefix (FFN gate/up + down) -> provider -> target. The
    # ~98 MB FFN weight stream is the production cache-evicting predecessor;
    # C3 - C2 isolates its effect on the target's weight residency.
    gprg = cap[GATEUP]["program"]
    ggrid = tuple(cap[GATEUP]["global_size"])
    gblock = tuple(cap[GATEUP]["local_size"])
    gbufs = _fresh_bufs(dev, cap[GATEUP]["buf_size"])

    fprg = cap[FFNDOWN]["program"]
    fgrid = tuple(cap[FFNDOWN]["global_size"])
    fblock = tuple(cap[FFNDOWN]["local_size"])
    fbufs = _fresh_bufs(dev, cap[FFNDOWN]["buf_size"])

    c3 = _measure(
        dev,
        [
          (gprg, gbufs, tuple(cap[GATEUP]["vals"]), ggrid, gblock),
          (fprg, fbufs, tuple(cap[FFNDOWN]["vals"]), fgrid, fblock),
          (pprg, tuple(prov_bufs), tuple(p["vals"]), pgrid, pblock),
        ],
        tprg, c2_bufs, tuple(t["vals"]), grid, block,
        args.n, args.timeout_s)[args.warmup:]

    result.update({
      "C0_median_us": round(statistics.median(c0), 3),
      "C2_median_us": round(statistics.median(c2), 3),
      "C2_minus_C0_us": round(statistics.median(c2) - statistics.median(c0), 3),
      "C3_median_us": round(statistics.median(c3), 3),
      "C3_minus_C2_us": round(statistics.median(c3) - statistics.median(c2), 3),
      "C3_minus_C0_us": round(statistics.median(c3) - statistics.median(c0), 3),
      "C0_us": [round(x, 3) for x in c0],
      "C2_us": [round(x, 3) for x in c2],
      "C3_us": [round(x, 3) for x in c3],
      "verdict": "MEASURED",
    })

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
