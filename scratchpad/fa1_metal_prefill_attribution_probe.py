#!/usr/bin/env python3
"""FA1: reconciled per-kernel attribution of Metal whole-model prefill, at start_pos=0 and 3584.

Why this differs from the retracted probe (`scratchpad/metal_prefill_attn_depth_probe.py`,
retraction: `718d5717d`, see docs/qwen3-8b-prefill-metal-attention-at-depth-20260731.md correction
banner): that probe used `Context(JIT=0)`, i.e. TinyJit's "jit ignore" branch
(tinygrad/engine/jit.py:555-559) on every call, and compared its `DEBUG=2` per-kernel capture (652ms)
against a *different run's* wall clock (9,446ms, from prefill_whole_synced.py's JIT-enabled, MetalGraph-
batched burst()). JIT=0 pays full Python-side graph construction/scheduling cost on every call, which
DEBUG=2's per-kernel `tm` timing does not count (it only times each kernel's own dispatch) -- that
mismatch, not attention itself, is most of what the 7% coverage was hiding, compounded by comparing two
different execution regimes.

RECONCILIATION METHOD (the mandatory discipline for this packet):
  - Keeps TinyJit's cache warm (real production path: use_flash=True, _prefill_v2=True, the exact
    `model(chunk, sp, temp, use_flash=True)` call both prefill_whole_synced.py and the retracted probe
    use) so wall clock is not dominated by recompilation.
  - Sets `Context(JIT=2)` for the ENTIRE lifetime of each start_pos's TinyJit instance (warmup calls
    included), which skips `graph_split_rewrite` (tinygrad/engine/jit.py:333) at JIT-capture time. That
    means every kernel replays as its own individually-committed, individually-labeled MTLCommandBuffer
    (tinygrad/runtime/ops_metal.py:130-141) instead of being folded into one opaque "batched N" Metal
    command buffer that ops_metal.py:56-58 explicitly does NOT profile
    ("NOTE: command buffers from MetalGraph are not profiled here"). This trades away MetalGraph's
    submission-overhead savings -- JIT=2 wall clock is consequently a DIFFERENT configuration than
    production JIT=1 (the 9,446/33,881ms figures), reported here separately and not conflated with it --
    in exchange for being able to reconcile at all.
  - Turns on `PROFILE=1` for the measured calls, which makes `MetalDevice.synchronize()` append one
    `ProfileRangeEvent` per (non-batched) command buffer with REAL Metal `GPUStartTime()`/`GPUEndTime()`
    timestamps (tinygrad/runtime/ops_metal.py:52-58) -- hardware ground truth, not a Python-side estimate.
  - Reconciles: sums the device-side per-kernel durations for a single measured call and divides by that
    SAME call's wall clock (perf_counter, Device.synchronize() bracketed, same configuration, same
    process). Coverage is reported for every round, not asserted.

CLASSIFICATION METHOD:
  The obvious first idea -- join each kernel's device-timing to the semantic role tags the model already
  attaches via `role_metadata(...)` (tinygrad/llm/model.py) and the automatic per-Tensor-method
  `_metadata_wrapper` (tinygrad/tensor.py:1439-1491), through `call.arg.metadata`
  (tinygrad/engine/realize.py:54-55) -- was tried and FAILED: empirically, `call.arg.metadata` is empty
  for every one of ~1090 kernel dispatches in this build, including the very first eager ("jit ignore",
  cnt==0) call before any JIT capture happens. This is not a JIT artifact; role_metadata's tags never
  reach CallInfo.metadata on this path. That avenue is abandoned and reported as a dead end, not silently
  dropped.

  Instead this uses the classification that SURVIVED the correction in
  docs/qwen3-8b-prefill-metal-attention-at-depth-20260731.md Sec 1.1: kernels are split by whether their
  identity (clean/ANSI-stripped Metal function name, which bakes in KV length for attention kernels) is
  DEPTH-VARYING (appears in the dispatch set at start_pos=0 XOR start_pos=3584 -- i.e. NOT common to
  both) vs DEPTH-INVARIANT (dispatched, byte-identical name, at both start_pos values). Only attention
  reads a KV window whose length depends on start_pos; QKVO projection and FFN GEMMs, and elementwise/
  norm/rope ops, operate on the fixed 512-token chunk and do not. This is the split that survived
  retraction ("What survives: the observation that attention kernels are recompiled per depth with the
  KV length in the kernel name, and that 24 kernels are depth-invariant" -- the *shares* computed from it
  were what got refuted, not the split itself, and that refutation was about coverage, not this method).

  depth-varying                                  -> "attention"
  depth-invariant AND kernel-type "r" (has a reduceop; codegen/opt/postrange.py:117:
    `k_type = "r" if self.reduceop is not None else "E"`, i.e. the kernel contains a genuine reduction/
    contraction -- QKVO and FFN matmuls all reduce over an input-feature axis)  -> "gemm"
  depth-invariant AND kernel-type "E" (no reduceop: elementwise/norm/rope/residual/cast), or a device
    "COPY ..." label                             -> "other"
  anything not classifiable by the above (should be empty; reported explicitly, not swept in)
                                                  -> "unclassified"

  This is NOT the retracted probe's mistake ("its first classifier put every r_ kernel in one bucket"):
  that classifier used r_/E_ as the ONLY signal, so it merged attention's own reduce kernels (softmax,
  QK^T, PV) into the same bucket as FFN/QKVO reduce kernels. Here, r_/E_ is applied ONLY inside the
  depth-invariant residual, after attention's kernels (all depth-varying, since KV length is baked into
  every attention kernel's shape) have already been split out by the primary criterion.
"""
from __future__ import annotations
import json, os, pathlib, sys, time
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = "/Users/julianabeleda/models/Qwen3-8B-Q4_K_M.gguf"
MAX_CONTEXT = 4608
CHUNK_N = 512
START_POSITIONS = tuple(int(x) for x in os.environ.get("FA1_START_POSITIONS", "0,3584").split(","))
WARMUPS = int(os.environ.get("FA1_WARMUPS", "4"))
MEASURED_ROUNDS = int(os.environ.get("FA1_ROUNDS", "3"))
OUT_DIR = pathlib.Path("/tmp/fa1_metal_prefill_attribution")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def kernel_type(name: str) -> str:
  """codegen/opt/postrange.py:117: k_type = 'r' if self.reduceop is not None else 'E'."""
  if name.upper().startswith("COPY"): return "COPY"
  if name.startswith("r_"): return "r"
  if name.startswith("E_"): return "E"
  return "?"


def main() -> None:
  from tinygrad import Tensor, Device
  from tinygrad.helpers import Context, to_function_name
  from tinygrad.device import Compiled
  from tinygrad.llm.generate import load_model_and_tokenizer

  print(f"loading {MODEL} ...", flush=True)
  model, _ = load_model_and_tokenizer(MODEL, MAX_CONTEXT, seed=20260617)
  for block in model.blk: block._use_flash, block._prefill_v2 = True, True
  temp = Tensor([0.0])
  chunk = Tensor([[(i * 7) % 1000 for i in range(CHUNK_N)]], dtype="int32").contiguous()
  dev = Device[Device.DEFAULT]

  def call(sp: int):
    return model(chunk, sp, temp, use_flash=True)

  per_sp: dict[int, dict] = {}

  for sp in START_POSITIONS:
    print(f"\n=== start_pos={sp} ===", flush=True)
    # Entire lifetime of this start_pos's TinyJit instance runs under JIT=2: this is what keeps
    # graph_split_rewrite from ever firing for it (decided once, at jit-capture time, cnt==1).
    with Context(JIT=2):
      for i in range(WARMUPS):
        call(sp).realize()
      dev.synchronize()

    rounds = []
    union_names: set[str] = set()
    for r in range(MEASURED_ROUNDS):
      with Context(JIT=2, PROFILE=1):
        dev.synchronize()
        ev_before = len(Compiled.profile_events)
        t0 = time.perf_counter()
        call(sp).realize()
        dev.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        prof_events = list(Compiled.profile_events[ev_before:])

      by_name_us: dict[str, float] = defaultdict(float)
      counts: dict[str, int] = defaultdict(int)
      for ev in prof_events:
        if type(ev).__name__ != "ProfileRangeEvent": continue
        if ev.en is None: continue
        by_name_us[str(ev.name)] += float(ev.en - ev.st)
        counts[str(ev.name)] += 1

      captured_us = sum(by_name_us.values())
      captured_ms = captured_us / 1000.0
      coverage = captured_ms / wall_ms if wall_ms else float("nan")
      rounds.append({
        "wall_ms": wall_ms, "captured_ms": captured_ms, "coverage": coverage,
        "n_kernel_names": len(by_name_us), "n_dispatches": sum(counts.values()),
        "by_name_us": dict(by_name_us), "counts": dict(counts),
      })
      union_names |= set(by_name_us.keys())
      print(f"  round {r}: wall={wall_ms:8.2f}ms captured={captured_ms:8.2f}ms "
            f"coverage={coverage*100:5.1f}%  dispatches={sum(counts.values())} unique_names={len(by_name_us)}",
            flush=True)

    wall_ms_med = sorted(r["wall_ms"] for r in rounds)[len(rounds) // 2]
    captured_ms_med = sorted(r["captured_ms"] for r in rounds)[len(rounds) // 2]
    # median-round per-name timings (fall back to the median-wall-clock round's own by_name_us)
    median_round = sorted(rounds, key=lambda r: r["wall_ms"])[len(rounds) // 2]
    per_sp[sp] = {
      "rounds": rounds, "wall_ms_median": wall_ms_med, "captured_ms_median": captured_ms_med,
      "coverage_median": captured_ms_med / wall_ms_med if wall_ms_med else float("nan"),
      "union_names": union_names, "median_by_name_us": median_round["by_name_us"],
    }
    print(f"  MEDIAN: wall={wall_ms_med:.2f}ms captured={captured_ms_med:.2f}ms "
          f"coverage={captured_ms_med/wall_ms_med*100:.1f}%", flush=True)

  if len(per_sp) >= 2:
    sps = sorted(per_sp.keys())
    name_sets = {sp: per_sp[sp]["union_names"] for sp in sps}
    all_sp_common = set.intersection(*name_sets.values())
    print("\n=== cross-depth classification ===")
    print(f"  depth-invariant (common to all {len(sps)} start_pos): {len(all_sp_common)} unique kernel names")
    for sp in sps:
      only = name_sets[sp] - all_sp_common
      print(f"  depth-varying, only at start_pos={sp}: {len(only)} unique kernel names")

    for sp in sps:
      by_name_us = per_sp[sp]["median_by_name_us"]
      bucket_us: dict[str, float] = defaultdict(float)
      bucket_names: dict[str, list[str]] = defaultdict(list)
      for nm, us in by_name_us.items():
        if nm not in all_sp_common:
          b = "attention"
        else:
          kt = kernel_type(nm)
          b = {"r": "gemm", "E": "other", "COPY": "other"}.get(kt, "unclassified")
        bucket_us[b] += us
        bucket_names[b].append(nm)
      total_us = sum(bucket_us.values())
      per_sp[sp]["classification"] = {
        b: {"us": bucket_us.get(b, 0.0), "ms": bucket_us.get(b, 0.0) / 1000.0,
            "share_of_captured": (bucket_us.get(b, 0.0) / total_us if total_us else float("nan")),
            "n_unique_kernel_names": len(bucket_names.get(b, []))}
        for b in ("attention", "gemm", "other", "unclassified")
      }
      print(f"\n  start_pos={sp} classification (share of median-round CAPTURED time):")
      for b, row in per_sp[sp]["classification"].items():
        print(f"    {b:14s}: {row['ms']:9.2f}ms  {row['share_of_captured']*100:5.1f}% of captured  "
              f"({row['n_unique_kernel_names']} unique kernel names)")
      unclassified_names = bucket_names.get("unclassified", [])
      if unclassified_names:
        print(f"    unclassified kernel names: {unclassified_names}")

  out_path = OUT_DIR / "fa1_attribution.json"

  def _default(o):
    if isinstance(o, set): return sorted(o)
    return str(o)
  out_path.write_text(json.dumps(per_sp, indent=2, default=_default))
  print(f"\nwrote {out_path}")


if __name__ == "__main__":
  main()
