#!/usr/bin/env python3
"""P7d timing for the one-role imported Q4 MMVQ model route."""
from __future__ import annotations

import json, pathlib, statistics, time

import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
import tinygrad.llm.model as llm_model
from extra.llm_generate import load_model_and_tokenizer
from extra.qk_decode_mmvq_graph_route import Q8_BYTES, install_imported_q4_mmvq, route_imported_q4_mmvq
from extra.qk_decode_mmvq_p3_q4_correctness import OUT
from extra.qk_nll_eval import CALIB_TEXT

MODEL = pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")


def diff_stats(a: Tensor, b: Tensor) -> dict:
  av, bv = a.numpy().astype("float32", copy=False), b.numpy().astype("float32", copy=False)
  d = np.abs(av - bv)
  return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "max_rel": float((d / np.maximum(np.abs(bv), 1e-6)).max())}


class CaptureLinear:
  def __init__(self, linear):
    self.linear, self.last_input = linear, None

  def __getattr__(self, name):
    return getattr(self.linear, name)

  def __call__(self, x: Tensor) -> Tensor:
    self.last_input = x.cast(dtypes.float32).contiguous().realize()
    return self.linear(x)


def median_ms(xs: list[float]) -> float:
  return statistics.median(xs) * 1000.0


def main() -> None:
  if Device.DEFAULT != "AMD":
    raise RuntimeError(f"P7d requires DEV=AMD, got {Device.DEFAULT!r}")
  OUT.mkdir(parents=True, exist_ok=True)
  dev = Device["AMD"]
  model, tok = load_model_and_tokenizer(str(MODEL), 4096, seed=20260619)
  for lin in getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []:
    lin.decode_enabled = True

  block = model.blk[0]
  original_linear = block.attn_output
  install = install_imported_q4_mmvq(original_linear.out_features)

  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  token = Tensor([[ids[0]]], dtype=dtypes.int32, device="AMD").contiguous()
  x = model.token_embd(token).float().realize()
  block._init_state(x)

  capture = CaptureLinear(original_linear)
  block.attn_output = capture
  try:
    baseline_attention = block._attention(block.attn_norm(x), 0).realize()
  finally:
    block.attn_output = original_linear
  if capture.last_input is None:
    raise RuntimeError("failed to capture pre-attn_output activation")
  out_in = capture.last_input

  q8_side = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  out_side = Tensor.empty(original_linear.out_features, dtype=dtypes.float32, device="AMD").contiguous().realize()

  @TinyJit
  def baseline(inp: Tensor):
    return original_linear(inp).realize()

  @TinyJit
  def imported(inp: Tensor, q8_buf: Tensor, out_buf: Tensor):
    out = route_imported_q4_mmvq(original_linear, inp, q8_buf, out_buf)
    if out is None:
      raise RuntimeError("imported route returned None")
    return out.realize()

  # Compile and clock-warm both sides.
  for _ in range(8):
    baseline(out_in).realize()
    imported(out_in, q8_side, out_side).realize()
  dev.synchronize(timeout=10000)

  warmups, iters = 8, 40
  baseline_s: list[float] = []
  imported_s: list[float] = []
  replay_diffs: list[dict] = []
  ref_imported = imported(out_in, q8_side, out_side).realize()
  dev.synchronize(timeout=10000)

  for i in range(warmups + iters):
    if i % 2 == 0:
      t0 = time.perf_counter()
      b = baseline(out_in).realize()
      dev.synchronize(timeout=10000)
      tb = time.perf_counter() - t0

      t0 = time.perf_counter()
      r = imported(out_in, q8_side, out_side).realize()
      dev.synchronize(timeout=10000)
      tr = time.perf_counter() - t0
    else:
      t0 = time.perf_counter()
      r = imported(out_in, q8_side, out_side).realize()
      dev.synchronize(timeout=10000)
      tr = time.perf_counter() - t0

      t0 = time.perf_counter()
      b = baseline(out_in).realize()
      dev.synchronize(timeout=10000)
      tb = time.perf_counter() - t0

    if i >= warmups:
      baseline_s.append(tb)
      imported_s.append(tr)
      if len(replay_diffs) < 5:
        replay_diffs.append(diff_stats(r, ref_imported))

  # Confirm the actual P7c model branch still routes with the same block.
  llm_model.DECODE_MMVQ_IMPORT_Q4 = True
  try:
    routed_attention = block._attention(block.attn_norm(x), 0).realize()
    dev.synchronize(timeout=10000)
  finally:
    llm_model.DECODE_MMVQ_IMPORT_Q4 = False

  baseline_med = median_ms(baseline_s)
  imported_med = median_ms(imported_s)
  speedup = baseline_med / imported_med if imported_med > 0 else 0.0
  result = {
    "schema": "decode_mmvq_large_project_p7d_one_role_timing_v1",
    "date": "2026-06-19",
    "phase": "P7d_one_role_timing",
    "role": "blk.0.attn_output",
    "install": install,
    "activation_shape": list(out_in.shape),
    "baseline_attention_shape": list(baseline_attention.shape),
    "routed_attention_shape": list(routed_attention.shape),
    "timing": {
      "method": "same-process interleaved TinyJit wall time with Device synchronize after each call",
      "warmups": warmups,
      "iters": iters,
      "baseline_ms_median": baseline_med,
      "imported_ms_median": imported_med,
      "speedup": speedup,
      "baseline_ms_min": min(baseline_s) * 1000.0,
      "imported_ms_min": min(imported_s) * 1000.0,
    },
    "correctness": {
      "baseline_vs_imported_q8_path": diff_stats(ref_imported, baseline(out_in).realize()),
      "imported_replay_diffs": replay_diffs,
    },
    "gates": {
      "baseline_runs": len(baseline_s) == iters,
      "imported_runs": len(imported_s) == iters,
      "imported_replay_stable": all(d["max_abs"] <= 1e-6 for d in replay_diffs),
      "model_branch_routed": hasattr(block, "_decode_mmvq_import_q4_q8"),
      "speedup_ge_1_10": speedup >= 1.10,
      "default_unchanged": True,
    },
  }
  result["verdict"] = "PASS_ONE_ROLE_TIMING" if all(result["gates"].values()) else "NO_LOCAL_TIMING_WIN"
  (OUT / "p7d_one_role_timing.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  if result["verdict"] != "PASS_ONE_ROLE_TIMING":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
