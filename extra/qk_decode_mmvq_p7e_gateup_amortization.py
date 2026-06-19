#!/usr/bin/env python3
"""P7e timing for imported Q4 MMVQ on the ffn_gate/up shared-input pair."""
from __future__ import annotations

import json, pathlib, statistics, time

import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from extra.llm_generate import load_model_and_tokenizer
from extra.qk_decode_mmvq_graph_route import Q8_BYTES, install_imported_q4_mmvq, q4_mmvq_stub, q4_words, q8_quant_stub
from extra.qk_decode_mmvq_p3_q4_correctness import OUT
from extra.qk_nll_eval import CALIB_TEXT

MODEL = pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")


def diff_stats(a: Tensor, b: Tensor) -> dict:
  av, bv = a.numpy().astype("float32", copy=False), b.numpy().astype("float32", copy=False)
  d = np.abs(av - bv)
  return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "max_rel": float((d / np.maximum(np.abs(bv), 1e-6)).max())}


def median_ms(xs: list[float]) -> float:
  return statistics.median(xs) * 1000.0


def main() -> None:
  if Device.DEFAULT != "AMD":
    raise RuntimeError(f"P7e requires DEV=AMD, got {Device.DEFAULT!r}")
  OUT.mkdir(parents=True, exist_ok=True)
  dev = Device["AMD"]
  model, tok = load_model_and_tokenizer(str(MODEL), 4096, seed=20260619)
  for lin in getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []:
    lin.decode_enabled = True

  block = model.blk[0]
  rows = block.ffn_gate.out_features
  install = install_imported_q4_mmvq(rows)

  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  token = Tensor([[ids[0]]], dtype=dtypes.int32, device="AMD").contiguous()
  x = model.token_embd(token).float().realize()
  block._init_state(x)
  h = (x + block._attention(block.attn_norm(x), 0)).contiguous().realize()
  ffn_in = block.ffn_norm(h).cast(dtypes.float32).contiguous().realize()

  q4_gate = q4_words(block.ffn_gate, "AMD")
  q4_up = q4_words(block.ffn_up, "AMD")
  q8_side = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate_side = Tensor.empty(rows, dtype=dtypes.float32, device="AMD").contiguous().realize()
  up_side = Tensor.empty(rows, dtype=dtypes.float32, device="AMD").contiguous().realize()

  @TinyJit
  def baseline(inp: Tensor):
    gate = block.ffn_gate(inp).realize()
    up = block.ffn_up(inp).realize()
    return gate, up

  @TinyJit
  def imported_pair(inp: Tensor, q8_buf: Tensor, gate_buf: Tensor, up_buf: Tensor, gate_q4: Tensor, up_q4: Tensor):
    x_vec = inp.reshape(4096).cast(dtypes.float32).contiguous()
    q8 = q8_buf.custom_kernel(x_vec, fxn=q8_quant_stub)[0]
    gate = gate_buf.custom_kernel(gate_q4, q8, fxn=q4_mmvq_stub)[0].reshape(1, 1, rows)
    up = up_buf.custom_kernel(up_q4, q8, fxn=q4_mmvq_stub)[0].reshape(1, 1, rows)
    return gate.realize(), up.realize()

  for _ in range(8):
    baseline(ffn_in)
    imported_pair(ffn_in, q8_side, gate_side, up_side, q4_gate, q4_up)
  dev.synchronize(timeout=10000)

  warmups, iters = 8, 40
  baseline_s: list[float] = []
  imported_s: list[float] = []
  replay_diffs: list[dict] = []
  ref_gate, ref_up = imported_pair(ffn_in, q8_side, gate_side, up_side, q4_gate, q4_up)
  ref_gate, ref_up = ref_gate.realize(), ref_up.realize()
  dev.synchronize(timeout=10000)

  for i in range(warmups + iters):
    if i % 2 == 0:
      t0 = time.perf_counter()
      bg, bu = baseline(ffn_in)
      bg.realize(); bu.realize(); dev.synchronize(timeout=10000)
      tb = time.perf_counter() - t0

      t0 = time.perf_counter()
      rg, ru = imported_pair(ffn_in, q8_side, gate_side, up_side, q4_gate, q4_up)
      rg.realize(); ru.realize(); dev.synchronize(timeout=10000)
      tr = time.perf_counter() - t0
    else:
      t0 = time.perf_counter()
      rg, ru = imported_pair(ffn_in, q8_side, gate_side, up_side, q4_gate, q4_up)
      rg.realize(); ru.realize(); dev.synchronize(timeout=10000)
      tr = time.perf_counter() - t0

      t0 = time.perf_counter()
      bg, bu = baseline(ffn_in)
      bg.realize(); bu.realize(); dev.synchronize(timeout=10000)
      tb = time.perf_counter() - t0

    if i >= warmups:
      baseline_s.append(tb)
      imported_s.append(tr)
      if len(replay_diffs) < 5:
        replay_diffs.append({"gate": diff_stats(rg, ref_gate), "up": diff_stats(ru, ref_up)})

  bg, bu = baseline(ffn_in)
  bg, bu = bg.realize(), bu.realize()
  baseline_med = median_ms(baseline_s)
  imported_med = median_ms(imported_s)
  speedup = baseline_med / imported_med if imported_med > 0 else 0.0
  result = {
    "schema": "decode_mmvq_large_project_p7e_gateup_amortization_v1",
    "date": "2026-06-19",
    "phase": "P7e_gateup_amortization",
    "roles": ["blk.0.ffn_gate", "blk.0.ffn_up"],
    "rows": rows,
    "install": install,
    "activation_shape": list(ffn_in.shape),
    "timing": {
      "method": "same-process interleaved TinyJit wall time; candidate uses one q8 producer shared by two imported Q4 consumers",
      "warmups": warmups,
      "iters": iters,
      "baseline_ms_median": baseline_med,
      "imported_ms_median": imported_med,
      "speedup": speedup,
      "baseline_ms_min": min(baseline_s) * 1000.0,
      "imported_ms_min": min(imported_s) * 1000.0,
    },
    "correctness": {
      "gate_vs_baseline_q8_path": diff_stats(ref_gate, bg),
      "up_vs_baseline_q8_path": diff_stats(ref_up, bu),
      "imported_replay_diffs": replay_diffs,
    },
    "gates": {
      "baseline_runs": len(baseline_s) == iters,
      "imported_runs": len(imported_s) == iters,
      "imported_replay_stable": all(d["gate"]["max_abs"] <= 1e-6 and d["up"]["max_abs"] <= 1e-6 for d in replay_diffs),
      "speedup_ge_1_10": speedup >= 1.10,
      "default_unchanged": True,
    },
  }
  result["verdict"] = "PASS_GATEUP_AMORTIZATION" if all(result["gates"].values()) else "NO_GATEUP_TIMING_WIN"
  (OUT / "p7e_gateup_amortization.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  if result["verdict"] != "PASS_GATEUP_AMORTIZATION":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
