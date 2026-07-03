#!/usr/bin/env python3
"""llama.cpp long-context KV slope benchmark.

Runs llama-bench text-generation at fixed context depths (`-d`) and fits:

  ms/token = A + B * ctx

Then compares B to the model's KV-cache byte formula. This is a reference
benchmark for llama.cpp behavior; it does not make tinygrad route decisions.

Outputs:
  bench/llama-kv-ctx-slope/<model-id>/{latest.json,summary.md}
"""
from __future__ import annotations

import argparse, json, math, pathlib, struct, subprocess
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_LLAMA_BENCH = "/home/ubuntu/env/llama.cpp/build/bin/llama-bench"


def _read_unpack(fmt:str, n:int, f) -> Any: return struct.unpack(fmt, f.read(n))[0]
def _read_str(f) -> str: return f.read(_read_u64(f)).decode("utf-8")
def _read_arr(f) -> list[Any]:
  item_reader, n = _READERS[_read_i32(f)], _read_u64(f)
  return [item_reader(f) for _ in range(n)]

_READERS:dict[int, Callable] = {8: _read_str, 9: _read_arr,
  **{t: (lambda f, fmt="<"+fmt, n=n: _read_unpack(fmt, n, f)) for t,fmt,n in
     [(0,"c",1),(1,"b",1),(2,"H",2),(3,"h",2),(4,"I",4),(5,"i",4),(6,"f",4),
      (7,"?",1),(10,"Q",8),(11,"q",8),(12,"d",8)]}}
_read_u32, _read_i32, _read_u64, _read_i64 = _READERS[4], _READERS[5], _READERS[10], _READERS[11]


def gguf_kv(path:pathlib.Path) -> dict[str, Any]:
  with path.open("rb") as f:
    magic, version, _n_tensors, n_kv = f.read(4), _read_i32(f), _read_i64(f), _read_i64(f)
    if magic != b"GGUF" or version not in (2, 3): raise ValueError(f"{path} is not a GGUF v2/v3 file")
    out:dict[str, Any] = {}
    for _ in range(n_kv):
      k, typ = _read_str(f), _read_i32(f)
      out[k] = _READERS[typ](f)
    return out


_DTYPE_BYTES = {
  "f32": 4.0, "f16": 2.0, "bf16": 2.0,
  "q8_0": 34.0 / 32.0,
  "q4_0": 18.0 / 32.0, "q4_1": 20.0 / 32.0,
  "q5_0": 22.0 / 32.0, "q5_1": 24.0 / 32.0,
  "iq4_nl": 18.0 / 32.0,
}


def model_meta(path:pathlib.Path, cache_k:str, cache_v:str) -> dict[str, Any]:
  kv = gguf_kv(path)
  arch = kv["general.architecture"]
  n_heads = int(kv[f"{arch}.attention.head_count"])
  n_kv_heads = int(kv[f"{arch}.attention.head_count_kv"])
  dim = int(kv[f"{arch}.embedding_length"])
  layers = int(kv[f"{arch}.block_count"] - kv.get(f"{arch}.nextn_predict_layers", 0))
  head_dim = int(kv.get(f"{arch}.attention.key_length_mla", kv.get(f"{arch}.attention.key_length", dim // n_heads)))
  bytes_k, bytes_v = _DTYPE_BYTES[cache_k], _DTYPE_BYTES[cache_v]
  kv_bytes_per_ctx_token = layers * n_kv_heads * head_dim * (bytes_k + bytes_v)
  return {
    "architecture": arch,
    "name": kv.get("general.name") or kv.get("general.basename") or path.stem,
    "layers": layers, "embedding_length": dim, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
    "head_dim": head_dim, "context_length": int(kv.get(f"{arch}.context_length", 0)),
    "cache_type_k": cache_k, "cache_type_v": cache_v,
    "bytes_per_k": bytes_k, "bytes_per_v": bytes_v,
    "kv_bytes_per_ctx_token": kv_bytes_per_ctx_token,
  }


def run_depth(args, depth:int) -> dict[str, Any]:
  cmd = [args.llama_bench, "-m", args.model, "-ngl", str(args.ngl), "-n", str(args.gen), "-p", "0",
         "-d", str(depth), "-r", str(args.reps), "-ctk", args.cache_type_k, "-ctv", args.cache_type_v,
         "-o", "json"]
  if args.flash_attn is not None: cmd += ["-fa", args.flash_attn]
  try:
    raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=args.timeout).decode()
    start = raw.find("[")
    rows = json.loads(raw[start:] if start >= 0 else raw)
    row = next(r for r in rows if r.get("n_gen") and not r.get("n_prompt"))
    tok_s = float(row["avg_ts"])
    return {
      "ctx": depth, "ok": True, "tok_s": tok_s, "stddev_tok_s": float(row.get("stddev_ts", 0.0)),
      "ms_per_token": 1000.0 / tok_s, "command": cmd,
      "llama": {k: row.get(k) for k in ("build_commit","build_number","gpu_info","backends","model_type",
                                         "model_size","model_n_params","type_k","type_v","flash_attn",
                                         "no_kv_offload","n_gpu_layers")},
    }
  except Exception as e:
    return {"ctx": depth, "ok": False, "error": str(e), "command": cmd}


def fit_linear(points:list[tuple[float, float]]) -> dict[str, float | None]:
  n = len(points)
  if n < 2: return {"a_ms": None, "b_ms_per_ctx": None, "r2": None}
  xs, ys = [p[0] for p in points], [p[1] for p in points]
  mx, my = sum(xs) / n, sum(ys) / n
  den = sum((x - mx) ** 2 for x in xs)
  if den == 0: return {"a_ms": None, "b_ms_per_ctx": None, "r2": None}
  b = sum((x - mx) * (y - my) for x, y in points) / den
  a = my - b * mx
  ss_tot = sum((y - my) ** 2 for y in ys)
  ss_res = sum((y - (a + b * x)) ** 2 for x, y in points)
  r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
  return {"a_ms": a, "b_ms_per_ctx": b, "r2": r2}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--id", default=None)
  ap.add_argument("--depths", default="512,1024,2048,4096,8192,16384,32768")
  ap.add_argument("--cache-type-k", default="f16", choices=sorted(_DTYPE_BYTES))
  ap.add_argument("--cache-type-v", default="f16", choices=sorted(_DTYPE_BYTES))
  ap.add_argument("--flash-attn", default=None, choices=("on", "off", "auto"))
  ap.add_argument("--target-id", default="amd_gfx1100")
  ap.add_argument("--baseline-json", default=None,
                  help="Optional f16/f16 slope artifact; used to compute storage-only B and quant residual.")
  ap.add_argument("--llama-bench", default=DEFAULT_LLAMA_BENCH)
  ap.add_argument("--ngl", type=int, default=99)
  ap.add_argument("--gen", type=int, default=128)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--timeout", type=int, default=1800)
  ap.add_argument("--out-root", default=str(ROOT / "bench" / "llama-kv-ctx-slope"))
  args = ap.parse_args()

  model_path = pathlib.Path(args.model)
  model_id = args.id or model_path.stem
  depths = [int(x) for x in args.depths.split(",") if x.strip()]
  meta = model_meta(model_path, args.cache_type_k, args.cache_type_v)

  rows = [run_depth(args, d) for d in depths]
  ok = [r for r in rows if r["ok"]]
  fit = fit_linear([(float(r["ctx"]), float(r["ms_per_token"])) for r in ok])
  b = fit["b_ms_per_ctx"]
  implied_bw = None
  if b is not None and b > 0:
    implied_bw = meta["kv_bytes_per_ctx_token"] / (b / 1000.0) / 1e9

  residual:dict[str, Any] = {}
  if args.baseline_json:
    base = json.loads(pathlib.Path(args.baseline_json).read_text())
    base_b = base.get("fit", {}).get("b_ms_per_ctx")
    base_bytes = base.get("meta", {}).get("kv_bytes_per_ctx_token")
    if base_b is None or base_bytes is None:
      raise ValueError("--baseline-json must contain fit.b_ms_per_ctx and meta.kv_bytes_per_ctx_token")
    storage_b = float(base_b) * float(meta["kv_bytes_per_ctx_token"]) / float(base_bytes)
    residual = {
      "baseline_path": str(pathlib.Path(args.baseline_json)),
      "baseline_b_ms_per_ctx": float(base_b),
      "baseline_kv_bytes_per_ctx_token": float(base_bytes),
      "baseline_cache_type_k": base.get("meta", {}).get("cache_type_k"),
      "baseline_cache_type_v": base.get("meta", {}).get("cache_type_v"),
      "storage_only_b_ms_per_ctx": storage_b,
      "quant_residual_b_ms_per_ctx": (float(b) - storage_b) if b is not None else None,
    }

  for r in rows:
    if not r["ok"]: continue
    ctx = int(r["ctx"])
    r["kv_storage_bytes"] = int(meta["kv_bytes_per_ctx_token"] * ctx)
    r["kv_read_bytes_per_token_est"] = int(meta["kv_bytes_per_ctx_token"] * ctx)
    if fit["a_ms"] is not None and b is not None:
      pred = float(fit["a_ms"]) + float(b) * ctx
      r["fit_ms_per_token"] = pred
      r["residual_ms"] = float(r["ms_per_token"]) - pred

  linear = fit["r2"] is not None and fit["r2"] > 0.97
  quantized_kv = args.cache_type_k not in ("f16", "bf16", "f32") or args.cache_type_v not in ("f16", "bf16", "f32")
  artifact = {
    "schema": "tinygrad.llama_kv_ctx_slope.v1",
    "model_id": model_id,
    "target_id": args.target_id,
    "model_path": str(model_path),
    "meta": meta,
    "config": {"depths": depths, "cache_type_k": args.cache_type_k, "cache_type_v": args.cache_type_v,
               "flash_attn": args.flash_attn or "default", "gen": args.gen, "reps": args.reps},
    "rows": rows,
    "fit": {**fit, "implied_kv_bandwidth_gb_s": implied_bw},
    "residual": residual,
    "verdict": "LLAMA_CTX_DECLINE_LINEAR_KV_SLOPE" if linear else "LLAMA_CTX_DECLINE_NEEDS_RESIDUAL_ANALYSIS",
    "interpretation_hint": ("quantized KV has a linear context slope, but storage bytes alone may not predict speed; "
                            "compare against an f16/bf16 baseline for quant/dequant or kernel overhead")
                           if quantized_kv else
                           "f16/bf16/f32 KV linear slope can be compared directly to plausible KV-read bandwidth",
  }

  out = pathlib.Path(args.out_root) / model_id
  out.mkdir(parents=True, exist_ok=True)
  (out / "latest.json").write_text(json.dumps(artifact, indent=2))
  lines = [
    "# llama.cpp KV context-slope benchmark",
    "",
    f"**Verdict:** {artifact['verdict']}",
    "",
    f"- model: `{model_id}`",
    f"- cache: K `{args.cache_type_k}`, V `{args.cache_type_v}`",
    f"- KV bytes/context token: `{meta['kv_bytes_per_ctx_token']:.0f}`",
    f"- fit: `ms/token = {fit['a_ms']:.6f} + {fit['b_ms_per_ctx']:.9f} * ctx`" if fit["a_ms"] is not None else "- fit: unavailable",
    f"- R^2: `{fit['r2']:.4f}`" if fit["r2"] is not None else "- R^2: unavailable",
    f"- implied KV bandwidth: `{implied_bw:.1f} GB/s`" if implied_bw else "- implied KV bandwidth: unavailable",
    *((
      f"- storage-only B from baseline: `{residual['storage_only_b_ms_per_ctx']:.9f} ms/ctx`",
      f"- quant residual B: `{residual['quant_residual_b_ms_per_ctx']:.9f} ms/ctx`",
    ) if residual and residual.get("quant_residual_b_ms_per_ctx") is not None else ()),
    "",
    "| ctx | ok | tok/s | ms/token | KV read/token | fit residual ms |",
    "|---:|:---:|---:|---:|---:|---:|",
  ]
  for r in rows:
    if r["ok"]:
      lines.append(f"| {r['ctx']} | yes | {r['tok_s']:.2f} | {r['ms_per_token']:.4f} | "
                   f"{r['kv_read_bytes_per_token_est']/1e9:.3f} GB | {r.get('residual_ms', 0.0):+.4f} |")
    else:
      lines.append(f"| {r['ctx']} | no | - | - | - | - |")
  (out / "summary.md").write_text("\n".join(lines) + "\n")
  print(f"wrote {out/'latest.json'} and {out/'summary.md'}")


if __name__ == "__main__":
  main()
