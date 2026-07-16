#!/usr/bin/env python3
"""Authority decode benchmark + matched-context llama.cpp comparison for the per-model bench docs.

Produces fixed-context authority artifacts alongside the older generate-window E2E artifacts:
  - decode: extra/qk/decode_runtime_overhead.py (clean W==D, TinyJit, synced, NMEAS=40, fixed context,
    shipped FLASH_DECODE_THRESHOLD so the owned-attention route fires at ctx>=512). tok_s_W per ctx.
  - llama.cpp: llama-bench tg128 at MATCHED depth (-d <ctx>) so decode is compared at the same context, and
    pp512 for prefill reference.
Key difference from model_e2e_bench: decode is measured at FIXED contexts (128 and 512), not a median over a
growing-context generate window, and llama is compared at the same depth. Adopting this for README tables is a
methodology decision, not a mechanical replacement.

Writes bench/models/qwen/data/amd-gfx1100/<id>.authority.json

Usage:
  python extra/llm/model_authority_bench.py --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --id qwen3-8b \
      --out bench/models/qwen/data/amd-gfx1100/qwen3-8b.authority.json
"""
from __future__ import annotations
import os, sys, json, argparse, subprocess, pathlib, tempfile

from extra.llm.llama_bench import (ARTIFACT_VERSION, atomic_write_json, build_llama_bench_cmd, model_identity,
  run_llama_bench_cmd, llama_pp_row, llama_tg_rows, summarize_row)

ROOT = pathlib.Path(__file__).resolve().parents[2]

def run_decode_authority(model:str, ckpts:str, reps:int, decode_tokens:int, artifact_dir:pathlib.Path) -> dict:
  env = {**os.environ, "DEV": "AMD", "JIT": "1", "PYTHONPATH": str(ROOT), "QK_MODEL": model, "QK_CKPTS": ckpts}
  artifact_dir.mkdir(parents=True, exist_ok=True)
  fd, name = tempfile.mkstemp(prefix="decode-authority-", suffix=".json", dir=artifact_dir)
  os.close(fd); os.unlink(name)
  depths = [int(x) for x in ckpts.split(",")]
  max_context = max(depths) + decode_tokens + 1
  cmd = [sys.executable, "extra/qk/decode_runtime_overhead.py", "--model", model, "--ckpts", ckpts,
         "--max-context", str(max_context), "--nmeas", str(decode_tokens), "--reps", str(reps), "--out", name]
  try:
    subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800, check=True)
    res = json.loads(pathlib.Path(name).read_text())
  finally:
    pathlib.Path(name).unlink(missing_ok=True)
  res.setdefault("producer_command", cmd)
  return res

def run_llama_matched(model:str, depths:list[int], reps:int=3, *, bin=None, ngl:int=99,
                      ctk:str="f16", ctv:str="f16", fa:bool=False, decode_tokens:int=128) -> dict:
  # decode at matched depths + pp512
  d_arg = ",".join(str(d) for d in depths)
  spec = ["-n", decode_tokens, "-d", d_arg, "-p", "512", "-ctk", ctk, "-ctv", ctv, "-fa", "1" if fa else "0"]
  kw = {"reps": reps, "ngl": ngl}
  if bin is not None: kw["bin"] = bin
  cmd = build_llama_bench_cmd(model, spec, **kw)
  rows = run_llama_bench_cmd(cmd)
  tg_rows = llama_tg_rows(rows)
  if any(int(r.get("n_gen", -1)) != decode_tokens for r in tg_rows):
    raise ValueError("llama.cpp decode token count does not match requested count")
  if any(r.get("model_filename") and pathlib.Path(r["model_filename"]).name != pathlib.Path(model).name for r in rows):
    raise ValueError("llama.cpp reported a different model filename")
  decode_by_depth = {str(r.get("n_depth", 0)): summarize_row(r, reps) for r in tg_rows}
  pp = llama_pp_row(rows)
  prefill = {"pp": pp["n_prompt"], **summarize_row(pp, reps)} if pp else None
  return {"decode_by_depth": decode_by_depth, "prefill_pp512": prefill, "reps": reps,
          "settings": {"depths": depths, "decode_tokens": decode_tokens, "ctk": ctk, "ctv": ctv, "fa": fa, "ngl": ngl},
          "tool": {"command": cmd, "binary": cmd[0], "build_commit": (rows[0].get("build_commit") if rows else None)},
          "raw_rows": rows}

def validate_matched(dec:dict, llama:dict, identity:dict, depths:list[int], reps:int, decode_tokens:int) -> None:
  if dec.get("schema") != "tinygrad.decode.fixed_depth.v2" or dec.get("artifact_version") != 2:
    raise ValueError("tinygrad producer returned an unsupported fixed-depth artifact")
  if not dec.get("model_identity", {}).get("path"): raise ValueError("tinygrad artifact has no model identity")
  if pathlib.Path(dec["model_identity"]["path"]).resolve() != pathlib.Path(identity["path"]):
    raise ValueError("tinygrad artifact model does not match requested model")
  for field in ("size_bytes", "mtime_ns"):
    if field in dec["model_identity"] and dec["model_identity"][field] != identity.get(field):
      raise ValueError(f"tinygrad artifact model {field} does not match requested model")
  if list(dec.get("ckpts", depths)) != depths: raise ValueError("tinygrad contexts do not match requested depths")
  if int(dec.get("reps", -1)) != reps: raise ValueError("tinygrad reps do not match requested reps")
  if int(dec.get("nmeas", -1)) != decode_tokens: raise ValueError("tinygrad decode token count does not match requested count")
  if sorted(map(int, llama["decode_by_depth"])) != sorted(depths): raise ValueError("llama.cpp did not return every matched depth")
  settings = llama.get("settings", {})
  if settings.get("depths") != depths or settings.get("decode_tokens") != decode_tokens or llama.get("reps") != reps:
    raise ValueError("llama.cpp artifact workload does not match requested workload")
  if dec.get("runtime_settings", {}).get("kv_cache") == "fp16" and (settings.get("ctk"), settings.get("ctv")) != ("f16", "f16"):
    raise ValueError("llama.cpp KV types do not match tinygrad fp16 KV")

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--id", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--ckpts", default="128,512")
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--decode-tokens", type=int, default=128)
  ap.add_argument("--ctk", default="f16"); ap.add_argument("--ctv", default="f16")
  ap.add_argument("--fa", action="store_true"); ap.add_argument("--ngl", type=int, default=99)
  args = ap.parse_args()

  ckpt_ints = [int(x) for x in args.ckpts.split(",")]
  identity = model_identity(args.model)
  dec = run_decode_authority(args.model, args.ckpts, args.reps, args.decode_tokens, pathlib.Path(args.out).parent)
  llama = run_llama_matched(args.model, ckpt_ints, args.reps, ngl=args.ngl, ctk=args.ctk, ctv=args.ctv,
                            fa=args.fa, decode_tokens=args.decode_tokens)
  validate_matched(dec, llama, identity, ckpt_ints, args.reps, args.decode_tokens)

  # build matched-context comparison rows
  comp = []
  for row in dec["rows"]:
    ctx = row["ctx"]
    tg = row["tok_s_W"]
    lc = llama["decode_by_depth"].get(str(ctx), {}).get("median_tok_s")
    comp.append({"ctx": ctx, "tinygrad_route": row["route"], "tinygrad_routes": row["routes"],
                 "flash_route": row["flash"], "tinygrad_tok_s_W": tg,
                 "llama_tok_s": lc, "ratio_pct": round(tg / lc * 100, 1) if (tg and lc) else None,
                 "host_sync_pct": row["host_sync_pct_of_wall"]})

  artifact = {
    "artifact_schema": "tinygrad.matched-llama-authority", "artifact_version": ARTIFACT_VERSION,
    "id": args.id,
    "model_identity": identity,
    "model_id": dec.get("model_id"),
    "hardware": dec.get("hardware"),
    "decode_authority": dec,
    "llama_cpp": llama,
    "decode_matched_comparison": comp,
    "provenance": {"command": [sys.executable, *sys.argv], "fixed_contexts": ckpt_ints,
      "decode_tokens": args.decode_tokens, "reps": args.reps,
      "tinygrad": {"command": dec.get("producer_command"), "runtime_settings": dec.get("runtime_settings"),
                   "route_by_context": {str(r["ctx"]): r["route"] for r in dec["rows"]}},
      "llama_cpp": llama["tool"] | {"kv": {"ctk": args.ctk, "ctv": args.ctv}, "fa": args.fa, "ngl": args.ngl}},
    "timing_authority": "clean W==D qk_decode_runtime_overhead (TinyJit, synced, NMEAS=40, fixed ctx, shipped flash threshold)",
    "note": "decode compared to llama.cpp tg128 at MATCHED depth (-d). Prefill authority (tuned PREFILL_V2 path) is "
            "recorded separately where measured (8B: qk_prefill_authority_refresh); llama prefill is pp512.",
  }
  outp = pathlib.Path(args.out)
  atomic_write_json(outp, artifact)
  for c in comp:
    r = f"{c['ratio_pct']}%" if c["ratio_pct"] else "—"
    host = f"{c['host_sync_pct']:.1f}%" if c["host_sync_pct"] is not None else "N/A"
    print(f"{args.id} ctx{c['ctx']}{'F' if c['flash_route'] else ' '}: tinygrad {c['tinygrad_tok_s_W']} vs "
          f"llama {c['llama_tok_s']} = {r}  (host-sync {host})")
  print(f"wrote {outp}")

if __name__ == "__main__":
  main()
