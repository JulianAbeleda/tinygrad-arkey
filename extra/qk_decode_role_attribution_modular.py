#!/usr/bin/env python3
"""Model-driven decode role attribution.

This is the non-hardcoded successor to the 8B-specific W1/system-residual tools.
It builds a role profile from the selected GGUF, classifies kernels by matching
kernel dimensions to that profile, and optionally captures eager per-kernel GPU
time for a decode step.

Examples:
  # Cheap profile/classifier proof, no GPU model load.
  PYTHONPATH=. python3 extra/qk_decode_role_attribution_modular.py \
    --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --id qwen3-14b --profile-only

  # Full capture. Route flags are intentionally external, so this works for
  # baseline, generated-anyshape, or future policy routes without script edits.
  DEV=AMD JIT=1 DECODE_Q4K_G3_ANYSHAPE=1 PYTHONPATH=. \
    python3 extra/qk_decode_role_attribution_modular.py \
    --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --id qwen3-14b --capture
"""
from __future__ import annotations

import argparse, collections, json, os, pathlib, subprocess, sys, textwrap
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extra.qk_decode_role_profile import DecodeRoleProfile, WeightRole, classify_kernel, profile_from_gguf, summarize_profile

DEFAULT_OUT = ROOT / "bench/qk-decode-role-attribution"
DEFAULT_CTXS = "512,4096"
DEFAULT_STEPS = 4
DEFAULT_MAX_CONTEXT = 4608

CHILD = r'''
import json, os
from tinygrad import Tensor, TinyJit, Context
from tinygrad.uop.ops import UOp
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent
from extra.llm_generate import load_model_and_tokenizer

MODEL=os.environ["QK_ATTR_MODEL"]
MAXC=int(os.environ.get("QK_ATTR_MAX_CONTEXT", "4608"))
CTX=int(os.environ["QK_ATTR_CTX"])
NSTEPS=int(os.environ.get("QK_ATTR_STEPS", "4"))

m,tok=load_model_and_tokenizer(MODEL, MAXC, seed=20260630)
for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
  lin.decode_enabled = True
use_flash = CTX >= int(os.environ.get("FLASH_DECODE_THRESHOLD", "512"))
for b in m.blk:
  b._use_flash, b._prefill_v2 = use_flash, False

v=UOp.variable("start_pos", 0, MAXC-1)
temp=Tensor([0.0])
step=TinyJit(m.forward)
tk=Tensor([[100]], dtype="int32").contiguous()
for i in range(4):
  step(tk, v.bind(CTX+i), temp).realize().item()

Compiled.profile_events=[]
with Context(PROFILE=1):
  for i in range(NSTEPS):
    m.forward(tk, v.bind(CTX+i), temp).realize().item()

agg={}; calls={}
for e in Compiled.profile_events:
  if isinstance(e, ProfileRangeEvent) and e.en is not None:
    nm=getattr(e.name, "name", None) or str(e.name)
    agg[nm]=agg.get(nm, 0.0)+float(e.en-e.st)
    calls[nm]=calls.get(nm, 0)+1
n=max(1, NSTEPS)
print("@@"+json.dumps({"ctx": CTX, "use_flash": use_flash, "nsteps": n,
  "per_kernel": {k: {"dur_per_step": round(agg[k]/n, 4), "calls_per_step": round(calls[k]/n, 2)}
                 for k in sorted(agg)}}))
'''


def _ctxs(s:str) -> list[int]:
  return [int(x) for x in s.split(",") if x.strip()]


def _model_id(path:str, explicit:str | None) -> str:
  return explicit or pathlib.Path(path).expanduser().stem.replace("-", "_").lower()


def capture(model:str, ctx:int, steps:int, max_context:int, extra_env:dict[str, str]) -> dict[str, Any]:
  env = {**os.environ, **extra_env, "DEV": os.environ.get("DEV", "AMD"), "JIT": os.environ.get("JIT", "1"),
         "PROFILE": "1", "PYTHONPATH": str(ROOT), "QK_ATTR_MODEL": str(pathlib.Path(model).expanduser()),
         "QK_ATTR_CTX": str(ctx), "QK_ATTR_STEPS": str(steps), "QK_ATTR_MAX_CONTEXT": str(max_context)}
  p = subprocess.run([sys.executable, "-c", CHILD], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=2400)
  line = [l for l in p.stdout.splitlines() if l.startswith("@@")]
  if not line:
    return {"failed": True, "returncode": p.returncode, "stdout_tail": p.stdout[-3000:], "stderr_tail": p.stderr[-3000:]}
  return json.loads(line[-1][2:])


def attribute_capture(cap:dict[str, Any], profile:DecodeRoleProfile) -> dict[str, Any]:
  pk = cap.get("per_kernel", {})
  total = sum(v["dur_per_step"] for v in pk.values()) or 1e-9
  rows = []
  for name, v in pk.items():
    c = classify_kernel(name, profile)
    dur = float(v["dur_per_step"])
    rows.append({**c, "calls_per_step": v["calls_per_step"], "dur_per_step": dur,
                 "pct_of_gpu_compute": round(100.0 * dur / total, 2)})
  rows.sort(key=lambda r: -r["dur_per_step"])

  by_bucket: dict[str, dict[str, Any]] = {}
  by_role: dict[str, dict[str, Any]] = {}
  for r in rows:
    for key, table in (("bucket", by_bucket), ("role", by_role)):
      name = r[key]
      e = table.setdefault(name, {"name": name, "dur": 0.0, "pct": 0.0, "quants": set(), "route_classes": set(),
                                  "kernels": 0, "bytes_per_step": 0})
      e["dur"] += r["dur_per_step"]
      e["pct"] += r["pct_of_gpu_compute"]
      e["quants"].add(r["quant"])
      e["route_classes"].add(r["route_class"])
      e["kernels"] += 1
      e["bytes_per_step"] += int(r.get("bytes_per_call") or 0) * float(r.get("calls_per_step") or 0)
  for table in (by_bucket, by_role):
    for e in table.values():
      e["dur"] = round(e["dur"], 4)
      e["pct"] = round(e["pct"], 2)
      e["quants"] = sorted(e["quants"])
      e["route_classes"] = sorted(e["route_classes"])
      e["bytes_per_step"] = int(e["bytes_per_step"])
  return {
    "ctx": cap.get("ctx"), "use_flash": cap.get("use_flash"), "total_dur_units": round(total, 4),
    "by_bucket": sorted(by_bucket.values(), key=lambda e: -e["dur"]),
    "by_role": sorted(by_role.values(), key=lambda e: -e["dur"]),
    "by_kernel_top": rows[:40],
  }


def self_test() -> dict[str, Any]:
  profile = DecodeRoleProfile(
    model_id="synthetic_qwen3_14b", model_path="<synthetic>", arch="qwen3",
    hidden=5120, ffn=17408, vocab=151936, layers=40,
    weights=(
      # Counts mimic repeated layer roles; exact count is not used for matching.
      WeightRole("ffn_gate_up", "blk.0.ffn_gate.weight", 17408, 5120, 12, "Q4_K", 80),
      WeightRole("ffn_down", "blk.0.ffn_down.weight", 5120, 17408, 14, "Q6_K", 40),
      WeightRole("attn_qo", "blk.0.attn_q.weight", 5120, 5120, 12, "Q4_K", 80),
      WeightRole("attn_kv", "blk.0.attn_k.weight", 1024, 5120, 12, "Q4_K", 80),
      WeightRole("lm_head", "output.weight", 151936, 5120, 14, "Q6_K", 1),
    ))
  cases = {
    "q4k_g3_lanemap_gemv_17408_5120": ("q4k_gemv", "ffn_gate_up", "Q4_K"),
    "q6k_coop_partial_5120_17408": ("q6k_gemv", "ffn_down", "Q6_K"),
    "q6k_coop_partial_151936_5120": ("lm_head", "lm_head", "Q6_K"),
    "r_32_4_1187": ("reduce_partial", "other", "unknown"),
  }
  rows = []
  ok = True
  for name, want in cases.items():
    got = classify_kernel(name, profile)
    have = (got["bucket"], got["role"], got["quant"])
    rows.append({"kernel": name, "want": want, "have": have, "reduce_class": got.get("reduce_class")})
    ok &= have == want
  return {"verdict": "QK_DECODE_ROLE_PROFILE_SELFTEST_PASS" if ok else "QK_DECODE_ROLE_PROFILE_SELFTEST_FAIL",
          "rows": rows}


def write_summary(out:pathlib.Path, rec:dict[str, Any]) -> None:
  lines = [
    f"# Decode Role Attribution — {rec['model_id']}",
    "",
    f"Verdict: **{rec['verdict']}**",
    "",
    "## Profile",
    "",
    f"- arch: `{rec['profile'].get('arch')}`",
    f"- hidden: `{rec['profile'].get('hidden')}`",
    f"- ffn: `{rec['profile'].get('ffn')}`",
    f"- vocab: `{rec['profile'].get('vocab')}`",
    f"- layers: `{rec['profile'].get('layers')}`",
    "",
    "## Role Shapes",
    "",
    "| role | shape rows x cols | quant | count |",
    "|---|---:|---|---:|",
  ]
  for w in rec["profile"]["weights"]:
    lines.append(f"| {w['role']} | {w['rows']} x {w['cols']} | {w['quant']} | {w['count']} |")
  if rec.get("per_ctx"):
    for ctx, cap in rec["per_ctx"].items():
      lines += ["", f"## ctx {ctx}", "", "| bucket | pct gpu-compute | quants | routes |",
                "|---|---:|---|---|"]
      if "by_bucket" in cap:
        for r in cap["by_bucket"][:12]:
          lines.append(f"| {r['name']} | {r['pct']} | {','.join(r['quants'])} | {','.join(r['route_classes'])} |")
      else:
        lines.append(f"| capture_failed | 100 | — | — |")
  lines += ["", "Classifier source: GGUF tensor table, not model-size constants.", ""]
  (out / "summary.md").write_text("\n".join(lines))


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default=os.environ.get("QK_MODEL"), help="GGUF path. Defaults to QK_MODEL.")
  ap.add_argument("--id", default=None, help="Artifact/model id.")
  ap.add_argument("--ctxs", default=os.environ.get("QK_CKPTS", DEFAULT_CTXS))
  ap.add_argument("--steps", type=int, default=int(os.environ.get("QK_ATTR_STEPS", DEFAULT_STEPS)))
  ap.add_argument("--max-context", type=int, default=int(os.environ.get("QK_ATTR_MAX_CONTEXT", DEFAULT_MAX_CONTEXT)))
  ap.add_argument("--out-root", default=str(DEFAULT_OUT))
  ap.add_argument("--profile-only", action="store_true", help="Write profile/classifier artifact but do not run GPU capture.")
  ap.add_argument("--capture", action="store_true", help="Run eager PROFILE capture for each context.")
  ap.add_argument("--self-test", action="store_true", help="Run the import-safe classifier self-test and exit.")
  args = ap.parse_args()

  if args.self_test:
    rec = self_test()
    print(json.dumps(rec, indent=2))
    return 0 if rec["verdict"].endswith("_PASS") else 1
  if not args.model:
    raise SystemExit("--model or QK_MODEL is required")

  model = str(pathlib.Path(args.model).expanduser())
  mid = _model_id(model, args.id)
  out = pathlib.Path(args.out_root) / mid
  out.mkdir(parents=True, exist_ok=True)
  profile = profile_from_gguf(model, mid)

  rec: dict[str, Any] = {
    "schema": "qk_decode_role_attribution_modular_v1",
    "model_id": mid, "model": model, "profile": profile.to_json(),
    "profile_summary": summarize_profile(profile),
    "capture": {"enabled": bool(args.capture), "ctxs": _ctxs(args.ctxs), "steps": args.steps,
                "max_context": args.max_context},
    "per_ctx": {},
  }
  (out / "profile.json").write_text(json.dumps(rec["profile_summary"], indent=2) + "\n")

  if args.capture:
    for ctx in _ctxs(args.ctxs):
      cap = capture(model, ctx, args.steps, args.max_context, {})
      rec["per_ctx"][str(ctx)] = attribute_capture(cap, profile) if "per_kernel" in cap else cap
  elif not args.profile_only:
    rec["note"] = "No GPU capture requested. Use --capture for per-kernel GPU attribution."

  ok = all(not v.get("failed") for v in rec["per_ctx"].values()) if rec["per_ctx"] else True
  rec["verdict"] = "QK_DECODE_ROLE_ATTRIBUTION_PASS" if ok else "QK_DECODE_ROLE_ATTRIBUTION_BLOCKED_CAPTURE"
  (out / "latest.json").write_text(json.dumps(rec, indent=2) + "\n")
  if rec.get("per_ctx"):
    (out / "kernel_taxonomy.json").write_text(json.dumps({k: v.get("by_bucket", []) for k, v in rec["per_ctx"].items()}, indent=2) + "\n")
  write_summary(out, rec)
  print(f"{mid}: {rec['verdict']} -> {out}")
  if args.capture:
    for ctx, cap in rec["per_ctx"].items():
      if "by_bucket" in cap:
        print(f"ctx{ctx}:", [(r["name"], r["pct"], r["quants"], r["route_classes"]) for r in cap["by_bucket"][:8]])
      else:
        print(f"ctx{ctx}: capture failed")
  else:
    roles = collections.Counter(w.role for w in profile.weights)
    print("profile roles:", dict(sorted(roles.items())))
    print(textwrap.dedent("""\
      Capture command example:
        DEV=AMD JIT=1 DECODE_Q4K_G3_ANYSHAPE=1 PYTHONPATH=. python3 extra/qk_decode_role_attribution_modular.py \\
          --model {model} --id {mid} --capture
    """).format(model=model, mid=mid).rstrip())
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
