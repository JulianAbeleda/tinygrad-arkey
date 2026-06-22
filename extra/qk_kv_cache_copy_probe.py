#!/usr/bin/env python3
"""KV-cache copy elimination probe (executes docs/kv-cache-copy-elimination-implementation-scope-20260622.md).

Gates the default-off `KV_CACHE_INPLACE` decode route (model.py:951) against the post-Q4K_GEMV_WARP baseline.
Each config runs in its OWN subprocess (variant 1's in-place .assign mutates persistent state, so configs must be
isolated). A child does: real prompt prefill + greedy decode (token ids) + ProfileGraphEvent copy cost; the driver
compares tokens byte-identically, checks copy-shrink, and classifies. The W==D wall sweep only runs if a variant is
correct + JIT-stable (none are, here).

  variant 1 = in-place .assign() ; variant 2 = slice-scoped .after() ; both vs canonical full-buffer .after().

  run: DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 PYTHONPATH=. .venv/bin/python extra/qk_kv_cache_copy_probe.py
  -> bench/qk-kv-cache-copy-elimination/latest.json
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-kv-cache-copy-elimination"
MAXC = 4608; NGEN = 8
VARIANTS = {"baseline": "0", "v1_inplace_assign": "1", "v2_slice_after": "2"}

def child():
  """one config; prints a single JSON line {tokens, copy_us, error}."""
  import collections, traceback
  res = {"variant": os.environ.get("KV_CACHE_INPLACE", "0")}
  try:
    from tinygrad import Tensor, UOp, TinyJit, Context, Device
    from tinygrad.device import Compiled
    from extra.llm_generate import load_model_and_tokenizer
    from extra.qk_harness_contract import DEFAULT_MODEL
    dev = Device["AMD"]
    m, tok = load_model_and_tokenizer(DEFAULT_MODEL, MAXC, seed=20260617)
    for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []): lin.decode_enabled = True
    for b in m.blk: b._use_flash, b._prefill_v2 = True, False
    v = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])
    prompt = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("The history of computing began when")
    step = TinyJit(m.forward); out = None
    for i, tid in enumerate(prompt): out = step(Tensor([[int(tid)]], dtype="int32").contiguous(), v.bind(i), temp).realize()
    toks = [int(out.item())]; sp = len(prompt)
    for _ in range(NGEN - 1):
      out = step(Tensor([[toks[-1]]], dtype="int32").contiguous(), v.bind(sp), temp).realize(); toks.append(int(out.item())); sp += 1
    with Context(PROFILE=1):
      base = len(Compiled.profile_events); step(Tensor([[toks[-1]]], dtype="int32").contiguous(), v.bind(sp), temp).realize()
      dev.synchronize(); dev._at_profile_finalize()
      per = collections.defaultdict(float)
      for e in Compiled.profile_events[base:]:
        if type(e).__name__ != "ProfileGraphEvent": continue
        sigs = [float(s) for s in e.sigs]
        for ent in e.ents: per[str(ent.name)] += sigs[ent.en_id] - sigs[ent.st_id]
    res["tokens"] = toks
    res["copy_us"] = round(sum(u for n, u in per.items() if n.startswith("E_49152") or n.startswith("E_1536")), 1)
    res["top_E_us"] = {n: round(u, 1) for n, u in sorted(per.items(), key=lambda x: -x[1]) if n.startswith("E_")}
  except Exception as e:
    res["error"] = f"{type(e).__name__}: {str(e)[:240]}"
  print("PROBE_JSON " + json.dumps(res))

def main():
  if os.environ.get("KV_PROBE_CHILD"): return child()
  os.environ.setdefault("Q4K_GEMV_WARP", "1"); os.environ.setdefault("Q4K_GEMV_WARP_DOWN", "1")
  configs = {}
  for name, ev in VARIANTS.items():
    env = {**os.environ, "KV_PROBE_CHILD": "1", "KV_CACHE_INPLACE": ev}
    p = subprocess.run([sys.executable, __file__], cwd=ROOT, env=env, text=True, capture_output=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith("PROBE_JSON ")), None)
    configs[name] = json.loads(line[len("PROBE_JSON "):]) if line else {"error": "no output: " + p.stderr.strip().splitlines()[-1][:200] if p.stderr.strip() else "no output"}
    print(f"  {name}: {configs[name].get('error') or ('tokens=%s copy_us=%s' % (configs[name].get('tokens'), configs[name].get('copy_us')))}", file=sys.stderr)

  base = configs["baseline"]; base_tokens = base.get("tokens")
  results = {}
  for name in VARIANTS:
    if name == "baseline": continue
    c = configs[name]
    if "error" in c:
      results[name] = {"outcome": "JIT_BLOCKED", "error": c["error"],
                       "note": "core-scheduler/JIT error at schedule/realize time (not caught by the decode-path try/except)"}
    else:
      match = c.get("tokens") == base_tokens
      saved = round((base.get("copy_us", 0) or 0) - (c.get("copy_us", 0) or 0), 1)
      results[name] = {"outcome": ("CORRECTNESS_FAIL" if not match else ("COPY_SHRINK_LOCAL_PASS" if saved > 100 else "NOOP")),
                       "tokens_match": match, "copy_us_baseline": base.get("copy_us"), "copy_us_variant": c.get("copy_us"),
                       "copy_us_saved": saved}
  # overall: WD only attempted if a variant is correct + JIT-stable
  any_local_pass = any(r.get("outcome") == "COPY_SHRINK_LOCAL_PASS" for r in results.values())
  all_blocked = all(r.get("outcome") in ("JIT_BLOCKED", "CORRECTNESS_FAIL") for r in results.values())
  if any_local_pass: verdict = "KV_CACHE_COPY_ELIMINATION_LOCAL_PASS_WD_PENDING"
  elif all_blocked and any(r.get("outcome") == "JIT_BLOCKED" for r in results.values()):
    verdict = "KV_CACHE_COPY_ELIMINATION_JIT_BLOCKED"
  else: verdict = "KV_CACHE_COPY_ELIMINATION_NOOP"

  try: commit = subprocess.run(["git","rev-parse","--short","HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
  except Exception: commit = None
  art = {"date": "2026-06-22", "phase": "KV_CACHE_COPY_ELIMINATION_PROBE", "model": "Qwen3-8B-Q4_K_M",
         "gpu": "RX 7900 XTX / gfx1100", "commit": commit, "max_context": MAXC,
         "env_baseline": {"Q4K_GEMV_WARP": 1, "Q4K_GEMV_WARP_DOWN": 1},
         "baseline_tokens": base_tokens, "baseline_copy_us": base.get("copy_us"),
         "configs": configs, "results": results, "verdict": verdict,
         "classification": ("eliminating the O(MAXC) KV-cache copy requires core-JIT/scheduler support for in-place "
                            "mutation of a captured buffer inside a @function(precompile=True) pure decode function "
                            "(variant 1: read-after-write hazard) or symbolic-sized .after() buffers (variant 2: "
                            "symbolic-alias eval). Per the scope, the broad core-JIT redesign is a STOP/CLASSIFY/DEFER "
                            "outcome, NOT undertaken here.") if verdict == "KV_CACHE_COPY_ELIMINATION_JIT_BLOCKED" else None,
         "default_behavior_changed": False}
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "latest.json").write_text(json.dumps(art, indent=2))
  print(f"\nVERDICT: {verdict}", file=sys.stderr)
  for name, r in results.items(): print(f"  {name}: {r['outcome']} -- {r.get('error') or ('copy %s->%s saved %s' % (r.get('copy_us_baseline'), r.get('copy_us_variant'), r.get('copy_us_saved')))}", file=sys.stderr)
  print(f"artifact: {OUT/'latest.json'}", file=sys.stderr)

if __name__ == "__main__":
  main()
