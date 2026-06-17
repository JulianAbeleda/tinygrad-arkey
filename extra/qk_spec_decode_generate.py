#!/usr/bin/env python3
# STATUS: REFUTED integration (not wired into defaults). Acceptance gate PASSED but naive two-model integration
# is jit-alternation runtime-bound (~0.24x). See docs/qk-spec-decode-integration-result-20260617.md.
"""Speculative-decoding generation prototype (gated, standalone -- does NOT modify model.py defaults). Realizes
the ~1.6x the acceptance gate projected, using incremental KV (no per-step recompile):
  - draft proposes K tokens via its jit'd T=1 decode (rollout_jit, symbolic start_pos, greedy temp=0)
  - target verifies all K+1 positions in one jit'd T=(K+1) pass (custom argmax jit over target.logits)
  - accept the longest matching prefix + 1 bonus; advance; both KV caches self-correct (each pass re-processes
    the last accepted token at its position, overwriting any stale speculative entry)
Greedy (temperature 0) -> spec output MUST equal target-only greedy (exactness check built in). Measures e2e
wall tok/s vs target-only baseline. First perf gate: >=1.2x.

Run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_spec_decode_generate.py --k 3 --n-new 64 --prompts 6
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, sys, time

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--target", default=os.environ.get("TARGET_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--draft", default=os.environ.get("DRAFT_MODEL", "/home/ubuntu/models/Qwen3-0.6B-Q8_0.gguf"))
  ap.add_argument("--prompts", default="bench/qk-spec-decode-acceptance/prompts.jsonl")
  ap.add_argument("--num-prompts", type=int, default=6)
  ap.add_argument("--k", type=int, default=3)
  ap.add_argument("--n-new", type=int, default=64)
  ap.add_argument("--max-ctx", type=int, default=1024)
  args = ap.parse_args()

  from tinygrad import Tensor, UOp, TinyJit, dtypes
  from extra.llm_generate import load_model_and_tokenizer
  K, MAXC = args.k, args.max_ctx
  target, tok = load_model_and_tokenizer(args.target, MAXC, seed=20260617)
  draft, _ = load_model_and_tokenizer(args.draft, MAXC, seed=20260617)
  temp0 = Tensor([0.0]); v_sp = UOp.variable("start_pos", 0, MAXC - 1)
  for b in target.blk: b._use_flash, b._prefill_v2 = False, False
  for lin in target._q4k_linears.linears: lin.decode_enabled = False
  verify_jit = TinyJit(lambda toks, spv: target.logits(toks, spv).argmax(-1).cast(dtypes.int32).realize())

  def spec_decode(prompt_ids, n_new):
    # prefill both (eager, one-time, fills caches to L0) -- UNTIMED (prefill is not the decode-speed question)
    for lin in draft._q4k_linears.linears: lin.decode_enabled = False
    target.logits(Tensor([prompt_ids], dtype="int32").contiguous(), 0).realize()
    draft.logits(Tensor([prompt_ids], dtype="int32").contiguous(), 0).realize()
    L = len(prompt_ids); last = int(prompt_ids[-1]); out = []
    passes = 0
    _t_decode = time.perf_counter()   # time ONLY the decode loop, not the prefill
    while len(out) < n_new:
      # draft propose K -- feed the DEVICE output token back (out->out), no per-step Tensor creation (that
      # per-step host creation was the spec 0.15x cause; normal decode is GPU-bound -- runtime accounting).
      proposed, cur_t, pos = [], Tensor([[last]], dtype="int32").contiguous(), L - 1
      for _ in range(K):
        cur_t = draft(cur_t, v_sp.bind(pos), temp0); proposed.append(int(cur_t.item())); pos += 1
      # cache proposed[-1] (an output, never an input) so the draft KV is valid through L+K-1 on full accept.
      draft(cur_t, v_sp.bind(pos), temp0).realize()
      tgt = verify_jit(Tensor([[last] + proposed], dtype="int32").contiguous(), v_sp.bind(L - 1))  # [1,K+1] argmax
      tg = [int(tgt[0, i].item()) for i in range(K + 1)]
      n = 0
      for i in range(K):
        if proposed[i] == tg[i]: n += 1
        else: break
      new = proposed[:n] + [tg[n]]                       # n accepted + 1 bonus/correction
      out += new; L += len(new); last = tg[n]; passes += 1
      if last == getattr(tok, "eos_id", -1): break
    return out[:n_new], passes, time.perf_counter() - _t_decode

  def baseline_decode(prompt_ids, n_new):
    # time decode-only: start the clock after the first token (prefill done)
    out = []; t_start = None
    for t in target.generate(list(prompt_ids), temperature=0.0):
      if t_start is None: t_start = time.perf_counter()   # prefill produced the first token; time the rest
      else: out.append(t)
      if len(out) >= n_new or t == getattr(tok, "eos_id", -1): break
    return out, time.perf_counter() - t_start

  prompts = [json.loads(l)["text"] for l in pathlib.Path(args.prompts).read_text().splitlines() if l.strip()][:args.num_prompts]
  # warm-up: compile both paths' jits once on the first prompt (discarded) so timed runs are warm, not compile-bound
  wids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(prompts[0])
  target._cached_tokens = []; baseline_decode(wids, 8)
  target._cached_tokens = []; spec_decode(wids, 8)
  rows, all_match = [], True
  for pi, ptext in enumerate(prompts):
    ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(ptext)
    if len(ids) + args.n_new + K + 2 > MAXC: ids = ids[:MAXC - args.n_new - K - 2]
    # per-prompt warm-up (prefill shape differs per prompt) then timed DECODE-ONLY run -- for both paths
    target._cached_tokens = []; baseline_decode(ids, 4)
    target._cached_tokens = []; base, base_dt = baseline_decode(ids, args.n_new)
    target._cached_tokens = []; spec_decode(ids, 4)
    target._cached_tokens = []; spec, passes, spec_dt = spec_decode(ids, args.n_new)
    match = spec[:len(base)] == base[:len(spec)]
    all_match = all_match and match
    rows.append({"prompt": pi, "tokens": len(spec), "passes": passes, "accepted_per_pass": round(len(spec) / passes, 2) if passes else 0,
                 "baseline_tok_s": round(len(base) / base_dt, 2), "spec_tok_s": round(len(spec) / spec_dt, 2),
                 "speedup": round((len(spec) / spec_dt) / (len(base) / base_dt), 3), "greedy_match": match})
    print(f"[{pi+1}/{len(prompts)}] base {rows[-1]['baseline_tok_s']:6.1f} | spec {rows[-1]['spec_tok_s']:6.1f} tok/s "
          f"-> {rows[-1]['speedup']}x | acc/pass {rows[-1]['accepted_per_pass']} | match={match}", file=sys.__stderr__)

  med_speedup = statistics.median([r["speedup"] for r in rows])
  out = {"target": pathlib.Path(args.target).name, "draft": pathlib.Path(args.draft).name, "K": K, "n_new": args.n_new,
         "rows": rows, "median_speedup": round(med_speedup, 3), "greedy_exact_all": all_match,
         "median_accepted_per_pass": round(statistics.median([r["accepted_per_pass"] for r in rows]), 2),
         "perf_gate_1p2x": bool(med_speedup >= 1.2 and all_match),
         "verdict": (f"PASS: spec decode {med_speedup:.2f}x median e2e (>=1.2x), greedy-exact={all_match}"
                     if (med_speedup >= 1.2 and all_match) else
                     f"BELOW GATE: {med_speedup:.2f}x median, greedy-exact={all_match}")}
  print(f"\nmedian speedup {med_speedup:.3f}x | greedy-exact {all_match} | {out['verdict']}", file=sys.__stderr__)
  print("@@RESULT@@" + json.dumps(out))

if __name__ == "__main__":
  main()
