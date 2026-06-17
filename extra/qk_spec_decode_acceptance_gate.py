#!/usr/bin/env python3
"""Speculative-decoding ACCEPTANCE GATE (offline, algorithmic -- not a generation integration). Question: can a
cheap draft (Qwen3-1.7B) propose enough tokens that Qwen3-8B accepts to make spec decoding worth building?

Greedy spec simulation per prompt: from the TARGET's accepted context, the draft proposes K tokens greedily;
the target verifies all K positions in one teacher-forced pass; accept the longest matching prefix + 1 bonus
(the target always contributes >=1 token). accepted_per_pass = matched_prefix + 1, in [1, K+1]. Re-sync each
pass (draft drafts from the target's sequence). Greedy only (no sampling correction -- that's exact-equivalent
for the acceptance metric at temperature 0). Correct-but-slow: re-prefills both models from 0 each step.

Gate: accepted/pass <1.2 hard-fail, >=1.3 weak, >=1.5 strong, >=2.0 excellent. Speed is recorded separately and
is NOT acceptance proof. Run:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_spec_decode_acceptance_gate.py --k 4
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, sys, time

def _greedy_logits_argmax(model, Tensor, seq, pos):
  lg = model.logits(Tensor([seq], dtype="int32").contiguous(), 0).realize()
  return int(lg[0, pos].argmax().item())

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--target", default=os.environ.get("TARGET_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--draft", default=os.environ.get("DRAFT_MODEL", "/home/ubuntu/models/Qwen3-1.7B-Q8_0.gguf"))
  ap.add_argument("--prompts", default="bench/qk-spec-decode-acceptance/prompts.jsonl")
  ap.add_argument("--max-prompts", type=int, default=16)
  ap.add_argument("--k", type=int, default=4)
  ap.add_argument("--max-steps", type=int, default=10)
  ap.add_argument("--max-ctx", type=int, default=512)
  args = ap.parse_args()

  from tinygrad import Tensor
  from extra.llm_generate import load_model_and_tokenizer
  K, MAXC = args.k, args.max_ctx
  target, tok = load_model_and_tokenizer(args.target, MAXC, seed=20260617)
  draft, _ = load_model_and_tokenizer(args.draft, MAXC, seed=20260617)
  prompts = [json.loads(l) for l in pathlib.Path(args.prompts).read_text().splitlines() if l.strip()][:args.max_prompts]

  per_prompt, per_pos_accept, reject_hist, t_draft, t_tgt = [], [0] * K, [0] * (K + 1), 0.0, 0.0
  total_accepted, total_passes = 0, 0
  for pi, p in enumerate(prompts):
    ctx = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(p["text"])
    acc_this = []
    for step in range(args.max_steps):
      if len(ctx) + K + 1 > MAXC: break
      # draft proposes K greedily from the target's accepted ctx
      d = list(ctx); proposed = []
      t0 = time.perf_counter()
      for _ in range(K):
        nt = _greedy_logits_argmax(draft, Tensor, d, len(d) - 1); proposed.append(nt); d.append(nt)
      t_draft += time.perf_counter() - t0
      # target verifies: greedy at positions len(ctx)-1 .. len(ctx)+K-1 over ctx+proposed (one pass)
      t1 = time.perf_counter()
      lg = target.logits(Tensor([ctx + proposed], dtype="int32").contiguous(), 0).realize()
      t_tgt += time.perf_counter() - t1
      tg = [int(lg[0, len(ctx) - 1 + i].argmax().item()) for i in range(K + 1)]
      na = 0
      for i in range(K):
        if proposed[i] == tg[i]: na += 1; per_pos_accept[i] += 1
        else: break
      reject_hist[na] += 1
      accepted = proposed[:na] + [tg[na]]   # matched prefix + the target's bonus/correction token
      ctx += accepted; acc_this.append(na + 1)
      total_accepted += na + 1; total_passes += 1
      if tg[na] == getattr(tok, "eos_id", -1): break
    per_prompt.append({"cat": p["cat"], "passes": len(acc_this), "mean_accepted_per_pass": round(statistics.mean(acc_this), 3) if acc_this else 0})
    print(f"[{pi+1}/{len(prompts)}] {p['cat']:6} passes={len(acc_this)} acc/pass={per_prompt[-1]['mean_accepted_per_pass']}", file=sys.__stderr__)

  agg = total_accepted / total_passes if total_passes else 0
  pp = [r["mean_accepted_per_pass"] for r in per_prompt if r["passes"]]
  verdict = ("hard_fail(<1.2)" if agg < 1.2 else "weak_pass(>=1.3)" if agg < 1.5 else "strong_pass(>=1.5)" if agg < 2.0 else "excellent(>=2.0)")
  if 1.2 <= agg < 1.3: verdict = "marginal(1.2-1.3)"
  out = {"target": pathlib.Path(args.target).name, "draft": pathlib.Path(args.draft).name, "tokenizer_compatible": True,
         "K": K, "prompts": len(prompts), "total_passes": total_passes,
         "accepted_per_target_pass": round(agg, 3), "per_prompt_mean_min_max": [round(min(pp), 3), round(max(pp), 3)] if pp else None,
         "per_prompt_stdev": round(statistics.pstdev(pp), 3) if len(pp) > 1 else None,
         "per_position_accept_rate": [round(per_pos_accept[i] / total_passes, 3) for i in range(K)] if total_passes else [],
         "rejected_at_position_hist": reject_hist,
         "draft_wall_s": round(t_draft, 2), "target_wall_s": round(t_tgt, 2),
         "verdict": verdict, "per_prompt": per_prompt}
  print(f"\nK={K}: accepted/target-pass = {agg:.3f}  ({verdict})  [{len(prompts)} prompts, {total_passes} passes]", file=sys.__stderr__)
  print(f"per-position accept rate: {out['per_position_accept_rate']}  reject@pos hist: {reject_hist}", file=sys.__stderr__)
  print("@@RESULT@@" + json.dumps(out))

if __name__ == "__main__":
  main()
