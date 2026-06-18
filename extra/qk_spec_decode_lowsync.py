"""Low-sync speculative decode (Phases 5-8). SPEC_DECODE arc. Default OFF; greedy-only; behind this script.

Uses the proven reusable proposal graph (Phase 4): draft proposes K via ONE captured TinyJit with device-token
feedback + K distinct rebindable start_pos vars (no per-step .item(), no recompile). Target verifies T=K+1 in one
jit. Accept = longest matching prefix + 1 correction. KV self-corrects (both caches re-process from the corrected
position next pass; draft full-accept hole closed by a (K+1)-th cache-only forward in the proposal graph).
~2 syncs/pass (proposal realize + verify realize) vs naive K+1.
"""
import os, time, argparse
from tinygrad import Tensor, UOp, TinyJit, dtypes
from extra.llm_generate import load_model_and_tokenizer

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--target", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--draft", default="/home/ubuntu/models/Qwen3-0.6B-Q8_0.gguf")
  ap.add_argument("--prompts", default="bench/qk-spec-decode-acceptance/prompts.jsonl")
  ap.add_argument("--num-prompts", type=int, default=4)
  ap.add_argument("--k", type=int, default=4)
  ap.add_argument("--n-new", type=int, default=48)
  ap.add_argument("--max-ctx", type=int, default=1024)
  args = ap.parse_args()
  K, MAXC = args.k, args.max_ctx
  target, tok = load_model_and_tokenizer(args.target, MAXC, seed=7)
  draft, _ = load_model_and_tokenizer(args.draft, MAXC, seed=7)
  for b in target.blk: b._use_flash, b._prefill_v2 = False, False
  v_sp = UOp.variable("start_pos", 0, MAXC - 1)
  verify_jit = TinyJit(lambda toks, spv: target.logits(toks, spv).argmax(-1).cast(dtypes.int32).realize())
  # proposal graph: K proposals + 1 cache-only forward (close the full-accept draft hole), distinct rebindable vars
  sps = [UOp.variable(f"sp{k}", 0, MAXC - 1) for k in range(K + 1)]
  @TinyJit
  def propose(tok0, *bs):
    t = tok0; outs = []
    for k in range(K):
      t = draft.logits(t, bs[k])[:, -1:, :].argmax(-1).cast(dtypes.int32); outs.append(t.reshape(1, 1))
    draft.logits(t, bs[K])  # cache-only: write KV for proposed[K-1] at base+K (closes full-accept hole)
    return outs[0].cat(*outs[1:], dim=1).realize()
  def prefill(m, ids): m.logits(Tensor([ids], dtype="int32").contiguous(), 0).realize()

  def spec_decode(prompt_ids, n_new):
    prefill(target, prompt_ids); prefill(draft, prompt_ids)
    L = len(prompt_ids); last = int(prompt_ids[-1]); out = []; passes = 0; syncs = 0
    t0 = time.perf_counter()
    while len(out) < n_new:
      props = propose(Tensor([[last]], dtype="int32").contiguous(), *[sps[k].bind(L - 1 + k) for k in range(K + 1)]); syncs += 1
      vin = Tensor([[last]], dtype="int32").cat(props, dim=1).contiguous()      # [1,K+1] on device
      tg = verify_jit(vin, v_sp.bind(L - 1)); syncs += 1                          # [1,K+1]
      proposed = props.tolist()[0]                                                 # ONE sync read (all K)
      tgt = tg.tolist()[0]                                                          # ONE sync read (all K+1)
      n = 0
      for i in range(K):
        if proposed[i] == tgt[i]: n += 1
        else: break
      new = proposed[:n] + [tgt[n]]
      out += new; L += len(new); last = tgt[n]; passes += 1
      if last == getattr(tok, "eos_id", -1): break
    return out[:n_new], passes, time.perf_counter() - t0, syncs

  rollout = TinyJit(lambda t, spv: target.logits(t, spv)[:, -1:, :].argmax(-1).cast(dtypes.int32).realize())
  def baseline_decode(prompt_ids, n_new):
    prefill(target, prompt_ids); L = len(prompt_ids); last = int(prompt_ids[-1]); out = []
    t0 = time.perf_counter()
    while len(out) < n_new:
      r = rollout(Tensor([[last]], dtype="int32").contiguous(), v_sp.bind(L - 1)); last = int(r[0, 0].item())
      out.append(last); L += 1
      if last == getattr(tok, "eos_id", -1): break
    return out[:n_new], time.perf_counter() - t0

  import json
  prompts = []
  with open(args.prompts) as f:
    for line in f:
      if line.strip(): prompts.append(json.loads(line))
      if len(prompts) >= args.num_prompts: break
  # warm
  wids = tok.encode(prompts[0].get("prompt", prompts[0].get("text", "Hello")))[:32]
  wids = list(wids)
  for _ in range(2): baseline_decode(wids, 40); spec_decode(wids, 40)  # ramp clock
  print(f"{'prompt':6} {'exact':6} {'base tok/s':11} {'spec tok/s':11} {'speedup':8} {'acc/pass':9} {'syncs/pass'}")
  tot_b = tot_s = 0.0
  for pi, p in enumerate(prompts):
    ids = list(tok.encode(p.get("prompt", p.get("text", "")))[:32])
    base, bdt = baseline_decode(ids, args.n_new)
    spec, passes, sdt, syncs = spec_decode(ids, args.n_new)
    exact = base == spec
    btks, stks = len(base) / bdt, len(spec) / sdt
    tot_b += btks; tot_s += stks
    print(f"{pi:6} {str(exact):6} {btks:11.1f} {stks:11.1f} {stks/btks:8.2f} {len(spec)/passes:9.2f} {syncs/passes:.1f}")
  print(f"AVG base {tot_b/len(prompts):.1f} tok/s | spec {tot_s/len(prompts):.1f} tok/s | speedup {tot_s/tot_b:.2f}x")

if __name__ == "__main__": main()
