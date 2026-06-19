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
  ap.add_argument("--measure-verify", action="store_true",
                  help="isolate verify cost: single T==1 pass vs T=K+1 fallback vs T=K+1 batched-GEMM (decode_enabled). Diagnostic; exits.")
  ap.add_argument("--measure-ctx", type=int, default=512)
  args = ap.parse_args()
  K, MAXC = args.k, args.max_ctx
  target, tok = load_model_and_tokenizer(args.target, MAXC, seed=7)
  draft, _ = load_model_and_tokenizer(args.draft, MAXC, seed=7)
  for b in target.blk: b._use_flash, b._prefill_v2 = False, False
  v_sp = UOp.variable("start_pos", 0, MAXC - 1)
  verify_jit = TinyJit(lambda toks, spv: target.logits(toks, spv).argmax(-1).cast(dtypes.int32).realize())

  if args.measure_verify:
    # DIAGNOSTIC (Track 3): is the T=K+1 verify slow because of a missing primitive, or because the harness
    # never set decode_enabled (so K>1 routes to the dense _fallback)? Time three ways, isolated, W==D-style.
    import statistics
    from tinygrad import Device
    dev = Device[Device.DEFAULT]
    CTX = args.measure_ctx
    base_ids = list(tok.encode("In the beginning was the word. " * 64))
    ids = (base_ids * (1 + (CTX + K + 2) // max(1, len(base_ids))))[:CTX]
    target.logits(Tensor([ids], dtype="int32").contiguous(), 0).realize()           # prefill to CTX
    last = int(ids[-1]); spb = v_sp.bind(CTX - 1)
    vin = Tensor([[last] + ids[:K]], dtype="int32").contiguous()                     # [1,K+1] concrete verify input
    one = Tensor([[last]], dtype="int32").contiguous()                              # [1,1] single-pass
    rj = TinyJit(lambda t, s: target.logits(t, s)[:, -1:, :].argmax(-1).cast(dtypes.int32).realize())
    def wd(fn, *a, M=40):
      for _ in range(8): fn(*a)                                                      # warm + capture + clock ramp
      dev.synchronize(); t0 = time.perf_counter()
      for _ in range(M): fn(*a)
      dev.synchronize(); return (time.perf_counter() - t0) / M * 1e3
    # decode_enabled=True == production decode state: single pass hits the shipped coop kernels (K==1);
    # the K+1 verify hits the batched GEMM (K>1). Capture+measure BOTH here before flipping the flag.
    for lin in target._q4k_linears.linears: lin.decode_enabled = True
    vj_bt = TinyJit(lambda t, s: target.logits(t, s).argmax(-1).cast(dtypes.int32).realize())
    one_ms = wd(rj, one, spb); bt_ms = wd(vj_bt, vin, spb); bt_tok = vj_bt(vin, spb).tolist()
    # decode_enabled=False == the (buggy) harness state: K+1 verify routes to the dense _fallback.
    for lin in target._q4k_linears.linears: lin.decode_enabled = False
    vj_fb = TinyJit(lambda t, s: target.logits(t, s).argmax(-1).cast(dtypes.int32).realize())
    fb_ms = wd(vj_fb, vin, spb); fb_tok = vj_fb(vin, spb).tolist()
    print(f"\n=== verify-cost isolation (ctx={CTX}, K={K}, T=K+1={K+1}) ===")
    print(f"single T==1 pass     : {one_ms:6.2f} ms ({1000/one_ms:5.1f} tok/s)")
    print(f"T=K+1 verify FALLBACK: {fb_ms:6.2f} ms  = {fb_ms/one_ms:.2f}x one pass (decode_enabled=False, dense)")
    print(f"T=K+1 verify BATCHED : {bt_ms:6.2f} ms  = {bt_ms/one_ms:.2f}x one pass (decode_enabled=True, GEMM)")
    print(f"batched vs fallback  : {fb_ms/bt_ms:.2f}x  | argmax tokens identical: {fb_tok == bt_tok}")
    print(f"INTERPRETATION: spec ceiling needs verify <= ~1.x one-pass; {K+1} serial T==1 calls would be ~{K+1}x.")
    return
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
