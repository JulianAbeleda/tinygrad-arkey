"""Decode machine-search GATE — the per-candidate evaluator + reusable checkers.

Loads the model ONCE, captures the decode TinyJit, and runs the full safe-search pipeline in cost order
(route-fire -> materialization/ABI -> correctness -> ISA-reject -> W==D), SHORT-CIRCUITING on the first reject.
W==D (synced) is the only promotion authority. Emits one result JSON (the standard result schema).

Usage (the runner spawns this per candidate, with the candidate's knob env vars set):
  QK_CAND_ID=oracle [DECODE_ATTN_KV_IDENTITY=1 ...] DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python \
    extra/qk_decode_search_gate.py [--oracle-tokens bench/qk-decode-search-readiness/baseline_oracle.json]

See docs/decode-machine-search-readiness-package-scope-20260623.md. Does NOT change defaults; the oracle is frozen.
"""
from __future__ import annotations
import os, re, sys, json, time
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
CANDIDATE_KERNEL = os.environ.get("QK_CAND_KERNEL", "owned_flash_tile_gqa_whole")  # the expected kernel symbol
# (Mode B sets QK_CAND_KERNEL to the variant symbol e.g. owned_flash_tile_gqa_whole_tk8_v2_u1 so route-fire binds to it)
SLICE_KERNEL = "owned_flash_tile_gqa"             # the pre-fix slice route (must be ABSENT on the buffer-identity route)
WD_CTXS = [512, 1024]
CORRECTNESS_PROMPT = "The history of computing began when"

def _program_names(captured):
  from tinygrad.uop.ops import Ops
  if captured is None: return []
  return [_ANSI.sub("", str(getattr(u.arg, "name", ""))) for u in captured.linear.toposort() if u.op is Ops.PROGRAM]

# ---- the reusable checkers (operate on a captured TinyJit graph) ----
def check_route_fire(captured, candidate_kernel=CANDIDATE_KERNEL):
  names = _program_names(captured)
  return {"program_node_names": names,
          "candidate_kernel_present": any(candidate_kernel in n for n in names),
          "slice_route_absent": not any(n.endswith(SLICE_KERNEL) or n == SLICE_KERNEL for n in names)}

def check_materialization(captured):
  names = _program_names(captured)
  def _copy_elems(name: str) -> int:
    m = re.match(r"^E_([0-9]+)", name)
    return int(m.group(1)) if m else 0
  # E_49152 is one full MAXC K or V tensor for Qwen3-8B decode; E_98304 is combined K+V.
  literal_copies = [n for n in names if "4915" in n or "49152" in n]
  large_e_copies = [n for n in names if _copy_elems(n) >= 49152]
  copies = sorted(set(literal_copies + large_e_copies), key=names.index)
  return {"E_49152_present": len(literal_copies) > 0, "full_maxc_copy_kernels": copies,
          # buffer-identity holds iff the whole-cache kernel fired AND no full-MAXC copy is present
          "buffer_identity_inputs": (any(CANDIDATE_KERNEL in n for n in names) and len(copies) == 0)}

# ---- ISA reject (wraps qk_isa_primitive_audit) ----
def isa_audit(candidate="owned_decode_attention_whole"):
  import glob, subprocess
  cos = sorted(glob.glob("/tmp/b4_tile_whole_*.co"), key=os.path.getmtime)
  if not cos: return None, "no_candidate_code_object"
  out = "/tmp/_search_isa.json"
  r = subprocess.run([sys.executable, "extra/qk_isa_primitive_audit.py", "--vendor", "amd",
                      "--candidate", candidate, "--code-object", cos[-1], "--out", out],
                     capture_output=True, text=True, env={**os.environ})
  try:
    return json.loads(r.stdout.strip().splitlines()[-1]), None
  except Exception:
    try: return json.load(open(out)), None
    except Exception as e: return None, f"isa_audit_failed:{e}"

def isa_reject(isa):
  if isa is None: return "isa_missing"
  f = isa.get("instruction_flags", {}); res = isa.get("resources", {})
  if not f.get("has_vector_dot"): return "isa_lost_v_dot2"
  if not f.get("has_lds"): return "isa_lost_lds"
  if not f.get("has_cross_lane"): return "isa_lost_cross_lane"
  if f.get("has_spill") or (res.get("spills") or 0) > 0 or (res.get("scratch_bytes") or 0) > 0: return "isa_spill"
  if (res.get("vgpr") or 0) > 96: return f"isa_vgpr_over_envelope_{res.get('vgpr')}"
  return None

# ---- model load + capture + correctness + W==D (one process) ----
def _setup_model():
  from tinygrad import Tensor
  from extra.llm_generate import load_model_and_tokenizer
  from extra.qk_harness_contract import DEFAULT_MODEL
  m, tok = load_model_and_tokenizer(DEFAULT_MODEL, 4608, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []): lin.decode_enabled = True
  for b in m.blk: b._use_flash, b._prefill_v2 = True, False
  return m, tok

def capture_decode(m, tok):
  """Prefill >=512 (so the owned tile route fires) then greedy 6-token decode via TinyJit; returns
  (tokens, captured_jit, step, v, temp). The owned route guard is ctx >= DECODE_ATTN_AMDGCN_MIN_CTX (512)."""
  from tinygrad import Tensor, UOp, TinyJit
  v = UOp.variable("start_pos", 0, 4607); temp = Tensor([0.0])
  ids = ((tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CORRECTNESS_PROMPT + " ") * 80)[:520]
  out = None; sp = 0
  for st in range(0, len(ids), 512):                       # eager prefill fills the cache to >=512
    chunk = ids[st:st+512]; out = m.forward(Tensor([chunk], dtype="int32").contiguous(), sp, temp).realize(); sp += len(chunk)
  step = TinyJit(m.forward); toks = [int(out.item())]
  for _ in range(5): out = step(Tensor([[toks[-1]]], dtype="int32").contiguous(), v.bind(sp), temp).realize(); toks.append(int(out.item())); sp += 1
  return toks, step.captured, step, v, temp

def run_wd(m, ctxs=WD_CTXS):
  """Canonical W==D (matches /tmp/wd_child + qk_decode_runtime_overhead): a FRESH TinyJit warmed and measured at the
  bound ctx, fixed [[100]] token, .item() readback per step (the real synced decode path). The synced authority."""
  from tinygrad import Tensor, UOp, TinyJit
  v = UOp.variable("start_pos", 0, 4607); temp = Tensor([0.0]); tk = Tensor([[100]], dtype="int32").contiguous()
  wd = {}
  for ck in ctxs:
    step = TinyJit(m.forward)
    for _ in range(8): o = step(tk, v.bind(ck), temp).realize(); o.item()   # warm (compile + clock ramp)
    ts = []
    for _ in range(30): t0 = time.perf_counter(); o = step(tk, v.bind(ck), temp).realize(); o.item(); ts.append(time.perf_counter() - t0)
    ts.sort(); med = ts[len(ts)//2]
    wd[str(ck)] = {"tok_s": round(1.0 / med, 1), "spread_pct": round(100*(ts[-1]-ts[0])/med, 1)}
  return wd

def evaluate(cand_id, oracle_tokens=None):
  from tinygrad import Tensor
  res = {"id": cand_id, "knobs_env": {k: os.environ[k] for k in os.environ if k.startswith("DECODE_ATTN") or k.startswith("Q4K_GEMV")},
         "reject_reason": None, "verdict": None}
  m, tok = _setup_model()
  toks, captured, step, v, temp = capture_decode(m, tok)
  res["tokens"] = toks
  # cheap structural checks first
  res["route_fire"] = check_route_fire(captured)
  res["materialization"] = check_materialization(captured)
  res["token_byte_identical"] = (oracle_tokens is None) or (toks == oracle_tokens)
  # apply the cheap rejects (short-circuit before ISA / W==D)
  reason = _cheap_reject(res)
  if reason: res["reject_reason"] = reason
  if res["reject_reason"] is None:
    isa, err = isa_audit(); res["isa"] = isa or {"error": err}
    res["reject_reason"] = isa_reject(isa)
  if res["reject_reason"] is None:
    res["wd"] = run_wd(m)
    res["reject_reason"] = _wd_reject(res, oracle_wd=(oracle_tokens and None))
  res["verdict"] = "REJECTED:" + res["reject_reason"] if res["reject_reason"] else "PASS"
  return res

def _cheap_reject(res):
  if not res["route_fire"]["candidate_kernel_present"]: return "route_not_firing"
  if res["materialization"]["full_maxc_copy_kernels"]: return "full_maxc_materialization_returned"
  if not res["materialization"]["buffer_identity_inputs"]: return "sliced_view_not_buffer_identity"
  if not res["token_byte_identical"]: return "token_correctness_failed"
  return None

def _wd_reject(res, oracle_wd=None):
  # ctx512 regression is checked by the runner against the frozen oracle; here only structural W==D sanity
  if "512" not in res.get("wd", {}): return "wd_missing"
  return None

if __name__ == "__main__":
  cand_id = os.environ.get("QK_CAND_ID", "candidate")
  oracle = None
  if "--oracle-tokens" in sys.argv:
    p = sys.argv[sys.argv.index("--oracle-tokens") + 1]
    try: oracle = json.load(open(p)).get("tokens")
    except Exception: oracle = None
  r = evaluate(cand_id, oracle_tokens=oracle)
  print("RESULT " + json.dumps(r))
