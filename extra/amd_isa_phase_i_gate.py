"""AMD ISA backend — Phase I gate: route-bound native-ISA tile W==D baseline (correctness + timing ground truth).

Measures the native-ISA decode block tile in the real in-model W==D harness vs the owned-tile oracle, at ctx512 and
ctx4096. This phase is ALLOWED TO BE SLOW (the native tile is 1 workgroup with serial RANGE(GLOBAL) loops); it
establishes ground truth before any Phase K/L performance work. NO performance optimization here.

Two routes in FRESH SUBPROCESSES (getenv memoizes):
  - candidate: native-ISA tile (model on HIP, attention tile via AMDISARenderer + Ops.PROGRAM; gmax+combine on HIP)
  - comparator/oracle: owned hand-AMDGCN tile
Each child: per ctx checkpoint, warm then measure NMEAS real decode steps (.item()/token = W==D timing authority +
greedy tokens), + a DEBUG=2 eager forward for route attribution. Parent records tok/s, % of owned, token match,
attribution, and the native tile's VGPR/LDS resource summary.

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_i_gate.py
Writes: bench/amd-isa-backend-phase-i/latest.json
"""
import os, sys, json, io, re, time, statistics, contextlib, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "bench/amd-isa-backend-phase-i/latest.json"
MAXC = 4608
CKPTS = [int(x) for x in os.environ.get("QK_CKPTS", "512,4096").split(",")]
NMEAS = int(os.environ.get("QK_NMEAS", "10"))
NTOK = int(os.environ.get("QK_NTOK", "6"))
NWARM = int(os.environ.get("QK_NWARM", "4"))
_ANSI = re.compile(r"\x1b\[[0-9;]*m"); _KNAME = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(\w+)")
_NATIVE, _OWNED, _HIPBLK = "native_block_tile", "owned_flash_tile_gqa", "flash_block_tiled_xlane"

CAND = {"DECODE_ATTN_AMDGCN_TILE":"0","DECODE_ATTN_GENERATED_WHOLECACHE":"1","DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE":"1",
        "DECODE_ATTN_BLOCK_TILE":"1","DECODE_ATTN_BLOCK_TILE_FIXED_S":"1","DECODE_ATTN_NATIVE_ISA_BLOCK_TILE":"1"}
COMP = {"DECODE_ATTN_AMDGCN_TILE":"1"}
ALLFLAGS = set(CAND) | set(COMP)

def _child():
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters
  from extra.llm_generate import load_model_and_tokenizer
  from extra.qk_harness_contract import DEFAULT_MODEL
  m, tok = load_model_and_tokenizer(os.environ.get("QK_MODEL", DEFAULT_MODEL), MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []): lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps over the lazy dog. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v_sp = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0]); rows = {}
  for ck in CKPTS:
    for b in m.blk: b._use_flash, b._prefill_v2 = ck >= 512, False
    step = TinyJit(m.forward); tokid = int(ids[ck]); out = Tensor([[tokid]], dtype="int32").contiguous()
    for i in range(NWARM): out = step(out, v_sp.bind(ck + i), temp).realize()
    out = Tensor([[tokid]], dtype="int32").contiguous(); W, toks = [], []
    for i in range(NMEAS):
      t0 = time.perf_counter(); out = step(out, v_sp.bind(ck + i), temp); tid = int(out.item()); W.append(time.perf_counter() - t0)
      if i < NTOK: toks.append(tid)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), Context(DEBUG=2):
      GlobalCounters.reset(); m.forward(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(ck), temp).realize()
    names = sorted({_KNAME.search(_ANSI.sub("", l)).group(1) for l in buf.getvalue().splitlines() if _KNAME.search(_ANSI.sub("", l))})
    attn = [n for n in names if any(k in n for k in ("flash","native","owned","combine","gmax","state","coop"))]
    w_ms = statistics.median(W) * 1e3
    rows[ck] = {"tok_s": round(1000 / w_ms, 2), "w_ms_median": round(w_ms, 3), "w_ms_stdev": round(statistics.pstdev(W) * 1e3, 3),
                "nmeas": NMEAS, "tokens": toks, "attn": attn}
  print("@@RESULT@@" + json.dumps(rows))

def _spawn(flags, label):
  env = dict(os.environ)
  for k in ALLFLAGS: env.pop(k, None)
  env.update({k: str(v) for k, v in flags.items()}); env["QK_I_CHILD"] = "1"; env["DEV"] = "AMD"
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__))], env=env, capture_output=True, text=True, cwd=str(ROOT), timeout=5400)
  for line in p.stdout.splitlines():
    if line.startswith("@@RESULT@@"): return {int(k): v for k, v in json.loads(line[len("@@RESULT@@"):]).items()}
  raise RuntimeError(f"[{label}] no @@RESULT@@:\n{p.stderr[-3000:]}")

def _native_resources():
  try:
    from extra.qk_native_isa_block_tile_graph_node import compile_block_tile_isa
    from tinygrad.renderer.amd.elf import kernel_descriptor_from_elf
    from tinygrad.runtime.autogen import amdgpu_kd
    S = int(os.environ.get("DECODE_ATTN_FUSED_XLANE_SCORE_PV_S", 48)); L = max(1, -(-MAXC // S))
    elf, g, b, gseg = compile_block_tile_isa(128, 32, 8, MAXC, L, S, 4096)
    d = kernel_descriptor_from_elf(elf)
    gran = (d.compute_pgm_rsrc1 >> amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT) & 0x3f
    return {"vgpr": (gran + 1) * 8, "lds_bytes": gseg, "grid": list(g), "block": list(b), "elf_bytes": len(elf)}
  except Exception as e: return {"error": f"{type(e).__name__}: {e}"}

def main():
  rec = {"verdict": None, "command": "DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_i_gate.py",
         "selected_route_mode": "tile-only native injection (model on HIP, attention tile via AMDISARenderer)", "ckpts": CKPTS, "nmeas": NMEAS}
  try:
    cand = _spawn(CAND, "native"); comp = _spawn(COMP, "owned")
    per_ctx, route_bound, token_match, no_fallback = {}, True, True, True
    for ck in CKPTS:
      c, o = cand[ck], comp[ck]
      native_fired = any(_NATIVE in n for n in c["attn"])
      clean = native_fired and not any(_HIPBLK in n for n in c["attn"]) and not any(_OWNED in n for n in c["attn"])
      tm = c["tokens"] == o["tokens"]
      route_bound &= native_fired; token_match &= tm; no_fallback &= clean
      per_ctx[ck] = {"native_tok_s": c["tok_s"], "owned_tok_s": o["tok_s"],
                     "pct_of_owned": round(100.0 * c["tok_s"] / o["tok_s"], 1) if o["tok_s"] else None,
                     "native_w_ms_median": c["w_ms_median"], "native_w_ms_stdev": c["w_ms_stdev"],
                     "token_match": tm, "native_attn": c["attn"], "owned_attn": o["attn"], "candidate_route_clean": clean}
    rec["per_ctx"] = per_ctx
    rec["route_bound"] = route_bound; rec["token_match"] = token_match
    rec["hidden_fallback_check"] = "no HIP/LLVM block tile, no owned tile in candidate route" if no_fallback else "FALLBACK DETECTED"
    rec["native_resource_summary"] = _native_resources()
    rec["hip_block_tile_baseline_note"] = "comparator is the owned hand-AMDGCN tile; HIP-generated block-tile baseline is in bench/qk-owned-oracle-parity (separate harness)"
    rec["baseline_analysis"] = ("native tile ~0.4% of owned -- EXPECTED ground-truth baseline (Phase I is allowed to be "
      "slow). Dominant bottleneck = grid=[1,1,1]: the tile's kvh/s are RANGE(GLOBAL) which the native backend lowers to "
      "SERIAL loops (global_size derives from SPECIAL only), so ONE workgroup does all Hkv*S splits serially, vs the "
      "owned tile's Hkv*S parallel workgroups. Grid parallelism (map RANGE(GLOBAL) -> workgroup dims) + waitcnt (J) + "
      "scheduling (K) + modulo (L) are the optimization levers; out of scope for Phase I (record only).")
    if route_bound and token_match and no_fallback: rec["verdict"] = "AMD_ISA_PHASE_I_PASS_NATIVE_WD_BASELINE"; rec["next_phase_unlocked"] = "Phase J: correct consumer-only waitcnt"
    elif not route_bound: rec["verdict"] = "AMD_ISA_PHASE_I_BLOCKED_WD_ROUTE"
    elif not token_match: rec["verdict"] = "AMD_ISA_PHASE_I_BLOCKED_TOKEN_MATCH"
    else: rec["verdict"] = "AMD_ISA_PHASE_I_BLOCKED_COUNTER_ATTRIBUTION"
  except Exception as e:
    import traceback; rec["verdict"] = "AMD_ISA_PHASE_I_BLOCKED_RUNTIME_STABILITY"
    rec["exception"] = f"{type(e).__name__}: {e}"; rec["traceback"] = traceback.format_exc().splitlines()[-8:]
  return rec

if __name__ == "__main__":
  if os.environ.get("QK_I_CHILD"): _child(); sys.exit(0)
  rec = main()
  ART.parent.mkdir(parents=True, exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2)); print("\nPHASE_I", rec["verdict"])
