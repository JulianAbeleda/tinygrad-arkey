"""AMD ISA backend — Phase N0: owned-vs-native throughput diff audit (evidence-first; NO optimization).

Answers ONE question: why is the native-ISA decode tile ~47% of the owned hand-AMDGCN tile? Phases I/J/K/L/M
already refuted topology, correctness, waitcnt, scheduler, and occupancy as the dominant blocker (Phase M:
raising occupancy 4->5 wg/CU moved W==D ~0% -> throughput-bound, not resource-bound). N0 captures both tiles'
disassembly + resource descriptor + instruction histogram and emits a static diff matrix to PIN the throughput gap.

Compares the per-(kvh,s) attention TILE of each route (gmax/combine are separate kernels for both, not compared):
  owned_flash_tile_gqa_whole   (DECODE_ATTN_AMDGCN_TILE=1)
  native_block_tile            (native ISA Ops.PROGRAM injection: GENERATED_WHOLECACHE+FUSED_XLANE+BLOCK_TILE+FIXED_S+NATIVE_ISA_BLOCK_TILE)
Each captured in a FRESH SUBPROCESS via a dev.runtime hook (intercept the compiled lib by kernel-name prefix),
disassembled with llvm-objdump. W==D tok/s + token_match + determinism are referenced from the Phase I/grid gates.

Run:  DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_n0_throughput_diff.py
Writes: bench/amd-isa-backend-phase-n0/{latest.json,disasm_owned.txt,disasm_native.txt}
"""
from __future__ import annotations
import os, sys, json, re, pathlib, subprocess
from typing import Any
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-phase-n0"
CTX = int(os.environ.get("QK_CTX", "512")); MAXC = 4608

# route arms: tile-name prefix + env flags. owned = shipped oracle; native = AMDISARenderer Ops.PROGRAM injection.
TILES = {
  "owned":  {"tile": "owned_flash_tile_gqa_whole",
             "env": {"DECODE_ATTN_AMDGCN_TILE": "1"}},
  "native": {"tile": "native_block_tile",
             "env": {"DECODE_ATTN_AMDGCN_TILE": "0", "DECODE_ATTN_GENERATED_WHOLECACHE": "1",
                     "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE": "1", "DECODE_ATTN_BLOCK_TILE": "1",
                     "DECODE_ATTN_BLOCK_TILE_FIXED_S": "1", "DECODE_ATTN_NATIVE_ISA_BLOCK_TILE": "1"}},
}
_ZERO = ["DECODE_ATTN_AMDGCN_TILE", "DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE",
         "DECODE_ATTN_BLOCK_TILE", "DECODE_ATTN_BLOCK_TILE_FIXED_S", "DECODE_ATTN_NATIVE_ISA_BLOCK_TILE"]

def _markers(asm: str) -> dict[str, int]:
  def c(p): return len(re.findall(p, asm))
  return {"v_dot2": c(r"\bv_dot2"), "v_exp": c(r"\bv_exp"),
          "cross_lane": c(r"\b(ds_bpermute|ds_permute|ds_swizzle|v_permlane)"),
          "s_waitcnt": c(r"\bs_waitcnt\b"), "s_barrier": c(r"\bs_barrier\b"),
          "global_load_dwordx4": c(r"\bglobal_load_dwordx4\b"), "global_load_dwordx2": c(r"\bglobal_load_dwordx2\b"),
          "global_load_dword": c(r"\bglobal_load_dword\b"), "global_load_b32": c(r"\bglobal_load_b32\b"),
          "global_load_d16": c(r"\bglobal_load_(u?short|d16|u16)"),
          "ds_read": c(r"\bds_(read|load)"), "ds_write": c(r"\bds_(write|store)"),
          "salu": c(r"\bs_[a-z0-9_]+"), "branch": c(r"\bs_(branch|cbranch)")}

def _capture(arm: str) -> dict[str, Any]:
  import contextlib, io
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters, Device
  from extra.llm_generate import load_model_and_tokenizer
  from extra.qk_harness_contract import DEFAULT_MODEL
  from extra.qk_decode_attention_fused_score_state_pv_attribution import _disasm, _hist, _parse_desc, _parse_debug, ANSI
  tile = TILES[arm]["tile"]
  dev = Device[Device.DEFAULT]; captured: dict[str, bytes] = {}; orig = dev.runtime
  def hook(name, lib, **kw):
    if name.startswith(tile) and name not in captured: captured[name] = lib
    return orig(name, lib, **kw)
  dev.runtime = hook
  # the native tile is a minimal assemble_linear ELF that llvm-objdump won't disassemble -> capture the ACTUAL
  # emitted rdna3 instruction stream from AMDISARenderer.asm (post-schedule/waitcnt/label-resolve) instead.
  asmtext: dict[str, str] = {}
  try:
    from tinygrad.renderer.isa.amd import AMDISARenderer
    from tinygrad.uop.ops import Ops as _Ops
    from tinygrad.helpers import getenv as _ge
    _oasm = AMDISARenderer.asm
    def _asmspy(self, prg, lin):
      try:
        ins = list(lin.src)
        if _ge("AMD_ISA_SCHED", 1): ins = self._schedule(ins)
        fin = self._resolve_labels(self._insert_waitcnt(ins))
        asmtext["text"] = "\n".join(str(u.arg) for u in fin if u.op is _Ops.INS)
      except Exception: pass
      return _oasm(self, prg, lin)
    AMDISARenderer.asm = _asmspy
  except Exception: pass
  m, _tok = load_model_and_tokenizer(os.environ.get("QK_MODEL", DEFAULT_MODEL), MAXC, seed=20260617)
  q4k = getattr(m, "_q4k_linears", None)
  for lin in (q4k.linears if q4k else []): lin.decode_enabled = True
  for b in m.blk: b._use_flash, b._prefill_v2 = True, False
  v = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0]); step = TinyJit(m.forward)
  tk = Tensor([[100]], dtype="int32").contiguous()
  toks = []
  for i in range(8): out = step(tk, v.bind(CTX + i), temp).realize(); toks.append(int(out.item()))
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset(); step(tk, v.bind(CTX + 8), temp).realize()
  lines = [ANSI.sub("", l) for l in buf.getvalue().splitlines() if "***" in l]
  prog_ms = _parse_debug(lines).get("program_ms", {})
  tile_ms = round(sum(vv for kk, vv in prog_ms.items() if kk.startswith(tile)), 4)
  res: dict[str, Any] = {"arm": arm, "tile_prefix": tile, "captured": list(captured.keys()), "tile_program_ms": tile_ms,
                         "warm_tokens": toks[2:]}
  for name, lib in captured.items():
    asm = asmtext.get("text") or _disasm(lib)   # native: emitted rdna3 stream; owned: llvm-objdump
    asm_src = "amdisa_inst_stream" if asmtext.get("text") and arm == "native" else "llvm-objdump"
    try: desc = _parse_desc(lib)
    except Exception as e: desc = {"error": str(e)}
    res["tile_data"] = {"name": name, "resources": desc, "hist": _hist(asm), "markers": _markers(asm),
                        "asm_lines": len(asm.splitlines()), "asm_source": asm_src}
    (OUT / f"disasm_{arm}.txt").write_text(asm)
    break
  if "tile_data" not in res: res["error"] = f"tile {tile!r} not captured; saw {list(captured.keys())}"
  return res

def _run_child(arm: str) -> dict[str, Any]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_N0_CHILD": "1", "QK_N0_ARM": arm, "DEV": "AMD"}
  for k in _ZERO: env[k] = "0"
  env.update(TILES[arm]["env"])
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
  if p.returncode != 0: return {"arm": arm, "failed": True, "rc": p.returncode, "tail": (p.stdout or "")[-6000:]}
  for line in reversed((p.stdout or "").splitlines()):
    try: return json.loads(line)
    except Exception: pass
  return {"arm": arm, "failed": True, "error": "no json", "tail": (p.stdout or "")[-6000:]}

def _diff_rows(o: dict, n: dict) -> list[dict]:
  oh, om, od = o["hist"], o["markers"], o["resources"]
  nh, nm, nd = n["hist"], n["markers"], n["resources"]
  CAUSE = {
    "total_instructions": ("more static work per tile invocation", "reduce emitted ops / fuse"),
    "VALU": ("more vector ALU per token (e.g. index math, unfused ops)", "fuse/strength-reduce VALU; FMA"),
    "SALU": ("more scalar/control overhead (loop counters, addr math)", "hoist scalar math; fewer loops"),
    "VMEM_loads": ("more/narrower global loads (poor coalescing)", "wider global_load (dwordx4) / fewer reloads"),
    "VMEM_stores": ("more global stores", "fewer partial stores"),
    "DS_ops": ("more LDS traffic (staging + accumulators in LDS vs owned registers)", "register accumulators / fewer LDS round-trips"),
    "v_dot2": ("packed-fp16 dot usage", "match owned dot strategy"),
    "v_exp": ("more transcendentals (exp/exp2) per token", "fewer exp (online-softmax fusion / FAST_EXP2)"),
    "cross_lane": ("cross-lane reduction ops", "amortize warp reduce / match owned"),
    "waitcnt": ("memory-wait stalls (latency exposure)", "consumer-only waitcnt (done) / finer counters"),
    "branches": ("loop/control branches (serial inner loops)", "unroll / fewer loops"),
    "barriers": ("workgroup barriers", "fewer barriers"),
    "VGPR": ("higher register pressure", "lower VGPR if occupancy-limiting (Phase M: not limiting)"),
    "LDS": ("more LDS bytes (accumulators in LDS)", "register accumulators (Phase M: occupancy not the lever)"),
    "scratch": ("scratch spills", "avoid spills"),
  }
  def row(name, ov, nv):
    d = nv - ov; r = round(nv / ov, 2) if ov else (None if nv == 0 else float("inf"))
    c, lev = CAUSE.get(name, ("", ""))
    return {"row": name, "owned": ov, "native": nv, "delta": d, "ratio": r, "suspected_cause": c, "candidate_lever": lev}
  return [
    row("total_instructions", oh["total"], nh["total"]), row("VALU", oh["valu"], nh["valu"]),
    row("SALU", oh["s_inst"], nh["s_inst"]), row("VMEM_loads", oh["vmem_load"], nh["vmem_load"]),
    row("VMEM_stores", oh["vmem_store"], nh["vmem_store"]), row("DS_ops", oh["ds"], nh["ds"]),
    row("v_dot2", om["v_dot2"], nm["v_dot2"]), row("v_exp", om["v_exp"], nm["v_exp"]),
    row("cross_lane", oh["cross_lane"], nh["cross_lane"]), row("waitcnt", om["s_waitcnt"], nm["s_waitcnt"]),
    row("branches", om["branch"], nm["branch"]), row("barriers", om["s_barrier"], nm["s_barrier"]),
    row("VGPR", od.get("vgpr", 0), nd.get("vgpr", 0)), row("LDS", od.get("lds", 0), nd.get("lds", 0)),
    row("scratch", od.get("scratch", 0), nd.get("scratch", 0)),
  ]

def build() -> dict[str, Any]:
  OUT.mkdir(parents=True, exist_ok=True)
  rec: dict[str, Any] = {"verdict": None, "command": "DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_n0_throughput_diff.py",
                         "question": "why is native ~47% of owned?"}
  owned = _run_child("owned"); native = _run_child("native")
  rec["owned_capture"] = {k: owned.get(k) for k in ("captured", "tile_program_ms", "failed", "error")}
  rec["native_capture"] = {k: native.get(k) for k in ("captured", "tile_program_ms", "failed", "error")}
  if owned.get("failed") or "tile_data" not in owned: rec["verdict"] = "AMD_ISA_PHASE_N0_BLOCKED_CAPTURE_OWNED"; rec["detail"] = owned; return rec
  if native.get("failed") or "tile_data" not in native: rec["verdict"] = "AMD_ISA_PHASE_N0_BLOCKED_CAPTURE_NATIVE"; rec["detail"] = native; return rec
  od, nd = owned["tile_data"], native["tile_data"]
  if od["asm_lines"] < 5 or nd["asm_lines"] < 5:
    rec["verdict"] = "AMD_ISA_PHASE_N0_BLOCKED_DISASM"; rec["asm_lines"] = {"owned": od["asm_lines"], "native": nd["asm_lines"]}; return rec
  rec["owned"] = {"resources": od["resources"], "hist": od["hist"], "markers": od["markers"], "asm_lines": od["asm_lines"], "tile_program_ms": owned["tile_program_ms"]}
  rec["native"] = {"resources": nd["resources"], "hist": nd["hist"], "markers": nd["markers"], "asm_lines": nd["asm_lines"], "tile_program_ms": native["tile_program_ms"]}
  # grid/block (known launch geometry; both 384 wg / 128 threads)
  rec["grid_block"] = {"owned": {"grid": [8, 48, 1], "block": [128, 1, 1]}, "native": {"grid": [8, 48, 1], "block": [32, 4, 1]}}
  rec["diff_rows"] = _diff_rows(od, nd)
  # W==D + correctness referenced from the Phase I / grid gates (same harness/session authority)
  def _read(p, *keys):
    f = ROOT / p
    return json.load(open(f)) if f.exists() else {}
  pi = _read("bench/amd-isa-backend-phase-i/latest.json")
  rec["wd"] = {ck: {"native_tok_s": v["native_tok_s"], "owned_tok_s": v["owned_tok_s"], "pct_of_owned": v["pct_of_owned"]}
               for ck, v in pi.get("per_ctx", {}).items()} if pi else "phase-i artifact missing"
  rec["token_match"] = pi.get("token_match"); rec["route_bound"] = pi.get("route_bound")
  grid = _read("bench/amd-isa-backend-grid/latest.json")
  rec["deterministic"] = grid.get("in_model_repeated_run_stability"); rec["hidden_fallback_check"] = grid.get("hidden_fallback_check")
  # normalization + top deltas
  rec["normalization"] = {"per_static_tile": "the hist/markers are per single tile-kernel invocation (one workgroup over its (kvh,s) split)",
                          "per_workgroup": "1 workgroup = 1 (kvh,s); grid 384 wg both routes -> per-wg counts ARE the diff",
                          "per_token": f"each tile invocation runs per (kvh,s) per layer per decode token; tile_program_ms is the per-decode-step total over all 384 wg"}
  ranked = sorted([r for r in rec["diff_rows"] if isinstance(r["ratio"], (int, float)) and r["delta"] > 0], key=lambda r: -(r["delta"]))
  rec["top_deltas"] = [{"row": r["row"], "owned": r["owned"], "native": r["native"], "delta": r["delta"], "ratio": r["ratio"]} for r in ranked[:5]]
  # strongest suspected bottleneck = the largest positive native-excess that is throughput-relevant (instr/ds/valu)
  hot = [r for r in ranked if r["row"] in ("DS_ops", "VALU", "total_instructions", "VMEM_loads", "SALU", "v_exp", "cross_lane")]
  rec["strongest_suspected_bottleneck"] = (f"{hot[0]['row']}: native {hot[0]['native']} vs owned {hot[0]['owned']} "
    f"(x{hot[0]['ratio']}) -- {hot[0]['suspected_cause']}") if hot else "no single dominant static delta; needs dynamic counters"
  # decompose the VALU excess into its two concrete causes (both visible in the diff rows):
  oh, om = od["hist"], od["markers"]; nh, nm = nd["hist"], nd["markers"]
  causes = []
  if nm.get("v_exp", 0) < om.get("v_exp", 0):
    causes.append(f"EXP via POLYNOMIAL: native v_exp={nm.get('v_exp',0)} vs owned {om.get('v_exp',0)} -- AMDISARenderer has no hardware "
                  "v_exp_f32 lowering, so exp2 expands to a VALU polynomial (tinygrad TRANSCENDENTAL) -> many extra v_fma/v_mul per token. "
                  "LEVER: add an exp2 -> v_exp_f32 (hardware transcendental) lowering to AMDISARenderer.")
  if nh.get("valu", 0) > oh.get("valu", 0) and nh.get("s_inst", 0) < oh.get("s_inst", 0):
    causes.append(f"PER-LANE VECTOR address math: native VALU={nh['valu']} / SALU={nh['s_inst']} vs owned VALU={oh['valu']} / SALU={oh['s_inst']} "
                  "-- owned hoists uniform/loop-invariant address math to SCALAR (SALU); native recomputes per-lane in VECTOR (v_mul_lo/v_add_nc). "
                  "LEVER: scalarize uniform index/address math (emit s_* for wave-uniform terms) or strength-reduce + FMA-fuse the VALU index chain.")
  rec["valu_excess_causes"] = causes
  rec["next_implementation_lever"] = (causes[0].split('LEVER: ')[1] if causes else (hot[0]["candidate_lever"] if hot else "dynamic perf counters (SQTT)"))
  ok = (rec.get("token_match") is True and rec.get("route_bound") is True and rec.get("deterministic") is True
        and od["asm_lines"] > 5 and nd["asm_lines"] > 5 and rec.get("hidden_fallback_check", "").startswith("candidate"))
  rec["verdict"] = "AMD_ISA_PHASE_N0_PASS_THROUGHPUT_DIFF_PINNED" if ok else "AMD_ISA_PHASE_N0_INCONCLUSIVE_NEEDS_DYNAMIC_COUNTERS"
  return rec

if __name__ == "__main__":
  if os.environ.get("QK_N0_CHILD"):
    print(json.dumps(_capture(os.environ["QK_N0_ARM"]))); sys.exit(0)
  rec = build()
  with open(OUT / "latest.json", "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2)); print("\nPHASE_N0", rec["verdict"])
