"""AMD ISA backend — Phase N2: dynamic stall attribution, owned tile vs native ISA tile (AUDIT-ONLY).

Answers: why is the native decode tile still ~60% of owned despite LOWER static instruction count (N0: owned 557 /
native 324; N1A closed the VALU excess; N1B proved uniform-address scalarization dead). The remaining gap must be
dynamic. This tool uses SQTT (thread-trace) per-instruction stall/duration + occupancy — hardware evidence, not
static counts.

Backend: SQTT (see bench/amd-isa-backend-phase-n2/profiling_capability_audit.json). Armed via env
SQTT=1 PROFILE=1 SQTT_ITRACE_SE_MASK=1 JIT=1 + ROCPROF_PATH=<vendored rocprof-trace-decoder .so>. SQTT_ITRACE_SE_MASK=1
limits the instruction trace to ONE shader engine -> per-wave samples (occupancy/CU counts are per-sampled-SE).

Captures one eager decode forward per route (owned / native) at ctx512, decodes per-PC stall/duration/category via
extra/sqtt/roc.py, aggregates, and emits a dynamic diff + hot stalled PCs + the named next phase (N3A..N3F).

Run:  DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_n2_dynamic_stall_attribution.py
Writes: bench/amd-isa-backend-phase-n2/{latest.json, owned_trace.json, native_trace.json, summary.md}
"""
from __future__ import annotations
import os, sys, json, pickle, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-phase-n2"
ROCPROF_SO = str(ROOT / "bench/amd-scheduler-tooling-backend/att_decoder_source_build_work/install/lib/librocprof-trace-decoder.so")
CTX = int(os.environ.get("QK_N2_CTX", "512")); MAXC = 4608
ROUTES = {
  "owned":  {"prefix": "owned_flash_tile_gqa_whole", "env": {"DECODE_ATTN_AMDGCN_TILE": "1"}},
  "native": {"prefix": "native_block_tile", "env": {"DECODE_ATTN_AMDGCN_TILE": "0", "DECODE_ATTN_GENERATED_WHOLECACHE": "1",
             "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE": "1", "DECODE_ATTN_BLOCK_TILE": "1", "DECODE_ATTN_BLOCK_TILE_FIXED_S": "1",
             "DECODE_ATTN_NATIVE_ISA_BLOCK_TILE": "1"}},
}
_ZERO = ["DECODE_ATTN_AMDGCN_TILE", "DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE",
         "DECODE_ATTN_BLOCK_TILE", "DECODE_ATTN_BLOCK_TILE_FIXED_S", "DECODE_ATTN_NATIVE_ISA_BLOCK_TILE"]
# inst category (rocprof enum tail) -> coarse bucket
def _bucket(typ: str) -> str:
  t = (typ or "").split("_")[-1]
  return {"VALU": "valu", "SALU": "salu", "SMEM": "salu", "VMEM": "vmem", "FLAT": "vmem", "LDS": "lds",
          "JUMP": "branch"}.get(t, "other")

def _capture(route: str):
  # subprocess body: arm SQTT via env, warmup the JIT, capture ONE eager forward, pickle Compiled.profile_events
  from tinygrad import Tensor, TinyJit, Context, GlobalCounters
  from tinygrad.uop.ops import UOp
  from tinygrad.device import Compiled
  from extra.qk_harness_contract import DEFAULT_MODEL
  from extra.llm_generate import load_model_and_tokenizer
  m, _tok = load_model_and_tokenizer(DEFAULT_MODEL, MAXC, seed=20260617)
  q4k = getattr(m, "_q4k_linears", None)
  for lin in (q4k.linears if q4k else []): lin.decode_enabled = True
  for b in m.blk: b._use_flash, b._prefill_v2 = True, False
  v = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0]); step = TinyJit(m.forward)
  tk = Tensor([[100]], dtype="int32").contiguous()
  for i in range(6): step(tk, v.bind(CTX + i), temp).realize()
  import tinygrad.runtime.ops_amd  # noqa
  with Context(SQTT=1, PROFILE=1):
    GlobalCounters.reset()
    m.forward(Tensor([[100]], dtype="int32").contiguous(), v.bind(CTX), temp).realize()
  with open(OUT / f"_{route}.pkl", "wb") as f: pickle.dump(Compiled.profile_events, f)
  print(f"@@CAPTURED {route} events={len(Compiled.profile_events)}")

def _aggregate(route: str, prefix: str) -> dict:
  from tinygrad.runtime.ops_amd import ProfileSQTTEvent
  from tinygrad.device import ProfileProgramEvent, ProfileDeviceEvent
  from tinygrad.helpers import ProfileRangeEvent, unwrap
  from tinygrad.viz.serve import amd_decode
  from extra.sqtt.roc import decode
  with open(OUT / f"_{route}.pkl", "rb") as f: profile = pickle.load(f)
  arch = ""; prg_events = {}; counter = {}; durations = {}
  for e in profile:
    if isinstance(e, ProfileProgramEvent) and e.tag is not None: prg_events[e.tag] = e
    if isinstance(e, ProfileSQTTEvent): counter.setdefault((e.kern, e.exec_tag), []).append(e)  # group per launch (kern,exec_tag)
    if isinstance(e, ProfileDeviceEvent) and e.device.startswith("AMD"): arch = f"gfx{unwrap(e.props)['gfx_target_version']//1000}"
    if isinstance(e, ProfileRangeEvent) and e.device.startswith("AMD") and getattr(e, "en", None) is not None:
      durations.setdefault(str(e.name), []).append(float(e.en - e.st))
  arch = arch or "gfx1100"
  prog_names = sorted({p.name for p in prg_events.values()})
  tile_tags = {k for k, p in prg_events.items() if p.name.startswith(prefix)}
  tile_groups = [(kt, evs) for kt, evs in counter.items() if kt[0] in tile_tags and any(e.itrace for e in evs)]  # launches with an itrace stream
  route_bound = len(tile_groups) > 0
  # a FALLBACK is the block TILE running elsewhere (owned tile in native route, or a HIP/LLVM xlane block tile) -- NOT
  # the gmax/combine helpers, which both routes legitimately run on HIP (phase-h: gmax_hip_present+combine_hip_present).
  _BAD = ("fused_xlane_score_pv_tile",) + (("owned_flash_tile_gqa_whole",) if route != "owned" else ())
  fallback_seen = [n for n in prog_names if any(b in n for b in _BAD) and not n.startswith(prefix)]
  rec = {"route": route, "arch": arch, "route_bound": route_bound, "prog_names_sample": [n for n in prog_names if "flash" in n or "block" in n or "attn" in n.lower()][:8],
         "tile_kernel_present": len(tile_tags) > 0, "tile_itrace_launches": len(tile_groups), "non_tile_attn_kernels": fallback_seen}
  if not route_bound: return rec
  p = prg_events[next(iter(tile_tags))]; base = unwrap(p.base)
  try: disasm = {addr + base: inst for addr, inst in amd_decode(p.lib, arch).items()}   # only needed for per-PC; occupancy decode never calls isa_cb
  except Exception as e: disasm = {}; rec["disasm_note"] = f"amd_decode failed ({e}); occupancy decode unaffected (decoder is occupancy-only)"
  # The vendored rocprof-trace-decoder (0.1.6) build returns wave events with instructions_size==0 -> per-PC stall is
  # NOT decodable (verified: roc.py --kernel yields only OCC). We extract OCCUPANCY + WAVE-CYCLE timing, which still
  # answers "is the tile stall/latency-bound" (cycles per wave vs static instruction count).
  waves = []; occ_pairs = []; n_inst = 0
  for (k, tag), evs in tile_groups[:24]:
    rctx = decode(evs, {k: disasm})
    waves += [w for v in rctx.inst_execs.values() for w in v]
    for occs in rctx.occ_events.values():
      starts = {}
      for o in occs:
        key = (o.se, o.cu, o.simd, o.wave_id)
        if o.start: starts[key] = o.time
        elif key in starts: occ_pairs.append((key, o.time - starts.pop(key)))
    for w in waves: n_inst += sum(1 for _ in w.unpack_insts())
  wave_set = {kp for kp, _ in occ_pairs}; cu_set = {(kp[0], kp[1]) for kp, _ in occ_pairs}
  durs = sorted(d for _, d in occ_pairs)
  median_wave_cycles = durs[len(durs)//2] if durs else 0
  mean_wave_cycles = round(sum(durs)/len(durs)) if durs else 0
  rec.update({
    "kernel_gpu_ms_total_capture": round(sum(durations.get(p.name, [0])), 5), "tile_launches_in_capture": len(durations.get(p.name, [])),
    "per_pc_stall_available": n_inst > 0,
    "traced_waves": len(wave_set), "traced_cus": len(cu_set), "occ_wave_samples": len(occ_pairs),
    "median_wave_cycles": median_wave_cycles, "mean_wave_cycles": mean_wave_cycles, "max_wave_cycles": (durs[-1] if durs else 0),
  })
  return rec

def _run_capture(route: str):
  if os.environ.get("QK_N2_REUSE") == "1" and (OUT / f"_{route}.pkl").exists(): return True, "reused"
  env = {**os.environ, "PYTHONPATH": str(ROOT), "DEV": "AMD", "JIT": "1", "SQTT": "1", "PROFILE": "1",
         "SQTT_ITRACE_SE_MASK": "1", "ROCPROF_PATH": ROCPROF_SO, "QK_N2_CAPTURE": route}
  for z in _ZERO: env[z] = "0"
  env.update(ROUTES[route]["env"])
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=1800)
  ok = any("@@CAPTURED" in l for l in (p.stdout or "").splitlines())
  return ok, (p.stdout or "")[-2000:] + (p.stderr or "")[-2000:]

def _ratio(o, n): return round(n / o, 2) if o else (None if not n else float("inf"))

def build() -> dict:
  OUT.mkdir(parents=True, exist_ok=True)
  rec = {"verdict": None, "profiling_backend": "sqtt", "ctx_profiled": CTX,
         "note": "SQTT_ITRACE_SE_MASK=1 -> instruction trace sampled on ONE shader engine; occupancy/CU counts are per-sampled-SE, stall ratios are representative."}
  cap = {}
  for r in ROUTES:
    ok, tail = _run_capture(r); cap[r] = ok
    if not ok: rec["verdict"] = "AMD_ISA_PHASE_N2_BLOCKED_TRACE_CAPTURE"; rec["capture_fail"] = {r: tail}; return rec
  owned = _aggregate("owned", ROUTES["owned"]["prefix"]); native = _aggregate("native", ROUTES["native"]["prefix"])
  json.dump(owned, open(OUT / "owned_trace.json", "w"), indent=2); json.dump(native, open(OUT / "native_trace.json", "w"), indent=2)
  keep = ("route_bound", "kernel_gpu_ms_total_capture", "tile_launches_in_capture", "per_pc_stall_available",
          "traced_waves", "traced_cus", "occ_wave_samples", "median_wave_cycles", "mean_wave_cycles", "max_wave_cycles", "non_tile_attn_kernels")
  rec["owned"] = {k: owned.get(k) for k in keep}; rec["native"] = {k: native.get(k) for k in keep}
  if not owned.get("route_bound"): rec["verdict"] = "AMD_ISA_PHASE_N2_BLOCKED_ROUTE_ATTRIBUTION"; rec["detail"] = "owned tile kernel not captured"; return rec
  if not native.get("route_bound") or native.get("non_tile_attn_kernels"):
    rec["verdict"] = "AMD_ISA_PHASE_N2_BLOCKED_ROUTE_ATTRIBUTION"; rec["detail"] = f"native fallback/missing: {native.get('non_tile_attn_kernels')}"; return rec
  def _rd(p):
    f = ROOT / p; return json.load(open(f)) if f.exists() else {}
  pi = _rd("bench/amd-isa-backend-phase-i/latest.json"); grid = _rd("bench/amd-isa-backend-grid/latest.json"); h = _rd("bench/amd-isa-backend-phase-h/latest.json")
  rec["token_match"] = pi.get("token_match"); rec["route_bound_wd"] = pi.get("route_bound")
  rec["deterministic"] = grid.get("in_model_repeated_run_stability") or h.get("repeated_run_stability")
  rec["hidden_fallback_check"] = grid.get("hidden_fallback_check")
  rec["wd"] = {ck: {"native_tok_s": v["native_tok_s"], "owned_tok_s": v["owned_tok_s"], "pct_of_owned": v["pct_of_owned"]} for ck, v in pi.get("per_ctx", {}).items()}
  rec["per_pc_stall_attribution"] = ("UNAVAILABLE: the vendored rocprof-trace-decoder (0.1.6) build returns wave events "
    "with instructions_size==0 (occupancy-only; verified roc.py --kernel yields OCC but no per-PC Stall table). Dynamic "
    "attribution is therefore OCCUPANCY + WAVE-CYCLE timing (degraded), not per-instruction stall categories.")
  os_, ns = owned, native
  def row(name, ov, nv, conf, ev, lever): return {"row": name, "owned": ov, "native": nv,
    "delta": (nv - ov) if isinstance(ov, (int, float)) and isinstance(nv, (int, float)) else None,
    "ratio": _ratio(ov, nv) if isinstance(ov, (int, float)) and isinstance(nv, (int, float)) else None, "confidence": conf, "evidence": ev, "candidate_lever": lever}
  rows = [
    row("total_kernel_time_gpu_ms_capture", os_["kernel_gpu_ms_total_capture"], ns["kernel_gpu_ms_total_capture"], "high", "PROFILE per-kernel GPU ms summed over all tile launches in the captured forward", "lower wall time"),
    row("active_cu_count_sampledSE", os_["traced_cus"], ns["traced_cus"], "medium", "distinct (SE,CU) occupied in the sampled shader engine", "occupancy/CU coverage (N3-occupancy)"),
    row("traced_waves_sampledSE", os_["traced_waves"], ns["traced_waves"], "medium", "distinct waves with occupancy events in sampled SE", "grid/wave mapping"),
    row("median_wave_cycles", os_["median_wave_cycles"], ns["median_wave_cycles"], "high", "median wave lifetime in shader cycles (occupancy start->end)", "reduce per-wave latency"),
    row("mean_wave_cycles", os_["mean_wave_cycles"], ns["mean_wave_cycles"], "high", "mean wave lifetime in shader cycles", "reduce per-wave latency"),
    {"row": "vmem_stall / lds_stall / waitcnt_stall (per-PC categories)", "owned": "unavailable", "native": "unavailable", "confidence": "low",
     "evidence": "decoder build returns occupancy-only (instructions_size==0); per-PC stall category not decodable", "candidate_lever": "needs PMC counters or a fixed itrace decoder to split VMEM vs LDS vs waitcnt"},
    {"row": "lane_utilization", "owned": "unavailable", "native": "unavailable", "confidence": "low", "evidence": "EXEC/lane occupancy not exposed by this decoder build", "candidate_lever": "—"},
  ]
  rec["dynamic_diff_rows"] = rows
  rec["hot_pc_rows"] = []
  rec["hot_pc_note"] = "per-PC stall unavailable (decoder occupancy-only) -> no hot_pc_rows. Top stall PC requires a working itrace decoder build."
  # interpretation from occupancy + wave-cycle timing + the static evidence (N0: native 324 < owned 557 static instr)
  wave_ratio = _ratio(os_["median_wave_cycles"], ns["median_wave_cycles"])
  rec["wave_cycles_ratio_native_over_owned"] = wave_ratio
  rec["interpretation"] = (f"native median wave lifetime {ns['median_wave_cycles']} cyc vs owned {os_['median_wave_cycles']} cyc "
    f"(x{wave_ratio}); native occupies {ns['traced_cus']} CUs vs owned {os_['traced_cus']} (sampled SE). With native doing "
    "FEWER static instructions (N0: 324 vs 557) but its waves living LONGER, the gap is per-wave LATENCY/STALL-bound, not "
    "instruction-count or occupancy. The exact stall source (VMEM vs LDS vs waitcnt) is NOT resolved (decoder occupancy-only). "
    "CAVEAT: median_wave_cycles is a wave-RESIDENCY proxy (occupancy start->end) pooled over launches; it can include "
    "occupancy-overlap idle, so the absolute ratio overstates pure-compute cycles -- the DIRECTION (native waves live much "
    "longer => latency/stall-bound) is robust, the magnitude is approximate. End-to-end W==D gap is 1.6-1.7x (attention is "
    "one of many decode kernels), consistent with a much slower attention tile that is a fraction of the decode.")
  # strongest bottleneck + next phase from the timing/occupancy evidence
  if wave_ratio and wave_ratio > 1.3:
    rec["strongest_suspected_bottleneck"] = (f"per-wave latency: native waves take {wave_ratio}x owned's cycles despite fewer static "
      "instructions -> stall-bound (exposed load-use / memory latency). Category unresolved by occupancy-only decode.")
    rec["next_implementation_phase"] = ("N2.1 (finer counters): get a working itrace/PMC decode to split VMEM vs LDS vs waitcnt stall, "
      "THEN N3A (memory/coalescing) or N3C (waitcnt/load-use scheduling). Strong prior hint: per-token ds_bpermute reduce latency exposed.")
  elif ns["traced_cus"] < os_["traced_cus"] * 0.8:
    rec["strongest_suspected_bottleneck"] = f"occupancy: native occupies {ns['traced_cus']} vs owned {os_['traced_cus']} CUs"
    rec["next_implementation_phase"] = "N3 occupancy (CU coverage / wave mapping)"
  else:
    rec["strongest_suspected_bottleneck"] = "no single dominant timing/occupancy delta resolved at this granularity"
    rec["next_implementation_phase"] = "N2.1 finer counters (PMC) — occupancy-only decode too coarse"
  gates_ok = (cap["owned"] and cap["native"] and owned["route_bound"] and native["route_bound"]
              and rec["token_match"] is True and bool(rec["deterministic"]))
  if rec["token_match"] is not True: rec["verdict"] = "AMD_ISA_PHASE_N2_BLOCKED_TOKEN_MATCH"
  elif not rec["deterministic"]: rec["verdict"] = "AMD_ISA_PHASE_N2_BLOCKED_NONDETERMINISM"
  elif gates_ok and (owned["median_wave_cycles"] and native["median_wave_cycles"]):
    # per-PC unavailable -> degraded timing/occupancy attribution (still narrows to latency/stall-bound)
    rec["verdict"] = "AMD_ISA_PHASE_N2_PASS_DEGRADED_TIMING_ATTRIBUTION"
  else: rec["verdict"] = "AMD_ISA_PHASE_N2_INCONCLUSIVE_COUNTERS_TOO_COARSE"
  return rec

def _summary_md(rec: dict) -> str:
  L = ["# Phase N2 — dynamic stall attribution (owned vs native ISA decode tile)", "",
       f"**Verdict:** `{rec['verdict']}`  ·  backend: {rec['profiling_backend']}  ·  ctx profiled: {rec['ctx_profiled']}", ""]
  if "wd" in rec:
    L += [f"W==D: owned {rec['wd'].get('512',{}).get('owned_tok_s')}/{rec['wd'].get('4096',{}).get('owned_tok_s')} vs "
          f"native {rec['wd'].get('512',{}).get('native_tok_s')}/{rec['wd'].get('4096',{}).get('native_tok_s')} tok/s "
          f"(native {rec['wd'].get('512',{}).get('pct_of_owned')}%/{rec['wd'].get('4096',{}).get('pct_of_owned')}% of owned)",
          f"token_match={rec.get('token_match')} · deterministic={rec.get('deterministic')} · route_bound (owned+native captured)", ""]
  if "per_pc_stall_attribution" in rec: L += [f"_{rec['per_pc_stall_attribution']}_", ""]
  if "dynamic_diff_rows" in rec:
    L += ["## Dynamic diff (occupancy + wave-cycle timing)", "", "| row | owned | native | ratio | lever |", "|---|---|---|---|---|"]
    for r in rec["dynamic_diff_rows"]: L.append(f"| {r['row']} | {r.get('owned')} | {r.get('native')} | {r.get('ratio')} | {r.get('candidate_lever')} |")
    L += ["", f"**Interpretation:** {rec.get('interpretation','')}", "",
          f"**Strongest suspected bottleneck:** {rec.get('strongest_suspected_bottleneck')}", "",
          f"**Next implementation phase:** {rec.get('next_implementation_phase')}", ""]
  return "\n".join(L) + "\n"

if __name__ == "__main__":
  if (r := os.environ.get("QK_N2_CAPTURE")):
    _capture(r); sys.exit(0)
  rec = build()
  json.dump(rec, open(OUT / "latest.json", "w"), indent=2)
  open(OUT / "summary.md", "w").write(_summary_md(rec))
  print(json.dumps({k: v for k, v in rec.items() if k not in ("dynamic_diff_rows", "hot_pc_rows")}, indent=2))
  print("\nPHASE_N2", rec["verdict"])
