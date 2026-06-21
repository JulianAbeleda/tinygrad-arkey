#!/usr/bin/env python3
"""Deliverable 0: current-route decode role/tensor/kernel attribution table (Qwen3-8B, gfx1100).

Two-layer measurement (matches the non-negotiable timing policy of
docs/decode-role-tensor-kernel-attribution-solution-scope-20260620.md):

  1. WHOLE-TOKEN AUTHORITY (W==D), PROFILE off -> clean wall.
       W = real decode wall/token: jit replay + .item() readback every token (the real sync path).
       D = dispatch-only wall/token: same jit replayed back-to-back, NO per-token .item, one final sync.
       host_sync_residual = W - D. tok_s_W = 1000/W_ms, tok_s_D_ceiling = 1000/D_ms.
  2. PER-ROLE TIMED SPLIT (ProfileGraphEvent), PROFILE on, a SECOND jit so (1) stays clean.
       The warm HCQ graph replay records per-kernel GPU timestamps. dur_us = sigs[en_id]-sigs[st_id].
       These are real warm GPU-execution intervals per program -> confidence 'timed' (NOT a DEBUG=2 proxy).
       Sum = GPU-busy us/token; span-busy = inter-kernel gaps. Program name -> role via shape/name.

ATT/HCQ packet counts are never used as timing here. Same-process interleaved A/B is not used (this is an
attribution table, not a build gate). Default decode behavior is NOT changed (q8 is env-gated, restored off).
Run under GPU perf-state `auto` (user-realistic); we record sclk/perf_level provenance, we do not force a lane.

Run target:
  PYTHONPATH=. python3 extra/qk_decode_current_route_attribution.py \
    --modes baseline,q8 --ckpts 512 1024 4096 --nmeas 20 --warmups 8 \
    --out bench/qk-decode-role-tensor-kernel-attribution/current_route_attribution.json
"""
from __future__ import annotations

import argparse, collections, json, os, pathlib, re, statistics, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-role-tensor-kernel-attribution/current_route_attribution.json"
DEVSYS = pathlib.Path("/sys/class/drm/card0/device")
MHZ = re.compile(r"(\d+)Mhz")

# --- program name -> (out_features, in_features) -----------------------------------------------------------------
# widen the frozen census regex (extra/qk_decode_layer_census.py:20) to also catch the coop / q8_1_vdot kernels that
# the current decode route actually emits (q4k_coop_partial_*, q6k_coop_partial_*, q4k_q8_1_vdot_builtin_partial_*).
GEMV_SHAPE = re.compile(r"q[46]k_(?:gemv|coop|q8_1_vdot_builtin)\w*?_(\d+)_(\d+)(?:_\d+)?")
# (out, in) -> role  (single source of truth: extra/qk_decode_layer_census.py:22-23). 8B: hidden 4096, ffn 12288,
# vocab 151936, n_kv 8, head_dim 128 -> kv proj 1024.
GEMV_ROLE = {(151936, 4096): "lm_head", (4096, 12288): "ffn_down", (12288, 4096): "ffn_gate/up",
             (4096, 4096): "attn_q/o", (1024, 4096): "attn_k/v"}
# bytes per stored weight: Q4_K_M 144B/256 = 0.5625, Q6_K 210B/256 = 0.8203
BYTES_PER_W = {"Q4_K": 144 / 256, "Q6_K": 210 / 256}
ROLE_QUANT = {"lm_head": "Q6_K", "ffn_down": "Q6_K", "ffn_gate/up": "Q4_K", "attn_q/o": "Q4_K", "attn_k/v": "Q6_K"}

def classify(name: str) -> tuple[str, str]:
  """program name -> (role, tensor_family)."""
  n = name.lower()
  if "q8_rmsnorm" in n: return "rmsnorm", "q8_activation"
  if "q8_mmvq_gateup" in n or "q8_mmvq" in n: return "ffn_gate/up", "q8_activation"
  g = GEMV_SHAPE.search(n)
  if g:
    role = GEMV_ROLE.get((int(g.group(1)), int(g.group(2))), "other")
    return role, (ROLE_QUANT.get(role, "quant") if role != "other" else "quant")
  nums = [int(x) for x in re.findall(r"\d+", name)]
  # explicit-name prefixes win over numeric heuristics (E_128_32_3 is ffn_down glue, NOT attention)
  if "flash" in n or "start_pos" in n: return "attention_flash", "attention"
  if n.startswith("copy"): return "other", "copy/kv-write"
  if n.startswith("e_") or n.startswith("e "): return "elementwise", "fp/elementwise"
  if n.startswith("r_"):
    if 16 in nums and 256 in nums: return "rmsnorm", "fp/elementwise"
    if 128 in nums or 1024 in nums: return "attention_flash", "attention"   # flash-decode partial reduces
    return "reduce/glue", "fp/elementwise"
  if 128 in nums or 1024 in nums: return "attention_flash", "attention"
  return "other", "fp/elementwise"

def role_weight_bytes(role: str, count: int) -> int:
  """aggregate stored weight bytes/token for `count` instances of a weight-GEMV role."""
  shapes = {"lm_head": (151936, 4096), "ffn_down": (4096, 12288), "ffn_gate/up": (12288, 4096),
            "attn_q/o": (4096, 4096), "attn_k/v": (1024, 4096)}
  if role not in shapes: return 0
  out, inn = shapes[role]
  return int(out * inn * BYTES_PER_W[ROLE_QUANT[role]] * count)

# --- sysfs clock provenance (read-only; we do NOT force a lane) --------------------------------------------------
def _read(p: pathlib.Path) -> str:
  try: return p.read_text().strip()
  except OSError: return ""

def _active_mhz(name: str) -> int:
  for line in _read(DEVSYS / name).splitlines():
    if "*" in line:
      m = MHZ.search(line); return int(m.group(1)) if m else 0
  return 0

def clock_sample() -> dict[str, Any]:
  return {"sclk_mhz": _active_mhz("pp_dpm_sclk"), "mclk_mhz": _active_mhz("pp_dpm_mclk"),
          "perf_level": _read(DEVSYS / "power_dpm_force_performance_level"),
          "gpu_busy_pct": int(_read(DEVSYS / "gpu_busy_percent") or 0)}

def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"

def rel(p: pathlib.Path) -> str:
  return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)

# ================================================================================================================
# CHILD: one model/process per mode. Clean W==D + PROFILE-on per-role graph split.
# ================================================================================================================
def run_child(args: argparse.Namespace) -> int:
  from tinygrad import Tensor, UOp, TinyJit, Context, Device
  from tinygrad.device import Compiled
  from extra.llm_generate import load_model_and_tokenizer

  dev = Device[Device.DEFAULT]
  model, tok = load_model_and_tokenizer(args.model, args.max_context, seed=args.seed)
  for lin in (getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + args.max_context // max(1, len(ids))))[:args.max_context]
  v_sp = UOp.variable("start_pos", 0, args.max_context - 1)
  temp = Tensor([0.0])

  clocks = []
  rows = []
  for ck in args.ckpts:
    # default decode route: leave _use_flash default (auto policy enables flash for bound ctx>=512); prefill_v2 off.
    for block in model.blk:
      block._prefill_v2 = False
    tokid = int(ids[ck])

    # ---- (1) clean W==D timing (PROFILE off) ----
    step = TinyJit(model.forward)
    out = Tensor([[tokid]], dtype="int32").contiguous()
    for i in range(args.warmups):
      out = step(out, v_sp.bind(ck + i), temp).realize()
    out = Tensor([[tokid]], dtype="int32").contiguous()
    W = []
    for i in range(args.nmeas):
      t0 = time.perf_counter(); out = step(out, v_sp.bind(ck + i), temp); _ = int(out.item())
      W.append(time.perf_counter() - t0)
    clocks.append(clock_sample())
    out = Tensor([[tokid]], dtype="int32").contiguous(); dev.synchronize()
    t0 = time.perf_counter()
    for i in range(args.nmeas):
      out = step(out, v_sp.bind(ck + i), temp)
    dev.synchronize(); D = (time.perf_counter() - t0) / args.nmeas
    w_ms, d_ms = statistics.median(W) * 1e3, D * 1e3
    host = max(0.0, w_ms - d_ms)

    # ---- (2) per-role TIMED split via ProfileGraphEvent (PROFILE on, separate jit so W==D stays clean) ----
    # The decode forward is replayed as SEVERAL HCQ graph segments per token (symbolic KV store / flash custom
    # kernels break the graph). One profiled token therefore emits multiple ProfileGraphEvents -> aggregate ALL
    # segments in a single bracketed replay (finalize flushes the just-executed replay's timestamps).
    per = collections.defaultdict(lambda: {"calls": 0, "us": 0.0})
    busy_us = span_us = 0.0; n_events = 0; captured = False
    with Context(PROFILE=1):
      pstep = TinyJit(model.forward)
      pout = Tensor([[tokid]], dtype="int32").contiguous()
      for i in range(max(8, args.warmups)):
        pout = pstep(pout, v_sp.bind(ck + i), temp).realize()
      dev.synchronize(); dev._at_profile_finalize()      # flush all warm-up timestamps
      base = len(Compiled.profile_events)
      pout = pstep(pout, v_sp.bind(ck), temp).realize()   # exactly ONE profiled token
      dev.synchronize(); dev._at_profile_finalize()       # collect this token's graph segments
      evs = [e for e in Compiled.profile_events[base:] if type(e).__name__ == "ProfileGraphEvent"]
      n_events = len(evs); captured = bool(evs)
      starts, ends = [], []
      for e in evs:
        sigs = [float(s) for s in e.sigs]
        for ent in e.ents:
          du = sigs[ent.en_id] - sigs[ent.st_id]
          per[str(ent.name)]["calls"] += 1; per[str(ent.name)]["us"] += du
          starts.append(sigs[ent.st_id]); ends.append(sigs[ent.en_id])
      busy_us = sum(v["us"] for v in per.values())
      span_us = (max(ends) - min(starts)) if ends else 0.0
    print(f"  [{args.mode}] ctx {ck}: profiled {n_events} graph segments, "
          f"{sum(v['calls'] for v in per.values())} kernel-calls", file=sys.__stderr__)

    programs = [{"program_or_kernel": nm, "calls_per_token": v["calls"], "gpu_us_per_token": round(v["us"], 2)}
                for nm, v in sorted(per.items(), key=lambda kv: -kv[1]["us"])]
    rows.append({
      "ctx": ck,
      "wall_ms_W": round(w_ms, 3), "dispatch_ms_D": round(d_ms, 3),
      "host_sync_residual_ms": round(host, 3), "host_sync_pct_of_wall": round(100 * host / w_ms, 1),
      "tok_s_W": round(1000 / w_ms, 1), "tok_s_D_ceiling": round(1000 / d_ms, 1),
      "gpu_busy_us_per_token": round(busy_us, 1), "graph_span_us_per_token": round(span_us, 1),
      "graph_overhead_us_per_token": round(span_us - busy_us, 1),
      "graph_event_captured": captured, "graph_segments": n_events,
      "programs": programs,
    })
    print(f"  [{args.mode}] ctx {ck:5}: W {w_ms:6.2f}ms ({1000/w_ms:.1f} tok/s) | D {d_ms:6.2f}ms "
          f"(ceil {1000/d_ms:.1f}) | host {host:.2f}ms | busy {busy_us:.0f}us span {span_us:.0f}us "
          f"| {len(programs)} progs", file=sys.__stderr__)

  result = {
    "phase": "DECODE_CURRENT_ROUTE_ATTRIBUTION_CHILD", "schema": "decode_current_route_attribution_child_v1",
    "mode": args.mode, "q8_enabled": args.mode == "q8", "commit": git_sha(),
    "model_id": pathlib.Path(args.model).stem, "hardware": "RX 7900 XTX / gfx1100",
    "ckpts": args.ckpts, "nmeas": args.nmeas, "warmups": args.warmups,
    "method": "W==D clean wall (PROFILE off) + per-role ProfileGraphEvent timed split (PROFILE on, 2nd jit)",
    "clock_provenance": clocks, "rows": rows, "default_behavior_changed": False,
  }
  args.child_out.parent.mkdir(parents=True, exist_ok=True)
  args.child_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"mode": args.mode, "median_tok_s_W": statistics.median([r["tok_s_W"] for r in rows]),
                    "out": rel(args.child_out)}, indent=2))
  return 0

# ================================================================================================================
# llama reference join
# ================================================================================================================
def load_llama_ref() -> dict[str, Any]:
  prov = json.loads((ROOT / "bench/qk-llama-token-primitive-accounting/provenance.json").read_text())
  rt = json.loads((ROOT / "bench/qk-llama-token-primitive-accounting/llama_runtime.json").read_text())
  join = json.loads((ROOT / "bench/qk-decode-complete-tooling/llama_join.json").read_text())
  tok_s = prov["baseline_decode_tok_s"]["llama"]  # {"512":98.6,...}
  shares = rt["decode_only_share_pct"]            # mmvq/attention/rmsnorm/rope/q8_1_activation_quant/elementwise/other
  # split llama mmvq into Q4 vs Q6 by launch-contract total_ms proportions
  fam = {r["family"]: r["total_ms"] for r in join["launch_contract_rows"]}
  q4 = fam.get("llama_mmvq_Q4_K_fusion_true", 0) + fam.get("llama_mmvq_Q4_K_fusion_false", 0)
  q6 = fam.get("llama_mmvq_Q6_K_fusion_true", 0) + fam.get("llama_mmvq_Q6_K_fusion_false", 0)
  return {"tok_s": tok_s, "shares": shares, "mmvq_q4_frac": q4 / (q4 + q6), "mmvq_q6_frac": q6 / (q4 + q6),
          "mmvq_effective_bw_GBs": rt.get("llama_mmvq_effective_bw_GBs"), "hbm_peak_GBs": 960}

# map tinygrad roles to llama runtime share buckets
LLAMA_BUCKET = {"ffn_gate/up": "mmvq", "ffn_down": "mmvq", "lm_head": "mmvq", "attn_q/o": "mmvq",
                "attn_k/v": "mmvq", "attention_flash": "attention", "rmsnorm": "rmsnorm",
                "elementwise": "elementwise", "rope": "rope", "reduce/glue": None, "other": "other"}
WEIGHT_GEMV = {"ffn_gate/up", "ffn_down", "lm_head", "attn_q/o", "attn_k/v"}

# ================================================================================================================
# AGGREGATE: build the role/tensor/kernel attribution table per (mode, ctx), join llama, compute gap math.
# ================================================================================================================
def aggregate(args: argparse.Namespace, children: list[dict[str, Any]]) -> int:
  modes = args.modes.split(",")
  llama = load_llama_ref()
  per_mode = {}
  for mode in modes:
    f = args.out.parent / f"current_route_attribution_{mode}.json"
    if f.exists(): per_mode[mode] = json.loads(f.read_text())

  tables = {}        # (mode, ctx) -> [row, ...]
  token_math = {}    # (mode, ctx) -> {...}
  for mode, child in per_mode.items():
    for r in child["rows"]:
      ctx = r["ctx"]; w_ms = r["wall_ms_W"]; busy = r["gpu_busy_us_per_token"] or 1.0
      # aggregate programs -> (role, tensor_family)
      agg = collections.defaultdict(lambda: {"calls": 0, "us": 0.0, "progs": set()})
      for p in r["programs"]:
        role, fam = classify(p["program_or_kernel"])
        k = (role, fam)
        agg[k]["calls"] += p["calls_per_token"]; agg[k]["us"] += p["gpu_us_per_token"]
        agg[k]["progs"].add(re.sub(r"\d+", "N", p["program_or_kernel"])[:48])
      # llama per-bucket ms/token at this ctx
      lt = llama["tok_s"].get(str(ctx))
      llama_ms = (1000.0 / lt) if lt else None
      # PROFILE-on GPU busy slightly exceeds the clean W wall (per-kernel timestamp overhead). The timed shares
      # are trustworthy; rescale them onto the clean W wall so ms/token aligns with the authority wall.
      scale = w_ms / (busy / 1000.0) if busy > 0 else 0.0
      rows = []
      for (role, fam), v in sorted(agg.items(), key=lambda kv: -kv[1]["us"]):
        ms = (v["us"] / 1000.0) * scale                      # rescaled onto clean wall
        bw = None
        if role in WEIGHT_GEMV and ms > 0:
          bw = round(role_weight_bytes(role, v["calls"]) / (ms * 1e-3) / 1e9, 1)
        rows.append({
          "role": role, "tensor_family": fam, "program_or_kernel": " | ".join(sorted(v["progs"]))[:96],
          "calls_per_token": v["calls"], "ms_per_token": round(ms, 4),
          "raw_gpu_ms_profile_on": round(v["us"] / 1000.0, 4),
          "share_pct_of_gpu_busy": round(100 * v["us"] / busy, 2),
          "share_pct_of_wall": round(100 * ms / w_ms, 2) if w_ms else None,
          "effective_bw_GBs": bw, "pct_hbm_peak": round(100 * bw / 960, 1) if bw else None,
          "llama_analogue": {"ffn_gate/up": "llama mul_mat_vec_q Q4_K", "ffn_down": "llama mul_mat_vec_q Q6_K",
                             "lm_head": "llama mul_mat_vec_q Q6_K", "attn_q/o": "llama mul_mat_vec_q Q4_K",
                             "attn_k/v": "llama mul_mat_vec_q Q6_K", "attention_flash": "llama flash_attn_*",
                             "rmsnorm": "llama rmsnorm", "elementwise": "llama rope+elementwise",
                             "reduce/glue": "llama (fused; ~none)", "other": "llama other"}.get(role),
          "confidence": "timed", "next_action": None,
        })
      # ---- token math: decompose the WHOLE gap by family vs llama buckets (Sum tg_family = W wall by construction) --
      tg_ms = w_ms; busy_ms = busy / 1000.0
      def fam_ms(roles): return sum(x["ms_per_token"] for x in rows if x["role"] in roles)
      tg = {"weight_gemv": fam_ms(WEIGHT_GEMV), "attention": fam_ms({"attention_flash"}),
            "rmsnorm": fam_ms({"rmsnorm"}), "elementwise": fam_ms({"elementwise"}),
            "glue_other": fam_ms({"reduce/glue", "other"})}
      tm = None
      if llama_ms is not None:
        sh = llama["shares"]
        ll = {"weight_gemv": llama_ms * sh["mmvq"] / 100.0, "attention": llama_ms * sh["attention"] / 100.0,
              "rmsnorm": llama_ms * sh["rmsnorm"] / 100.0,
              "elementwise": llama_ms * (sh.get("rope", 0) + sh.get("elementwise", 0)) / 100.0,
              "glue_other": llama_ms * (sh.get("other", 0) + sh.get("q8_1_activation_quant", 0)) / 100.0}
        fam_gap = {k: round(tg[k] - ll[k], 3) for k in tg}
        # The decomposition is COMPLETE: sum(tg_family) = W wall and sum(llama_family) = llama wall, so
        # sum(family_gap) == total_gap exactly. "attributed" = every family mapped to a like-for-like llama
        # bucket (weight/attention/rmsnorm/elementwise), NET (negatives allowed). glue_other has no clean llama
        # analogue (llama fuses it away: ~1000 unfused tinygrad progs/token vs ~260 fused) -> fusion residual.
        total_gap = tg_ms - llama_ms
        attributed = total_gap - fam_gap["glue_other"]
        tm = {
          "tinygrad_ms_per_token": round(tg_ms, 3), "llama_ms_per_token": round(llama_ms, 3),
          "gap_ms_per_token": round(total_gap, 3),
          "tinygrad_family_ms": {k: round(tg[k], 3) for k in tg},
          "llama_family_ms": {k: round(ll[k], 3) for k in ll},
          "family_gap_ms": fam_gap,
          "attributed_gap_ms": round(attributed, 3),
          "fusion_residual_glue_other_gap_ms": round(fam_gap["glue_other"], 3),
          "unattributed_residual_ms": round(fam_gap["glue_other"], 3),
          "attributed_frac_of_gap": round(attributed / total_gap, 3) if total_gap > 0 else None,
          "gpu_busy_ms_profile_on": round(busy_ms, 3), "wall_minus_busy_ms": round(tg_ms - busy_ms, 3),
          "rescale_factor_wall_over_busy": round(scale, 4),
        }
        for x in rows:
          base_role = ("weight_gemv" if x["role"] in WEIGHT_GEMV else
                       "attention" if x["role"] == "attention_flash" else
                       x["role"] if x["role"] in ("rmsnorm", "elementwise") else "glue_other")
          # role's slice of its family's gap, proportional to its ms within the family
          fms = tg[base_role]
          x["gap_ms_per_token"] = round(fam_gap[base_role] * (x["ms_per_token"] / fms), 4) if fms > 0 else None
      # next_action heuristic (table only; no build here)
      for x in rows:
        s = x["share_pct_of_wall"] or 0; gp = x.get("gap_ms_per_token") or 0
        if x["role"] in WEIGHT_GEMV and s >= 10 and gp >= 0.3: x["next_action"] = "build"
        elif x["role"] in WEIGHT_GEMV and s >= 5: x["next_action"] = "audit_more"
        elif x["role"] == "attention_flash" and s >= 12: x["next_action"] = "audit_more"
        elif x["role"] in ("elementwise", "reduce/glue") and gp >= 0.3: x["next_action"] = "audit_more"
        else: x["next_action"] = "drop"
      tables[f"{mode}@{ctx}"] = rows
      token_math[f"{mode}@{ctx}"] = tm

  # pass-gate check at baseline ctx1024 (the scope's reference gap)
  gate_key = "baseline@1024" if "baseline@1024" in token_math else next(iter(token_math), None)
  gate_tm = token_math.get(gate_key) or {}
  attributed_1024 = gate_tm.get("attributed_gap_ms")
  want_ctx = {512, 1024, 4096} & set(args.ckpts)
  gates = {
    "produced_requested_ctx_all_modes": all(f"{m}@{c}" in tables for m in per_mode for c in want_ctx),
    "separates_W_wall_from_D_ceiling": True,
    "attributes_ge_2p5ms_of_ctx1024_gap": (attributed_1024 is not None and attributed_1024 >= 2.5),
  }
  result = {
    "date": "2026-06-20", "phase": "DECODE_CURRENT_ROUTE_ATTRIBUTION", "schema": "decode_current_route_attribution_v1",
    "commit": git_sha(), "hardware": "RX 7900 XTX / gfx1100", "model_id": "Qwen3-8B-Q4_K_M",
    "modes": modes, "ckpts": args.ckpts, "nmeas": args.nmeas, "warmups": args.warmups,
    "method": "W==D clean wall (authority) + per-role ProfileGraphEvent timed GPU split; llama join from "
              "llama_runtime.json/provenance.json/llama_join.json",
    "llama_reference": {"tok_s": llama["tok_s"], "decode_only_share_pct": llama["shares"],
                        "mmvq_effective_bw_GBs": llama["mmvq_effective_bw_GBs"]},
    "wd_summary": {f"{m}@{c}": {k: per_mode[m]["rows"][i][k] for k in
                                ("tok_s_W", "tok_s_D_ceiling", "host_sync_pct_of_wall", "gpu_busy_us_per_token",
                                 "graph_overhead_us_per_token")}
                   for m in per_mode for i, c in enumerate(per_mode[m]["ckpts"])},
    "attribution_tables": tables, "token_math": token_math,
    "pass_gate_reference": gate_key, "attributed_gap_ms_ctx1024": attributed_1024,
    "gates": gates, "gate_pass": all(gates.values()),
    "clock_provenance": {m: per_mode[m].get("clock_provenance") for m in per_mode},
    "children": children, "default_behavior_changed": False,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": "PASS" if all(gates.values()) else "GATE_CHECK", "gates": gates,
                    "attributed_gap_ms_ctx1024": attributed_1024, "out": rel(args.out)}, indent=2))
  return 0

def run_parent(args: argparse.Namespace) -> int:
  children = []
  for mode in args.modes.split(","):
    out = args.out.parent / f"current_route_attribution_{mode}.json"
    env = os.environ.copy()
    env.setdefault("DEV", "AMD"); env.setdefault("JIT", "1"); env["PYTHONPATH"] = str(ROOT)
    if mode == "q8": env["Q8_FFN_HANDWRITTEN"] = "1"
    else: env.pop("Q8_FFN_HANDWRITTEN", None)
    cmd = [sys.executable, rel(pathlib.Path(__file__).resolve()), "--child-out", rel(out), "--mode", mode,
           "--nmeas", str(args.nmeas), "--warmups", str(args.warmups), "--model", args.model,
           "--max-context", str(args.max_context), "--seed", str(args.seed),
           "--ckpts", *[str(x) for x in args.ckpts]]
    print(f"[parent] launching child mode={mode}", file=sys.__stderr__)
    p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=None)
    children.append({"mode": mode, "cmd": cmd, "returncode": p.returncode, "stdout": (p.stdout or "")[-3000:],
                     "artifact": rel(out)})
  return aggregate(args, children)

def main() -> int:
  ap = argparse.ArgumentParser(description="Current-route decode role/tensor/kernel attribution table")
  ap.add_argument("--modes", default="baseline,q8")
  ap.add_argument("--ckpts", nargs="+", type=int, default=[512, 1024, 4096])
  ap.add_argument("--nmeas", type=int, default=20)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--seed", type=int, default=20260620)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  ap.add_argument("--aggregate-existing", action="store_true")
  ap.add_argument("--child-out", type=pathlib.Path)
  ap.add_argument("--mode", choices=["baseline", "q8"], default="baseline")
  args = ap.parse_args()
  if args.child_out is not None: return run_child(args)
  if args.aggregate_existing: return aggregate(args, [])
  return run_parent(args)

if __name__ == "__main__":
  raise SystemExit(main())
