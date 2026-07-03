"""Project-wide machine-search LEDGER — one schema for every search lane (decode, prefill, codegen, cross-shape,
small-op). Turns fragmented per-lane results into a single durable memory:

  candidate -> lane -> knobs -> gates -> authority benchmark -> verdict -> artifact links -> learned rule

Append-only JSONL at bench/qk-project-search-ledger/ledger.jsonl; schema at .../schema.json. The authority benchmark
field MUST be a whole-path synced metric (W==D / whole-prefill) or an explicit non-promotion microprimitive note --
never a local/PROFILE/no-sync timing (per the harness SOP). See docs/project-search-ledger-contract-20260623.md.

  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_project_search_ledger.py --seed   # (re)build from known results
"""
from __future__ import annotations
import os, sys, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-project-search-ledger"; OUT.mkdir(parents=True, exist_ok=True)
LEDGER = OUT / "ledger.jsonl"; SCHEMA = OUT / "schema.json"

FIELDS = ["candidate_id", "lane", "primitive_class", "knobs", "oracle", "correctness", "route_identity",
          "materialization_abi", "isa", "local_diagnostic", "authority_benchmark", "verdict", "stop_reason",
          "artifact_links", "learned_rule"]
LANES = ["decode", "prefill", "codegen", "cross-shape", "small-op"]
PRIMITIVE_CLASSES = ["attention", "GEMM", "ABI", "fusion", "route_policy", "codegen_microprimitive"]

def validate(e: dict):
  return [f for f in FIELDS if f not in e]

def entry(**kw):
  e = {f: kw.get(f) for f in FIELDS}
  assert e["lane"] in LANES, f"bad lane {e['lane']}"
  assert not validate(e), validate(e)
  return e

def write_schema():
  json.dump({"_schema": "project-wide machine-search ledger entry", "date": "2026-06-23", "fields": FIELDS,
             "lanes": LANES, "primitive_classes": PRIMITIVE_CLASSES,
             "rules": {"authority_benchmark": "MUST be whole-path synced (W==D / whole-prefill) or an explicit non-promotion microprimitive note; never local/PROFILE/no-sync",
                       "correctness": "byte-identical greedy (decode) or rel_rmse + byte-identical greedy (GEMM); checked BEFORE speed",
                       "stop_reason": "the first failed cost-ordered gate (or 'passed all gates')",
                       "learned_rule": "the durable, transferable lesson (links into structure/Development/performance-primitive-research-principles.md when promoted)"}},
            open(SCHEMA, "w"), indent=2)

def ingest_decode_mode_a(entries):
  rf = ROOT / "bench/qk-decode-machine-search/results.jsonl"
  if not rf.exists(): return
  for line in open(rf):
    r = json.loads(line); wd = r.get("wd") or {}
    entries.append(entry(
      candidate_id=f"decode/modeA/{r['id']}", lane="decode", primitive_class="route_policy",
      knobs=r.get("knobs_env"), oracle="bench/qk-decode-search-readiness/baseline_oracle.json",
      correctness=("byte-identical" if r.get("token_byte_identical") else ("n/a" if r.get("reject_reason") else "FAIL")),
      route_identity=("present" if (r.get("route_fire") or {}).get("candidate_kernel_present") else "absent"),
      materialization_abi=("E_49152_absent" if not (r.get("materialization") or {}).get("E_49152_present") else "E_49152_PRESENT"),
      isa=(r.get("isa") or {}).get("verdict"), local_diagnostic=None,
      authority_benchmark={"W==D_tok_s": {c: wd.get(c, {}).get("tok_s") for c in wd}, "delta_vs_oracle_pct_1024": r.get("delta_vs_oracle_pct_1024")},
      verdict=r.get("verdict"), stop_reason=(r.get("reject_reason") or "passed all gates"),
      artifact_links=["docs/decode-machine-search-execution-result-20260623.md", "bench/qk-decode-machine-search/leaderboard.json"],
      learned_rule=("default S=48/base is the policy optimum (no policy knob beats it within spread)" if r["id"]=="S48_base_control" else None)))

def seed():
  entries = []
  ingest_decode_mode_a(entries)
  # shipped wins + experiments (the durable cross-lane memory)
  entries.append(entry(candidate_id="decode/buffer_identity_whole_cache", lane="decode", primitive_class="ABI",
    knobs={"tile": "generated_live_split_g4_8b_kvboth"},
    oracle="pre-fix slice route (owned_flash_tile_gqa, materializes E_49152)", correctness="byte-identical 64-tok x2 prompts",
    route_identity="owned_flash_tile_gqa_whole present", materialization_abi="E_49152 REMOVED (buffer identity)",
    isa="AMD_ISA_PRIMITIVE_CONFIRMED (60 VGPR, 0 spill, v_dot2/LDS/cross-lane)", local_diagnostic=None,
    authority_benchmark={"W==D_delta_pct": {"512": 18.7, "1024": 17.4, "2048": 16.3, "4096": 13.3}, "vs_llama": "102-105%"},
    verdict="WON_SHIPPED_DEFAULT_ON", stop_reason="passed all gates + W==D transfer",
    artifact_links=["docs/owned-tile-buffer-identity-kv-read-result-20260623.md"],
    learned_rule="BUFFER-IDENTITY ABI RULE (principle #12): pass whole buffers (not sliced views) across precompiled-call boundaries; callify materializes slices, reads buffer-identity directly"))
  entries.append(entry(candidate_id="prefill/kv_proj_de_wg_starve", lane="prefill", primitive_class="GEMM",
    knobs={"out_f<=1024": "waves_n=1, wn=4 (BN 128->64, 2x workgroups)"},
    oracle="bench/qk-prefill-post-decode-parity-frontier (graph-GEMM pre-fix) + vendored Tensile",
    correctness="rel_rmse 2.08e-4 + byte-identical greedy vs WMMA reference", route_identity="prefill_graph_gemm fires (default-on within PREFILL_V2 gfx1100)",
    materialization_abi="n/a (prefill GEMM)", isa="WMMA(v_dot)+LDS; +23% VALU residual (deterministic leanness, not searched)",
    local_diagnostic="isolated GEMM host-bound (discarded per SOP)",
    authority_benchmark={"whole_prefill_tok_s": {"512": 3554, "1024": 3468, "2048": 3221, "4096": 2796}, "vs_tensile": "99.5%", "vs_llama": "91-116%"},
    verdict="WON_SHIPPED (graph-GEMM route)", stop_reason="passed; whole-prefill +3-4% transfer",
    artifact_links=["docs/prefill-per-role-transfer-attribution-result-20260623.md"],
    learned_rule="ONE-CONFIG-FITS-ALL WG-starves small-N roles; per-shape tile config (not search) is the bounded fix; in-model GPU-busy is authority (isolated GEMM is host-bound)"))
  entries.append(entry(candidate_id="codegen/lds_cross_lane_v_dot2_expressibility", lane="codegen", primitive_class="codegen_microprimitive",
    knobs={"render": "tinygrad-native fp16 workgroup reduction"}, oracle="owned tile ISA (v_dot2 + ds_bpermute + LDS)",
    correctness="ISA-evidence only (no W==D claim)", route_identity="n/a", materialization_abi="n/a",
    isa="tinygrad emits LDS(ds_load/ds_store) NATIVELY; does NOT emit v_dot2 or ds_bpermute(cross-lane)",
    local_diagnostic="disasm of compiled reduce kernel", authority_benchmark="non-promotion (codegen expressibility)",
    verdict="EXPRESSIBILITY_MAPPED (LDS native; v_dot2+cross-lane are the gaps)", stop_reason="ISA evidence sufficient",
    artifact_links=["bench/qk-native-codegen-experiment/lds_cross_lane_result.json", "docs/machine-code-translation-roadmap-result-20260623.md"],
    learned_rule="native-codegen targets are exactly v_dot2 lowering + cross-lane reduce; LDS staging already native"))
  write_schema()
  with open(LEDGER, "w") as fh:
    for e in entries: fh.write(json.dumps(e) + "\n")
  return entries

if __name__ == "__main__":
  if "--seed" in sys.argv or not LEDGER.exists():
    es = seed()
    print(f"LEDGER seeded: {len(es)} entries -> {LEDGER}")
    by_lane = {}
    for e in es: by_lane.setdefault(e["lane"], []).append(e["verdict"])
    for lane, vs in by_lane.items(): print(f"  {lane}: {len(vs)} ({', '.join(sorted(set(v.split('(')[0].split(':')[0].strip() for v in vs)))[:90]})")
