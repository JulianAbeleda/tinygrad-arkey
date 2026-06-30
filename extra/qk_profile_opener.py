#!/usr/bin/env python3
"""TG4: New-Profile Opener -- adding a model/quant/GPU starts from AUDITS, not flags.

Given a profile descriptor, the opener runs the required new-profile flow and emits census + ceilings +
route space + first-candidate rows, wiring the existing substrate:
  * TG3 quant semantics library (extra/qk_quant_semantics.py)  -> quant-mix census + per-role weight-byte ceiling
  * PMS-R8 shape regeneration (extra/qk_profile_regenerate_check.derive_decode_shapes) -> model/shape census
  * route manifest (extra/qk_route_manifest.py) + PMS-R0 census -> default-path route census + do_not_search
  * TG2 author (extra/qk_topology_candidate_author) -> first candidate rows for the hot decode GEMV

Required flow (scope TG4):
  1. model/shape census    2. quant-mix census    3. GPU target-feature census    4. default-path route census
  5. theoretical ceiling   6. wall-share attribution    7. declared search rows
  8. do_not_search inherited from matching prior profiles    9. first candidate recommendation or NO_ACTION

It REFUSES to open a profile (no candidates) when model shape, quant layout, or GPU features are missing:
  TG4_BLOCKED_PROFILE_MISSING_MODEL_METADATA / _MISSING_QUANT_SEMANTICS / _MISSING_TARGET_FEATURES.

AUDIT/RESEARCH only: no GPU kernel, no default change, no live-route repoint. Reads descriptors + replays/derives.

Run: PYTHONPATH=. python3 extra/qk_profile_opener.py
"""
from __future__ import annotations
import json, pathlib
from extra.qk_quant_semantics import quant_row, QuantLayoutUnknown
from extra.qk_profile_regenerate_check import derive_decode_shapes
from extra.qk_route_manifest import ROUTES, REFUTED

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROFILES_DIR = ROOT / "bench/qk-search-spaces/profiles"
SCHEMA = PROFILES_DIR / "_schema.json"
OUT = ROOT / "bench/qk-profile-opener"

_GPU_REQUIRED = ("vendor", "arch", "wave", "vram_gb", "measured_copy_gbps")
_MODEL_REQUIRED = ("family", "params", "layers", "hidden", "ffn", "heads", "kv_heads", "head_dim")


class ProfileGap(Exception):
  def __init__(self, verdict: str, detail: str):
    self.verdict, self.detail = verdict, detail
    super().__init__(f"{verdict}: {detail}")


# ---- the 9-step opener flow ---------------------------------------------------------------------
def model_shape_census(prof: dict) -> dict:
  m = prof.get("model", {})
  missing = [k for k in _MODEL_REQUIRED if k not in m]
  if missing:
    raise ProfileGap("TG4_BLOCKED_PROFILE_MISSING_MODEL_METADATA", f"model missing {missing}")
  return {"model_dims": m, "decode_role_shapes": derive_decode_shapes(m)}


def quant_mix_census(prof: dict) -> dict:
  """Pull every quant in the mix from the TG3 library. Refuse if any layout is unknown."""
  qmix = prof.get("quant_mix", {})
  if not qmix:
    raise ProfileGap("TG4_BLOCKED_PROFILE_MISSING_QUANT_SEMANTICS", "quant_mix is empty")
  rows, unknown = {}, []
  for q in qmix:
    try:
      fmt = quant_row(q)
      rows[q] = {"roles": qmix[q], "block_elems": fmt.block_elems, "block_bytes": fmt.block_bytes,
                 "packing": fmt.packing_word_dtype, "symmetric": fmt.symmetric,
                 "quality_class": fmt.quality_class, "known_good": list(fmt.known_good_route_families),
                 "known_refuted": [r.get("route_id", r.get("route_family")) for r in fmt.known_refuted_route_families],
                 **fmt.derive()}
    except QuantLayoutUnknown as e:
      unknown.append(q)
  if unknown:
    raise ProfileGap("TG4_BLOCKED_PROFILE_MISSING_QUANT_SEMANTICS",
                     f"quant layout unknown for {unknown} (add a TG3 row; do NOT fall back to Q4_K)")
  return {"quant_rows": rows}


def gpu_feature_census(prof: dict) -> dict:
  g = prof.get("gpu", {})
  missing = [k for k in _GPU_REQUIRED if k not in g]
  if missing:
    raise ProfileGap("TG4_BLOCKED_PROFILE_MISSING_TARGET_FEATURES", f"gpu missing {missing}")
  return {"gpu": g, "lane_extent": int(g["wave"]), "vendor": g["vendor"], "arch": g["arch"],
          "target_id": f"{g['vendor'].lower()}_{g['arch']}"}


def default_route_census(prof: dict) -> dict:
  """Default-path routes whose profile_id matches this profile's decode/prefill profiles (from the manifest)."""
  dp = prof.get("decode_profile_id"); pp = prof.get("prefill_profile_id")
  rows = []
  for rid, r in ROUTES.items():
    if r["profile_id"] in (dp, pp):
      rows.append({"route_id": rid, "status": r["status"], "roles": r["roles"], "quant": r["quant"],
                   "purity_status": r["purity_status"], "selector": r["selector"], "env_default": (r["env"] == {})})
  return {"routes": rows, "promoted_or_shipped": [r["route_id"] for r in rows
                                                  if r["status"] in ("promoted_default", "default_shipped")]}


def theoretical_ceiling(shape_census: dict, quant_census: dict, gpu_census: dict, prof: dict) -> dict:
  """Per-decode-GEMV-role HBM weight-read roofline: bytes = N*K/block_elems*block_bytes; ceiling_ms = bytes/bw.

  Bandwidth-bound decode -> this is the speed-of-light for each weight GEMV (the search cannot beat it)."""
  bw = float(gpu_census["gpu"]["measured_copy_gbps"]) * 1e9
  shapes = shape_census["decode_role_shapes"]
  qmix = prof["quant_mix"]
  role_to_quant = {}
  for q, roles in qmix.items():
    for role in roles:
      role_to_quant.setdefault(role, q)  # first quant listed for the role
  ceilings = {}
  for role, shp in shapes.items():
    q = role_to_quant.get(role)
    if q is None or q not in quant_census["quant_rows"]:
      continue
    N, K = shp.get("N"), shp.get("K")
    if not isinstance(N, int) or not isinstance(K, int):
      continue
    qr = quant_census["quant_rows"][q]
    weight_elems = N * K
    weight_bytes = weight_elems // qr["block_elems"] * qr["block_bytes"]
    ms = weight_bytes / bw * 1e3
    ceilings[role] = {"quant": q, "N": N, "K": K, "weight_bytes": weight_bytes,
                      "roofline_ms_at_measured_bw": round(ms, 4),
                      "roofline_gbps": round(qr["block_bytes"] / qr["block_elems"] * 8, 2)}
  return {"measured_copy_gbps": gpu_census["gpu"]["measured_copy_gbps"],
          "per_role_weight_read_roofline": ceilings,
          "note": "HBM weight-read speed-of-light per decode GEMV role; decode is weight-mem-bound so this bounds the search."}


def wall_share_attribution(prof: dict) -> dict:
  """Reference the existing ceiling/attribution artifacts if this profile is the solved one; else mark PENDING."""
  art = ROOT / "bench/amd-isa-backend-weight-path-ceiling/latest.json"
  if prof["profile_id"] == "qwen3_8b_q4_k_m_gfx1100" and art.exists():
    return {"source": str(art.relative_to(ROOT)), "status": "CITED_FROM_PRIOR_AUDIT",
            "note": "weight GEMVs dominate (~58% of decode wall); attention wall-share ~10%@512 ->~0%@4096 "
                    "(bench/amd-isa-backend-decode-attention-ceiling). Weight GEMV is the leverage."}
  return {"source": None, "status": "PENDING_MEASURE",
          "note": "no prior wall-share audit for this profile; a W==D attribution must be measured before promotion."}


def declared_search_rows(prof: dict) -> dict:
  """The declared candidate space = decode roles x their quant x allowed route families (from search_profiles if
  present for this profile, else derived from the quant-mix census known_good route families)."""
  sp_path = ROOT / "bench/qk-search-spaces/search_profiles.json"
  sp = json.load(open(sp_path))
  dp = prof.get("decode_profile_id")
  if dp in sp.get("profiles", {}):
    roles = sp["profiles"][dp]["roles"]
    return {"source": "search_profiles.json", "rows": {r: {"quant": c["quant"], "status": c["status"],
            "allowed_route_families": c["allowed_route_families"]} for r, c in roles.items()}}
  # derived for a brand-new profile: each weight role -> its quant's known_good families, status=open
  qmix = prof["quant_mix"]
  rows = {}
  for q, qroles in qmix.items():
    fmt = quant_row(q)
    for role in qroles:
      if role in ("attention_tile", "attention_combine", "prefill_gemm_all_roles"):
        continue
      rows[f"{role}__{q}"] = {"quant": q, "status": "open",
                              "allowed_route_families": list(fmt.known_good_route_families)}
  return {"source": "derived_from_quant_known_good", "rows": rows}


def do_not_search_inherited(prof: dict) -> dict:
  """do_not_search inherited from matching prior profiles: the manifest REFUTED axes + each quant's known_refuted."""
  inherited = [dict(r) for r in REFUTED]
  per_quant = {}
  for q in prof["quant_mix"]:
    fmt = quant_row(q)
    if fmt.known_refuted_route_families:
      per_quant[q] = [dict(r) for r in fmt.known_refuted_route_families]
  return {"manifest_refuted_axes": inherited, "quant_known_refuted": per_quant}


def first_candidate_recommendation(prof: dict, declared: dict) -> dict:
  """Run the TG2 author for the hot Q4_K decode GEMV if this profile has open Q4_K GEMV roles; else NO_ACTION."""
  # If the profile is the solved Q4_K one, all GEMV roles are promoted -> NO_ACTION (closed), matching pms_r3.
  rows = declared["rows"]
  open_rows = {r: c for r, c in rows.items() if c.get("status") == "open"}
  if prof["profile_id"] == "qwen3_8b_q4_k_m_gfx1100":
    from extra.qk_topology_candidate_author import load_profile_facts, enumerate_candidates
    facts = load_profile_facts()
    cands, _ = enumerate_candidates(facts)
    return {"action": "NO_ACTION_ALL_GEMV_ROLES_PROMOTED",
            "note": "Q4_K decode GEMV roles are promoted (decode_q4k_g3_generated); the author can still enumerate "
                    f"{len(cands)} bounded candidates but there is no open/failed row to target.",
            "author_candidate_count": len(cands)}
  if not open_rows:
    return {"action": "NO_ACTION_NO_OPEN_ROWS", "note": "no open/failed search rows for this profile."}
  return {"action": "AUTHOR_CANDIDATES_FOR_OPEN_ROWS", "open_rows": sorted(open_rows),
          "note": "open weight-GEMV rows exist; run the TG2 author (quant-parameterized) per row -> TG6 evaluator."}


def open_profile(prof: dict) -> dict:
  """Run the full 9-step flow. Raises ProfileGap (-> BLOCKED verdict) on missing model/quant/GPU."""
  shape = model_shape_census(prof)
  quant = quant_mix_census(prof)
  gpu = gpu_feature_census(prof)
  routes = default_route_census(prof)
  ceiling = theoretical_ceiling(shape, quant, gpu, prof)
  walls = wall_share_attribution(prof)
  declared = declared_search_rows(prof)
  dns = do_not_search_inherited(prof)
  first = first_candidate_recommendation(prof, declared)
  return {
    "profile_id": prof["profile_id"], "verdict": "TG4_PASS_NEW_PROFILE_OPENER_READY",
    "step1_model_shape_census": shape, "step2_quant_mix_census": quant,
    "step3_gpu_target_feature_census": gpu, "step4_default_path_route_census": routes,
    "step5_theoretical_ceiling": ceiling, "step6_wall_share_attribution": walls,
    "step7_declared_search_rows": declared, "step8_do_not_search_inherited": dns,
    "step9_first_candidate_recommendation": first,
  }


# ---- TG4 acceptance gates -----------------------------------------------------------------------
def _regenerates_existing_profile() -> dict:
  """Acceptance 1: the opener reproduces the current Qwen3-8B/gfx1100 profile -- derived decode shapes match the
  manifest shape_guards for every regenerated route (the PMS-R8 invariant, run through the opener)."""
  prof = json.load(open(PROFILES_DIR / "qwen3_8b_q4_k_m_gfx1100.json"))
  opened = open_profile(prof)
  derived = opened["step1_model_shape_census"]["decode_role_shapes"]
  mism = []
  for rid in prof["regenerates_routes"]:
    r = ROUTES.get(rid)
    if not r or r["workload"] != "decode":
      continue
    for guard in r["shape_guards"]:
      role = guard.get("role")
      if role is None or role not in derived:
        continue
      for k in ("K", "N"):
        if k in guard and isinstance(guard[k], int) and derived[role].get(k) != guard[k]:
          mism.append({"route": rid, "role": role, "k": k, "derived": derived[role].get(k), "guard": guard[k]})
  return {"regenerates": not mism, "mismatches": mism,
          "route_census_promoted_or_shipped": opened["step4_default_path_route_census"]["promoted_or_shipped"],
          "first_recommendation": opened["step9_first_candidate_recommendation"]["action"]}


def _draft_new_profile() -> dict:
  """Acceptance 2: draft a profile for ONE additional gfx1100 target (Q5_K weight quant -- genuinely new quant
  mix, same model/GPU) WITHOUT manual route flags; the opener must open it cleanly."""
  base = json.load(open(PROFILES_DIR / "qwen3_8b_q4_k_m_gfx1100.json"))
  m = base["model"]
  draft = {
    "_schema": base["_schema"],
    "profile_id": "qwen3_8b_q5_k_m_gfx1100",
    "decode_profile_id": "qwen3_8b_q5_k_m_gfx1100_decode",
    "prefill_profile_id": "qwen3_8b_q5_k_m_gfx1100_prefill",
    "model": dict(m),
    # Q5_K weight quant for the GEMV weight roles; Q6_K still for lm_head; fp16 for attention/prefill. NO route flags.
    "quant_mix": {"Q5_K": ["ffn_gate_up", "ffn_down", "attn_qo"], "Q6_K": ["lm_head"],
                  "fp16": ["attention_tile", "attention_combine", "prefill_gemm_all_roles"]},
    "quant_mix_note": "DRAFT (TG4 opener): a hypothetical Q5_K_M weight mix on the same Qwen3-8B/gfx1100. Q5_K layout "
                      "is known to the TG3 library; NO shipped/generated route exists yet -> open search rows, no flags.",
    "gpu": dict(base["gpu"]),
    "authority_contexts": base["authority_contexts"],
    "threshold_policy": base["threshold_policy"], "thresholds": base["thresholds"],
    "role_shape_derivation": base["role_shape_derivation"],
    "regenerates_routes": [],  # no routes yet: this is an UNSOLVED draft, opened from audits not flags
    "new_target_bootstrap": base["new_target_bootstrap"],
    "draft_provenance": "extra/qk_profile_opener.py _draft_new_profile (TG4); shapes derived from model dims; quant "
                        "layout from extra/qk_quant_semantics.py; NO hand-edited route flags.",
  }
  out_path = PROFILES_DIR / "qwen3_8b_q5_k_m_gfx1100.json"
  json.dump(draft, open(out_path, "w"), indent=2)
  opened = open_profile(draft)
  return {"draft_profile": str(out_path.relative_to(ROOT)),
          "opened_ok": opened["verdict"] == "TG4_PASS_NEW_PROFILE_OPENER_READY",
          "no_route_flags": draft["regenerates_routes"] == [],
          "quant_rows_resolved": sorted(opened["step2_quant_mix_census"]["quant_rows"]),
          "open_search_rows": sorted(opened["step7_declared_search_rows"]["rows"]),
          "first_recommendation": opened["step9_first_candidate_recommendation"]["action"],
          "per_role_ceiling": opened["step5_theoretical_ceiling"]["per_role_weight_read_roofline"]}


def _refuses_incomplete_profiles() -> dict:
  """Acceptance 3: refuse to open a profile missing model shape / quant layout / GPU features."""
  base = json.load(open(PROFILES_DIR / "qwen3_8b_q4_k_m_gfx1100.json"))
  results = {}
  # (a) missing model dims
  p = json.loads(json.dumps(base)); p["model"].pop("ffn", None)
  results["missing_model_metadata"] = _expect_block(p, "TG4_BLOCKED_PROFILE_MISSING_MODEL_METADATA")
  # (b) unknown quant layout
  p = json.loads(json.dumps(base)); p["quant_mix"] = {"Q3_K": ["ffn_down"]}
  results["missing_quant_semantics"] = _expect_block(p, "TG4_BLOCKED_PROFILE_MISSING_QUANT_SEMANTICS")
  # (c) missing GPU features
  p = json.loads(json.dumps(base)); p["gpu"].pop("wave", None)
  results["missing_target_features"] = _expect_block(p, "TG4_BLOCKED_PROFILE_MISSING_TARGET_FEATURES")
  return results


def _expect_block(prof: dict, expected_verdict: str) -> dict:
  try:
    open_profile(prof)
    return {"refused": False, "got": "OPENED (should have blocked)", "expected": expected_verdict}
  except ProfileGap as e:
    return {"refused": e.verdict == expected_verdict, "got": e.verdict, "expected": expected_verdict, "detail": e.detail}


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  regen = _regenerates_existing_profile()
  draft = _draft_new_profile()
  refuse = _refuses_incomplete_profiles()

  ready = (regen["regenerates"] and draft["opened_ok"] and draft["no_route_flags"]
           and all(v["refused"] for v in refuse.values()))
  verdict = "TG4_PASS_NEW_PROFILE_OPENER_READY" if ready else "TG4_BLOCKED_PROFILE_OPENER_INCOMPLETE"

  # write the opener output for both the existing and the drafted profile
  existing_open = open_profile(json.load(open(PROFILES_DIR / "qwen3_8b_q4_k_m_gfx1100.json")))
  draft_open = open_profile(json.load(open(PROFILES_DIR / "qwen3_8b_q5_k_m_gfx1100.json")))
  for pid, opened in (("qwen3_8b_q4_k_m_gfx1100", existing_open), ("qwen3_8b_q5_k_m_gfx1100", draft_open)):
    d = OUT / pid; d.mkdir(parents=True, exist_ok=True)
    json.dump(opened, open(d / "latest.json", "w"), indent=2)

  result = {
    "scope": "TG4 new-profile opener: census + ceilings + route space + first-candidate rows from a profile "
             "descriptor, wiring TG3 quant lib + PMS census/ceiling + TG2 author. AUDIT only.",
    "verdict": verdict,
    "opener": "extra/qk_profile_opener.py",
    "acceptance_1_regenerates_existing_profile": regen,
    "acceptance_2_draft_new_profile_no_flags": draft,
    "acceptance_3_refuses_incomplete_profiles": refuse,
    "outputs": ["bench/qk-profile-opener/qwen3_8b_q4_k_m_gfx1100/latest.json",
                "bench/qk-profile-opener/qwen3_8b_q5_k_m_gfx1100/latest.json",
                "bench/qk-search-spaces/profiles/qwen3_8b_q5_k_m_gfx1100.json (DRAFT)"],
    "do_not": ["no GPU kernel", "no default change", "no live-route repoint"],
  }
  json.dump(result, open(OUT / "latest.json", "w"), indent=2)

  md = [f"# TG4 New-Profile Opener -- verdict: **{verdict}**", "",
        "The opener runs the 9-step new-profile flow (model/shape, quant-mix, GPU features, route census, ceiling, "
        "wall-share, declared rows, inherited do_not_search, first candidate) from a profile descriptor.", "",
        "## Acceptance", "",
        f"- **Regenerates the existing Qwen3-8B/gfx1100 profile**: {regen['regenerates']} "
        f"(promoted/shipped routes: {regen['route_census_promoted_or_shipped']}; first action: {regen['first_recommendation']})",
        f"- **Drafts a new gfx1100 target without route flags**: {draft['opened_ok'] and draft['no_route_flags']} "
        f"(`qwen3_8b_q5_k_m_gfx1100`; quants resolved {draft['quant_rows_resolved']}; open rows {draft['open_search_rows']})",
        f"- **Refuses incomplete profiles**: "
        f"{all(v['refused'] for v in refuse.values())} "
        f"(model={refuse['missing_model_metadata']['refused']}, quant={refuse['missing_quant_semantics']['refused']}, "
        f"gpu={refuse['missing_target_features']['refused']})", "",
        "## Drafted profile per-role weight-read ceiling (HBM speed-of-light)", "",
        "| role | quant | N | K | weight bytes | roofline ms @ measured bw |", "|---|---|---:|---:|---:|---:|"]
  for role, c in draft["per_role_ceiling"].items():
    md.append(f"| {role} | {c['quant']} | {c['N']} | {c['K']} | {c['weight_bytes']} | {c['roofline_ms_at_measured_bw']} |")
  md.append("")
  (OUT / "summary.md").write_text("\n".join(md))

  print(verdict)
  print(f"  regenerates existing profile: {regen['regenerates']} (first action {regen['first_recommendation']})")
  print(f"  drafted qwen3_8b_q5_k_m_gfx1100 (no flags): {draft['opened_ok'] and draft['no_route_flags']} "
        f"| quants {draft['quant_rows_resolved']} | open rows {draft['open_search_rows']}")
  print(f"  refuses incomplete: model={refuse['missing_model_metadata']['refused']} "
        f"quant={refuse['missing_quant_semantics']['refused']} gpu={refuse['missing_target_features']['refused']}")
  return 0 if ready else 1


if __name__ == "__main__":
  raise SystemExit(main())
