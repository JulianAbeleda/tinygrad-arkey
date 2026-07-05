#!/usr/bin/env python3
"""PMS-R8: prove the existing Qwen3-8B/gfx1100 routes REGENERATE from the profile descriptor.

Derives every decode/prefill role shape (M/N/K) from the profile's MODEL DIMS (hidden/ffn/heads/kv_heads/head_dim/vocab)
and confirms the derived shapes match the route manifest shape_guards (decode) and the search_profiles role shapes
(prefill) -- i.e. the routes are a FUNCTION of the profile, not hand-edited flags. Also checks the descriptor against
bench/qk-search-spaces/profiles/_schema.json. No GPU.

Run:  PYTHONPATH=. python3 extra/qk/profile_regenerate_check.py
"""
from __future__ import annotations
import json, pathlib
from extra.qk.route_manifest import ROUTES

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROFILES_DIR = ROOT / "bench/qk-search-spaces/profiles"
PROFILE = PROFILES_DIR / "qwen3_8b_q4_k_m_gfx1100.json"
SCHEMA = PROFILES_DIR / "_schema.json"
SEARCH_PROFILES = ROOT / "bench/qk-search-spaces/search_profiles.json"


def derive_decode_shapes(m: dict) -> dict:
  return {
    "attn_qo":     {"K": m["hidden"], "N": m["heads"] * m["head_dim"]},
    "attn_kv":     {"K": m["hidden"], "N": m["kv_heads"] * m["head_dim"]},
    "ffn_gate_up": {"K": m["hidden"], "N": m["ffn"]},
    "ffn_down":    {"K": m["ffn"], "N": m["hidden"]},
    "lm_head":     {"N": m["vocab"]},
  }


def _guard_match(derived: dict, guard: dict) -> bool:
  for k in ("K", "N"):
    if k not in guard: continue
    g = guard[k]
    if isinstance(g, str) and g.startswith(">="):
      if not (k in derived and derived[k] >= int(g[2:])): return False
    elif g == "*":
      continue
    else:
      if derived.get(k) != g: return False
  return True


def main() -> int:
  prof = json.load(open(PROFILE))
  m = prof["model"]
  derived = derive_decode_shapes(m)
  checks, errors = [], []

  # 1. schema: required top-level + model fields present
  schema = json.load(open(SCHEMA))
  for k in schema["required_top_level"]:
    if k not in prof: errors.append(f"profile missing required top-level {k!r}")
  for k in schema["model"]["required"]:
    if k not in m: errors.append(f"profile.model missing {k!r}")
  if prof.get("gpu", {}).get("measured_copy_gbps") != 820:
    errors.append("gpu.measured_copy_gbps must be 820 (scope PMS-R8 pin)")

  # 2. decode routes regenerate: every manifest shape_guard (role-tagged) matches the derived shape
  for rid in prof["regenerates_routes"]:
    if ROUTES[rid]["workload"] != "decode": continue
    for guard in ROUTES[rid]["shape_guards"]:
      role = guard.get("role")
      if role is None or role not in derived: continue
      ok = _guard_match(derived[role], guard)
      checks.append({"route": rid, "role": role, "derived": derived[role], "guard": guard, "match": ok})
      if not ok: errors.append(f"{rid}.{role}: derived {derived[role]} != guard {guard}")

  # 3. prefill role shapes regenerate from model dims (search_profiles declares per-role M/N/K)
  sp = json.load(open(SEARCH_PROFILES))
  pre = sp["profiles"]["qwen3_8b_q4_k_m_gfx1100_prefill"]["roles"]
  ub = prof["role_shape_derivation"]["prefill_ubatch"]
  for role, rmeta in pre.items():
    if role not in derived: continue
    want = {"M": ub, **derived[role]}
    got = rmeta["shape"]
    ok = all(got.get(k) == v for k, v in want.items())
    checks.append({"route": "prefill_pipe_role_selective_generated", "role": role, "derived": want,
                   "guard": got, "match": ok})
    if not ok: errors.append(f"prefill.{role}: derived {want} != search_profiles {got}")

  regen_ok = not errors
  verdict = "PMS_R8_PASS_PROFILE_DRIVEN_SEARCH_READY" if regen_ok else "PMS_R8_BLOCKED_PROFILE_SCHEMA_GAPS"
  result = {
    "scope": "PMS-R8 profile-driven search readiness (routes regenerate from the profile descriptor)",
    "verdict": verdict, "profile": str(PROFILE.relative_to(ROOT)),
    "model_dims": m, "derived_decode_shapes": derived,
    "authority_contexts": prof["authority_contexts"], "threshold_policy": prof["threshold_policy"],
    "gpu": prof["gpu"], "quant_mix": prof["quant_mix"],
    "regeneration_checks": checks, "errors": errors,
    "routes_regenerated": prof["regenerates_routes"],
    "new_target_bootstrap": prof["new_target_bootstrap"],
  }
  OUT = PROFILES_DIR / "qwen3_8b_q4_k_m_gfx1100.regen.json"
  json.dump(result, open(OUT, "w"), indent=2)
  print(verdict, "|", len([c for c in checks if c["match"]]), "/", len(checks), "shape regenerations match")
  for e in errors: print("  ERROR:", e)
  return 0 if regen_ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
