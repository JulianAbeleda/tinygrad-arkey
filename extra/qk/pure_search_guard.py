#!/usr/bin/env python3
"""TG-P6: PURE_MACHINE_SEARCH_ONLY diagnostic guard.

Proves whether a run's selected hot-route defaults are all machine-authored/generated (pure search) or whether any
handwritten route is on the default path. It resolves the EFFECTIVE route for each hot family given the environment
(the rollback flags flip a generated default back to its handwritten oracle), maps it to the manifest provenance, and
reports violations with the route id + replacement scope.

Purity contract (docs/pure-machine-search.md; extra/qk/route_manifest.py):
  allowed on a pure default path : machine_authored_generated | tinygrad_scheduler_generated
  not strict-pure generated      : compiler_primitive_spec_owned
  forbidden                      : external_handwritten_kernel | hand_authored_uop_template | rollback_oracle

When PURE_MACHINE_SEARCH_ONLY=1 the model raises at init if any hot default is impure (unless the specific rollback
was explicitly requested AND PURE_MACHINE_SEARCH_ALLOW_ROLLBACK=1). This module is pure data+logic (no GPU, no
kernels); model.py calls assert_pure_machine_search() at Transformer init.
"""
from __future__ import annotations

import os
from typing import Any

from extra.qk.route_manifest import ROUTES, FINAL_DEFAULT_PROVENANCE
from extra.qk.pure_kernel_surface_audit import route_surface_row

def _enabled(env: dict[str, Any], key: str) -> bool:
  return str(env.get(key, "0")).strip().lower() not in ("0", "false", "off", "no", "")


def _env_flag(env: dict[str, Any], key: str, default: int) -> bool:
  """Mirror tinygrad.getenv(key, default): an UNSET key resolves to the runtime DEFAULT value, not 0. This is the ONLY
  place the guard encodes a runtime getenv default; test/unit/test_pure_search_guard_boundary.py pins these against the
  real decode_routes.py getenv defaults so a flipped default (e.g. BUBBLEBEAM_FUTURESIGHT -> 0) fails the suite instead
  of silently diverging from the shipped route."""
  v = env.get(key)
  if v is None: return bool(default)
  return str(v).strip().lower() not in ("0", "false", "off", "no", "")


def _decode_q4k_rolled_back(e: dict[str, Any]) -> bool:
  # decode_routes.py q4k_primitive_linear_call: generated G3 fires when getenv("BUBBLEBEAM_FUTURESIGHT", 1) is truthy
  # AND getenv("Q4K_GEMV_SCHEDULER") is unset. Otherwise decode falls to the ORDINARY tinygrad graph (pure -- the hand
  # owned-warp rollback kernels were deleted 2026-07-06). DECODE_Q4K_INKERNEL_COMBINE_KV / DECODE_Q4K_SPLIT_K_KV only
  # pick GENERATED G3 sub-variants (still pure), so they are NOT rollbacks and do not appear here.
  return not (_env_flag(e, "BUBBLEBEAM_FUTURESIGHT", 1) and not _enabled(e, "Q4K_GEMV_SCHEDULER"))


def _decode_attention_rolled_back(e: dict[str, Any]) -> bool:
  # decode_routes.py flash_decode_attention_route: the generated live-split route is on by getenv("DECODE_LIVE_SPLIT",
  # 1). DECODE_LIVE_SPLIT=0 de-selects it; runtime then fails loud (no handwritten flash fallback remains).
  return not _env_flag(e, "DECODE_LIVE_SPLIT", 1)


# Each hot route family resolves to an EFFECTIVE route id from the environment. `rollback_active(env)` is True when the
# env leaves the generated default; `generated`/`oracle` are the manifest route ids for the two arms. The decode
# rollback predicates read the REAL decode_routes.py env gates (with real getenv defaults) so the guard's model tracks
# the actual selector rather than a hardcoded constant; the boundary test drives the real dispatcher to prove it. The
# handwritten decode rollback kernels were deleted (no backups), so the decode "oracle" arm is the family's own
# generated route (its canonical manifest route); the boundary test is what catches an impure/de-selected decode
# default, since a rollback here lands on the pure ordinary graph or fails loud rather than a hand kernel.
HOT_FAMILIES = [
  {"family": "decode_q4k_gemv", "generated": "decode_q4k_g3_generated", "oracle": "decode_q4k_g3_generated",
   "rollback_active": _decode_q4k_rolled_back},
  # Q6_K shipped hand-kernel rollback was deleted (no backups): generated Q6_K decode is unconditional -- no env
  # de-selects it in decode_routes.py q6k_primitive_linear_call.
  {"family": "decode_q6k_gemv", "generated": "decode_q6k_coop_generated", "oracle": "decode_q6k_coop_generated",
   "rollback_active": lambda e: False},
  {"family": "prefill_gemm", "generated": "prefill_v2_scheduler_matmul_default", "oracle": "prefill_pipe_role_selective_generated",
   "effective": "prefill_gemm"},
  # Q4_K quantized prefill (14B/32B memory-safe default). The direct-packed default is descriptor-owned; the opt-in
  # PREFILL_Q4K_WMMA_FUSED route remains raw-ISA WMMA and is not selected here.
  {"family": "prefill_q4k", "generated": "prefill_q4k_direct_tile4x4_default", "oracle": "prefill_q4k_direct_tile4x4_default",
   "rollback_active": lambda e: False},
  {"family": "decode_attention", "generated": "decode_flash_live_split_g4_8b_kvboth", "oracle": "decode_flash_live_split_g4_8b_kvboth",
   "rollback_active": _decode_attention_rolled_back},
]


def _prefill_gemm_effective(env: dict[str, Any]) -> tuple[str, bool]:
  if not _enabled(env, "PREFILL_GRAPH_GEMM"):
    return "prefill_v2_scheduler_matmul_default", False
  if (_enabled(env, "PREFILL_WMMA_PIPE_PRIMITIVE") and _enabled(env, "PREFILL_WMMA_LDS_PRIMITIVE") and
      _enabled(env, "PREFILL_DBUF")):
    return "prefill_wmma_pipe_lds_dbuf_primitive_generated", False
  if _enabled(env, "PREFILL_WMMA_PIPE_PRIMITIVE"):
    return "prefill_wmma_pipe_primitive_generated", False
  if _enabled(env, "PREFILL_WMMA_LDS_PRIMITIVE") and _enabled(env, "PREFILL_DBUF"):
    return "prefill_wmma_lds_dbuf_primitive_mixed", False
  return "prefill_pipe_role_selective_generated", True


def _provenance(rid: str) -> str:
  return str(ROUTES.get(rid, {}).get("provenance", "unknown"))


def _replacement_scope(rid: str) -> str:
  return str(ROUTES.get(rid, {}).get("replacement_scope", "") or ROUTES.get(rid, {}).get("note", ""))


def effective_routes(env: dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """The effective route id + provenance for each hot family under `env` (default os.environ)."""
  e = os.environ if env is None else env
  out = []
  for fam in HOT_FAMILIES:
    if fam.get("effective") == "prefill_gemm":
      rid, rolled_back = _prefill_gemm_effective(e)
    else:
      rolled_back = fam["rollback_active"](e)
      rid = fam["oracle"] if rolled_back else fam["generated"]
    prov = _provenance(rid)
    surface = route_surface_row(rid)
    out.append({"family": fam["family"], "effective_route": rid, "provenance": prov,
                "surface_class": surface["surface_class"], "strict_pure": surface["strict_pure"],
                "manifest_pure": prov in FINAL_DEFAULT_PROVENANCE,
                "rolled_back_to_oracle": rolled_back, "pure": surface["strict_pure"]})
  return out


def pure_search_violations(env: dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """Hot families whose effective route is NOT machine-authored/generated (impure on the default path)."""
  viols = []
  for r in effective_routes(env):
    if not r["pure"]:
      viols.append({"family": r["family"], "route_id": r["effective_route"], "provenance": r["provenance"],
                    "surface_class": r.get("surface_class", "unknown"),
                    "rolled_back_to_oracle": r["rolled_back_to_oracle"],
                    "replacement_scope": _replacement_scope(r["effective_route"]),
                    "reason": ("explicit rollback to handwritten oracle" if r["rolled_back_to_oracle"]
                               else f"selected surface is not strict pure machine search ({r.get('surface_class', 'unknown')})")})
  return viols


def assert_pure_machine_search(env: dict[str, Any] | None = None) -> None:
  """Enforce PURE_MACHINE_SEARCH_ONLY: raise if any hot default is impure. Called from model.py Transformer init.
  A rollback is tolerated only when PURE_MACHINE_SEARCH_ALLOW_ROLLBACK=1 explicitly requests it."""
  e = os.environ if env is None else env
  if str(e.get("PURE_MACHINE_SEARCH_ONLY", "0")) != "1":
    return
  allow_rollback = str(e.get("PURE_MACHINE_SEARCH_ALLOW_ROLLBACK", "0")) == "1"
  viols = pure_search_violations(e)
  if allow_rollback:
    viols = [v for v in viols if not v["rolled_back_to_oracle"]]
  # always export the route report for the run
  report = {"pure_machine_search_only": True, "effective_routes": effective_routes(e), "violations": viols}
  print("PURE_MACHINE_SEARCH_ONLY route report: " + str({r["family"]: (r["effective_route"], "pure" if r["pure"] else "IMPURE")
                                                          for r in report["effective_routes"]}))
  if viols:
    lines = [f"  - {v['family']}: selected {v['route_id']} (provenance={v['provenance']}, "
             f"surface={v['surface_class']}) is not machine-authored/"
             f"generated; {v['reason']}. Replacement scope: {v['replacement_scope'][:120]}" for v in viols]
    raise RuntimeError("PURE_MACHINE_SEARCH_ONLY=1 but the default path is not pure:\n" + "\n".join(lines))


if __name__ == "__main__":
  import json
  print(json.dumps({"effective_routes": effective_routes(), "violations": pure_search_violations()}, indent=2))
