#!/usr/bin/env python3
"""TG-P6: PURE_MACHINE_SEARCH_ONLY diagnostic guard.

Proves whether a run's selected hot-route defaults are all machine-authored/generated (pure search) or whether any
handwritten route is on the default path. It resolves the EFFECTIVE route for each hot family given the environment
(the rollback flags flip a generated default back to its handwritten oracle), maps it to the manifest provenance, and
reports violations with the route id + replacement scope.

Purity contract (docs/pure-machine-search.md; extra/qk/route_manifest.py):
  allowed on a pure default path : machine_authored_generated | tinygrad_scheduler_generated
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

# Each hot route family resolves to an EFFECTIVE route id from the environment. `rollback_active(env)` is True when the
# env selects the handwritten oracle instead of the generated default; `generated_route`/`oracle_route` are the two
# manifest route ids. `pure_default` names the generated route that SHOULD be selected on a pure path.
HOT_FAMILIES = [
  {"family": "decode_q4k_gemv", "generated": "decode_q4k_g3_generated", "oracle": "decode_q4k_owned_warp",
   "rollback_active": lambda e: str(e.get("BUBBLEBEAM_FUTURESIGHT", "1")) == "0"},
  # decode_q6k_coop_shipped rollback DELETED (no backups): generated Q6_K decode is now unconditional.
  {"family": "decode_q6k_gemv", "generated": "decode_q6k_coop_generated", "oracle": "decode_q6k_coop_generated",
   "rollback_active": lambda e: False},
  {"family": "prefill_gemm", "generated": "prefill_pipe_role_selective_generated", "oracle": "prefill_pipe_role_selective_generated",
   "rollback_active": lambda e: False},
  # Q4_K quantized prefill (14B/32B memory-safe default). The direct-packed default is a hand-authored UOp template and
  # the opt-in PREFILL_Q4K_WMMA_FUSED route is raw-ISA WMMA -> both impure until the generated MMQ/WMMA substrate lands.
  {"family": "prefill_q4k", "generated": "prefill_q4k_direct_tile4x4_default", "oracle": "prefill_q4k_direct_tile4x4_default",
   "rollback_active": lambda e: False},
  # attention: 8B long-context decode now defaults to the generated live-split + fused-combine + KV_BOTH route. The
  # only rollback here is to generic generated tinygrad flash decode; the retired owned HIP tile is not selected.
  {"family": "decode_attention", "generated": "decode_flash_live_split_g4_8b_kvboth", "oracle": "decode_attention_generic_flash_generated",
   "rollback_active": lambda e: str(e.get("DECODE_LIVE_SPLIT", "1")) == "0"},
]


def _provenance(rid: str) -> str:
  return str(ROUTES.get(rid, {}).get("provenance", "unknown"))


def _replacement_scope(rid: str) -> str:
  return str(ROUTES.get(rid, {}).get("replacement_scope", "") or ROUTES.get(rid, {}).get("note", ""))


def effective_routes(env: dict[str, Any] | None = None) -> list[dict[str, Any]]:
  """The effective route id + provenance for each hot family under `env` (default os.environ)."""
  e = os.environ if env is None else env
  out = []
  for fam in HOT_FAMILIES:
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
