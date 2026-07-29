"""Route manifest -- the single declarative source of truth for which decode/prefill routes exist on this fork, what
selects each one, what it rolls back to, and its current disposition. PMS-R1 of
docs/pure-machine-search-remaining-hot-kernels-scope-20260630.md (supersedes the Phase-1 draft of
docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md).

For why machine-searched routes are frozen into static runtime artifacts, see
README.md#why-this-is-machine-search-even-though-the-runtime-is-static.

This module is DATA + tiny helpers. It changes NO defaults and runs NO kernels; gates import it instead of copying
ad-hoc env maps. For each route, `env` is what you SET to force that route onto the active path; an empty `env` ({})
means the route is ALREADY the shipped default (no flag needed). `rollback` is the exact env to leave it.

CURRENT-STATE PIN (verified 2026-07-03 against tinygrad/llm/decode_routes.py + tinygrad/llm/decode_kernels.py +
tinygrad/llm/flash_decode_attention.py + tinygrad/llm/prefill_graph_gemm.py; legacy extra modules are research oracles):

  * Decode Q4_K GEMV default = the GENERATED G3 LaneMap route. decode_routes.py:q4k_primitive_linear_call reads
    `getenv("BUBBLEBEAM_FUTURESIGHT", 1)` (DEFAULT-ON, flipped in commit 81370ae38). For eligible Q4_K shapes
    the G3 route fires FIRST and short-circuits before the owned-warp guards. So `decode_q4k_g3_generated` is the
    promoted default; the owned warp kernel is the rollback/reference one flag away (BUBBLEBEAM_FUTURESIGHT=0).
  * An earlier draft of this file had G3/owned default-status INVERTED (it predated the default flip). This version
    pins the real state: G3 = default, owned warp = rollback.

Token-identity / speed-equivalence proof: bench/amd-isa-backend-g3-weight-promotion/latest.json
(AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT; lag -0.13..+0.41% across ctx 512-4096; token_match + route_clean all ctx).
"""
from __future__ import annotations
import hashlib, json, pathlib
from collections.abc import Mapping
from types import MappingProxyType

from tinygrad.llm.prefill_candidate_runtime import canonical_candidate_set_identity as _production_candidate_set_identity

PROFILE_DECODE = "qwen3_8b_q4_k_m_gfx1100_decode"
PROFILE_DECODE_LARGE = "qwen3_14b_32b_q4_k_m_gfx1100_decode"
PROFILE_PREFILL = "qwen3_8b_q4_k_m_gfx1100_prefill"

# status vocabulary:
#   promoted_default        -> generated/search-selected route that is now the shipped default
#   default_shipped         -> shipped default whose writer is owned/hand asm (not yet replaced by a generated route)
#   rollback_reference      -> kept one flag away as the rollback/oracle for a promoted route; NOT on the default path
#   superseded_rollback     -> previously promoted, kept as the A/B rollback target for a newer default
#   refuted                 -> built, token-correct, route-bound, but speed-refuted; default-off, do not re-search as-is
#   correct_not_fast        -> generated/correct/route-bound but below promotion speed; infrastructure/research, not shipped
#
# purity_status vocabulary (docs/pure-machine-search.md definitions):
#   search_generated_promoted | owned_reference | owned_default | search_selected_specialized_route | refuted | research
#
# provenance vocabulary (strict default-purity audit):
#   machine_authored_generated  -> emitted from profile/grammar/search-owned lowering; allowed as final default
#   tinygrad_scheduler_generated -> ordinary tinygrad graph lowering; allowed as final default
#   compiler_primitive_spec_owned -> compiler/search spec owns route lifecycle; reusable backend atom may emit ASM
#   hand_authored_uop_template  -> Python UOp custom_kernel body written by humans; transitional default only
#   external_handwritten_kernel -> HIP/ASM/C++/precompiled binary or explicit instruction emitter; not final default
#   rollback_oracle            -> handwritten/specialized route retained only as rollback/reference
ROUTE_PROVENANCE = (
  "machine_authored_generated", "tinygrad_scheduler_generated", "hand_authored_uop_template",
  "compiler_primitive_spec_owned", "external_handwritten_kernel", "rollback_oracle",
)
FINAL_DEFAULT_PROVENANCE = {"machine_authored_generated", "tinygrad_scheduler_generated"}
TRANSITIONAL_DEFAULT_PROVENANCE = {"hand_authored_uop_template"}
FORBIDDEN_DEFAULT_PROVENANCE = {"external_handwritten_kernel", "rollback_oracle"}

# purity_status is a DERIVED human-facing label, not an independent axis. It is a pure function of (status, provenance)
# so it can never silently drift from them (F4: it used to be a hand-maintained third vocabulary that disagreed with
# the route's real status/provenance). derive_purity_status is the single source; validate_manifest() enforces that
# every stored purity_status equals the derived value, and the census sources its purity_status from here.
PURITY_STATUS_VOCAB = ("search_generated_promoted", "refuted", "research")

def derive_purity_status(status: str, provenance: str) -> str:
  if status in ("promoted_default", "default_shipped") and provenance in FINAL_DEFAULT_PROVENANCE:
    return "search_generated_promoted"
  if status == "refuted":
    return "refuted"
  return "research"

def _load_routes() -> dict:
  """Load the ROUTES data table from the sibling JSON file.

  The table itself (workload/profile/shape_guards/env/rollback/etc. per route) is pure data -- no computed
  values, no tuples, nothing but dicts/lists/strings/bools/None -- so it lives in route_manifest.json to keep
  it out of sz.py's .py/.js LOC budget. PROFILE_DECODE/PROFILE_DECODE_LARGE/PROFILE_PREFILL are written into
  the JSON as their plain string VALUES (the JSON loader has no access to these Python names).
  """
  return json.loads((pathlib.Path(__file__).parent / "route_manifest.json").read_text())

ROUTES = _load_routes()

# Closed / refuted axes -- do not re-search without a NEW premise (PMS-R3 do_not_search carries these forward).
REFUTED = [
  {"axis": "q6k_direct_half_warp_route", "disposition": "refuted: W==D regression -4.77..-6.06% (median -5.44%)",
   "citation": "bench/amd-isa-backend-q6k-direct-speed/latest.json", "route_id": "decode_q6k_direct_refuted"},
  {"axis": "q4k_offline_layout_reshuffle", "disposition": "deprioritized: G3 matches owned, no layout gap to recover",
   "citation": "bench/amd-isa-backend-g3-weight-promotion/search_space_update.json"},
  {"axis": "prefill_q4k_simple_uop_cooperative_lane_tile", "domain": "prefill",
   "disposition": "refuted on 14B ffn_gate_up: lane_partials 0.99 GB/s; direct_warp sweep best 1.29 GB/s vs current direct-packed floor ~2.11 GB/s",
   "citation": "docs/prefill-lessons-ledger.md", "route_id": "prefill_q4k_generated_tile_research"},
  {"axis": "prefill_q4k_direct_group_reduce_current_uop", "domain": "prefill",
   "disposition": "refuted for the manual direct-output recurrence: GROUP:0:10 looked fast but is numerically wrong on real 14B ffn_gate (rel_rmse ~1.26); use PREFILL_Q4K_REDUCE_OUT=1 for correct-but-not-fast grouped semantics",
   "citation": "docs/prefill-lessons-ledger.md", "route_id": "prefill_q4k_direct_tile4x4_default"},
  {"axis": "attention_combine_fused_lifecycle", "domain": "attention", "disposition": "exhausted/low-leverage (combine overlaps in-graph; fused is codegen-walled)",
   "citation": "docs/decode-two-kernel-problem-audit-result-20260625.md"},
  {"axis": "g5_block_tile_as_default", "compatibility_aliases": ("g5_block_tile_8b_as_default",), "domain": "attention", "disposition": "correct_not_fast: token-identical + route-bound but 87.6% of owned @ctx512 / 95.6% @ctx4096 (TG-P5)",
   "citation": "bench/tg-p5-attention-generated-default/latest.json", "legacy_route_id": "decode_flash_block_tile_g5_8b_refuted"},
  {"axis": "g5_block_tile_L_geometry", "compatibility_aliases": ("g5_block_tile_8b_L_geometry",), "domain": "attention", "disposition": "refuted: L=128 is the geometry optimum (87.7%/95.9%); larger L monotonically worse (69%/75.6% at L=576, occupancy-starved) -- the generated route needs ~36 splits for parallelism so it over-launches at low ctx (TG-P8.2)",
   "citation": "bench/tg-p8-generated-8b-attention-parity/geometry_search.json"},
  {"axis": "g5_block_tile_combine_lifecycle_cap", "compatibility_aliases": ("g5_block_tile_8b_combine_lifecycle_cap",), "domain": "attention", "disposition": "blocking: the generated 3-kernel gmax+combine lifecycle is 556us/fwd (83% of the ctx4096 attention delta) vs owned's fused 224us -> BINDING cap at ctx4096 (95.9%); a perfect tile saves only 112us. Combine COLLAPSE is refuted (guardrail #3); reopen only with a NEW non-collapse coordination primitive (TG-P8.1/P8.2)",
   "citation": "bench/tg-p8-generated-8b-attention-parity/latest.json"},
  {"axis": "live_split_geometry_tile", "compatibility_aliases": ("live_split_geometry_8b_tile",), "domain": "attention", "disposition": "SOLVED/PROMOTED: live-context split geometry (fixed S, per=ceildiv(Tc,S)) is expressible in generated UOp; the live-split route plus KV_BOTH staging and fused combine is now the 8B generated default. extra/llm_research/live_split_geometry.py",
   "citation": "bench/tg-p9-pure-attention-primitive-route/live_split_tile_microgate.json"},
  {"axis": "split_preserving_lse_combine", "compatibility_aliases": ("split_preserving_lse_combine_8b",), "domain": "attention", "disposition": "EMITTER_BLOCKED (TG-P9.4): a split-preserving generated combine (de-dup the per-d fexp / fuse gmax without collapsing Hq*S or Hq*Hd) mis-vectorizes the reduction-accumulator REG to a non-assignable make_float4(...) store; REG_STORE_DEVEC=1 compiles but NaNs. The ctx4096 556us combine cap cannot be removed in current AMD codegen. Reopen: a codegen fix keeping the reduction-accumulator REG scalar for a multi-reduce/weight-sharing combine.",
   "citation": "bench/tg-p9-pure-attention-primitive-route/combine_microgate.json"},
]

# ---- DEFERRED capability frontier: blocked-but-OPEN, NOT refuted on merits. The pure-search north-star (replace the
#      two hand-written decode kernels with a fully searched native route) is gated on renderer lowering of these
#      primitives; reopen each when the capability lands. Distinct from REFUTED (lost on its merits) and from shipped. ----
DEFERRED_CAPABILITIES = [
  {"capability": "v_dot2_lowering", "status": "deferred", "domain": "codegen",
   "blocks": "native packed-fp16 dot in a searched decode-attention / GEMV kernel",
   "reopen_when": "the renderer lowers v_dot2 (packed fp16 dot) so the search space can emit it natively"},
  {"capability": "cross_lane_mixed_reduce", "status": "deferred", "domain": "codegen",
   "blocks": "native cross-lane reduction lowering for a searched tile (LDS path already native; ds_bpermute tree TODO)",
   "reopen_when": "the renderer lowers the cross-lane reduction so a searched kernel can own the reduction topology"},
]

# ---- tiny helpers (no side effects unless you call apply_route) ----
ROUTE_COMPATIBILITY_ALIASES = (
  {"canonical_route_id": "decode_flash_live_split_g4_kvboth",
   "compatibility_aliases": ("decode_flash_live_split_g4_8b_kvboth",)},
  {"canonical_route_id": "prefill_q4k_q8_1_hybrid_mmq_atom",
   "compatibility_aliases": ("prefill_14b_q4k_q8_1_hybrid_mmq_atom",)},
)

def _route_alias_map() -> dict[str, str]:
  return {alias: row["canonical_route_id"] for row in ROUTE_COMPATIBILITY_ALIASES
          for alias in row["compatibility_aliases"]}

class _CanonicalRouteTable(dict):
  """Canonical keys with exact legacy reads for old artifact/registry consumers."""
  def __contains__(self, route_id) -> bool:
    return dict.__contains__(self, route_id) or route_id in _route_alias_map()
  def __getitem__(self, route_id):
    return dict.__getitem__(self, _route_alias_map().get(route_id, route_id))
  def get(self, route_id, default=None):
    try: return self[route_id]
    except KeyError: return default

ROUTES = _CanonicalRouteTable(ROUTES)

def immutable_route_registry():
  """Return an immutable manifest snapshot for explicit selector/admission use."""
  def freeze(value):
    if isinstance(value, dict): return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)): return tuple(freeze(v) for v in value)
    return value
  return freeze(dict(ROUTES))

def canonical_route_id(route_id: str, registry: Mapping|None = None) -> str:
  """Resolve one complete legacy spelling. Prefix, case-folded, and partial matches are intentionally unsupported."""
  aliases = _route_alias_map()
  registry = immutable_route_registry() if registry is None else registry
  if route_id in registry: return route_id
  if route_id in aliases: return aliases[route_id]
  raise KeyError(f"unknown route_id {route_id!r}; known: {sorted((*registry, *aliases))}")

def route(route_id: str, registry: Mapping|None = None) -> dict:
  registry = immutable_route_registry() if registry is None else registry
  return registry[canonical_route_id(route_id, registry)]

def route_env(route_id: str) -> dict:
  """The env vars to SET to force this route onto the active path ({} means it is the shipped default)."""
  return dict(route(route_id).get("env", {}))

def rollback_env(route_id: str) -> dict:
  """The env vars that leave this route for its rollback target ({} means it IS a rollback target)."""
  return dict(route(route_id).get("rollback", {}))

# Semantic policy identity deliberately excludes benchmark/model labels.  Legacy artifacts may still carry those
# labels, but they are exposed only as provenance by promoted_prefill_candidate_policy().
_PROVENANCE_KEYS = frozenset(("profile", "profile_id", "profiles", "model", "model_id", "model_name",
                              "model_path", "filename", "size_label", "model_size"))

def _semantic_json(value) -> str:
  def clean(x):
    if isinstance(x, Mapping):
      return {str(k):clean(v) for k,v in sorted(x.items(), key=lambda item: str(item[0]))
              if str(k).lower().replace("-", "_") not in _PROVENANCE_KEYS}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if x is None or isinstance(x, (str, bool, int, float)): return x
    raise TypeError(f"semantic identity requires JSON values, got {type(x).__name__}")
  return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

def _identity(namespace: str, value) -> str:
  return f"{namespace}:sha256:" + hashlib.sha256(_semantic_json(value).encode("ascii")).hexdigest()

def canonical_capability_identity(capability: Mapping) -> str:
  """Identity of the complete scanned target/route capability contract; labels never participate."""
  if not isinstance(capability, Mapping) or not capability: raise ValueError("capability must be a non-empty mapping")
  _target(capability.get("target"))
  return _identity("capability", capability)

def _target(value) -> dict:
  if not isinstance(value, Mapping): raise ValueError("exact scanned target facts are required")
  try: out = {"backend": str(value["backend"]), "arch": str(value["arch"]), "wave_size": int(value["wave_size"])}
  except (KeyError, TypeError, ValueError): raise ValueError("target requires backend/arch/wave_size scanner facts") from None
  if not out["backend"] or not out["arch"] or out["wave_size"] <= 0:
    raise ValueError("target requires backend/arch/wave_size scanner facts")
  # Preserve additional scanned geometry/resource facts as material identity inputs.
  return {**value, **out}

def _shape(row: Mapping) -> tuple[int, int, int]:
  shape = row.get("shape", row)
  try: out = tuple(int(shape.get(lo, shape.get(lo.upper()))) for lo in ("m", "n", "k"))
  except (AttributeError, TypeError, ValueError): raise ValueError("policy row requires exact positive M/N/K") from None
  if any(v <= 0 for v in out): raise ValueError("policy row requires exact positive M/N/K")
  return out  # type: ignore[return-value]

def _inventory_rows(inventory: Mapping) -> list[dict]:
  rows = inventory.get("rows")
  if not isinstance(rows, list) or not rows: raise ValueError("inventory requires non-empty rows")
  out = []
  for row in rows:
    if not isinstance(row, Mapping): raise ValueError("malformed inventory row")
    phase, role = row.get("phase", "prefill"), row.get("role")
    quant = row.get("quant_format", row.get("quant"))
    if not all(isinstance(x, str) and x for x in (phase, role, quant)): raise ValueError("inventory row lacks phase/role/quant")
    target = _target(row.get("target", inventory.get("target")))
    out.append({"phase": phase, "role": role, "quant": quant, "shape": dict(zip(("m", "n", "k"), _shape(row))),
                "target": target, "tensor_identities": sorted(row.get("tensor_identities", ())),
                "packed_abi": row.get("packed_abi", row.get("layout")), "call_count": row.get("call_count")})
  return out

def canonical_inventory_identity(inventory: Mapping) -> str:
  """Canonical identity of exact routed inventory content, independent of model/profile spelling."""
  rows = sorted(_inventory_rows(inventory), key=lambda r: (r["phase"], r["role"], r["quant"], *r["shape"].values()))
  derived = _identity("inventory", rows)
  recorded = inventory.get("inventory_identity")
  if recorded is not None:
    if not isinstance(recorded, str) or not recorded: raise ValueError("invalid inventory identity")
    if recorded.startswith("inventory:sha256:") and recorded != derived:
      raise ValueError("inventory identity mismatch")
    # Validate the un-namespaced identity emitted by prefill.workload_inventory without making its profile provenance
    # or model path semantic. The manifest identity additionally includes phase and scanned target facts.
    if inventory.get("schema") == "qk.packed_prefill_workload_inventory.v1":
      legacy_rows = []
      try:
        for row in inventory["rows"]:
          legacy_rows.append({"role": row["role"], "quant_format": row["quant_format"],
            "shape": {x:row["shape"][x] for x in ("m", "n", "k")},
            "layout": {x:row["layout"][x] for x in ("logical", "packed", "block_elems", "block_bytes")},
            "tensor_identities": sorted(row["tensor_identities"]), "call_count": row["call_count"],
            "source_bytes": row["source_bytes"]})
      except (KeyError, TypeError): raise ValueError("malformed workload inventory row") from None
      legacy_rows.sort(key=lambda x: (x["role"], x["quant_format"], x["shape"]["m"], x["shape"]["n"], x["shape"]["k"]))
      expected = hashlib.sha256(json.dumps(legacy_rows, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
      if recorded != expected: raise ValueError("inventory identity mismatch")
  return derived

def _research_candidate_set_identity(candidate_set: Mapping) -> str:
  """Research-only identity for mixed candidate/fallback experiment manifests."""
  entries = candidate_set.get("entries")
  fallbacks = candidate_set.get("fallbacks", [])
  if not isinstance(entries, list) or not isinstance(fallbacks, list) or not entries and not fallbacks:
    raise ValueError("candidate set requires candidate entries or declared fallbacks")
  canonical = []
  for entry in entries:
    if not isinstance(entry, Mapping) or not isinstance(entry.get("payload"), Mapping): raise ValueError("malformed candidate entry")
    canonical.append({"canonical_identity": entry.get("canonical_identity"), "payload": entry["payload"]})
  if not fallbacks: return _identity("candidate_set", sorted(canonical, key=_semantic_json))
  declared = [_validated_fallback(row) for row in fallbacks]
  return _identity("candidate_set", {"entries":sorted(canonical, key=_semantic_json),
                                     "fallbacks":sorted(declared, key=_semantic_json)})


def canonical_candidate_set_identity(candidate_set: Mapping) -> str:
  """Use production identity for an admitted set; retain research identity for mixed rollback manifests."""
  if isinstance(candidate_set, Mapping) and set(candidate_set) == {"schema", "entries"}:
    return _production_candidate_set_identity(candidate_set)
  return _research_candidate_set_identity(candidate_set)

def _validated_fallback(row: Mapping) -> dict:
  """Validate one exact, evidence-bearing rollback declaration and return its semantic content."""
  if not isinstance(row, Mapping) or not isinstance(row.get("workload"), Mapping):
    raise ValueError("malformed declared fallback")
  route_id, evidence = row.get("route_id"), row.get("evidence")
  if not isinstance(route_id, str) or not route_id or not isinstance(evidence, Mapping) or not evidence:
    raise ValueError("declared fallback requires route and evidence")
  if evidence.get("status") != "qualified": raise ValueError("declared fallback evidence is not qualified")
  evidence_identity = _identity("fallback_evidence", evidence)
  if row.get("evidence_identity") != evidence_identity: raise ValueError("declared fallback evidence identity mismatch")
  content = {"workload":row["workload"], "route_id":route_id, "evidence_identity":evidence_identity}
  fallback_identity = _identity("fallback", content)
  if row.get("fallback_identity") != fallback_identity: raise ValueError("declared fallback identity mismatch")
  return {**content, "fallback_identity":fallback_identity}

def canonical_policy_rows(inventory: Mapping, capability: Mapping, candidate_set: Mapping, *,
                          route_id: str = "prefill_wmma_lds_dbuf_generated") -> tuple[dict, ...]:
  """Bind candidates to exact inventory/capability facts. Missing, duplicate, or mismatched coverage fails closed."""
  inv_rows, cap_id = _inventory_rows(inventory), canonical_capability_identity(capability)
  capability_target = _target(capability["target"])
  inv_id, set_id = canonical_inventory_identity(inventory), canonical_candidate_set_identity(candidate_set)
  candidates = {}
  for entry in candidate_set["entries"]:
    workload = entry["payload"].get("workload", {})
    key = (workload.get("phase", "prefill"), workload.get("role"),
           workload.get("quant_format", workload.get("quant", workload.get("dtypes", {}).get("b"))), _shape(workload),
           _semantic_json(_target(workload.get("target"))))
    if key in candidates: raise ValueError(f"duplicate structural candidate {key!r}")
    candidates[key] = entry
  fallbacks = {}
  for raw in candidate_set.get("fallbacks", []):
    fallback = _validated_fallback(raw)
    workload = fallback["workload"]
    key = (workload.get("phase", "prefill"), workload.get("role"),
           workload.get("quant_format", workload.get("quant")), _shape(workload),
           _semantic_json(_target(workload.get("target"))))
    if key in candidates or key in fallbacks: raise ValueError(f"ambiguous candidate/fallback binding {key!r}")
    fallbacks[key] = fallback
  rows = []
  for inv in inv_rows:
    if _semantic_json(inv["target"]) != _semantic_json(capability_target):
      raise ValueError("inventory target does not match scanned capability target")
    supported_phases = capability.get("phases", (capability.get("phase"),))
    supported_quants = capability.get("quant_formats", (capability.get("quant_format"), capability.get("quant")))
    if inv["phase"] not in supported_phases or inv["quant"] not in supported_quants:
      raise ValueError("inventory phase/quant is outside the scanned capability contract")
    key = (inv["phase"], inv["role"], inv["quant"], _shape(inv), _semantic_json(inv["target"]))
    entry = candidates.get(key)
    fallback = fallbacks.get(key)
    if entry is None and fallback is None: raise ValueError(f"candidate set does not cover exact inventory row {key[:4]!r}")
    binding = ({"binding_kind":"candidate", "candidate_identity":entry.get("canonical_identity"),
                "selected_route":route_id, "route_aliases":[route_id]} if entry is not None else
               {"binding_kind":"fallback", "fallback_identity":fallback["fallback_identity"],
                "fallback_evidence_identity":fallback["evidence_identity"],
                "selected_route":fallback["route_id"], "route_aliases":[fallback["route_id"]]})
    rows.append({"phase": inv["phase"], "role": inv["role"], "quant": inv["quant"], "shape": inv["shape"],
      "target": inv["target"], "capability_identity": cap_id, "inventory_identity": inv_id,
      "candidate_set_identity": set_id, **binding})
  if len(candidates) + len(fallbacks) != len(rows):
    raise ValueError("candidate set contains rows outside the exact inventory")
  return tuple(rows)

def lookup_policy_row(policy_rows, *, phase: str, role: str, quant: str, shape, target: Mapping,
                      capability_identity: str, inventory_identity: str, candidate_set_identity: str) -> dict | None:
  """Exact structural lookup. No wildcard, profile, status, or default-on fallback is permitted."""
  wanted = (phase, role, quant, _shape({"shape": shape}), _semantic_json(_target(target)), capability_identity,
            inventory_identity, candidate_set_identity)
  matches = [row for row in policy_rows if (row.get("phase"), row.get("role"), row.get("quant"), _shape(row),
    _semantic_json(row.get("target", {})), row.get("capability_identity"), row.get("inventory_identity"),
    row.get("candidate_set_identity")) == wanted]
  if len(matches) > 1: raise ValueError("ambiguous exact manifest policy lookup")
  return dict(matches[0]) if matches else None

def promoted_prefill_candidate_policy() -> dict:
  """Return the single promoted full-kernel candidate policy consumed by runtime and search selection.

  The manifest owns promotion; the candidate-set artifact owns exact payloads and canonical identities. Keeping those
  concerns separate lets machine search replace a candidate set without duplicating its schedules in runtime code.
  """
  route_id = "prefill_wmma_lds_dbuf_generated"
  row = route(route_id)
  if row.get("status") != "promoted_default" or row.get("provenance") != "tinygrad_scheduler_generated":
    raise RuntimeError(f"promoted prefill candidate policy is not eligible: route={route_id} "
                       f"status={row.get('status')!r} provenance={row.get('provenance')!r}")
  relpath = pathlib.Path(str(row["candidate_set_path"]))
  path = pathlib.Path(__file__).resolve().parents[2] / relpath
  if not path.is_file(): raise FileNotFoundError(f"promoted prefill candidate set is missing: {path}")
  artifact = json.loads(path.read_text())
  entries = artifact.get("entries", ())
  if artifact.get("schema") != "boltbeam.full_kernel_candidate_set.v1" or not isinstance(entries, list) or not entries:
    raise RuntimeError("promoted prefill candidate set has an invalid schema or no entries")
  profiles = tuple(sorted({str(entry["payload"]["workload"]["profile"]) for entry in entries}))
  roles = tuple(str(role) for role in row.get("candidate_roles", ()))
  if set(roles) != set(row.get("roles", ())):
    raise RuntimeError(f"promoted prefill candidate roles drifted from route roles: {roles!r} vs {row.get('roles')!r}")
  artifact_roles = {str(entry["payload"]["workload"]["role"]) for entry in entries}
  if artifact_roles != set(roles):
    raise RuntimeError(f"promoted prefill candidate artifact roles drifted from policy: {sorted(artifact_roles)!r} vs {roles!r}")
  return {
    "route_id": route_id, "route_aliases": (route_id,), "candidate_set_path": str(path),
    "candidate_set_identity": canonical_candidate_set_identity(artifact),
    # Compatibility artifact metadata only. It is intentionally not a semantic support predicate.
    "candidate_profiles": profiles, "provenance_profiles": profiles, "candidate_roles": roles,
    "semantic_policy_rows": (),
  }

def promoted_prefill_candidate_policy_rows(inventory: Mapping, capability: Mapping) -> tuple[dict, ...]:
  """Read the legacy promoted artifact and bind it to caller-supplied exact semantic facts."""
  policy = promoted_prefill_candidate_policy()
  artifact = json.loads(pathlib.Path(policy["candidate_set_path"]).read_text())
  return canonical_policy_rows(inventory, capability, artifact, route_id=policy["route_id"])

def apply_route(route_id: str, env: dict | None = None) -> dict:
  """Materialize a research route onto an explicit configuration copy.

  No ambient process configuration is consulted. ``strict_fallback`` routes
  set ``QK_STRICT_FALLBACK=1`` (fail-loud).
  """
  out = dict({} if env is None else env)
  out.update({k: str(v) for k, v in route_env(route_id).items()})
  if route(route_id).get("strict_fallback"): out.setdefault("QK_STRICT_FALLBACK", "1")
  return out

def is_refuted(axis: str) -> bool:
  return any(r["axis"] == axis or axis in r.get("compatibility_aliases", ()) for r in REFUTED)

def default_routes() -> list[str]:
  """Routes on the live default path (promoted generated default OR owned shipped default)."""
  return [rid for rid, r in ROUTES.items() if r["status"] in ("promoted_default", "default_shipped")]

def routes_by_status(status: str) -> list[str]:
  return [rid for rid, r in ROUTES.items() if r["status"] == status]

def route_provenance(route_id: str) -> str:
  prov = str(route(route_id).get("provenance", ""))
  if prov not in ROUTE_PROVENANCE:
    raise ValueError(f"route {route_id!r} has invalid provenance {prov!r}; expected one of {ROUTE_PROVENANCE}")
  return prov

def default_purity_report() -> dict:
  """Strict final-default purity report. This is intentionally allowed to FAIL today.

  A generated/search route can be fast and correct but still fail final-default purity if its implementation is an
  external handwritten kernel. A hand-authored UOp route is reported as transitional debt: allowed to keep shipping
  only while its replacement scope is explicit.
  """
  defaults = default_routes()
  rows, forbidden, transitional = [], [], []
  for rid in defaults:
    r, prov = route(rid), route_provenance(rid)
    row = {"route_id": rid, "status": r["status"], "provenance": prov,
           "replacement_scope": r.get("replacement_scope", ""), "final_default_allowed": prov in FINAL_DEFAULT_PROVENANCE}
    rows.append(row)
    if prov in FORBIDDEN_DEFAULT_PROVENANCE: forbidden.append(rid)
    if prov in TRANSITIONAL_DEFAULT_PROVENANCE: transitional.append(rid)
  verdict = "TINYGRAD_DEFAULT_PURITY_PASS" if not forbidden and not transitional else "TINYGRAD_DEFAULT_PURITY_FAIL"
  return {"verdict": verdict, "default_routes": defaults, "rows": rows,
          "forbidden_default_routes": forbidden, "transitional_default_routes": transitional,
          "final_default_allowed_provenance": sorted(FINAL_DEFAULT_PROVENANCE)}

def validate_manifest() -> list[str]:
  errors: list[str] = []
  for rid, r in ROUTES.items():
    prov = r.get("provenance")
    if prov not in ROUTE_PROVENANCE:
      errors.append(f"{rid}: invalid or missing provenance {prov!r}")
    if r["status"] in ("promoted_default", "default_shipped"):
      if prov == "rollback_oracle":
        errors.append(f"{rid}: default route cannot be provenance=rollback_oracle")
      if prov in ("hand_authored_uop_template", "external_handwritten_kernel") and not r.get("replacement_scope"):
        errors.append(f"{rid}: non-pure default provenance={prov} requires replacement_scope")
    if r.get("research_only"):
      if r["status"] != "research":
        errors.append(f"{rid}: research_only route cannot claim final status {r['status']!r}")
      if not {"correctness evidence", "resource evidence", "timing evidence"}.issubset(
          {part.strip() for part in str(r.get("authority_gate", "")).split("+")}):
        errors.append(f"{rid}: research_only route must require correctness/resource/timing evidence")
    # purity_status is derived, not declared: any stored value must equal derive_purity_status(status, provenance).
    if "purity_status" in r:
      expected = derive_purity_status(r["status"], str(prov))
      if r["purity_status"] != expected:
        errors.append(f"{rid}: purity_status={r['purity_status']!r} drifted from derived {expected!r} "
                      f"(status={r['status']}, provenance={prov})")
  return errors

def to_manifest_dict() -> dict:
  return {"_schema": "default route manifest (PMS-R1)", "generated_by": "extra/llm_research/route_manifest.py",
          "profiles": {"decode": PROFILE_DECODE, "prefill": PROFILE_PREFILL},
          "provenance_vocabulary": list(ROUTE_PROVENANCE),
          "routes": ROUTES, "refuted_axes": REFUTED,
          "default_routes": default_routes(),
          "promoted_defaults": routes_by_status("promoted_default"),
          "owned_defaults": routes_by_status("default_shipped"),
          "default_purity": default_purity_report()}

def dump(out_path: str | None = None) -> str:
  """Write the canonical manifest json (bench/qk-search-spaces/default_route_manifest.json by default)."""
  root = pathlib.Path(__file__).resolve().parents[2]
  p = pathlib.Path(out_path) if out_path else (root / "bench/qk-search-spaces/default_route_manifest.json")
  p.parent.mkdir(parents=True, exist_ok=True)
  json.dump(to_manifest_dict(), open(p, "w"), indent=2)
  return str(p)

# ---- refuted axes: REFUTED is the SINGLE SOURCE; do_not_search (search_profiles.json) and the quant
#      known_refuted_route_families must agree with it (enforced by qk_search_space_manifest_check). ----
def disposition_class(disp: str) -> str:
  """Normalize a disposition to its class token (refuted / deprioritized / exhausted / correct_not_fast / small / ...)."""
  return (disp or "").split(":")[0].split("/")[0].split()[0].lower()

def refuted_index() -> dict[str, str]:
  """{key -> disposition_class} for every refuted axis, keyed by route_id when present else axis."""
  return {(r.get("route_id") or r["axis"]): disposition_class(r["disposition"]) for r in REFUTED}

def dump_refuted(out_path: str | None = None) -> str:
  """Write the canonical refuted-axes json FROM REFUTED (the single source for do_not_search / quant known_refuted)."""
  root = pathlib.Path(__file__).resolve().parents[2]
  p = pathlib.Path(out_path) if out_path else (root / "bench/qk-search-spaces/refuted_axes.json")
  p.parent.mkdir(parents=True, exist_ok=True)
  json.dump({"_schema": "canonical refuted axes (generated FROM qk_route_manifest.REFUTED)",
             "generated_by": "extra/llm_research/route_manifest.py:dump_refuted",
             "agreement_key": "route_id when present else axis; disposition compared by class token",
             "refuted_axes": REFUTED}, open(p, "w"), indent=2)
  return str(p)

if __name__ == "__main__":
  if (errs := validate_manifest()):
    raise SystemExit("manifest validation failed:\n- " + "\n- ".join(errs))
  path = dump()
  rpath = dump_refuted()
  print(f"wrote default route manifest to {path}")
  print(f"wrote canonical refuted axes to {rpath}")
  print("default routes:", default_routes())
  print("promoted (generated/search-selected) defaults:", routes_by_status("promoted_default"))
  print("default purity:", default_purity_report()["verdict"])
  print(f"{len(ROUTES)} routes, {len(REFUTED)} refuted axes")
