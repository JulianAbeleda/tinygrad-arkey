"""LR-061: prove prefill/decode admission agrees with the route manifest, rather than unifying them.

The scope originally said "unify prefill and decode admission". That premise was corrected before this was
written: the two paths are not two copies of one algorithm. Decode admits through frozen candidate dataclasses
with a `bind()` that returns None or a binding (`tinygrad/llm/decode_routes.py`); prefill admits through
predicate functions over a linear's attributes (`tinygrad/llm/prefill_routes.py`). Merging those would be a
rewrite of both, justified by a symmetry that is not there.

What IS there is duplication of a different kind, and it is the kind that actually breaks: every admission
guard is stated twice -- once as a field on the candidate that enforces it, and once in
`extra/llm_research/route_manifest.json`, which is the file the audit, the promotion machinery and `KernelSpec` all
treat as authoritative. Nothing checked that the two agree. `_Q4KDecodeCandidate.k_multiple = 1024` and the
manifest's `(K//256)%4==0` are the same constraint written in two notations, and one of those notations is a
prose string that no code parses.

So this is a consistency gate, not a unification. It fails when a guard is changed in one place and not the
other. It deliberately does not try to make the two representations into one -- that is LR-062's boundary
question, and doing it here would mean editing the manifest that everything else trusts.

Coverage is partial and the gaps are named below rather than papered over: of the four decode candidates,
only Q4_K states its numeric guard in a form a machine can compare. See UNCHECKED_GUARDS.
"""
from __future__ import annotations

import dataclasses
import re

import pytest

from tinygrad.llm import decode_routes
from extra.llm_research import route_manifest


def _candidates() -> dict[str, object]:
  """Every frozen decode-candidate singleton in decode_routes, by attribute name."""
  out = {}
  for name in dir(decode_routes):
    obj = getattr(decode_routes, name)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type) and hasattr(obj, "route_id") and hasattr(obj, "bind"):
      out[name] = obj
  return out


def _guards(route_id: str) -> list[dict]:
  return [dict(g) for g in route_manifest.route(route_id).get("shape_guards", ())]


# Guards a candidate enforces that the manifest does NOT state in any comparable form. Each one is a place where
# admission can change and no gate will notice. Listed explicitly so the list can only shrink deliberately: a new
# unchecked guard makes test_unchecked_guard_list_is_exhaustive fail.
UNCHECKED_GUARDS = {
  # manifest states roles and per-role K/N, but no K-multiple or row-tile rule at all
  "Q6K_DECODE_CANDIDATE": {"batch", "tokens", "k_multiple", "row_tile"},
  # decode is 1 batch x 1 token by definition; the manifest does not restate it
  "Q4K_DECODE_CANDIDATE": {"batch", "tokens"},
  # the 48-wide live split and KV_BOTH staging are implementation geometry, absent from the manifest
  "FLASH_DECODE_CANDIDATE": {"split_size", "staging", "query_group_size", "stage_width"},
  "FLASH_DECODE_G5_CANDIDATE": {"split_size", "staging", "query_group_size", "stage_width"},
}

# Guard fields a test above actually compares against the manifest.
CHECKED_GUARDS = {
  "Q4K_DECODE_CANDIDATE": {"k_multiple", "n_multiple"},
}

# Fields that are identity or target rather than shape guards; checked by their own tests below.
_NON_GUARD_FIELDS = {"candidate_id", "route_id", "quant", "target", "query_heads", "kv_heads", "head_dim"}


def test_every_decode_candidate_names_a_route_that_exists():
  """The weakest link and the most likely to rot: a candidate whose route_id was renamed in the manifest would
  still admit traffic, while every manifest-driven tool reported on a route nothing selects."""
  assert _candidates(), "no decode candidates discovered -- this gate would vacuously pass"
  for name, cand in _candidates().items():
    route_manifest.route(cand.route_id)  # raises KeyError if absent


def test_candidate_quant_matches_the_manifest():
  for name, cand in _candidates().items():
    quant = getattr(cand, "quant", None)
    if quant is None: continue          # the flash candidates carry no quant field; manifest says fp16
    assert quant.name in route_manifest.route(cand.route_id)["quant"], \
      f"{name} admits {quant!r} but manifest route {cand.route_id!r} does not list it"


def test_query_head_count_matches_the_manifest_shape_guard():
  """FLASH_DECODE_CANDIDATE.query_heads=32 / G5=40 against the manifest's Hq. These are the two guards that are
  already stated numerically on both sides, so they are the ones a drift gate can actually hold."""
  checked = 0
  for name, cand in _candidates().items():
    qh = getattr(cand, "query_heads", None)
    if qh is None: continue
    hqs = {g["Hq"] for g in _guards(cand.route_id) if "Hq" in g}
    assert hqs, f"{name} enforces query_heads={qh} but manifest route {cand.route_id!r} states no Hq"
    assert qh in hqs, f"{name} admits Hq={qh}, manifest says {sorted(hqs)}"
    checked += 1
  assert checked == 2, f"expected the two flash candidates to be checked here, got {checked}"


# --------------------------------------------------------------------------------------------------------------
# The numeric-condition comparison. Only Q4_K states its rule in a parsable form today.
# --------------------------------------------------------------------------------------------------------------

_DIV_MOD = re.compile(r"\(\s*([A-Za-z_]\w*)\s*//\s*(\d+)\s*\)\s*%\s*(\d+)\s*==\s*0")
_PLAIN_MOD = re.compile(r"(?<![\w)])([A-Za-z_]\w*)\s*%\s*(\d+)\s*==\s*0")


def parse_multiples(condition: str) -> dict[str, int]:
  """Extract {symbol: required multiple} from a manifest condition string.

  Handles the two forms the manifest actually uses: `(K//256)%4==0`, which requires K to be a multiple of
  256*4, and `N%32==0`, which requires a multiple of 32. Anything else is ignored rather than guessed at -- a
  condition this cannot read must show up as an unchecked guard, not as a silent pass.
  """
  out: dict[str, int] = {}
  for sym, div, mod in _DIV_MOD.findall(condition):
    out[sym] = int(div) * int(mod)
  for sym, mod in _PLAIN_MOD.findall(condition):
    out.setdefault(sym, int(mod))
  return out


def test_parse_multiples_reads_both_manifest_notations():
  got = parse_multiples("DECODE_Q4K_G3_ANYSHAPE=1 and (K//256)%4==0 and N%32==0")
  assert got == {"K": 1024, "N": 32}


def test_parse_multiples_ignores_what_it_cannot_read():
  assert parse_multiples("ctx >= 512 and role == 'ffn_down'") == {}


def test_q4k_k_and_n_multiples_match_the_manifest_condition():
  """`k_multiple=1024` / `n_multiple=32` on the candidate versus `(K//256)%4==0 and N%32==0` in the manifest --
  the same constraint in two notations, which until now nothing compared."""
  cand = _candidates()["Q4K_DECODE_CANDIDATE"]
  conds = [g["condition"] for g in _guards(cand.route_id) if "condition" in g]
  assert conds, "manifest route decode_q4k_g3_generated no longer states a parsable condition"
  multiples: dict[str, int] = {}
  for c in conds: multiples.update(parse_multiples(c))
  assert multiples.get("K") == cand.k_multiple, \
    f"candidate requires K % {cand.k_multiple} == 0, manifest requires K % {multiples.get('K')} == 0"
  assert multiples.get("N") == cand.n_multiple, \
    f"candidate requires N % {cand.n_multiple} == 0, manifest requires N % {multiples.get('N')} == 0"


def test_unchecked_guard_list_is_exhaustive():
  """Every candidate field is either compared against the manifest by a test above, or declared unchecked. A new
  guard field with no manifest counterpart fails here, which is the point: it should be a decision, not a
  default."""
  for name, cand in _candidates().items():
    # Flash candidates are compatibility binders over the executor-owned route
    # descriptor. Inspect that sole configuration authority, not the facade's
    # implementation fields (`route`, `target`).
    descriptor = cand.route if hasattr(cand, "route") else cand
    fields = {f.name for f in dataclasses.fields(descriptor)} - _NON_GUARD_FIELDS
    declared = UNCHECKED_GUARDS.get(name, set()) | CHECKED_GUARDS.get(name, set())
    assert fields == declared, (
      f"{name}: guard fields {sorted(fields)} but only {sorted(declared)} are accounted for. "
      f"Either compare the new field against the manifest (CHECKED_GUARDS) or declare it unchecked "
      f"(UNCHECKED_GUARDS), with a reason.")


def test_the_gate_is_not_vacuous():
  """Guards against the failure mode where a refactor renames the candidates and every loop above iterates over
  nothing while still reporting green."""
  cands = _candidates()
  assert len(cands) == 4, f"expected 4 decode candidates, found {sorted(cands)}"
  assert sum(1 for c in cands.values() if getattr(c, "quant", None)) == 2


@pytest.mark.parametrize("route_id", ["decode_q4k_g3_generated", "decode_q6k_coop_generated",
                                      "decode_flash_live_split_g4_kvboth", "decode_flash_live_split_g5_kvboth"])
def test_every_admitted_route_is_a_decode_workload(route_id):
  """A prefill route reached through the decode admission path would be a real routing bug and is cheap to
  exclude here, where the route ids are already in hand."""
  assert route_manifest.route(route_id)["workload"] == "decode"
