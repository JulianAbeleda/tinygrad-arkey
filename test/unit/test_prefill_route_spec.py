"""Round-trip proof for the prefill two-wrapper flag-collapse (Phase 1 SKELETON).

Proves the canonical wrapper and the resolver agree with the manifest, byte-for-byte:
  * CanonicalRoute.select(name) == route_manifest.ROUTES[route_id]["env"] for every named route.
  * resolve_prefill_route(that env).route_name == name (env -> spec round-trip).
No runtime behaviour is exercised; this is a pure data contract test.
"""
import unittest

from extra.qk.route_manifest import ROUTES
from extra.qk.prefill_route_select import CanonicalRoute, ROUTE_NAME_TO_ID
from tinygrad.codegen.opt.prefill_route_spec import (
  resolve_prefill_route, PrefillRouteSpec,
  ROUTE_HYBRID, ROUTE_SPEC_OWNED, ROUTE_MIXED, ROUTE_PIPE_RESEARCH, ROUTE_DEFAULT, ROUTE_NAMES,
)

# (route_name, expected selector booleans). The env comes from the manifest, not restated here.
_EXPECTED = {
  #                graph_gemm  pipe   lds    dbuf
  ROUTE_HYBRID:   (True,      False, False, False),
  ROUTE_SPEC_OWNED: (True,    True,  True,  True),
  ROUTE_MIXED:    (True,      False, True,  True),
  ROUTE_DEFAULT:  (False,     False, False, False),
}


class TestPrefillRouteRoundTrip(unittest.TestCase):
  def test_names_cover_all_routes(self):
    self.assertEqual(set(ROUTE_NAME_TO_ID), set(_EXPECTED))
    self.assertEqual(set(ROUTE_NAMES), set(_EXPECTED) | {ROUTE_PIPE_RESEARCH})

  def test_select_matches_manifest_env(self):
    # CanonicalRoute.select(name) is byte-identical to the manifest's defining env for that route_id.
    for name, route_id in ROUTE_NAME_TO_ID.items():
      with self.subTest(route=name):
        self.assertEqual(CanonicalRoute.select(name), dict(ROUTES[route_id]["env"]))

  def test_env_resolves_back_to_route_name(self):
    # env -> spec: the resolver classifies the canonical env back to its own route_name and boolean shape.
    for name, (gg, pipe, lds, dbuf) in _EXPECTED.items():
      with self.subTest(route=name):
        spec = resolve_prefill_route(CanonicalRoute.select(name))
        self.assertIsInstance(spec, PrefillRouteSpec)
        self.assertEqual(spec.route_name, name)
        self.assertEqual((spec.graph_gemm, spec.wmma_pipe_primitive, spec.wmma_lds_primitive, spec.dbuf),
                         (gg, pipe, lds, dbuf))

  def test_empty_env_is_default(self):
    spec = resolve_prefill_route({})
    self.assertEqual(spec.route_name, ROUTE_DEFAULT)
    self.assertFalse(spec.graph_gemm or spec.wmma_pipe_primitive or spec.wmma_lds_primitive or spec.dbuf)

  def test_select_rejects_unknown_route(self):
    with self.assertRaises(KeyError):
      CanonicalRoute.select("not_a_route")

  def test_pipe_research_is_not_canonically_selectable(self):
    with self.assertRaises(KeyError):
      CanonicalRoute.select(ROUTE_PIPE_RESEARCH)

  def test_select_returns_a_copy(self):
    # Mutating the returned dict must not corrupt the manifest (defensive-copy contract).
    env = CanonicalRoute.select(ROUTE_SPEC_OWNED)
    env["PREFILL_GRAPH_GEMM"] = "999"
    self.assertEqual(ROUTES[ROUTE_NAME_TO_ID[ROUTE_SPEC_OWNED]]["env"]["PREFILL_GRAPH_GEMM"], "1")


if __name__ == "__main__":
  unittest.main()
