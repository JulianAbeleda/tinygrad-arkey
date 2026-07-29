import inspect
import unittest

from tinygrad.llm.production_route_interface import select_production_route
from tinygrad.llm.production_route_policy import GENERIC_ROUTE_ID, ROUTES_BY_ID, ProductionRouteRequest


def request(workload, role, quant, shape, backend="AMD", architecture="gfx1100"):
  return ProductionRouteRequest(workload, role, quant, backend, architecture, shape)


class TestProductionRouteInterface(unittest.TestCase):
  def test_g4_and_g5_keep_their_exact_selected_configs(self):
    g4 = select_production_route(request("decode", "attention_tile", "fp16", {"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128, "context": 512}))
    g5 = select_production_route(request("decode", "attention_combine", "fp16", {"B": 1, "Hq": 40, "Hkv": 8, "Hd": 128, "context": 4096}))
    self.assertEqual(g4.trace["route_id"], "decode_flash_live_split_g4_kvboth")
    self.assertEqual(g4.trace["selected_config"]["split_size"], 48)
    self.assertEqual(g5.trace["route_id"], "decode_flash_live_split_g5_kvboth")
    self.assertEqual(g5.trace["selected_config"], {"B": 1, "Hq": 40, "Hkv": 8, "Hd": 128, "split_size": 32, "query_group_size": 2, "staging": "KV_BOTH", "stage_width": 4})

  def test_prefill_flash_and_generic_fallback_are_explicit(self):
    selected = select_production_route(request("prefill", "attention_tile", "fp16", {"B": 1, "Hq": 40, "Hkv": 8, "Hd": 128, "q_tokens": 512, "kv_tokens": 1024}))
    self.assertTrue(selected.uses_selected_route)
    self.assertEqual(selected.trace["route_id"], "prefill_flash_attention_generated")
    fallback = select_production_route(request("prefill", "attention_tile", "fp16", {"B": 1, "Hq": 40, "Hkv": 8, "Hd": 128, "q_tokens": 256, "kv_tokens": 256}))
    self.assertFalse(fallback.uses_selected_route)
    self.assertEqual(fallback.trace["route_id"], GENERIC_ROUTE_ID)
    self.assertEqual(fallback.trace["fallback_reason"], "no_exact_current_route")

  def test_policy_is_master_owned_and_has_current_route_ids(self):
    self.assertEqual(len(ROUTES_BY_ID), 9)
    source = inspect.getsource(__import__("tinygrad.llm.production_route_policy", fromlist=["*"]))
    self.assertNotIn("extra.llm_research", source)
    self.assertIn("packed_wmma_prefill_generated", ROUTES_BY_ID)

