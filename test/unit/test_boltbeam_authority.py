from extra.llm_research.boltbeam_authority import load_promoted_routes

def test_authority_ledger_has_promoted_routes():
  routes = load_promoted_routes()
  assert len(routes) == 26
  assert routes["q6_ffn_down_streamk_destination.v1"]["state"] == "exact"
