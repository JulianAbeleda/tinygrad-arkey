import pytest

from extra.llm_research.decode.nv_sampler_feedback_tail_census import census

def _node(name, duration=1.0): return {"name":name, "duration_us":duration}

def _payload():
  nodes=[_node("noop") for _ in range(948)]
  nodes[0],nodes[1]=_node("E_hash", 2),_node("E_2_hash", 3)
  nodes[943]=_node("q6k_gen_coop_151936_4096_inkernel_hash")
  for i,(name,duration) in enumerate(zip(("E_1187_32_4_hash", "r_32_4_1187_hash", "r_128_16_8_1187_hash", "r_16_8_hash"),(4,5,6,7)),944): nodes[i]=_node(name,duration)
  edges=[{"from":a,"to":b} for a,b in ((943,944),(944,945),(944,946),(945,946),(946,947))]
  return {"nodes":nodes,"edges":edges}

def test_qualified_sampler_tail_is_a_serial_four_program_boundary():
  got=census(_payload())
  assert got["status"] == "PASS"
  assert got["sampler"]["tail_device_us"] == 22
  assert got["sampler"]["serial_dependencies"] == [[943,944],[944,945],[944,946],[945,946],[946,947]]

def test_rejects_missing_reduction_edge():
  payload=_payload(); payload["edges"].pop()
  with pytest.raises(ValueError, match="dependencies changed"): census(payload)
