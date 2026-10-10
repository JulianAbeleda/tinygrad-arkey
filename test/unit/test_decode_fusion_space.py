"""The decode fusion families: each names the nodes it covers and the operands it reads in parameter order, and a
shape's verdict is the emitter's own validate (constructed, not compiled)."""
from extra.llm_research import decode_fusion_space as fz


def test_every_family_reads_its_nodes_inputs_and_weights():
  for f in fz.FAMILIES:
    inside = {n.name for n in f.covers}
    reads = {r for n in f.covers for r in n.reads}
    outside = {r for r in reads if r.startswith("weight:") or r.split(":", 1)[1] not in inside}
    assert set(f.operands) == outside, f.family
    assert f.weights() and all(f"weight:{w}" in f.operands for w in f.weights()), f.family


def test_a_shape_verdict_is_the_emitters_own_words():
  verdicts = {v["family"]: v for v in fz.shape_verdicts([{"n": 96, "k": 300}])}
  assert all(r["refused"] for v in verdicts.values() for r in v["rows"])
  ok = {v["family"]: v for v in fz.shape_verdicts([{"n": 64, "k": 1024}])}
  assert any(r["refused"] is None for r in ok["q4_w1w3.v1"]["rows"])


def test_value_lengths_follow_their_consumers():
  fused = fz.BY_FAMILY["q4_g3_ffn_down_fused.v1"]
  assert fz._value_sizes(fused, 8, 16) == {"attn_residual": 8, "ffn_down": 8, "silu_mul": 16, "ffn_gate": 16, "ffn_up": 16}
  assert fz._value_sizes(fz.BY_FAMILY["q4_w1w3.v1"], 8, 16) == {"ffn_gate": 8, "ffn_up": 8, "ffn_norm": 16}
