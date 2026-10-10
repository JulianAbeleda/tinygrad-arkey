"""A next-n (MTP) block in the GGUF is not a runtime block: the prefill inventory leaves its tensors out."""
from tinygrad.llm.model import derive_selected_gguf_prefill_inventory


def _meta(blocks):
  infos = []
  for b in range(blocks):
    infos += [(f"blk.{b}.attn_q.weight", (64, 64), 12, 0), (f"blk.{b}.ffn_down.weight", (256, 64), 12, 0)]
  return {"tensor_infos": infos + [("output.weight", (64, 1000), 14, 0), ("token_embd.weight", (64, 1000), 14, 0)]}


def test_nextn_block_is_left_out():
  kv = {"general.architecture": "qwen35", "qwen35.block_count": 3, "qwen35.nextn_predict_layers": 1, "qwen35.embedding_length": 64}
  inv = derive_selected_gguf_prefill_inventory(kv, _meta(3))
  names = {r["tensor_identity"] for r in inv["rows"]}
  assert not any(n.startswith("blk.2.") for n in names) and any(n.startswith("blk.1.") for n in names)
