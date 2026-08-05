import pathlib

from scratchpad.llama_cuda_ffn_down_oracle import ENTRIES
from scratchpad.cuda_decode_ffn_down_llama_graph_ab import FUSED, population

ROOT=pathlib.Path(__file__).resolve().parents[2]

def test_exact_fused_entries_cover_both_down_quant_types():
  assert "ggml_type12" in ENTRIES["Q4_K"] and "ELb1ELb0" in ENTRIES["Q4_K"]
  assert "ggml_type14" in ENTRIES["Q6_K"] and "ELb1ELb0" in ENTRIES["Q6_K"]

def test_semantic_manifest_pins_disjoint_18_plus_18_down_populations():
  manifest=ROOT/"docs/task_workflow/output/nv-decode-llama-tinygrad-semantic-call-manifest-20260804.json"
  assert population(manifest,"q4")==18
  assert population(manifest,"q6")==18

def test_q6_scatter_adapter_matches_row_major_row_pos_layout():
  src=(ROOT/"scratchpad/cuda_decode_ffn_down_llama_graph_adapter.cu").read_text()
  assert "i&15" in src and "in[i>>4]" in src
  assert "i<16*n" in src

def test_graph_ab_has_explicit_full_semantic_entries():
  assert FUSED["q4"]==ENTRIES["Q4_K"]
  assert FUSED["q6"]==ENTRIES["Q6_K"]
