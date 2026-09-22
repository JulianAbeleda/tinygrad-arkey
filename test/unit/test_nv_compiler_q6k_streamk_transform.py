from pathlib import Path
from extra.llm_research.prefill.nv_compiler_q6k_streamk_transform import (active_fixup_source, transform_compiler_q6k_to_streamk,
  transform_compiler_q6k_wide_to_streamk, wide_active_fixup_source)

FIXTURE=Path("/home/ubuntu/boltbeam-runs/packed-q8-completion-20260830/q6-baseline/down.cu")

def test_q6_transform_preserves_imma_body_and_owns_streamk_contract():
  if not FIXTURE.exists(): return
  original=FIXTURE.read_text(); transformed=transform_compiler_q6k_to_streamk(original,unroll=8)
  assert transformed.count("mma.sync.aligned.m16n8k32") == original.count("mma.sync.aligned.m16n8k32")
  assert "q6k_imma_stream(" in transformed and "Ridx0 = k_begin" in transformed
  assert "partials+(slot*2048)" in transformed and "owner*2+((owner_tail&&owner_has_head)?1:0)" in transformed
  assert "tile=active[blockIdx.x]" in active_fixup_source()

def test_q6_wide_transform_owns_128_square_partial_layout():
  fixture=Path("/tmp/q6wide-down2/g_128_128_2_4_256.cu")
  if not fixture.exists(): return
  transformed=transform_compiler_q6k_wide_to_streamk(fixture.read_text(),unroll=2)
  assert "partials+(slot*16384)" in transformed and "gidx0=tile%32,gidx1=tile/32" in transformed
  assert "z<16384" in wide_active_fixup_source()
