from pathlib import Path

from extra.llm_research.prefill.nv_compiler_q4k_streamk_transform import active_fixup_source, transform_compiler_q4k_to_streamk

FIXTURE=Path("/home/ubuntu/boltbeam-runs/packed-q8-completion-20260830/q4-emitter-baseline.cu")

def test_transform_preserves_imma_body_and_adds_exact_owner_workspace_contract():
  if not FIXTURE.exists(): return
  original=FIXTURE.read_text()
  transformed=transform_compiler_q4k_to_streamk(original)
  assert transformed.count("mma.sync.aligned.m16n8k32") == original.count("mma.sync.aligned.m16n8k32")
  assert 'q4k_imma_stream(' in transformed
  assert "blockIdx.x; /* 170 persistent owners */" in transformed
  assert "for (int Ridx0 = k_begin; Ridx0 < k_end; Ridx0++)" in transformed
  assert "partial_ids[slot]=tile" in transformed
  assert "partial_ids[owner*2]=-1" in transformed
  assert "partials+(slot*16384)" in transformed
  assert "owner*2+((owner_tail&&owner_has_head)?1:0)" in transformed

def test_active_fixup_has_no_empty_tile_launches():
  source=active_fixup_source()
  assert "tile=active[blockIdx.x]" in source and "if(s0<0)return" not in source

def test_transform_owns_qualified_unroll_choice():
  if not FIXTURE.exists(): return
  transformed=transform_compiler_q4k_to_streamk(FIXTURE.read_text(),unroll=8)
  assert "#pragma unroll 8\n  for (int Ridx0 = k_begin; Ridx0 < k_end; Ridx0++)" in transformed
