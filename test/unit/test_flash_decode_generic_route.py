"""The live-split flash decode route takes the model's attention shape: the kernel's own validate admits it, a
measured tuning is a preference, and a refused shape has words, not a crash."""
from tinygrad.llm.decode_routes import FLASH_DECODE_CANDIDATE, FLASH_DECODE_G5_CANDIDATE, _FlashDecodeCandidate
from tinygrad.llm.flash_decode_attention import (FLASH_DECODE_G4, FLASH_DECODE_G5, FlashDecodeCapability, flash_decode_route_for,
                                                 flash_decode_route_refusal, llama_vec_wide_shape_refusal)

CAP = FlashDecodeCapability(True, True)


def _bind(route, shape):
  return _FlashDecodeCandidate(route).bind(1, *shape, "NV", capability=CAP) if route is not None else None


def test_the_measured_shapes_keep_their_tuned_route_and_binding():
  for tuned, old in ((FLASH_DECODE_G4, FLASH_DECODE_CANDIDATE), (FLASH_DECODE_G5, FLASH_DECODE_G5_CANDIDATE)):
    shape = (tuned.query_heads, tuned.kv_heads, tuned.head_dim)
    assert flash_decode_route_for(*shape) == tuned
    assert _bind(flash_decode_route_for(*shape), shape) == old.bind(1, *shape, "NV", capability=CAP)


def test_any_shape_the_kernel_accepts_gets_the_default_geometry():
  for shape in ((16, 8, 128), (24, 8, 128), (14, 2, 64), (8, 8, 64)):
    route = flash_decode_route_for(*shape)
    assert route is not None and (route.query_heads, route.kv_heads, route.head_dim) == shape
    assert (route.split_size, route.query_group_size, route.stage_width) == \
      (FLASH_DECODE_G4.split_size, FLASH_DECODE_G4.query_group_size, FLASH_DECODE_G4.stage_width)
    assert _bind(route, shape) is not None and not _bind(route, shape).llama_vec_wide


def test_a_refused_shape_says_why_in_the_kernels_words():
  assert flash_decode_route_for(32, 8, 96) is None
  assert "Hd must be divisible by lane_width*dot_pair_width" in flash_decode_route_refusal(32, 8, 96)
  assert flash_decode_route_for(30, 8, 128) is None and "divisible by Hkv" in flash_decode_route_refusal(30, 8, 128)
  assert flash_decode_route_refusal(16, 8, 128) is None


def test_the_wide_substrate_states_its_own_shape():
  assert llama_vec_wide_shape_refusal(FLASH_DECODE_G4.query_heads, FLASH_DECODE_G4.kv_heads, FLASH_DECODE_G4.head_dim) is None
  assert llama_vec_wide_shape_refusal(16, 8, 128)
