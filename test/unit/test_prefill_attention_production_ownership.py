import ast
from pathlib import Path

import pytest

from tinygrad.llm import fused_attention
from tinygrad.schedule.wmma.flash_prefill import FlashPrefillAttentionSpec


ROOT = Path(__file__).parents[2]
PRODUCTION_CLOSURE = (
  "tinygrad/llm/fused_attention.py",
  "tinygrad/llm/flash_prefill_attention.py",
  "tinygrad/schedule/wmma/flash_prefill.py",
  "tinygrad/schedule/wmma/__init__.py",
  "tinygrad/schedule/wmma/kernels.py",
  "tinygrad/renderer/isa/amd.py",
)


class _AttentionTensor:
  def __init__(self, shape):
    self.shape = shape
    self.sdpa_calls = []

  def scaled_dot_product_attention(self, k, v, **kwargs):
    self.sdpa_calls.append((k, v, kwargs))
    return ("ordinary_sdpa", self.shape)


def _research_imports(path: Path) -> list[str]:
  imports = []
  for node in ast.walk(ast.parse(path.read_text())):
    names = ([node.module] if isinstance(node, ast.ImportFrom) else
             [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
    imports.extend(name for name in names if name and (name == "extra" or name.startswith("extra.llm_research")))
  return imports


def test_fused_prefill_production_closure_does_not_import_research_package():
  for relative in PRODUCTION_CLOSURE:
    assert _research_imports(ROOT / relative) == [], relative


@pytest.mark.parametrize("hq", (32, 40))
def test_both_promoted_qwen_grids_have_the_same_production_descriptor(hq):
  q, k = _AttentionTensor((1, hq, 512, 128)), _AttentionTensor((1, 8, 512, 128))
  grid = fused_attention.prefill_grid_spec(q, k)
  assert grid is not None
  assert (grid.q_heads, grid.kv_heads, grid.group_ratio, grid.q_tokens, grid.kv_tokens, grid.head_dim) == \
         (hq, 8, hq // 8, 512, 512, 128)

  descriptor = FlashPrefillAttentionSpec(Hq=hq, Hkv=8, Hd=128, q_tokens=512, kv_tokens=512,
    causal=True, scale=128 ** -0.5, valid_kv=512, query_start=0)
  assert descriptor.validate() is descriptor
  assert descriptor.target == "amd_gfx1100"
  assert descriptor.emitted_kernel_names == ("amd_gfx1100_q16_grid_hd128_loop_attention",)


@pytest.mark.parametrize("hq,use_custom", ((24, True), (32, False)))
def test_unselected_prefill_routes_use_ordinary_sdpa(monkeypatch, hq, use_custom):
  q, k, v = (_AttentionTensor((1, hq, 512, 128)), _AttentionTensor((1, 8, 512, 128)),
             _AttentionTensor((1, 8, 512, 128)))
  monkeypatch.setattr(fused_attention, "custom_kernel_attention",
                      lambda *args, **kwargs: pytest.fail("custom route must not run"))
  mask = object()
  result = fused_attention.route_prefill_attention(q, k, v, mask=mask, ctx=object(), use_custom_kernel=use_custom)
  assert result == ("ordinary_sdpa", q.shape)
  assert q.sdpa_calls == [(k, v, {"attn_mask": mask, "enable_gqa": True})]


def test_rejected_promoted_candidate_falls_back_to_ordinary_sdpa(monkeypatch):
  q, k, v = (_AttentionTensor((1, 40, 512, 128)), _AttentionTensor((1, 8, 512, 128)),
             _AttentionTensor((1, 8, 512, 128)))

  def reject(*args, **kwargs):
    raise NotImplementedError("descriptor rejected")

  monkeypatch.setattr(fused_attention, "custom_kernel_attention", reject)
  assert fused_attention.route_prefill_attention(q, k, v, ctx=object(), use_custom_kernel=True) == \
         ("ordinary_sdpa", q.shape)
  assert len(q.sdpa_calls) == 1
