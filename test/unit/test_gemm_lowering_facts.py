"""tinygrad exports what its precontract tile lowering can emit (boltbeam.gemm_lowering_facts.v1); BoltBeam derives the
dense bf16 search space from those facts and the GPU's.  The hand caps are gone: every limit reads the renderer."""
import json
from pathlib import Path

import pytest

from extra.llm_research.gemm_lowering_facts import lowering_facts
from tinygrad.renderer.cuda import CUDARenderer

ROOT = Path(__file__).resolve().parents[2]


def test_lowering_facts_read_the_renderer_and_descriptor():
  facts = lowering_facts("NV", "sm_120")
  assert facts["schema"] == "boltbeam.gemm_lowering_facts.v1"
  assert (facts["mma_m"], facts["mma_n"], facts["mma_k"]) == (16, 8, 16)            # mma.sync m16n8k16
  assert (facts["a_fragment_regs"], facts["b_fragment_regs"], facts["accumulator_regs"]) == (4, 2, 4)
  assert facts["static_lds_bytes"] == CUDARenderer.max_static_local_bytes
  assert facts["runtime_lds_bytes"] == CUDARenderer.max_runtime_local_bytes
  assert facts["async_copy"] is True and CUDARenderer.async_copy_ops is not None
  assert facts["matrix_fragments"] is True and 128 in facts["swizzle_row_bytes"]
  assert facts["ragged_split_k"] is False and facts["stream_k"] is False


def test_lowering_facts_follow_the_renderer(monkeypatch):
  monkeypatch.setattr(CUDARenderer, "async_copy_ops", None)
  monkeypatch.setattr(CUDARenderer, "max_runtime_local_bytes", 65536)
  facts = lowering_facts("NV", "sm_120")
  assert facts["async_copy"] is False and facts["runtime_lds_bytes"] == 65536


def test_nv_capability_rows_take_the_static_lds_cap_from_the_renderer():
  import extra.llm_research.runtime_specs as rs
  rows = [row for (backend, _), table in rs._CAPABILITY_ROWS.items() if backend in ("NV", "CUDA")
          for name, row in table.items() if name != "async_ring_matrix"]
  assert rows and all(row.max_lds_bytes == CUDARenderer.max_static_local_bytes for row in rows)
  assert "max_lds_bytes=49152" not in (ROOT / "extra/llm_research/runtime_specs.py").read_text()


def test_the_dense_candidate_set_re_mints_byte_identically():
  from extra.llm_research.mint_typed_candidate_template import _DENSE_BF16_ARTIFACT, mint_dense_bf16
  assert json.dumps(mint_dense_bf16(), indent=2, sort_keys=True) + "\n" == _DENSE_BF16_ARTIFACT.read_text()


def _search():
  import extra.llm_research.prefill.dense_bf16_geometry_search as search
  try: search._strategy()
  except ImportError as exc: pytest.skip(f"BoltBeam checkout unavailable: {exc}")
  return search


def test_every_promoted_route_is_in_the_derived_search_space():
  search = _search()
  rows = json.loads(search.SELECTION.read_text())["rows"]
  assert len(rows) == 48
  for row in rows:
    pipe = tuple(row.get("pipeline", search.SYNC2))
    assert search.feasible(tuple(row["geometry"]), row["split_k"], row["m"], row["n"], row["k"], pipe), row


def test_climb_neighbours_move_one_axis_inside_the_derived_space():
  search = _search()
  config = ((64, 128, 64, 2, 2), 1, (3, True, True, True))
  neighbours = search.ring_neighbors(config, 64, 17536, 3136)
  assert ((64, 128, 64, 2, 2), 1, (4, True, True, True)) in neighbours
  assert all(search.feasible(*c[:2], 64, 17536, 3136, c[2]) for c in neighbours)
  moved = [sum(a != b for a, b in zip(c[0] + (c[1],) + c[2][:1], config[0] + (config[1],) + config[2][:1])) for c in neighbours]
  assert moved and all(x == 1 or (x == 2 and c[0][:3] == config[0][:3]) for x, c in zip(moved, neighbours))


def test_production_tinygrad_imports_neither_boltbeam_nor_the_exporter():
  """Dependency direction: tinygrad/ compiles and exports; the strategy (BoltBeam) is research-side only."""
  offenders = [str(p.relative_to(ROOT)) for p in (ROOT / "tinygrad").rglob("*.py")
               if any(s in p.read_text() for s in ("import boltbeam", "from boltbeam", "gemm_lowering_facts"))]
  assert offenders == []
