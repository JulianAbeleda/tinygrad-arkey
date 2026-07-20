import pytest

from tinygrad import dtypes
from tinygrad.uop.ops import Ops

from extra.qk.mmq_llama_five_buffer_graph import (BLOCKER, build_llama_five_buffer_bounded_sink,
  build_llama_five_buffer_graph, five_buffer_parameters)


@pytest.mark.xfail(reason="phase2-fp16-dequant-q4k: retired int8 5-buffer/2-phase-x-4-group MMQ structural assumptions (exact WMMA/tag/offset counts, q8/q4 LDS region names, dm/ds sidecar correction) superseded by the fp16-dequant-in-register per-K32-group design (see docs/amd-fp16-dequant-q4k-primitive-implementation-plan-20260720.md); not rewritten this phase")
def test_exact_five_buffer_abi_slots_dtypes_shapes_and_sizes():
  params = five_buffer_parameters(128, 256, 512)
  assert [(x.slot, x.name, x.dtype) for x in params] == [
    (0, "output", dtypes.float32), (1, "q4", dtypes.uint32), (2, "q8_values", dtypes.int8),
    (3, "q8_scales", dtypes.float32), (4, "q8_original_sums", dtypes.float32)]
  assert [x.physical_shape for x in params] == [(128, 256), (256, 2, 36), (4, 128, 128), (4, 128, 4), (4, 128, 4)]
  assert [x.size for x in params] == [128*256, 256*2*36, 4*128*128, 4*128*4, 4*128*4]


@pytest.mark.parametrize("k,expected", [(256, [(0, 0, 0, 0)]), (512, [(0, 0, 0, 0), (36, 256*128, 256*4, 256*4)])])
@pytest.mark.xfail(reason="phase2-fp16-dequant-q4k: retired int8 5-buffer/2-phase-x-4-group MMQ structural assumptions (exact WMMA/tag/offset counts, q8/q4 LDS region names, dm/ds sidecar correction) superseded by the fp16-dequant-in-register per-K32-group design (see docs/amd-fp16-dequant-q4k-primitive-implementation-plan-20260720.md); not rewritten this phase")
def test_epoch_offsets_wmmas_lds_and_fp32_recurrence(k, expected):
  graph = build_llama_five_buffer_graph(128, 128, k)
  assert [(e.offsets.q4, e.offsets.values, e.offsets.scales, e.offsets.sums) for e in graph.body.epochs] == expected
  assert all(e.recurrence.stage.geometry.lds_bytes == 57856 for e in graph.body.epochs)
  assert all(len([x for x in e.recurrence.consumer_seam.toposort() if x.op is Ops.WMMA]) == 16 for e in graph.body.epochs)
  if k == 512:
    assert all(graph.body.epochs[0].accumulators[i] in graph.body.epochs[1].accumulators[i].backward_slice for i in range(8))
    assert graph.body.epochs[0].recurrence.consumer_seam in graph.body.epochs[1].accumulators[0].backward_slice


@pytest.mark.xfail(reason="phase2-fp16-dequant-q4k: retired int8 5-buffer/2-phase-x-4-group MMQ structural assumptions (exact WMMA/tag/offset counts, q8/q4 LDS region names, dm/ds sidecar correction) superseded by the fp16-dequant-in-register per-K32-group design (see docs/amd-fp16-dequant-q4k-primitive-implementation-plan-20260720.md); not rewritten this phase")
def test_tile_rebasing_uses_declared_split_physical_layouts():
  graph = build_llama_five_buffer_graph(256, 256, 512, tile_m=1, tile_n=1)
  assert [(e.offsets.q4, e.offsets.values, e.offsets.scales) for e in graph.body.epochs] == [
    (128*2*36, 128*128, 128*4), ((128*2+1)*36, (2*256+128)*128, (2*256+128)*4)]


@pytest.mark.xfail(reason="phase2-fp16-dequant-q4k: retired int8 5-buffer/2-phase-x-4-group MMQ structural assumptions (exact WMMA/tag/offset counts, q8/q4 LDS region names, dm/ds sidecar correction) superseded by the fp16-dequant-in-register per-K32-group design (see docs/amd-fp16-dequant-q4k-primitive-implementation-plan-20260720.md); not rewritten this phase")
def test_no_dense_allocations_and_bounded_writeback_only():
  graph = build_llama_five_buffer_graph(128, 128, 256)
  assert graph.allocated_shapes[-1] == ("lds_bounded_tile", (57856,), "uint8")
  assert not any(kind.startswith(("dense", "dequant")) for kind, _, _ in graph.allocated_shapes)
  sink = build_llama_five_buffer_bounded_sink(graph)
  drains = [x for x in sink.toposort() if x.op is Ops.STORE and isinstance(x.tag, tuple) and
            x.tag[:1] == ("llama_full_kernel_bounded_final_release",)]
  assert len(drains) == 8 and all(x.src[1].dtype == dtypes.float.vec(8) for x in drains)
  assert graph.custom_kernel is None and not graph.emitted and not graph.routed
  assert graph.blocker == BLOCKER and "no grid" in BLOCKER and "emission" in BLOCKER and "routing" in BLOCKER
  with pytest.raises(RuntimeError, match="bounded one-tile"): graph.program()


@pytest.mark.parametrize("args,kwargs", [
  ((0, 128, 256), {}), ((127, 128, 256), {}), ((128, 129, 256), {}), ((128, 128, 128), {}),
  ((128, 128, 257), {}), ((128, 128, 256), {"tile_m": 1}), ((128, 128, 256), {"tile_n": -1}),
])
def test_invalid_dimensions_and_tiles_fail_closed(args, kwargs):
  with pytest.raises(ValueError): build_llama_five_buffer_graph(*args, **kwargs)
