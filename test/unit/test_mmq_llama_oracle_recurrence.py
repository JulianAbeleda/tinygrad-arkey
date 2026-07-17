from dataclasses import replace

from tinygrad import dtypes
import extra.qk.kernel_lds as lds
from tinygrad.uop.ops import Ops, UOp

import test.unit.test_hierarchical_packed_record_stage as stage_fixture
from extra.qk.mmq_llama_oracle_recurrence import (LLAMA_SOURCE_COMMIT, build_llama_oracle_recurrence,
                                                  prove_llama_oracle_recurrence)
from extra.qk.mmq_llama_record_producers import RecordProducerInstanceWitness, record_producer_instance_value


def _schedule(template, threads, source_k):
  thread = (threads.wave_m+threads.wave_n)*32+threads.lane
  out = []
  for binding in template.fields:
    field = template.transform.produced.component(binding.field)
    vectors_per_row, width = field.size_bytes//binding.vector_bytes, binding.vector_bytes//field.dtype.itemsize
    sources = tuple(template.source(x) for x in binding.sources)
    for iteration in range(128*vectors_per_row//256):
      linear = thread+iteration*256
      row, vector = linear//vectors_per_row, linear%vectors_per_row
      k = UOp.const(dtypes.weakint, source_k)+vector*width
      value = binding.producer(sources, row, k, width)
      if template.role == "B":
        value = record_producer_instance_value(value, RecordProducerInstanceWitness(
          "llama-q8-ds4-producer-instance.v1", "B", binding.field, source_k//128, iteration, iteration,
          row, k, row, vector))
      out.append(lds.PackedRecordCooperativeStore(binding.field, iteration, row, k, row, vector, value))
  return tuple(out)


def _graph(monkeypatch):
  original = stage_fixture.build_hierarchical_packed_record_stage
  def scheduled(*args, **kwargs):
    schedule = lds.PackedRecordCooperativeSchedule("oracle-test-exact-cover", _schedule, ("wave_m", "wave_n", "lane"))
    kwargs["templates"] = tuple(replace(x, cooperative_schedule=schedule) for x in kwargs["templates"])
    return original(*args, **kwargs)
  monkeypatch.setattr(stage_fixture, "build_hierarchical_packed_record_stage", scheduled)
  return build_llama_oracle_recurrence(stage_fixture._fixture()[3])


def _replace_group(graph, ordinal, mutation):
  pi, gi = divmod(ordinal, 4)
  phase = graph.phases[pi]
  groups = phase.groups[:gi]+(mutation(phase.groups[gi]),)+phase.groups[gi+1:]
  return replace(graph, phases=graph.phases[:pi]+(replace(phase, groups=groups),)+graph.phases[pi+1:])


def test_exact_k256_source_pinned_recurrence_and_dependencies(monkeypatch):
  graph = _graph(monkeypatch)
  assert LLAMA_SOURCE_COMMIT == "ac4cddeb0dbd778f650bf568f6f08344a06abe3a"
  assert [x.k for x in graph.groups] == list(range(0, 256, 32))
  assert [[x.wmmas[i].tag[-1] for i in range(2)] for x in graph.groups] == [[k, k+16] for k in range(0, 256, 32)]
  assert all(x.zero.dtype == dtypes.int.vec(8) and x.zero.op is Ops.CONST and x.zero.arg == 0 for x in graph.groups)
  assert len({id(x.zero) for x in graph.groups}) == 8
  for phase in graph.phases:
    assert all(x in phase.release.backward_slice for x in phase.groups[3].update)
  assert len(graph.initial) == 8 and all(x.dtype == dtypes.float for x in graph.initial)
  assert all(len(x.update) == 8 and all(y.dtype == dtypes.float for y in x.update) for x in graph.groups)
  assert all(graph.stage.subtile_n in x.ranges for x in graph.groups[-1].update)
  assert not any(x.op is Ops.STACK and x.dtype == dtypes.float.vec(8) for x in graph.consumer_seam.toposort())
  assert graph.phases[0].release in graph.phases[1].producer.backward_slice
  assert graph.phases[1].publish in graph.phases[1].groups[0].wmmas[0].backward_slice
  assert graph.phases[1].release in graph.consumer_seam.backward_slice
  assert prove_llama_oracle_recurrence(graph).passed


def test_q4_dm_follows_each_j_major_c_element_and_q8_ds_stays_lane_owned(monkeypatch):
  graph = _graph(monkeypatch)
  stage = graph.stage
  persistent, overwriteable = stage.descriptor.plan.persistent.name, stage.descriptor.plan.overwriteable.name
  region_names = {x.role:x.region for x in stage.regions}
  q4, q8 = stage.geometry.lds_region(region_names[persistent]), stage.geometry.lds_region(region_names[overwriteable])
  assert q4.records is not None and q8.records is not None
  dm_field = q4.records.component(next(x.field for x in stage.groups[0].sidecars if x.role == persistent))
  ds_field = q8.records.component(next(x.field for x in stage.groups[0].sidecars if x.role == overwriteable))
  wave_m, wave_n, lane = stage.threads.wave_m, stage.threads.wave_n, stage.threads.lane
  subtile_m, subtile_n = stage.subtile_m, stage.subtile_n
  sm = stage.geometry.tile[0] // (stage.geometry.waves[0]*16)
  sn = stage.geometry.tile[1] // (stage.geometry.waves[1]*16)

  # Every wave/lane/l combination must load Q4 metadata from tile_C::get_i(l),
  # not from the fragment row (lane%16).
  rec = graph.groups[0]
  for wm in range(stage.geometry.waves[0]):
    for ln in range(stage.geometry.wave_size):
      replacements = {wave_m:UOp.const(dtypes.weakint, wm), wave_n:UOp.const(dtypes.weakint, 0),
                      lane:UOp.const(dtypes.weakint, ln), subtile_m:UOp.const(dtypes.weakint, 0)}
      for element, dm in enumerate(rec.dm):
        address = dm.src[0].src[0].src[1].substitute(replacements).simplify()
        row = (wm*sm)*16 + ln//16 + 2*element
        expected = q4.base + dm_field.offset_bytes + row*q4.records.stride_bytes
        assert address.op is Ops.CONST and address.arg == expected

  # Group offsets advance one half2 while row ownership remains unchanged.
  for rec in graph.groups:
    offset = rec.k*dm_field.size_bytes//stage.descriptor.outer_k
    address = rec.dm[0].src[0].src[0].src[1].substitute({
      wave_m:UOp.const(dtypes.weakint, 0), wave_n:UOp.const(dtypes.weakint, 0),
      lane:UOp.const(dtypes.weakint, 0), subtile_m:UOp.const(dtypes.weakint, 0)}).simplify()
    assert address.arg == q4.base + dm_field.offset_bytes + offset

  # Q8 ds is one shared half2 per group/subtile/lane and must not acquire an
  # output-element dependency from the Q4 correction.
  assert all(len(rec.dm) == 8 and len({id(x) for x in rec.dm}) == 8 for rec in graph.groups)
  for st in range(sn):
    for ln in range(stage.geometry.wave_size):
      address = graph.groups[0].ds.src[0].src[0].src[1].substitute({
        wave_m:UOp.const(dtypes.weakint, 0), wave_n:UOp.const(dtypes.weakint, 0),
        lane:UOp.const(dtypes.weakint, ln), subtile_n:UOp.const(dtypes.weakint, st)}).simplify()
      expected = q8.base + ds_field.offset_bytes + (st*16+ln%16)*q8.records.stride_bytes
      assert address.op is Ops.CONST and address.arg == expected


def test_fragment_group_substep_and_sidecar_mutations_fail(monkeypatch):
  graph = _graph(monkeypatch)
  bad_fragment = _replace_group(graph, 0, lambda x: replace(x, fragments=((x.fragments[0][1], x.fragments[0][0]), x.fragments[1])))
  bad_group = _replace_group(graph, 2, lambda x: replace(x, group=7))
  bad_substep = _replace_group(graph, 3, lambda x: replace(x, wmmas=(x.wmmas[1], x.wmmas[0])))
  bad_sidecar = _replace_group(graph, 4, lambda x: replace(x, dm=x.ds))
  for candidate in (bad_fragment, bad_group, bad_substep, bad_sidecar):
    assert not prove_llama_oracle_recurrence(candidate).passed


def test_algebra_and_barrier_wiring_mutations_fail(monkeypatch):
  graph = _graph(monkeypatch)
  rec = graph.groups[5]
  wrong_algebra = _replace_group(graph, 5, lambda x: replace(x, update=x.previous[:-1]+(x.wmmas[1].gep(0).cast(dtypes.float),)))
  assert not prove_llama_oracle_recurrence(wrong_algebra).passed

  p0 = graph.phases[0]
  detached_release = UOp(Ops.BARRIER, dtypes.void, p0.groups[2].update)
  assert not prove_llama_oracle_recurrence(replace(graph, phases=(replace(p0, release=detached_release), graph.phases[1]))).passed
  # Recreate the rejected construction: metadata wrapper depends on the new release, while the publish and WMMAs still
  # consume the old stage producer/publish path.  The connected-graph proof must reject it.
  p1 = graph.phases[1]
  detached_producer = graph.stage.phases[1].producer.after(p0.release)
  old_publish_group = replace(p1, producer=detached_producer, publish=graph.stage.phases[1].publish)
  assert not prove_llama_oracle_recurrence(replace(graph, phases=(p0, old_publish_group))).passed

  detached_final = replace(graph, consumer_seam=UOp(Ops.BARRIER, dtypes.void, graph.groups[-1].update))
  assert not prove_llama_oracle_recurrence(detached_final).passed


def test_descriptor_renderer_signed_contract_and_fresh_seed_fail_closed(monkeypatch):
  graph = _graph(monkeypatch)
  shared = _replace_group(graph, 1, lambda x: replace(x, zero=graph.groups[0].zero))
  assert not prove_llama_oracle_recurrence(shared).passed
  unsigned_tc = replace(graph.stage.tc, dtype_in=dtypes.uchar)
  assert not prove_llama_oracle_recurrence(replace(graph, stage=replace(graph.stage, tc=unsigned_tc))).passed
