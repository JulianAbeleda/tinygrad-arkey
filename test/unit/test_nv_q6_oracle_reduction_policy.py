import inspect
import numpy as np

from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.prefill.nv_q6_oracle_reduction_policy import (
  M, N, OWNERS, ROWS, COLS, TILES, TILE_ELEMS, SEGMENT_ORDER_BUILDER_PATCH,
  ast_signature, build_packed_one_body_ast, build_reduction_schedule, direct_final_writeback,
  emit_ordered_fixup, fold_values, supports_nonfinal_first)


def test_reduction_schedule_freezes_llama_fold_order_and_plane_major_storage():
  schedule=build_reduction_schedule()
  assert len(schedule.records)==294
  assert len(schedule.predecessor_slots)==166
  assert len(schedule.final_slots)==128
  assert sorted(map(len,schedule.ordered_by_tile)).count(2)==90
  assert sorted(map(len,schedule.ordered_by_tile)).count(3)==38
  assert all(row[-1].is_final and all(not x.is_final for x in row[:-1]) for row in schedule.ordered_by_tile)
  assert all(tuple(x.owner for x in row[:-1])==tuple(sorted((x.owner for x in row[:-1]),reverse=True))
             for row in schedule.ordered_by_tile)
  assert all(x.slot==x.plane*OWNERS+x.owner for x in schedule.records)
  three=[row for row in schedule.ordered_by_tile if len(row)==3]
  assert len(three)==38
  assert all(sorted(x.slot for x in row).index(row[-1].slot)==1 for row in three)
  slots,counts,final_indices=schedule.arrays()
  assert slots.shape==(TILES,3) and counts.shape==(TILES,) and final_indices.shape==(TILES,)
  assert np.array_equal(final_indices,counts-1)
  assert np.all((slots>=0)&(slots<2*OWNERS))


def test_fold_values_is_explicit_sequential_fp32():
  values=(np.array([1.0e20,-0.0],np.float32),np.array([-1.0e20,0.0],np.float32),np.array([3.25,-0.0],np.float32))
  got=fold_values(values)
  want=np.zeros((2,),np.float32)
  for value in values:np.add(want,value,out=want)
  assert np.array_equal(got.view(np.uint32),want.view(np.uint32))


def test_direct_final_adapter_changes_only_terminal_writeback_shape():
  base=build_packed_one_body_ast("ascending")
  candidate=direct_final_writeback(base,UOp.placeholder((M*N,),dtypes.float32,3),name="q6_reduction_test")
  before,after=ast_signature(base),ast_signature(candidate)
  assert before["wmma"]==after["wmma"]==256
  assert before["barriers"]==after["barriers"]==4
  assert before["ranges"]==after["ranges"]
  assert before["terminal_store_count"]==64 and after["terminal_store_count"]==128
  assert before["parameter_store_counts"][0]==64
  assert after["parameter_store_counts"][0]==64 and after["parameter_store_counts"][3]==64


def test_ordered_fixups_are_bounded_and_have_explicit_fadd_chains():
  p=lambda n,t,i:UOp.placeholder((n,),t,i)
  all_ast=emit_ordered_fixup(p(M*N,dtypes.float32,0),p(2*OWNERS*TILE_ELEMS,dtypes.float32,1),
    p(TILES*3,dtypes.int32,2),p(TILES,dtypes.int32,3))
  direct_ast=emit_ordered_fixup(p(M*N,dtypes.float32,0),p(2*OWNERS*TILE_ELEMS,dtypes.float32,1),
    p(TILES*3,dtypes.int32,2),p(TILES,dtypes.int32,3),p(TILES,dtypes.int32,4),direct_final=True)
  for ast in (all_ast,direct_ast):
    topo=ast.toposort()
    assert sum(x.op is Ops.STORE for x in topo)==64
    assert sum(x.op is Ops.CUSTOMI and x.arg=="__fadd_rn({0},{1})" for x in topo)==192
    assert sum(x.op is Ops.SPECIAL for x in topo)==2
  assert sum(x.op is Ops.LOAD for x in direct_ast.toposort())>sum(x.op is Ops.LOAD for x in all_ast.toposort())


def test_nonfinal_first_is_exposed_and_preserves_the_physical_kernel_shape():
  from extra.llm_research.prefill.nv_q6_oracle_broad_cta import q6_oracle_broad_cta_kernel
  exposed="streamk_segment_order" in inspect.signature(q6_oracle_broad_cta_kernel).parameters
  assert exposed and supports_nonfinal_first()
  ascending=build_packed_one_body_ast("ascending")
  nonfinal_first=build_packed_one_body_ast("nonfinal_first")
  ascending_signature,nonfinal_signature=ast_signature(ascending),ast_signature(nonfinal_first)
  for field in ("wmma","barriers","ranges","terminal_store_count","parameter_store_counts"):
    assert ascending_signature[field]==nonfinal_signature[field]
  assert "nonfinal_first" not in ascending.arg.name
  assert "nonfinal_first" in nonfinal_first.arg.name
