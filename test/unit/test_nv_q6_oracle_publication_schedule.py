import collections
from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.prefill.bench_nv_q6_oracle_publication_gates import _classify_q8_panel1
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import ROWS,COLS,q6_oracle_broad_cta_kernel

def _kernel(*,prefetch_second_panel=True,combined_initial_publish=False):
  out=UOp.placeholder((ROWS*COLS,),dtypes.float32,0)
  blocks=UOp.placeholder((ROWS*105,),dtypes.uint16,1)
  q8=UOp.placeholder((2*COLS*36,),dtypes.uint32,2)
  ast=q6_oracle_broad_cta_kernel(out,blocks,q8,prefetch_second_panel=prefetch_second_panel,
    combined_initial_publish=combined_initial_publish,factor_dA=False,oracle_publisher=True,
    weight_scale_contract="trusted_fp16_packed")
  return ast,q8

def _pointer_root(node):
  while node.src: node=node.src[0]
  return node

def _panel1_load_wmma_counts(ast,q8):
  loads=[u for u in ast.toposort() if u.op is Ops.LOAD and _pointer_root(u.src[0]) is q8]
  return collections.Counter(sum(x.op is Ops.WMMA for x in u.toposort()) for u in loads)

def _op_count(ast,op): return sum(x.op is op for x in ast.toposort())

def test_admitted_early_panel1_preload_remains_before_half0():
  early,q8_early=_kernel(prefetch_second_panel=True)
  serial,q8_serial=_kernel(prefetch_second_panel=False)
  assert _panel1_load_wmma_counts(early,q8_early) == collections.Counter({0:18})
  assert _panel1_load_wmma_counts(serial,q8_serial) == collections.Counter({128:18})
  assert _op_count(early,Ops.BARRIER) == _op_count(serial,Ops.BARRIER) == 5

def test_combined_initial_publication_removes_only_one_ast_barrier():
  separate,q8_separate=_kernel(combined_initial_publish=False)
  combined,q8_combined=_kernel(combined_initial_publish=True)
  assert _op_count(separate,Ops.BARRIER) == 5
  assert _op_count(combined,Ops.BARRIER) == 4
  assert _op_count(separate,Ops.WMMA) == _op_count(combined,Ops.WMMA) == 256
  assert _panel1_load_wmma_counts(separate,q8_separate) == _panel1_load_wmma_counts(combined,q8_combined) == collections.Counter({0:18})

def test_panel1_sass_classifier_requires_two_exact_18_store_planes():
  lines=[];pc=0
  def add(op):
    nonlocal pc
    lines.append(f"        /*{pc:04x}*/                   {op}")
    pc+=0x10
  for i in range(18): add(f"LDG.E R{i}, desc[UR14][R2.64+0x{i*0x400:x}]")
  for i in range(18): add(f"STS [R12+0x{0x9800+i*0x400:x}], R{i}")
  for _ in range(10): add("NOP")
  for i in range(18): add(f"LDG.E R{i+32}, desc[UR14][R2.64+0x{0x4800+i*0x400:x}]")
  for _ in range(7): add("NOP")
  for i in range(18): add(f"STS [R40+0x{0x9800+i*0x400:x}], R{i+32}")
  got=_classify_q8_panel1("\n".join(lines))
  assert got["classified"]
  assert got["panel1_loads"] == got["panel1_stores"] == 18
  assert got["panel1_load_to_store_span_instructions"] == 25
