import contextlib, pathlib, re, signal, subprocess, tempfile

import pytest

from tinygrad import dtypes
from tinygrad.codegen import full_rewrite_to_sink, to_program
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import KernelInfo, Ops, ParamArg, UOp


TARGET=Target.parse("NV:CUDA:sm_120")
CONTROL_SHA256="ccda2dfe85bdc9d964076a0aa0de7e10a929730af12b1d027a510277fe7ade1e"
NVDISASM=pathlib.Path(".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm")
INSN_RE=re.compile(r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\b",re.MULTILINE)


@contextlib.contextmanager
def _timeout(seconds:int):
  def expired(_signum,_frame): raise TimeoutError(f"const_restrict microgate exceeded {seconds}s")
  old=signal.signal(signal.SIGALRM,expired); signal.setitimer(signal.ITIMER_REAL,seconds)
  try: yield
  finally: signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old)


def _ast(qualified:bool=False, write_source:bool=False, second_qualified:bool=False):
  lid=UOp.special(32,"lidx0")
  out=UOp.placeholder((32,),dtypes.uint32,0)
  source=UOp.placeholder((32,),dtypes.uint32,1)
  if qualified: source=source.const_restrict()
  roots=[out[lid].store(source[lid].load())]
  if write_source: roots.append(source[lid].store(UOp.const(dtypes.uint32,0)))
  if second_qualified:
    other=UOp.placeholder((32,),dtypes.uint32,2).const_restrict()
    roots.append(out[lid+1].store(other[lid].load()))
  return UOp.sink(*roots,arg=KernelInfo(name="const_restrict_control",opts_to_apply=()))


def _source(qualified:bool=False) -> str:
  program=to_program(_ast(qualified),CUDARenderer(TARGET))
  return next(x.arg for x in program.src if x.op is Ops.SOURCE)


def test_const_restrict_api_and_source_are_explicit():
  source=UOp.placeholder((32,),dtypes.uint32,1).const_restrict()
  assert source.op is Ops.PARAM and isinstance(source.arg,ParamArg) and source.arg.const_restrict
  rendered=_source(True)
  assert "const unsigned int *__restrict__ data1_32" in rendered
  assert "unsigned int* data0_32" in rendered and "const unsigned int *__restrict__ data0_32" not in rendered


def test_unmarked_source_is_byte_identical():
  import hashlib
  rendered=_source(False)
  assert hashlib.sha256(rendered.encode()).hexdigest() == CONTROL_SHA256
  assert "const " not in rendered and "__restrict__" not in rendered


def test_const_restrict_api_rejects_wrong_owner_space_type_and_duplicate():
  with pytest.raises(ValueError,match="pointer PARAM owner"): UOp.const(dtypes.uint32,0).const_restrict()
  with pytest.raises(ValueError,match="pointer PARAM owner"): UOp.param(4,dtypes.uint32).const_restrict()
  with pytest.raises(ValueError,match="GLOBAL pointer PARAM"):
    UOp(Ops.PARAM,dtypes.uint32.ptr(32,AddrSpace.LOCAL),arg=ParamArg(5,addrspace=AddrSpace.LOCAL)).const_restrict()
  with pytest.raises(ValueError,match="scalar 32-bit pointee"): UOp.placeholder((32,),dtypes.half,6).const_restrict()
  owner=UOp.placeholder((32,),dtypes.uint32,7).const_restrict()
  with pytest.raises(ValueError,match="already const_restrict"): owner.const_restrict()


def test_const_restrict_fails_closed_for_writes_multiple_or_unsupported_renderer():
  with pytest.raises(RuntimeError,match="written in this kernel"):
    full_rewrite_to_sink(_ast(True,write_source=True),CUDARenderer(TARGET))
  with pytest.raises(RuntimeError,match="exactly one annotated"):
    full_rewrite_to_sink(_ast(True,second_qualified=True),CUDARenderer(TARGET))
  renderer=CUDARenderer(TARGET); renderer.supports_const_restrict_pointer=False
  with pytest.raises(RuntimeError,match="cannot preserve const_restrict"):
    full_rewrite_to_sink(_ast(True),renderer)


def test_const_restrict_store_preflight_uses_destination_pointer_lineage_only():
  lid=UOp.special(32,"lidx0")
  source=UOp.placeholder((32,),dtypes.uint32,1).const_restrict()
  loaded=source[lid].load()
  local=UOp(Ops.DEFINE_LOCAL,dtypes.uint32.ptr(32,AddrSpace.LOCAL),arg=41)
  # Model publication ordering by carrying the loaded value beside the LOCAL base.
  # It is a dependency, not an alias owner of the STORE destination.
  published_local=UOp(Ops.AFTER,local.dtype,(local,loaded))
  publication=published_local[lid].store(loaded)
  ast=UOp.sink(publication,arg=KernelInfo(name="const_restrict_local_publication",opts_to_apply=()))
  program=to_program(ast,CUDARenderer(TARGET))
  rendered=next(x.arg for x in program.src if x.op is Ops.SOURCE)
  assert "const unsigned int *__restrict__ data1_32" in rendered


def test_const_restrict_rejects_direct_and_derived_alias_writes():
  source=UOp.placeholder((32,),dtypes.uint32,1).const_restrict()
  zero=UOp.const(dtypes.uint32,0)
  direct=UOp(Ops.STORE,dtypes.void,(source,zero))
  with pytest.raises(RuntimeError,match="written in this kernel"):
    full_rewrite_to_sink(UOp.sink(direct,arg=KernelInfo(name="const_restrict_direct_write",opts_to_apply=())),CUDARenderer(TARGET))

  cast_alias=UOp(Ops.CAST,source.dtype,(source,))
  derived=cast_alias[UOp.const(dtypes.int,1)].store(zero)
  with pytest.raises(RuntimeError,match="written in this kernel"):
    full_rewrite_to_sink(UOp.sink(derived,arg=KernelInfo(name="const_restrict_derived_write",opts_to_apply=())),CUDARenderer(TARGET))


def test_const_restrict_rejects_ambiguous_destination_pointer_base():
  source=UOp.placeholder((32,),dtypes.uint32,1).const_restrict()
  other=UOp.placeholder((32,),dtypes.uint32,2)
  ambiguous=UOp(Ops.NOOP,source.dtype,(source,other))
  store=UOp(Ops.STORE,dtypes.void,(ambiguous,UOp.const(dtypes.uint32,0)))
  with pytest.raises(RuntimeError,match="ambiguous PARAM ownership"):
    full_rewrite_to_sink(UOp.sink(store,arg=KernelInfo(name="const_restrict_ambiguous_write",opts_to_apply=())),CUDARenderer(TARGET))


@pytest.mark.skipif(not NVDISASM.is_file(),reason="bundled nvdisasm unavailable")
def test_const_restrict_changes_one_nv_load_to_constant_without_extra_loads():
  with _timeout(30):
    opcodes=[]
    for qualified in (False,True):
      binary=NVRTCCompiler("sm_120",ptx=False,cache_key=f"const_restrict_pointer_{int(qualified)}").compile(_source(qualified))
      with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(binary); f.flush()
        sass=subprocess.run([str(NVDISASM),"-c",f.name],check=True,text=True,stdout=subprocess.PIPE,timeout=15).stdout
      opcodes.append([x for x in INSN_RE.findall(sass) if x.startswith("LDG")])
  assert opcodes == [["LDG.E"],["LDG.E.CONSTANT"]]
