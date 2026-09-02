import pathlib, re, shutil, subprocess, tempfile
import pytest

from tinygrad import dtypes
from tinygrad.codegen import full_rewrite_to_sink, to_program
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import KernelInfo, Ops, RuntimeLocalAllocation, StrictAfter, UOp
from tinygrad.uop.symbolic import _commutative_key

TARGET=Target.parse("NV:CUDA:sm_120")
INSN_RE=re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b", re.MULTILINE)


def _ast(strict:bool, live_dependency:bool=False):
  lid=UOp.special(32, "lidx0")
  out=UOp.placeholder((32,), dtypes.uint32, 0)
  dependency=UOp.placeholder((32,), dtypes.float32, 1)
  source=UOp.placeholder((32,), dtypes.uint32, 2)
  dependency_out=UOp.placeholder((32,), dtypes.float32, 3)
  scratch=UOp.placeholder((32,), dtypes.uint32, 4, addrspace=AddrSpace.LOCAL).replace(tag=RuntimeLocalAllocation(128))
  dep=dependency[lid]+1.0
  ordered=source[lid.strict_after(dep)] if strict else source.after(dep)[lid]
  publish=scratch[lid].store(ordered)
  consume=out[lid].store(scratch.after(publish)[lid^1])
  root=dependency_out.after(consume)[lid].store(dep) if live_dependency else consume
  return UOp.sink(root, arg=KernelInfo(name=f"strict_after_{int(strict)}", opts_to_apply=()))


def _source(strict:bool, live_dependency:bool=False):
  program=to_program(_ast(strict, live_dependency), CUDARenderer(TARGET))
  return next(x.arg for x in program.src if x.op is Ops.SOURCE)


def _uses_slot(u:UOp, slot:int):
  return any(x.op is Ops.PARAM and x.arg.slot == slot for x in u.backward_slice_with_self)


def test_strict_after_survives_rewrite_while_ordinary_after_may_drop():
  strict=full_rewrite_to_sink(_ast(True), CUDARenderer(TARGET))
  ordinary=full_rewrite_to_sink(_ast(False), CUDARenderer(TARGET))
  edges=[x for x in strict.toposort() if x.op is Ops.AFTER and isinstance(x.arg, StrictAfter)]
  assert len(edges) == 1 and len(edges[0].src) == 2 and _uses_slot(edges[0].src[1], 1)
  assert any(x.op is Ops.LOAD and _uses_slot(x, 1) for x in strict.toposort())
  assert not any(x.op is Ops.LOAD and _uses_slot(x, 1) for x in ordinary.toposort())


def test_strict_after_dependency_is_opaque_to_commutative_key():
  value=UOp.special(32, "lidx0")
  dependency=UOp.const(dtypes.float32, 1.0)
  for _ in range(256): dependency=dependency*1.0001+0.0001
  assert _commutative_key(value.strict_after(dependency)) == _commutative_key(value)


def test_cuda_strict_after_renders_dependency_before_ordered_load():
  source=_source(True, live_dependency=True); lines=source.splitlines()
  dep=next(i for i,x in enumerate(lines) if "data1" in x and " = (*" in x)
  dep_value=next(i for i,x in enumerate(lines) if "+1.0f" in x)
  compiler_edge=next(i for i,x in enumerate(lines) if 'asm volatile("xor.b32' in x)
  ordered=next(i for i,x in enumerate(lines) if "data2" in x and " = (*" in x)
  consumer=next(i for i,x in enumerate(lines) if "*(buf0" in x and " = " in x and i > ordered)
  assert "alu0" in lines[compiler_edge]
  assert dep < dep_value < compiler_edge < ordered < consumer
  assert "__syncthreads" not in source and "membar" not in source.lower()
  ordinary=_source(False)
  assert 'asm volatile(""' not in ordinary
  assert not any("data1" in x and " = (*" in x for x in ordinary.splitlines())


def test_nvrtc_strict_after_final_sass_order():
  nvdisasm=shutil.which("nvdisasm")
  bundled=pathlib.Path(".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm")
  if nvdisasm is None and bundled.is_file(): nvdisasm=str(bundled)
  if nvdisasm is None: pytest.skip("nvdisasm unavailable")
  binary=NVRTCCompiler("sm_120", ptx=False, cache_key="strict_after_sass_v2").compile(_source(True, live_dependency=True))
  with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
    f.write(binary); f.flush()
    sass=subprocess.run([nvdisasm, "-c", f.name], check=True, text=True, stdout=subprocess.PIPE).stdout
  ins=[(int(pc,16), op) for pc,op in INSN_RE.findall(sass)]
  ldg=[pc for pc,op in ins if op.startswith("LDG")]
  fadd=[pc for pc,op in ins if op.startswith("FADD")]
  sts=[pc for pc,op in ins if op.startswith("STS")]
  assert len(ldg) == 2 and fadd and sts and ldg[0] < fadd[0] < ldg[1] < sts[0]
  assert not any(op.startswith(("BAR", "MEMBAR")) for _,op in ins)
  assert not any(op.startswith(("LDL", "STL")) for _,op in ins)
