import contextlib, pathlib, re, shutil, signal, subprocess, tempfile, time

import pytest

from tinygrad import dtypes
from tinygrad.codegen import full_rewrite_to_sink, to_program
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import KernelInfo, LoadSchedule, Ops, RuntimeLocalAllocation, UOp

TARGET=Target.parse("NV:CUDA:sm_120")
INSN_RE=re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(?:(@!?P\d+)\s+)?([A-Z][A-Z0-9_.]*)\b", re.MULTILINE)
FANIN, ROUNDS = 8, 32


@contextlib.contextmanager
def _hard_timeout(seconds:int):
  def expired(_signum, _frame): raise TimeoutError(f"opaque load scheduling exceeded {seconds}s")
  old=signal.signal(signal.SIGALRM, expired)
  signal.setitimer(signal.ITIMER_REAL, seconds)
  try: yield
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, old)


def _large_ast():
  lid=UOp.special(32, "lidx0")
  out=UOp.placeholder((32,), dtypes.uint32, 0)
  phase=UOp.placeholder((32*FANIN*ROUNDS,), dtypes.float32, 1)
  panel1=UOp.placeholder((32,), dtypes.uint32, 2)
  phase_out=UOp.placeholder((32,), dtypes.float32, 3)
  scratch=UOp.placeholder((32,), dtypes.uint32, 4, addrspace=AddrSpace.LOCAL).replace(tag=RuntimeLocalAllocation(128))
  acc=[phase[lid+i*32] for i in range(FANIN)]
  for k in range(1, ROUNDS):
    acc=[x*1.00001+phase[lid+(k*FANIN+i)*32] for i,x in enumerate(acc)]
  token=sum(acc[1:], acc[0])+12345.0
  ordered=panel1[lid].load().schedule_after(token)
  publish=scratch[lid].store(ordered)
  consume=out[lid].store(scratch.after(publish)[lid^1])
  root=phase_out.after(consume)[lid].store(token)
  return UOp.sink(root, arg=KernelInfo(name="opaque_load_schedule_large", opts_to_apply=()))


def _control_ast(strict:bool):
  lid=UOp.special(32, "lidx0")
  out=UOp.placeholder((32,), dtypes.uint32, 0)
  dependency=UOp.placeholder((32,), dtypes.float32, 1)
  source=UOp.placeholder((32,), dtypes.uint32, 2)
  dependency_out=UOp.placeholder((32,), dtypes.float32, 3)
  dep=dependency[lid]+7.0
  loaded=source[lid].load()
  ordered=loaded.strict_after(dep) if strict else loaded.after(dep)
  consume=out[lid].store(ordered)
  root=dependency_out.after(consume)[lid].store(dep)
  return UOp.sink(root, arg=KernelInfo(name=f"opaque_load_control_{int(strict)}", opts_to_apply=()))


def _source(ast:UOp):
  program=to_program(ast, CUDARenderer(TARGET))
  return next(x.arg for x in program.src if x.op is Ops.SOURCE)


def _uses_slot(u:UOp, slot:int):
  return any(x.op is Ops.PARAM and x.arg.slot == slot for x in u.backward_slice_with_self)


def _nvdisasm():
  ret=shutil.which("nvdisasm")
  bundled=pathlib.Path(".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm")
  return str(bundled) if ret is None and bundled.is_file() else ret


def test_large_dag_opaque_load_schedule_is_bounded_and_lexical():
  with _hard_timeout(30):
    ast=_large_ast(); raw_nodes=len(ast.toposort())
    st=time.perf_counter(); rewritten=full_rewrite_to_sink(ast, CUDARenderer(TARGET)); rewrite_s=time.perf_counter()-st
    rewritten_nodes=len(rewritten.toposort())
    st=time.perf_counter(); source=_source(ast); render_s=time.perf_counter()-st
  scheduled=[x for x in rewritten.toposort() if x.op is Ops.LOAD and
             any(s.op is Ops.AFTER and isinstance(s.arg, LoadSchedule) for s in x.src[1:])]
  assert len(scheduled) == 1 and len(scheduled[0].src) == 2
  assert not _uses_slot(scheduled[0].src[0], 1) and _uses_slot(scheduled[0].src[1], 1)
  assert rewritten_nodes <= raw_nodes*4 and rewritten_nodes < 20_000
  assert rewrite_s < 10 and render_s < 15
  lines=source.splitlines()
  phase_end=source.rindex("12345.0f")
  token_edge=source.index('asm volatile("xor.b32', phase_end)
  ordered=source.index("(*(data2", token_edge)
  consumer=source.index("*(buf0", ordered)
  assert phase_end < token_edge < ordered < consumer and source.count("(*(data2") == 1
  assert not any(x in source.lower() for x in ("__syncthreads", "membar", "atomic"))
  print(f"LARGE_DAG raw={raw_nodes} rewritten={rewritten_nodes} rewrite_s={rewrite_s:.4f} render_s={render_s:.4f}")


def test_large_dag_opaque_load_schedule_nv_sass_order():
  nvdisasm=_nvdisasm()
  if nvdisasm is None: pytest.skip("nvdisasm unavailable")
  with _hard_timeout(30):
    source=_source(_large_ast())
    st=time.perf_counter()
    binary=NVRTCCompiler("sm_120", ptx=False, cache_key="opaque_load_schedule_large_v2").compile(source)
    compile_s=time.perf_counter()-st
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
      f.write(binary); f.flush()
      sass=subprocess.run([nvdisasm, "-c", f.name], check=True, text=True, stdout=subprocess.PIPE).stdout
  ins=[(int(pc,16), pred, op) for pc,pred,op in INSN_RE.findall(sass)]
  phase_ops=[pc for pc,_,op in ins if op.startswith(("FADD", "FFMA"))]
  stores=[pc for pc,_,op in ins if op.startswith("STS")]
  token_ops=[pc for pc,_,op in ins if op.startswith("LOP3") and phase_ops and pc > max(phase_ops) and stores and pc < min(stores)]
  ordered=[pc for pc,_,op in ins if op.startswith("LDG") and token_ops and pc > min(token_ops) and stores and pc < min(stores)]
  assert phase_ops and len(token_ops) >= 1 and len(ordered) == 1 and stores
  assert max(phase_ops) < min(token_ops) < ordered[0] < min(stores)
  assert not any(op.startswith(("BAR", "MEMBAR", "ATOM", "LDL", "STL")) for _,_,op in ins)
  assert compile_s < 15
  print(f"LARGE_DAG compile_s={compile_s:.4f} phase_pc=0x{max(phase_ops):x} token_pc=0x{min(token_ops):x} "
        f"ordered_pc=0x{ordered[0]:x} sts_pc=0x{min(stores):x}")


def test_ordinary_and_value_strict_after_are_not_load_schedule_boundaries():
  ordinary=_source(_control_ast(False)); strict=_source(_control_ast(True))
  assert "schedule_index" not in ordinary and "schedule_index" not in strict
  strict_lines=strict.splitlines()
  load=next(i for i,x in enumerate(strict_lines) if "data2" in x and " = (*" in x)
  edge=next(i for i,x in enumerate(strict_lines) if 'asm volatile("xor.b32' in x)
  assert load < edge


def test_unsupported_renderer_fails_closed_before_lowering():
  renderer=CUDARenderer(TARGET); renderer.supports_load_schedule=False
  with pytest.raises(RuntimeError, match="cannot preserve schedule_after load ordering"):
    full_rewrite_to_sink(_large_ast(), renderer)
