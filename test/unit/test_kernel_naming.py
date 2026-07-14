import hashlib, json, re, subprocess, sys

from tinygrad.codegen import to_program, to_program_cache
from tinygrad.codegen.opt.postrange import apply_opts
from tinygrad.dtype import dtypes
from tinygrad.renderer import Target
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.uop.ops import KernelCandidateContext, KernelInfo, Ops, UOp


def _sink(value: int, info: KernelInfo|None=None) -> UOp:
  out = UOp.param(0, dtypes.float.ptr(1))
  return out.index(UOp.const(dtypes.int, 0), ptr=True).store(UOp.const(dtypes.float, value)).sink(arg=info or KernelInfo())


def _identity(value: int) -> tuple[str, str, str]:
  # Exercise naming on every call instead of accepting a to_program cache hit.
  to_program_cache.clear()
  program = to_program(_sink(value), PythonRenderer(Target("PYTHON")))
  source = next(x.arg for x in program.src if x.op is Ops.SOURCE)
  binary = next(x.arg for x in program.src if x.op is Ops.BINARY)
  return program.arg.function_name, hashlib.sha256(source.encode()).hexdigest(), hashlib.sha256(binary).hexdigest()


def test_generated_names_are_structural_stable_and_valid():
  first, repeated, distinct = _identity(1), _identity(1), _identity(2)
  assert first == repeated
  assert first[0] != distinct[0]
  assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", first[0])


def test_generated_identity_matches_fresh_process_after_other_compiles():
  _identity(2)
  expected = _identity(1)
  script = """
import hashlib, json
from tinygrad.codegen import to_program
from tinygrad.dtype import dtypes
from tinygrad.renderer import Target
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp
out = UOp.param(0, dtypes.float.ptr(1))
sink = out.index(UOp.const(dtypes.int, 0), ptr=True).store(UOp.const(dtypes.float, 1)).sink(arg=KernelInfo())
program = to_program(sink, PythonRenderer(Target('PYTHON')))
source = next(x.arg for x in program.src if x.op is Ops.SOURCE)
binary = next(x.arg for x in program.src if x.op is Ops.BINARY)
print(json.dumps((program.arg.function_name, hashlib.sha256(source.encode()).hexdigest(), hashlib.sha256(binary).hexdigest())))
"""
  actual = tuple(json.loads(subprocess.check_output([sys.executable, "-c", script], text=True)))
  assert actual == expected


def test_candidate_context_does_not_change_structural_name_and_override_is_preserved():
  renderer = PythonRenderer(Target("PYTHON"))
  contexts = [KernelCandidateContext("boltbeam.full_kernel_candidate.v1", digit * 64) for digit in ("1", "2")]
  names = [apply_opts(_sink(1, KernelInfo(opts_to_apply=(), candidate_context=context)), renderer).arg.function_name for context in contexts]
  assert names[0] == names[1]
  assert apply_opts(_sink(1, KernelInfo(name="explicit-name", opts_to_apply=())), renderer).arg.name == "explicit-name"
