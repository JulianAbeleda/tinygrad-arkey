"""Generic ownership gate for computed inputs to opaque PROGRAM calls."""
import numpy as np

from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.callify import (_direct_readonly_model_parameter_carrier, _exact_readonly_model_parameter_carrier, _program_after_output_slot_rebind,
                              _readonly_program_input_param_slots, _writable_function_param_slots)
from tinygrad.engine.realize import compile_linear
from tinygrad.function import _computed_program_inputs, function
from tinygrad.llm.memory_semantics import model_parameter
from tinygrad.uop.ops import CallInfo, Ops, ProgramInfo, UOp


def _finalized_increment_program():
  source = Tensor.empty(8, dtype=dtypes.int32, device="CPU").realize()
  call, = compile_linear((source + 1).contiguous().schedule_linear()).src
  assert call.src[0].op is Ops.PROGRAM and isinstance(call.src[0].arg, ProgramInfo)
  assert call.src[0].arg.outs == (0,) and call.src[0].arg.ins == (1,)
  return call.src[0]


class _TwoProjectionProgram:
  def __init__(self):
    self.program = _finalized_increment_program()
    self.outputs = [Tensor.empty(8, dtype=dtypes.int32, device="CPU").realize() for _ in range(32)]
    self.output_uops = tuple(output.uop.buf_uop for output in self.outputs)
    self.cursor = 0

  def project(self, value:Tensor) -> Tensor:
    output = self.outputs[self.cursor]
    self.cursor += 1
    return output.uop_program(value, fxn=lambda *_: self.program)[0]


def _values(start:int) -> Tensor:
  return Tensor(list(range(start, start+8)), dtype=dtypes.int32, device="CPU").contiguous().realize()


def _model_values(start:int) -> Tensor:
  return model_parameter(_values(start))


def test_program_input_hoist_fails_closed_on_write_or_undeclared_slots():
  base = UOp.new_buffer("CPU", 8, dtypes.int32)
  computed = (base + 1).contiguous()
  output = UOp.new_buffer("CPU", 8, dtypes.int32)
  body = UOp(Ops.SINK)
  device = UOp(Ops.DEVICE, arg="CPU")
  def call(info): return UOp(Ops.PROGRAM, src=(body, device), arg=info).call(output, computed)

  read = call(ProgramInfo(globals=(0, 1), outs=(0,), ins=(1,)))
  assert _computed_program_inputs(read, (base,)) == [computed]
  read_write = call(ProgramInfo(globals=(0, 1), outs=(0, 1), ins=(1,)))
  undeclared = call(ProgramInfo(globals=(0, 1), outs=(0,), ins=()))
  assert _computed_program_inputs(read_write, (base,)) == []
  assert _computed_program_inputs(undeclared, (base,)) == []


def test_program_output_rebind_requires_one_unaliased_write_only_slot():
  base = UOp.param(0, dtypes.int32, (8,), "CPU")
  source = UOp.param(1, dtypes.int32, (8,), "CPU")
  target = UOp.param(2, dtypes.int32, (8,), "CPU")
  body, device = UOp(Ops.SINK), UOp(Ops.DEVICE, arg="CPU")
  def result(info, *args):
    program = UOp(Ops.PROGRAM, src=(body, device), arg=info)
    return base.after(program.call(*args))

  write_only = ProgramInfo(globals=(0, 1), outs=(0,), ins=(1,))
  assert _program_after_output_slot_rebind(result(write_only, base, source), target) == (base, target)
  read_write = ProgramInfo(globals=(0, 1), outs=(0,), ins=(0, 1))
  assert _program_after_output_slot_rebind(result(read_write, base, source), target) is None
  assert _program_after_output_slot_rebind(result(write_only, base, base), target) is None


def test_readonly_model_parameter_admission_is_exact_and_fails_closed():
  weight = _model_values(0)
  assert _exact_readonly_model_parameter_carrier(weight.uop)
  assert _direct_readonly_model_parameter_carrier(weight.contiguous().uop) is weight.uop
  assert not _exact_readonly_model_parameter_carrier(weight[1:].uop)
  assert not _exact_readonly_model_parameter_carrier((weight + 1).contiguous().uop)

  output = UOp.param(1, dtypes.int32, (8,), "CPU")
  source = UOp.param(0, dtypes.int32, (8,), "CPU")
  body, device = UOp(Ops.SINK), UOp(Ops.DEVICE, arg="CPU")
  def result(info, value=source):
    program = UOp(Ops.PROGRAM, src=(body, device), arg=info)
    return output.after(program.call(output, value))

  readonly = ProgramInfo(globals=(0, 1), outs=(0,), ins=(1,))
  assert _readonly_program_input_param_slots((result(readonly),)) == frozenset({0})
  assert _writable_function_param_slots((result(readonly),)) == frozenset({1})
  assert _readonly_program_input_param_slots((result(readonly, source.contiguous()),)) == frozenset({0})
  # One ordinary tensor use or a read/write ABI use closes the direct route.
  assert _readonly_program_input_param_slots((result(readonly), source+1)) == frozenset()
  read_write = ProgramInfo(globals=(0, 1), outs=(0, 1), ins=(1,))
  assert _readonly_program_input_param_slots((result(read_write),)) == frozenset()
  assert _writable_function_param_slots((result(read_write),)) == frozenset({0, 1})


def test_readonly_program_slots_do_not_cross_nested_function_param_namespaces():
  outer_source = UOp.param(0, dtypes.int32, (8,), "CPU")
  output = UOp.param(2, dtypes.int32, (8,), "CPU")
  body, device = UOp(Ops.SINK), UOp(Ops.DEVICE, arg="CPU")
  info = ProgramInfo(globals=(0, 1), outs=(0,), ins=(1,))
  read = output.after(UOp(Ops.PROGRAM, src=(body, device), arg=info).call(output, outer_source))

  # The nested body deliberately reuses PARAM slot 0 for ordinary ALU.  Its
  # namespace must not contaminate the enclosing FUNCTION's slot-0 proof.
  nested_param = UOp.param(0, dtypes.int32, (8,), "CPU")
  nested_arg = UOp.param(1, dtypes.int32, (8,), "CPU")
  nested = UOp(Ops.FUNCTION, dtypes.void, (UOp.maketuple(nested_param+1), nested_arg), CallInfo(precompile=True))
  assert _readonly_program_input_param_slots((read, nested.gettuple(0))) == frozenset({0})

  # A model input may also be forwarded through several FUNCTION boundaries
  # before the exact read-only PROGRAM.  Prove that ABI transitively without
  # descending into or conflating the child PARAM namespace.
  nested_output = UOp.param(1, dtypes.int32, (8,), "CPU")
  nested_read = nested_output.after(UOp(Ops.PROGRAM, src=(body, device), arg=info).call(nested_output, nested_param))
  forwarding = UOp(Ops.FUNCTION, dtypes.void, (UOp.maketuple(nested_read), outer_source), CallInfo(precompile=True))
  assert _readonly_program_input_param_slots((forwarding.gettuple(0),)) == frozenset({0})

  nested_rmw = nested_output.after(UOp(Ops.PROGRAM, src=(body, device),
    arg=ProgramInfo(globals=(0, 1), outs=(0, 1), ins=(1,))).call(nested_output, nested_param))
  forwarding_rmw = UOp(Ops.FUNCTION, dtypes.void, (UOp.maketuple(nested_rmw), outer_source), CallInfo(precompile=True))
  assert _writable_function_param_slots((forwarding_rmw.gettuple(0),)) == frozenset({0})


def test_canonical_model_parameter_program_inputs_are_zero_copy_and_fresh():
  provider = _TwoProjectionProgram()

  @function(precompile=True, allow_implicit=True)
  def pair(left, right):
    return provider.project(left.contiguous()), provider.project(right.contiguous())

  first_weights = (_model_values(0), _model_values(10))
  second_weights = (_model_values(20), _model_values(30))
  before = tuple(x.numpy().copy() for x in first_weights + second_weights)
  first, second = pair(*first_weights), pair(*second_weights)
  Tensor.realize(*first, *second)
  for got, expected in zip(first+second, before): np.testing.assert_array_equal(got.numpy(), expected+1)
  for got, expected in zip(first_weights + second_weights, before): np.testing.assert_array_equal(got.numpy(), expected)

  # No transport kernel may stand between either canonical weight and the
  # opaque reader; its ProgramInfo.ins argument must be the exact allocation.
  probe_weights = (_model_values(40), _model_values(50))
  canonical = {x.uop.buf_uop for x in probe_weights}
  linear = compile_linear(Tensor.schedule_linear(*pair(*probe_weights)))
  native = [call for call in linear.src if call.src[0].arg.name == provider.program.arg.name]
  assert len(native) == 2 and len(linear.src) == 2
  assert {call.src[2].buf_uop for call in native} == canonical
  assert all(call.src[0].arg.ins == (1,) and call.src[0].arg.outs == (0,) for call in native)


def test_canonical_model_parameter_program_input_survives_jit_replay():
  provider = _TwoProjectionProgram()

  @function(precompile=True, allow_implicit=True)
  def pair(left, right):
    return provider.project(left.contiguous()), provider.project(right.contiguous())

  @TinyJit
  def run(left, right):
    a, b = pair(left, right)
    return (a+b).realize()

  for start in (0, 20, 40, 60, 80):
    left, right = _model_values(start), _model_values(start+10)
    got = run(left, right)
    np.testing.assert_array_equal(got.numpy(), np.arange(start, start+8, dtype=np.int32)*2 + 12)


def test_explicit_params_and_computed_values_bind_exact_program_inputs():
  provider = _TwoProjectionProgram()

  @function(precompile=True, allow_implicit=True)
  def explicit_pair(left, right):
    return provider.project(left), provider.project(right)

  @function(precompile=True, allow_implicit=True)
  def computed_pair(value):
    shared = (value * 3).contiguous()
    return provider.project(shared), provider.project(shared + 7)

  a, b = _values(0), _values(10)
  control_a, control_b = explicit_pair(a, b)
  first = computed_pair(a)
  second = computed_pair(b)
  Tensor.realize(control_a, control_b, *first, *second)

  np.testing.assert_array_equal(control_a.numpy(), np.arange(8, dtype=np.int32) + 1)
  np.testing.assert_array_equal(control_b.numpy(), np.arange(10, 18, dtype=np.int32) + 1)
  expected_first = (np.arange(8, dtype=np.int32) * 3 + 1, np.arange(8, dtype=np.int32) * 3 + 8)
  expected_second = (np.arange(10, 18, dtype=np.int32) * 3 + 1, np.arange(10, 18, dtype=np.int32) * 3 + 8)
  for got, expected in zip(first, expected_first): np.testing.assert_array_equal(got.numpy(), expected)
  for got, expected in zip(second, expected_second): np.testing.assert_array_equal(got.numpy(), expected)
  assert len({id(x) for x in provider.output_uops[:6]}) == 6

  # Every opaque read is backed by a prior producer's exact output buffer.
  linear = compile_linear(Tensor.schedule_linear(*computed_pair(_values(20))))
  produced:set = set()
  opaque_reads = []
  for call in linear.src:
    info = call.src[0].arg
    for slot in info.outs: produced.add(call.src[slot+1].buf_uop)
    if info.name == provider.program.arg.name:
      opaque_reads.extend((call.src[slot+1].buf_uop, call.src[slot+1].buf_uop in produced)
                          for slot in info.ins if slot not in info.outs)
  assert len(opaque_reads) == 2 and all(had_prior_writer for _, had_prior_writer in opaque_reads)


def test_computed_program_input_survives_capture_and_replay():
  provider = _TwoProjectionProgram()

  @function(precompile=True, allow_implicit=True)
  def computed_pair(value):
    shared = (value * 3).contiguous()
    return provider.project(shared), provider.project(shared + 7)

  @TinyJit
  def run(value):
    left, right = computed_pair(value)
    # One downstream consumer exercises the production shape: both native
    # results remain internal graph values and the returned allocation is
    # refreshed on capture and every replay.
    return (left + right).realize()

  for start in (0, 10, 20, 30, 40):
    combined = run(_values(start))
    base = np.arange(start, start+8, dtype=np.int32) * 3
    np.testing.assert_array_equal(combined.numpy(), 2*base + 9)


def test_single_computed_program_output_is_returned_not_stale():
  program = _finalized_increment_program()
  output = Tensor.empty(8, dtype=dtypes.int32, device="CPU").realize()
  output_before = output.numpy().copy()

  @function(precompile=True, allow_implicit=True)
  def project(value):
    return output.uop_program((value*3).contiguous(), fxn=lambda *_: program)[0]

  value = _values(0)
  np.testing.assert_array_equal(project(value).realize().numpy(), np.arange(8, dtype=np.int32)*3 + 1)
  np.testing.assert_array_equal(output.numpy(), output_before)


def test_nested_single_computed_program_output_is_returned_not_stale():
  program = _finalized_increment_program()
  output = Tensor.empty(8, dtype=dtypes.int32, device="CPU").realize()
  output_before = output.numpy().copy()

  @function(precompile=True, allow_implicit=True)
  def inner(value):
    return output.uop_program((value*3).contiguous(), fxn=lambda *_: program)[0]

  @function(precompile=True, allow_implicit=True)
  def outer(value):
    return inner(value)

  @TinyJit
  def run(value):
    return outer(value).realize()

  for start in (0, 10, 20, 30, 40):
    np.testing.assert_array_equal(run(_values(start)).numpy(), np.arange(start, start+8, dtype=np.int32)*3 + 1)
  np.testing.assert_array_equal(output.numpy(), output_before)
