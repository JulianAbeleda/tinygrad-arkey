import pytest
from types import SimpleNamespace

import tinygrad.llm.kernel_program as kernel_program
from tinygrad import dtypes
from tinygrad.llm.kernel_program import (KernelProgram, KernelProgramProvenance, OutputSpec, execute_oracle_program,
                                          execute_promoted_program, execute_research_program,
                                          execute_research_program_outputs)


class Output:
  def __init__(self): self.calls = []

  def uop_program(self, *inputs, fxn):
    self.calls.append((inputs, fxn))
    return ("output-zero", "output-one", "output-two")


def emitter(*args): return args


def program(provenance): return KernelProgram("route", "program", provenance, emitter)


@pytest.mark.parametrize("provenance", (KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
                                          KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED))
def test_promoted_program_delegates_once_and_returns_output_zero(provenance):
  output, first, second = Output(), object(), object()
  assert execute_promoted_program(output, first, second, program=program(provenance)) == "output-zero"
  assert output.calls == [((first, second), emitter)]


@pytest.mark.parametrize("provenance", (KernelProgramProvenance.HAND_AUTHORED_ORACLE,
                                          KernelProgramProvenance.RESEARCH_ONLY))
def test_promoted_program_rejects_oracle_and_research(provenance):
  output = Output()
  with pytest.raises(ValueError, match="execute_promoted_program"):
    execute_promoted_program(output, program=program(provenance))
  assert output.calls == []


@pytest.mark.parametrize("provenance", (KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
                                          KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED,
                                          KernelProgramProvenance.RESEARCH_ONLY))
def test_oracle_program_rejects_non_oracle_provenance(provenance):
  with pytest.raises(ValueError, match="execute_oracle_program"):
    execute_oracle_program(Output(), program=program(provenance))


@pytest.mark.parametrize("provenance", (KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
                                          KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED,
                                          KernelProgramProvenance.HAND_AUTHORED_ORACLE))
def test_research_program_rejects_non_research_provenance(provenance):
  with pytest.raises(ValueError, match="execute_research_program"):
    execute_research_program(Output(), program=program(provenance))


def test_oracle_and_research_programs_delegate_once_to_their_matching_boundaries():
  oracle_output, research_output = Output(), Output()
  assert execute_oracle_program(oracle_output, object(), program=program(KernelProgramProvenance.HAND_AUTHORED_ORACLE)) == "output-zero"
  assert execute_research_program(research_output, object(), program=program(KernelProgramProvenance.RESEARCH_ONLY)) == "output-zero"
  assert len(oracle_output.calls) == len(research_output.calls) == 1


def test_research_outputs_preserves_all_outputs_with_one_delegation():
  output, first, second = Output(), object(), object()
  result = execute_research_program_outputs(output, first, second,
                                             program=program(KernelProgramProvenance.RESEARCH_ONLY))
  assert result == ("output-zero", "output-one", "output-two")
  assert output.calls == [((first, second), emitter)]


@pytest.mark.parametrize("provenance", (KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
                                          KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED,
                                          KernelProgramProvenance.HAND_AUTHORED_ORACLE))
def test_research_outputs_rejects_non_research_provenance(provenance):
  output = Output()
  with pytest.raises(ValueError, match="execute_research_program_outputs"):
    execute_research_program_outputs(output, program=program(provenance))
  assert output.calls == []


@pytest.mark.parametrize("route_id,program_id,emitter_value", (("", "program", emitter), ("route", "", emitter),
                                                                  ("route", "program", None)))
def test_kernel_program_requires_identifiers_and_callable_emitter(route_id, program_id, emitter_value):
  with pytest.raises(ValueError):
    KernelProgram(route_id, program_id, KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emitter_value)


def test_trace_facts_are_stable_and_exclude_emitter():
  value = program(KernelProgramProvenance.MACHINE_SEARCH_GENERATED)
  assert value.to_dict() == {"route_id": "route", "program_id": "program", "provenance": "machine_search_generated"}
  assert "emitter" not in value.to_dict()


def test_promoted_program_allocates_output_from_program_output_spec(monkeypatch):
  allocated = []

  class FakeEmpty:
    def __init__(self, *shape, dtype, device):
      allocated.append((shape, dtype, device))

    def uop_program(self, *inputs, fxn):
      return ("allocated-output",)

  monkeypatch.setattr(kernel_program, "Tensor", SimpleNamespace(empty=FakeEmpty))
  first, second = SimpleNamespace(device="GPU:0"), SimpleNamespace(device="GPU:0")
  spec = OutputSpec((4, 8), dtypes.float32)
  promoted = KernelProgram("route", "program", KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emitter,
                           output_spec=spec)
  assert execute_promoted_program(None, first, second, program=promoted) == "allocated-output"
  assert allocated == [((4, 8), dtypes.float32, "GPU:0")]


def test_positional_output_overrides_program_output_spec():
  output, first = Output(), object()
  promoted = KernelProgram("route", "program", KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emitter,
                           output_spec=OutputSpec((3,), dtypes.float32))
  assert execute_promoted_program(output, first, program=promoted) == "output-zero"
  assert output.calls == [((first,), emitter)]


def test_promoted_program_requires_output_or_program_output_spec():
  with pytest.raises(ValueError, match="output_spec"):
    execute_promoted_program(None, program=program(KernelProgramProvenance.MACHINE_SEARCH_GENERATED))


@pytest.mark.parametrize("shape,dtype", [((), dtypes.float32), ((0,), dtypes.float32),
                                         ((4, "x"), dtypes.float32), ((4,), None)])
def test_output_spec_validates_shape_and_dtype(shape, dtype):
  with pytest.raises(ValueError):
    OutputSpec(shape, dtype)


def test_kernel_program_requires_output_spec_value_type():
  with pytest.raises(ValueError, match="output_spec"):
    KernelProgram("route", "program", KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emitter,
                  output_spec=object())
