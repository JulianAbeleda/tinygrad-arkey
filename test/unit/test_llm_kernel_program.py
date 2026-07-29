import pytest

from tinygrad.llm.kernel_program import (KernelProgram, KernelProgramProvenance, execute_oracle_program,
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
