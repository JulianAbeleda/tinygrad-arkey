from extra.qk.packed_wmma_compile_gate import (CandidateEvidence, EmitterDescriptor, ProgramEvidence, ResourceEvidence,
  TensorEvidence, classify_packed_wmma_candidate, classify_registered_packed_wmma_candidate)


EMITTER = EmitterDescriptor("future.qk.fused", ("Q4_K", "Q6_K"))


def _candidate(**program_changes):
  values = {"name": "q4k_fused", "claimed_contraction": True,
            "instruction_families": ("v_wmma_f32_16x16x16_f16_w32",),
            "inputs": ("activation", "packed_weight"), "packed_inputs": ("packed_weight",),
            "prerequisites": (), "materializations": (), "resources": ResourceEvidence(40_960, 0, 0, 0)}
  values.update(program_changes)
  return CandidateEvidence("candidate-1", "Q4_K", 64, 256, (ProgramEvidence(**values),))


def test_valid_compile_evidence_passes_without_gpu():
  result = classify_packed_wmma_candidate(_candidate(), emitter=EMITTER)
  assert result.passed and result.contraction_program == "q4k_fused"
  assert result.to_json()["schema"] == "packed-wmma-compile-gate.v1"


def test_registered_primitive_still_blocks_without_compiled_candidate():
  result = classify_registered_packed_wmma_candidate("Q4_K", None)
  assert result.blocked and not result.passed
  assert result.reasons == ("no compiled generated candidate is available for Q4_K",)
  mismatch = classify_registered_packed_wmma_candidate("Q6_K", _candidate())
  assert not mismatch.passed and any("quant format" in reason for reason in mismatch.reasons)


def test_requires_exactly_one_contraction_program_and_fp16_wmma():
  duplicate = _candidate()
  duplicate = CandidateEvidence(duplicate.candidate_id, duplicate.quant_format, duplicate.n, duplicate.k,
                                duplicate.programs * 2)
  assert "found 2" in classify_packed_wmma_candidate(duplicate, emitter=EMITTER).reasons[0]
  missing = classify_packed_wmma_candidate(_candidate(instruction_families=("v_dot4_i32_i8",)), emitter=EMITTER)
  assert any("required fp16 WMMA" in reason for reason in missing.reasons)


def test_rejects_lost_packed_input_and_full_fp16_decode():
  lost = classify_packed_wmma_candidate(_candidate(inputs=("activation",)), emitter=EMITTER)
  assert any("not preserved" in reason for reason in lost.reasons)
  full = TensorEvidence("decoded_w", "float16", 64 * 256, "dequantized_weight")
  decoded = classify_packed_wmma_candidate(_candidate(prerequisites=(full,), materializations=(full,)), emitter=EMITTER)
  assert sum("full N*K fp16 decoded-weight" in reason for reason in decoded.reasons) == 2
  contraction = _candidate().programs[0]
  separate_decode = ProgramEvidence("decode_full_weight", False, (), ("packed_weight",), ("packed_weight",),
                                    materializations=(full,), resources=ResourceEvidence(0))
  split = CandidateEvidence("candidate-1", "Q4_K", 64, 256, (separate_decode, contraction))
  result = classify_packed_wmma_candidate(split, emitter=EMITTER)
  assert any("decoded-weight materialization" in reason for reason in result.reasons)


def test_resource_checks_are_bounded_and_metadata_conditional():
  # Missing optional scratch/spill metadata is accepted; LDS evidence is mandatory.
  assert classify_packed_wmma_candidate(_candidate(resources=ResourceEvidence(1024)), emitter=EMITTER).passed
  for resources, text in ((ResourceEvidence(None), "LDS usage"), (ResourceEvidence(65_537), "exceeds bound"),
                          (ResourceEvidence(0, scratch_bytes=4), "scratch_bytes"),
                          (ResourceEvidence(0, vgpr_spills=1), "vgpr_spills")):
    result = classify_packed_wmma_candidate(_candidate(resources=resources), emitter=EMITTER)
    assert not result.passed and any(text in reason for reason in result.reasons)


def test_mapping_records_allow_future_emitters_to_plug_in():
  candidate = {"candidate_id": "q6-future", "quant_format": "Q6_K", "n": 16, "k": 256, "programs": [{
    "name": "q6_fused", "claimed_contraction": True,
    "instruction_families": ["v_wmma_f32_16x16x16_f16"], "inputs": ["x", "q6_bytes"],
    "packed_inputs": ["q6_bytes"], "resources": {"lds_bytes": 0}}]}
  result = classify_registered_packed_wmma_candidate("Q6_K", candidate, emitters={"Q6_K": {
    "emitter_id": "future.q6", "quant_formats": ["Q6_K"], "fused_packed_operand": True}})
  assert result.passed
