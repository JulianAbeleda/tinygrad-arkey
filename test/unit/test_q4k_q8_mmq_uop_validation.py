import json, subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import extra.qk.q4k_q8_mmq_uop_validation as validation
from extra.qk.q4k_q8_mmq_uop import (describe_q4k_q8_mmq_uop, describe_q4k_q8_mmq_wmma,
  emit_q4k_q8_mmq_uop, emit_q4k_q8_mmq_wmma)
from tinygrad import dtypes
from tinygrad.uop.ops import UOp


def _graph(mode):
  m,n,k = (2,3,256) if mode == validation.SCALAR_MODE else (16,16,256)
  emitter = (emit_q4k_q8_mmq_uop(describe_q4k_q8_mmq_uop(m,n,k)) if mode == validation.SCALAR_MODE
             else emit_q4k_q8_mmq_wmma(describe_q4k_q8_mmq_wmma()))
  return emitter(UOp.placeholder((m,n),dtypes.float32,0), UOp.placeholder((n*36,),dtypes.uint32,1),
    UOp.placeholder((m*k,),dtypes.int8,2), UOp.placeholder((m*8,),dtypes.float32,3))


def _passing(mode):
  scalar = mode == validation.SCALAR_MODE
  return {"mode":mode, "uop":validation.inspect_uop_graph(_graph(mode)), "program_count":1,
    "kernel_count_delta":1, "kernel_name":validation.CANDIDATE_KERNEL_NAMES[mode], "fallback_used":False,
    "numeric":{"reference":"independent_packed_byte", "allclose":True},
    "isa":{"classification":"scalar_direct" if scalar else "wmma", "signed_integer_wmma":not scalar}}


def test_authored_graph_modes_are_distinct_and_route_local_wmma_free():
  scalar, wmma = validation.inspect_uop_graph(_graph(validation.SCALAR_MODE)), validation.inspect_uop_graph(_graph(validation.WMMA_MODE))
  assert scalar["classification"] == "scalar_direct_uop" and scalar["tc_opt_count"] == 0
  assert wmma["classification"] == "generic_tc_candidate"
  assert wmma["tc_opt"] == {"axis":0, "arg":[-1,2,1]}
  assert scalar["route_local_wmma_count"] == wmma["route_local_wmma_count"] == 0
  assert scalar["computed_store_index"] and wmma["computed_store_index"]


@pytest.mark.parametrize("mode", validation.MODES)
def test_mode_specific_admission_requires_program_counter_launch_name_numeric_and_isa(mode):
  assert validation.admit_evidence(_passing(mode),mode=mode)["passed"]
  for key,value in (("fallback_used",True),("program_count",2),("kernel_count_delta",0)):
    row=_passing(mode); row[key]=value
    assert not validation.admit_evidence(row,mode=mode)["passed"]
  row=_passing(mode); row["numeric"]["allclose"]=False
  assert not validation.admit_evidence(row,mode=mode)["passed"]


def test_scalar_and_wmma_evidence_cannot_cross_admit():
  assert not validation.admit_evidence(_passing(validation.SCALAR_MODE),mode=validation.WMMA_MODE)["passed"]
  assert not validation.admit_evidence(_passing(validation.WMMA_MODE),mode=validation.SCALAR_MODE)["passed"]
  row=_passing(validation.WMMA_MODE); row["isa"]={"classification":"scalar_direct","signed_integer_wmma":False}
  assert not validation.admit_evidence(row,mode=validation.WMMA_MODE)["passed"]
  row=_passing(validation.SCALAR_MODE); row["isa"]={"classification":"wmma","signed_integer_wmma":True}
  assert not validation.admit_evidence(row,mode=validation.SCALAR_MODE)["passed"]


def test_independent_reference_decodes_packed_bytes_without_tinygrad():
  raw=np.zeros(144,dtype=np.uint8)
  raw[:4]=np.frombuffer(np.array([.5,.25],dtype="<f2").tobytes(),dtype=np.uint8)
  raw[4:8]=2; raw[16:]=0x33
  got=validation.independent_packed_byte_reference(raw.view(np.uint32),np.ones((1,256),dtype=np.int8),
    np.ones((1,8),dtype=np.float32),m=1,n=1,k=256)
  assert got[0,0] == 4*32*.5*2*3


def test_isa_classifier_requires_direct_evidence_and_signed_integer_flags():
  assert validation.classify_isa("")["classification"] == "missing"
  assert validation.classify_isa("void kernel() {}")["classification"] == "unknown"
  assert validation.classify_isa("v_add_f32 v1, v2, v3\nglobal_store_dword v0, v1")["classification"] == "scalar_direct"
  unsigned=validation.classify_isa("v_wmma_i32_16x16x16_iu8 operands, 0")
  assert unsigned["classification"] == "wmma" and not unsigned["signed_integer_wmma"]
  signed=validation.classify_isa("v_wmma_i32_16x16x16_iu8 operands, 3")
  assert signed["classification"] == "wmma" and signed["signed_integer_wmma"]


def test_parent_worker_timeout_bad_json_and_mode_forwarding_are_cpu_safe(monkeypatch):
  def timeout(*args,**kwargs): raise subprocess.TimeoutExpired(args[0],kwargs["timeout"])
  monkeypatch.setattr(validation.subprocess,"run",timeout)
  assert "timed out" in validation.run_amd_validation(mode=validation.WMMA_MODE,timeout_seconds=.01)["blocker"]
  calls=[]
  monkeypatch.setattr(validation.subprocess,"run",lambda cmd,**kw: calls.append(cmd) or SimpleNamespace(
    returncode=0,stdout=json.dumps(_passing(validation.WMMA_MODE)),stderr=""))
  assert validation.run_amd_validation(mode=validation.WMMA_MODE)["passed"]
  assert calls[0][-2:] == ["--mode",validation.WMMA_MODE]
  monkeypatch.setattr(validation.subprocess,"run",lambda *a,**k: SimpleNamespace(returncode=0,stdout="not-json",stderr=""))
  assert "invalid JSON" in validation.run_amd_validation(mode=validation.SCALAR_MODE)["blocker"]
