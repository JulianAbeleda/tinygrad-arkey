from extra.llm_research.decode.nv_overlap_resource_join import build, complementary_cta_bound, resource_complete
from extra.llm_research.decode.route_b3_dag_attribution import CallRecord, CallAccess, attach_compiled_descriptors, B3AttributionError
import pytest


def _descriptor(regs: int = 32, block: list[int] | None = None, smem: int = 0) -> dict:
  block = block or [32, 1, 1]
  return {"binary_sha256": "0" * 64, "grid": [1, 1, 1], "block": block,
          "registers_per_thread": regs, "static_smem_bytes": smem,
          "dynamic_smem_bytes": 0, "local_mem_bytes": 0}


def _base_inputs() -> tuple[dict, dict]:
  manifest = {"rows": []}
  timeline = {"classes": {"quantize_q8_1": {"hidden_behind_mmq_us": 1.0}}}
  return manifest, timeline


def test_resource_join_refuses_partial_resource_rows():
  assert not resource_complete({"grid":[1,1,1],"block":[32,1,1],"registers_per_thread":32,"static_smem_bytes":0,"dynamic_smem_bytes":0,"local_mem_bytes":None})

def test_descriptor_collector_is_occurrence_exact_and_observational():
  calls=[CallRecord(0,"a",[CallAccess("x",0,4,True)]),CallRecord(1,"b",[CallAccess("x",0,4,False)])]
  d={i:{"binary_sha256":"0"*64,"grid":[1,1,1],"block":[32,1,1],"registers_per_thread":1,"static_smem_bytes":0,"dynamic_smem_bytes":0,"local_mem_bytes":0} for i in range(2)}
  assert [x["index"] for x in attach_compiled_descriptors(calls,d)] == [0,1]
  with pytest.raises(B3AttributionError): attach_compiled_descriptors(calls,{0:d[0]})


def test_complementary_cta_bound_scales_registers_and_threads():
  support = _descriptor(regs=32, block=[128, 1, 1])
  host = _descriptor(regs=32, block=[128, 1, 1])
  # 65536 regs / (128*32 + 128*32) = 8; 1536 threads / 256 = 6; smem 0.
  assert complementary_cta_bound(support, host) == 6
  # Register blow-up alone can sink the bound to zero.
  fat = _descriptor(regs=255, block=[256, 1, 1])
  assert complementary_cta_bound(fat, host) == 0


def test_build_fails_closed_without_pair_list():
  manifest, timeline = _base_inputs()
  census = {"nodes": [{"index": 0, "descriptor": _descriptor()}], "summary": {"node_count": 1}}
  out = build(manifest, timeline, census, pairs=None)
  assert out["status"] == "INCONCLUSIVE_FAIL_CLOSED"
  assert out["tinygrad_complete_resource_rows"] == 1


def test_build_fails_closed_below_gate_recovery():
  manifest, timeline = _base_inputs()
  census = {"nodes": [
    {"index": 0, "descriptor": _descriptor(regs=32, block=[128, 1, 1])},
    {"index": 1, "descriptor": _descriptor(regs=32, block=[128, 1, 1])},
  ], "summary": {"node_count": 2}}
  pairs = [{"support_id": 0, "support_name": "reduce_output_rmsnorm_32_128",
            "host_id": 1, "host_name": "q4k_g3_lanemap_gemv_4096_4096",
            "host_family": "q4k", "hideable_us": 4.0, "recovery_us": 3.97}]
  out = build(manifest, timeline, census, pairs=pairs)
  assert out["status"] == "INCONCLUSIVE_FAIL_CLOSED"
  assert out["positive_bound_pairs"] == 1
  assert out["best_pair"]["recovery_us"] == 3.97


def test_build_gpu_eligible_only_above_gate():
  manifest, timeline = _base_inputs()
  census = {"nodes": [
    {"index": 0, "descriptor": _descriptor(regs=32, block=[128, 1, 1])},
    {"index": 1, "descriptor": _descriptor(regs=32, block=[128, 1, 1])},
  ], "summary": {"node_count": 2}}
  pairs = [{"support_id": 0, "support_name": "reduce_output_rmsnorm_32_128",
            "host_id": 1, "host_name": "q4k_g3_lanemap_gemv_4096_4096",
            "host_family": "q4k", "hideable_us": 60.0, "recovery_us": 60.0}]
  out = build(manifest, timeline, census, pairs=pairs)
  assert out["status"] == "GPU_ELIGIBLE"
  assert out["best_pair"]["complementary_cta_bound"] > 0
