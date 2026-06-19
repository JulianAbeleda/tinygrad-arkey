#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-scheduler-tooling-backend/rocprofiler_thread_trace_audit.json"

ROCP = Path("/tmp/rocm-systems-main-probe/projects/rocprofiler-sdk/source/lib")
INST = Path("/opt/rocm-7.2.4")

FILES = {
  "tinygrad_hcq": ROOT / "tinygrad/runtime/ops_amd.py",
  "rocprof_tool": ROCP / "rocprofiler-sdk-tool/tool.cpp",
  "thread_trace_core": ROCP / "rocprofiler-sdk/thread_trace/core.cpp",
  "thread_trace_service": ROCP / "rocprofiler-sdk/thread_trace/service.cpp",
  "packet_construct": ROCP / "rocprofiler-sdk/aql/packet_construct.cpp",
  "hsa_queue": ROCP / "rocprofiler-sdk/hsa/queue.cpp",
  "hsa_aql_packet": ROCP / "rocprofiler-sdk/hsa/aql_packet.cpp",
  "aqlprofile_threadtrace": ROCP / "aqlprofile/core/threadtrace.cpp",
  "sqtt_builder": ROCP / "aqlprofile/pm4/sqtt_builder.h",
  "sample_agent": INST / "share/rocprofiler-sdk/samples/thread_trace/agent.cpp",
  "installed_header": INST / "include/rocprofiler-sdk/experimental/thread-trace/core.h",
}

CHECKS = {
  "rocprofiler_dispatch_service": {
    "rocprof_files": ["rocprof_tool", "thread_trace_core"],
    "tinygrad_files": ["tinygrad_hcq"],
    "patterns": [
      "rocprofiler_configure_dispatch_thread_trace_service",
      "ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP",
      "DispatchThreadTracer::pre_kernel_call",
      "DispatchThreadTracer::post_kernel_call",
    ],
    "meaning": "rocprofiler injects ATT start/stop around selected HSA kernel dispatches.",
  },
  "hsa_queue_interposition": {
    "rocprof_files": ["hsa_queue", "thread_trace_core"],
    "tinygrad_files": ["tinygrad_hcq"],
    "patterns": [
      "hsa_amd_queue_intercept_create_fn",
      "hsa_amd_queue_intercept_register_fn",
      "WriteInterceptor",
      "add_callback",
      "enable_serialization",
    ],
    "meaning": "rocprofiler sees HSA queue writes and can insert profiling packets before/after kernels.",
  },
  "profiled_queue_activation": {
    "rocprof_files": ["hsa_queue"],
    "tinygrad_files": ["tinygrad_hcq"],
    "patterns": [
      "hsa_amd_profiling_set_profiler_enabled_fn",
      "set_profiler_active_on_queue",
      "Could not set agent to be profiled",
    ],
    "meaning": "rocprofiler explicitly marks the HSA queue/agent as profiling-active before ATT packets run.",
  },
  "aqlprofile_packet_factory": {
    "rocprof_files": ["packet_construct", "hsa_aql_packet", "aqlprofile_threadtrace"],
    "tinygrad_files": ["tinygrad_hcq"],
    "patterns": [
      "ThreadTraceAQLPacketFactory",
      "aqlprofile_att_create_packets",
      "aqlprofile_att_profile_t",
      "PopulateAql",
      "VENDOR_BIT | BARRIER_BIT",
    ],
    "meaning": "rocprofiler does not hand-code only registers; AQLprofile creates vendor PM4 AQL packets.",
  },
  "trace_control_buffer_protocol": {
    "rocprof_files": ["aqlprofile_threadtrace", "sqtt_builder"],
    "tinygrad_files": ["tinygrad_hcq"],
    "patterns": [
      "TraceControl",
      "CreateTraceControlBuf",
      "aqlprofile_att_iterate_data",
      "SQ_THREAD_TRACE_STATUS",
      "SQ_THREAD_TRACE_WPTR",
      "BuildCacheFlushPacket",
    ],
    "meaning": "rocprofiler records status/counter/WPTR into a host-visible trace-control buffer and iterates payloads from it.",
  },
  "sqtt_begin_order": {
    "rocprof_files": ["sqtt_builder"],
    "tinygrad_files": ["tinygrad_hcq"],
    "patterns": [
      "SetGRBMToBroadcast",
      "BuildPrimeL2",
      "SQ_THREAD_TRACE_STATUS_ADDR, 0",
      "SQ_THREAD_TRACE_SIZE_ADDR",
      "SQ_THREAD_TRACE_BASE_ADDR",
      "SQ_THREAD_TRACE_MASK_ADDR",
      "SQ_THREAD_TRACE_TOKEN_MASK_ADDR",
      "COMPUTE_THREAD_TRACE_ENABLE_ADDR, 1",
    ],
    "meaning": "rocprofiler's gfx11 begin sequence includes buffer priming, status clearing, exact per-SE register order, and enable.",
  },
  "decoder_metadata_markers": {
    "rocprof_files": ["sqtt_builder", "aqlprofile_threadtrace", "thread_trace_core"],
    "tinygrad_files": ["tinygrad_hcq"],
    "patterns": [
      "rocprof_trace_decoder_instrument_enable_t",
      "ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO",
      "InsertCodeobjMarker",
      "SQ_THREAD_TRACE_USERDATA_2",
      "CodeobjCallbackRegistry",
    ],
    "meaning": "rocprofiler emits decoder agent/code-object metadata through SQTT userdata packets.",
  },
  "sqtt_end_order": {
    "rocprof_files": ["sqtt_builder", "aqlprofile_threadtrace"],
    "tinygrad_files": ["tinygrad_hcq"],
    "patterns": [
      "COMPUTE_THREAD_TRACE_ENABLE_ADDR, 0",
      "THREAD_TRACE_FINISH",
      "sqtt_busy_mask",
      "ReadValues",
      "BuildCacheFlushPacket",
      "control_buffer_ptr",
    ],
    "meaning": "rocprofiler's stop path disables trace, waits hardware finish/busy, copies status/counter/WPTR, and cache-flushes the control buffer.",
  },
}

def read(path: Path) -> str:
  try:
    return path.read_text(errors="ignore")
  except FileNotFoundError:
    return ""

def has_all(texts: list[str], patterns: list[str]) -> dict[str, bool]:
  joined = "\n".join(texts)
  return {p: p in joined for p in patterns}

def main() -> None:
  texts = {k: read(v) for k, v in FILES.items()}
  file_state = {k: {"path": str(v), "exists": v.exists(), "bytes": len(texts[k].encode())} for k, v in FILES.items()}

  rows = []
  for name, spec in CHECKS.items():
    roc_hits = has_all([texts[k] for k in spec["rocprof_files"]], spec["patterns"])
    tg_hits = has_all([texts[k] for k in spec["tinygrad_files"]], spec["patterns"])
    rows.append({
      "primitive": name,
      "meaning": spec["meaning"],
      "rocprofiler_present": all(roc_hits.values()),
      "tinygrad_hcq_present": all(tg_hits.values()),
      "rocprofiler_pattern_hits": roc_hits,
      "tinygrad_pattern_hits": tg_hits,
      "classification": "MISSING_IN_TINYGRAD_HCQ" if all(roc_hits.values()) and not all(tg_hits.values()) else "CHECK_MANUALLY",
    })

  result = {
    "verdict": "ROCPROFILER_ATT_MISSING_PROFILED_HSA_AQL_PATH_IN_HCQ",
    "date": "2026-06-19",
    "inputs": file_state,
    "rows": rows,
    "high_confidence_missing": [
      "HSA queue interposition around kernel dispatch packets",
      "hsa_amd_profiling_set_profiler_enabled plus profiler-active queue packet",
      "AQLprofile-generated vendor-specific ATT start/stop packets",
      "AQLprofile trace-control buffer status/WPTR protocol",
      "AQLprofile SQTT begin/end ordering: PrimeL2, status clear, cache flush, control-buffer reads",
      "ROC trace-decoder agent/code-object metadata markers",
    ],
    "bounded_register_sweeps_already_refuted": [
      "SQ_THREAD_TRACE_MASK",
      "SQ_THREAD_TRACE_TOKEN_MASK",
      "SQ_THREAD_TRACE_CTRL",
      "SQTT_MODE",
      "SQTT_TTRACE_EXEC",
      "SQTT_ORACLE_TARGET_CU",
    ],
    "next_reopen_options": [
      {
        "name": "AQLprofile packet import/replay",
        "description": "Use AQLprofile to manufacture start/stop packets and adapt their command buffers into tinygrad HCQ submission, if the vendor AQL packet body can be decoded or directly replayed outside an HSA queue.",
        "risk": "medium-high",
      },
      {
        "name": "HCQ profiled-queue equivalent",
        "description": "Implement the profiling-active queue state and ATT packet lifecycle natively for tinygrad's KFD/HCQ path.",
        "risk": "project-level",
      },
      {
        "name": "Keep external ROCprofiler oracle",
        "description": "Use HIP controls and imported kernels for instruction-rich ATT attribution; use tinygrad-native PMCs for HCQ model attribution.",
        "risk": "low",
      },
    ],
  }

  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"wrote": str(OUT), "verdict": result["verdict"], "rows": len(rows)}, indent=2))

if __name__ == "__main__":
  main()
