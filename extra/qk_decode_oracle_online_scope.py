#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_online_scope_result.json"


def tool(name: str, *fallbacks: str) -> str | None:
  for item in (name, *fallbacks):
    found = shutil.which(item)
    if found: return found
    p = Path(item)
    if p.exists(): return str(p)
  return None


def main() -> int:
  source_basis = [
    {
      "source": "LLVM AMDGPU Usage",
      "url": "https://llvm.org/docs/AMDGPUUsage.html",
      "usable_fact": "AMDGPU code objects expose metadata/note records, kernel descriptors, and next-free VGPR/SGPR symbols/directives.",
      "maps_to": ["NINFO-1", "NINFO-2", "NINFO-4"],
    },
    {
      "source": "ROCm workload optimization",
      "url": "https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html",
      "usable_fact": "Saved ROCm code objects can be disassembled with llvm-objdump --disassemble-all.",
      "maps_to": ["NINFO-2"],
    },
    {
      "source": "ROCprofiler-SDK rocprofv3",
      "url": "https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html",
      "usable_fact": "Kernel trace output includes LDS_Block_Size, Scratch_Size, VGPR_Count, Accum_VGPR_Count, SGPR_Count, workgroup, and grid fields.",
      "maps_to": ["NINFO-1", "NINFO-6"],
    },
    {
      "source": "ROCprofiler-SDK thread trace",
      "url": "https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.1.1/how-to/using-thread-trace.html",
      "usable_fact": "Thread trace targets instruction timing, execution path, wave scheduling, stall timing, and hotspots.",
      "maps_to": ["NINFO-3", "NINFO-5"],
    },
    {
      "source": "ROCprof Compute Viewer",
      "url": "https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/amd-mainline/how-to/using_compute_viewer.html",
      "usable_fact": "Viewer output covers ISA source visualization, hotspots, memory-op to waitcnt dependencies, and occupancy visualization.",
      "maps_to": ["NINFO-3", "NINFO-4", "NINFO-5"],
    },
  ]

  tool_inventory = {
    "llvm_objdump": tool("llvm-objdump", "/opt/rocm/llvm/bin/llvm-objdump", "/opt/rocm-7.2.4/llvm/bin/llvm-objdump"),
    "llvm_readobj": tool("llvm-readobj", "/opt/rocm/llvm/bin/llvm-readobj", "/opt/rocm-7.2.4/llvm/bin/llvm-readobj"),
    "llvm_objcopy": tool("llvm-objcopy", "/opt/rocm/llvm/bin/llvm-objcopy", "/opt/rocm-7.2.4/llvm/bin/llvm-objcopy"),
    "clang_offload_bundler": tool("clang-offload-bundler", "/opt/rocm/llvm/bin/clang-offload-bundler", "/opt/rocm-7.2.4/llvm/bin/clang-offload-bundler"),
    "rocprofv3": tool("rocprofv3", "/opt/rocm/bin/rocprofv3", "/opt/rocm-7.2.4/bin/rocprofv3"),
    "rocprof_compute_viewer": tool("rocprof-compute-viewer"),
  }

  phases: list[dict[str, Any]] = [
    {
      "phase": "OES-1 oracle kernel identification",
      "goal": "Identify the exact HIP/LLD q8 gate/up oracle kernel symbol and code object used by the measured oracle row.",
      "online_basis": ["rocprofv3 kernel trace reports kernel names and launch/resource fields"],
      "commands": [
        "rocprofv3 --kernel-trace --output-format csv -- <oracle runner>",
        "parse kernel_trace.csv for q8/MMVQ/gate/up candidate rows",
      ],
      "outputs": ["kernel symbol", "dispatch id", "code object path or in-memory URI", "launch grid/workgroup", "VGPR/SGPR/LDS/scratch fields if present"],
      "pass_gate": "one q8 gate/up oracle dispatch accounts for the oracle timing row and has a stable symbol/code-object identity",
      "unblocks": ["NINFO-1", "NINFO-6"],
    },
    {
      "phase": "OES-2 code object extraction",
      "goal": "Recover the loadable gfx1100 oracle code object for that symbol.",
      "online_basis": ["ROCm docs show saved code objects can be disassembled with llvm-objdump --disassemble-all"],
      "commands": [
        "if bundled: clang-offload-bundler --unbundle --targets=hipv4-amdgcn-amd-amdhsa--gfx1100",
        "if runtime URI: copy saved in-memory code object from profiler/debugger output",
        "sha256sum <oracle.co>",
      ],
      "outputs": ["code object path", "sha256", "ELF header/e_flags", "symbol list"],
      "pass_gate": "llvm-objdump can read the recovered code object and the selected symbol is present",
      "unblocks": ["NINFO-1", "NINFO-2"],
    },
    {
      "phase": "OES-3 metadata and descriptor extraction",
      "goal": "Extract resource envelope from code-object metadata, notes, descriptors, and/or rocprofv3 kernel trace fields.",
      "online_basis": ["LLVM documents AMDGPU metadata/note records and next-free VGPR/SGPR symbols", "rocprofv3 kernel trace includes VGPR/SGPR/LDS/scratch columns"],
      "commands": [
        "llvm-readobj --notes --symbols --sections <oracle.co>",
        "llvm-objdump --syms --section-headers <oracle.co>",
        "parse rocprofv3 kernel_trace.csv resource columns for selected dispatch",
      ],
      "outputs": ["VGPR count", "SGPR count", "accum VGPR count", "LDS block size", "scratch size", "kernarg size", "workgroup", "grid", "descriptor symbols"],
      "pass_gate": "oracle resource envelope is comparable against native, best-static, and C7C-best in one table",
      "unblocks": ["NINFO-1", "NINFO-4", "NINFO-6"],
    },
    {
      "phase": "OES-4 semantic ISA map",
      "goal": "Disassemble the selected oracle kernel and annotate it into q8 semantic stages.",
      "online_basis": ["ROCm docs prescribe llvm-objdump --disassemble-all for saved code objects"],
      "commands": [
        "llvm-objdump --disassemble-all --no-show-raw-insn <oracle.co> > oracle.disasm",
        "slice selected symbol range",
        "classify instructions into load, unpack/select, dot4, scale/min, reduction, wait, branch, store",
      ],
      "outputs": ["ordered ISA table", "stage boundaries", "operand register map", "waitcnt reasons", "branch/predicate purpose"],
      "pass_gate": "one unimplemented semantic mechanism is named, or oracle reduces to already-tested native patterns",
      "unblocks": ["NINFO-2", "NINFO-4"],
    },
    {
      "phase": "OES-5 PC timeline and stall attribution",
      "goal": "Use thread trace/ATT to map stalls and hotspots back to q8 oracle or native PCs.",
      "online_basis": ["thread trace provides instruction timing/path/stall timing", "Compute Viewer shows ISA, hotspots, waitcnt dependencies, and occupancy"],
      "commands": [
        "rocprofv3 --att --kernel-trace -- <oracle runner>",
        "import ui_output_agent_*_dispatch_* into rocprof-compute-viewer or parse exported trace-decoder artifacts",
        "join PCs to oracle.disasm",
      ],
      "outputs": ["PC-to-ISA join", "stage stall histogram", "waitcnt dependency evidence", "occupancy view"],
      "pass_gate": "one stall stage or dependency family with plausible >=30us movement is identified",
      "unblocks": ["NINFO-3", "NINFO-5"],
    },
    {
      "phase": "OES-6 fair oracle comparison",
      "goal": "Run oracle, native, best-static, and C7C-best under one timing/clock methodology.",
      "online_basis": ["rocprofv3 kernel trace provides HIP-visible oracle launch/resource data; native uses tinygrad HCQ timing/PMC"],
      "commands": [
        "run oracle HIP artifact and native candidates in as close to one-clock interleaved setup as possible",
        "record clock provenance and resource metadata for each row",
      ],
      "outputs": ["same-run timing table", "resource table", "clock provenance", "correctness policy"],
      "pass_gate": "oracle target remains stable and the remaining gap is assigned to body schedule, resource envelope, launch/runtime, or route policy",
      "unblocks": ["NINFO-6", "NINFO-7"],
    },
  ]

  deliverables = [
    "bench/qk-decode-primitive-transfer/oracle_kernel_trace.csv",
    "bench/qk-decode-primitive-transfer/oracle_code_object_manifest.json",
    "bench/qk-decode-primitive-transfer/oracle_resource_envelope.json",
    "bench/qk-decode-primitive-transfer/oracle_semantic_isa_map.json",
    "bench/qk-decode-primitive-transfer/oracle_thread_trace_summary.json",
    "docs/decode-oracle-extraction-result-YYYYMMDD.md",
  ]

  gates = {
    "sources_cover_code_object_metadata": True,
    "sources_cover_disassembly": True,
    "sources_cover_kernel_trace_resources": True,
    "sources_cover_thread_trace_pc_timeline": True,
    "phases_defined": len(phases) == 6,
    "reopen_mapping_defined": all(row["unblocks"] for row in phases),
  }

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_ORACLE_EXTRACTION_SCOPE_FROM_ONLINE_SOURCES",
    "schema": "decode_oracle_online_scope_v1",
    "verdict": "SCOPE_DECODE_ORACLE_EXTRACTION_READY" if all(gates.values()) else "BLOCKED_DECODE_ORACLE_EXTRACTION_SCOPE_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "source_basis": source_basis,
    "tool_inventory": tool_inventory,
    "phases": phases,
    "deliverables": deliverables,
    "decision_policy": {
      "resume_native_only_if": "one DNR-3C9 reopen gate passes from OES outputs",
      "otherwise": "keep native DNR-3C parked and move to route-level decode work",
      "do_not_do": [
        "do not use static opcode similarity as the search objective",
        "do not start BEAM/search before OES names a measurable objective",
        "do not promote a native candidate unless timing and oracle/resource evidence agree",
      ],
    },
    "gates": gates,
  }

  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "tool_inventory": tool_inventory,
    "phases": [row["phase"] for row in phases],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
