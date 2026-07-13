"""Compile-only evidence capture for the pure register-resident prefill route.

This module consumes an already lowered compiler result.  It has no runtime,
device-program, or dispatch entry point: callers must provide final source,
binary, disassembly, descriptor facts, and allocator intervals explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, re
from typing import Any, Mapping

from tinygrad.codegen.opt.amd_resource_artifact import (AMDPhysicalInterval, AMDResourceFacts,
  join_amd_resource_artifact)

SCHEMA = "prefill-pure-register-compile.v1"
PROOF_SCHEMA = "prefill-pure-register-instruction-order.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOAD = re.compile(r"\b(?:global|flat|buffer)_load\w*\b", re.I)
_WAIT = re.compile(r"\bs_waitcnt\b[^\n]*\bvmcnt\s*\(\s*0\s*\)", re.I)
_WMMA = re.compile(r"\bv_wmma\w*\b", re.I)
_LDS = re.compile(r"\bds_(?:read|write|load|store)\w*\b", re.I)
_PACK_DEST = re.compile(r"\bv_pack_b32_f16\s+v(\d+)\b", re.I)
_MOV_SOURCE = re.compile(r"\bv_mov_b32\w*\s+v\d+\s*,\s*v(\d+)\b", re.I)


def _final_field(program: Any, name: str) -> Any:
  """Read a field from a final-program record without invoking compiler/runtime APIs."""
  if isinstance(program, Mapping): return program.get(name)
  return getattr(program, name, None)


def _required_final_mapping(program: Any, name: str, authority: str) -> Mapping[str, Any]:
  value = _final_field(program, name)
  if not isinstance(value, Mapping): raise ValueError(f"final program lacks {name} mapping")
  if value.get("authority") != authority:
    raise ValueError(f"{name} lacks authoritative {authority} descriptor")
  return value


def capture_final_program_compile_only(program: Any, *, pipeline: Mapping[str, Any], wait: Mapping[str, Any],
                                      abi_contract: Mapping[str, Any], surface: Mapping[str, Any],
                                      runtime_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Adapt one actual final AMD ISA assembler result into the existing capture gate.

  ``program`` is a mapping or object produced by the assembler/compiler and must
  expose ``candidate_identity``, ``target``, ``abi``, ``source``, ``binary``,
  ``disassembly``, plus ``descriptor`` and ``allocator`` mappings.  The latter
  must carry ``authority`` values ``final_code_object_descriptor`` and
  ``final_regalloc`` respectively.  Descriptor ``resources`` and allocator
  ``intervals`` may be typed objects or their exact serialized JSON forms.
  Missing or estimated fields are rejected; this adapter never compiles,
  disassembles, allocates, or dispatches.
  """
  if isinstance(program, (str, bytes)) or program is None:
    raise TypeError("final AMD program record/object is required")
  descriptor = _required_final_mapping(program, "descriptor", "final_code_object_descriptor")
  allocator = _required_final_mapping(program, "allocator", "final_regalloc")
  resources = descriptor.get("resources")
  if isinstance(resources, Mapping):
    try:
      resources = AMDResourceFacts(**{name: resources.get(name) for name in
        ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills",
         "workgroup_threads", "wavefront_size")})
    except (TypeError, ValueError) as exc:
      raise ValueError("final descriptor lacks typed resources") from exc
  if not isinstance(resources, AMDResourceFacts):
    raise ValueError("final descriptor lacks typed resources")
  rows = allocator.get("intervals")
  if isinstance(rows, list):
    try: rows = tuple(AMDPhysicalInterval(x["logical_role"], x["bank"], x["start"], x["end"], x.get("purpose", "value")) for x in rows)
    except (KeyError, TypeError, ValueError) as exc: raise ValueError("final allocator intervals are malformed") from exc
  if not isinstance(rows, tuple) or not rows or any(not isinstance(x, AMDPhysicalInterval) for x in rows):
    raise ValueError("final allocator lacks typed intervals")
  fields = {name: _final_field(program, name) for name in
            ("candidate_identity", "target", "abi", "source", "binary", "disassembly")}
  if any(value is None for value in fields.values()): raise ValueError("final AMD program record is incomplete")
  evidence = FinalCompileEvidence(**fields, resources=resources, intervals=rows)
  return capture_compile_only(evidence, pipeline=pipeline, wait=wait, abi_contract=abi_contract,
                              surface=surface, runtime_binding=runtime_binding)


@dataclass(frozen=True)
class FinalCompileEvidence:
  """All authorities needed to capture one final program without executing it."""
  candidate_identity: str
  target: str
  abi: str
  source: str
  binary: bytes
  disassembly: str
  resources: AMDResourceFacts
  intervals: tuple[AMDPhysicalInterval, ...]
  resource_authority: str = "final_code_object_descriptor"
  allocator_authority: str = "final_regalloc"

  def __post_init__(self):
    if _SHA256.fullmatch(self.candidate_identity) is None: raise ValueError("candidate identity must be lowercase SHA-256")
    if not all(isinstance(x, str) and x for x in (self.target, self.abi, self.source, self.disassembly)):
      raise ValueError("target, ABI, source, and disassembly must be non-empty")
    if not isinstance(self.binary, bytes) or not self.binary: raise ValueError("final binary must be non-empty bytes")
    if self.target != "gfx1100" or self.abi != "amdgpu_kernel":
      raise ValueError("milestone one requires gfx1100 amdgpu_kernel ABI")
    if not isinstance(self.resources, AMDResourceFacts): raise TypeError("resources must be final AMDResourceFacts")
    if not isinstance(self.intervals, tuple) or not self.intervals: raise ValueError("final allocator intervals are required")
    if self.resource_authority != "final_code_object_descriptor": raise ValueError("host-estimated resources are forbidden")
    if self.allocator_authority != "final_regalloc": raise ValueError("non-final allocator intervals are forbidden")


def instruction_order_proof(disassembly: str, intervals: tuple[AMDPhysicalInterval, ...]) -> dict[str, Any]:
  """Prove final instruction order and absence of LDS from final disassembly."""
  if not isinstance(disassembly, str) or not disassembly.strip(): raise ValueError("final disassembly is required")
  if not isinstance(intervals, tuple) or not intervals: raise ValueError("final allocator intervals are required")
  lines = tuple(line.strip() for line in disassembly.splitlines() if line.strip())
  stage_regs = {reg for row in intervals if row.bank.value == "vgpr" and row.logical_role in ("A", "B")
                for reg in range(row.start, row.end)}
  def stage_write(line: str) -> bool:
    return (match := _PACK_DEST.search(line)) is not None and int(match.group(1)) in stage_regs
  def stage_read(line: str) -> bool:
    return (match := _MOV_SOURCE.search(line)) is not None and int(match.group(1)) in stage_regs
  ab = tuple(row for row in intervals if row.logical_role in ("A", "B"))
  direct = bool(ab) and all(row.purpose == "direct_wmma_fragment" for row in ab)
  patterns = (((_LOAD.search, "global_load"), (_WAIT.search, "vmcnt0_wait"), (_WMMA.search, "wmma")) if direct else
              ((_LOAD.search, "global_load"), (_WAIT.search, "vmcnt0_wait"), (stage_write, "stage_write"),
               (stage_read, "stage_read"), (_WMMA.search, "wmma")))
  positions: dict[str, int] = {}
  cursor = 0
  for matches, name in patterns:
    found = next((idx for idx in range(cursor, len(lines)) if matches(lines[idx])), None)
    if found is None: break
    positions[name], cursor = found, found + 1
  missing = [name for _, name in patterns if name not in positions]
  lds_lines = [idx for idx, line in enumerate(lines) if _LDS.search(line)]
  errors = ([f"final disassembly lacks ordered {name}" for name in missing] +
            (["final disassembly contains LDS transport"] if lds_lines else []))
  return {"schema": PROOF_SCHEMA, "authority": "final_disassembly", "passed": not errors, "errors": errors,
          "fragment_transport": "direct_global" if direct else "staged_register",
          "disassembly_sha256": hashlib.sha256(disassembly.encode()).hexdigest(),
          "positions": positions, "lds_instruction_lines": lds_lines}


def capture_compile_only(evidence: FinalCompileEvidence, *, pipeline: Mapping[str, Any], wait: Mapping[str, Any],
                         abi_contract: Mapping[str, Any], surface: Mapping[str, Any],
                         runtime_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Create immutable, gate-compatible evidence.  This function never dispatches a GPU."""
  if not isinstance(evidence, FinalCompileEvidence): raise TypeError("FinalCompileEvidence is required")
  for name, value in (("pipeline", pipeline), ("wait", wait), ("abi_contract", abi_contract), ("surface", surface)):
    if not isinstance(value, Mapping): raise TypeError(f"{name} must be a mapping")
  resource = join_amd_resource_artifact(target=evidence.target, abi=evidence.abi, source=evidence.source,
    binary=evidence.binary, candidate_identity=evidence.candidate_identity, resources=evidence.resources,
    intervals=evidence.intervals)
  proof = instruction_order_proof(evidence.disassembly, evidence.intervals)
  errors = list(proof["errors"])
  if evidence.resources.lds_bytes != 0: errors.append("final program uses LDS")
  if evidence.resources.scratch_bytes != 0: errors.append("final program uses scratch")
  if evidence.resources.vgpr_spills or evidence.resources.sgpr_spills: errors.append("final program spills registers")
  row = {"schema": SCHEMA, "canonical_identity": evidence.candidate_identity,
         "binary_sha256": resource.binary_sha256, "passed": not errors, "errors": errors,
         "program": {"source_sha256": resource.source_sha256, "binary_sha256": resource.binary_sha256,
                     "target": evidence.target, "abi": evidence.abi},
         "target_evidence": {"authority": "final_program", "target": evidence.target, "abi": evidence.abi},
         "surface": dict(surface), "pipeline": dict(pipeline), "wait": dict(wait), "abi": dict(abi_contract),
         "resource_artifact": resource.to_json(), "instruction_order_proof": proof,
         "capture": {"mode": "compile_only", "dispatch_permitted": False,
                     "resource_authority": evidence.resource_authority,
                     "allocator_authority": evidence.allocator_authority}}
  if runtime_binding is not None: row["runtime_binding"] = dict(runtime_binding)
  return row
