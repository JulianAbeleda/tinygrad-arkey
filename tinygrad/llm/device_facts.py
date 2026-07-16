from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib, json, re, subprocess
from typing import Any, Callable, Mapping


SCHEMA_VERSION = 1
Probe = Callable[[str], Mapping[str, Any]]


def _optional_int(value: Any) -> int | None:
  if value is None: return None
  value = int(value)
  return value if value >= 0 else None


@dataclass(frozen=True)
class DeviceCapabilities:
  wave_size: int | None = None
  max_workgroup_threads: int | None = None
  max_workgroup_dimensions: tuple[int, int, int] | None = None
  lds_bytes: int | None = None
  lds_allocation_granularity: int | None = None
  global_allocation_granularity: int | None = None

  def to_json(self) -> dict[str, Any]: return asdict(self)

  @classmethod
  def from_json(cls, value: Mapping[str, Any]) -> DeviceCapabilities:
    dims = value.get("max_workgroup_dimensions")
    return cls(*(_optional_int(value.get(k)) for k in ("wave_size", "max_workgroup_threads")),
               None if dims is None else tuple(int(x) for x in dims),
               _optional_int(value.get("lds_bytes")), _optional_int(value.get("lds_allocation_granularity")),
               _optional_int(value.get("global_allocation_granularity")))


@dataclass(frozen=True)
class ProbeRecord:
  source: str
  observed_at: str
  state: str = "ok"
  error: str | None = None

  def __post_init__(self):
    if self.state not in ("ok", "partial", "unknown", "error"): raise ValueError(f"invalid probe state {self.state!r}")

  def to_json(self) -> dict[str, Any]: return asdict(self)
  @classmethod
  def from_json(cls, value: Mapping[str, Any]) -> ProbeRecord: return cls(**value)


@dataclass(frozen=True)
class DeviceFacts:
  selected_device: str
  backend: str | None
  architecture: str | None
  total_vram_bytes: int | None
  free_vram_bytes: int | None
  capabilities: DeviceCapabilities
  target_probe: ProbeRecord
  memory_probe: ProbeRecord
  errors: tuple[str, ...] = ()
  schema_version: int = SCHEMA_VERSION

  @property
  def state(self) -> str:
    if self.errors or self.target_probe.state == "error" or self.memory_probe.state == "error": return "error"
    required = (self.backend, self.architecture, self.total_vram_bytes, self.free_vram_bytes)
    return "ok" if all(x is not None for x in required) else "unknown"

  def canonical_hardware(self) -> dict[str, Any]:
    """Stable machine facts only: deliberately excludes free VRAM and probe metadata."""
    return {"schema_version": self.schema_version, "selected_device": self.selected_device, "backend": self.backend,
            "architecture": self.architecture, "total_vram_bytes": self.total_vram_bytes,
            "capabilities": self.capabilities.to_json()}

  @property
  def canonical_hardware_identity(self) -> str:
    encoded = json.dumps(self.canonical_hardware(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

  def planning_snapshot(self) -> dict[str, Any]:
    # Material live availability belongs in the search key, observation time
    # does not. This lets an exact-fact cache survive a repeated scan when the
    # hardware and free bytes are unchanged.
    return {**self.canonical_hardware(), "canonical_hardware_identity": self.canonical_hardware_identity,
            "free_vram_bytes": self.free_vram_bytes, "state": self.state,
            "target_probe": {"source": self.target_probe.source, "state": self.target_probe.state,
                             "error": self.target_probe.error},
            "memory_probe": {"source": self.memory_probe.source, "state": self.memory_probe.state,
                             "error": self.memory_probe.error}, "errors": list(self.errors)}

  def to_json(self) -> dict[str, Any]:
    return {"schema_version": self.schema_version, "selected_device": self.selected_device, "backend": self.backend,
            "architecture": self.architecture, "total_vram_bytes": self.total_vram_bytes,
            "free_vram_bytes": self.free_vram_bytes, "capabilities": self.capabilities.to_json(),
            "target_probe": self.target_probe.to_json(), "memory_probe": self.memory_probe.to_json(), "errors": list(self.errors)}

  @classmethod
  def from_json(cls, value: Mapping[str, Any]) -> DeviceFacts:
    return cls(str(value["selected_device"]), value.get("backend"), value.get("architecture"),
               _optional_int(value.get("total_vram_bytes")), _optional_int(value.get("free_vram_bytes")),
               DeviceCapabilities.from_json(value.get("capabilities", {})), ProbeRecord.from_json(value["target_probe"]),
               ProbeRecord.from_json(value["memory_probe"]), tuple(str(x) for x in value.get("errors", ())),
               int(value.get("schema_version", SCHEMA_VERSION)))


def scan_device_facts(selected_device: str | None = None, *, target_probe: Probe | None = None,
                      memory_probe: Probe | None = None, clock: Callable[[], datetime] | None = None) -> DeviceFacts:
  device = selected_device or _selected_device()
  now = (clock or (lambda: datetime.now(timezone.utc)))().astimezone(timezone.utc).isoformat()
  target, target_record = _run_probe(target_probe or _tinygrad_target_probe, device, "tinygrad-device", now)
  memory, memory_record = _run_probe(memory_probe or _rocm_smi_memory_probe, device, "rocm-smi", now)
  errors: list[str] = []
  try: total, free = _optional_int(memory.get("total_vram_bytes")), _optional_int(memory.get("free_vram_bytes"))
  except (TypeError, ValueError, OverflowError) as exc:
    total = free = None
    errors.append(f"invalid VRAM probe values: {exc}")
  if total is not None and free is not None and free > total: errors.append(f"free VRAM ({free}) exceeds total VRAM ({total})")
  caps = DeviceCapabilities.from_json(target.get("capabilities", target))
  backend = target.get("backend")
  architecture = target.get("architecture", target.get("arch"))
  return DeviceFacts(device, None if backend is None else str(backend), None if architecture is None else str(architecture),
                     total, free, caps, target_record, memory_record, tuple(errors))


def _run_probe(probe: Probe, device: str, source: str, now: str) -> tuple[Mapping[str, Any], ProbeRecord]:
  try:
    value = probe(device)
    if not isinstance(value, Mapping): raise TypeError("probe result is not a mapping")
    supplied_source = str(value.get("provenance", source))
    return value, ProbeRecord(supplied_source, now, "ok" if value else "unknown")
  except Exception as exc: return {}, ProbeRecord(source, now, "error", f"{type(exc).__name__}: {exc}")


def _selected_device() -> str:
  from tinygrad.device import Device
  return Device.DEFAULT


def _tinygrad_target_probe(device: str) -> Mapping[str, Any]:
  from tinygrad.device import Device
  opened = Device[device]
  renderer = getattr(opened, "renderer", None)
  backend = device.split(":", 1)[0].upper()
  arch = next((getattr(obj, name, None) for obj in (opened, renderer) if obj is not None
               for name in ("arch", "architecture") if getattr(obj, name, None) is not None), None)
  # Renderer fields are the compiler's effective limits.  Some renderers do
  # not expose wave/workgroup facts, so augment AMD with the selected HSA
  # agent reported by rocminfo.  These are observations, not architecture
  # lookup-table defaults: an unavailable probe remains unknown and candidate
  # capability matching must fail closed.
  capabilities = {
    "wave_size": getattr(renderer, "wave_size", None),
    "max_workgroup_threads": getattr(renderer, "max_workgroup_threads", None),
    "max_workgroup_dimensions": getattr(renderer, "max_workgroup_dimensions", None),
    "lds_bytes": getattr(renderer, "shared_max", None),
    "lds_allocation_granularity": getattr(renderer, "lds_allocation_granularity", None),
    "global_allocation_granularity": getattr(getattr(opened, "allocator", None), "allocation_granularity", None),
  }
  provenance = "tinygrad-device"
  if backend == "AMD":
    try:
      proc = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10, check=True)
      ordinal = int(device.split(":", 1)[1]) if ":" in device else 0
      observed = _parse_rocminfo_gpu_capabilities(proc.stdout, ordinal)
      capabilities = {name: value if value is not None else observed.get(name)
                      for name, value in capabilities.items()}
      provenance += " + rocminfo"
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
      pass
  return {"backend": backend, "architecture": arch, **capabilities, "provenance": provenance}


def _parse_rocminfo_gpu_capabilities(output: str, ordinal: int) -> dict[str, Any]:
  if ordinal < 0: raise ValueError("GPU ordinal must be non-negative")
  # Agent headers delimit CPU and GPU records.  Select by GPU ordinal so
  # AMD:N is resolved from runtime discovery rather than an architecture name.
  agents = re.split(r"(?m)^\*{7}\s*\nAgent\s+\d+\s*\n\*{7}\s*$", output)
  gpu_agents = [row for row in agents if re.search(r"(?m)^\s*Device Type:\s*GPU\s*$", row)]
  if ordinal >= len(gpu_agents): raise IndexError(f"rocminfo has no GPU ordinal {ordinal}")
  row = gpu_agents[ordinal]
  def field(label: str) -> int | None:
    match = re.search(rf"(?m)^\s*{re.escape(label)}:\s*(\d+)", row)
    return None if match is None else int(match.group(1))
  dims = None
  dim_block = re.search(r"Workgroup Max Size per Dimension:\s*\n"
                        r"\s*x\s+(\d+).*\n\s*y\s+(\d+).*\n\s*z\s+(\d+)", row)
  if dim_block is not None: dims = tuple(int(x) for x in dim_block.groups())
  # GROUP pool size is reported in KB.  Renderer.shared_max, when present,
  # remains the effective compiler limit and therefore takes precedence.
  group = re.search(r"Segment:\s*GROUP.*?\n\s*Size:\s*(\d+)", row, re.S)
  global_pool = re.search(r"Segment:\s*GLOBAL.*?\n\s*Alloc Granule:\s*(\d+)\s*(KB|MB|B)", row, re.S|re.I)
  global_granularity = None
  if global_pool is not None:
    scale = {"B": 1, "KB": 1024, "MB": 1024*1024}[global_pool.group(2).upper()]
    global_granularity = int(global_pool.group(1))*scale
  return {"wave_size": field("Wavefront Size"), "max_workgroup_threads": field("Workgroup Max Size"),
          "max_workgroup_dimensions": dims, "lds_bytes": None if group is None else int(group.group(1))*1024,
          "lds_allocation_granularity": None, "global_allocation_granularity": global_granularity}


def _rocm_smi_memory_probe(device: str) -> Mapping[str, Any]:
  proc = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=10, check=True)
  index = int(device.split(":", 1)[1]) if ":" in device else 0
  card = re.compile(rf"^\s*(?:GPU\[?{index}\]?|card{index})\s*[: ]", re.I)
  lines = [line for line in proc.stdout.splitlines() if card.search(line)] or proc.stdout.splitlines()
  total = used = None
  for line in lines:
    match = re.search(r"VRAM Total (Used )?Memory.*?:\s*(\d+)", line, re.I)
    if match:
      if match.group(1): used = int(match.group(2))
      else: total = int(match.group(2))
  return {"total_vram_bytes": total, "free_vram_bytes": None if total is None or used is None else total-used,
          "provenance": "rocm-smi --showmeminfo vram"}


# Explicit alias for callers that prefer the autoscan terminology used by the planner documentation.
autoscan_device_facts = scan_device_facts


__all__ = ["DeviceCapabilities", "DeviceFacts", "ProbeRecord", "autoscan_device_facts", "scan_device_facts"]
