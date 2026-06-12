from __future__ import annotations

from tinygrad.helpers import DEV, OSX, getenv

def q4k_device_label(device:str|None=None) -> str:
  return (device or repr(DEV) or getenv("DEV", "")).upper()

def q4k_remote_risk_reason(device:str|None=None) -> str|None:
  label = q4k_device_label(device)
  if OSX: return "macOS/TinyGPU path"
  if getenv("REMOTE", ""): return "REMOTE PCI path"
  if getenv("APL_REMOTE_SOCK", ""): return "TinyGPU APL remote socket path"
  if any(x in label for x in ("PCI", "REMOTE", "TINYGPU", "APL")):
    return f"remote-risk device {label!r}"
  if not label or not label.startswith("AMD"):
    return f"non-native AMD device {label or '<default>'!r}"
  return None

def assert_q4k_native_sweep_allowed(device:str|None=None, what:str="Q4_K sweep") -> None:
  if (reason := q4k_remote_risk_reason(device)) is not None:
    raise RuntimeError(f"Refusing {what} on {reason}. Run Q4_K search/tuning only on native Ubuntu AMD.")

def assert_q4k_risky_search_allowed(device:str|None=None, what:str="Q4_K risky search") -> None:
  if getenv("Q4K_ALLOW_RISKY_SEARCH", 0) != 1:
    raise RuntimeError(f"{what} is disabled by default. Set Q4K_ALLOW_RISKY_SEARCH=1 only on native Ubuntu AMD.")
  assert_q4k_native_sweep_allowed(device, what)
