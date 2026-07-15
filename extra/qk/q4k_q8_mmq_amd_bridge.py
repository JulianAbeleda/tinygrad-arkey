"""Reproducible, fail-closed launcher configuration for the generated MMQ harness.

This module only prepares an invocation.  It never imports a live AMD device,
constructs a program, or dispatches work.  The harness remains the owner of
spawn isolation and guarded execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AMDExecutionConfig:
  rocm_root: Path = Path("/opt/rocm")
  device: str = "AMD"
  timeout_seconds: float = 30.0

  def __post_init__(self) -> None:
    if self.device != "AMD":
      raise ValueError("the MMQ bridge only admits DEV=AMD")
    if self.timeout_seconds <= 0:
      raise ValueError("timeout_seconds must be positive")

  @property
  def hipcc(self) -> Path:
    return self.rocm_root / "bin" / "hipcc"

  @property
  def clang(self) -> Path:
    return self.rocm_root / "llvm" / "bin" / "clang"

  def environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    pythonpath = env.get("PYTHONPATH", "")
    env.update({"DEV": self.device, "ROCM_PATH": str(self.rocm_root),
                "HIP_PATH": str(self.rocm_root), "HIPCC": str(self.hipcc),
                "PYTHONPATH": str(ROOT) if not pythonpath else f"{ROOT}{os.pathsep}{pythonpath}"})
    return env

  def preflight(self, *, runner=subprocess.run) -> dict[str, object]:
    """Report tools and GPU visibility without creating a tinygrad device."""
    tools = {name: shutil.which(name) for name in ("rocminfo", "hipcc")}
    tools["rocm_hipcc"] = str(self.hipcc) if self.hipcc.is_file() else None
    tools["rocm_clang"] = str(self.clang) if self.clang.is_file() else None
    info = runner(("rocminfo",), capture_output=True, text=True, check=False, timeout=10)
    text = (info.stdout or "") + "\n" + (info.stderr or "")
    gpu_agents = [line.strip() for line in text.splitlines()
                  if "Device Type:" in line and "GPU" in line]
    return {"rocm_root": str(self.rocm_root), "tools": tools,
            "gpu_agent_visible": bool(gpu_agents),
            "gpu_agents": gpu_agents,
            "blocker": None if gpu_agents else "rocminfo exposes no AMD GPU agent"}

  def command(self, *, bootstrap: str | Path, dispatch: bool = False) -> list[str]:
    """Build an explicit launcher command; dispatch is opt-in and guarded."""
    if not dispatch:
      raise ValueError("dispatch is opt-in: pass dispatch=True explicitly")
    bootstrap_path = Path(bootstrap)
    if not bootstrap_path.is_file():
      raise FileNotFoundError(bootstrap_path)
    return [sys.executable, "-m", "extra.qk.q4k_q8_mmq_generated_harness",
            "--bootstrap", str(bootstrap_path), "--dispatch", "--timeout", str(self.timeout_seconds)]


def guarded_dispatch_command(config: AMDExecutionConfig, *, bootstrap: str | Path,
                             base_env: Mapping[str, str] | None = None) -> tuple[list[str], dict[str, str]]:
  """Return a command only when preflight sees a GPU and device use is enabled."""
  env = config.environment(base_env)
  if env.get("ALLOW_DEVICE_USAGE") != "1":
    raise RuntimeError("refusing dispatch: ALLOW_DEVICE_USAGE=1 is required")
  if not config.preflight()["gpu_agent_visible"]:
    raise RuntimeError("refusing dispatch: rocminfo exposes no AMD GPU agent")
  return config.command(bootstrap=bootstrap, dispatch=True), env
