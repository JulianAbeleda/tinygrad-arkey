#!/usr/bin/env python3
"""Central prefill authority harness policy.

This module is import-light: no tinygrad import and no GPU access. It owns the
profiles and subprocess shape for whole-prefill timing so callers do not clone
smoke/authority defaults. Runtime route selection belongs to the loaded plan.
"""
from __future__ import annotations

from dataclasses import dataclass
import pathlib

from extra.qk.model_profiles import MODEL_PROFILES, ModelProfile, profile_by_id, profile_from_model_path

ROOT = pathlib.Path(__file__).resolve().parents[2]

DEFAULT_MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
DEFAULT_MODEL_PROFILE = "qwen3_8b_q4k_m_gfx1100"
AUTHORITY_START_POSITIONS = (0, 512, 1024, 2048, 3584)
AUTHORITY_WHOLE_LENGTHS = (512, 1024, 2048, 4096)
SMOKE_START_POSITIONS = (0,)
SMOKE_WHOLE_LENGTHS = (512,)
PREFILL_MODES = ("authority", "smoke")


@dataclass(frozen=True)
class PrefillRunProfile:
  mode: str
  K: int
  warmups: int
  rounds: int
  start_positions: tuple[int, ...]
  whole_lengths: tuple[int, ...]
  chunk_n: int = 512
  max_context: int = 4608

  def validate(self) -> None:
    if self.mode not in PREFILL_MODES: raise ValueError(f"prefill mode must be one of {PREFILL_MODES}, got {self.mode!r}")
    if self.K < 1 or self.warmups < 0 or self.rounds < 1:
      raise ValueError("K >= 1, warmups >= 0, and rounds >= 1 are required")
    if self.chunk_n <= 0 or self.max_context <= 0: raise ValueError("chunk_n and max_context must be positive")
    if not self.start_positions or not self.whole_lengths:
      raise ValueError("start_positions and whole_lengths must be non-empty")
    if any(sp < 0 for sp in self.start_positions): raise ValueError("start_positions must be non-negative")
    if any(length <= 0 for length in self.whole_lengths): raise ValueError("whole_lengths must be positive")
    max_start = max(self.start_positions)
    if max_start + self.chunk_n > self.max_context:
      raise ValueError(f"max start_pos {max_start} + chunk_n {self.chunk_n} exceeds max_context {self.max_context}")


@dataclass(frozen=True)
class SixRowResearchHarnessConfig:
  """Explicit subprocess authority for the default-off mixed-route smoke."""
  policy_path: str
  frozen_bundles: tuple[str, ...]
  fallback_program_identities: tuple[str, ...]
  inventory_path: str = str(ROOT / "bench/prefill-pure-full-kernel/qwen3-14b-mixed-quant-candidate-inventory-v1.json")

  def validate(self, profile: PrefillRunProfile) -> None:
    if not self.policy_path or not self.inventory_path or not self.frozen_bundles or not self.fallback_program_identities:
      raise ValueError("six-row research smoke requires policy, inventory, frozen bundle, and fallback program authorities")
    # The frozen AMDProgram is eager and is not captured as a TinyJit LINEAR.
    # Until replay integration exists, only one direct non-JIT smoke call is truthful.
    if profile.mode != "smoke" or profile.K != 1 or profile.warmups != 0 or profile.rounds != 1 or \
       profile.start_positions != (0,) or profile.whole_lengths != (512,):
      raise ValueError("six-row research is smoke-only and requires K=1, warmups=0, rounds=1, start=0, whole=512")


@dataclass(frozen=True)
class PrefillModelHarnessProfile:
  model_profile: ModelProfile
  default_model: str
  env_overrides: dict[str, str]
  note: str

  @property
  def id(self) -> str: return self.model_profile.id

  @property
  def env(self) -> dict[str, str]:
    return dict(self.env_overrides)


MODEL_HARNESS_PROFILE_ROWS = (
  PrefillModelHarnessProfile(profile_by_id("qwen3_8b_q4k_m_gfx1100"), DEFAULT_MODEL, {},
                             "8B automatic prefill authority path"),
  PrefillModelHarnessProfile(profile_by_id("qwen3_14b_q4k_m_gfx1100"), "/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf",
    {}, "14B automatic memory-safe authority path"),
)
if {row.id for row in MODEL_HARNESS_PROFILE_ROWS} != {profile.id for profile in MODEL_PROFILES}:
  raise ValueError("every model profile must have exactly one prefill harness record")
MODEL_HARNESS_PROFILES: dict[str, PrefillModelHarnessProfile] = {row.id:row for row in MODEL_HARNESS_PROFILE_ROWS}
MODEL_HARNESS_ALIASES = tuple(profile.size_label.lower() for profile in MODEL_PROFILES)


def resolve_prefill_model_profile(profile_id: str | None = None, *, model_path: str | None = None) -> PrefillModelHarnessProfile:
  if profile_id:
    try: return MODEL_HARNESS_PROFILES[profile_by_id(profile_id).id]
    except KeyError as exc:
      raise KeyError(f"unknown prefill model profile {profile_id!r}; known={sorted(MODEL_HARNESS_PROFILES)}") from exc
  profile = profile_from_model_path(model_path or DEFAULT_MODEL,
                                    default_profile_id=DEFAULT_MODEL_PROFILE if model_path is None else None)
  return MODEL_HARNESS_PROFILES[profile.id]


def csv_ints(raw: str) -> tuple[int, ...]:
  vals = tuple(int(x) for x in raw.replace(" ", "").split(",") if x)
  if not vals: raise ValueError("expected at least one comma-separated integer")
  return vals


def prefill_run_profile(mode: str = "authority", *, K: int | None = None, warmups: int | None = None,
                        rounds: int | None = None, start_positions: tuple[int, ...] | None = None,
                        whole_lengths: tuple[int, ...] | None = None, chunk_n: int = 512,
                        max_context: int = 4608) -> PrefillRunProfile:
  mode = mode.strip().lower()
  smoke = mode == "smoke"
  prof = PrefillRunProfile(
    mode=mode,
    K=K if K is not None else (1 if smoke else 8),
    warmups=warmups if warmups is not None else (1 if smoke else 4),
    rounds=rounds if rounds is not None else (1 if smoke else 3),
    start_positions=start_positions if start_positions is not None else (SMOKE_START_POSITIONS if smoke else AUTHORITY_START_POSITIONS),
    whole_lengths=whole_lengths if whole_lengths is not None else (SMOKE_WHOLE_LENGTHS if smoke else AUTHORITY_WHOLE_LENGTHS),
    chunk_n=chunk_n,
    max_context=max_context,
  )
  prof.validate()
  return prof


def prefill_authority_argv(model_path: str, profile: PrefillRunProfile, *, model_profile_id: str | None = None,
                           pin_clock: bool = False, artifact: bool = True, require_route: str | None = None,
                           six_row_research: SixRowResearchHarnessConfig | None = None,
                           artifact_path: str | None = None) -> list[str]:
  profile.validate()
  if not artifact and artifact_path: raise ValueError("prefill artifact path cannot be combined with artifact=False")
  if six_row_research is not None: six_row_research.validate(profile)
  model_profile = resolve_prefill_model_profile(model_profile_id, model_path=model_path)
  argv = ["extra/qk/prefill_whole_synced.py", "--model", model_path, "--mode", profile.mode,
          "--model-profile", model_profile.id,
          "-K", str(profile.K), "--warmups", str(profile.warmups), "--rounds", str(profile.rounds),
          "--start-positions", ",".join(str(x) for x in profile.start_positions),
          "--whole-lengths", ",".join(str(x) for x in profile.whole_lengths),
          "--max-context", str(profile.max_context)]
  if pin_clock: argv.append("--pin-clock")
  if not artifact: argv.append("--no-artifact")
  elif artifact_path: argv.extend(("--artifact", artifact_path))
  if require_route: argv.extend(("--require-route", require_route))
  if six_row_research is not None:
    argv.extend(("--six-row-research-policy", six_row_research.policy_path,
                 "--six-row-research-inventory", six_row_research.inventory_path))
    for declaration in six_row_research.frozen_bundles:
      argv.extend(("--six-row-frozen-bundle", declaration))
    for declaration in six_row_research.fallback_program_identities:
      argv.extend(("--six-row-fallback-program", declaration))
  return argv


def prefill_subprocess_env(extra: dict | None = None, *, model_profile_id: str | None = None,
                           model_path: str | None = None) -> dict[str, str]:
  model_profile = resolve_prefill_model_profile(model_profile_id, model_path=model_path)
  env = {"PYTHONPATH": str(ROOT), **model_profile.env}
  for k, v in (extra or {}).items(): env[str(k)] = str(v)
  return env
