"""Interim NV split-phase PDL render policy loader and placement (CPU half).

Backing scaffolding behind NV_SPLIT_PHASE=1 for
`tinygrad/renderer/cuda.py`: the renderer reads a versioned JSON policy file
(env `NV_SPLIT_PHASE_POLICY`) instead of the process-wide NV_PDL_* name
prefixes. This is explicitly not the production interface; the target shape is
per-program split policy carried in graph metadata (scope doc section 5.3).

Schema `tinygrad.nv_split_phase_policy.v1`:

.. code-block:: json

  {
    "schema": "tinygrad.nv_split_phase_policy.v1",
    "commit": "<optional>",
    "programs": [
      {"match": "<exact name or prefix:rule>", "wait_position": "entry"|"prologue",
       "trigger_policy": "end"|"start", "wait_anchor": "<optional literal substring>",
       "emit_wait": "<optional bool, default true>",
       "emit_trigger": "<optional bool, default true>",
       "profile": {
         "va": "<scratch buffer device VA>",
         "trigger_offset": "<8-aligned byte offset>",
         "grid_start_offset": "<8-aligned byte offset>",
         "wait_exit_offset": "<8-aligned byte offset>"
       }}
    ]
  }

Semantics, per first matching program rule:

- `wait_position=entry` prepends `griddepcontrol.wait` at instruction zero.
- `wait_position=prologue` inserts it immediately before the first kernel
  source line containing the literal `wait_anchor`; a prologue rule without an
  anchor, or an anchor that never appears in the rendered kernel, refuses to
  emit and raises rather than guessing.
- `trigger_policy=end` appends `griddepcontrol.launch_dependents`;
  `trigger_policy=start` prepends it. When both land at the top, wait is
  emitted first so the kernel satisfies its incoming dependency before
  announcing readiness to dependents.
- `emit_wait` / `emit_trigger` are optional booleans (default `true`).
  A probe policy can set `emit_wait=false` for a pure producer and
  `emit_trigger=false` for a pure consumer so the injected halves match the
  scheduler's per-edge roles instead of always emitting both instructions.
- Optional `profile` (Stage 3 timestamped capture, behind the same gate):
  when present, the policy loader additionally injects `%globaltimer`
  read/write blocks so the real-route capture can record the producer trigger
  time, the consumer grid-start time, and the consumer wait-exit time.
  `va` is the device VA of the probe scratch buffer; each `*_offset` is an
  8-byte-aligned byte offset into it where the u64 timestamp lands. Offsets
  must be pairwise non-overlapping (each slot is 8 bytes). `trigger_offset`
  writes immediately after `griddepcontrol.launch_dependents`;
  `grid_start_offset` writes at instruction 0 (before any wait);
  `wait_exit_offset` writes immediately after `griddepcontrol.wait`. The
  injected block is guarded to thread 0 of block 0 and never touches kernel
  numerics.

The emitted asm strings are byte-identical to the legacy NV_PDL_* path.
"""

import json, os
from functools import lru_cache

POLICY_SCHEMA = "tinygrad.nv_split_phase_policy.v1"
_WAIT_ASM = '  asm volatile("griddepcontrol.wait;");'
_TRIGGER_ASM = '  asm volatile("griddepcontrol.launch_dependents;");'
_PROFILE_FIELDS = ("trigger_offset", "grid_start_offset", "wait_exit_offset")
_PROFILE_GUARD_OPEN = "  if (threadIdx.x == 0 && blockIdx.x == 0) {"
_PROFILE_GUARD_CLOSE = "  }"
_PROFILE_GT_READ = "    unsigned long long _nv_gt = 0ULL;"
_PROFILE_GT_ASM = '    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_nv_gt));'


def _nv_pdl_match(name:str, rule:str) -> bool:
  return name == rule or (rule.startswith("prefix:") and name.startswith(rule.removeprefix("prefix:")))


def _profile_timer_lines(va:int) -> list[str]:
  return [_PROFILE_GUARD_OPEN, _PROFILE_GT_READ, _PROFILE_GT_ASM,
          f"    *(volatile unsigned long long*)(unsigned long long){va}ULL = _nv_gt;",
          _PROFILE_GUARD_CLOSE]


def validate_profile(profile:dict, path:str, prog_idx:int) -> None:
  """Validate a Stage 3 policy profile block; raises ValueError on any issue."""
  if not isinstance(profile, dict):
    raise ValueError(f"NV_SPLIT_PHASE_POLICY {path}: programs[{prog_idx}].profile must be an object")
  va = profile.get("va")
  if not isinstance(va, int) or va <= 0:
    raise ValueError(f"NV_SPLIT_PHASE_POLICY {path}: programs[{prog_idx}].profile.va must be a positive int")
  offsets = []
  for key in _PROFILE_FIELDS:
    if key not in profile: continue
    val = profile[key]
    if not isinstance(val, int) or val < 0 or val % 8 != 0:
      raise ValueError(f"NV_SPLIT_PHASE_POLICY {path}: programs[{prog_idx}].profile.{key} must be a non-negative 8-aligned int")
    offsets.append(val)
  for i, a in enumerate(offsets):
    for b in offsets[i+1:]:
      if abs(a - b) < 8:
        raise ValueError(f"NV_SPLIT_PHASE_POLICY {path}: programs[{prog_idx}].profile offsets overlap (each slot is 8 bytes)")


def _validate_policy(policy:dict, path:str) -> dict:
  if policy.get("schema") != POLICY_SCHEMA:
    raise ValueError(f"NV_SPLIT_PHASE_POLICY {path}: expected schema {POLICY_SCHEMA!r}, got {policy.get('schema')!r}")
  programs = policy.get("programs")
  if not isinstance(programs, list):
    raise ValueError(f"NV_SPLIT_PHASE_POLICY {path}: 'programs' must be a list")
  for i, prog in enumerate(programs):
    if not isinstance(prog, dict):
      raise ValueError(f"NV_SPLIT_PHASE_POLICY {path}: programs[{i}] must be an object")
    if not isinstance(prog.get("match"), str) or not prog["match"]:
      raise ValueError(f"NV_SPLIT_PHASE_POLICY {path}: programs[{i}] needs a non-empty 'match'")
    if prog.get("wait_position") not in ("entry", "prologue"):
      raise ValueError(f"NV_SPLIT_PHASE_POLICY {path}: programs[{i}] 'wait_position' must be 'entry' or 'prologue', got {prog.get('wait_position')!r}")
    if prog.get("trigger_policy") not in ("end", "start"):
      raise ValueError(f"NV_SPLIT_PHASE_POLICY {path}: programs[{i}] 'trigger_policy' must be 'end' or 'start', got {prog.get('trigger_policy')!r}")
    for key in ("emit_wait", "emit_trigger"):
      if key in prog and not isinstance(prog[key], bool):
        raise ValueError(f"NV_SPLIT_PHASE_POLICY {path}: programs[{i}] '{key}' must be a bool when present")
    if prog.get("wait_position") == "prologue" and (not isinstance(prog.get("wait_anchor"), str) or not prog["wait_anchor"]):
      raise ValueError(f"NV_SPLIT_PHASE_POLICY {path}: programs[{i}] 'prologue' requires a non-empty literal 'wait_anchor'")
    if "profile" in prog: validate_profile(prog["profile"], path, i)
  return policy


@lru_cache(maxsize=None)
def _load_policy_file(path:str, mtime:int) -> dict:
  with open(path, encoding="utf-8") as f: return _validate_policy(json.load(f), path)


def get_nv_split_phase_policy(path:str|None=None) -> dict|None:
  """Memoized loader for the NV_SPLIT_PHASE_POLICY JSON file (None when unset)."""
  path = path or os.environ.get("NV_SPLIT_PHASE_POLICY", "")
  if not path: return None
  try: mtime = os.stat(path).st_mtime_ns
  except OSError as e: raise FileNotFoundError(f"NV_SPLIT_PHASE_POLICY {path}: cannot stat policy file") from e
  return _load_policy_file(path, mtime)


def _apply_wait(kernel:list[str], prog:dict, name:str) -> list[str]:
  if prog.get("emit_wait") is False: return kernel
  if prog["wait_position"] == "entry": return [_WAIT_ASM, *kernel]
  anchor = prog.get("wait_anchor")
  for i, line in enumerate(kernel):
    if anchor in line: return [*kernel[:i], _WAIT_ASM, *kernel[i:]]
  raise RuntimeError(f"NV_SPLIT_PHASE policy: program {name!r} wait_position=prologue, anchor {anchor!r} not found in rendered kernel; refusing to guess")


def _apply_trigger(kernel:list[str], trigger_policy:str) -> list[str]:
  if trigger_policy is False: return kernel
  return ([_TRIGGER_ASM, *kernel] if trigger_policy == "start" else [*kernel, _TRIGGER_ASM])


def _insert_after(kernel:list[str], anchor:str, lines:list[str], name:str, what:str) -> list[str]:
  for i, line in enumerate(kernel):
    if line == anchor:
      return [*kernel[:i+1], *lines, *kernel[i+1:]]
  raise RuntimeError(f"NV_SPLIT_PHASE policy: program {name!r} profile {what} anchor {anchor!r} not found in rendered kernel; refusing to guess")


def _apply_profile(kernel:list[str], prog:dict, name:str) -> list[str]:
  profile = prog["profile"]
  va = int(profile["va"])
  if "grid_start_offset" in profile:
    kernel = [*_profile_timer_lines(va + profile["grid_start_offset"]), *kernel]
  if "wait_exit_offset" in profile:
    kernel = _insert_after(kernel, _WAIT_ASM, _profile_timer_lines(va + profile["wait_exit_offset"]), name, "wait_exit")
  if "trigger_offset" in profile:
    kernel = _insert_after(kernel, _TRIGGER_ASM, _profile_timer_lines(va + profile["trigger_offset"]), name, "trigger")
  return kernel


def apply_nv_split_phase_policy(name:str, kernel:list[str], policy:dict|None=None) -> list[str]:
  """Emit PDL asm for `name` per the first matching rule; unmatched stays untouched."""
  if policy is None: policy = get_nv_split_phase_policy()
  if policy is None: return kernel
  for prog in policy["programs"]:
    if not _nv_pdl_match(name, prog["match"]): continue
    trigger_policy = prog["trigger_policy"] if prog.get("emit_trigger", True) else False
    kernel = _apply_wait(_apply_trigger(list(kernel), trigger_policy), prog, name)
    if "profile" in prog: kernel = _apply_profile(kernel, prog, name)
    return kernel
  return kernel
