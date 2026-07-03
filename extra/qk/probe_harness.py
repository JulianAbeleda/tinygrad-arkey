"""Shared probe harness — one IO + verdict helper for one-shot bench probes (audit S1 de-clone).

Historically every `qk_amd_bb5a*` / `qk_decode_*tooling` probe cloned its own `read_json(rel, default)` +
`write_json(name, data)` + an inline `{phase, gate_pass, next_action, ...}` verdict dict. This module is the single
source for that pattern so a new probe adds a *call*, not a copy.

`probe_io(out_dir)` returns `(read_json, write_json)` byte-identical to the historical clones:
  - `write_json(name, data)` -> `out_dir/name`, `json.dumps(data, indent=2, sort_keys=True) + "\\n"` (the trailing
    newline is the probe-artifact convention -- note llm_eval_common.write_json omits it, so probe IO lives here, not
    there);
  - `read_json(rel, default=None)` -> `ROOT/rel`, returns `default` if missing else the parsed JSON.

No tinygrad import (probes set the AMD env before importing tinygrad; this stays import-light).
"""
from __future__ import annotations
import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]


def probe_io(out_dir: "str | pathlib.Path"):
  """Return `(read_json, write_json)` bound to a probe's output dir. Byte-identical to the historical clone helpers."""
  out_dir = pathlib.Path(out_dir)

  def write_json(name: str, data: Any) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

  def read_json(rel: str, default: Any = None) -> Any:
    p = ROOT / rel
    if not p.exists(): return default
    return json.loads(p.read_text())

  return read_json, write_json


def emit_verdict(phase: str, gate_pass: bool, next_action: str, **extra: Any) -> dict:
  """The historical inline verdict-row template, defined once. Bespoke per-probe keys go in `extra`."""
  return {"phase": phase, "gate_pass": bool(gate_pass), "next_action": next_action, **extra}
