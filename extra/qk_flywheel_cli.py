#!/usr/bin/env python3
"""Unified CLI dispatcher for the AMD decode flywheel judging tooling.

A single entry point over the consolidated flywheel modules. Each subcommand
delegates to the owning module's existing ``main()`` so argument parsing and
behavior are identical to invoking ``python -m extra.qk_flywheel_<x>`` directly
(those per-module entry points stay valid; this only adds one boring surface
over them, per the anti-re-sprawl rule).

Usage:
  python -m extra.qk_flywheel_cli <command> [args...]
  python -m extra.qk_flywheel_cli --list

A new experiment is a new row in a source table, not a new ``main()`` or a new
script. New subcommands here should be rare and only when a genuinely new
module is added.
"""
from __future__ import annotations

import importlib, sys

# command name -> module that owns its main(). The module's own argparse surface
# is the single source of truth for each command's arguments.
COMMANDS: dict[str, str] = {
  "dataset": "extra.qk_flywheel_dataset",
  "dataset-v1": "extra.qk_flywheel_dataset_v1",
  "feature-enrich": "extra.qk_flywheel_feature_enrich",
  "targeted-outcomes": "extra.qk_flywheel_targeted_outcomes",
  "triage-eval": "extra.qk_flywheel_triage_eval",
  "cost-model": "extra.qk_flywheel_cost_model",
  "feature-audit": "extra.qk_flywheel_feature_audit",
  "coverage-plan": "extra.qk_flywheel_coverage_plan",
  "triage-sft": "extra.qk_flywheel_triage_sft",
  "shadow": "extra.qk_flywheel_shadow",
}


def _usage() -> str:
  lines = ["usage: python -m extra.qk_flywheel_cli <command> [args...]", "", "commands:"]
  lines += [f"  {name}" for name in COMMANDS]
  lines += ["", "run `<command> --help` for that command's arguments"]
  return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
  args = list(sys.argv[1:] if argv is None else argv)
  if not args or args[0] in ("-h", "--help", "--list"):
    print(_usage())
    return 0 if args and args[0] in ("-h", "--help", "--list") else 2
  command, rest = args[0], args[1:]
  if command not in COMMANDS:
    print(f"unknown command {command!r}\n", file=sys.stderr)
    print(_usage(), file=sys.stderr)
    return 2
  module = importlib.import_module(COMMANDS[command])
  # Delegate to the owning module's main(): rewrite argv so its argparse sees the
  # command's own args, and restore prog to the unified surface for clean errors.
  saved = sys.argv
  sys.argv = [f"qk_flywheel_cli {command}"] + rest
  try:
    return int(module.main() or 0)
  finally:
    sys.argv = saved


if __name__ == "__main__":
  raise SystemExit(main())
