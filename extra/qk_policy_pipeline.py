#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, re, statistics, subprocess, sys, time
from datetime import datetime

from extra.qk_ansor import build_policy_entries, make_policy_cache
from extra.qk_decode_summary import _md as decode_summary_md, parse_log as parse_decode_log
from extra.qk_layout import read_metadata

LLAMA_REFS = {"8B": 101.2, "14B": 65.8, "32B": 30.8}
MODEL_RE = re.compile(r"Qwen3-(?P<size>[0-9.]+B)-(?P<quant>[^/]+)\.gguf$", re.IGNORECASE)

def _model_label(model:pathlib.Path) -> str:
  m = MODEL_RE.search(model.name)
  if m is not None: return f"{m.group('size').lower()}-{m.group('quant').lower()}"
  return model.stem.lower().replace("_", "-")

def _model_size(model:pathlib.Path) -> str:
  m = MODEL_RE.search(model.name)
  return m.group("size").upper() if m is not None else "unknown"

def _git_commit(repo:pathlib.Path) -> str:
  return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True).strip()

def _run(cmd:list[str], repo:pathlib.Path, log:pathlib.Path, env:dict[str, str]|None=None, timeout:float|None=None) -> dict:
  log.parent.mkdir(parents=True, exist_ok=True)
  merged_env = {**os.environ, **(env or {})}
  st = time.perf_counter()
  proc = subprocess.run(cmd, cwd=repo, env=merged_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
  elapsed = time.perf_counter() - st
  log.write_text(proc.stdout)
  if proc.returncode != 0:
    raise RuntimeError(f"{log}: command failed rc={proc.returncode}\n{proc.stdout[-4000:]}")
  return {"cmd": cmd, "log": str(log), "elapsed_s": round(elapsed, 3), "returncode": proc.returncode}

def _mean(xs:list[float]) -> float:
  return statistics.mean(xs) if xs else 0.0

def _stdev(xs:list[float]) -> float:
  return statistics.stdev(xs) if len(xs) >= 2 else 0.0

def _runs_stable(rows:list[dict]) -> tuple[bool, list[str]]:
  reasons = []
  vals = [r["avg_tok_s"] for r in rows]
  mean = _mean(vals)
  for r in rows:
    if r["avg_tok_s"] < 0.90 * mean:
      reasons.append(f"{r['label']} avg {r['avg_tok_s']:.2f} is >10% below mean {mean:.2f}")
    if r.get("avg_last64_tok_s") is not None and r["avg_last64_tok_s"] < 0.90 * r["avg_tok_s"]:
      reasons.append(f"{r['label']} last64 {r['avg_last64_tok_s']:.2f} collapsed below 90% of avg {r['avg_tok_s']:.2f}")
    if r.get("avg_last16_tok_s") is not None and r["avg_last16_tok_s"] < 0.85 * r["avg_tok_s"]:
      reasons.append(f"{r['label']} last16 {r['avg_last16_tok_s']:.2f} collapsed below 85% of avg {r['avg_tok_s']:.2f}")
  return not reasons, reasons

def _decode_stats(rows:list[dict], prefix:str, window:int|None=None) -> dict:
  selected = [r for r in rows if r["label"].startswith(prefix)]
  decision_rows = selected[-window:] if window is not None and len(selected) > window else selected
  if not decision_rows: decision_rows = selected
  vals = [r["avg_tok_s"] for r in decision_rows]
  return {
    "runs": len(decision_rows),
    "total_runs": len(selected),
    "decision_labels": [r["label"] for r in decision_rows],
    "avg_tok_s": _mean(vals),
    "stdev_tok_s": _stdev(vals),
    "min_tok_s": min(vals) if vals else 0.0,
    "max_tok_s": max(vals) if vals else 0.0,
    "stable": _runs_stable(decision_rows)[0],
    "stability_reasons": _runs_stable(decision_rows)[1],
    "rows": decision_rows,
  }

def _policy_summary(path:pathlib.Path) -> dict:
  data = json.loads(path.read_text())
  if "summary" not in data: raise ValueError(f"{path}: missing policy parity summary")
  return data["summary"]

def _policy_storage(path:pathlib.Path) -> dict:
  return json.loads(path.read_text()).get("storage_policy", {})

def _ab_match(path:pathlib.Path) -> bool:
  return bool(json.loads(path.read_text())["match"])

def _run_decode(args, mode:str, policy:pathlib.Path|None, idx:int) -> pathlib.Path:
  log = args.out / f"{mode}-run{idx}.log"
  env = {"DEV": args.device, "JIT": "1", "PYTHONPATH": "."}
  if mode == "explicit":
    if args.reference_mode == "explicit":
      env |= {"Q4K_PRIMITIVE": "1", "Q6K_PRIMITIVE": "1", "Q4K_PRIMITIVE_DEBUG": "1", "Q6K_PRIMITIVE_DEBUG": "1"}
  elif mode == "generated":
    if policy is None: raise ValueError("generated decode requires policy")
    env |= {"QK_GENERATED_POLICY": str(policy), "QK_GENERATED_POLICY_DEBUG": "1"}
  else:
    raise ValueError(mode)
  _run([sys.executable, "-m", "tinygrad.llm", "--model", str(args.model), "--warmup", "--benchmark", str(args.benchmark)],
       args.repo, log, env=env, timeout=args.decode_timeout)
  return log

def _run_decode_repeats(args, mode:str, policy:pathlib.Path|None, start_idx:int=1, count:int|None=None) -> list[pathlib.Path]:
  logs = []
  for idx in range(start_idx, start_idx + (count or args.repeats)):
    logs.append(_run_decode(args, mode, policy, idx))
  return logs

def _existing_decode_logs(out:pathlib.Path, mode:str) -> list[pathlib.Path]:
  logs = []
  idx = 1
  while True:
    path = out / f"{mode}-run{idx}.log"
    if not path.exists(): break
    logs.append(path)
    idx += 1
  return logs

def _write_decode_summary(out:pathlib.Path, logs:list[tuple[str, pathlib.Path]]) -> list[dict]:
  rows = [parse_decode_log(label, path) for label, path in logs]
  (out / "decode-summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
  (out / "decode-summary.md").write_text(decode_summary_md(rows))
  return rows

def _apply_policy_cap_from_search(args, search_json:pathlib.Path, policy:pathlib.Path) -> bool:
  if args.policy_max_storage_mb is None: return False
  report = json.loads(search_json.read_text())
  cap_bytes = int(args.policy_max_storage_mb * 1024 * 1024)
  entries, storage_policy = build_policy_entries(args.model, args.repo, read_metadata(args.model), report["descriptors"], cap_bytes)
  capped = make_policy_cache(args.model, args.repo, entries, storage_policy)
  if policy.exists():
    old = json.loads(policy.read_text())
    old_stable = {k: v for k, v in old.items() if k != "created_at"}
    capped_stable = {k: v for k, v in capped.items() if k != "created_at"}
    if old_stable == capped_stable: return False
  text = json.dumps(capped, indent=2, sort_keys=True)
  policy.write_text(text)
  return True

def _profile_specs(args, policy:pathlib.Path) -> list[tuple[str, str, dict[str, str]]]:
  base_env = {"DEV": args.device, "DEBUG": "2", "JIT": "1", "PYTHONPATH": "."}
  if args.reference_mode == "explicit":
    reference_name = "q4q6-primitive"
    reference_env = base_env | {"Q4K_PRIMITIVE": "1", "Q6K_PRIMITIVE": "1"}
  elif args.reference_mode == "generic":
    reference_name = "baseline"
    reference_env = dict(base_env)
  else:
    raise ValueError(args.reference_mode)
  generated_env = base_env | {"QK_GENERATED_POLICY": str(policy)}
  model = args.model_size.lower()
  return [
    ("reference-batched", f"{model}-{reference_name}-batched-debug2.log", reference_env),
    ("generated-batched", f"{model}-generated-batched-debug2.log", generated_env),
    ("reference-named", f"{model}-{reference_name}-named-debug2.log", reference_env | {"JIT_BATCH_SIZE": "1"}),
    ("generated-named", f"{model}-generated-named-debug2.log", generated_env | {"JIT_BATCH_SIZE": "1"}),
  ]

def _run_profile(args, policy:pathlib.Path) -> dict:
  specs = _profile_specs(args, policy)
  logs = []
  for _, filename, env in specs:
    log = args.out / filename
    _run([sys.executable, "-m", "tinygrad.llm", "--model", str(args.model), "--warmup", "--benchmark", str(args.profile_tokens)],
         args.repo, log, env=env, timeout=args.profile_timeout)
    logs.append(log)
  report_json, report_md = args.out / "profile-report.json", args.out / "profile-report.md"
  _run([sys.executable, "extra/q4_k_profile_report.py", *[str(x) for x in logs], "--json", str(report_json), "--out", str(report_md),
        "--steady-drop", "1"], args.repo, args.out / "profile-report.log", env={"PYTHONPATH": "."}, timeout=args.profile_timeout)
  return {"logs": [str(x) for x in logs], "json": str(report_json), "md": str(report_md)}

def _decide(args, decode_rows:list[dict], parity_summary:dict, ab_match:bool, profile:dict|None) -> dict:
  explicit = _decode_stats(decode_rows, "explicit", args.repeats)
  generated = _decode_stats(decode_rows, "generated", args.repeats)
  gain = generated["avg_tok_s"] / explicit["avg_tok_s"] - 1.0 if explicit["avg_tok_s"] else 0.0
  reasons = []
  status = "reject"
  if parity_summary.get("generated_unsupported", 0) != 0:
    status = "invalid"
    reasons.append(f"generated_unsupported={parity_summary.get('generated_unsupported')}")
  elif not ab_match:
    status = "invalid"
    reasons.append("greedy output A/B mismatch")
  elif not explicit["stable"]:
    status = "needs-rerun"
    reasons += [f"explicit unstable: {x}" for x in explicit["stability_reasons"]]
  elif not generated["stable"]:
    status = "needs-rerun"
    reasons += [f"generated unstable: {x}" for x in generated["stability_reasons"]]
  elif gain >= args.accept_gain:
    if gain >= args.profile_gain and profile is None:
      status = "needs-profile"
      reasons.append(f"gain {gain:.2%} requires profile")
    else:
      status = "accept"
      reasons.append(f"generated beats explicit by {gain:.2%}")
  elif gain >= -args.tie_band:
    status = "tie"
    reasons.append(f"generated within tie band: {gain:.2%}")
  else:
    status = "reject"
    reasons.append(f"generated slower than explicit by {-gain:.2%}")
  return {
    "status": status,
    "reasons": reasons,
    "gain": gain,
    "explicit": {k: v for k, v in explicit.items() if k != "rows"},
    "generated": {k: v for k, v in generated.items() if k != "rows"},
    "parity_summary": parity_summary,
    "reference_mode": args.reference_mode,
    "ab_match": ab_match,
    "profile": profile,
  }

def _top_up_until_stable(args, policy:pathlib.Path, explicit_logs:list[pathlib.Path], generated_logs:list[pathlib.Path]) -> tuple[list[pathlib.Path], list[pathlib.Path], list[dict]]:
  max_runs = args.repeats + args.max_extra_repeats
  while True:
    decode_rows = _write_decode_summary(args.out, [(f"explicit{i+1}", p) for i, p in enumerate(explicit_logs)] +
                                        [(f"generated{i+1}", p) for i, p in enumerate(generated_logs)])
    explicit = _decode_stats(decode_rows, "explicit", args.repeats)
    generated = _decode_stats(decode_rows, "generated", args.repeats)
    need_explicit = not explicit["stable"] and len(explicit_logs) < max_runs
    need_generated = not generated["stable"] and len(generated_logs) < max_runs
    if not need_explicit and not need_generated: return explicit_logs, generated_logs, decode_rows
    if need_explicit:
      explicit_logs.append(_run_decode(args, "explicit", None, len(explicit_logs) + 1))
    if need_generated:
      generated_logs.append(_run_decode(args, "generated", policy, len(generated_logs) + 1))

def _write_readme(args, decision:dict) -> None:
  llama_ref = LLAMA_REFS.get(args.model_size)
  pct = "" if llama_ref is None else f"{decision['generated']['avg_tok_s'] / llama_ref * 100:.1f}%"
  lines = [
    f"# QK Policy Pipeline: {args.model.name}",
    "",
    f"Date: {datetime.now().date().isoformat()}",
    "",
    f"- commit: `{_git_commit(args.repo)}`",
    f"- device: `{args.device}`",
    f"- model size: `{args.model_size}`",
    f"- reference mode: `{args.reference_mode}`",
    f"- generated policy: `policy.json`",
    "",
    "## Decision",
    "",
    f"- status: `{decision['status']}`",
    f"- gain: `{decision['gain']*100:.2f}%`",
    f"- explicit mean: `{decision['explicit']['avg_tok_s']:.2f} tok/s`",
    f"- explicit decision window: `{', '.join(decision['explicit'].get('decision_labels', []))}`",
    f"- generated mean: `{decision['generated']['avg_tok_s']:.2f} tok/s`",
    f"- generated decision window: `{', '.join(decision['generated'].get('decision_labels', []))}`",
    f"- generated percent of llama.cpp reference: `{pct}`" if pct else "- generated percent of llama.cpp reference: `n/a`",
    "",
    "Reasons:",
    "",
    *[f"- {x}" for x in decision["reasons"]],
    "",
    "## Reproduction",
    "",
    "```sh",
    f"DEV={args.device} JIT=1 QK_GENERATED_POLICY={args.out / 'policy.json'} PYTHONPATH=. \\",
    f"  .venv/bin/python -m tinygrad.llm --model {args.model} --warmup --benchmark {args.benchmark}",
    "```",
    "",
    "## Artifacts",
    "",
    "- `search.json`, `policy.json`, `semantic-report.md`",
    "- `policy-parity.json`, `policy-parity.md`",
    "- `decode-summary.json`, `decode-summary.md`",
    "- `output-ab.json`, `output-ab.log`",
  ]
  if decision.get("profile"):
    lines += ["- `profile-report.json`, `profile-report.md`"]
  lines += ["", "## Decode Summary", "", (args.out / "decode-summary.md").read_text()]
  lines += ["", "## Policy Parity Summary", "", "```json",
            json.dumps(decision["parity_summary"], indent=2, sort_keys=True), "```", ""]
  if (storage:=decision.get("storage_policy")):
    lines += ["", "## Storage Policy", "", "```json", json.dumps(storage, indent=2, sort_keys=True), "```", ""]
  args.out.joinpath("README.md").write_text("\n".join(lines))

def _write_blocked_readme(args, decision:dict) -> None:
  lines = [
    f"# QK Policy Pipeline: {args.model.name}",
    "",
    f"Date: {datetime.now().date().isoformat()}",
    "",
    f"- commit: `{_git_commit(args.repo)}`",
    f"- device: `{args.device}`",
    f"- model size: `{args.model_size}`",
    f"- reference mode: `{args.reference_mode}`",
    f"- generated policy: `policy.json`",
    "",
    "## Decision",
    "",
    f"- status: `{decision['status']}`",
    "",
    "Reasons:",
    "",
    *[f"- {x}" for x in decision["reasons"]],
    "",
    "## Artifacts",
    "",
    "- `search.json`, `policy.json`, `semantic-report.md`",
    "- `policy-parity.json`, `policy-parity.md`",
  ]
  if (args.out / "explicit-run1.log").exists():
    lines += ["- `explicit-run1.log`"]
  lines += ["", "## Policy Parity Summary", "", "```json",
            json.dumps(decision["parity_summary"], indent=2, sort_keys=True), "```", ""]
  if (storage:=decision.get("storage_policy")):
    lines += ["", "## Storage Policy", "", "```json", json.dumps(storage, indent=2, sort_keys=True), "```", ""]
  if (args.out / "explicit-run1.log").exists():
    lines += ["", "## Failure Tail", "", "```", "\n".join((args.out / "explicit-run1.log").read_text(errors="replace").splitlines()[-32:]), "```"]
  args.out.joinpath("README.md").write_text("\n".join(lines))

def _memory_blocked(error:Exception) -> bool:
  msg = str(error)
  return "MemoryError" in msg or "no memory is available" in msg or "Allocation of" in msg

def run_pipeline(args) -> dict:
  args.repo = args.repo.resolve()
  args.model = args.model.expanduser().resolve()
  args.model_size = _model_size(args.model)
  if args.out is None:
    args.out = args.repo / "bench" / f"qk-policy-pipeline-{datetime.now().date().strftime('%Y%m%d')}" / _model_label(args.model)
  args.out.mkdir(parents=True, exist_ok=True)

  policy = args.out / "policy.json"
  search_json = args.out / "search.json"
  policy_changed = False
  if args.reuse and search_json.exists() and args.policy_max_storage_mb is not None:
    policy_changed = _apply_policy_cap_from_search(args, search_json, policy)
  elif not (args.reuse and policy.exists() and search_json.exists()):
    cmd = [sys.executable, "extra/qk_ansor.py", "--model", str(args.model), "--device", args.device, "--level", str(args.level),
           "--iters", str(args.iters), "--skip-stopped", "--json", str(search_json), "--policy-json", str(policy),
           "--timeout", str(args.candidate_timeout)]
    if args.policy_max_storage_mb is not None: cmd += ["--policy-max-storage-mb", str(args.policy_max_storage_mb)]
    _run(cmd, args.repo, args.out / "search.log",
         env={"DEV": args.device, "PYTHONPATH": ".", "Q4K_ALLOW_RISKY_SEARCH": "1"}, timeout=args.search_timeout)
  if not (args.reuse and (args.out / "semantic-report.md").exists()):
    _run([sys.executable, "extra/qk_semantic_report.py", str(search_json), "--md", str(args.out / "semantic-report.md"),
          "--title", f"QK Policy Pipeline Search: {args.model.name}"], args.repo, args.out / "semantic-report.log",
         env={"PYTHONPATH": "."}, timeout=120)
  if not (args.reuse and not policy_changed and (args.out / "policy-parity.json").exists()):
    _run([sys.executable, "extra/qk_policy_parity.py", "--model", str(args.model), "--policy", str(policy),
          "--json", str(args.out / "policy-parity.json"), "--md", str(args.out / "policy-parity.md")],
         args.repo, args.out / "policy-parity.log", env={"PYTHONPATH": "."}, timeout=600)

  try:
    explicit_logs = _existing_decode_logs(args.out, "explicit") if args.reuse and not policy_changed else []
    generated_logs = _existing_decode_logs(args.out, "generated") if args.reuse and not policy_changed else []
    if len(explicit_logs) < args.repeats:
      explicit_logs += _run_decode_repeats(args, "explicit", None, len(explicit_logs) + 1, args.repeats - len(explicit_logs))
    if len(generated_logs) < args.repeats:
      generated_logs += _run_decode_repeats(args, "generated", policy, len(generated_logs) + 1, args.repeats - len(generated_logs))
    explicit_logs, generated_logs, decode_rows = _top_up_until_stable(args, policy, explicit_logs, generated_logs)
  except RuntimeError as e:
    if not _memory_blocked(e): raise
    parity_summary = _policy_summary(args.out / "policy-parity.json")
    decision = {
      "status": "blocked",
      "reasons": [f"decode blocked by GPU memory during primitive install: {str(e).splitlines()[-1]}"],
      "parity_summary": parity_summary,
      "storage_policy": _policy_storage(policy),
      "reference_mode": args.reference_mode,
      "ab_match": None,
      "profile": None,
      "model": str(args.model), "model_size": args.model_size, "out": str(args.out), "policy": str(policy),
      "commit": _git_commit(args.repo), "created_at": datetime.now().isoformat(),
    }
    (args.out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
    _write_blocked_readme(args, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision

  if not (args.reuse and not policy_changed and (args.out / "output-ab.json").exists()):
    _run([sys.executable, "extra/q4_k_output_ab.py", "--model", str(args.model), "--tokens", str(args.ab_tokens),
          "--timeout", str(args.ab_timeout), "--candidate-policy", str(policy), "--policy-debug",
          "--json", str(args.out / "output-ab.json")], args.repo, args.out / "output-ab.log",
         env={"DEV": args.device, "JIT": "1", "PYTHONPATH": "."}, timeout=args.ab_timeout + 60)

  parity_summary = _policy_summary(args.out / "policy-parity.json")
  storage_policy = _policy_storage(policy)
  ab_match = _ab_match(args.out / "output-ab.json")
  preliminary = _decide(args, decode_rows, parity_summary, ab_match, None)
  profile = None
  if args.profile == "always" or (args.profile == "auto" and preliminary["gain"] >= args.profile_gain and preliminary["status"] != "invalid"):
    if args.reuse and not policy_changed and (args.out / "profile-report.json").exists():
      profile = {"json": str(args.out / "profile-report.json"), "md": str(args.out / "profile-report.md"), "reused": True}
    else:
      profile = _run_profile(args, policy)
  decision = _decide(args, decode_rows, parity_summary, ab_match, profile)
  decision |= {"model": str(args.model), "model_size": args.model_size, "out": str(args.out), "policy": str(policy),
               "storage_policy": storage_policy, "commit": _git_commit(args.repo), "created_at": datetime.now().isoformat()}
  (args.out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
  _write_readme(args, decision)
  print(json.dumps(decision, indent=2, sort_keys=True))
  return decision

def main() -> None:
  parser = argparse.ArgumentParser(description="Generate, validate, benchmark, and accept/reject a QK generated policy")
  parser.add_argument("--model", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path)
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--level", type=int, default=2)
  parser.add_argument("--iters", type=int, default=2)
  parser.add_argument("--benchmark", type=int, default=128)
  parser.add_argument("--reference-mode", choices=("explicit", "generic"), default="explicit",
                      help="explicit compares against Q4K/Q6K primitive flags; generic compares against the fused graph baseline")
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--max-extra-repeats", type=int, default=2,
                      help="run up to this many additional samples per mode until the latest --repeats window is stable")
  parser.add_argument("--ab-tokens", type=int, default=32)
  parser.add_argument("--profile", choices=("auto", "always", "never"), default="auto")
  parser.add_argument("--profile-tokens", type=int, default=8)
  parser.add_argument("--accept-gain", type=float, default=0.03)
  parser.add_argument("--tie-band", type=float, default=0.03)
  parser.add_argument("--profile-gain", type=float, default=0.20)
  parser.add_argument("--candidate-timeout", type=float, default=120)
  parser.add_argument("--policy-max-storage-mb", type=float,
                      help="cap generated primitive policy to this much persistent packed-weight storage")
  parser.add_argument("--search-timeout", type=float)
  parser.add_argument("--decode-timeout", type=float, default=1800)
  parser.add_argument("--profile-timeout", type=float, default=1800)
  parser.add_argument("--ab-timeout", type=float, default=1800)
  parser.add_argument("--reuse", action="store_true", help="reuse existing stage artifacts in --out and regenerate decision/README")
  args = parser.parse_args()
  run_pipeline(args)

if __name__ == "__main__":
  main()
