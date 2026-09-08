#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, subprocess, time
from typing import Any

from extra.llm.bench.sft_smoke_train import load_sft_rows, split_rows


def completed_text(events_text:str) -> tuple[str|None, str|None]:
  completed, failure = None, None
  for line in events_text.splitlines():
    event = json.loads(line)
    if event.get("event") == "completed": completed = event.get("text", "")
    if event.get("event") == "failed": failure = str(event.get("code", "unknown"))
  return completed, failure


def evaluate(harness:pathlib.Path, endpoint:str, model:str, system_prompt:str, rows:list[dict[str, Any]], out:pathlib.Path,
             *, eval_every:int=3, timeout_s:float=120) -> dict[str, Any]:
  _, eval_rows, eval_source_ids = split_rows(rows, eval_every=eval_every)
  events_dir = out / "events"
  events_dir.mkdir(parents=True, exist_ok=True)
  results = []
  for row in eval_rows:
    source_id = row["source_id"]
    command = [str(harness), "--endpoint", endpoint, "--model", model, "--session-id", f"self-train-{model}-{source_id}",
               "--system-prompt", system_prompt]
    request = json.dumps({"schema":"gameterm.harness.turn.v1", "request_id":f"request-{source_id}", "text":row["prompt"]}) + "\n"
    started = time.monotonic()
    proc = subprocess.run(command, input=request, text=True, capture_output=True, timeout=timeout_s)
    elapsed_s = time.monotonic() - started
    (events_dir / f"{source_id}.jsonl").write_text(proc.stdout)
    (events_dir / f"{source_id}.stderr").write_text(proc.stderr)
    actual, failure = completed_text(proc.stdout)
    expected = row["completion"]
    results.append({"id":row["id"], "source_id":source_id, "prompt":row["prompt"], "expected":expected,
                    "actual":actual, "exact":actual is not None and actual.strip() == expected,
                    "failure":failure, "exit_code":proc.returncode, "elapsed_s":elapsed_s})
  exact = sum(result["exact"] for result in results)
  ledger = {"kind":"gameterm_qwen_self_training_eval", "model":model, "endpoint":endpoint,
            "system_prompt":system_prompt, "eval_every":eval_every, "eval_source_ids":eval_source_ids,
            "score":{"exact_correct":exact, "total":len(results), "exact_accuracy":exact / len(results)}, "results":results}
  out.mkdir(parents=True, exist_ok=True)
  (out / "run.jsonl").write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in results))
  (out / "score-ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
  return ledger


def main() -> int:
  parser = argparse.ArgumentParser(description="Run source-separated Qwen self-training evaluation through GameTerm")
  parser.add_argument("--harness", type=pathlib.Path, required=True)
  parser.add_argument("--endpoint", required=True)
  parser.add_argument("--model", required=True)
  parser.add_argument("--system-prompt", required=True)
  parser.add_argument("--input", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--eval-every", type=int, default=3)
  parser.add_argument("--timeout", type=float, default=120)
  args = parser.parse_args()
  ledger = evaluate(args.harness, args.endpoint, args.model, args.system_prompt, load_sft_rows(args.input), args.out,
                    eval_every=args.eval_every, timeout_s=args.timeout)
  print(json.dumps(ledger, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
