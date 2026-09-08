#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, signal, subprocess, sys, time, urllib.request
from typing import Any

from extra.llm.bench.gameterm_self_training_eval import evaluate
from extra.llm.bench.sft_smoke_train import load_sft_rows, split_rows


def write_json_atomic(path:pathlib.Path, value:Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
  temporary.replace(path)


def load_ledger(path:pathlib.Path, config:dict[str, Any]) -> dict[str, Any]:
  if not path.exists(): return {"kind":"gameterm_qwen_self_training_loop", "status":"running", "config":config, "rounds":[]}
  ledger = json.loads(path.read_text())
  if ledger.get("config") != config: raise ValueError("existing controller config differs; choose a new output directory")
  ledger["status"] = "running"
  return ledger


def wait_ready(endpoint:str, process:subprocess.Popen, timeout_s:float=90) -> None:
  status_url = endpoint.rsplit("/v1/chat/completions", 1)[0] + "/runtime/status"
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    if process.poll() is not None: raise RuntimeError(f"server exited before readiness with code {process.returncode}")
    try:
      with urllib.request.urlopen(status_url, timeout=2) as response:
        if json.loads(response.read()).get("loaded"): return
    except Exception: pass
    time.sleep(1)
  raise TimeoutError(f"server did not become ready within {timeout_s}s")


def warm_server(endpoint:str, model:str, system_prompt:str, prompt:str, max_tokens:int, out:pathlib.Path,
                timeout_s:float=300) -> dict[str, Any]:
  body = {"model":model, "messages":[{"role":"system", "content":system_prompt}, {"role":"user", "content":prompt}],
          "stream":False, "max_tokens":max_tokens, "temperature":0}
  request = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
  started = time.monotonic()
  with urllib.request.urlopen(request, timeout=timeout_s) as response: result = json.loads(response.read())
  record = {"elapsed_s":time.monotonic()-started, "request":body, "response":result}
  write_json_atomic(out, record)
  return record


def stop_process(process:subprocess.Popen) -> None:
  if process.poll() is not None: return
  try: os.killpg(process.pid, signal.SIGINT)
  except ProcessLookupError: return
  try: process.wait(timeout=15)
  except subprocess.TimeoutExpired:
    try: os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    process.wait(timeout=5)


def run(args:argparse.Namespace) -> dict[str, Any]:
  out, started = args.out.resolve(), time.time()
  out.mkdir(parents=True, exist_ok=True)
  config = {"model":str(args.model.resolve()), "input":str(args.input.resolve()), "harness":str(args.harness.resolve()),
            "system_prompt":args.system_prompt, "device":args.device, "max_context":args.max_context,
            "steps_per_round":args.steps_per_round, "lr":args.lr, "lr_decay":args.lr_decay,
            "rank":args.rank, "alpha":args.alpha, "eval_every":args.eval_every,
            "port":args.port, "server_max_tokens":args.server_max_tokens, "prefill_chunk_size":args.prefill_chunk_size,
            "init_adapter":str(args.init_adapter.resolve()) if args.init_adapter is not None else None}
  ledger_path = out / "controller-ledger.json"
  ledger = load_ledger(ledger_path, config)
  ledger["pid"] = os.getpid()
  ledger["last_started_at_unix"] = started
  write_json_atomic(ledger_path, ledger)
  rows = load_sft_rows(args.input)
  _, eval_rows, _ = split_rows(rows, eval_every=args.eval_every)
  endpoint = f"http://127.0.0.1:{args.port}/v1/chat/completions"
  deadline = time.monotonic() + args.max_hours * 3600
  previous_adapter = pathlib.Path(ledger["rounds"][-1]["adapter"]) if ledger["rounds"] else args.init_adapter
  while len(ledger["rounds"]) < args.max_rounds and time.monotonic() < deadline:
    index = len(ledger["rounds"])
    round_dir = out / f"round-{index:03d}"
    train_dir, eval_dir = round_dir / "training", round_dir / "gameterm-eval"
    round_dir.mkdir(parents=True, exist_ok=True)
    learning_rate = args.lr * (args.lr_decay ** index)
    train_command = [sys.executable, "-m", "extra.llm.bench.qwen_lora_smoke_train", "--model", str(args.model),
      "--input", str(args.input), "--out", str(train_dir), "--device", args.device, "--max-context", str(args.max_context),
      "--seed", str(args.seed + index), "--eval-every", str(args.eval_every), "--steps", str(args.steps_per_round),
      "--lr", str(learning_rate), "--rank", str(args.rank), "--alpha", str(args.alpha), "--system-prompt", args.system_prompt,
      "--completion-scope", "all"]
    if previous_adapter is not None: train_command += ["--init-adapter", str(previous_adapter)]
    write_json_atomic(round_dir / "train-command.json", train_command)
    train_started = time.monotonic()
    with (round_dir / "training.stdout").open("w") as stdout, (round_dir / "training.stderr").open("w") as stderr:
      train_result = subprocess.run(train_command, cwd=args.repo, env={**os.environ, "DEV":args.device}, stdout=stdout, stderr=stderr)
    summary_path, adapter = train_dir / "summary.json", train_dir / "adapter"
    if not summary_path.is_file() or not (adapter / "adapter.npz").is_file():
      raise RuntimeError(f"round {index}: trainer produced no complete checkpoint (exit {train_result.returncode})")
    summary = json.loads(summary_path.read_text())
    if not summary.get("checks", {}).get("finite", False): raise RuntimeError(f"round {index}: non-finite training result")

    server_command = [sys.executable, "-m", "tinygrad.llm", "--model", str(args.model), "--adapter", str(adapter),
      "--max_context", str(args.max_context), "--serve", str(args.port), "--no-warmup",
      "--default-max-tokens", str(args.server_max_tokens), "--prefill-chunk-size", str(args.prefill_chunk_size)]
    write_json_atomic(round_dir / "server-command.json", server_command)
    server = None
    try:
      server_stdout, server_stderr = (round_dir / "server.stdout").open("w"), (round_dir / "server.stderr").open("w")
      server = subprocess.Popen(server_command, cwd=args.repo, env={**os.environ, "DEV":args.device},
                                stdout=server_stdout, stderr=server_stderr, start_new_session=True)
      wait_ready(endpoint, server, args.server_ready_timeout)
      warmup = warm_server(endpoint, f"qwen-self-train-round-{index}", args.system_prompt, eval_rows[0]["prompt"],
                           args.server_max_tokens, round_dir / "warmup.json", args.warmup_timeout)
      score = evaluate(args.harness, endpoint, f"qwen-self-train-round-{index}", args.system_prompt, rows, eval_dir,
                       eval_every=args.eval_every, timeout_s=args.harness_timeout)
    finally:
      if server is not None: stop_process(server)
      if 'server_stdout' in locals(): server_stdout.close()
      if 'server_stderr' in locals(): server_stderr.close()
    round_record = {"round":index, "adapter":str(adapter.resolve()), "trainer_exit_code":train_result.returncode,
                    "learning_rate":learning_rate, "elapsed_s":time.monotonic()-train_started,
                    "training_status":summary.get("status"), "training_initial_eval":summary.get("initial_eval"),
                    "training_final_eval":summary.get("final_eval"), "adapter_sha256":summary.get("adapter_sha256", {}).get("after"),
                    "warmup_elapsed_s":warmup["elapsed_s"], "gameterm_score":score["score"],
                    "gameterm_failures":sum(result["failure"] is not None for result in score["results"])}
    ledger["rounds"].append(round_record)
    previous_adapter = adapter
    ledger["best_exact_correct"] = max(row["gameterm_score"]["exact_correct"] for row in ledger["rounds"])
    ledger["status"] = "passed" if score["score"]["exact_correct"] == score["score"]["total"] else "running"
    write_json_atomic(ledger_path, ledger)
    if ledger["status"] == "passed": break
  if ledger["status"] != "passed":
    ledger["status"] = "time_limit" if time.monotonic() >= deadline else "max_rounds"
  ledger["stopped_at_unix"] = time.time()
  ledger.pop("pid", None)
  write_json_atomic(ledger_path, ledger)
  return ledger


def main() -> int:
  parser = argparse.ArgumentParser(description="Resumable overnight Qwen training and GameTerm evaluation loop")
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--model", type=pathlib.Path, required=True)
  parser.add_argument("--input", type=pathlib.Path, required=True)
  parser.add_argument("--harness", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--system-prompt", required=True)
  parser.add_argument("--device", default="NV")
  parser.add_argument("--max-context", type=int, default=64)
  parser.add_argument("--steps-per-round", type=int, default=96)
  parser.add_argument("--lr", type=float, default=0.0005)
  parser.add_argument("--lr-decay", type=float, default=0.8)
  parser.add_argument("--rank", type=int, default=8)
  parser.add_argument("--alpha", type=float, default=8)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--eval-every", type=int, default=3)
  parser.add_argument("--max-hours", type=float, default=8)
  parser.add_argument("--max-rounds", type=int, default=12)
  parser.add_argument("--port", type=int, default=8091)
  parser.add_argument("--server-max-tokens", type=int, default=4)
  parser.add_argument("--prefill-chunk-size", type=int, default=64)
  parser.add_argument("--server-ready-timeout", type=float, default=90)
  parser.add_argument("--warmup-timeout", type=float, default=300)
  parser.add_argument("--harness-timeout", type=float, default=120)
  parser.add_argument("--init-adapter", type=pathlib.Path)
  args = parser.parse_args()
  for name in ("steps_per_round", "max_rounds", "server_max_tokens", "prefill_chunk_size"):
    if getattr(args, name) < 1: parser.error(f"--{name.replace('_', '-')} must be positive")
  if args.max_hours <= 0 or not 0 < args.lr_decay <= 1: parser.error("max-hours must be positive and lr-decay must be in (0, 1]")
  try:
    ledger = run(args)
  except KeyboardInterrupt:
    return 130
  print(json.dumps(ledger, indent=2, sort_keys=True))
  return 0 if ledger["status"] in ("passed", "max_rounds", "time_limit") else 1


if __name__ == "__main__": raise SystemExit(main())
