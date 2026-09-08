#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, pathlib, time
from typing import Any

import numpy as np

from tinygrad import Device, Tensor, nn
from tinygrad.llm.adapter import adapter_parameters, install_lora, load_adapter, save_adapter
from tinygrad.llm.model import Transformer
from tinygrad.llm.runtime_state import SimpleTokenizer
from extra.llm.bench.sft_smoke_train import load_sft_rows, split_rows


def sha256_file(path:pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(1024 * 1024): digest.update(chunk)
  return digest.hexdigest()


def build_completion_examples(rows:list[dict[str, Any]], tok:SimpleTokenizer, *, system_prompt:str="",
                              completion_scope:str="first") -> list[dict[str, Any]]:
  if completion_scope not in ("first", "all"): raise ValueError("completion_scope must be first or all")
  examples = []
  for row in rows:
    prefix = tok.prefix()
    if system_prompt: prefix += tok.role("system") + tok.encode(system_prompt) + tok.end_turn()
    prefix += tok.role("user") + tok.encode(row["prompt"]) + tok.end_turn() + tok.role("assistant")
    completion = tok.encode(row["completion"])
    if not completion: raise ValueError(f"{row['id']}: completion tokenized to zero tokens")
    targets = completion[:1] if completion_scope == "first" else completion + [tok.eos_id]
    for token_index, target in enumerate(targets):
      examples.append({"id": row["id"] if completion_scope == "first" else f"{row['id']}:token-{token_index}",
                       "row_id": row["id"], "source_id": row["source_id"],
                       "tokens": prefix + completion[:token_index], "target": target, "token_index": token_index,
                       "target_text": tok.decode([target]), "completion_tokens": completion})
  return examples


def build_first_completion_examples(rows:list[dict[str, Any]], tok:SimpleTokenizer, *, system_prompt:str="") -> list[dict[str, Any]]:
  return build_completion_examples(rows, tok, system_prompt=system_prompt, completion_scope="first")


def adapter_digest(adapters:list[Any]) -> str:
  digest = hashlib.sha256()
  for adapter in adapters:
    for name, array in sorted(adapter.state_arrays().items()):
      digest.update(name.encode())
      digest.update(array.tobytes(order="C"))
  return digest.hexdigest()


def token_loss(model:Transformer, example:dict[str, Any], *, device:str) -> tuple[Tensor, Tensor]:
  tokens = Tensor([example["tokens"]], device=device, dtype="int32")
  logits = model.logits(tokens, 0)[:, -1, :]
  target = Tensor([example["target"]], device=device, dtype="int32")
  return logits.sparse_categorical_crossentropy(target), logits


def evaluate(model:Transformer, examples:list[dict[str, Any]], *, device:str, tok:SimpleTokenizer|None=None) -> dict[str, Any]:
  losses, predictions, correct = [], [], 0
  for example in examples:
    loss, logits = token_loss(model, example, device=device)
    losses.append(float(loss.numpy()))
    predicted = int(logits.argmax(axis=-1).item())
    correct += predicted == example["target"]
    predictions.append({"id": example["id"], "row_id":example["row_id"], "token_index":example["token_index"],
                        "correct":predicted == example["target"], "predicted_id": predicted,
                        "predicted_text": tok.decode([predicted]) if tok is not None else None,
                        "target_id": example["target"], "target_text": example["target_text"]})
  rows = {prediction["row_id"] for prediction in predictions}
  exact_rows = sum(all(prediction["correct"] for prediction in predictions if prediction["row_id"] == row_id) for row_id in rows)
  return {"loss": float(np.mean(losses)), "accuracy": correct / len(examples), "sequence_accuracy":exact_rows / len(rows),
          "sequences":len(rows), "examples": len(examples), "losses": losses, "predictions": predictions}


def run(model_path:pathlib.Path, rows:list[dict[str, Any]], out:pathlib.Path, *, device:str, max_context:int,
        seed:int, eval_every:int, steps:int, lr:float, rank:int, alpha:float, system_prompt:str="",
        completion_scope:str="first", init_adapter:pathlib.Path|None=None) -> dict[str, Any]:
  Tensor.manual_seed(seed)
  file_before = sha256_file(model_path)
  load_started = time.perf_counter()
  model, kv = Transformer.from_gguf(model_path, max_context)
  load_s = time.perf_counter() - load_started
  tok = SimpleTokenizer.from_gguf_kv(kv)
  train_rows, eval_rows, eval_source_ids = split_rows(rows, eval_every=eval_every)
  train_examples = build_completion_examples(train_rows, tok, system_prompt=system_prompt, completion_scope=completion_scope)
  eval_examples = build_completion_examples(eval_rows, tok, system_prompt=system_prompt, completion_scope=completion_scope)
  sequence_length = max(max(len(x["tokens"]) for x in train_examples), max(len(x["tokens"]) for x in eval_examples))
  if sequence_length > max_context: raise ValueError(f"tokenized sequence length {sequence_length} exceeds max_context {max_context}")

  base_output = model.output
  adapters = load_adapter(model, init_adapter, device=device) if init_adapter is not None else \
    install_lora(model, ["output"], rank=rank, alpha=alpha, seed=seed+1, device=device)
  params = adapter_parameters(adapters)
  optimizer = nn.optim.Adam(params, lr=lr, fused=False)
  adapter_before = adapter_digest(adapters)
  initial_eval = evaluate(model, eval_examples, device=device, tok=tok)

  rng = np.random.default_rng(seed+2)
  train_losses = []
  started = time.perf_counter()
  previous_training = Tensor.training
  Tensor.training = True
  try:
    for _ in range(steps):
      example = train_examples[int(rng.integers(0, len(train_examples)))]
      loss, _ = token_loss(model, example, device=device)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_losses.append(float(loss.numpy()))
  finally:
    Tensor.training = previous_training
  train_s = time.perf_counter() - started

  final_eval = evaluate(model, eval_examples, device=device, tok=tok)
  adapter_after = adapter_digest(adapters)
  out.mkdir(parents=True, exist_ok=True)
  save_adapter(out / "adapter", adapters, base_model=str(model_path), source="gameterm-shaped-jsonl", seed=seed,
               extra={"training_scope": f"output_{completion_scope}_completion_tokens", "sequence_length": sequence_length,
                      "init_adapter":str(init_adapter) if init_adapter is not None else None})

  class ReloadModel: pass
  reloaded = ReloadModel()
  reloaded.output = base_output
  loaded = load_adapter(reloaded, out / "adapter", device=device)
  reload_digest = adapter_digest(loaded)
  file_after = sha256_file(model_path)
  checks = {
    "finite": all(np.isfinite(x) for x in initial_eval["losses"] + final_eval["losses"] + train_losses),
    "heldout_loss_decreased": final_eval["loss"] < initial_eval["loss"],
    "adapter_changed": adapter_before != adapter_after,
    "adapter_reload_exact": reload_digest == adapter_after,
    "optimizer_scope_adapter_only": len(optimizer.params) == len(params) and all(a is b for a, b in zip(optimizer.params, params)),
    "base_model_file_unchanged": file_before == file_after,
  }
  summary = {
    "kind": "tinygrad_qwen_output_lora_mvp", "status": "pass" if all(checks.values()) else "fail",
    "model": str(model_path), "model_sha256": {"before": file_before, "after": file_after}, "device": device,
    "scope": f"Qwen3-8B output LoRA; completion_scope={completion_scope}", "rows": len(rows),
    "completion_scope":completion_scope, "init_adapter":str(init_adapter) if init_adapter is not None else None,
    "system_prompt": system_prompt,
    "train_rows": len(train_rows), "eval_rows": len(eval_rows), "eval_source_ids": eval_source_ids,
    "eval_targets": [{"id": x["id"], "target_id": x["target"], "target_text": x["target_text"],
                      "completion_tokens": x["completion_tokens"]} for x in eval_examples],
    "train_examples":len(train_examples), "eval_examples":len(eval_examples),
    "sequence_length": sequence_length, "initial_eval": initial_eval, "final_eval": final_eval,
    "training": {"steps": steps, "lr": lr, "rank": rank, "alpha": alpha, "load_s": load_s, "train_s": train_s,
                 "first_loss": train_losses[0], "last_loss": train_losses[-1]},
    "adapter_sha256": {"before": adapter_before, "after": adapter_after, "reloaded": reload_digest},
    "checks": checks,
    "limitations": (["trains only the first assistant completion token"] if completion_scope == "first" else []) +
                   ["adapts only the output projection", "uses synthetic, host-labeled harness turns with unique held-out prompt wording"],
  }
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(
    "# Qwen3-8B tinygrad LoRA MVP\n\n"
    "This is the first real-model training gate: Qwen3-8B Q4_K_M remains frozen while tinygrad trains an output LoRA on NVIDIA.\n\n"
    f"- status: `{summary['status']}`\n- held-out loss: `{initial_eval['loss']:.6f}` -> `{final_eval['loss']:.6f}`\n"
    f"- adapter changed/reloaded exactly: `{checks['adapter_changed']}` / `{checks['adapter_reload_exact']}`\n"
    f"- model file unchanged: `{checks['base_model_file_unchanged']}`\n- steps: `{steps}` in `{train_s:.3f}s`\n\n"
    f"Completion scope is `{completion_scope}` and only the output projection is adapted. Transformer-block QLoRA remains a later stage.\n")
  return summary


def main() -> int:
  parser = argparse.ArgumentParser(description="Train the minimum real Qwen output LoRA with tinygrad")
  parser.add_argument("--model", type=pathlib.Path, required=True)
  parser.add_argument("--input", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--device", default=Device.DEFAULT)
  parser.add_argument("--max-context", type=int, default=64)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--eval-every", type=int, default=3)
  parser.add_argument("--steps", type=int, default=12)
  parser.add_argument("--lr", type=float, default=0.0001)
  parser.add_argument("--rank", type=int, default=8)
  parser.add_argument("--alpha", type=float, default=8.0)
  parser.add_argument("--system-prompt", default="")
  parser.add_argument("--completion-scope", choices=("first", "all"), default="first")
  parser.add_argument("--init-adapter", type=pathlib.Path)
  args = parser.parse_args()
  summary = run(args.model.expanduser().resolve(), load_sft_rows(args.input), args.out, device=args.device,
                max_context=args.max_context, seed=args.seed, eval_every=args.eval_every, steps=args.steps,
                lr=args.lr, rank=args.rank, alpha=args.alpha, system_prompt=args.system_prompt,
                completion_scope=args.completion_scope, init_adapter=args.init_adapter)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__": raise SystemExit(main())
