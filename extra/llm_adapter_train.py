#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, os, pathlib, time
from typing import Any

import numpy as np

from tinygrad import Tensor
from tinygrad.nn.optim import Adam

from extra.llm_adapter import adapter_parameters, install_lora, save_adapter
from extra.llm_eval_common import build_prompt_ids, md_text
from extra.llm_sft_smoke_train import load_sft_rows, split_rows

def configure_env(args:argparse.Namespace) -> None:
  os.environ["DEV"] = args.device
  os.environ["JIT"] = "1"
  os.environ["PYTHONPATH"] = "."
  os.environ["QK_PRIMITIVE_STORAGE"] = args.storage
  os.environ.pop("QK_GENERATED_POLICY", None)
  os.environ.pop("QK_GENERATED_POLICY_DEBUG", None)
  os.environ.pop("Q4K_PRIMITIVE_DEBUG", None)
  os.environ.pop("Q6K_PRIMITIVE_DEBUG", None)
  if args.mode == "generated":
    if args.policy is None: raise ValueError("--policy is required for mode=generated")
    os.environ["Q4K_PRIMITIVE"] = "0"
    os.environ["Q6K_PRIMITIVE"] = "0"
    os.environ["QK_GENERATED_POLICY"] = str(args.policy)
  elif args.mode == "explicit":
    os.environ["Q4K_PRIMITIVE"] = "1"
    os.environ["Q6K_PRIMITIVE"] = "1"
  elif args.mode == "baseline":
    os.environ["Q4K_PRIMITIVE"] = "0"
    os.environ["Q6K_PRIMITIVE"] = "0"
  else:
    raise ValueError(f"unknown mode {args.mode!r}")

def _build_examples(rows:list[dict[str, Any]], tok:Any, prompt_format:str, max_context:int, target_positions:str="all",
                    append_eos:bool=False) -> list[dict[str, Any]]:
  if target_positions not in ("all", "last"): raise ValueError("target_positions must be 'all' or 'last'")
  examples = []
  for row in rows:
    completion_ids = tok.encode(row["completion"])
    if append_eos:
      if getattr(tok, "eos_id", None) is None: raise ValueError("append_eos requires tokenizer eos_id")
      completion_ids = completion_ids + [tok.eos_id]
    if not completion_ids: continue
    prompt_ids = build_prompt_ids(tok, row["prompt"], prompt_format)
    positions = [len(completion_ids) - 1] if target_positions == "last" else range(len(completion_ids))
    for pos in positions:
      ids = prompt_ids + completion_ids[:pos]
      if len(ids) <= 0 or len(ids) >= max_context: continue
      examples.append({
        "id": f"{row['id']}:tok{pos}", "row_id": row["id"], "source_id": row["source_id"],
        "prompt_tokens": len(prompt_ids), "completion_tokens": len(completion_ids),
        "completion_pos": pos, "input_tokens": len(ids), "ids": ids,
        "target": completion_ids[pos], "tags": row.get("tags", []),
      })
  if not examples: raise ValueError("no adapter training examples built")
  return examples

def split_adapter_rows(rows:list[dict[str, Any]], *, eval_every:int=5) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
  splits = [row.get("split") for row in rows]
  if any(split is not None for split in splits):
    if not all(split in ("train", "eval") for split in splits):
      raise ValueError("all adapter rows with split metadata must use split='train' or split='eval'")
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval"]
    if not train_rows or not eval_rows: raise ValueError("split metadata produced an empty train or eval set")
    return train_rows, eval_rows, sorted({row["source_id"] for row in eval_rows})
  return split_rows(rows, eval_every=eval_every)

def _loss_one(model:Any, example:dict[str, Any]) -> Tensor:
  logits = model.logits(Tensor([example["ids"]], dtype="int32"), 0)[:, -1, :]
  return logits.sparse_categorical_crossentropy(Tensor([example["target"]], dtype="int32"))

def _eval_loss(model:Any, examples:list[dict[str, Any]], limit:int) -> dict[str, float]:
  losses = []
  correct = 0
  used = examples[:limit]
  for example in used:
    logits = model.logits(Tensor([example["ids"]], dtype="int32"), 0)[:, -1, :]
    losses.append(float(logits.sparse_categorical_crossentropy(Tensor([example["target"]], dtype="int32")).numpy()))
    correct += int(logits.argmax(axis=1).item() == example["target"])
  if not losses: raise ValueError("no examples evaluated")
  return {"loss": float(sum(losses) / len(losses)), "accuracy": correct / len(losses), "examples": len(losses)}

def _array_delta_l2(before:dict[str, np.ndarray], after:dict[str, np.ndarray]) -> float:
  total = 0.0
  for key, arr in after.items():
    diff = arr - before[key]
    total += float((diff * diff).sum())
  return math.sqrt(total)

def _adapter_arrays(adapters) -> dict[str, np.ndarray]:
  arrays: dict[str, np.ndarray] = {}
  for adapter in adapters: arrays.update(adapter.state_arrays())
  return arrays

def train_adapter(args:argparse.Namespace) -> tuple[dict[str, Any], Any]:
  configure_env(args)
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer

  Tensor.manual_seed(args.seed)
  rows = load_sft_rows(args.input)
  if args.max_rows > 0: rows = rows[:args.max_rows]
  train_rows, eval_rows, eval_source_ids = split_adapter_rows(rows, eval_every=args.eval_every)

  model, kv = Transformer.from_gguf(pathlib.Path(args.model).expanduser(), args.max_context)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  adapters = install_lora(model, args.targets, rank=args.rank, alpha=args.alpha, seed=args.seed)
  opt = Adam(adapter_parameters(adapters), lr=args.lr, fused=False)

  train_examples = _build_examples(train_rows, tok, args.prompt_format, args.max_context, args.target_positions, args.append_eos)
  eval_examples = _build_examples(eval_rows, tok, args.prompt_format, args.max_context, args.target_positions, args.append_eos)
  before_arrays = {k:v.copy() for k,v in _adapter_arrays(adapters).items()}
  st = time.perf_counter()
  initial = {
    "train": _eval_loss(model, train_examples, args.eval_limit),
    "eval": _eval_loss(model, eval_examples, args.eval_limit),
  }
  history = []
  rng = np.random.default_rng(args.seed)
  with Tensor.train():
    for step in range(args.steps):
      example = train_examples[int(rng.integers(0, len(train_examples)))]
      loss = _loss_one(model, example)
      opt.zero_grad()
      loss.backward()
      opt.step()
      if step == 0 or (step + 1) % max(1, args.steps // 4) == 0:
        history.append({"step": step + 1, "example": example["id"], "loss": float(loss.numpy())})
  final = {
    "train": _eval_loss(model, train_examples, args.eval_limit),
    "eval": _eval_loss(model, eval_examples, args.eval_limit),
  }
  after_arrays = _adapter_arrays(adapters)
  adapter_delta_l2 = _array_delta_l2(before_arrays, after_arrays)
  train_loss_delta = initial["train"]["loss"] - final["train"]["loss"]
  eval_loss_delta = initial["eval"]["loss"] - final["eval"]["loss"]
  status = "pass" if adapter_delta_l2 > 0 and train_loss_delta > args.min_train_loss_delta else "fail"
  summary = {
    "kind": "llm_adapter_train_summary",
    "status": status,
    "adapter_kind": "output_lora",
    "base_model": str(args.model),
    "mode": args.mode,
    "policy": str(args.policy) if args.policy else None,
    "storage": args.storage,
    "prompt_format": args.prompt_format,
    "input": str(args.input),
    "targets": args.targets,
    "rank": args.rank,
    "alpha": args.alpha,
    "rows": len(rows),
    "train_rows": len(train_rows),
    "eval_rows": len(eval_rows),
    "eval_source_ids": eval_source_ids,
    "train_examples": len(train_examples),
    "eval_examples": len(eval_examples),
    "target_positions": args.target_positions,
    "append_eos": args.append_eos,
    "optimizer": {"name": "Adam", "lr": args.lr, "steps": args.steps, "seed": args.seed},
    "initial": initial,
    "final": final,
    "deltas": {
      "train_loss": train_loss_delta,
      "eval_loss": eval_loss_delta,
      "train_accuracy": final["train"]["accuracy"] - initial["train"]["accuracy"],
      "eval_accuracy": final["eval"]["accuracy"] - initial["eval"]["accuracy"],
      "adapter_l2": adapter_delta_l2,
    },
    "elapsed_s": time.perf_counter() - st,
    "history": history,
    "artifacts": {"config": "adapter.json", "weights": "adapter.npz"},
    "note": "Output-head LoRA V0; trains adapter tensors only and does not claim model-quality improvement.",
  }
  return summary, adapters

def summary_markdown(summary:dict[str, Any]) -> str:
  lines = [
    "# LLM Adapter Train",
    "",
    "This is the first real Qwen adapter-training gate. It trains output-head",
    "LoRA tensors only, with the base GGUF model frozen. It is not QLoRA and",
    "does not claim model-quality improvement.",
    "",
    "## Summary",
    "",
    f"- status: `{summary['status']}`",
    f"- adapter: `{summary['adapter_kind']}` rank `{summary['rank']}` alpha `{summary['alpha']}`",
    f"- model: `{summary['base_model']}`",
    f"- policy: `{summary['policy']}`",
    f"- rows: `{summary['rows']}` (`{summary['train_rows']}` train, `{summary['eval_rows']}` eval)",
    f"- examples: `{summary['train_examples']}` train, `{summary['eval_examples']}` eval",
    f"- adapter L2 delta: `{summary['deltas']['adapter_l2']:.6f}`",
    "",
    "## Metrics",
    "",
    "| split | initial loss | final loss | loss delta | initial acc | final acc | acc delta |",
    "|---|---:|---:|---:|---:|---:|---:|",
  ]
  for split in ("train", "eval"):
    init, final = summary["initial"][split], summary["final"][split]
    lines.append(
      f"| `{split}` | {init['loss']:.4f} | {final['loss']:.4f} | {init['loss'] - final['loss']:.4f} | "
      f"{init['accuracy']:.4f} | {final['accuracy']:.4f} | {final['accuracy'] - init['accuracy']:.4f} |")
  lines += ["", "## History", "", "| step | example | loss |", "|---:|---|---:|"]
  for row in summary["history"]:
    lines.append(f"| {row['step']} | `{row['example']}` | {row['loss']:.4f} |")
  lines += ["", "## Eval Source IDs", "", md_text(", ".join(summary["eval_source_ids"])), ""]
  return "\n".join(lines)

def write_artifact(out:pathlib.Path, summary:dict[str, Any], adapters) -> None:
  out.mkdir(parents=True, exist_ok=True)
  save_adapter(out, adapters, base_model=summary["base_model"], source=summary["input"],
               seed=summary["optimizer"]["seed"], extra={"train_summary": "train-summary.json"})
  (out / "train-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(summary_markdown(summary))

def main() -> int:
  parser = argparse.ArgumentParser(description="Train an output-head LoRA adapter on rollout-derived SFT rows")
  parser.add_argument("--model", type=pathlib.Path, required=True)
  parser.add_argument("--policy", type=pathlib.Path)
  parser.add_argument("--input", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--mode", choices=("generated", "explicit", "baseline"), default="generated")
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--storage", default="shared")
  parser.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
  parser.add_argument("--targets", nargs="+", default=["output"])
  parser.add_argument("--rank", type=int, default=4)
  parser.add_argument("--alpha", type=float, default=8.0)
  parser.add_argument("--lr", type=float, default=0.05)
  parser.add_argument("--steps", type=int, default=8)
  parser.add_argument("--seed", type=int, default=20260613)
  parser.add_argument("--max-rows", type=int, default=20)
  parser.add_argument("--eval-every", type=int, default=5)
  parser.add_argument("--eval-limit", type=int, default=8)
  parser.add_argument("--target-positions", choices=("all", "last"), default="all")
  parser.add_argument("--append-eos", action="store_true", help="train one EOS target after each completion so rollout can stop")
  parser.add_argument("--max-context", type=int, default=4096)
  parser.add_argument("--min-train-loss-delta", type=float, default=0.0)
  args = parser.parse_args()

  summary, adapters = train_adapter(args)
  write_artifact(args.out, summary, adapters)
  print(summary_markdown(summary))
  return 0 if summary["status"] == "pass" else 1

if __name__ == "__main__":
  raise SystemExit(main())
