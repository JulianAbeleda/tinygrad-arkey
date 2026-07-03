#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, time
from typing import Any

import numpy as np

from tinygrad import Device, Tensor
from tinygrad.nn.optim import Adam

from extra.llm.eval_common import md_text, read_id_jsonl

def _load_jsonl(path:pathlib.Path) -> list[dict[str, Any]]:
  rows = read_id_jsonl(path)
  if not rows: raise ValueError(f"{path}: no rows")
  return rows

def load_sft_rows(path:pathlib.Path) -> list[dict[str, Any]]:
  rows = _load_jsonl(path)
  for idx, row in enumerate(rows):
    for key in ("prompt", "completion", "source_id"):
      if not isinstance(row.get(key), str) or not row[key]: raise ValueError(f"{path}: row {idx} missing string {key}")
  return rows

def split_rows(rows:list[dict[str, Any]], *, eval_every:int=5) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
  if eval_every < 2: raise ValueError("eval_every must be >= 2")
  ids = sorted({row["source_id"] for row in rows})
  eval_ids = set(ids[::eval_every])
  train = [row for row in rows if row["source_id"] not in eval_ids]
  eval_rows = [row for row in rows if row["source_id"] in eval_ids]
  if not train or not eval_rows: raise ValueError("split produced an empty train or eval set")
  return train, eval_rows, sorted(eval_ids)

def _training_text(row:dict[str, Any]) -> str:
  return f"<user>\n{row['prompt']}\n<assistant>\n{row['completion']}"

def build_byte_examples(rows:list[dict[str, Any]], *, context_bytes:int=4, vocab_size:int=128) -> tuple[np.ndarray, np.ndarray]:
  if context_bytes < 1: raise ValueError("context_bytes must be >= 1")
  if vocab_size < 2 or vocab_size > 256: raise ValueError("vocab_size must be in [2, 256]")
  feature_dim = context_bytes * vocab_size + 1
  xs: list[np.ndarray] = []
  ys: list[int] = []
  for row in rows:
    data = _training_text(row).encode("utf-8", "replace")
    if len(data) < 2: continue
    padded = [0] * context_bytes + list(data)
    for i in range(context_bytes, len(padded) - 1):
      feat = np.zeros(feature_dim, dtype=np.float32)
      feat[-1] = 1.0
      for j in range(context_bytes):
        feat[j * vocab_size + (padded[i - context_bytes + 1 + j] % vocab_size)] = 1.0
      xs.append(feat)
      ys.append(padded[i + 1] % vocab_size)
  if not xs: raise ValueError("no byte examples built")
  return np.stack(xs), np.asarray(ys, dtype=np.int32)

def _metrics(x:np.ndarray, y:np.ndarray, w:Tensor, b:Tensor, *, device:str, max_examples:int) -> dict[str, float]:
  x_eval, y_eval = x[:max_examples], y[:max_examples]
  logits = Tensor(x_eval, device=device) @ w + b
  loss = logits.sparse_categorical_crossentropy(Tensor(y_eval, device=device))
  acc = (logits.argmax(axis=1) == Tensor(y_eval, device=device)).mean()
  return {"loss": float(loss.numpy()), "accuracy": float(acc.numpy()), "examples": int(len(y_eval))}

def train_probe(rows:list[dict[str, Any]], *, device:str, steps:int=80, batch_size:int=512, lr:float=0.04, seed:int=123,
                eval_every:int=5, context_bytes:int=4, vocab_size:int=128, max_eval_examples:int=4096) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  if steps < 1: raise ValueError("steps must be >= 1")
  if batch_size < 1: raise ValueError("batch_size must be >= 1")
  train_rows, eval_rows, eval_source_ids = split_rows(rows, eval_every=eval_every)
  x_train, y_train = build_byte_examples(train_rows, context_bytes=context_bytes, vocab_size=vocab_size)
  x_eval, y_eval = build_byte_examples(eval_rows, context_bytes=context_bytes, vocab_size=vocab_size)
  feature_dim = x_train.shape[1]

  rng = np.random.default_rng(seed)
  w = Tensor((rng.standard_normal((feature_dim, vocab_size)) * 0.01).astype(np.float32), device=device).is_param_()
  b = Tensor(np.zeros((vocab_size,), dtype=np.float32), device=device).is_param_()
  opt = Adam([w, b], lr=lr, fused=False)

  start = time.perf_counter()
  initial = {
    "train": _metrics(x_train, y_train, w, b, device=device, max_examples=max_eval_examples),
    "eval": _metrics(x_eval, y_eval, w, b, device=device, max_examples=max_eval_examples),
  }
  history = []
  with Tensor.train():
    for step in range(steps):
      idx = rng.integers(0, len(y_train), size=batch_size)
      logits = Tensor(x_train[idx], device=device) @ w + b
      loss = logits.sparse_categorical_crossentropy(Tensor(y_train[idx], device=device))
      opt.zero_grad()
      loss.backward()
      opt.step()
      if step == 0 or (step + 1) % max(1, steps // 4) == 0:
        history.append({"step": step + 1, "batch_loss": float(loss.numpy())})
  final = {
    "train": _metrics(x_train, y_train, w, b, device=device, max_examples=max_eval_examples),
    "eval": _metrics(x_eval, y_eval, w, b, device=device, max_examples=max_eval_examples),
  }
  elapsed_s = time.perf_counter() - start
  train_loss_delta = initial["train"]["loss"] - final["train"]["loss"]
  eval_loss_delta = initial["eval"]["loss"] - final["eval"]["loss"]
  status = "pass" if train_loss_delta > 0.5 and eval_loss_delta > 0.5 and final["eval"]["accuracy"] > 0.2 else "fail"
  summary = {
    "kind": "llm_sft_smoke_train_summary",
    "status": status,
    "model": "byte_context_softmax",
    "purpose": "training-loop smoke test over rollout-derived SFT rows; not a Qwen adapter",
    "device": device,
    "rows": len(rows),
    "train_rows": len(train_rows),
    "eval_rows": len(eval_rows),
    "eval_source_ids": eval_source_ids,
    "train_examples": int(len(y_train)),
    "eval_examples": int(len(y_eval)),
    "context_bytes": context_bytes,
    "vocab_size": vocab_size,
    "feature_dim": int(feature_dim),
    "optimizer": {"name": "Adam", "lr": lr, "steps": steps, "batch_size": batch_size, "seed": seed},
    "initial": initial,
    "final": final,
    "deltas": {
      "train_loss": train_loss_delta,
      "eval_loss": eval_loss_delta,
      "train_accuracy": final["train"]["accuracy"] - initial["train"]["accuracy"],
      "eval_accuracy": final["eval"]["accuracy"] - initial["eval"]["accuracy"],
    },
    "elapsed_s": elapsed_s,
    "history": history,
    "artifacts": {"weights": "model.npz"},
  }
  return summary, {"weight": w.numpy(), "bias": b.numpy()}

def summary_markdown(summary:dict[str, Any]) -> str:
  lines = [
    "# LLM SFT Smoke Train",
    "",
    "This is a tinygrad optimization smoke test over rollout-derived SFT rows.",
    "It validates training/eval plumbing only; it is not a Qwen fine-tune or",
    "adapter.",
    "",
    "## Summary",
    "",
    f"- status: `{summary['status']}`",
    f"- rows: `{summary['rows']}` (`{summary['train_rows']}` train, `{summary['eval_rows']}` eval)",
    f"- byte examples: `{summary['train_examples']}` train, `{summary['eval_examples']}` eval",
    f"- model: `{summary['model']}`",
    f"- device: `{summary['device']}`",
    f"- optimizer: `{summary['optimizer']['name']}`, steps `{summary['optimizer']['steps']}`, batch `{summary['optimizer']['batch_size']}`, lr `{summary['optimizer']['lr']}`",
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
  lines += ["", "## History", "", "| step | batch loss |", "|---:|---:|"]
  for row in summary["history"]:
    lines.append(f"| {row['step']} | {row['batch_loss']:.4f} |")
  lines += ["", "## Eval Source IDs", "", md_text(", ".join(summary["eval_source_ids"])), ""]
  return "\n".join(lines)

def write_artifact(out:pathlib.Path, summary:dict[str, Any], weights:dict[str, np.ndarray]) -> None:
  out.mkdir(parents=True, exist_ok=True)
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(summary_markdown(summary))
  np.savez_compressed(out / "model.npz", **weights)

def main() -> int:
  parser = argparse.ArgumentParser(description="Train a tiny byte-context SFT smoke model over rollout-derived rows")
  parser.add_argument("--input", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--device", default=Device.DEFAULT)
  parser.add_argument("--steps", type=int, default=80)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--lr", type=float, default=0.04)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--eval-every", type=int, default=5)
  parser.add_argument("--context-bytes", type=int, default=4)
  parser.add_argument("--vocab-size", type=int, default=128)
  parser.add_argument("--max-eval-examples", type=int, default=4096)
  args = parser.parse_args()

  rows = load_sft_rows(args.input)
  summary, weights = train_probe(rows, device=args.device, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                                 seed=args.seed, eval_every=args.eval_every, context_bytes=args.context_bytes,
                                 vocab_size=args.vocab_size, max_eval_examples=args.max_eval_examples)
  summary["input"] = str(args.input)
  write_artifact(args.out, summary, weights)
  print(summary_markdown(summary))
  return 0 if summary["status"] == "pass" else 1

if __name__ == "__main__":
  raise SystemExit(main())
