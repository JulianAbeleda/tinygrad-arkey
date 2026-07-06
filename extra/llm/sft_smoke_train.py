#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np

from tinygrad import Device, Tensor
from extra.llm.eval_common import md_text, read_id_jsonl


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
  rows = read_id_jsonl(path)
  if not rows:
    raise ValueError(f"{path}: no rows")
  return rows


def load_sft_rows(path: pathlib.Path) -> list[dict[str, Any]]:
  rows = _load_jsonl(path)
  for idx, row in enumerate(rows):
    for key in ("prompt", "completion", "source_id"):
      if not isinstance(row.get(key), str) or not row[key]:
        raise ValueError(f"{path}: row {idx} missing string {key}")
  return rows


def split_rows(rows: list[dict[str, Any]], *, eval_every: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
  if eval_every < 2:
    raise ValueError("eval_every must be >= 2")
  ids = sorted({row["source_id"] for row in rows})
  eval_ids = set(ids[::eval_every])
  train = [row for row in rows if row["source_id"] not in eval_ids]
  eval_rows = [row for row in rows if row["source_id"] in eval_ids]
  if not train or not eval_rows:
    raise ValueError("split produced an empty train or eval set")
  return train, eval_rows, sorted(eval_ids)


def _training_text(row: dict[str, Any]) -> str:
  return f"<user>\n{row['prompt']}\n<assistant>\n{row['completion']}"


def build_byte_examples(rows: list[dict[str, Any]], *, context_bytes: int = 4, vocab_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
  if context_bytes < 1:
    raise ValueError("context_bytes must be >= 1")
  if vocab_size < 2 or vocab_size > 256:
    raise ValueError("vocab_size must be in [2, 256]")

  feature_dim = context_bytes * vocab_size + 1
  xs: list[np.ndarray] = []
  ys: list[int] = []
  for row in rows:
    data = _training_text(row).encode("utf-8", "replace")
    if len(data) < 2:
      continue
    padded = [0] * context_bytes + list(data)
    for i in range(context_bytes, len(padded) - 1):
      feat = np.zeros(feature_dim, dtype=np.float32)
      feat[-1] = 1.0
      for j in range(context_bytes):
        feat[j * vocab_size + (padded[i - context_bytes + 1 + j] % vocab_size)] = 1.0
      xs.append(feat)
      ys.append(padded[i + 1] % vocab_size)

  if not xs:
    raise ValueError("no byte examples built")
  return np.stack(xs), np.asarray(ys, dtype=np.int32)


def _metrics(x: np.ndarray, y: np.ndarray, w: Tensor, b: Tensor, *, device: str, max_examples: int) -> dict[str, float]:
  x_eval, y_eval = x[:max_examples], y[:max_examples]
  logits = Tensor(x_eval, device=device) @ w + b
  target = Tensor(y_eval, device=device)
  loss = logits.sparse_categorical_crossentropy(target)
  acc = (logits.argmax(axis=1) == target).mean()
  return {
    "loss": float(loss.numpy()),
    "accuracy": float(acc.numpy()),
    "examples": int(len(y_eval)),
  }


def run_smoke_probe(
    rows: list[dict[str, Any]],
    *,
    device: str,
    seed: int = 123,
    eval_every: int = 5,
    context_bytes: int = 4,
    vocab_size: int = 128,
    max_eval_examples: int = 4096,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  train_rows, eval_rows, eval_source_ids = split_rows(rows, eval_every=eval_every)
  x_train, y_train = build_byte_examples(train_rows, context_bytes=context_bytes, vocab_size=vocab_size)
  x_eval, y_eval = build_byte_examples(eval_rows, context_bytes=context_bytes, vocab_size=vocab_size)
  feature_dim = x_train.shape[1]

  rng = np.random.default_rng(seed)
  w = Tensor((rng.standard_normal((feature_dim, vocab_size), dtype=np.float32).astype(np.float32) * 0.01), device=device)
  b = Tensor(np.zeros((vocab_size,), dtype=np.float32), device=device)

  start = time.perf_counter()
  initial_train = _metrics(x_train, y_train, w, b, device=device, max_examples=max_eval_examples)
  initial_eval = _metrics(x_eval, y_eval, w, b, device=device, max_examples=max_eval_examples)
  elapsed_s = time.perf_counter() - start

  checks = (
    np.isfinite(initial_train["loss"]),
    np.isfinite(initial_eval["loss"]),
    0.0 <= initial_train["accuracy"] <= 1.0,
    0.0 <= initial_eval["accuracy"] <= 1.0,
  )
  status = "pass" if all(checks) else "fail"

  summary = {
    "kind": "llm_sft_smoke_inference_summary",
    "status": status,
    "mode": "inference_only",
    "model": "byte_context_softmax",
    "purpose": "inference smoke test over SFT rows",
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
    "initial": {
      "train": initial_train,
      "eval": initial_eval,
    },
    "deltas": {
      "train_loss": 0.0,
      "eval_loss": 0.0,
      "train_accuracy": 0.0,
      "eval_accuracy": 0.0,
    },
    "elapsed_s": elapsed_s,
    "artifacts": {"weights": "model.npz"},
  }
  return summary, {"weight": w.numpy(), "bias": b.numpy()}


def summary_markdown(summary: dict[str, Any]) -> str:
  lines = [
    "# LLM SFT Smoke Train (inference only)",
    "",
    "This smoke script validates tinygrad inference plumbing over rollout-derived SFT rows.",
    "It does not run backward, call optimizers, or update parameters.",
    "",
    "## Summary",
    "",
    f"- status: `{summary['status']}`",
    f"- rows: `{summary['rows']}` (`{summary['train_rows']}` train, `{summary['eval_rows']}` eval)",
    f"- byte examples: `{summary['train_examples']}` train, `{summary['eval_examples']}` eval",
    f"- model: `{summary['model']}`",
    f"- device: `{summary['device']}`",
    f"- mode: `{summary['mode']}`",
    "",
    "## Metrics",
    "",
    "| split | loss | accuracy | examples |",
    "|---|---:|---:|---:|",
  ]
  for split in ("train", "eval"):
    metrics = summary["initial"][split]
    lines.append(f"| `{split}` | {metrics['loss']:.4f} | {metrics['accuracy']:.4f} | {metrics['examples']} |")

  lines += [
    "",
    "## Eval Source IDs",
    "",
    md_text(", ".join(summary["eval_source_ids"])),
    "",
  ]
  return "\n".join(lines)


def write_artifact(out: pathlib.Path, summary: dict[str, Any], weights: dict[str, np.ndarray]) -> None:
  out.mkdir(parents=True, exist_ok=True)
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(summary_markdown(summary))
  np.savez_compressed(out / "model.npz", **weights)


def main() -> int:
  parser = argparse.ArgumentParser(description="Run an inference-only SFT smoke check over rollout-derived rows")
  parser.add_argument("--input", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--device", default=Device.DEFAULT)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--eval-every", type=int, default=5)
  parser.add_argument("--context-bytes", type=int, default=4)
  parser.add_argument("--vocab-size", type=int, default=128)
  parser.add_argument("--max-eval-examples", type=int, default=4096)

  # Legacy training-oriented arguments are intentionally ignored in this inference-only branch.
  parser.add_argument("--steps", type=int, default=80, help="legacy; ignored")
  parser.add_argument("--batch-size", type=int, default=512, help="legacy; ignored")
  parser.add_argument("--lr", type=float, default=0.04, help="legacy; ignored")

  args = parser.parse_args()

  rows = load_sft_rows(args.input)
  summary, weights = run_smoke_probe(
    rows,
    device=args.device,
    seed=args.seed,
    eval_every=args.eval_every,
    context_bytes=args.context_bytes,
    vocab_size=args.vocab_size,
    max_eval_examples=args.max_eval_examples,
  )
  summary["input"] = str(args.input)
  write_artifact(args.out, summary, weights)
  print(summary_markdown(summary))
  return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
  raise SystemExit(main())
