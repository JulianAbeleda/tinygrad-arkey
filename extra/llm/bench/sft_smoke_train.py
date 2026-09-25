#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, pathlib, time
from typing import Any

import numpy as np

from tinygrad import Device, Tensor, nn
from tinygrad.llm.adapter import adapter_parameters, install_lora, load_adapter, save_adapter
from extra.llm.bench.eval_common import md_text, read_id_jsonl


def _load_jsonl(path:pathlib.Path) -> list[dict[str, Any]]:
  rows = read_id_jsonl(path)
  if not rows: raise ValueError(f"{path}: no rows")
  return rows


def load_sft_rows(path:pathlib.Path, *, native:bool=False) -> list[dict[str, Any]]:
  rows = _load_jsonl(path)
  for idx, row in enumerate(rows):
    if native and 'messages' in row:
      from tinygrad.llm.chat import validate_messages
      if 'prompt' in row or 'completion' in row: raise ValueError('ambiguous legacy and native training row')
      validate_messages(row['messages'], row.get('tools', []), target=True)
      if row['messages'][-1]['role'] != 'assistant': raise ValueError('native row requires final assistant target')
      required = ('source_id',)
    else: required = ('prompt', 'completion', 'source_id')
    for key in required:
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


def _prefix(row:dict[str, Any]) -> bytes:
  return f"<user>\n{row['prompt']}\n<assistant>\n".encode("utf-8", "replace")


def build_byte_examples(rows:list[dict[str, Any]], *, context_bytes:int=8, vocab_size:int=256) -> tuple[np.ndarray, np.ndarray]:
  """Build a bounded causal probe whose labels contain assistant bytes only.

  Prompt bytes remain visible in the causal context, but never become targets.
  The byte probe tests the training/adapter lifecycle without pretending to be
  the Qwen tokenizer or a useful language model.
  """
  if context_bytes < 1: raise ValueError("context_bytes must be >= 1")
  if vocab_size < 2 or vocab_size > 256: raise ValueError("vocab_size must be in [2, 256]")
  feature_dim = context_bytes * vocab_size + 1
  xs:list[np.ndarray] = []
  ys:list[int] = []
  for row in rows:
    prefix = _prefix(row)
    completion = (row["completion"] + "\n").encode("utf-8", "replace")
    padded = bytes(context_bytes) + prefix + completion
    first_target = context_bytes + len(prefix)
    for target_pos in range(first_target, len(padded)):
      feat = np.zeros(feature_dim, dtype=np.float32)
      feat[-1] = 1.0
      for slot, value in enumerate(padded[target_pos-context_bytes:target_pos]): feat[slot*vocab_size + value % vocab_size] = 1.0
      xs.append(feat)
      ys.append(padded[target_pos] % vocab_size)
  if not xs: raise ValueError("no completion examples built")
  return np.stack(xs), np.asarray(ys, dtype=np.int32)


class FrozenByteModel:
  """A deterministic frozen base with the same output module seam as the LLM."""
  def __init__(self, feature_dim:int, vocab_size:int, *, seed:int, device:str):
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(feature_dim)
    self.output = nn.Linear(feature_dim, vocab_size)
    self.output.weight = Tensor((rng.standard_normal((vocab_size, feature_dim))*scale).astype(np.float32), device=device).realize()
    self.output.bias = Tensor.zeros(vocab_size, device=device).realize()

  def __call__(self, x:Tensor) -> Tensor: return self.output(x)


def _array_digest(*arrays:np.ndarray) -> str:
  digest = hashlib.sha256()
  for array in arrays:
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
  return digest.hexdigest()


def _base_arrays(model:FrozenByteModel) -> tuple[np.ndarray, np.ndarray]:
  base = model.output.base if hasattr(model.output, "base") else model.output
  return base.weight.numpy(), base.bias.numpy()


def _adapter_digest(adapters:list[Any]) -> str:
  return _array_digest(*(array.copy() for adapter in adapters for array in adapter.state_arrays().values()))


def _metrics(x:np.ndarray, y:np.ndarray, model:FrozenByteModel, *, device:str, max_examples:int) -> dict[str, float|int]:
  x_eval, y_eval = x[:max_examples], y[:max_examples]
  logits = model(Tensor(x_eval, device=device))
  target = Tensor(y_eval, device=device)
  loss = logits.sparse_categorical_crossentropy(target)
  return {"loss": float(loss.numpy()), "accuracy": float((logits.argmax(axis=1) == target).mean().numpy()), "examples": len(y_eval)}


def run_smoke_train(rows:list[dict[str, Any]], *, device:str, seed:int=123, eval_every:int=5, context_bytes:int=8,
                    vocab_size:int=256, max_eval_examples:int=4096, steps:int=120, batch_size:int=256,
                    lr:float=0.02, rank:int=16, alpha:float=16.0, adapter_out:pathlib.Path|None=None,
                    source:str="in-memory") -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  if steps < 1 or batch_size < 1: raise ValueError("steps and batch_size must be >= 1")
  train_rows, eval_rows, eval_source_ids = split_rows(rows, eval_every=eval_every)
  x_train, y_train = build_byte_examples(train_rows, context_bytes=context_bytes, vocab_size=vocab_size)
  x_eval, y_eval = build_byte_examples(eval_rows, context_bytes=context_bytes, vocab_size=vocab_size)
  feature_dim = x_train.shape[1]

  model = FrozenByteModel(feature_dim, vocab_size, seed=seed, device=device)
  base_before = _base_arrays(model)
  base_digest_before = _array_digest(*base_before)
  adapters = install_lora(model, ["output"], rank=rank, alpha=alpha, seed=seed+1, device=device)
  optimizer = nn.optim.Adam(adapter_parameters(adapters), lr=lr)
  adapter_digest_before = _adapter_digest(adapters)
  initial_train = _metrics(x_train, y_train, model, device=device, max_examples=max_eval_examples)
  initial_eval = _metrics(x_eval, y_eval, model, device=device, max_examples=max_eval_examples)

  rng = np.random.default_rng(seed+2)
  losses:list[float] = []
  started = time.perf_counter()
  previous_training = Tensor.training
  Tensor.training = True
  try:
    for step in range(steps):
      indices = rng.integers(0, len(y_train), size=min(batch_size, len(y_train)))
      logits = model(Tensor(x_train[indices], device=device))
      loss = logits.sparse_categorical_crossentropy(Tensor(y_train[indices], device=device))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if step in (0, steps-1): losses.append(float(loss.numpy()))
  finally:
    Tensor.training = previous_training
  elapsed_s = time.perf_counter() - started

  final_train = _metrics(x_train, y_train, model, device=device, max_examples=max_eval_examples)
  final_eval = _metrics(x_eval, y_eval, model, device=device, max_examples=max_eval_examples)
  adapter_digest_after = _adapter_digest(adapters)
  base_after = _base_arrays(model)
  base_digest_after = _array_digest(*base_after)
  if adapter_out is not None:
    save_adapter(adapter_out, adapters, base_model="frozen-byte-probe-v1", source=source, seed=seed,
                 extra={"context_bytes": context_bytes, "vocab_size": vocab_size, "feature_dim": feature_dim})
    fresh = FrozenByteModel(feature_dim, vocab_size, seed=seed, device=device)
    load_adapter(fresh, adapter_out, device=device)
    probe = Tensor(x_eval[:min(64, len(x_eval))], device=device)
    reload_max_abs = float(np.max(np.abs(model(probe).numpy() - fresh(probe).numpy())))
  else: reload_max_abs = None

  checks = {
    "finite": all(np.isfinite(float(x["loss"])) for x in (initial_train, initial_eval, final_train, final_eval)),
    "train_loss_decreased": final_train["loss"] < initial_train["loss"],
    "eval_loss_decreased": final_eval["loss"] < initial_eval["loss"],
    "base_unchanged": base_digest_before == base_digest_after,
    "adapter_changed": adapter_digest_before != adapter_digest_after,
    "adapter_reload_exact": reload_max_abs is None or reload_max_abs <= 1e-6,
  }
  summary = {
    "kind": "tinygrad_lora_self_training_mvp", "status": "pass" if all(checks.values()) else "fail", "mode": "train",
    "model": "frozen_byte_context_plus_lora", "purpose": "prove tinygrad optimizer and LoRA lifecycle over harness-shaped SFT rows",
    "device": device, "rows": len(rows), "train_rows": len(train_rows), "eval_rows": len(eval_rows),
    "eval_source_ids": eval_source_ids, "train_examples": len(y_train), "eval_examples": len(y_eval),
    "context_bytes": context_bytes, "vocab_size": vocab_size, "feature_dim": feature_dim,
    "training": {"steps": steps, "batch_size": batch_size, "lr": lr, "rank": rank, "alpha": alpha,
                 "first_batch_loss": losses[0], "last_batch_loss": losses[-1], "elapsed_s": elapsed_s},
    "initial": {"train": initial_train, "eval": initial_eval}, "final": {"train": final_train, "eval": final_eval},
    "deltas": {"train_loss": final_train["loss"]-initial_train["loss"], "eval_loss": final_eval["loss"]-initial_eval["loss"],
               "train_accuracy": final_train["accuracy"]-initial_train["accuracy"],
               "eval_accuracy": final_eval["accuracy"]-initial_eval["accuracy"]},
    "base_sha256": {"before": base_digest_before, "after": base_digest_after},
    "adapter_sha256": {"before": adapter_digest_before, "after": adapter_digest_after},
    "reload_max_abs": reload_max_abs, "checks": checks,
    "artifacts": {"adapter": "adapter", "frozen_base": "frozen_base.npz"},
  }
  return summary, {"weight": base_after[0], "bias": base_after[1]}


def summary_markdown(summary:dict[str, Any]) -> str:
  lines = [
    "# Tinygrad LoRA self-training MVP", "",
    "This is a real optimizer/update proof over rollout-shaped SFT rows. It uses a deterministic frozen byte-context base",
    "to qualify completion-only masking, tinygrad backward/Adam, LoRA-only mutation, adapter persistence, and held-out evaluation.",
    "It is substrate evidence, not a claim that Qwen3-8B has been fine-tuned.", "", "## Summary", "",
    f"- status: `{summary['status']}`", f"- rows: `{summary['rows']}` (`{summary['train_rows']}` train, `{summary['eval_rows']}` eval)",
    f"- completion examples: `{summary['train_examples']}` train, `{summary['eval_examples']}` eval",
    f"- device: `{summary['device']}`", f"- steps/rank: `{summary['training']['steps']}` / `{summary['training']['rank']}`",
    f"- frozen base unchanged: `{summary['checks']['base_unchanged']}`",
    f"- adapter reload max abs: `{summary['reload_max_abs']}`", "", "## Metrics", "",
    "| split | initial loss | final loss | delta | initial accuracy | final accuracy |", "|---|---:|---:|---:|---:|---:|",
  ]
  for split in ("train", "eval"):
    before, after = summary["initial"][split], summary["final"][split]
    lines.append(f"| `{split}` | {before['loss']:.4f} | {after['loss']:.4f} | {after['loss']-before['loss']:.4f} | {before['accuracy']:.4f} | {after['accuracy']:.4f} |")
  lines += ["", "## Held-out source IDs", "", md_text(", ".join(summary["eval_source_ids"])), ""]
  return "\n".join(lines)


def write_artifact(out:pathlib.Path, summary:dict[str, Any], base:dict[str, np.ndarray]) -> None:
  out.mkdir(parents=True, exist_ok=True)
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(summary_markdown(summary))
  np.savez_compressed(out / "frozen_base.npz", **base)


def main() -> int:
  parser = argparse.ArgumentParser(description="Run the tinygrad LoRA self-training MVP over rollout-derived rows")
  parser.add_argument("--input", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--device", default=Device.DEFAULT)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--eval-every", type=int, default=5)
  parser.add_argument("--context-bytes", type=int, default=8)
  parser.add_argument("--vocab-size", type=int, default=256)
  parser.add_argument("--max-eval-examples", type=int, default=4096)
  parser.add_argument("--steps", type=int, default=120)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--lr", type=float, default=0.02)
  parser.add_argument("--rank", type=int, default=16)
  parser.add_argument("--alpha", type=float, default=16.0)
  args = parser.parse_args()
  rows = load_sft_rows(args.input)
  summary, base = run_smoke_train(rows, device=args.device, seed=args.seed, eval_every=args.eval_every,
    context_bytes=args.context_bytes, vocab_size=args.vocab_size, max_eval_examples=args.max_eval_examples,
    steps=args.steps, batch_size=args.batch_size, lr=args.lr, rank=args.rank, alpha=args.alpha,
    adapter_out=args.out / "adapter", source=str(args.input))
  summary["input"] = str(args.input)
  write_artifact(args.out, summary, base)
  print(summary_markdown(summary))
  return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__": raise SystemExit(main())
