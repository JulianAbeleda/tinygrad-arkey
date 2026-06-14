#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re, time
from dataclasses import dataclass
from typing import Any

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import Adam

from extra.llm_adapter import adapter_parameters, install_lora, save_adapter
from extra.llm_adapter_train import _adapter_arrays, _array_delta_l2, _build_examples, _plain_logits, configure_env, split_adapter_rows
from extra.llm_eval_common import md_text
from extra.llm_sft_smoke_train import load_sft_rows

@dataclass
class CachedExample:
  example: dict[str, Any]
  hidden: np.ndarray

def suffix_start_from_targets(model:Any, targets:list[str]) -> int:
  if len(targets) != 1: raise ValueError("suffix training supports exactly one target group, for example --targets last1_ffn")
  m = re.fullmatch(r"last(\d+)_ffn", targets[0])
  if m is None: raise ValueError("suffix training requires a lastN_ffn target group")
  if not hasattr(model, "blk") or not isinstance(model.blk, list): raise ValueError("model has no block list")
  count = int(m.group(1))
  if count < 1: raise ValueError("lastN_ffn count must be >= 1")
  if count > len(model.blk): raise ValueError(f"{targets[0]} requested {count} blocks, model has {len(model.blk)}")
  return len(model.blk) - count

def block_forward_plain(block:Any, x:Tensor, start_pos:int) -> Tensor:
  block._init_state(x)
  h = x + block._attention(block.attn_norm(x), start_pos)
  return (h + block._feed_forward(block.ffn_norm(h))).contiguous()

def prefix_hidden_tensor(model:Any, ids:list[int], suffix_start:int) -> Tensor:
  x = model.token_embd(Tensor([ids], dtype="int32")).float()
  for block in model.blk[:suffix_start]: x = block(x, 0)
  return x

def prefix_hidden_numpy(model:Any, ids:list[int], suffix_start:int, cache_dtype:str) -> np.ndarray:
  x = prefix_hidden_tensor(model, ids, suffix_start)
  if cache_dtype == "float16": x = x.cast(dtypes.float16)
  elif cache_dtype == "float32": x = x.cast(dtypes.float32)
  else: raise ValueError(f"unknown cache dtype {cache_dtype!r}")
  return x.realize().numpy().copy()

def suffix_logits_from_hidden(model:Any, hidden:Tensor, suffix_start:int) -> Tensor:
  x = hidden.float()
  for block in model.blk[suffix_start:]: x = block_forward_plain(block, x, 0)
  return model.output(model.output_norm(x))

def _last_logits_from_hidden(model:Any, hidden_np:np.ndarray, suffix_start:int, device:str|None) -> Tensor:
  return suffix_logits_from_hidden(model, Tensor(hidden_np, device=device), suffix_start)[:, -1, :]

def _loss_one_suffix(model:Any, cached:CachedExample, suffix_start:int, device:str|None) -> Tensor:
  logits = _last_logits_from_hidden(model, cached.hidden, suffix_start, device)
  return logits.sparse_categorical_crossentropy(Tensor([cached.example["target"]], dtype="int32"))

def _eval_loss_suffix(model:Any, cached_examples:list[CachedExample], suffix_start:int, device:str|None, limit:int) -> dict[str, float]:
  losses = []
  correct = 0
  used = cached_examples[:limit]
  for cached in used:
    logits = _last_logits_from_hidden(model, cached.hidden, suffix_start, device)
    losses.append(float(logits.sparse_categorical_crossentropy(Tensor([cached.example["target"]], dtype="int32")).numpy()))
    correct += int(logits.argmax(axis=1).item() == cached.example["target"])
  if not losses: raise ValueError("no cached examples evaluated")
  return {"loss": float(sum(losses) / len(losses)), "accuracy": correct / len(losses), "examples": len(losses)}

def build_suffix_cache(model:Any, examples:list[dict[str, Any]], suffix_start:int, cache_dtype:str) -> tuple[list[CachedExample], dict[str, Any]]:
  by_row: dict[str, dict[str, Any]] = {}
  for example in examples:
    row_key = str(example.get("row_id", example["id"]))
    if row_key not in by_row or len(example["ids"]) > len(by_row[row_key]["ids"]): by_row[row_key] = example
  hidden_by_row: dict[str, np.ndarray] = {}
  token_counts = []
  with Tensor.train(False):
    for row_key, example in by_row.items():
      hidden_by_row[row_key] = prefix_hidden_numpy(model, example["ids"], suffix_start, cache_dtype)
  cached: list[CachedExample] = []
  for example in examples:
    row_key = str(example.get("row_id", example["id"]))
    base_hidden = hidden_by_row[row_key]
    if by_row[row_key]["ids"][:len(example["ids"])] != example["ids"]:
      raise ValueError(f"{example['id']}: cached row prefix is not a prefix of this example")
    hidden = base_hidden[:, :len(example["ids"]), :].copy()
    cached.append(CachedExample(example=example, hidden=hidden))
    token_counts.append(int(hidden.shape[1]))
  total_bytes = sum(int(row.hidden.nbytes) for row in cached)
  return cached, {
    "examples": len(cached),
    "prefix_cache_entries": len(hidden_by_row),
    "prefix_cache_bytes": sum(int(row.nbytes) for row in hidden_by_row.values()),
    "dtype": cache_dtype,
    "hidden_shape_tail": list(cached[0].hidden.shape[2:]) if cached else [],
    "total_hidden_bytes": total_bytes,
    "avg_tokens": None if not token_counts else float(sum(token_counts) / len(token_counts)),
    "max_tokens": None if not token_counts else max(token_counts),
  }

def parity_check(model:Any, examples:list[dict[str, Any]], suffix_start:int, cache_dtype:str, limit:int, tol:float,
                 device:str) -> dict[str, Any]:
  if limit <= 0: return {"status": "skipped", "examples": 0}
  rows = []
  max_abs = 0.0
  mismatches = 0
  with Tensor.train(False):
    for example in examples[:limit]:
      full = _plain_logits(model, Tensor([example["ids"]], dtype="int32"), 0)[:, -1, :].realize().numpy()
      hidden = prefix_hidden_numpy(model, example["ids"], suffix_start, cache_dtype)
      suffix = _last_logits_from_hidden(model, hidden, suffix_start, device).realize().numpy()
      row_max_abs = float(np.max(np.abs(full - suffix)))
      full_argmax, suffix_argmax = int(np.argmax(full, axis=1)[0]), int(np.argmax(suffix, axis=1)[0])
      mismatch = full_argmax != suffix_argmax
      max_abs = max(max_abs, row_max_abs)
      mismatches += int(mismatch)
      rows.append({
        "id": example["id"], "input_tokens": example["input_tokens"], "target": example["target"],
        "max_abs": row_max_abs, "full_argmax": full_argmax, "suffix_argmax": suffix_argmax,
        "argmax_match": not mismatch,
      })
  status = "pass" if max_abs <= tol and mismatches == 0 else "fail"
  return {"status": status, "examples": len(rows), "tol": tol, "max_abs": max_abs, "argmax_mismatches": mismatches, "rows": rows}

def train_suffix_adapter(args:argparse.Namespace) -> tuple[dict[str, Any], Any]:
  configure_env(args)
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer

  Tensor.manual_seed(args.seed)
  rows = load_sft_rows(args.input)
  if args.max_rows > 0: rows = rows[:args.max_rows]
  train_rows, eval_rows, eval_source_ids = split_adapter_rows(rows, eval_every=args.eval_every)

  model, kv = Transformer.from_gguf(pathlib.Path(args.model).expanduser(), args.max_context)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  suffix_start = suffix_start_from_targets(model, args.targets)

  train_examples = _build_examples(train_rows, tok, args.prompt_format, args.max_context, args.target_positions, args.append_eos)
  eval_examples = _build_examples(eval_rows, tok, args.prompt_format, args.max_context, args.target_positions, args.append_eos)
  parity = parity_check(model, eval_examples, suffix_start, args.cache_dtype, args.parity_limit, args.parity_tol, args.device)
  if parity["status"] == "fail" and not args.allow_failed_parity:
    raise RuntimeError(f"suffix parity failed: max_abs={parity['max_abs']} mismatches={parity['argmax_mismatches']}")

  adapters = install_lora(model, args.targets, rank=args.rank, alpha=args.alpha, seed=args.seed)
  installed_targets = [adapter.target for adapter in adapters]
  opt = Adam(adapter_parameters(adapters), lr=args.lr, fused=False)

  st = time.perf_counter()
  train_cache, train_cache_summary = build_suffix_cache(model, train_examples, suffix_start, args.cache_dtype)
  eval_cache, eval_cache_summary = build_suffix_cache(model, eval_examples, suffix_start, args.cache_dtype)
  before_arrays = {k:v.copy() for k,v in _adapter_arrays(adapters).items()}
  initial = {
    "train": _eval_loss_suffix(model, train_cache, suffix_start, args.device, args.eval_limit),
    "eval": _eval_loss_suffix(model, eval_cache, suffix_start, args.device, args.eval_limit),
  }
  history = []
  rng = np.random.default_rng(args.seed)
  with Tensor.train():
    for step in range(args.steps):
      cached = train_cache[int(rng.integers(0, len(train_cache)))]
      loss = _loss_one_suffix(model, cached, suffix_start, args.device)
      opt.zero_grad()
      loss.backward()
      opt.step()
      if step == 0 or (step + 1) % max(1, args.steps // 4) == 0:
        history.append({"step": step + 1, "example": cached.example["id"], "loss": float(loss.numpy())})
  final = {
    "train": _eval_loss_suffix(model, train_cache, suffix_start, args.device, args.eval_limit),
    "eval": _eval_loss_suffix(model, eval_cache, suffix_start, args.device, args.eval_limit),
  }
  after_arrays = _adapter_arrays(adapters)
  adapter_delta_l2 = _array_delta_l2(before_arrays, after_arrays)
  train_loss_delta = initial["train"]["loss"] - final["train"]["loss"]
  eval_loss_delta = initial["eval"]["loss"] - final["eval"]["loss"]
  status = "pass" if parity["status"] == "pass" and adapter_delta_l2 > 0 and train_loss_delta > args.min_train_loss_delta else "fail"
  summary = {
    "kind": "llm_adapter_suffix_train_summary",
    "status": status,
    "adapter_kind": "suffix_lora",
    "base_model": str(args.model),
    "mode": args.mode,
    "policy": str(args.policy) if args.policy else None,
    "storage": args.storage,
    "prompt_format": args.prompt_format,
    "input": str(args.input),
    "targets": args.targets,
    "suffix_start_block": suffix_start,
    "suffix_blocks": len(model.blk) - suffix_start,
    "installed_targets": installed_targets,
    "installed_adapters": len(adapters),
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
    "cache": {"train": train_cache_summary, "eval": eval_cache_summary},
    "parity": parity,
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
    "note": "Suffix-cache LoRA gate; caches frozen prefix hidden states and trains only the selected lastN_ffn suffix.",
  }
  return summary, adapters

def summary_markdown(summary:dict[str, Any]) -> str:
  lines = [
    "# LLM Suffix Adapter Train",
    "",
    "This gate caches hidden states at the selected suffix boundary, then trains",
    "LoRA tensors only inside the suffix. It is a diagnostic path for internal",
    "adapters when full-model adapter backprop is too slow or too memory-heavy.",
    "",
    "## Summary",
    "",
    f"- status: `{summary['status']}`",
    f"- adapter: `{summary['adapter_kind']}` rank `{summary['rank']}` alpha `{summary['alpha']}`",
    f"- model: `{summary['base_model']}`",
    f"- targets: `{', '.join(summary['targets'])}`",
    f"- suffix start block: `{summary['suffix_start_block']}` (`{summary['suffix_blocks']}` blocks)",
    f"- parity: `{summary['parity']['status']}` max_abs `{summary['parity'].get('max_abs')}`",
    f"- cache bytes: train `{summary['cache']['train']['total_hidden_bytes']}`, eval `{summary['cache']['eval']['total_hidden_bytes']}`",
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
  parser = argparse.ArgumentParser(description="Train internal LoRA adapters with a cached-prefix suffix trainer")
  parser.add_argument("--model", type=pathlib.Path, required=True)
  parser.add_argument("--policy", type=pathlib.Path)
  parser.add_argument("--input", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--mode", choices=("generated", "explicit", "baseline"), default="baseline")
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--storage", default="shared")
  parser.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
  parser.add_argument("--targets", nargs="+", default=["last1_ffn"])
  parser.add_argument("--rank", type=int, default=4)
  parser.add_argument("--alpha", type=float, default=8.0)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--steps", type=int, default=64)
  parser.add_argument("--seed", type=int, default=20260613)
  parser.add_argument("--max-rows", type=int, default=20)
  parser.add_argument("--eval-every", type=int, default=5)
  parser.add_argument("--eval-limit", type=int, default=8)
  parser.add_argument("--target-positions", choices=("all", "last"), default="all")
  parser.add_argument("--append-eos", action="store_true", help="train one EOS target after each completion so rollout can stop")
  parser.add_argument("--max-context", type=int, default=4096)
  parser.add_argument("--cache-dtype", choices=("float32", "float16"), default="float32")
  parser.add_argument("--parity-limit", type=int, default=4)
  parser.add_argument("--parity-tol", type=float, default=1e-2)
  parser.add_argument("--allow-failed-parity", action="store_true")
  parser.add_argument("--min-train-loss-delta", type=float, default=0.0)
  args = parser.parse_args()

  summary, adapters = train_suffix_adapter(args)
  write_artifact(args.out, summary, adapters)
  print(summary_markdown(summary))
  return 0 if summary["status"] == "pass" else 1

if __name__ == "__main__":
  raise SystemExit(main())
