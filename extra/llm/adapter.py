#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, re
from typing import Any

import numpy as np

from tinygrad import Tensor, dtypes

def _module_at(root:Any, path:str) -> Any:
  obj = root
  for part in path.split("."):
    obj = obj[int(part)] if isinstance(obj, list) and part.isdigit() else getattr(obj, part)
  return obj

def _set_module_at(root:Any, path:str, value:Any) -> None:
  if "." in path:
    parent_path, attr = path.rsplit(".", 1)
    parent = _module_at(root, parent_path)
  else:
    parent, attr = root, path
  if isinstance(parent, list) and attr.isdigit(): parent[int(attr)] = value
  else: setattr(parent, attr, value)

def _dense_ffn_targets(model:Any, block_idxs:list[int]) -> list[str]:
  paths: list[str] = []
  for idx in block_idxs:
    block = model.blk[idx]
    missing = [name for name in ("ffn_gate", "ffn_up", "ffn_down") if not hasattr(block, name)]
    if missing: raise ValueError(f"blk.{idx}: dense FFN target group missing {missing}")
    paths += [f"blk.{idx}.ffn_gate", f"blk.{idx}.ffn_up", f"blk.{idx}.ffn_down"]
  return paths

def expand_lora_targets(model:Any, targets:list[str]) -> list[str]:
  expanded: list[str] = []
  for target in targets:
    if target == "output":
      expanded.append("output")
    elif m := re.fullmatch(r"last(\d+)_ffn", target):
      if not hasattr(model, "blk") or not isinstance(model.blk, list): raise ValueError(f"{target}: model has no block list")
      count = int(m.group(1))
      if count < 1: raise ValueError(f"{target}: block count must be >= 1")
      if count > len(model.blk): raise ValueError(f"{target}: requested {count} blocks, model has {len(model.blk)}")
      expanded += _dense_ffn_targets(model, list(range(len(model.blk) - count, len(model.blk))))
    elif "." in target:
      expanded.append(target)
    else:
      raise ValueError(f"unknown LoRA target or target group {target!r}")
  deduped = list(dict.fromkeys(expanded))
  if not deduped: raise ValueError("LoRA target expansion installed zero modules")
  return deduped

class LoRALinear:
  def __init__(self, base:Any, target:str, rank:int, alpha:float, *, seed:int=0, device:str|None=None,
               lora_a:np.ndarray|None=None, lora_b:np.ndarray|None=None, detach_base:bool=True):
    if rank < 1: raise ValueError("LoRA rank must be >= 1")
    if not hasattr(base, "weight"): raise ValueError(f"{target}: base module has no weight")
    if len(base.weight.shape) != 2: raise ValueError(f"{target}: expected 2D base weight, got {base.weight.shape}")
    self.base, self.target, self.rank, self.alpha, self.detach_base = base, target, rank, float(alpha), bool(detach_base)
    self.out_features, self.in_features = int(base.weight.shape[0]), int(base.weight.shape[1])
    rng = np.random.default_rng(seed)
    if lora_a is None: lora_a = (rng.standard_normal((self.in_features, rank)) * 0.01).astype(np.float32)
    if lora_b is None:
      lora_b = np.zeros((rank, self.out_features), dtype=np.float32) if self.detach_base else \
        (rng.standard_normal((rank, self.out_features)) * 0.001).astype(np.float32)
    if tuple(lora_a.shape) != (self.in_features, rank):
      raise ValueError(f"{target}: lora_a shape {lora_a.shape} != {(self.in_features, rank)}")
    if tuple(lora_b.shape) != (rank, self.out_features):
      raise ValueError(f"{target}: lora_b shape {lora_b.shape} != {(rank, self.out_features)}")
    self.lora_a = Tensor(lora_a, device=device, dtype=dtypes.float32).is_param_()
    self.lora_b = Tensor(lora_b, device=device, dtype=dtypes.float32).is_param_()

  @property
  def scale(self) -> float:
    return self.alpha / self.rank

  @property
  def weight(self) -> Tensor:
    return self.base.weight

  @property
  def bias(self) -> Tensor|None:
    return getattr(self.base, "bias", None)

  def parameters(self) -> list[Tensor]:
    return [self.lora_a, self.lora_b]

  def __call__(self, x:Tensor) -> Tensor:
    base_out = self.base(x)
    x_lora = x
    if self.detach_base:
      base_out = base_out.detach()
      x_lora = x_lora.detach()
    delta = (x_lora.cast(dtypes.float32) @ self.lora_a) @ self.lora_b
    return base_out + delta * self.scale

  def state_arrays(self) -> dict[str, np.ndarray]:
    return {f"{self.target}.lora_a": self.lora_a.numpy(), f"{self.target}.lora_b": self.lora_b.numpy()}

  def metadata(self) -> dict[str, Any]:
    return {
      "target": self.target, "rank": self.rank, "alpha": self.alpha,
      "in_features": self.in_features, "out_features": self.out_features,
      "detach_base": self.detach_base,
    }

def install_lora(model:Any, targets:list[str], *, rank:int, alpha:float, seed:int=0, device:str|None=None) -> list[LoRALinear]:
  adapters: list[LoRALinear] = []
  expanded = expand_lora_targets(model, targets)
  detach_base = expanded == ["output"]
  for idx, target in enumerate(expanded):
    base = _module_at(model, target)
    adapter = LoRALinear(base, target, rank, alpha, seed=seed + idx, device=device, detach_base=detach_base)
    _set_module_at(model, target, adapter)
    adapters.append(adapter)
  if not adapters: raise ValueError("LoRA install produced zero adapters")
  return adapters

def adapter_parameters(adapters:list[LoRALinear]) -> list[Tensor]:
  return [param for adapter in adapters for param in adapter.parameters()]

def save_adapter(path:pathlib.Path, adapters:list[LoRALinear], *, base_model:str, source:str, seed:int, extra:dict[str, Any]|None=None) -> None:
  path.mkdir(parents=True, exist_ok=True)
  config = {
    "kind": "llm_lora_adapter",
    "version": 1,
    "base_model": base_model,
    "source": source,
    "seed": seed,
    "targets": [adapter.metadata() for adapter in adapters],
    "weights": "adapter.npz",
  }
  if extra: config["extra"] = extra
  arrays: dict[str, np.ndarray] = {}
  for adapter in adapters: arrays.update(adapter.state_arrays())
  (path / "adapter.json").write_text(json.dumps(config, indent=2, sort_keys=True))
  np.savez_compressed(path / "adapter.npz", **arrays)

def load_adapter(model:Any, path:pathlib.Path, *, device:str|None=None) -> list[LoRALinear]:
  config_path = path / "adapter.json"
  config = json.loads(config_path.read_text())
  if config.get("kind") != "llm_lora_adapter": raise ValueError(f"{config_path}: expected kind=llm_lora_adapter")
  weights = np.load(path / config.get("weights", "adapter.npz"))
  adapters: list[LoRALinear] = []
  for entry in config.get("targets", []):
    target = entry["target"]
    lora_a = weights[f"{target}.lora_a"]
    lora_b = weights[f"{target}.lora_b"]
    base = _module_at(model, target)
    adapter = LoRALinear(base, target, int(entry["rank"]), float(entry["alpha"]), device=device, lora_a=lora_a, lora_b=lora_b,
                         detach_base=bool(entry.get("detach_base", target == "output")))
    if int(entry.get("in_features", adapter.in_features)) != adapter.in_features:
      raise ValueError(f"{target}: adapter in_features mismatch")
    if int(entry.get("out_features", adapter.out_features)) != adapter.out_features:
      raise ValueError(f"{target}: adapter out_features mismatch")
    _set_module_at(model, target, adapter)
    adapters.append(adapter)
  if not adapters: raise ValueError(f"{config_path}: no adapter targets")
  return adapters
