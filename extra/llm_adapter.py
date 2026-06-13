#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
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

class LoRALinear:
  def __init__(self, base:Any, target:str, rank:int, alpha:float, *, seed:int=0, device:str|None=None,
               lora_a:np.ndarray|None=None, lora_b:np.ndarray|None=None):
    if rank < 1: raise ValueError("LoRA rank must be >= 1")
    if not hasattr(base, "weight"): raise ValueError(f"{target}: base module has no weight")
    if len(base.weight.shape) != 2: raise ValueError(f"{target}: expected 2D base weight, got {base.weight.shape}")
    self.base, self.target, self.rank, self.alpha = base, target, rank, float(alpha)
    self.out_features, self.in_features = int(base.weight.shape[0]), int(base.weight.shape[1])
    rng = np.random.default_rng(seed)
    if lora_a is None: lora_a = (rng.standard_normal((self.in_features, rank)) * 0.01).astype(np.float32)
    if lora_b is None: lora_b = np.zeros((rank, self.out_features), dtype=np.float32)
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
    base_out = self.base(x).detach()
    x_detached = x.detach().cast(dtypes.float32)
    delta = (x_detached @ self.lora_a) @ self.lora_b
    return base_out + delta * self.scale

  def state_arrays(self) -> dict[str, np.ndarray]:
    return {f"{self.target}.lora_a": self.lora_a.numpy(), f"{self.target}.lora_b": self.lora_b.numpy()}

  def metadata(self) -> dict[str, Any]:
    return {
      "target": self.target, "rank": self.rank, "alpha": self.alpha,
      "in_features": self.in_features, "out_features": self.out_features,
    }

def install_lora(model:Any, targets:list[str], *, rank:int, alpha:float, seed:int=0, device:str|None=None) -> list[LoRALinear]:
  adapters: list[LoRALinear] = []
  for idx, target in enumerate(targets):
    base = _module_at(model, target)
    adapter = LoRALinear(base, target, rank, alpha, seed=seed + idx, device=device)
    _set_module_at(model, target, adapter)
    adapters.append(adapter)
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
    adapter = LoRALinear(base, target, int(entry["rank"]), float(entry["alpha"]), device=device, lora_a=lora_a, lora_b=lora_b)
    if int(entry.get("in_features", adapter.in_features)) != adapter.in_features:
      raise ValueError(f"{target}: adapter in_features mismatch")
    if int(entry.get("out_features", adapter.out_features)) != adapter.out_features:
      raise ValueError(f"{target}: adapter out_features mismatch")
    _set_module_at(model, target, adapter)
    adapters.append(adapter)
  if not adapters: raise ValueError(f"{config_path}: no adapter targets")
  return adapters
