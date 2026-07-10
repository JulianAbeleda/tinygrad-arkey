from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LinearRoleShape:
  role: str
  phase: str
  quant: str
  M: int
  N: int
  K: int
  tensor_patterns: tuple[str, ...] = ()

  @property
  def mnk(self) -> tuple[int, int, int]:
    return (self.M, self.N, self.K)

  def to_json(self) -> dict[str, Any]:
    return {"role": self.role, "phase": self.phase, "quant": self.quant, "M": self.M, "N": self.N, "K": self.K,
            "tensor_patterns": list(self.tensor_patterns)}


@dataclass(frozen=True)
class AttentionShape:
  B: int
  Hq: int
  Hkv: int
  Hd: int

  def to_json(self) -> dict[str, int]:
    return {"B": self.B, "Hq": self.Hq, "Hkv": self.Hkv, "Hd": self.Hd}


@dataclass(frozen=True)
class ModelProfile:
  id: str
  family: str
  size_label: str
  quant: str
  device_profile: str
  roles: tuple[LinearRoleShape, ...]
  attention: AttentionShape

  def role_shape(self, role: str, *, phase: str = "prefill") -> LinearRoleShape:
    for shape in self.roles:
      if shape.role == role and shape.phase == phase: return shape
    raise KeyError(f"profile {self.id!r} has no {phase} role {role!r}")

  def to_json(self) -> dict[str, Any]:
    return {"id": self.id, "family": self.family, "size_label": self.size_label, "quant": self.quant,
            "device_profile": self.device_profile, "roles": [role.to_json() for role in self.roles],
            "attention": self.attention.to_json()}


def _qwen3_q4_prefill_roles(*, hidden_size: int, intermediate_size: int,
                            tensor_suffix: str) -> tuple[LinearRoleShape, ...]:
  kv_size = 1024
  return (
    LinearRoleShape("attn_kv", "prefill", "Q4_K_M", 512, kv_size, hidden_size,
                    (f"blk.*.attn_k.{tensor_suffix}", f"blk.*.attn_v.{tensor_suffix}")),
    LinearRoleShape("attn_qo", "prefill", "Q4_K_M", 512, hidden_size, hidden_size,
                    (f"blk.*.attn_q.{tensor_suffix}", f"blk.*.attn_output.{tensor_suffix}")),
    LinearRoleShape("ffn_down", "prefill", "Q4_K_M", 512, hidden_size, intermediate_size,
                    (f"blk.*.ffn_down.{tensor_suffix}",)),
    LinearRoleShape("ffn_gate_up", "prefill", "Q4_K_M", 512, intermediate_size, hidden_size,
                    (f"blk.*.ffn_gate.{tensor_suffix}", f"blk.*.ffn_up.{tensor_suffix}")),
  )


QWEN3_8B_Q4_K_M_GFX1100 = ModelProfile(
  id="qwen3_8b_q4k_m_gfx1100",
  family="qwen3",
  size_label="8B",
  quant="Q4_K_M",
  device_profile="gfx1100",
  roles=_qwen3_q4_prefill_roles(hidden_size=4096, intermediate_size=12288, tensor_suffix="weight"),
  attention=AttentionShape(B=1, Hq=32, Hkv=8, Hd=128),
)

QWEN3_14B_Q4_K_M_GFX1100 = ModelProfile(
  id="qwen3_14b_q4k_m_gfx1100",
  family="qwen3",
  size_label="14B",
  quant="Q4_K_M",
  device_profile="gfx1100",
  roles=_qwen3_q4_prefill_roles(hidden_size=5120, intermediate_size=17408, tensor_suffix="weight"),
  attention=AttentionShape(B=1, Hq=40, Hkv=8, Hd=128),
)

MODEL_PROFILES: tuple[ModelProfile, ...] = (
  QWEN3_8B_Q4_K_M_GFX1100,
  QWEN3_14B_Q4_K_M_GFX1100,
)

_PROFILE_ALIASES = {
  "qwen3_8b_q4_k_m_gfx1100": "qwen3_8b_q4k_m_gfx1100",
  "qwen3_14b_q4_k_m_gfx1100": "qwen3_14b_q4k_m_gfx1100",
}
_PROFILES_BY_ID = {profile.id: profile for profile in MODEL_PROFILES}
_PROFILES_BY_CONFIG = {
  (profile.family, profile.quant, profile.device_profile, profile.attention.Hq * profile.attention.Hd,
   profile.role_shape("ffn_gate_up").N, profile.attention.Hq, profile.attention.Hkv, profile.attention.Hd): profile
  for profile in MODEL_PROFILES
}


def profile_by_id(profile_id: str) -> ModelProfile:
  return _PROFILES_BY_ID[_PROFILE_ALIASES.get(profile_id, profile_id)]


def qwen3_8b_q4k_m_gfx1100_profile() -> ModelProfile:
  return QWEN3_8B_Q4_K_M_GFX1100


def qwen3_14b_q4k_m_gfx1100_profile() -> ModelProfile:
  return QWEN3_14B_Q4_K_M_GFX1100


def _config_get(config: Any, key: str, default: Any = None) -> Any:
  if isinstance(config, dict): return config.get(key, default)
  return getattr(config, key, default)


def profile_from_transformer_config(config: Any, *, quant: str, device_profile: str) -> ModelProfile:
  family = str(_config_get(config, "model_type", _config_get(config, "family", "qwen3"))).lower()
  if family == "qwen2": family = "qwen3"
  hidden_size = int(_config_get(config, "hidden_size", _config_get(config, "dim")))
  intermediate_size = int(_config_get(config, "intermediate_size", _config_get(config, "hidden_dim")))
  num_attention_heads = int(_config_get(config, "num_attention_heads", _config_get(config, "n_heads")))
  num_key_value_heads = int(_config_get(config, "num_key_value_heads", _config_get(config, "n_kv_heads")))
  head_dim = int(_config_get(config, "head_dim", hidden_size // num_attention_heads))
  key = (family, quant, device_profile, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads,
         head_dim)
  try:
    return _PROFILES_BY_CONFIG[key]
  except KeyError as exc:
    raise KeyError(f"no model profile for {key!r}") from exc


def prefill_role_shapes(profile: ModelProfile) -> tuple[LinearRoleShape, ...]:
  return tuple(role for role in profile.roles if role.phase == "prefill")


def attention_shape(profile: ModelProfile) -> AttentionShape:
  return profile.attention
