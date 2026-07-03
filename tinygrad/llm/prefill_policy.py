from __future__ import annotations

_PREFILL_V2_VALIDATED_UBATCH = (512,)

def prefill_v2_auto_decision(total_vram_bytes:int|None, est_fp16_bytes:int, q4_bytes:int, kv_bytes:int,
                             min_total_gb:float=23.0, margin_gb:float=3.0) -> tuple[bool, str]:
  # Enable PREFILL_V2 only when the full Q4 + fp16-covered weights + KV footprint fits with runtime headroom.
  if total_vram_bytes is None: return (False, "VRAM unknown (rocm-smi unavailable) -> conservative OFF")
  need = q4_bytes + est_fp16_bytes + kv_bytes
  tot_gb, need_gb = total_vram_bytes/1e9, need/1e9
  if total_vram_bytes < min_total_gb*1e9:
    return (False, f"total {tot_gb:.1f}GB < {min_total_gb:.0f}GB floor -> OFF (PREFILL_V2 +fp16 would risk OOM)")
  if total_vram_bytes < need + margin_gb*1e9:
    return (False, f"need {need_gb:.1f}GB + {margin_gb:.0f}GB margin > {tot_gb:.1f}GB total -> OFF")
  return (True, f"need {need_gb:.1f}GB + {margin_gb:.0f}GB margin <= {tot_gb:.1f}GB total -> ON")

def prefill_concrete_kv_auto_decision(server_profile:bool, prefill_v2_on:bool) -> tuple[bool, str]:
  # Precompile pays off only across repeated/long generation, so auto keys off the explicit server profile.
  if not prefill_v2_on: return (False, "PREFILL_V2 off -> concrete-KV moot, OFF")
  if server_profile: return (True, "server profile + PREFILL_V2 on -> precompile concrete jits, ON")
  return (False, "no server profile (one-shot assumed) -> OFF; set PREFILL_SERVER_PROFILE=1 or PREFILL_CONCRETE_KV=1")

def prefill_v2_validate_ubatch(ubatch:int) -> None:
  if ubatch not in _PREFILL_V2_VALIDATED_UBATCH:
    raise ValueError(f"PREFILL_V2 only validates PREFILL_UBATCH in {_PREFILL_V2_VALIDATED_UBATCH} (got {ubatch}); "
                     f"the warmstart TC schedule is shape-specific. Re-measure per-shape opts for {ubatch} first "
                     f"and add it to _PREFILL_V2_VALIDATED_UBATCH.")

def prefill_v2_realize_bytes(shapes:list[tuple[int,int]]) -> int:
  return sum(o * i for o, i in shapes) * 2
