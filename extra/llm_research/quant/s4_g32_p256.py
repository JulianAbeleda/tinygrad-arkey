"""Research-only S4_G32_P256 format and sidecar substrate (Gate 1).

The wire format is deliberately boring: eight FP16 scales (16 bytes) followed by 128 bytes
of low-nibble-first signed four-bit codes, for 144 bytes per 256 weights.
This module has no device or route dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json, math, struct
from typing import Iterable, Sequence

FORMAT_ID = "S4_G32_P256"
BLOCK_WEIGHTS, GROUP_WEIGHTS, BLOCK_BYTES, ALIGNMENT = 256, 32, 144, 16

def pack_block(weights: Sequence[float]) -> bytes:
  if len(weights) != BLOCK_WEIGHTS: raise ValueError("S4 block must contain 256 weights")
  if not all(math.isfinite(float(x)) for x in weights): raise ValueError("S4 weights must be finite")
  scales, codes = [], bytearray(128)
  for g in range(8):
    ws = [float(x) for x in weights[g*32:(g+1)*32]]
    scale = max((abs(x) for x in ws), default=0.0) / 7.0
    # Quantize against the exact FP16 value that is stored on the wire.
    scale = struct.unpack("<e", struct.pack("<e", scale))[0]
    scales.append(scale)
    for j, x in enumerate(ws):
      q = 0 if scale == 0 else max(-8, min(7, int(round(x / scale))))
      # Signed two's-complement nibble; -8 is representable, +8 is not.
      q &= 0xF
      i = g*16 + j//2
      codes[i] = (codes[i] & 0xF0) | q if j % 2 == 0 else (codes[i] & 0x0F) | (q << 4)
  return struct.pack("<8e", *scales) + bytes(codes)

def decode_block(block: bytes) -> list[float]:
  if len(block) != BLOCK_BYTES: raise ValueError("invalid S4 block size")
  scales = struct.unpack("<8e", block[:16]); codes = block[16:]
  out = []
  for g, scale in enumerate(scales):
    for j in range(32):
      n = codes[g*16+j//2] >> (4*(j%2)) & 0xF
      out.append(float(scale) * (n - 16 if n >= 8 else n))
  return out

def reference_decode_block(block: bytes) -> list[float]:
  """Independent scalar oracle: separate indexing and signed-nibble path."""
  if len(block) != BLOCK_BYTES: raise ValueError("invalid S4 block size")
  out = []
  for i in range(BLOCK_WEIGHTS):
    group, within = divmod(i, GROUP_WEIGHTS)
    scale = struct.unpack_from("<e", block, group * 2)[0]
    byte = block[16 + group * 16 + (within >> 1)]
    nibble = (byte >> 4) if (within & 1) else (byte & 15)
    signed = nibble - 16 if nibble > 7 else nibble
    out.append(scale * signed)
  return out

def pack_tensor(weights: Iterable[float], *, cols: int | None = None) -> bytes:
  vals = [float(x) for x in weights]
  if cols is not None and (cols <= 0 or cols % BLOCK_WEIGHTS or len(vals) % cols):
    raise ValueError("tensor rows must contain a whole number of 256-weight blocks")
  if len(vals) % BLOCK_WEIGHTS: raise ValueError("tensor length must be a multiple of 256")
  return b"".join(pack_block(vals[i:i+256]) for i in range(0, len(vals), 256))

def decode_tensor(payload: bytes) -> list[float]:
  if len(payload) % BLOCK_BYTES: raise ValueError("payload is not block aligned")
  return [x for i in range(0, len(payload), BLOCK_BYTES) for x in decode_block(payload[i:i+BLOCK_BYTES])]

def block_dot(block: bytes, activations: Sequence[float]) -> float:
  if len(activations) != BLOCK_WEIGHTS: raise ValueError("activation block must contain 256 values")
  return sum(w*a for w,a in zip(reference_decode_block(block), activations))

def sha256(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def _align(n: int, alignment: int = ALIGNMENT) -> int: return (n + alignment - 1) // alignment * alignment

@dataclass(frozen=True)
class TensorEntry:
  name: str; source_type: str; role: str; rows: int; cols: int
  byte_offset: int; payload_bytes: int; padded_bytes: int; payload_sha256: str
  tied_weight_owner: str | None = None

@dataclass(frozen=True)
class SidecarManifest:
  schema: str
  source_model_sha256: str
  source_tensor_table_sha256: str
  converter_config_sha256: str
  format_id: str
  tensors: tuple[TensorEntry, ...]
  calibration_manifest_hash: str | None = None
  promotable: bool = True

  def to_bytes(self) -> bytes:
    d = asdict(self); d["tensors"] = [asdict(x) for x in self.tensors]
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
  @classmethod
  def from_bytes(cls, data: bytes) -> "SidecarManifest":
    d = json.loads(data); d["tensors"] = tuple(TensorEntry(**x) for x in d["tensors"])
    return cls(**d)

def build_one_tensor_sidecar(name: str, weights: Iterable[float], *, rows: int, cols: int,
  source_model_sha256: str, source_tensor_table_sha256: str, converter_config_sha256: str,
  source_type: str = "higher_precision", role: str = "dense", tied_weight_owner: str | None = None,
  posthoc_q4: bool = False) -> tuple[SidecarManifest, bytes]:
  # Materialize once while preserving generator support.
  vals = list(weights)
  if len(vals) != rows*cols: raise ValueError("rows*cols does not match weights")
  payload = pack_tensor(vals, cols=cols)
  if posthoc_q4 and source_type == "higher_precision": source_type = "posthoc_q4_dequant"
  entry = TensorEntry(name, source_type, role, rows, cols, 0, len(payload), _align(len(payload)), sha256(payload), tied_weight_owner)
  manifest = SidecarManifest("s4-sidecar/v1", source_model_sha256, source_tensor_table_sha256,
    converter_config_sha256, FORMAT_ID, (entry,), promotable=not posthoc_q4)
  return manifest, payload

def validate_sidecar(manifest: SidecarManifest, payload: bytes) -> None:
  if manifest.schema != "s4-sidecar/v1" or manifest.format_id != FORMAT_ID: raise ValueError("unsupported schema/format")
  for value in (manifest.source_model_sha256, manifest.source_tensor_table_sha256, manifest.converter_config_sha256):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value): raise ValueError("invalid source/config hash")
  if manifest.calibration_manifest_hash is not None:
    value = manifest.calibration_manifest_hash
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value): raise ValueError("invalid calibration hash")
  names, ranges = set(), []
  for t in manifest.tensors:
    if t.name in names or not t.name: raise ValueError("duplicate/empty tensor name")
    names.add(t.name)
    if t.rows <= 0 or t.cols <= 0 or t.cols % BLOCK_WEIGHTS or t.payload_bytes != t.rows*t.cols//BLOCK_WEIGHTS*BLOCK_BYTES: raise ValueError("invalid tensor geometry")
    if len(t.payload_sha256) != 64 or any(c not in "0123456789abcdef" for c in t.payload_sha256): raise ValueError("invalid payload hash")
    if t.byte_offset % ALIGNMENT or t.payload_bytes % BLOCK_BYTES: raise ValueError("unaligned tensor entry")
    if t.byte_offset + t.payload_bytes > len(payload): raise ValueError("truncated payload")
    if sha256(payload[t.byte_offset:t.byte_offset+t.payload_bytes]) != t.payload_sha256: raise ValueError("payload hash mismatch")
    if t.padded_bytes < t.payload_bytes or t.padded_bytes % ALIGNMENT: raise ValueError("invalid padding")
    ranges.append((t.byte_offset, t.byte_offset+t.padded_bytes))
  ordered = sorted(ranges)
  for i, (start, end) in enumerate(ordered):
    if i and start < ordered[i-1][1]: raise ValueError("overlapping tensor entries")
