"""Research-only zero-copy view of llama tile-major Q8_1 records."""
from __future__ import annotations
from dataclasses import dataclass
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp
from tinygrad.codegen.opt.packed_weight import Q8ActivationRecordTransform


@dataclass(frozen=True)
class TileMajorQ8_1RecordSpec:
  M: int = 512
  K: int = 4096
  block_bytes: int = 144
  tail_bytes: int = 128 * 144

  @property
  def blocks(self) -> int: return self.M * self.K // 128
  @property
  def payload_bytes(self) -> int: return self.blocks * self.block_bytes
  @property
  def record_bytes(self) -> int: return self.payload_bytes + self.tail_bytes
  @property
  def record_u32(self) -> int: return self.record_bytes // 4

  def validate(self, record: Tensor) -> None:
    if record.dtype != dtypes.uint32: raise ValueError("record must be uint32")
    if record.numel() < self.record_u32: raise ValueError(f"record too small: {record.numel()} < {self.record_u32}")
    if self.payload_bytes % 4 or self.record_bytes % 16: raise ValueError("record alignment invariant failed")

  def views(self, record: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return logical qs[M,K], d[M,K/32], s[M,K/32] views without materializing."""
    self.validate(record)
    raw = record.bitcast(dtypes.uint8)[:self.payload_bytes]
    blocks = raw.reshape(self.blocks, self.block_bytes)
    d = blocks[:, 0:8].bitcast(dtypes.float16).reshape(self.blocks, 4)
    s = blocks[:, 8:16].bitcast(dtypes.float16).reshape(self.blocks, 4)
    qs = blocks[:, 16:144].reshape(self.M, self.K)
    return qs, d.reshape(self.M, self.K // 32), s.reshape(self.M, self.K // 32)


DEFAULT_TILE_MAJOR_Q8_1 = TileMajorQ8_1RecordSpec()

@dataclass(frozen=True)
class TileMajorActivationCarrierSpec:
  transform: object
  logical_shape: tuple[int,int] = (512,4096)
  record_bytes: int = 2377728
  logical_dtype: object = dtypes.int8
  abi: str = "q8_1.logical_mk_to_s8.v1"
  def __post_init__(self):
    if self.logical_shape != self.transform.logical_shape or self.record_bytes != self.transform.packed_bytes:
      raise ValueError("tile-major carrier shape/record size mismatch")
    if self.abi != "q8_1.logical_mk_to_s8.v1": raise ValueError("unsupported tile-major carrier ABI")

def tile_major_q8_carrier(record: Tensor, spec: TileMajorActivationCarrierSpec) -> Tensor:
  return Tensor(UOp.packed_activation_carrier(record.uop, spec))


@dataclass(frozen=True)
class TileMajorQ8FragmentProvider:
  """Logical Q8 fragment address provider for the existing Q4 accumulator ABI."""
  spec: TileMajorQ8_1RecordSpec = DEFAULT_TILE_MAJOR_Q8_1
  abi_identity: str = "q8_1.logical_mk_to_s8.v1"

  def addresses(self, m: int, k: int) -> dict:
    if not (0 <= m < self.spec.M and 0 <= k < self.spec.K): raise IndexError((m, k))
    block, lane = divmod(k, 128)
    block = block * self.spec.M + m
    base = block * self.spec.block_bytes
    return {"block": block, "lane": lane, "q_byte": base + 16 + lane,
            "d_byte": base + ((lane//32)*4), "s_byte": base + 8 + ((lane//32)*4)}

  def validate(self) -> None:
    probes = [(0, 0), (0, 31), (0, 32), (self.spec.M - 1, self.spec.K - 1),
              (127, 4095), (128, 0)]
    for m, k in probes:
      a = self.addresses(m, k)
      if not (4 <= a["q_byte"] < self.spec.payload_bytes): raise ValueError("q offset out of bounds")
      if a["d_byte"] % 2 or a["s_byte"] % 2: raise ValueError("metadata alignment failed")
    if self.addresses(0, 127)["block"] != 0 or self.addresses(0, 128)["block"] != self.spec.M:
      raise ValueError("logical block mapping failed")


@dataclass(frozen=True)
class TileMajorQ8ActivationRecordTransform(Q8ActivationRecordTransform):
  """Q8 transform implementing the compiler fragment-provider protocol."""
  @property
  def abi(self): return "q8_1.logical_mk_to_s8.v1"
  @property
  def packed_bytes(self): return self.rows * self.k // 128 * 144 + 128 * 144
  @property
  def storage_units(self): return self.packed_bytes // 4
  @property
  def values_bytes(self): return self.rows * self.k // 32 * 36
  def value(self, source, row, k):
    self._check_coords(row, k); block, lane = k//128, k%128; block = block*self.rows + row
    byte = block*144 + 16 + lane; word = self._load(source, byte//4)
    return word.rshift((byte%4)*8).bitwise_and(0xff).cast(dtypes.uint8).bitcast(dtypes.char)
  def metadata(self, source, row, k_base):
    self._check_coords(row, k_base); chunk, subgroup = k_base//128, k_base%128; block = chunk*self.rows+row; subgroup //= 32
    base = block*144 + subgroup*4; word = self._load(source, base//4)
    d = word.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float)
    s = word.rshift(16).bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float)
    return d, s, (row*self.k+k_base)//32
  def to_json(self):
    return {"abi": self.abi, "rows": self.rows, "k": self.k, "group_elems": 32,
            "block_bytes": 144, "tail_bytes": 128*144, "packed_bytes": self.packed_bytes}


def encode_decode_contract() -> dict:
  """Small deterministic CPU-side contract summary; no GPU/materialization."""
  spec = DEFAULT_TILE_MAJOR_Q8_1
  return {"spec": {"M": spec.M, "K": spec.K, "blocks": spec.blocks,
                    "block_bytes": spec.block_bytes, "payload_bytes": spec.payload_bytes,
                    "tail_bytes": spec.tail_bytes, "record_bytes": spec.record_bytes},
          "fields": {"d": [0, 2, "float16"], "s": [2, 4, "float16"], "qs": [4, 36, "int8"]},
          "zero_copy": True, "tail": "opaque llama allocation tail; excluded from logical views"}
