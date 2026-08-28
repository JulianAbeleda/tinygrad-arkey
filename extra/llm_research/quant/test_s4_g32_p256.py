import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from s4_g32_p256 import *

def test_fixed_geometry_and_decode():
  w = [(i % 31) - 15 for i in range(256)]
  b = pack_block(w)
  assert len(b) == 144 and len(decode_block(b)) == 256
  assert decode_block(b) == reference_decode_block(b)
  assert decode_block(pack_block([0.0]*256)) == [0.0]*256

def test_determinism_and_dot():
  w = [((i*17)%101-50)/13 for i in range(256)]
  assert pack_block(w) == pack_block(w)
  a = [1.0/(i+1) for i in range(256)]
  assert block_dot(pack_block(w), a) == pytest.approx(sum(x*y for x,y in zip(decode_block(pack_block(w)),a)))

def test_manifest_hash_alignment_corruption_and_nonpromotable():
  m, p = build_one_tensor_sidecar("x", [float(i) for i in range(256)], rows=1, cols=256,
    source_model_sha256="a"*64, source_tensor_table_sha256="b"*64, converter_config_sha256="c"*64, posthoc_q4=True)
  validate_sidecar(m, p)
  assert not m.promotable and m.tensors[0].payload_sha256 == sha256(p)
  with pytest.raises(ValueError): validate_sidecar(m, p[:-1])
  bad = bytearray(p); bad[20] ^= 1
  with pytest.raises(ValueError): validate_sidecar(m, bytes(bad))

def test_shape_and_block_checks():
  with pytest.raises(ValueError): pack_block([0]*255)
  with pytest.raises(ValueError): pack_block([0]*255+[float("nan")])
  with pytest.raises(ValueError): pack_tensor([0]*255)
  with pytest.raises(ValueError): pack_tensor([0]*256, cols=128)
  with pytest.raises(ValueError): build_one_tensor_sidecar("x", [0]*256, rows=2, cols=256,
    source_model_sha256="a"*64, source_tensor_table_sha256="b"*64, converter_config_sha256="c"*64)
