import random, struct
from tinygrad.codegen.opt.q6_mma_tile import Q6KMMATile

def test_q6_mma_tile_stages_every_code_and_metadata_exactly():
  rng=random.Random(7); tile=Q6KMMATile()
  blocks=bytearray(rng.randbytes(tile.rows*tile.block_bytes))
  for row in range(tile.rows): struct.pack_into("<e",blocks,row*tile.block_bytes+208,(row-31)/16)
  staged=tile.stage_cpu(bytes(blocks))
  assert len(staged)==64*76
  for row in range(64):
    base=row*210
    for txi in range(32):
      ql=int.from_bytes(blocks[base+4*txi:base+4*txi+4],"little")
      qhi=(txi//16)*8+txi%8; qh=int.from_bytes(blocks[base+128+4*qhi:base+132+4*qhi],"little"); shift=(txi&8)>>2
      kq0=2*txi-txi%16
      for word,q in ((kq0,(ql&0x0f0f0f0f)|(((qh>>shift)<<4)&0x30303030)),
                     (kq0+16,((ql>>4)&0x0f0f0f0f)|((qh>>shift)&0x30303030))):
        want=sum((((q>>(8*j))&255)-32&255)<<(8*j) for j in range(4))
        assert staged[row*76+word]==want
    assert staged[row*76+64]==struct.unpack("<I",struct.pack("<f",(row-31)/16))[0]
    for word in range(4): assert staged[row*76+65+word]==int.from_bytes(blocks[base+192+4*word:base+196+4*word],"little")

def test_q6_mma_tile_cuda_lowering_uses_packed_lane_map():
  src=Q6KMMATile().emit_cuda()
  assert src.count("__vsubss4") == 2 and "tile[4864]" in src and "row*76" in src
