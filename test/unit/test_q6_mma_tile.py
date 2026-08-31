import random, struct
from tinygrad.codegen.opt.q6_mma_tile import Q6KMMATile

def test_q6_mma_tile_stages_every_code_and_metadata_exactly():
  rng=random.Random(7); tile=Q6KMMATile()
  blocks=bytearray(rng.randbytes(tile.rows*tile.block_bytes))
  for row in range(tile.rows): struct.pack_into("<e",blocks,row*tile.block_bytes+208,(row-31)/16)
  staged=tile.stage_cpu(bytes(blocks))
  assert len(staged)==64*77
  for row in range(64):
    base=row*210
    for txi in range(4):
      ql=int.from_bytes(blocks[base+4*txi:base+4*txi+4],"little")
      qh=int.from_bytes(blocks[base+128+4*txi:base+132+4*txi],"little")
      for word,q in ((2*txi,(ql&0x0f0f0f0f)|((qh<<4)&0x30303030)),
                     (2*txi+8,((ql>>4)&0x0f0f0f0f)|(qh&0x30303030))):
        want=sum((((q>>(8*j))&255)-32&255)<<(8*j) for j in range(4))
        assert staged[row*77+word]==want
    assert staged[row*77+64]==struct.unpack("<I",struct.pack("<f",(row-31)/16))[0]
    for word in range(4): assert staged[row*77+66+word]==int.from_bytes(blocks[base+192+4*word:base+196+4*word],"little")

def test_q6_mma_tile_cuda_lowering_uses_packed_lane_map():
  src=Q6KMMATile().emit_cuda()
  assert src.count("__vsubss4") == 2 and "tile[4928]" in src and "row*77" in src
