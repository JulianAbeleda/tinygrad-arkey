from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from tinygrad.dtype import AddrSpace, PtrDType, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import Ops, UOp

REQUIRED_WMMA_PROOF_FIELDS = ("role", "lds_buffer_id", "dbuf_slot", "k_phase", "logical_row_or_col", "byte_len",
                              "producer_epoch", "overwrite_epoch")

@dataclass(frozen=True)
class PrefillAMDISARendererPolicy:
  name: str = "prefill"

  def wmma_frag_proof_from_elem(self, e:UOp) -> dict|None:
    if e.op is Ops.GEP: e = e.src[0]
    tags = [e.tag]
    if e.op is Ops.LOAD and e.src:
      tags.append(e.src[0].tag)
      idx = e.src[0].src[0] if e.src[0].op is Ops.CAST else e.src[0]
      if idx.op is Ops.INDEX: tags.append(idx.tag)
    tag = next((t for t in tags if isinstance(t, tuple) and t and t[0] == "wmma_frag_proof"), None)
    if tag is None: return None
    try: return {k: v for k, v in tag[1:]}
    except Exception: return None

  def wmma_frag_buffer_proof_from_desc(self, desc:Any, role:str, h:Any) -> dict|None:
    proof = self.wmma_frag_buffer_proof_from_tag(None if desc is None else desc.buf.tag, desc, role)
    if proof is not None or desc is None or not getenv("PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC", 0): return proof
    byte_len = 32
    byte_start = desc.const_bytes
    ptr_key = h.lds_key_uop(desc.buf)
    window = (ptr_key, byte_start, byte_start + byte_len)
    return {
      "role": role, "lds_buffer_id": ptr_key, "nbuf": None, "dbuf_slot": ("lds_byte_window", window),
      "k_phase": ("lds_byte_window", window), "logical_row_or_col": (role, byte_start, byte_len),
      "byte_start": byte_start, "byte_len": byte_len, "producer_epoch": ("lds_write_before_barrier", window),
      "overwrite_epoch": ("next_write_to_lds_window", window),
    }

  def wmma_frag_buffer_proof_from_tag(self, tag:Any, desc:Any, role:str) -> dict|None:
    if desc is None: return None
    if not (isinstance(tag, tuple) and tag and tag[0] == "wmma_frag_buffer_proof"): return None
    try: base = {k: v for k, v in tag[1:]}
    except Exception: return None
    if base.get("role") != role: return None
    tile_elems = base.get("tile_elems")
    nbuf = base.get("nbuf")
    if not isinstance(tile_elems, int) or not isinstance(nbuf, int) or tile_elems <= 0: return None
    slot_elems = tile_elems * max(int(base.get("tile_count", 1)), 1)
    slot_bytes = slot_elems * desc.itemsize
    dbuf_slot = desc.const_bytes // slot_bytes if slot_bytes > 0 else None
    return {
      "role": role, "lds_buffer_id": base.get("lds_buffer_id"), "nbuf": nbuf, "dbuf_slot": dbuf_slot,
      "k_phase": ("slot_mod", dbuf_slot), "logical_row_or_col": (role, desc.const_bytes % max(slot_bytes, 1)),
      "byte_start": desc.const_bytes, "byte_len": 32, "producer_epoch": ("lds_buffer", base.get("lds_buffer_id"), dbuf_slot),
      "overwrite_epoch": ("lds_buffer", base.get("lds_buffer_id"), dbuf_slot, "next"),
    }

  def wmma_frag_buffer_proof_from_elem(self, e:UOp, desc:Any, role:str, h:Any) -> dict|None:
    if e.op is Ops.GEP: e = e.src[0]
    if e.op is not Ops.LOAD or not e.src: return None
    addr = e.src[0]
    idx = addr.src[0] if addr.op is Ops.CAST else addr
    if idx.op is not Ops.INDEX or not idx.src: return None
    proof = self.wmma_frag_store_epoch_proof(idx, desc, role, h)
    if proof is not None: return proof
    return self.wmma_frag_buffer_proof_from_tag(addr.tag, desc, role) or self.wmma_frag_buffer_proof_from_tag(idx.tag, desc, role) or \
           self.wmma_frag_buffer_proof_from_tag(idx.src[0].tag, desc, role)

  def wmma_frag_store_epoch_proof(self, idx:UOp, desc:Any, role:str, h:Any) -> dict|None:
    if desc is None or not getenv("PREFILL_WMMA_AB_PROOF_FROM_LDS_STORES", 0): return None
    base = idx.src[0].src[0] if idx.src[0].op is Ops.AFTER and idx.src[0].src else None
    barrier = next((s for s in idx.src[0].src[1:] if s.op is Ops.BARRIER), None) if idx.src[0].op is Ops.AFTER else None
    if base is None or barrier is None: return None
    byte_start, byte_end = desc.const_bytes, desc.const_bytes + 32
    producers:list[tuple] = []
    overlapping:list[tuple[int, int]] = []
    for st in barrier.toposort():
      if st.op is not Ops.STORE or len(st.src) < 2: continue
      sidx = st.src[0].src[0] if st.src[0].op is Ops.CAST and st.src[0].src else st.src[0]
      if sidx.op is not Ops.INDEX or sidx.addrspace != AddrSpace.LOCAL or len(sidx.src) < 2: continue
      if h.reg_base(sidx.src[0]) is not desc.buf: continue
      _sdyn, sconst = h.const_base(sidx.src[1])
      width = h.uop_byte_width(st.src[1]) or desc.itemsize
      s0, s1 = desc.base_bytes + sconst * desc.itemsize, desc.base_bytes + sconst * desc.itemsize + width
      if s1 <= byte_start or s0 >= byte_end: continue
      overlapping.append((s0, s1))
      producers.append((h.lds_key_uop(st.src[1]), s0, s1, id(st)))
    if not producers: return None
    for b in range(byte_start, byte_end):
      if sum(1 for s0, s1 in overlapping if s0 <= b < s1) != 1: return None
    producer_epoch = tuple(sorted(producers, key=lambda x: (x[1], x[2], x[3])))
    ptr_key = h.lds_key_uop(desc.buf)
    window = (ptr_key, byte_start, byte_end, producer_epoch)
    return {
      "role": role, "lds_buffer_id": ptr_key, "nbuf": None, "dbuf_slot": ("lds_store_epoch", window),
      "k_phase": ("lds_store_epoch", window), "logical_row_or_col": (role, byte_start, 32),
      "byte_start": byte_start, "byte_len": 32, "producer_epoch": producer_epoch,
      "overwrite_epoch": ("next_write_to_lds_window", ptr_key, byte_start, byte_end, id(barrier)),
    }

  def wmma_frag_proof_key(self, role:str, carrier:UOp, h:Any) -> tuple|None:
    try: elems = h.wmma_elems(carrier, 16)
    except Exception: return None
    proofs = [self.wmma_frag_proof_from_elem(e) for e in elems]
    if any(p is None for p in proofs): return None
    p0 = proofs[0]
    if p0.get("role") != role or any(p0.get(k) is None for k in REQUIRED_WMMA_PROOF_FIELDS): return None
    if any(p != p0 for p in proofs[1:]): return None
    return tuple((k, p0[k]) for k in REQUIRED_WMMA_PROOF_FIELDS)

  def wmma_frag_proof_reuse_key(self, ctx:Any, role:str, carrier:UOp, h:Any) -> tuple|None:
    try: elems = h.wmma_elems(carrier, 16)
    except Exception: return None
    addrs = [h.wmma_half_addr(e) for e in elems]
    if any(a is None for a in addrs): return None
    idx0, ptr0, expr0, c0 = addrs[0]
    if any(ptr is not ptr0 or expr is not expr0 or c != c0 + i for i, (_idx, ptr, expr, c) in enumerate(addrs)): return None
    proof = self.wmma_frag_proof_from_elem(elems[0])
    desc = h.decompose_lds_index(ctx, idx0, None) if idx0.addrspace == AddrSpace.LOCAL else None
    if proof is None:
      proof = self.wmma_frag_buffer_proof_from_elem(elems[0], desc, role, h) or self.wmma_frag_buffer_proof_from_desc(desc, role, h)
    if proof is None: return None
    if proof.get("role") != role or any(proof.get(k) is None for k in REQUIRED_WMMA_PROOF_FIELDS): return None
    return tuple((k, proof[k]) for k in REQUIRED_WMMA_PROOF_FIELDS)

  def wmma_frag_phase_reuse_key(self, ctx:Any, role:str, carrier:UOp, h:Any) -> tuple|None:
    if not getenv("PREFILL_WMMA_AB_PHASE_SCOPED_KEY", 0): return None
    try: elems = h.wmma_elems(carrier, 16)
    except Exception: return None
    addrs = [h.wmma_half_addr(e) for e in elems]
    if any(a is None for a in addrs): return None
    idx0, ptr0, expr0, c0 = addrs[0]
    if any(ptr is not ptr0 or expr is not expr0 or c != c0 + i for i, (_idx, ptr, expr, c) in enumerate(addrs)): return None
    desc = h.decompose_lds_index(ctx, idx0, None) if idx0.addrspace == AddrSpace.LOCAL else None
    proof = self.wmma_frag_proof_from_elem(elems[0])
    if proof is None:
      proof = self.wmma_frag_buffer_proof_from_elem(elems[0], desc, role, h) or self.wmma_frag_buffer_proof_from_desc(desc, role, h)
    if proof is None or desc is None or proof.get("role") != role: return None
    if getenv("PREFILL_WMMA_PHASE_EXACT_WINDOW", 0):
      return (("role", role), ("lds_buffer_id", proof.get("lds_buffer_id")), ("byte_start", desc.const_bytes), ("byte_len", 32))
    tile_bytes = getenv("PREFILL_WMMA_PHASE_TILE_BYTES_A" if role == "A" else "PREFILL_WMMA_PHASE_TILE_BYTES_B", 128)
    if tile_bytes <= 0: return None
    rel = desc.const_bytes - desc.base_bytes
    return (("role", role), ("lds_buffer_id", proof.get("lds_buffer_id")), ("phase_row_or_col", rel // tile_bytes), ("byte_len", 32))

  def dbuf_stage_store_key(self, st:UOp, h:Any) -> tuple|None:
    if st.op is not Ops.STORE or not st.src: return None
    if isinstance(st.tag, tuple) and st.tag[:1] == ("tc_local_stage_store",): return st.tag
    idx = st.src[0]
    while idx.op in (Ops.AFTER, Ops.CAST) and idx.src: idx = idx.src[0]
    if idx.op is Ops.NOOP and isinstance(idx.tag, tuple) and idx.tag[:1] == ("tc_local_stage_store",): return idx.tag
    if isinstance(idx.tag, tuple) and idx.tag[:1] == ("tc_local_stage_store",): return idx.tag
    return h.lds_proof_key(idx, 8) if idx.op is Ops.INDEX and idx.addrspace == AddrSpace.LOCAL else None

  def dbuf_stage_value_key(self, st:UOp, h:Any) -> tuple|None:
    if st.op is not Ops.STORE or len(st.src) < 2: return None
    val = st.src[1]
    if val.op is Ops.AFTER and val.src: val = val.src[0]
    if val.op is Ops.NOOP and isinstance(val.arg, tuple) and len(val.arg) == 2 and val.arg[0] == "global_b128":
      return ("global_b128", h.lds_key_uop(val.arg[1]))
    if val.op is Ops.NOOP and val.dtype.count == 4 and val.dtype.scalar().itemsize == 4 and all(h.is_vpack_int32(s) for s in val.src):
      return ("vpack4", tuple(tuple(h.lds_key_uop(x) for x in p.src[:2]) for p in val.src))
    return None

  def dbuf_stage_candidate(self, carrier:UOp, h:Any) -> tuple[UOp|None, str]:
    try:
      elems = h.wmma_elems(carrier, 16)
      addrs = [h.wmma_half_addr(e) for e in elems]
    except Exception:
      return None, "not_wmma_memory_carrier"
    if any(a is None for a in addrs): return None, "non_memory_lane"
    for idx, ptr, _expr, _const in addrs:
      if not isinstance(ptr.dtype, PtrDType) or ptr.dtype.addrspace != AddrSpace.LOCAL: return None, "not_local_lds_operand"
      for st in idx.src[0].toposort():
        if st.op is Ops.STORE and len(st.src) >= 2:
          val = st.src[1]
          if val.op is Ops.NOOP and isinstance(val.arg, tuple) and len(val.arg) == 2 and val.arg[0] == "global_b128":
            return st, "global_b128"
          vv = val.src[0] if val.op is Ops.AFTER and val.src else val
          if vv.op is Ops.NOOP and vv.dtype.count == 4 and vv.dtype.scalar().itemsize == 4 and all(h.is_vpack_int32(s) for s in vv.src):
            return st, "vpack_vec4"
    return None, "no_matching_store_value"

  def dbuf_stage_candidates(self, carrier:UOp, h:Any) -> tuple[list[UOp], str]:
    try:
      elems = h.wmma_elems(carrier, 16)
      addrs = [h.wmma_half_addr(e) for e in elems]
    except Exception:
      return [], "not_wmma_memory_carrier"
    if any(a is None for a in addrs): return [], "non_memory_lane"
    idx0, ptr0, _expr0, c0 = addrs[0]
    if not isinstance(ptr0.dtype, PtrDType) or ptr0.dtype.addrspace != AddrSpace.LOCAL: return [], "not_local_lds_operand"
    target_consts = {c0, c0 + 8}
    out: list[tuple[int, UOp]] = []
    for st in idx0.src[0].toposort():
      if st.op is not Ops.STORE or len(st.src) < 2: continue
      sk = self.dbuf_stage_store_key(st, h)
      if sk is None or len(sk) < 3 or sk[2] not in target_consts: continue
      val = st.src[1]
      ok = val.op is Ops.NOOP and isinstance(val.arg, tuple) and len(val.arg) == 2 and val.arg[0] == "global_b128"
      vv = val.src[0] if val.op is Ops.AFTER and val.src else val
      ok = ok or (vv.op is Ops.NOOP and vv.dtype.count == 4 and vv.dtype.scalar().itemsize == 4 and all(h.is_vpack_int32(s) for s in vv.src))
      if ok: out.append((int(sk[2]), st))
    if not out: return [], "no_matching_store_value"
    return [st for _, st in sorted(out, key=lambda x: x[0])], "ok"

  def dbuf_stage_owner_key(self, role:str, slot:int|None, vkey:tuple|None, phase_i:int|None) -> tuple|None:
    if slot is None or vkey is None: return None
    return ("stage_owner", ("role", role), ("source", vkey), ("logical_phase", phase_i), ("lds_slot", slot))

PREFILL_AMD_ISA_RENDERER_POLICY = PrefillAMDISARendererPolicy()
