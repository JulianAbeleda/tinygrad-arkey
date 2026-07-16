from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from tinygrad.dtype import AddrSpace, PtrDType, dtypes
from tinygrad.codegen.opt import KernelOptError
from tinygrad.codegen.opt.prefill_value_key import PrefillSourceValueKey
from tinygrad.uop.ops import Ops, UOp
from extra.qk.mmq_llama_record_producers import RecordProducerInstanceWitness

REQUIRED_WMMA_PROOF_FIELDS = ("role", "lds_buffer_id", "dbuf_slot", "k_phase", "logical_row_or_col", "byte_len",
                              "producer_epoch", "overwrite_epoch")

@dataclass(frozen=True)
class PrefillAMDISARendererPolicy:
  name: str = "prefill"

  def prefill_source_value_key(self, *tags:Any) -> PrefillSourceValueKey|None:
    """Consume only explicit, typed source identity; address equivalence is not identity."""
    keys:list[PrefillSourceValueKey] = []
    for tag in tags:
      if not isinstance(tag, tuple): continue
      claims = [item for item in tag[1:] if isinstance(item, tuple) and item and item[0] == "value_key"]
      if not claims: continue
      if len(claims) != 1 or len(claims[0]) != 2:
        raise KernelOptError("AMD prefill source-value metadata has malformed or duplicate value_key fields")
      key = claims[0][1]
      if not isinstance(key, PrefillSourceValueKey):
        raise KernelOptError("AMD prefill source-value metadata must carry PrefillSourceValueKey")
      role_claims = [item[1] for item in tag[1:] if isinstance(item, tuple) and len(item) == 2 and item[0] == "role"]
      if role_claims and (len(set(role_claims)) != 1 or role_claims[0] != key.role):
        raise KernelOptError("AMD prefill source-value metadata role conflicts with typed key")
      fields = {item[0]: item[1] for item in tag[1:] if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)}
      if "nbuf" in fields and fields["nbuf"] != 1:
        raise KernelOptError("AMD prefill source-value metadata requires nbuf=1")
      if "lds_buffer_id" in fields and key.buffer_id != ("lds", fields["lds_buffer_id"], 0):
        raise KernelOptError("AMD prefill source-value metadata LDS buffer conflicts with typed key")
      if "owned_stage" in fields and fields["owned_stage"] != f"{key.role}_IDENTITY":
        raise KernelOptError("AMD prefill source-value metadata conflicts with owned identity stage")
      keys.append(key)
    if not keys: return None
    if any(key != keys[0] for key in keys[1:]):
      raise KernelOptError("AMD prefill source-value metadata carries conflicting typed keys")
    return keys[0]

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
    return self.wmma_frag_buffer_proof_from_tag(None if desc is None else desc.buf.tag, desc, role)

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
    """Derive the fragment contract from the staged record that feeds ``idx``.

    The store tag is only a provenance witness; identity is taken from the
    structural LDS base and the byte address.  Do not accept a tag unless the
    index's dependency graph contains exactly one matching stage record.
    """
    if desc is None or idx.op is not Ops.INDEX or not idx.src: return None
    if idx.addrspace != AddrSpace.LOCAL: return None
    try: lds_id = h.lds_key_uop(desc.buf)
    except Exception: return None
    if role == "B":
      # Q8 staging is a vector copy expressed as four scalar STOREs.  The
      # constant part of those addresses names the four lanes, while the typed
      # destination_vector coordinate names the complete record extent.  Prove
      # membership from those native values, then choose the latest visible
      # producer epoch from dependency order.  In particular, do not use the
      # descriptive STORE tag to authorize this composed half-record carrier.
      groups:dict[tuple, list[tuple[int, UOp, RecordProducerInstanceWitness]]] = {}
      order = {u: n for n, u in enumerate(idx.toposort())}
      for u in order:
        if u.op is not Ops.STORE or not isinstance(u.arg, RecordProducerInstanceWitness): continue
        w = u.arg
        if w.role != role or w.schema != "llama-q8-ds4-producer-instance.v1": continue
        st_idx = u.src[0].src[0] if u.src[0].op is Ops.CAST else u.src[0]
        if st_idx.op is not Ops.INDEX or st_idx.addrspace != AddrSpace.LOCAL: continue
        try:
          if h.lds_key_uop(h.reg_base(st_idx.src[0])) != lds_id: continue
          _base, c = h.const_base(st_idx.src[1])
          st_bytes = c * st_idx.src[0].dtype.base.itemsize
        except Exception: continue
        key = (w.role, w.field, w.phase, w.slot, w.iteration, w.schema,
               w.source_row, w.source_k, w.destination_row, w.destination_vector)
        groups.setdefault(key, []).append((st_bytes, u, w))
      candidates = []
      for lanes in groups.values():
        lanes.sort(key=lambda x: x[0])
        starts = [x[0] for x in lanes]
        w = lanes[0][2]
        if w.field != "qs" or len(starts) < 2 or starts != list(range(starts[0], starts[0]+len(starts))): continue
        try: vector_count = int(w.destination_vector.vmax) - int(w.destination_vector.vmin) + 1
        except Exception: continue
        if vector_count <= 0: continue
        record_bytes = vector_count * len(starts)
        if starts[0] <= desc.const_bytes and desc.const_bytes + 16 <= starts[0] + record_bytes:
          candidates.append(lanes)
      if not candidates: return None
      # Multiple epochs can be visible through an effect chain.  A legal
      # overwrite has a unique latest producer group in that native chain.
      latest = max(candidates, key=lambda lanes: max(order[x[1]] for x in lanes))
      latest_closure = set().union(*(x[1].backward_slice_with_self for x in latest))
      if any(any(x[1] not in latest_closure for x in lanes) for lanes in candidates if lanes is not latest): return None
      w = latest[0][2]
      return {"role": role, "lds_buffer_id": lds_id,
        "dbuf_slot": w.slot, "k_phase": ("stage_epoch", w.phase, w.slot),
        "logical_row_or_col": (role, desc.const_bytes), "byte_start": desc.const_bytes,
        "byte_len": 16, "producer_epoch": ("stage", lds_id, w.phase, w.slot),
        "overwrite_epoch": ("stage", lds_id, w.phase, w.slot, "next"),
        "field": w.field, "iteration": w.iteration, "schema": w.schema}
    matches = []
    stores = []
    for u in idx.toposort():
      tag = u.tag
      if u.op is not Ops.STORE or not (isinstance(tag, tuple) and tag[:1] == ("hierarchical_record_store",)):
        continue
      # hierarchical_record_store(role, region, phase, slot, schema)
      if len(tag) != 6 or tag[1] != role or not isinstance(tag[4], int): continue
      st_idx = u.src[0].src[0] if u.src[0].op is Ops.CAST else u.src[0]
      if st_idx.op is not Ops.INDEX or st_idx.addrspace != AddrSpace.LOCAL: continue
      try: st_buf = h.reg_base(st_idx.src[0])
      except Exception: continue
      if h.lds_key_uop(st_buf) != lds_id: continue
      try:
        _base, _c = h.const_base(st_idx.src[1])
        st_bytes = _c * st_idx.src[0].dtype.base.itemsize
      except Exception: continue
      # A producer tag identifies a class of stores only.  Accept a Q8 B
      # store as an epoch witness only when its value dependency contains
      # exactly one typed, coordinate-bearing producer instance.
      witnesses = (u.arg,) if isinstance(u.arg, RecordProducerInstanceWitness) else ()
      if tag[1] == "B":
        if len(witnesses) != 1: continue
        w = witnesses[0]
        if (w.role, w.field, w.phase, w.slot, w.iteration, w.schema) != (tag[1], tag[2], tag[3], tag[4], tag[4], "llama-q8-ds4-producer-instance.v1"):
          continue
      stores.append((tag, st_bytes, u))
    # A/B records are emitted as scalar stores, while the WMMA carrier reads
    # a fragment-sized window inside the record.  Normalize that representation
    # using the structural distance to the next record in the same phase.
    for tag, st_bytes, store in stores:
      peers = sorted(x[1] for x in stores if x[0][3] == tag[3] and x[0][1] == tag[1])
      following = [x for x in peers if x > st_bytes]
      span = following[0] - st_bytes if following else 0
      # The carrier's first half is the coordinate anchor.  The remaining
      # lanes may cross scalar record boundaries; wmma_frag_proof_reuse_key
      # separately proves their exact contiguous address chain.
      if span <= 0 or not (st_bytes <= desc.const_bytes < st_bytes + span): continue
      matches.append((tag, st_bytes, store, span))
    if len(matches) != 1: return None
    tag, st_bytes, store, span = matches[0]
    # Require the staged producer to be in the load's dependency closure; the
    # address match alone is not a proof of visibility or store ordering.
    if store not in idx.toposort(): return None
    witness_nodes = []
    if role == "B":
      witness_nodes = (store.arg,) if isinstance(store.arg, RecordProducerInstanceWitness) else ()
      if len(witness_nodes) != 1 or witness_nodes[0].schema != "llama-q8-ds4-producer-instance.v1": return None
    # The stage record's slot is the epoch witness; the load address is the
    # coordinate witness.  The next record boundary is the non-overlap witness.
    slot = tag[4]
    if role == "B":
      field, iteration, schema = witness_nodes[0].field, witness_nodes[0].iteration, witness_nodes[0].schema
    else:
      # Q4 A records do not carry the Q8 producer-instance witness. Their
      # hierarchical store tag is already structurally matched above and is
      # the exact field/phase/slot/schema authority for this staged record.
      field, iteration, schema = tag[2], slot, tag[5]
    return {"role": role, "lds_buffer_id": lds_id,
      "dbuf_slot": slot, "k_phase": ("stage_epoch", tag[3], slot),
      "logical_row_or_col": (role, desc.const_bytes), "byte_start": desc.const_bytes,
      "byte_len": 4, "producer_epoch": ("stage", lds_id, tag[3], slot),
      "overwrite_epoch": ("stage", lds_id, tag[3], slot, "next"),
      "field": field, "iteration": iteration, "schema": schema}

  def wmma_frag_proof_key(self, role:str, carrier:UOp, h:Any) -> tuple|None:
    try: elems = h.wmma_elems(carrier, 16)
    except Exception: return None
    proofs = [self.wmma_frag_proof_from_elem(e) for e in elems]
    if any(p is None for p in proofs): return None
    p0 = proofs[0]
    if p0.get("role") != role or any(p0.get(k) is None for k in REQUIRED_WMMA_PROOF_FIELDS): return None
    # Producer coordinates identify the four scalar records, not four
    # fragments.  Normalize them to one 16-byte carrier only after proving
    # that all lanes share the epoch and typed producer identity.
    identity = ("role", "lds_buffer_id", "dbuf_slot", "k_phase", "producer_epoch", "overwrite_epoch",
                "field", "iteration", "schema")
    if any(any(p.get(k) is None for k in ("field", "iteration", "schema")) for p in proofs): return None
    if any(tuple(p.get(k) for k in identity) != tuple(p0.get(k) for k in identity) for p in proofs): return None
    starts = [p.get("byte_start") for p in proofs]
    if any(not isinstance(x, int) or p.get("byte_len") != 4 for x, p in zip(starts, proofs)): return None
    counts = {x: starts.count(x) for x in set(starts)}
    if len(counts) != 4 or sorted(counts.values()) != [4, 4, 4, 4]: return None
    base = min(counts)
    if sorted(counts) != [base, base+4, base+8, base+12]: return None
    normalized = dict(p0, logical_row_or_col=(role, base), byte_len=16)
    return tuple((k, normalized[k]) for k in REQUIRED_WMMA_PROOF_FIELDS)

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
          if vv.op is Ops.NOOP and vv.dtype.count == 4 and vv.dtype.scalar().itemsize == 4 and \
             all(s.op is Ops.INS and s.arg.name == "V_PACK" and s.dtype is dtypes.int32 for s in vv.src):
            return st, "vpack_vec4"
    return None, "no_matching_store_value"

PREFILL_AMD_ISA_RENDERER_POLICY = PrefillAMDISARendererPolicy()
