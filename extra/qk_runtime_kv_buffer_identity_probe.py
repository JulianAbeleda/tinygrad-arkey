#!/usr/bin/env python3
"""Phase-1/2 probe for docs/runtime-kv-buffer-identity-rebase-scope-20260623.md.
Compares cache_kv identity after the model's canonical-store prefill (BAKES) vs a fresh assign-fill (ADVANCES),
and tests the pristine-buffer rebase fix at the prefill->decode handoff.
  run: DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 RUNTIME_KV_CACHE=1 PYTHONPATH=. .venv/bin/python extra/qk_runtime_kv_buffer_identity_probe.py
"""
import json, pathlib, sys
from collections import Counter
import numpy as np
from tinygrad import Tensor, UOp, TinyJit, dtypes
OUT = pathlib.Path(__file__).resolve().parents[1] / "bench/qk-runtime-managed-kv-cache"
MAXC = 4608

def fingerprint(t):
    u = t.uop; o = {"id_tensor": id(t), "id_uop": id(u), "uop_op": str(u.op), "dtype": str(u.dtype), "shape": str(tuple(u.shape))}
    try: o["base_op"], o["id_base"] = str(u.base.op), id(u.base)
    except Exception as e: o["base_err"] = str(e)[:40]
    for attr in ("buffer",):
        try: b = getattr(u, attr); o[f"{attr}_id"] = id(b); o[f"{attr}_str"] = str(b)[:90]
        except Exception as e: o[f"{attr}_err"] = str(e)[:50]
    try: bb = u.base.buffer; o["base_buffer_id"] = id(bb); o["base_buffer_str"] = str(bb)[:90]
    except Exception as e: o["base_buffer_err"] = str(e)[:50]
    seen, ops, bufids = set(), [], []
    def walk(x, d):
        if d > 10 or id(x) in seen: return
        seen.add(id(x)); ops.append(str(x.op).split(".")[-1])
        if str(x.op).endswith("BUFFER"):
            try: bufids.append(id(x.buffer))
            except Exception: bufids.append(("uop", id(x)))
        for s in x.src: walk(s, d + 1)
    walk(u, 0)
    o["op_counts"] = dict(Counter(ops)); o["buffer_uop_ids"] = bufids
    return o

def main():
    from extra.llm_generate import load_model_and_tokenizer
    from extra.qk_harness_contract import DEFAULT_MODEL
    res = {"date": "2026-06-23", "phase": "BUFFER_IDENTITY_DIFF+REBASE", "gpu": "gfx1100"}
    m, tok = load_model_and_tokenizer(DEFAULT_MODEL, MAXC, seed=20260617)
    for lin in (getattr(m, '_q4k_linears', None).linears if getattr(m, '_q4k_linears', None) else []): lin.decode_enabled = True
    for b in m.blk: b._use_flash, b._prefill_v2 = True, False
    v = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])
    ids = ((tok.prefix() if hasattr(tok, 'prefix') else []) + tok.encode("the quick brown fox jumps. " * 500))[:2050]

    # ---- canonical-store-fill state (batched prefill -> bakes) ----
    o = m.forward(Tensor([ids], dtype="int32").contiguous(), 0, temp).realize()
    res["canonical_store_fill"] = fingerprint(m.blk[0].cache_kv)

    # ---- assign-fill state (fresh buffer, same data) ----
    data0 = m.blk[0].cache_kv.numpy()
    fresh = Tensor.zeros(2, 1, 8, MAXC, 128, dtype=dtypes.float16).contiguous().realize()
    fresh.assign(Tensor(data0)).realize()
    res["assign_fill"] = fingerprint(fresh)

    # identity diff
    cf, af = res["canonical_store_fill"], res["assign_fill"]
    diffs = {k: [cf.get(k), af.get(k)] for k in set(cf) | set(af) if cf.get(k) != af.get(k) and not k.startswith("id_") and k not in ("buffer_id", "base_buffer_id", "buffer_uop_ids", "buffer_str", "base_buffer_str")}
    res["identity_diff_fields"] = diffs

    # ---- Phase 2: rebase fix -- replace every block cache with a fresh assign-filled buffer, then decode ----
    toks0 = [int(o.item())]
    for b in m.blk:
        d = b.cache_kv.numpy()
        nf = Tensor.zeros(2, 1, 8, MAXC, 128, dtype=dtypes.float16).contiguous().realize()
        nf.assign(Tensor(d)).realize()
        b.cache_kv = nf
    res["rebased_cache_fingerprint"] = fingerprint(m.blk[0].cache_kv)
    dec = TinyJit(m.forward); toks = list(toks0); sp = len(ids)
    for _ in range(7):
        o = dec(Tensor([[toks[-1]]], dtype="int32").contiguous(), v.bind(sp), temp).realize(); toks.append(int(o.item())); sp += 1
    c = m.blk[0].cache_kv.numpy()
    written = [p for p in range(2048, 2060) if np.any(c[:, 0, :, p, :] != 0)]
    res["rebase_decode"] = {"tokens": toks, "decode_positions_written": written,
                            "advances": written == list(range(len(ids), len(ids) + 8)) or len([p for p in written if p >= len(ids)]) >= 6}
    res["verdict_diff"] = "BUFFER_IDENTITY_DIFF_FOUND" if diffs else "BUFFER_IDENTITY_DIFF_NOT_FOUND"
    res["verdict_rebase"] = "PRISTINE_REBASE_ADVANCES" if res["rebase_decode"]["advances"] else "PRISTINE_REBASE_COPY_STILL_BAKES"
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "buffer_identity_diff.json").write_text(json.dumps(res, indent=2, default=str))
    print("identity_diff_fields:", json.dumps(diffs, default=str), file=sys.stderr)
    print("canonical buffer_uop_ids:", cf.get("buffer_uop_ids"), "| assign:", af.get("buffer_uop_ids"), file=sys.stderr)
    print("REBASE tokens:", toks, file=sys.stderr)
    print("REBASE positions written:", written, file=sys.stderr)
    print("VERDICT diff:", res["verdict_diff"], "| rebase:", res["verdict_rebase"], file=sys.stderr)

if __name__ == "__main__": main()
