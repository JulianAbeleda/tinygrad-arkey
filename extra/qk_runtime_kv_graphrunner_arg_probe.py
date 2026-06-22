#!/usr/bin/env python3
"""GraphRunner arg-patch probe (docs/runtime-kv-graphrunner-arg-patch-result-20260623.md).
Instruments the full-model RUNTIME_KV decode at the HCQ-graph kernarg level: is 'start_pos' in the graph's
patched vars? do the kv_append / owned_flash calls have var_vals_replace entries? does start_pos change per replay?
  run: DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 RUNTIME_KV_CACHE=1 PYTHONPATH=. .venv/bin/python extra/qk_runtime_kv_graphrunner_arg_probe.py
"""
import json, pathlib, sys
import numpy as np
OUT = pathlib.Path(__file__).resolve().parents[1] / "bench/qk-runtime-managed-kv-cache"

GRAPHS = []   # metadata per HCQGraph instance, captured at __init__
CALLS = []    # start_pos per graph __call__

def install_hooks():
    import tinygrad.runtime.graph.hcq as H
    Graph = H.HCQGraph
    _orig_init = Graph.__init__
    _orig_call = Graph.__call__
    def init(self, *a, **k):
        _orig_init(self, *a, **k)
        meta = {"graph_id": id(self), "n_calls": len(self.calls), "vars": list(self.vars),
                "start_pos_in_vars": "start_pos" in self.vars, "n_var_vals_replace": len(self.var_vals_replace)}
        prog_calls = []
        for j in range(len(self.calls)):
            ast = self.calls[j][1]
            if ast.op.name == "PROGRAM":
                nm = str(getattr(ast.arg, "name", ""))
                if "kv_append" in nm or "owned_flash" in nm:
                    pvars = [str(v.expr) for v in ast.arg.vars]
                    prog_calls.append({"j": j, "name": nm, "prog_vars": pvars,
                                       "in_var_vals_replace": j in self.var_vals_replace,
                                       "replace": self.var_vals_replace.get(j)})
        meta["append_tile_calls"] = prog_calls
        meta["n_kv_append"] = sum("kv_append" in c["name"] for c in prog_calls)
        meta["n_owned_flash"] = sum("owned_flash" in c["name"] for c in prog_calls)
        GRAPHS.append(meta)
    def call(self, input_uops, var_vals, wait=False):
        if any("kv_append" in str(getattr(self.calls[j][1].arg, "name", "")) for j in range(len(self.calls)) if self.calls[j][1].op.name == "PROGRAM"):
            CALLS.append({"graph_id": id(self), "start_pos": var_vals.get("start_pos"), "n_vars_passed": len(var_vals)})
        return _orig_call(self, input_uops, var_vals, wait)
    Graph.__init__, Graph.__call__ = init, call

def main():
    install_hooks()
    from tinygrad import Tensor, UOp, TinyJit
    from extra.llm_generate import load_model_and_tokenizer
    from extra.qk_harness_contract import DEFAULT_MODEL
    res = {"date": "2026-06-23", "phase": "GRAPHRUNNER_ARG_PATCH", "gpu": "gfx1100"}
    m, tok = load_model_and_tokenizer(DEFAULT_MODEL, 4608, seed=20260617)
    for lin in (getattr(m, '_q4k_linears', None).linears if getattr(m, '_q4k_linears', None) else []): lin.decode_enabled = True
    for b in m.blk: b._use_flash, b._prefill_v2 = True, False
    v = UOp.variable("start_pos", 0, 4607); temp = Tensor([0.0])
    ids = ((tok.prefix() if hasattr(tok, 'prefix') else []) + tok.encode("the quick brown fox jumps. " * 500))[:2048]
    o = None; sp = 0
    for st in range(0, len(ids), 512):
        o = m.forward(Tensor([ids[st:st+512]], dtype="int32").contiguous(), sp, temp).realize(); sp += len(ids[st:st+512])
    GRAPHS.clear(); CALLS.clear()   # only care about the DECODE graphs
    dec = TinyJit(m.forward); toks = [int(o.item())]
    for _ in range(5):
        o = dec(Tensor([[toks[-1]]], dtype="int32").contiguous(), v.bind(sp), temp).realize(); toks.append(int(o.item())); sp += 1
    res["decode_tokens"] = toks
    res["graphs_with_append"] = [g for g in GRAPHS if g["n_kv_append"] > 0]
    res["call_start_pos_sequence"] = CALLS
    sps = [c["start_pos"] for c in CALLS]
    res["start_pos_changes_across_calls"] = len(set(sps)) > 1
    res["analysis"] = {
        "n_decode_graphs_total": len(GRAPHS),
        "n_graphs_with_kv_append": len(res["graphs_with_append"]),
        "start_pos_in_every_append_graph_vars": all(g["start_pos_in_vars"] for g in res["graphs_with_append"]) if res["graphs_with_append"] else None,
        "every_append_call_in_var_vals_replace": all(c["in_var_vals_replace"] for g in res["graphs_with_append"] for c in g["append_tile_calls"] if "kv_append" in c["name"]) if res["graphs_with_append"] else None,
        "start_pos_values_seen": sps,
    }
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "graphrunner_arg_probe.json").write_text(json.dumps(res, indent=2, default=str))
    a = res["analysis"]
    print("decode tokens:", toks, file=sys.stderr)
    print("n decode graphs:", a["n_decode_graphs_total"], "| with kv_append:", a["n_graphs_with_kv_append"], file=sys.stderr)
    print("start_pos in every append-graph .vars:", a["start_pos_in_every_append_graph_vars"], file=sys.stderr)
    print("every kv_append call in var_vals_replace:", a["every_append_call_in_var_vals_replace"], file=sys.stderr)
    print("start_pos values across replays:", a["start_pos_values_seen"], "| changes:", res["start_pos_changes_across_calls"], file=sys.stderr)
    for g in res["graphs_with_append"][:2]:
        print(f"  graph n_kv_append={g['n_kv_append']} n_owned_flash={g['n_owned_flash']} start_pos_in_vars={g['start_pos_in_vars']} vars={g['vars'][:6]}", file=sys.stderr)
        for c in g["append_tile_calls"][:4]: print(f"    {c['name']}: prog_vars={c['prog_vars']} in_replace={c['in_var_vals_replace']} replace={c['replace']}", file=sys.stderr)

if __name__ == "__main__": main()
