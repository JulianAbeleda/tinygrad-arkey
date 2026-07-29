"""Public prefill-attention topology search request and deterministic BLOCKED export."""
from __future__ import annotations
import argparse
from extra.llm_research.attention_topology_contract import blocked_record, export, request

TARGETS = {"Hq32": {"head_dim": 128, "q_heads": 32, "kv_heads": 8}, "Hq40": {"head_dim": 128, "q_heads": 40, "kv_heads": 8}}

def search_request(target: str): return request("prefill", target, TARGETS[target])
def blocked(target: str): return blocked_record("prefill", target, TARGETS[target])

if __name__ == "__main__":
  p = argparse.ArgumentParser(); p.add_argument("target", choices=sorted(TARGETS)); p.add_argument("--out", required=True); a = p.parse_args()
  export(blocked(a.target), a.out)
