import json
from decimal import Decimal
from tinygrad.device import ProfileGraphEntry
from tinygrad.runtime.graph.hcq import graph_profile_payload

def test_graph_profile_payload_is_json_safe_and_preserves_dispatches():
  entries = [ProfileGraphEntry("AMD", "ffn_down_kernel", 0, 1),
             ProfileGraphEntry("AMD", "rmsnorm", 2, 3)]
  payload = graph_profile_payload(entries, [[0], []], [Decimal(10), Decimal(17), Decimal(20), Decimal(23)])
  assert payload["schema"] == "tinygrad.hcq_graph_profile.v1"
  assert [row["name"] for row in payload["entries"]] == ["ffn_down_kernel", "rmsnorm"]
  assert [row["duration"] for row in payload["entries"]] == ["7", "3"]
  json.dumps(payload)
