import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
M3 = ROOT / "extra/llm_research/decode/m3_machine_search"

sys.path.insert(0, str(M3))
from check_export import FORBIDDEN, validate

def test_m3_export_is_finite_generic_and_honest(tmp_path=None):
  # Works under pytest and as a standalone focused check without pytest installed.
  import tempfile
  tmp = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
  first, second = tmp / "first.json", tmp / "second.json"
  for out in (first, second): subprocess.run([sys.executable, str(M3 / "export_search.py"), "--out", str(out)], check=True, cwd=ROOT)
  assert first.read_bytes() == second.read_bytes()
  request = json.loads((M3 / "search_request.json").read_text()); export = json.loads(first.read_text())
  validate(request, export)
  assert export["routes"]["decode_q4k_g3_generated"]["population_size"] == 3
  assert export["routes"]["decode_q6k_coop_generated"]["population_size"] == 7
  assert export["promotion"]["status"] == "blocked"
  assert export["promotion"]["missing_executor"] == "route-bound generic topology-plan executor"
  assert export["promotion"]["missing_primitive_lowerers"] == ["q4k_packed_block_dot", "q6k_packed_block_dot", "external_reduce"]
  catalog = json.loads((ROOT / "tinygrad/llm/generated/catalog.json").read_text())
  assert not any("m3." in json.dumps(item) for item in catalog["artifacts"])
  for result in export["routes"].values():
    assert result["selected"] == result["ranked_population"][0]
    for row in result["ranked_population"]:
      assert row["plan"]["plan_schema"] == "tinygrad.generic_topology_plan.v1"
      assert not any(name in json.dumps(row["plan"]) for name in FORBIDDEN)
