import pathlib, re


ROOT = pathlib.Path(__file__).resolve().parents[2]
ADAPTERS = {
  ROOT / "tinygrad/codegen/experimental.py",
}
ROUTE_IMPORT = re.compile(r"(?:^\s*(?:from|import)\s+extra\.llm_research|extra\.llm_research)", re.MULTILINE)


def test_tinygrad_llm_has_no_research_import_or_dynamic_reference():
  offenders = []
  for path in (ROOT / "tinygrad/llm").rglob("*.py"):
    if path in ADAPTERS: continue
    if ROUTE_IMPORT.search(path.read_text()):
      offenders.append(str(path.relative_to(ROOT)))
  assert offenders == []


def test_route_ops_adapter_was_retired():
  assert not (ROOT / "tinygrad/llm/route_ops.py").exists()
