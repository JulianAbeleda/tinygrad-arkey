import pathlib, re


ROOT = pathlib.Path(__file__).resolve().parents[2]
ADAPTERS = {
  ROOT / "tinygrad/llm/route_ops.py",
  ROOT / "tinygrad/codegen/experimental.py",
}
ROUTE_IMPORT = re.compile(r"^\s*(?:from|import)\s+extra\.(?:qk|q4_|q6_|q8_)", re.MULTILINE)


def test_tinygrad_qk_extra_imports_are_behind_adapters():
  offenders = []
  for path in (ROOT / "tinygrad").rglob("*.py"):
    if path in ADAPTERS: continue
    if ROUTE_IMPORT.search(path.read_text()):
      offenders.append(str(path.relative_to(ROOT)))
  assert offenders == []
