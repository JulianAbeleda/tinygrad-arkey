import pathlib

import sz


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_default_authored_budget_gate_is_minimization_target():
  assert sz.DEFAULT_MAX_LINE_COUNT == 30000


def test_vendored_exclusion_stays_narrow():
  assert sz.EXCLUDE == ["tinygrad/viz/assets"]


def test_runtime_autogen_modules_declare_generated_marker():
  missing = []
  for path in (ROOT / "tinygrad/runtime/autogen").rglob("*.py"):
    if path.name == "__init__.py": continue
    if not sz.is_generated(path): missing.append(str(path.relative_to(ROOT)))
  assert missing == []


def test_runtime_autogen_init_files_are_authored():
  for path in (ROOT / "tinygrad/runtime/autogen").rglob("__init__.py"):
    assert not sz.is_generated(path), str(path.relative_to(ROOT))
