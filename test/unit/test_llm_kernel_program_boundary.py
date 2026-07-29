"""Static architecture gates for the LLM kernel-program execution boundary."""
import ast
from pathlib import Path


_BOUNDARY = Path("tinygrad/llm/kernel_program.py")
_RESTRICTED_EXECUTORS = frozenset(("execute_oracle_program", "execute_research_program", "execute_research_program_outputs"))


def _repo_root() -> Path:
  for candidate in Path(__file__).resolve().parents:
    if (candidate / "pyproject.toml").is_file() and (candidate / "tinygrad").is_dir(): return candidate
  raise RuntimeError("could not locate repository root")


def _python_sources(root: Path) -> list[Path]:
  sources = []
  for path in root.rglob("*.py"):
    relative = path.relative_to(root)
    if "__pycache__" in relative.parts: continue
    sources.append(path)
  return sorted(sources)


def _tree(path: Path) -> ast.AST:
  return ast.parse(path.read_text(), filename=str(path))


def _relative(root: Path, paths: list[Path]) -> list[str]:
  return [str(path.relative_to(root)) for path in paths]


def _custom_kernel_callers(paths: list[Path]) -> list[Path]:
  return [path for path in paths if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                                        node.func.attr == "custom_kernel" for node in ast.walk(_tree(path)))]


def _restricted_executor_users(paths: list[Path]) -> list[Path]:
  users = []
  for path in paths:
    for node in ast.walk(_tree(path)):
      if isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id in _RESTRICTED_EXECUTORS) or
                                         (isinstance(node.func, ast.Attribute) and node.func.attr in _RESTRICTED_EXECUTORS)):
        users.append(path)
        break
      if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("kernel_program") and any(
          alias.name in _RESTRICTED_EXECUTORS or alias.name == "*" for alias in node.names):
        users.append(path)
        break
  return users


def test_llm_custom_kernel_transport_is_confined_to_kernel_program_boundary():
  root = _repo_root()
  llm_sources = _python_sources(root / "tinygrad" / "llm")
  callers = _custom_kernel_callers(llm_sources)
  assert _relative(root, callers) == [str(_BOUNDARY)], "direct .custom_kernel calls must stay inside the kernel-program boundary"


def test_production_llm_does_not_import_or_call_oracle_or_research_execution():
  root = _repo_root()
  production_sources = [path for path in _python_sources(root / "tinygrad" / "llm") if path.relative_to(root) != _BOUNDARY]
  assert not _restricted_executor_users(production_sources), (
    "production LLM source may use only execute_promoted_program; oracle/research execution belongs outside production")
