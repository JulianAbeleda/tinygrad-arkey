"""Static architecture gates for the LLM kernel-program execution boundary."""
import ast
from pathlib import Path


_BOUNDARY = Path("tinygrad/llm/kernel_program.py")
# Closed-default research-admission surfaces. These modules host route admission
# candidates whose research-only spellings are unreachable unless a harness
# installs an explicit lease marker on a concrete linear. The promoted spelling
# (Q4 FFN-down fp16 geometry) uses execute_promoted_program; the remaining
# research branches stay here because they dispatch through the model forward.
_RESEARCH_ADMISSION_BOUNDARIES = frozenset((
  Path("tinygrad/llm/decode_routes.py"),
  Path("tinygrad/llm/q4k_ffn_down_mmvq.py"),
  Path("tinygrad/llm/qk_norm_rope_mmvq.py"),
))
_RESTRICTED_EXECUTORS = frozenset(("execute_oracle_program", "execute_research_program", "execute_research_program_outputs"))


def _repo_root() -> Path:
  for candidate in Path(__file__).resolve().parents:
    if (candidate / "pyproject.toml").is_file() and (candidate / "tinygrad").is_dir(): return candidate
  raise RuntimeError("could not locate repository root")


def _python_sources(root: Path, *, exclude_tests: bool = False) -> list[Path]:
  sources = []
  for path in root.rglob("*.py"):
    relative = path.relative_to(root)
    if "__pycache__" in relative.parts: continue
    if exclude_tests and any(part in {"test", "tests"} or part.startswith("test_") for part in relative.parts): continue
    sources.append(path)
  return sorted(sources)


def _tree(path: Path) -> ast.AST:
  return ast.parse(path.read_text(), filename=str(path))


def _relative(root: Path, paths: list[Path]) -> list[str]:
  return [str(path.relative_to(root)) for path in paths]


def _attribute_callers(paths: list[Path], attribute: str) -> list[Path]:
  return [path for path in paths if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                                        node.func.attr == attribute for node in ast.walk(_tree(path)))]


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


def test_llm_uses_no_legacy_custom_kernel_transport():
  root = _repo_root()
  llm_sources = _python_sources(root / "tinygrad" / "llm")
  callers = _attribute_callers(llm_sources, "custom_kernel")
  assert not callers, "production LLM source must not call the removed Tensor .custom_kernel API"


def test_llm_uop_program_transport_is_confined_to_kernel_program_boundary():
  root = _repo_root()
  llm_sources = _python_sources(root / "tinygrad" / "llm")
  callers = _attribute_callers(llm_sources, "uop_program")
  assert _relative(root, callers) == [str(_BOUNDARY)], "direct .uop_program calls must stay inside the kernel-program boundary"


def test_production_llm_does_not_import_or_call_oracle_or_research_execution():
  root = _repo_root()
  production_sources = [path for path in _python_sources(root / "tinygrad" / "llm")
                        if path.relative_to(root) != _BOUNDARY
                        and path.relative_to(root) not in _RESEARCH_ADMISSION_BOUNDARIES]
  assert not _restricted_executor_users(production_sources), (
    "production LLM source may use only execute_promoted_program; oracle/research execution belongs "
    "outside production except in the closed-default research-admission boundaries")


def test_research_runtime_sources_do_not_call_legacy_custom_kernel_directly():
  root = _repo_root()
  research_sources = _python_sources(root / "extra" / "llm_research", exclude_tests=True)
  callers = _attribute_callers(research_sources, "custom_kernel")
  assert not callers, "research runtime, benchmark, qualification, and campaign sources must use typed program execution"
