"""The pre-commit hook judges what a commit authors (working agreement Rules 2 and 3).

A merge re-judges nothing already committed on either branch; a file the merge itself changes is checked like
any commit. These run the real .githooks/pre-commit in a throwaway repository.
"""
import os, pathlib, shutil, subprocess

import pytest

HOOK = pathlib.Path(__file__).resolve().parents[2] / ".githooks" / "pre-commit"
HARNESS = "extra/llm_research/decode/nv_example_ab.py"


def _git(repo, *args, check=True):
  # Drop GIT_* so a caller running inside a git hook cannot point these commands at the outer repository.
  env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
  return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check, env=env)


def _stage(repo, path, text):
  target = repo / path
  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(text)
  _git(repo, "add", path)


@pytest.fixture
def repo(tmp_path):
  if shutil.which("git") is None: pytest.skip("git is not installed")
  _git(tmp_path, "init", "-q", "-b", "exp")
  for key, value in (("user.name", "test"), ("user.email", "test@example.com"), ("commit.gpgsign", "false")):
    _git(tmp_path, "config", key, value)
  _stage(tmp_path, "README", "base\n")
  _git(tmp_path, "commit", "-q", "-m", "base")
  # A harness without the contract, committed on another branch before the hook is installed.
  _git(tmp_path, "checkout", "-q", "-b", "side")
  _stage(tmp_path, HARNESS, "print('no contract')\n")
  _git(tmp_path, "commit", "-q", "-m", "side harness")
  _git(tmp_path, "checkout", "-q", "exp")
  _stage(tmp_path, "notes.txt", "exp moved on\n")
  _git(tmp_path, "commit", "-q", "-m", "exp work")
  hooks = tmp_path.parent / (tmp_path.name + "-hooks")
  hooks.mkdir()
  shutil.copy2(HOOK, hooks / "pre-commit")
  _git(tmp_path, "config", "core.hooksPath", str(hooks))
  return tmp_path


def _commit(repo, message):
  return _git(repo, "commit", "-q", "-m", message, check=False)


def test_a_merge_does_not_rejudge_a_harness_committed_on_the_other_branch(repo):
  _git(repo, "merge", "--no-ff", "--no-commit", "side")
  out = _commit(repo, "Merge side")
  assert out.returncode == 0, out.stderr


def test_a_merge_that_changes_the_harness_itself_is_judged(repo):
  _git(repo, "merge", "--no-ff", "--no-commit", "side")
  _stage(repo, HARNESS, "print('edited in the merge, still no contract')\n")
  out = _commit(repo, "Merge side")
  assert out.returncode == 1 and "Rule 3" in out.stderr


def test_an_ordinary_commit_of_a_harness_without_the_contract_is_refused(repo):
  _stage(repo, "extra/llm_research/decode/nv_other_ab.py", "print('no contract')\n")
  out = _commit(repo, "new harness")
  assert out.returncode == 1 and "Rule 3" in out.stderr


@pytest.mark.parametrize("in_merge", [False, True])
def test_code_without_a_test_is_refused_in_a_commit_and_in_a_merge(repo, in_merge):
  if in_merge: _git(repo, "merge", "--no-ff", "--no-commit", "side")
  _stage(repo, "tinygrad/example.py", "x = 1\n")
  out = _commit(repo, "code only")
  assert out.returncode == 1 and "Rule 2" in out.stderr
