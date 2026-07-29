"""Production LLM CLI boundary: the shipped entrypoint must not depend on extra/llm runtime code."""
import os, subprocess, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_llm_help_uses_tinygrad_cli_owner():
  env = {**os.environ, "PYTHONPATH": ROOT, "PYTHONDONTWRITEBYTECODE": "1"}
  proc = subprocess.run([sys.executable, "-m", "tinygrad.llm", "--help"], cwd=ROOT, env=env,
                        text=True, capture_output=True)
  assert proc.returncode == 0, proc.stderr
  assert "usage" in proc.stdout.lower()

def test_generation_and_adapter_modules_are_core_owned():
  pytest.importorskip("numpy")
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.generate import load_model_and_tokenizer
  from tinygrad.llm.adapter import expand_lora_targets
  assert SimpleTokenizer and load_model_and_tokenizer and expand_lora_targets
