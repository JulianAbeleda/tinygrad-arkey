import pathlib
import pytest

from extra.qk.decode.run_manifest import collect_run_manifest, validate_manifest

def _run(argv, cwd):
  answers = {("git", "status", "--porcelain=v1"): " M tracked.py\n?? new.py",
             ("git", "branch", "--show-current"): "feature/test",
             ("git", "rev-parse", "HEAD"): "abc123"}
  return answers[tuple(argv)]

def test_collect_records_known_worktree_dirty_path_and_command(tmp_path, monkeypatch):
  model = tmp_path / "model.gguf"; model.write_bytes(b"known model")
  original_read_text = pathlib.Path.read_text
  monkeypatch.setattr(pathlib.Path, "read_text", lambda path, *args, **kwargs:
                      "known-boot\n" if path == pathlib.Path("/proc/sys/kernel/random/boot_id")
                      else original_read_text(path, *args, **kwargs))
  manifest = collect_run_manifest(task_id="LUNA-003", command_argv=["llama-bench", "-ngl", "99"], model_path=str(model),
    backend="HIP", device="AMD Radeon", architecture="gfx1100", positive_control={"known_command": True}, classification="PASS",
    stdout_path="stdout.log", stderr_path="stderr.log", primary_artifact_path="artifact.json", kernel_or_route_identity="known-route",
    notes="fixture", worktree=str(tmp_path), power_before={"state": "auto"}, power_after={"state": "auto"}, run=_run)
  assert manifest["branch"] == "feature/test"
  assert manifest["git_dirty_paths"] == ["tracked.py", "new.py"]
  assert manifest["command_argv"] == ["llama-bench", "-ngl", "99"]
  assert manifest["worktree"] == str(tmp_path.resolve())

def test_manifest_rejects_missing_positive_controls():
  with pytest.raises(ValueError, match="positive_control"):
    validate_manifest({key: "x" for key in ("schema", "task_id", "created_unix_ns", "branch", "commit", "worktree", "git_dirty_paths", "command_argv", "environment_overrides", "model_path", "model_size_bytes", "model_mtime_ns", "model_identity_sha256", "backend", "device", "architecture", "boot_id", "lock_path", "lock_owner_pid", "power_before", "power_after", "start_time", "end_time", "exit_code", "classification", "positive_control", "stdout_path", "stderr_path", "primary_artifact_path", "kernel_or_route_identity", "notes")})
