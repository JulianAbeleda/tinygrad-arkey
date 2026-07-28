import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

from extra.usbgpu.tools.with_gpu_lock import LOCK_UNAVAILABLE_EXIT, LOCK_SCHEMA, run_locked


def _script(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
  path = tmp_path / "child.py"
  path.write_text(body)
  return path


def _command(script: pathlib.Path, *args: str) -> list[str]:
  return [sys.executable, str(script), *args]


def test_child_receives_metadata_and_inherited_lock_fd(tmp_path):
  lock = tmp_path / "gpu.lock"
  observed = tmp_path / "observed.json"
  child = _script(tmp_path, """
import json, os, pathlib
fd = int(os.environ['TINYGRAD_GPU_LOCK_FD'])
payload = json.loads(pathlib.Path(os.environ['TINYGRAD_GPU_LOCK_PATH']).read_text())
pathlib.Path(os.sys.argv[1]).write_text(json.dumps({
  'fd_open': os.fstat(fd).st_ino > 0,
  'nonce_matches': payload['nonce'] == os.environ['TINYGRAD_GPU_LOCK_NONCE'],
  'schema': payload['schema'],
  'argv': payload['argv'],
}))
""")
  command = _command(child, str(observed))
  assert run_locked(command, lock_path=lock) == 0
  got = json.loads(observed.read_text())
  assert got == {"fd_open": True, "nonce_matches": True, "schema": LOCK_SCHEMA, "argv": command}
  assert json.loads(lock.read_text()) == {}


def test_nonblocking_contention_reports_tempfail_and_keeps_holder_metadata(tmp_path):
  lock = tmp_path / "gpu.lock"
  ready = tmp_path / "ready"
  child = _script(tmp_path, """
import pathlib, sys, time
pathlib.Path(sys.argv[1]).touch()
time.sleep(3)
""")
  holder = subprocess.Popen(_command(pathlib.Path(__file__).parents[2] / "extra/usbgpu/tools/with_gpu_lock.py",
                                     "--lock-path", str(lock), "--", sys.executable, str(child), str(ready)))
  try:
    deadline = time.monotonic() + 2
    while not ready.exists() and time.monotonic() < deadline: time.sleep(0.01)
    assert ready.exists()
    assert run_locked([sys.executable, "-c", "raise SystemExit(0)"], lock_path=lock) == LOCK_UNAVAILABLE_EXIT
    assert json.loads(lock.read_text())["schema"] == LOCK_SCHEMA
  finally:
    holder.terminate()
    assert holder.wait(timeout=5) == 143


def test_bounded_wait_acquires_after_holder_exits(tmp_path):
  lock = tmp_path / "gpu.lock"
  child = _script(tmp_path, "import time; time.sleep(0.15)\n")
  holder = subprocess.Popen(_command(pathlib.Path(__file__).parents[2] / "extra/usbgpu/tools/with_gpu_lock.py",
                                     "--lock-path", str(lock), "--", sys.executable, str(child)))
  time.sleep(0.05)
  try:
    assert run_locked([sys.executable, "-c", "raise SystemExit(7)"], lock_path=lock, wait_s=2) == 7
  finally:
    holder.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="process groups and flock are POSIX-only")
def test_term_is_forwarded_to_child_process_group(tmp_path):
  lock = tmp_path / "gpu.lock"
  ready = tmp_path / "ready"
  child = _script(tmp_path, """
import pathlib, signal, sys, time
def stopped(signum, frame):
  pathlib.Path(sys.argv[2]).write_text(str(signum))
  raise SystemExit(0)
signal.signal(signal.SIGTERM, stopped)
pathlib.Path(sys.argv[1]).touch()
while True: time.sleep(1)
""")
  runner = pathlib.Path(__file__).parents[2] / "extra/usbgpu/tools/with_gpu_lock.py"
  stopped = tmp_path / "stopped"
  proc = subprocess.Popen(_command(runner, "--lock-path", str(lock), "--", sys.executable, str(child), str(ready), str(stopped)))
  try:
    deadline = time.monotonic() + 2
    while not ready.exists() and time.monotonic() < deadline: time.sleep(0.01)
    assert ready.exists()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=5) == 0
    assert stopped.read_text() == str(signal.SIGTERM)
  finally:
    if proc.poll() is None: proc.kill()
