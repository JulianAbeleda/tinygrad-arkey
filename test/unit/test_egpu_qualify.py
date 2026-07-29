import copy, importlib.util, json, os, pathlib, subprocess
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("qualify", ROOT / "extra/usbgpu/tests/qualify.py")
qualify = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(qualify)
STATUS = json.loads((ROOT / "extra/usbgpu/protocol/fixtures/keepalive-status-v1.json").read_text())["valid"]
POWER_STATUS = json.loads((ROOT / "extra/usbgpu/protocol/fixtures/power-residency-status-v2.json").read_text())["valid"]


def status(ticks=0, *, generation=1, **patch):
  return copy.deepcopy(STATUS) | {"provider_generation":generation, "attempts":3600+ticks, "successes":3600+ticks,
                                  "last_attempt_monotonic_ns":3600000000000+ticks*1000000000,
                                  "last_success_monotonic_ns":3600000000000+ticks*1000000000} | patch


def power(ticks=0, *, generation=1, **patch):
  return copy.deepcopy(POWER_STATUS) | {"provider_generation":generation, "last_transition_monotonic_ns":1100000000,
                                       "last_canary_success_monotonic_ns":3600000000000+ticks*1000000000} | patch


def test_lock_metadata_requires_matching_inherited_descriptor_and_parent(tmp_path):
  lock = tmp_path / "lock"
  payload = {"schema":"tinygrad.gpu.lock.v1", "nonce":"ok", "pid":123}
  lock.write_text(json.dumps(payload))
  with pytest.raises(qualify.QualificationError): qualify.validate_lock({})
  with open(lock) as handle:
    got = qualify.validate_lock({"TINYGRAD_GPU_LOCK_FD":str(handle.fileno()), "TINYGRAD_GPU_LOCK_PATH":str(lock),
                                 "TINYGRAD_GPU_LOCK_NONCE":"ok"}, parent_pid=123)
  assert got == payload
  other = tmp_path / "other"; other.write_text(json.dumps(payload))
  with open(other) as handle, pytest.raises(qualify.QualificationError, match="does not name"):
    qualify.validate_lock({"TINYGRAD_GPU_LOCK_FD":str(handle.fileno()), "TINYGRAD_GPU_LOCK_PATH":str(lock),
                           "TINYGRAD_GPU_LOCK_NONCE":"ok"}, parent_pid=123)


def test_environment_is_exact_not_implicit():
  assert qualify.validate_environment(qualify.REQUIRED_ENV) == qualify.REQUIRED_ENV
  with pytest.raises(qualify.QualificationError, match="environment mismatch"):
    qualify.validate_environment(qualify.REQUIRED_ENV | {"DEV":"CPU"})
  with pytest.raises(qualify.QualificationError, match="forbids"):
    qualify.validate_environment(qualify.REQUIRED_ENV | {"REMOTE_KEEPALIVE_S":"1"})


def test_a1_terms_only_exact_server_and_proves_socket_absence(tmp_path):
  app = tmp_path / "TinyGPU"; app.touch()
  samples = iter([status(), status(120)])
  power_samples = iter([power(), power(120)])
  killed = []
  state = [(1, [str(app), "server", "sock"]), (2, [str(app), "serverish"])]
  def processes(): return [row for row in state if row[0] not in killed]
  result = qualify.run_gate("A1", status_reader=samples.__next__, power_status_reader=power_samples.__next__, endpoint_reader=lambda:True, process_reader=processes,
                            terminator=killed.append, sleeper=lambda _:None, installed_executable=app, socket_reader=lambda:False)
  assert result["status"] == "passed" and killed == [1] and len(result["samples"]) == 2


def test_a1_fails_if_socket_remains_reachable(tmp_path):
  app = tmp_path / "TinyGPU"; app.touch()
  result = qualify.run_gate("A1", status_reader=lambda:status(), endpoint_reader=lambda:True, process_reader=lambda:[],
                            installed_executable=app, socket_reader=lambda:True)
  assert result["status"] == "failed" and "socket remained reachable" in result["first_failure"]["message"]


@pytest.mark.parametrize("patch", [
  {"active_dma_allocations":1}, {"state":"active_degraded"}, {"enabled":False}, {"timer_error":1},
  {"last_identity_dword":"0x00000000"}, {"failures":1, "successes":3599},
])
def test_unhealthy_status_is_first_failure(patch):
  result = qualify.run_gate("A2", status_reader=lambda:status(**patch), endpoint_reader=lambda:True,
                            minimal_command=["minimal"], runner=lambda _:0)
  assert result["status"] == "failed" and result["first_failure"]["type"] == "QualificationError"


def test_endpoint_visibility_is_never_optional():
  result = qualify.run_gate("A2", status_reader=lambda:status(), endpoint_reader=lambda:False,
                            minimal_command=["minimal"], runner=lambda _:0)
  assert result["status"] == "failed" and "endpoint disappeared" in result["first_failure"]["message"]


def test_manual_replug_prompts_once_and_requires_new_generation():
  prompts=[]; samples=iter([status(), status(generation=2), status(1, generation=2)])
  result = qualify.run_gate("A9", status_reader=samples.__next__, endpoint_reader=lambda:True, manual_prompt=prompts.append,
                            minimal_command=["minimal"], runner=lambda _:0)
  assert result["status"] == "passed" and prompts == ["manual replug"] and result["manual_action_required"]


def test_atomic_json_replaces_complete_file_and_duplicate_json_is_rejected(tmp_path):
  out=tmp_path/"nested/result.json"; qualify.atomic_json(out,{"ok":True})
  assert json.loads(out.read_text()) == {"ok":True}
  with pytest.raises(qualify.QualificationError, match="malformed JSON"):
    qualify.decode_json(b'{"a":1,"a":2}', max_bytes=100)


def test_a3_runs_configured_churn_and_checks_each_boundary():
  samples=iter([status(), status(1), status(2)]); commands=[]
  result = qualify.run_gate("A3", status_reader=samples.__next__, endpoint_reader=lambda:True,
                            runner=lambda cmd:commands.append(cmd) or 0, minimal_command=["minimal"], churn_count=2, churn_idle_s=0)
  assert result["status"] == "passed" and commands == [["minimal"], ["minimal"]] and len(result["endpoint_checks"]) == 3


def test_a4_requires_and_runs_immediate_a5_under_same_clock():
  missing = qualify.run_gate("A4", status_reader=lambda:status(), endpoint_reader=lambda:True, idle_s=1)
  assert missing["status"] == "failed" and "immediate A5" in missing["first_failure"]["message"]
  now=[0]
  def sleep(seconds): now[0] += seconds
  samples=iter([status(), status(1), status(1)])
  result=qualify.run_gate("A4", status_reader=samples.__next__, endpoint_reader=lambda:True, sleeper=sleep, clock=lambda:now[0],
                          idle_s=1, sample_interval_s=1, include_post_idle=True, minimal_command=["minimal"], runner=lambda _:0)
  assert result["status"] == "passed" and result["A5"] == "passed" and len(result["samples"]) == 3


def test_a8_composes_internal_a2_and_a6_checks(tmp_path):
  model = tmp_path / "model.gguf"; model.write_bytes(b"model")
  samples=iter([status(i) for i in range(6)]); commands=[]; now=[0]
  result = qualify.run_gate("A8", status_reader=samples.__next__, endpoint_reader=lambda:True, model=model, idle_s=1,
                            minimal_command=["minimal"], bench_command=["bench"], runner=lambda cmd:commands.append(cmd) or 0,
                            clock=lambda:now[0], sleeper=lambda _:now.__setitem__(0, 1))
  assert result["status"] == "passed" and len(commands) == 2 and "--prefill" in commands[1]


def test_a0_requires_direct_handshake_and_validates_it(tmp_path):
  app=tmp_path/"TinyGPU"; app.touch()
  missing=qualify.run_gate("A0", status_reader=lambda:status(), endpoint_reader=lambda:True, installed_executable=app)
  assert missing["status"] == "failed"
  hello={"schema":"tinygpu.handshake.v1", "protocol_major":1, "protocol_minor":0, "capabilities":11, "server_build_id":"test"}
  provenance={"schema":"tinygpu.development-install.provenance.v1", "source_commit":"abc"}
  result=qualify.run_gate("A0", status_reader=lambda:status(), power_status_reader=lambda:power(), endpoint_reader=lambda:True,
                          installed_executable=app, handshake_reader=lambda:hello, install_provenance=provenance)
  assert result["status"] == "passed" and result["provenance"]["handshake"] == hello


@pytest.mark.parametrize("patch", [
  {"override_probe_prejoin_error":0}, {"override_probe_postjoin_error":-536870212}, {"power_request_confirmed":False},
  {"last_observed_power_flags":65536}, {"last_canary_success_monotonic_ns":1000000000},
  {"unexpected_downgrade_count":1}, {"publishable":False},
])
def test_a0_rejects_unhealthy_power_residency(patch, tmp_path):
  app=tmp_path/"TinyGPU"; app.touch()
  hello={"schema":"tinygpu.handshake.v1", "protocol_major":1, "protocol_minor":0, "capabilities":11, "server_build_id":"test"}
  result=qualify.run_gate("A0", status_reader=lambda:status(), power_status_reader=lambda:power(**patch), endpoint_reader=lambda:True,
                          installed_executable=app, handshake_reader=lambda:hello, install_provenance={"schema":"test"})
  assert result["status"] == "failed" and "power" in result["first_failure"]["message"]


def test_install_provenance_binds_live_binary_hashes_and_commit(tmp_path):
  app_root=tmp_path/"TinyGPU.app"; app=app_root/"Contents/MacOS/TinyGPU"
  dext=app_root/"Contents/Library/SystemExtensions/org.tinygrad.arkey.tinygpu.driver2.dext/org.tinygrad.arkey.tinygpu.driver2"
  app.parent.mkdir(parents=True); dext.parent.mkdir(parents=True)
  app.write_bytes(b"app"); dext.write_bytes(b"dext")
  transcript=tmp_path/"install.txt"
  transcript.write_text("\n".join((
    "schema=tinygpu.development-install.provenance.v1", "source_commit=abc", "=== activated ===",
    "Identifier=org.tinygrad.arkey.tinygpu.installer", "Identifier=org.tinygrad.arkey.tinygpu.driver2",
    "Signature=adhoc", "[activated enabled]", f"{qualify.sha256(app)}  {app.resolve()}", f"{qualify.sha256(dext)}  {dext.resolve()}",
  )))
  got=qualify.validate_install_provenance(transcript, app, source_commit="abc")
  assert got["source_commit"] == "abc" and got["app_sha256"] == qualify.sha256(app)
  app.write_bytes(b"replaced")
  with pytest.raises(qualify.QualificationError, match="does not match"):
    qualify.validate_install_provenance(transcript, app, source_commit="abc")


def test_acceptance_cli_has_no_test_shortcuts_or_endpoint_override():
  result=subprocess.run([os.sys.executable, str(ROOT/"extra/usbgpu/tests/qualify.py"), "--help"], check=True, capture_output=True, text=True)
  for forbidden in ("--idle-s", "--decode-duration-s", "--endpoint-command", "--status-command", "--out-dir"):
    assert forbidden not in result.stdout


def test_classification_records_workload_failure_with_status_evidence(tmp_path):
  model=tmp_path/"model.gguf"; model.write_bytes(b"model"); samples=iter([status(), status(1)])
  result=qualify.run_gate("A10", status_reader=samples.__next__, endpoint_reader=lambda:True, model=model,
                          bench_command=["bench"], runner=lambda _:1)
  assert result["status"] == "recorded" and result["first_failure"] is None
  assert result["classification"]["workload_returncode"] == 1 and len(result["samples"]) == 2


def test_minimal_subprocess_output_must_be_exact():
  samples=iter([status(), status()])
  bad=subprocess.CompletedProcess(["minimal"], 0, b"[2, 5, 10, 17]\n", b"")
  result=qualify.run_gate("A2", status_reader=samples.__next__, endpoint_reader=lambda:True,
                          minimal_command=["minimal"], runner=lambda _:bad)
  assert result["status"] == "failed" and "exact four-value" in result["first_failure"]["message"]
