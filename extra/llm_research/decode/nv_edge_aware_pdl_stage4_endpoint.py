#!/usr/bin/env python3
"""Stage 4 endpoint bracket driver for the NV edge-aware PDL runtime hook.

Each arm runs a fresh-process control/candidate/control bracket.  Every child
run is serialized under ``flock -w 120 /tmp/gpu-bench.lock`` and launched with
``timeout`` so a hung or faulting probe is stopped and recorded verbatim
instead of blocking the queue.  The driver itself never touches the GPU or
holds the lock; it only serializes children and validates the evidence.

The child worker reuses the Phase B capture harness
(``nv_pdl_phase_b_probe``) for the identical model, token corpus, checksum
authority, and HCQ graph-profile timeline, then computes the endpoint ledger
with the same interval semantics as ``cuda_graph_timeline_ledger`` and the
anchor census from ``nv_inter_anchor_analysis``:

- ``s1_exposure_us``: per-layer O.start - Q.end, summed over anchor layers;
- ``union_us``: interval union of all timed node [start, end] intervals;
- ``overlap_mass_us``: node sum minus interval union;
- ``dead_device_time_us``: token span minus interval union (internal gaps);
- ``wall_us``: median host wall per token (PROFILE=1 instrumentation tax
  included; never presented as an unprofiled wall);
- ``token_sha256``: SHA-256 of the sampled token-id sequence;
- ``lock_session_id``: driver-generated bracket session id plus child pid.

Every numeric field is labeled ``observed``, ``inferred``, or ``unmeasured``.
No endpoint run is authorized until the Stage 2 gate evidence passes:
``--gate-evidence`` is checked before any child is spawned and the bracket is
refused when the verdict is missing, refuted, or not clean.

Arm matrix (section 7 of the scope doc):

1. ``off``                      -- NV_SPLIT_PHASE unset (baseline)
2. ``edge_1q_end_entry``        -- NV_SPLIT_PHASE=1, trigger=end, wait=entry, 1 queue
3. ``edge_2q_end_entry``        -- same, 2 queues
4. ``edge_2q_start_entry``      -- trigger=start, wait=entry, 2 queues
5. ``edge_2q_end_prologue``     -- trigger=end, wait=prologue (needs wait_anchor), 2 queues
6. ``one_graph_best_prior``     -- GRAPH_ONE_KERNEL=1 plus edge arm, own control
7. ``cuda_matched_grid``        -- placeholder; named-unavailable until the
   Stage 2/Phase C matched-grid data is linked.  No endpoint claim.

Usage (driver spawns children; the driver itself is CPU-only):

    python extra/llm_research/decode/nv_edge_aware_pdl_stage4_endpoint.py \
      --gate-evidence docs/task_workflow/evidence/nv-edge-aware-pdl-runtime-hook-20260821/stage2_semantic_gate.json
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, subprocess, sys, time, uuid

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

SCHEMA = "tinygrad.nv_edge_aware_pdl_stage4_endpoint.v1"
DEFAULT_MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
LOCK = "/tmp/gpu-bench.lock"
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.exists():
  PYTHON = pathlib.Path(sys.executable)

S1_RECOVERY_US = 150.0
DEFAULT_TIMEOUT_S = 1200

SEMANTIC_GATE_SCHEMA = "tinygrad.nv_edge_aware_pdl_stage2_semantic.v1"
CAPABILITY_GATE_SCHEMA = "tinygrad.nv_edge_aware_pdl_stage2_capability.v1"
PASSED_VERDICTS = {"passed", "pass", "true", "ok"}

PRODUCERS = ("prefix:q4k_", "prefix:q6k_")
CONSUMERS = ("prefix:reduce_output_rmsnorm", "prefix:E_", "prefix:r_",
             "prefix:flash_", "prefix:rmsnorm_q8_1_llama_provider")

# Arm spec table: control runs always use the `off` env (NV_SPLIT_PHASE unset).
ARMS: dict[str, dict] = {
  "off": {"queues": 2, "split": None, "trigger": "end", "wait": "entry", "one_graph": False},
  "edge_1q_end_entry": {"queues": 1, "split": "1", "trigger": "end", "wait": "entry", "one_graph": False},
  "edge_2q_end_entry": {"queues": 2, "split": "1", "trigger": "end", "wait": "entry", "one_graph": False},
  "edge_2q_start_entry": {"queues": 2, "split": "1", "trigger": "start", "wait": "entry", "one_graph": False},
  "edge_2q_end_prologue": {"queues": 2, "split": "1", "trigger": "end", "wait": "prologue", "one_graph": False},
  "one_graph_best_prior": {"queues": 2, "split": "1", "trigger": "end", "wait": "entry", "one_graph": True},
  "cuda_matched_grid": {"placeholder": True},
}

SPLIT_KEYS = ("NV_SPLIT_PHASE", "NV_SPLIT_PHASE_TRIGGER_POLICY", "NV_SPLIT_PHASE_WAIT_POSITION",
              "NV_SPLIT_PHASE_POLICY", "NV_SPLIT_PHASE_CENSUS_JSON", "NV_SPLIT_PHASE_CENSUS_ONLY",
              "NV_SPLIT_PHASE_LATCH_BASE", "NV_SPLIT_PHASE_LATCH_COUNT")
LEGACY_PDL_KEYS = ("NV_PDL_PRODUCER_PROGRAMS", "NV_PDL_CONSUMER_PROGRAMS",
                   "NV_PDL_TRIGGER_POSITION", "NV_PDL_LATCH_ID")
RUN_KEYS = SPLIT_KEYS + LEGACY_PDL_KEYS + ("GRAPH_ONE_KERNEL", "JIT_BATCH_SIZE",
                                           "HCQ_GRAPH_PROFILE_JSON", "HCQ_NUM_COMPUTE",
                                           "PROFILE", "DEV", "NV_STAGE4_LOCK_SESSION")


def _git_commit() -> str:
  return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                        capture_output=True, text=True).stdout.strip()


def _atomic_json(path: pathlib.Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(f".{path.name}.tmp")
  tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  tmp.replace(path)


def _append_jsonl(path: pathlib.Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")


def _median(values: list[float]) -> float | None:
  return round(statistics.median(values), 3) if values else None


def L(value: float | int | None, label: str = "observed") -> dict:
  """Label a number: observed / inferred / unmeasured.  None -> unmeasured."""
  if value is None:
    return {"value": None, "label": "unmeasured"}
  return {"value": round(float(value), 3), "label": label}


def _interval_union(ivs: list[tuple[float, float]]) -> float:
  merged: list[list[float]] = []
  for start, end in sorted(ivs):
    if end < start:
      raise ValueError(f"interval end precedes start: {start} > {end}")
    if not merged or start > merged[-1][1]:
      merged.append([start, end])
    elif end > merged[-1][1]:
      merged[-1][1] = end
  return round(sum(b - a for a, b in merged), 3)


def token_sha(token_ids: list[int]) -> str:
  return hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()


def _bracket_session() -> str:
  return f"{int(time.time())}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Arm environment assembly (pure CPU)
# ---------------------------------------------------------------------------

def arm_env(arm: str, role: str, policy_path: pathlib.Path | None,
            census_path: pathlib.Path | None, profile_jsonl: pathlib.Path,
            lock_session: str, queues: int | None = None,
            prologue_anchor: str | None = None) -> tuple[dict[str, str], str | None]:
  """Return ``(env_vars, unavailable_reason)`` for one child run.

  ``role`` is ``control`` or ``candidate``.  Controls always use the ``off``
  environment.  A candidate that the current construction cannot express
  returns a named reason instead of env vars; the caller records the arm as
  named-unavailable and does not spawn a child.
  """
  spec = ARMS[arm]
  env: dict[str, str] = {
    "DEV": "NV",
    "PROFILE": "1",
    "HCQ_GRAPH_PROFILE_JSON": str(profile_jsonl),
    "NV_STAGE4_LOCK_SESSION": lock_session,
  }
  q = queues if queues is not None else int(spec.get("queues", 2))
  env["HCQ_NUM_COMPUTE"] = str(q)

  if role == "control":
    # A one-graph candidate needs a one-graph control: the grouping knob is
    # part of the arm under test, not part of the split-phase candidate.
    if spec.get("one_graph"):
      env["GRAPH_ONE_KERNEL"] = "1"
    return env, None
  if spec.get("split") is None:
    return env, None

  if spec.get("placeholder"):
    return env, "placeholder arm; no endpoint run defined"

  trigger = str(spec["trigger"])
  wait = str(spec["wait"])
  if wait == "prologue" and not prologue_anchor:
    return env, "prologue wait_anchor not configured; stage0 plan supplies semantic anchors only, no literal rendered-source anchor"
  if policy_path is None:
    return env, "policy JSON path not provided"
  if census_path is None:
    return env, "census JSON path not provided"

  env["NV_SPLIT_PHASE"] = "1"
  env["NV_SPLIT_PHASE_TRIGGER_POLICY"] = trigger
  env["NV_SPLIT_PHASE_WAIT_POSITION"] = wait
  env["NV_SPLIT_PHASE_POLICY"] = str(policy_path)
  env["NV_SPLIT_PHASE_CENSUS_JSON"] = str(census_path)
  if spec.get("one_graph"):
    env["GRAPH_ONE_KERNEL"] = "1"
  return env, None


# ---------------------------------------------------------------------------
# Policy JSON generation (pure CPU, validated with the renderer loader)
# ---------------------------------------------------------------------------

def build_policy(arm: str, commit: str, trigger: str, wait: str,
                 prologue_anchor: str | None = None) -> dict:
  """Build a ``tinygrad.nv_split_phase_policy.v1`` policy body for the arm."""
  from nv_edge_aware_pdl_render_policy import POLICY_SCHEMA
  if wait == "prologue" and not prologue_anchor:
    raise ValueError("prologue wait requires a literal wait_anchor")
  programs = []
  for match in (*PRODUCERS, *CONSUMERS):
    prog: dict = {"match": match, "wait_position": wait, "trigger_policy": trigger}
    if wait == "prologue":
      prog["wait_anchor"] = prologue_anchor
    programs.append(prog)
  return {"schema": POLICY_SCHEMA, "commit": commit, "programs": programs}


def write_policy(arm: str, trigger: str, wait: str, evidence_dir: pathlib.Path,
                 prologue_anchor: str | None = None) -> tuple[pathlib.Path, str | None]:
  """Validate and write the arm policy JSON.  Returns (path, unavailable_reason)."""
  from nv_edge_aware_pdl_render_policy import _validate_policy
  if wait == "prologue" and not prologue_anchor:
    return evidence_dir / f"stage4_policy_{arm}.json", (
      "prologue wait_anchor not configured; no literal rendered-source anchor available")
  policy = build_policy(arm, _git_commit(), trigger, wait, prologue_anchor)
  path = evidence_dir / f"stage4_policy_{arm}.json"
  _atomic_json(path, policy)
  _validate_policy(json.loads(path.read_text(encoding="utf-8")), str(path))
  return path, None


# ---------------------------------------------------------------------------
# Stage 2 gate precheck (pure CPU)
# ---------------------------------------------------------------------------

def check_gate(path: pathlib.Path) -> tuple[bool, str]:
  """Verify the Stage 2 gate JSON verdict.  Returns (accepted, reason).

  Accepts either the semantic gate
  (``tinygrad.nv_edge_aware_pdl_stage2_semantic.v1``, verdict must be a
  passing value) or the capability gate
  (``tinygrad.nv_edge_aware_pdl_stage2_capability.v1``, every probe verdict
  must be ``supported``).  Anything else, including a missing file, wrong
  schema, a ``None`` verdict, or any refuted/named-unavailable probe, fails
  with a named reason and no endpoint run proceeds.
  """
  if path is None or not pathlib.Path(path).exists():
    return False, f"gate evidence not found: {path}"
  try:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as e:
    return False, f"gate evidence unreadable: {e}"
  schema = data.get("schema")
  if schema == SEMANTIC_GATE_SCHEMA:
    verdict = data.get("verdict")
    if not isinstance(verdict, dict) and verdict is None:
      verdict = (data.get("semantics") or {}).get("verdict")
    if isinstance(verdict, dict):
      result = verdict.get("result")
    else:
      # Accept a bare passing verdict string for hand-written gate files.
      result = verdict
    if result is None:
      return False, "semantic gate verdict result is None (results recorded, verdict not set)"
    if str(result).lower() in PASSED_VERDICTS:
      return True, f"semantic gate passed (result={result!r})"
    return False, f"semantic gate result {result!r} is not a passing value"
  if schema == CAPABILITY_GATE_SCHEMA:
    probes = data.get("probes") or {}
    if not isinstance(probes, dict):
      return False, "capability gate 'probes' is not an object"
    bad = {name: spec.get("verdict") for name, spec in sorted(probes.items())
           if spec.get("verdict") != "supported"}
    if bad:
      return False, f"capability gate not clean: probes not fully supported: {bad}"
    return True, "capability gate clean (every probe verdict is supported)"
  return False, f"gate schema {schema!r} is not a Stage 2 semantic/capability gate"


# ---------------------------------------------------------------------------
# Ledger computation (pure CPU; unit-tested without a device)
# ---------------------------------------------------------------------------

def compute_ledger(nodes: list[dict]) -> dict:
  """Endpoint ledger from a node timeline (start_us/end_us per node)."""
  ivs = [(float(n["start_us"]), float(n["end_us"])) for n in nodes]
  node_sum = round(sum(b - a for a, b in ivs), 3)
  union = _interval_union(ivs)
  span = round(max(b for _, b in ivs) - min(a for a, _ in ivs), 3)
  return {
    "node_count": {"value": len(nodes), "label": "observed"},
    "node_sum_us": L(node_sum),
    "union_us": L(union),
    "overlap_mass_us": L(round(node_sum - union, 3)),
    "dead_device_time_us": L(round(span - union, 3)),
    "span_us": L(span),
  }


def unmeasured_ledger() -> dict:
  """Ledger shape for rows that produced no timeline (unavailable/fault rows)."""
  return {
    "node_count": {"value": None, "label": "unmeasured"},
    "node_sum_us": L(None), "union_us": L(None),
    "overlap_mass_us": L(None), "dead_device_time_us": L(None), "span_us": L(None),
  }


def compute_s1(dag_nodes: list[dict], timeline: list[dict]) -> dict:
  """Per-layer S1 exposure: O.start - Q.end summed over matched layers."""
  from nv_inter_anchor_analysis import tg_anchor_ids
  anchors = tg_anchor_ids({"nodes": dag_nodes})
  by_id = {int(n["id"]): n for n in timeline}
  q_ids = sorted(anchors["Q"])
  o_ids = sorted(anchors["O"])
  layers = min(len(q_ids), len(o_ids))
  if layers == 0:
    return {"layers": 0, "s1_exposure_us": L(None), "s1_positive_us": L(None),
            "note": "no matched Q/O anchor layers in timeline"}
  total = 0.0
  positive = 0.0
  missing = 0
  for qid, oid in zip(q_ids[:layers], o_ids[:layers]):
    q, o = by_id.get(int(qid)), by_id.get(int(oid))
    if q is None or o is None:
      missing += 1
      continue
    exp = float(o["start_us"]) - float(q["end_us"])
    total += exp
    positive += max(0.0, exp)
  label = "observed" if missing == 0 else "inferred"
  return {
    "layers": layers,
    "s1_exposure_us": L(total, label),
    "s1_positive_us": L(positive, label),
    "note": None if missing == 0 else f"{missing}/{layers} anchor pairs missing from timeline",
  }


def unmeasured_s1() -> dict:
  return {"layers": None, "s1_exposure_us": L(None), "s1_positive_us": L(None),
          "note": "no timeline captured; S1 unmeasured"}


# ---------------------------------------------------------------------------
# Child worker (runs under flock in a fresh process; GPU work lives here)
# ---------------------------------------------------------------------------

def _read_census(census_path: pathlib.Path) -> dict | None:
  if not census_path.exists():
    return None
  payloads = []
  for line in census_path.read_text(encoding="utf-8").splitlines():
    if line.strip():
      payloads.append(json.loads(line))
  if not payloads:
    return None
  summary: dict = {}
  by_graph: dict[int, dict] = {}
  for p in payloads:
    size = int(p.get("graph_size", -1))
    if size not in by_graph:
      by_graph[size] = {"graphs": 0, "rows": 0, "candidate_armed": 0, "reasons": {}}
    g = by_graph[size]
    g["graphs"] += 1
    g["rows"] += len(p.get("rows", []))
    for row in p.get("rows", []):
      reason = str(row.get("reason", "unknown"))
      g["reasons"][reason] = g["reasons"].get(reason, 0) + 1
      if reason == "candidate_armed":
        g["candidate_armed"] += 1
  summary["payloads"] = len(payloads)
  summary["by_graph_size"] = by_graph
  summary["total_candidate_armed"] = sum(g["candidate_armed"] for g in by_graph.values())
  return summary


def _timeline_nodes(dag: dict, records: list[dict]) -> list[dict]:
  """Match profile records to DAG groups by entry-count signature (last cycle)."""
  groups: list = []
  for node in dag.get("nodes", []):
    gid = node.get("group_id")
    if gid not in groups:
      groups.append(gid)
  sizes = [sum(1 for n in dag.get("nodes", []) if n.get("group_id") == g) for g in groups]
  run_len = len(groups)
  matches = [i for i in range(len(records) - run_len + 1)
             if [len(r.get("entries") or []) for r in records[i:i + run_len]] == sizes]
  if not matches:
    raise RuntimeError(f"no profile window matches DAG group signature {sizes}")
  start = matches[-1]
  raw: dict[int, tuple[float, float]] = {}
  token_start: float | None = None
  for gi, gid in enumerate(groups):
    members = sorted([n for n in dag.get("nodes", []) if n.get("group_id") == gid],
                     key=lambda n: int(n["id"]))
    entries = (records[start + gi].get("entries") or [])
    if len(entries) != len(members):
      raise RuntimeError(f"group {gid}: {len(entries)} profile entries for {len(members)} nodes")
    for idx, node in enumerate(members):
      s, e = float(entries[idx]["start"]), float(entries[idx]["end"])
      raw[int(node["id"])] = (s, e)
      token_start = s if token_start is None else min(token_start, s)
  nodes = []
  for node in dag.get("nodes", []):
    s, e = raw[int(node["id"])]
    nodes.append({"id": int(node["id"]), "name": str(node.get("name", "")),
                  "group_id": node.get("group_id"),
                  "start_us": round(s - token_start, 3), "end_us": round(e - token_start, 3)})
  return nodes


def run_child(args: argparse.Namespace) -> int:
  """Measurement worker; runs inside the lock held by the driver's flock."""
  import nv_pdl_phase_b_probe as pb
  from tinygrad import Device
  from tinygrad.helpers import Context

  from extra.llm_research.decode import full_token_dag_capture as ftc
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt

  for key in LEGACY_PDL_KEYS:
    os.environ.pop(key, None)
  os.environ["DEV"] = "NV"
  os.environ["PROFILE"] = "1"
  os.environ["HCQ_GRAPH_PROFILE_JSON"] = str(args.profile_jsonl)

  capture_args = argparse.Namespace(model=args.model, max_context=args.max_context,
                                    depth=args.depth, tokens=args.tokens)
  pb._install_probes()
  args.profile_jsonl.unlink(missing_ok=True)

  model = _load(args.model, args.max_context)
  gen = model.generate(_prompt(args.model, args.depth), chunk_size=32, temperature=0.0)
  started_wall = time.perf_counter()

  def harness() -> None:
    dev = Device["NV"]
    for _ in range(args.tokens):
      started = time.perf_counter()
      pb.PROBE["tokens"].append(int(next(gen)))
      pb.PROBE["per_token_host_us"].append((time.perf_counter() - started) * 1e6)
    dev.synchronize()
    with Context(DEBUG=0):
      pb.PROBE["tokens"].append(int(next(gen)))  # flush: exports the measured cycle
    dev.synchronize()

  dag: dict = {}
  original_select = ftc._select_dag
  ftc._select_dag = pb._select_decode_dag
  try:
    with ftc.capture_full_token_dag(harness) as captured:
      dag = captured
  finally:
    ftc._select_dag = original_select
    gen.close()
  wall_s = time.perf_counter() - started_wall

  records = pb._load_records(args.profile_jsonl)
  timeline = _timeline_nodes(dag, records)
  ledger = compute_ledger(timeline)
  s1 = compute_s1(dag["nodes"], timeline)
  census = _read_census(args.census_json) if args.census_json else None

  per_token = [round(x, 3) for x in pb.PROBE["per_token_host_us"]]
  row = {
    "schema": SCHEMA,
    "arm": args.arm,
    "role": args.role,
    "bracket_index": args.bracket_index,
    "commit": _git_commit(),
    "date": "2026-08-21",
    "device": "NV",
    "model": args.model,
    "depth": args.depth,
    "max_context": args.max_context,
    "env": {k: os.environ.get(k, "") for k in sorted(RUN_KEYS)},
    "lock": {"session_id": os.environ.get("NV_STAGE4_LOCK_SESSION", ""), "child_pid": os.getpid()},
    "wall_us": L(_median(per_token), "observed"),
    "wall_note": "PROFILE=1 instrumentation tax included; not an unprofiled token wall",
    "bracket_wall_s": L(wall_s),
    "token_sha256": token_sha(pb.PROBE["tokens"]),
    "token_count": {"value": len(pb.PROBE["tokens"]), "label": "observed"},
    "ledger": ledger,
    "s1": s1,
    "census": census,
    "profile_records_total": len(records),
    "timeline_node_count": len(timeline),
    "graph_group_signature": [sum(1 for n in dag.get("nodes", []) if n.get("group_id") == g)
                              for g in dict.fromkeys(n.get("group_id") for n in dag.get("nodes", []))],
  }
  _atomic_json(args.out, row)
  _append_jsonl(args.rows_jsonl, row)
  print(json.dumps({
    "arm": args.arm, "role": args.role, "index": args.bracket_index,
    "sha": row["token_sha256"][:12], "s1": row["s1"]["s1_exposure_us"]["value"],
    "union_us": row["ledger"]["union_us"]["value"],
    "wall_us": row["wall_us"]["value"], "census_armed": (census or {}).get("total_candidate_armed"),
  }, indent=2))
  return 0


# ---------------------------------------------------------------------------
# Driver: gate, policy files, spawn children, bracket validation
# ---------------------------------------------------------------------------

def _spawn_child(args: argparse.Namespace, arm: str, role: str, index: int,
                 lock_session: str, env_extra: dict[str, str],
                 out: pathlib.Path, profile_jsonl: pathlib.Path,
                 census_path: pathlib.Path | None) -> dict:
  """Spawn one fresh-process child under timeout + flock; never run in-place."""
  env = dict(os.environ)
  for key in RUN_KEYS:
    env.pop(key, None)
  env.update(env_extra)
  cmd = [
    "timeout", str(args.timeout), "flock", "-w", "120", LOCK,
    "env",
    str(PYTHON), str(pathlib.Path(__file__).resolve()),
    "--mode", "child",
    "--arm", arm, "--role", role, "--bracket-index", str(index),
    "--model", args.model, "--depth", str(args.depth),
    "--max-context", str(args.max_context), "--tokens", str(args.tokens),
    "--profile-jsonl", str(profile_jsonl),
    "--rows-jsonl", str(args.rows_jsonl),
    "--out", str(out),
  ]
  if census_path is not None:
    cmd += ["--census-json", str(census_path)]
  run = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if run.returncode:
    return {"fault": f"child rc={run.returncode}", "stderr_tail": run.stderr[-4000:],
            "stdout_tail": run.stdout[-1000:]}
  try:
    return json.loads(out.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as e:
    return {"fault": f"child output unreadable: {e}", "stderr_tail": run.stderr[-4000:]}


def _bracket_verdict(control_rows: list[dict], candidate_row: dict | None) -> tuple[str, str]:
  """Belief-flip gate from scope section 8, applied to the observed bracket."""
  if candidate_row is None:
    return "named-unavailable", "candidate run produced no row"
  if "fault" in candidate_row:
    return "named-unavailable", f"candidate fault: {candidate_row['fault']}"
  controls = [r for r in control_rows if "fault" not in r]
  if len(controls) < 2:
    return "named-unavailable", f"need two clean control rows, got {len(controls)}"
  try:
    base_s1 = _median([r["s1"]["s1_exposure_us"]["value"] for r in controls
                       if r["s1"]["s1_exposure_us"]["value"] is not None])
    cand_s1 = candidate_row["s1"]["s1_exposure_us"]["value"]
    base_wall = _median([r["wall_us"]["value"] for r in controls
                         if r["wall_us"]["value"] is not None])
    cand_wall = candidate_row["wall_us"]["value"]
  except KeyError:
    return "named-unavailable", "control/candidate rows missing ledger fields"
  if base_s1 is None or cand_s1 is None:
    return "unmeasured", "S1 exposure unavailable in control or candidate row"
  s1_delta = round(base_s1 - cand_s1, 3)
  wall_delta = round(cand_wall - base_wall, 3) if (cand_wall is not None and base_wall is not None) else None
  if s1_delta >= S1_RECOVERY_US and wall_delta is not None and wall_delta <= 0.0:
    return "supported", (f"S1 recovery {s1_delta}us >= {S1_RECOVERY_US}us with flat/negative "
                         f"wall ({wall_delta:+.3f}us); identical-token bracket")
  return "refuted", (f"S1 delta {s1_delta:+.3f}us (needs >= {S1_RECOVERY_US}us) "
                     f"and/or wall delta {wall_delta if wall_delta is not None else 'unmeasured'}us")


def run_driver(args: argparse.Namespace) -> int:
  gate_ok, gate_reason = check_gate(args.gate_evidence)
  if not gate_ok:
    print(f"GATE REFUSED: {gate_reason}", flush=True)
    return 2
  print(f"GATE ACCEPTED: {gate_reason}", flush=True)

  arms = [a for a in args.arms.split(",") if a] or list(ARMS)
  for arm in arms:
    if arm not in ARMS:
      raise SystemExit(f"unknown arm {arm!r}; choices: {','.join(ARMS)}")

  args.evidence_dir.mkdir(parents=True, exist_ok=True)
  brackets: dict[str, dict] = {}
  rows: list[dict] = []

  for arm in arms:
    spec = ARMS[arm]
    if spec.get("placeholder"):
      rows.append({
        "schema": SCHEMA, "arm": arm, "role": "candidate", "bracket_index": 1,
        "commit": _git_commit(), "date": "2026-08-21", "device": "NV",
        "verdict": "named-unavailable",
        "verdict_reason": "CUDA matched-grid new-hook comparison is a separate synthetic probe; "
                         "not linked until Stage 2/Phase C matched data is joined. No endpoint claim.",
        "wall_us": L(None), "token_sha256": None,
        "ledger": unmeasured_ledger(), "s1": unmeasured_s1(),
      })
      brackets[arm] = {"verdict": "named-unavailable", "reason": rows[-1]["verdict_reason"], "rows": [rows[-1]]}
      continue

    trigger = str(spec["trigger"])
    wait = str(spec["wait"])
    policy_path, policy_reason = write_policy(arm, trigger, wait, args.evidence_dir, args.prologue_anchor)
    lock_session = _bracket_session()
    bracket_rows: list[dict] = []
    parts: dict[str, dict] = {}
    for index, role in enumerate(("control", "candidate", "control")):
      suffix = f"stage4_{arm}_{role}_{index}.json"
      out = args.evidence_dir / suffix
      profile = args.evidence_dir / f"stage4_{arm}_{role}_{index}.profile.jsonl"
      census = args.evidence_dir / f"stage4_{arm}_{role}_{index}.census.jsonl"
      census.unlink(missing_ok=True)
      profile.unlink(missing_ok=True)
      env_extra, reason = arm_env(
        arm, role, policy_path if role == "candidate" else None,
        census if role == "candidate" else None, profile, lock_session,
        queues=args.queues, prologue_anchor=args.prologue_anchor)
      if reason is not None:
        row = {
          "schema": SCHEMA, "arm": arm, "role": role, "bracket_index": index,
          "commit": _git_commit(), "date": "2026-08-21", "device": "NV",
          "verdict": "named-unavailable", "verdict_reason": reason,
          "wall_us": L(None), "token_sha256": None,
          "ledger": unmeasured_ledger(), "s1": unmeasured_s1(),
        }
        bracket_rows.append(row)
        rows.append(row)
        parts[f"{role}_{index}"] = row
        print(f"{arm} {role}[{index}] UNAVAILABLE: {reason}", flush=True)
        continue
      row = _spawn_child(args, arm, role, index, lock_session, env_extra, out, profile,
                         census if role == "candidate" else None)
      bracket_rows.append(row)
      rows.append(row)
      parts[f"{role}_{index}"] = row
      if "fault" in row:
        print(f"{arm} {role}[{index}] FAULT rc={row['fault']}\n{row.get('stderr_tail','')[-2000:]}", flush=True)
      else:
        print(f"{arm} {role}[{index}] sha={row['token_sha256'][:12]} "
              f"s1={row['s1']['s1_exposure_us']['value']} wall={row['wall_us']['value']} "
              f"census_armed={(row.get('census') or {}).get('total_candidate_armed')}", flush=True)

    hashes = [r.get("token_sha256") for r in bracket_rows if "fault" not in r and r.get("token_sha256")]
    tokens_equal = bool(hashes) and len(set(hashes)) == 1
    candidate = parts.get("candidate_1")
    controls = [parts.get("control_0"), parts.get("control_2")]
    verdict, reason = _bracket_verdict(controls, candidate)
    brackets[arm] = {
      "verdict": verdict,
      "reason": reason,
      "tokens_equal": tokens_equal,
      "token_hashes": hashes,
      "policy": str(policy_path),
      "policy_unavailable_reason": policy_reason,
      "lock_session_id": lock_session,
      "rows": bracket_rows,
    }

  accepted = all(b["verdict"] in ("supported", "refuted") for b in brackets.values()) and \
             all(b["tokens_equal"] for b in brackets.values() if "tokens_equal" in b)
  payload = {
    "schema": SCHEMA,
    "commit": _git_commit(),
    "date": "2026-08-21",
    "device": "NV",
    "model": args.model,
    "gate": {"evidence": str(args.gate_evidence), "accepted": gate_ok, "reason": gate_reason},
    "verdict_policy": {
      "s1_recovery_us": S1_RECOVERY_US,
      "note": "scope section 8 belief-flip: S1 recovery >= 150us with flat/negative wall supports Direction A",
    },
    "brackets": {arm: {k: v for k, v in b.items() if k != "rows"} for arm, b in brackets.items()},
    "rows": rows,
    "accepted": accepted,
  }
  merged = args.evidence_dir / "stage4_endpoint_merged.json"
  _atomic_json(merged, payload)
  print(json.dumps({"accepted": accepted, "brackets": {
    arm: {"verdict": b["verdict"], "reason": b["reason"], "tokens_equal": b.get("tokens_equal")}
    for arm, b in brackets.items()}}, indent=2), flush=True)
  print(f"rows jsonl: {args.rows_jsonl}", flush=True)
  print(f"merged: {merged}", flush=True)
  return 0 if accepted else 1


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--mode", choices=("driver", "child"), default="driver")
  # Driver args
  ap.add_argument("--gate-evidence", type=pathlib.Path, default=None,
                  help="Stage 2 gate JSON (semantic or capability). Required in driver mode.")
  ap.add_argument("--arms", default="", help="comma-separated subset of arms; default all")
  ap.add_argument("--queues", type=int, default=None, help="override HCQ_NUM_COMPUTE for all runs")
  ap.add_argument("--prologue-anchor", default=None,
                  help="literal rendered-source substring for the prologue wait arm")
  ap.add_argument("--evidence-dir", type=pathlib.Path,
                  default=ROOT / "docs/task_workflow/evidence/nv-edge-aware-pdl-runtime-hook-20260821")
  ap.add_argument("--rows-jsonl", type=pathlib.Path, default=None)
  ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
  # Child args
  ap.add_argument("--arm", choices=tuple(ARMS))
  ap.add_argument("--role", choices=("control", "candidate"))
  ap.add_argument("--bracket-index", type=int, default=0)
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--tokens", type=int, default=8)
  ap.add_argument("--profile-jsonl", type=pathlib.Path, default=None)
  ap.add_argument("--census-json", type=pathlib.Path, default=None)
  ap.add_argument("--out", type=pathlib.Path, default=None)
  args = ap.parse_args()

  if args.mode == "child":
    if args.profile_jsonl is None or args.out is None or args.rows_jsonl is None:
      raise SystemExit("child mode requires --profile-jsonl, --rows-jsonl, --out")
    return run_child(args)

  if args.gate_evidence is None:
    raise SystemExit("driver mode requires --gate-evidence (no endpoint run is authorized "
                     "until the Stage 2 gate verdict passes)")
  if args.rows_jsonl is None:
    args.rows_jsonl = args.evidence_dir / "stage4_endpoint_rows.jsonl"
  return run_driver(args)


if __name__ == "__main__":
  raise SystemExit(main())
