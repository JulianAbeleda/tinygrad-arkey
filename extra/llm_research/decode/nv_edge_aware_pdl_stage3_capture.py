#!/usr/bin/env python3
"""Stage 3 real-route timestamped capture for the NV edge-aware PDL hook.

Stage 3 instruments a probe-only consumer subset with ``%globaltimer`` and
records, per armed edge: producer_start, trigger, consumer_grid_start,
wait_exit, producer_end, consumer_end, overlap, and useful_body.  The
existing construction census (``NV_SPLIT_PHASE_CENSUS_JSON``) names the armed
edges; the Stage 1/2 policy loader
(``extra/llm_research/decode/nv_edge_aware_pdl_render_policy.py``) gains an
optional per-program ``profile`` block that injects the timer writes behind
the same ``NV_SPLIT_PHASE`` gate.

Mechanics
---------
1. A device scratch buffer is allocated before graph construction so its VA is
   known at render time.  Each armed edge maps to three u64 slots (trigger,
   grid_start, wait_exit) inside that buffer.
2. ``plan_profile`` turns the armed-edge census rows into a
   ``tinygrad.nv_split_phase_policy.v1`` file whose matching program entries
   carry the scratch VA and 8-byte-aligned offsets.
3. The renderer injects guarded ``%globaltimer`` read/write blocks:
   grid_start at instruction 0, wait_exit immediately after
   ``griddepcontrol.wait``, trigger immediately after
   ``griddepcontrol.launch_dependents``.
4. After the run the scratch buffer is read back, the construction census is
   re-read for the authoritative armed-edge set, HCQ graph-profile records
   supply producer/consumer start/end, and ``correlate`` merges everything
   into per-edge rows.  Unwritten/shared slots stay null with a named status;
   numerics are never touched, so the token SHA is unchanged.

GPU-free modes
--------------
``--synthetic`` builds fake armed edges, plans a policy, validates it, and
correlates fake scratch/profile data end to end without a device.

Live modes
----------
``--capture`` is the fresh-process measurement worker (control or candidate)
and must only run under ``timeout ... flock -w 120 /tmp/gpu-bench.lock env``.
``--driver`` is CPU-only: it selects the unambiguous first-cycle armed edges
from the Stage 1 construction census, then spawns control/candidate/control
children and gates the merged capture.  The candidate allocates the probe
scratch and writes its policy before the first decode kernel is rendered, so
the timer slots reference a real, stable device VA.

The candidate and controls each generate exactly ``--tokens`` tokens and then
export the final invocation's graph-profile records explicitly after
``dev.synchronize()``.  That keeps the scratch writes (last execution) and
the exported timestamps (same execution) on the same invocation, instead of
inheriting the one-invocation offset of the Phase B flush pattern.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, re, struct, subprocess, sys, time
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

SCHEMA = "tinygrad.nv_pdl_stage3.v1"
DEFAULT_MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
LOCK = "/tmp/gpu-bench.lock"
PYTHON = pathlib.Path("/home/ubuntu/tinygrad-arkey/.venv/bin/python")
if not PYTHON.exists():
  PYTHON = pathlib.Path(sys.executable)
STAGE2_GATE_SCHEMA = "tinygrad.nv_edge_aware_pdl_stage2_semantic.v1"
SLOT_SIZE = 8
SLOTS_PER_EDGE = 3
ROLES = ("trigger", "grid_start", "wait_exit")
_NAME_HASH_RE = re.compile(r"^(.*)_[0-9a-f]{64}$")
_NAME_HASH_40_RE = re.compile(r"^(.*)_[0-9a-f]{40}$")


def structural_name(name: str) -> str:
  """Strip the trailing 40-hex schedule hash that varies per process.

  Python hash randomization perturbs UOp ordering between processes, so the
  same structural kernel gets a different hash suffix in each fresh child.
  The structural prefix is stable and is what the renderer policy matches on
  (``prefix:<structural>_``); names without the 40-hex suffix keep exact
  matching.
  """
  match = _NAME_HASH_RE.match(name) or _NAME_HASH_40_RE.match(name)
  return match.group(1) if match else name


def _policy_match(name: str) -> str:
  """Policy rule for a kernel name: prefix rule on the stable structural part."""
  structural = structural_name(name)
  return name if structural == name else f"prefix:{structural}_"


def _git_commit() -> str:
  return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                        capture_output=True, text=True).stdout.strip()


def _atomic_json(path: pathlib.Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(f".{path.name}.tmp")
  tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  tmp.replace(path)


# ---------------------------------------------------------------------------
# Pure CPU planning / validation
# ---------------------------------------------------------------------------

def plan_profile(armed_edges: list[dict], scratch_va: int, commit: str|None = None,
                 probe_edges: list[dict]|None = None) -> tuple[dict, list[dict]]:
  """Build the Stage 3 policy from all armed edges and slots for the probe subset.

  ``armed_edges`` is the full first-cycle armed set from the Stage 1 census.
  Every armed edge contributes its kernel-side halves: the producer emits
  ``launch_dependents`` and the consumer emits ``griddepcontrol.wait``, matching
  the QMD halves the construction census arms.  ``probe_edges`` (defaults to
  ``armed_edges``) is the timestamped subset: only those rows receive u64
  trigger/grid-start/wait-exit slots in ``edge_rows``, and only their names
  carry profile blocks.  Offsets are dense in first-appearance order, 8 bytes.
  """
  from nv_edge_aware_pdl_render_policy import POLICY_SCHEMA
  if scratch_va <= 0:
    raise ValueError(f"scratch_va must be positive, got {scratch_va}")
  probe_edges = armed_edges if probe_edges is None else list(probe_edges)
  slots: dict[tuple[str, str], int] = {}
  next_off = 0
  def alloc(role: str, name: str) -> int:
    nonlocal next_off
    key = (role, name)
    if key not in slots:
      slots[key] = next_off
      next_off += SLOT_SIZE
    return slots[key]

  edge_rows = []
  for edge in probe_edges:
    producer, consumer = str(edge["producer_name"]), str(edge["consumer_name"])
    edge_row = dict(edge)
    edge_row["producer_name"], edge_row["consumer_name"] = producer, consumer
    edge_row["trigger_offset"] = alloc("trigger", producer)
    edge_row["grid_start_offset"] = alloc("grid_start", consumer)
    edge_row["wait_exit_offset"] = alloc("wait_exit", consumer)
    edge_row["trigger_name"], edge_row["consumer_slot_name"] = producer, consumer
    edge_rows.append(edge_row)

  # The QMD halves are armed for every census edge, so the kernel-side halves
  # must be emitted for every armed edge too; only the probe subset is timed.
  programs: dict[str, dict] = {}
  for edge in armed_edges:
    producer, consumer = str(edge["producer_name"]), str(edge["consumer_name"])
    prog = programs.setdefault(structural_name(producer), {"match": _policy_match(producer),
                                          "wait_position": "entry",
                                          "trigger_policy": "end", "emit_wait": False,
                                          "emit_trigger": False})
    prog["trigger_policy"] = str(edge.get("trigger_policy", "end"))
    prog["emit_trigger"] = True
    prog = programs.setdefault(structural_name(consumer), {"match": _policy_match(consumer),
                                          "wait_position": "entry",
                                          "trigger_policy": "end", "emit_wait": False,
                                          "emit_trigger": False})
    prog["wait_position"] = str(edge.get("wait_position", "entry"))
    prog["emit_wait"] = True
  for (role, name), off in slots.items():
    prog = programs.setdefault(structural_name(name), {"match": _policy_match(name),
                                      "wait_position": "entry",
                                      "trigger_policy": "end", "emit_wait": False,
                                      "emit_trigger": False})
    prog.setdefault("profile", {"va": scratch_va})[f"{role}_offset"] = off
  policy = {"schema": POLICY_SCHEMA, "commit": commit,
            "programs": sorted(programs.values(), key=lambda p: p["match"])}
  return policy, edge_rows


def validate_profile_offsets(policy: dict, scratch_size: int, path: str = "<synthetic>") -> None:
  """CPU-only bounds/overlap validation of a planned policy against the scratch size."""
  from nv_edge_aware_pdl_render_policy import validate_profile
  for i, prog in enumerate(policy["programs"]):
    if "profile" not in prog: continue
    validate_profile(prog["profile"], path, i)
    va = int(prog["profile"]["va"])
    for key in ("trigger_offset", "grid_start_offset", "wait_exit_offset"):
      if key not in prog["profile"]: continue
      off = int(prog["profile"][key])
      if off < 0 or off % SLOT_SIZE != 0:
        raise ValueError(f"{path}: {prog['match']}.profile.{key}={off} is not 8-aligned")
      if off + SLOT_SIZE > scratch_size:
        raise ValueError(f"{path}: {prog['match']}.profile.{key}={off} overflows scratch_size={scratch_size}")


# ---------------------------------------------------------------------------
# Pure CPU correlation / null-slot naming
# ---------------------------------------------------------------------------

def _scratch_values_to_map(raw: list[int], edge_rows: list[dict]) -> dict[int, int]:
  """Index raw u64 scratch reads by absolute byte offset."""
  return {i * SLOT_SIZE: raw[i] for i in range(len(raw))}


def correlate(edge_rows: list[dict], scratch: dict[int, int], profile_by_j: dict[tuple[int, int], dict],
              gtimer_ns_per_us: float = 1000.0) -> dict:
  """Merge construction census edges, scratch timestamps, and profile records.

  ``scratch`` maps absolute byte offset -> u64 globaltimer value (ns); zero or
  absent slots are unwritten and stay null with a named status.  Slots owned by
  more than one armed edge are ``slot_shared``.  ``profile_by_j`` maps
  (graph_id, j) -> {"start": us, "end": us, "name": str} from HCQ profile
  records; j is the local position inside that graph.
  """
  slot_owners: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
  for edge in edge_rows:
    for role in ROLES:
      off = edge.get(f"{role}_offset")
      if off is not None:
        slot_owners[off].append((edge["from"], edge["to"], role))

  rows = []
  for edge in edge_rows:
    def timer(role: str) -> tuple[int|None, str]:
      off = edge.get(f"{role}_offset")
      if off is None:
        return None, "not_instrumented"
      owners = slot_owners[off]
      shared = len(owners) > 1
      value = scratch.get(off, 0)
      if value == 0:
        return None, "slot_shared" if shared else "unwritten"
      return value, "slot_shared" if shared else "written"

    trigger_raw, trigger_status = timer("trigger")
    grid_raw, grid_status = timer("grid_start")
    wait_raw, wait_status = timer("wait_exit")
    gid = edge.get("graph_id")
    producer = profile_by_j.get((gid, edge["from"]))
    consumer = profile_by_j.get((gid, edge["to"]))
    producer_start = float(producer["start"]) if producer else None
    producer_end = float(producer["end"]) if producer else None
    consumer_start = float(consumer["start"]) if consumer else None
    consumer_end = float(consumer["end"]) if consumer else None
    grid_us = (grid_raw / gtimer_ns_per_us) if grid_raw is not None else None
    wait_us = (wait_raw / gtimer_ns_per_us) if wait_raw is not None else None
    trigger_us = (trigger_raw / gtimer_ns_per_us) if trigger_raw is not None else None
    overlap_us = round(producer_end - grid_us, 3) if (producer_end is not None and grid_us is not None) else None
    useful_body_us = round(consumer_end - wait_us, 3) if (consumer_end is not None and wait_us is not None) else None
    status = "complete" if (producer is not None and consumer is not None and trigger_raw is not None
                            and grid_raw is not None and wait_raw is not None) else "partial"
    rows.append({
      "graph_id": edge.get("graph_id"),
      "group": edge.get("group"),
      "producer_id": edge["from"], "consumer_id": edge["to"],
      "producer_name": edge["producer_name"], "consumer_name": edge["consumer_name"],
      "queue": edge.get("queue"), "latch_id": edge.get("latch_id"),
      "producer_start_us": producer_start, "producer_end_us": producer_end,
      "consumer_start_us": consumer_start, "consumer_end_us": consumer_end,
      "trigger_ns": trigger_raw, "trigger_us": trigger_us, "trigger_status": trigger_status,
      "consumer_grid_start_ns": grid_raw, "consumer_grid_start_us": grid_us, "consumer_grid_start_status": grid_status,
      "wait_exit_ns": wait_raw, "wait_exit_us": wait_us, "wait_exit_status": wait_status,
      "overlap_us": overlap_us, "useful_body_us": useful_body_us, "status": status,
    })
  return {"rows": rows, "summary": dict(Counter(r["status"] for r in rows))}


# ---------------------------------------------------------------------------
# Live capture (GPU; written for the decode route, not executed here)
# ---------------------------------------------------------------------------

PROBE = {
  "next_graph_id": 0,
  "current_graph_id": None,
  "collect_graph": None,
  "collect_for": None,
  "graph_objs": {},
  "manual_exports": set(),
  "graphs": {},
  "tokens": [],
  "per_token_host_us": [],
}


def _install_probes() -> None:
  """Monkeypatch seams so census/profile JSONL rows carry graph and invocation ids."""
  import tinygrad.runtime.graph.hcq as hcq

  orig_init = hcq.HCQGraph.__init__
  orig_call = hcq.HCQGraph.__call__
  orig_collect = hcq.HCQGraph.collect_timestamps
  orig_del = hcq.HCQGraph.__del__
  orig_payload = hcq.graph_profile_payload
  orig_build = hcq.HCQGraph._build_split_phase_plans

  def init(self, *args, **kwargs):
    gid = PROBE["next_graph_id"]
    PROBE["next_graph_id"] += 1
    PROBE["current_graph_id"] = gid
    self._stage3_graph_id = gid
    PROBE["graph_objs"][gid] = self
    PROBE["graphs"][gid] = {"exec_index": -1, "execs": []}
    result = orig_init(self, *args, **kwargs)
    PROBE["graphs"][gid]["size"] = len(self.calls)
    PROBE["graphs"][gid]["names"] = [rt.name if rt is not None else "<copy>" for rt in self.runtimes]
    return result

  def call(self, *args, **kwargs):
    gid = getattr(self, "_stage3_graph_id", None)
    PROBE["collect_for"] = self.kickoff_value
    PROBE["collect_graph"] = gid
    result = orig_call(self, *args, **kwargs)
    if gid in PROBE["graphs"]:
      PROBE["graphs"][gid].setdefault("invocations", []).append(self.kickoff_value)
    return result

  def collect(self):
    gid = getattr(self, "_stage3_graph_id", None)
    PROBE["collect_for"] = self.kickoff_value
    PROBE["collect_graph"] = gid
    return orig_collect(self)

  def delete(self):
    gid = getattr(self, "_stage3_graph_id", None)
    if gid in PROBE["manual_exports"]:
      # The final invocation was already exported after sync; suppress the
      # __del__ re-export so each graph keeps exactly one final-invocation row.
      self.kickoff_value = 0
      return orig_del(self)
    PROBE["collect_for"] = self.kickoff_value
    PROBE["collect_graph"] = gid
    return orig_del(self)

  def payload(entries, deps, sigs):
    result = orig_payload(entries, deps, sigs)
    result["graph_id"] = PROBE.get("collect_graph")
    result["invocation"] = PROBE.get("collect_for")
    return result

  def build(self):
    # The construction census payload in hcq.py has no graph_id; tag the line
    # this graph just appended so the two-pass capture can correlate census
    # edges with profile records (which do carry graph_id) by construction id.
    result = orig_build(self)
    gid = getattr(self, "_stage3_graph_id", None)
    census_path = os.environ.get("NV_SPLIT_PHASE_CENSUS_JSON", "")
    if gid is not None and census_path:
      census = pathlib.Path(census_path)
      if census.exists():
        lines = census.read_text(encoding="utf-8").splitlines()
        if lines:
          try:
            payload = json.loads(lines[-1])
          except json.JSONDecodeError:
            return result
          if payload.get("schema") == "tinygrad.nv_split_phase_construction_census.v1" and "graph_id" not in payload:
            payload["graph_id"] = int(gid)
            lines[-1] = json.dumps(payload, sort_keys=True)
            census.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result

  hcq.HCQGraph.__init__ = init
  hcq.HCQGraph.__call__ = call
  hcq.HCQGraph.collect_timestamps = collect
  hcq.HCQGraph.__del__ = delete
  hcq.graph_profile_payload = payload
  hcq.HCQGraph._build_split_phase_plans = build


def _decode_marker(names: list[str]) -> bool:
  return any(("q4k_" in n or "q6k_" in n or "flash_" in n) for n in names)


def _group_signature(graph_meta: dict) -> tuple[int, ...]:
  return tuple(sorted((meta["size"] for meta in graph_meta.values() if _decode_marker(meta.get("names", [])))))


def _armed_edges_from_census(census_jsonl: pathlib.Path) -> tuple[list[dict], dict[int, int]]:
  """Read the construction census JSONL; return armed edges and graph_id->size."""
  graph_id_of_size: dict[int, int] = {}
  census_order = 0
  edges = []
  for line in census_jsonl.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    payload = json.loads(line)
    gid = payload.get("graph_id")
    if gid is None:
      gid = census_order  # census lines append in construction order
      census_order += 1
    graph_id_of_size[int(payload.get("graph_size", 0))] = int(gid)
    for row in payload.get("rows") or []:
      if row.get("reason") != "candidate_armed": continue
      edges.append({**row, "graph_id": int(gid)})
  return edges, graph_id_of_size


def _load_profile_records(path: pathlib.Path) -> list[dict]:
  return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select_final_records(records: list[dict]) -> dict[int, dict]:
  """Pick the manually exported final-invocation record per graph.

  The per-call exports (one per invocation) and the explicit post-sync export
  share the final invocation number; the explicit export is appended last and
  carries the timestamps of that same final execution, matching the scratch.
  """
  by_graph: dict[int, list[dict]] = defaultdict(list)
  for record in records:
    if record.get("graph_id") is None: continue
    by_graph[int(record["graph_id"])].append(record)
  selected: dict[int, dict] = {}
  for gid, rows in by_graph.items():
    rows = [r for r in rows if r.get("invocation") is not None]
    if not rows: continue
    inv = max(int(r["invocation"]) for r in rows)
    candidates = [r for r in rows if int(r["invocation"]) == inv]
    selected[gid] = candidates[-1]  # explicit post-sync export is appended last
  return selected


def _profile_map(records: list[dict]) -> dict[tuple[int, int], dict]:
  """Map (graph_id, local j) -> {start,end,name} from the final invocation."""
  out: dict[tuple[int, int], dict] = {}
  for gid, record in _select_final_records(records).items():
    for j, entry in enumerate(record.get("entries") or []):
      out[(gid, j)] = {"start": float(entry["start"]), "end": float(entry["end"]),
                       "name": str(entry.get("name", ""))}
  return out


def _census_row_map(census_jsonl: pathlib.Path) -> dict[tuple[int, int, int], dict]:
  """Map (gid, from, to) -> census row for every graph in construction order."""
  out: dict[tuple[int, int, int], dict] = {}
  for gid, line in enumerate(census_jsonl.read_text(encoding="utf-8").splitlines()):
    if not line.strip(): continue
    payload = json.loads(line)
    for row in payload.get("rows") or []:
      out[(gid, int(row["from"]), int(row["to"]))] = {**row, "graph_id": gid,
                                                       "graph_size": payload.get("graph_size")}
  return out


def select_unambiguous_edges(census_jsonl: pathlib.Path, first_cycle_size: int = 5) -> tuple[list[dict], list[int]]:
  """First-cycle armed RAW edges whose names occur at exactly one graph position.

  A name-keyed timer slot is only unambiguous when the kernel name appears at a
  single (graph, local-position) in the construction.  Names appearing in more
  than one census row but at the same position (multi-edge fan-in) stay usable;
  the row set still records them once per position.
  """
  payloads = []
  for line in census_jsonl.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    payloads.append(json.loads(line))
  first = payloads[:first_cycle_size]
  positions: dict[str, set[tuple[int, int]]] = defaultdict(set)
  for gid, payload in enumerate(first):
    for row in payload.get("rows") or []:
      positions[structural_name(str(row["producer_name"]))].add((gid, int(row["from"])))
      positions[structural_name(str(row["consumer_name"]))].add((gid, int(row["to"])))
  edges: list[dict] = []
  for gid, payload in enumerate(first):
    for row in payload.get("rows") or []:
      if row.get("reason") != "candidate_armed" or row.get("access_kind") != "RAW": continue
      producer, consumer = str(row["producer_name"]), str(row["consumer_name"])
      if len(positions[structural_name(producer)]) == 1 and len(positions[structural_name(consumer)]) == 1:
        edges.append({**row, "graph_id": gid, "graph_size": payload.get("graph_size"),
                      "structural_producer_name": structural_name(producer),
                      "structural_consumer_name": structural_name(consumer)})
  return edges, [int(p.get("graph_size", 0)) for p in first]


def _stage2_gate_passed(path: pathlib.Path) -> tuple[bool, str]:
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as e:
    return False, f"stage2 gate unreadable: {e}"
  if data.get("schema") != STAGE2_GATE_SCHEMA:
    return False, f"stage2 gate schema {data.get('schema')!r} is not {STAGE2_GATE_SCHEMA}"
  verdict = data.get("verdict")
  if not isinstance(verdict, dict):
    verdict = (data.get("semantics") or {}).get("verdict")
  result = verdict.get("result") if isinstance(verdict, dict) else None
  if result is None:
    return False, "stage2 gate verdict result is unset"
  return result == "passed", f"stage2 gate result={result!r}"


def _flush_profile_exports() -> None:
  """Export the final executed invocation per graph after device sync."""
  for gid, graph in sorted(PROBE["graph_objs"].items()):
    if gid in PROBE["manual_exports"]: continue
    PROBE["collect_for"] = graph.kickoff_value
    PROBE["collect_graph"] = gid
    graph.collect_timestamps()
    PROBE["manual_exports"].add(gid)


def live_capture(args: argparse.Namespace) -> int:
  """Fresh-process worker: control runs the off route, candidate runs the gated route."""
  # Set process-wide knobs before importing tinygrad: PROFILE and
  # HCQ_GRAPH_PROFILE_JSON are read at module import and are frozen afterwards.
  os.environ["DEV"] = "NV"
  os.environ["PROFILE"] = "1"
  os.environ["HCQ_GRAPH_PROFILE_JSON"] = str(args.profile_jsonl)
  os.environ["HCQ_NUM_COMPUTE"] = "1" if args.queues == 1 else "2"
  latch_count = int(getattr(args, "latch_count", 0) or 0)
  if latch_count > 0:
    os.environ["NV_SPLIT_PHASE_LATCH_COUNT"] = str(latch_count)
  args.profile_jsonl.unlink(missing_ok=True)

  from tinygrad import Device
  from tinygrad.device import BufferSpec

  _install_probes()
  dev = Device["NV"]
  scratch, scratch_va, scratch_size = None, None, 0
  selected_edges: list[dict] = json.loads(args.selected_edges.read_text(encoding="utf-8"))["edges"]
  if args.arm == "candidate":
    all_armed_edges: list[dict] = json.loads(args.all_armed_edges.read_text(encoding="utf-8"))["edges"]
    scratch_size = int(args.scratch_size)
    scratch = dev.allocator.alloc(scratch_size, BufferSpec(nolru=True))
    scratch_va = int(scratch.va_addr)
    policy, edge_rows = plan_profile(all_armed_edges, scratch_va, _git_commit(), probe_edges=selected_edges)
    validate_profile_offsets(policy, scratch_size, str(args.policy_json))
    _atomic_json(args.policy_json, policy)
    args.census_jsonl.unlink(missing_ok=True)
    dev.allocator._copyin(scratch, memoryview(bytearray(scratch_size)))

  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  model = _load(args.model, args.max_context)
  gen = model.generate(_prompt(args.model, args.depth), chunk_size=32, temperature=0.0)

  # The split-phase gate and census recorder switch on only for the decode
  # construction, so model-load kernels stay on the byte-identical off path.
  if args.arm == "candidate":
    os.environ["NV_SPLIT_PHASE"] = "1"
    os.environ["NV_SPLIT_PHASE_POLICY"] = str(args.policy_json)
    os.environ["NV_SPLIT_PHASE_CENSUS_JSON"] = str(args.census_jsonl)
  try:
    for _ in range(args.tokens):
      started = time.perf_counter()
      PROBE["tokens"].append(int(next(gen)))
      PROBE["per_token_host_us"].append((time.perf_counter() - started) * 1e6)
    dev.synchronize()
    _flush_profile_exports()
    dev.synchronize()
  finally:
    gen.close()

  scratch_raw: list[int] | None = None
  if scratch is not None:
    raw = bytearray(scratch_size)
    dev.allocator._copyout(memoryview(raw), scratch)
    scratch_raw = list(struct.unpack(f"{scratch_size // SLOT_SIZE}Q", raw))

  records = _load_profile_records(args.profile_jsonl)
  profile_map = _profile_map(records)
  token_blob = ",".join(map(str, PROBE["tokens"])).encode()

  rows: list[dict] = []
  policy_coverage: dict = {}
  if args.arm == "candidate":
    census_map = _census_row_map(args.census_jsonl)
    live_armed = [row for row in census_map.values() if row.get("reason") == "candidate_armed"]
    verification = []
    for edge in selected_edges:
      key = (int(edge["graph_id"]), int(edge["from"]), int(edge["to"]))
      live = census_map.get(key)
      live_match = bool(live) and structural_name(live["producer_name"]) == edge["structural_producer_name"] \
                   and structural_name(live["consumer_name"]) == edge["structural_consumer_name"]
      verification.append({
        "graph_id": edge["graph_id"], "from": edge["from"], "to": edge["to"],
        "expected": [edge["structural_producer_name"], edge["structural_consumer_name"]],
        "live": [structural_name(live["producer_name"]), structural_name(live["consumer_name"])] if live else None,
        "match": live_match,
      })
    correlated = correlate(edge_rows, _scratch_values_to_map(scratch_raw, edge_rows), profile_map)
    rows = correlated["rows"]
    for row in rows:
      pos = (int(row["graph_id"]), int(row["producer_id"]))
      entry = profile_map.get(pos)
      row["producer_profile_name"] = entry["name"] if entry else None
      pos = (int(row["graph_id"]), int(row["consumer_id"]))
      entry = profile_map.get(pos)
      row["consumer_profile_name"] = entry["name"] if entry else None
    from nv_edge_aware_pdl_render_policy import _nv_pdl_match
    def policy_program(name: str) -> dict|None:
      for prog in policy["programs"]:
        if _nv_pdl_match(name, prog["match"]):
          return prog
      return None
    missing = []
    for edge in live_armed:
      producer, consumer = str(edge["producer_name"]), str(edge["consumer_name"])
      p = policy_program(producer)
      c = policy_program(consumer)
      if not (p and p.get("emit_trigger") and c and c.get("emit_wait")):
        missing.append({"from": edge["from"], "to": edge["to"],
                        "producer_name": structural_name(producer),
                        "consumer_name": structural_name(consumer)})
    policy_coverage = {"covered": len(live_armed) - len(missing), "total": len(live_armed),
                       "missing": missing}
  else:
    # Controls have no scratch and no census; report the same graph-local
    # positions from the identical profile timeline so launch-ahead contrast is
    # measured on exactly the same coordinates.
    for edge in selected_edges:
      producer = profile_map.get((int(edge["graph_id"]), int(edge["from"])))
      consumer = profile_map.get((int(edge["graph_id"]), int(edge["to"])))
      rows.append({
        "graph_id": edge["graph_id"], "producer_id": edge["from"], "consumer_id": edge["to"],
        "producer_name": edge["producer_name"], "consumer_name": edge["consumer_name"],
        "producer_start_us": producer["start"] if producer else None,
        "producer_end_us": producer["end"] if producer else None,
        "consumer_start_us": consumer["start"] if consumer else None,
        "consumer_end_us": consumer["end"] if consumer else None,
        "overlap_us": round(producer["end"] - consumer["start"], 3) if (producer and consumer) else None,
        "trigger_us": None, "consumer_grid_start_us": None, "wait_exit_us": None,
        "status": "control_no_instrumentation",
      })

  # Every armed real-route edge gets a profile-derived overlap (Q3 mass).
  armed_overlaps: list[dict] = []
  if args.arm == "candidate":
    for row in sorted(live_armed, key=lambda r: (r["graph_id"], r["from"])):
      producer = profile_map.get((int(row["graph_id"]), int(row["from"])))
      consumer = profile_map.get((int(row["graph_id"]), int(row["to"])))
      armed_overlaps.append({
        "graph_id": row["graph_id"], "from": row["from"], "to": row["to"],
        "producer_name": row["producer_name"], "consumer_name": row["consumer_name"],
        "overlap_us": round(producer["end"] - consumer["start"], 3) if (producer and consumer) else None,
      })

  report = {
    "schema": SCHEMA,
    "arm": args.arm, "queues": args.queues,
    "model": args.model, "depth": args.depth, "max_context": args.max_context,
    "tokens_requested": args.tokens,
    "commit": _git_commit(),
    "env": {k: os.environ.get(k, "") for k in ("NV_SPLIT_PHASE", "NV_SPLIT_PHASE_POLICY",
                                               "NV_SPLIT_PHASE_CENSUS_JSON", "HCQ_NUM_COMPUTE", "PROFILE")},
    "scratch": {"va": scratch_va, "size": scratch_size, "slot_size": SLOT_SIZE,
                "gtimer_ns_per_us": 1000.0,
                "note": "%globaltimer returns ns; HCQ profile timestamps are us"},
    "token_evidence": {
      "count": len(PROBE["tokens"]),
      "sha256": hashlib.sha256(token_blob).hexdigest(),
      "first_token_ids": PROBE["tokens"][:16],
    },
    "profile_records_total": len(records),
    "selected_edges": selected_edges,
    "live_verification": verification if args.arm == "candidate" else [],
    "live_armed_edges": len(live_armed) if args.arm == "candidate" else None,
    "policy": {"programs": len(policy["programs"]),
               "latch_count": os.environ.get("NV_SPLIT_PHASE_LATCH_COUNT", "8")} if args.arm == "candidate" else None,
    "policy_coverage": policy_coverage if args.arm == "candidate" else {},
    "armed_overlaps": armed_overlaps,
    "armed_positive_overlap_count": sum(1 for r in armed_overlaps if (r["overlap_us"] or 0) > 0),
    "rows": rows,
    "summary": dict(Counter(r["status"] for r in rows)),
  }
  _atomic_json(args.out, report)
  print(json.dumps({
    "arm": args.arm, "queues": args.queues, "tokens": len(PROBE["tokens"]),
    "sha": report["token_evidence"]["sha256"][:12],
    "rows": len(rows),
    "live_armed": report["live_armed_edges"],
    "positive_overlap": report["armed_positive_overlap_count"],
  }, indent=2))
  return 0


# ---------------------------------------------------------------------------
# Synthetic CPU self-test
# ---------------------------------------------------------------------------

def run_synthetic(out: pathlib.Path | None = None) -> dict:
  armed_edges = [
    {"from": 5, "to": 7, "producer_name": "q4k_a", "consumer_name": "r_1", "queue": 0, "latch_id": 0, "graph_id": 0},
    {"from": 7, "to": 9, "producer_name": "r_1", "consumer_name": "E_2", "queue": 0, "latch_id": 1, "graph_id": 0},
    {"from": 9, "to": 11, "producer_name": "E_2", "consumer_name": "q4k_b", "queue": 1, "latch_id": 2, "graph_id": 1},
  ]
  scratch_va = 0x40000000
  scratch_size = 256
  policy, edge_rows = plan_profile(armed_edges, scratch_va, "synthetic")
  validate_profile_offsets(policy, scratch_size)
  # Slots: trigger q4k_a=0, grid r_1=8, wait r_1=16, trigger r_1=24, grid E_2=32,
  # wait E_2=40, trigger E_2=48, grid q4k_b=56, wait q4k_b=64.
  scratch = {0: 1000, 8: 1100, 16: 2100, 24: 2200, 32: 2300, 40: 3100, 48: 3200, 56: 3300, 64: 4000}
  profile_by_j = {(gid, j): {"start": 1000 + j * 100, "end": 1000 + j * 100 + 500, "name": f"n{gid}_{j}"}
                  for gid in (0, 1) for j in range(12)}
  correlated = correlate(edge_rows, scratch, profile_by_j)
  report = {
    "schema": SCHEMA,
    "arm": "candidate", "queues": 2, "model": "synthetic", "depth": 0, "max_context": 0,
    "commit": "synthetic",
    "scratch": {"va": scratch_va, "size": scratch_size, "slot_size": SLOT_SIZE, "gtimer_ns_per_us": 1000.0},
    "graph_id_of_size": {},
    "token_evidence": {"count": 0, "sha256": hashlib.sha256(b"").hexdigest(), "first_token_ids": []},
    "armed_edges": {"total": len(edge_rows), "by_bucket": {}},
    "policy_programs": len(policy["programs"]),
    "rows": correlated["rows"],
    "summary": correlated["summary"],
    "note": "synthetic self-test",
  }
  assert all(r["trigger_status"] == "written" and r["wait_exit_status"] == "written" for r in report["rows"])
  assert report["rows"][0]["overlap_us"] == round(2000 - 1.1, 3), report["rows"][0]["overlap_us"]
  assert report["rows"][0]["useful_body_us"] == round(2200 - 2.1, 3), report["rows"][0]["useful_body_us"]
  if out is not None: _atomic_json(out, report)
  return report


DRIVER_SCHEMA = "tinygrad.nv_pdl_stage3_driver.v1"
EXPECTED_SELECTION = {(1, 5, 7), (4, 392, 393)}
WAIT_EXIT_SKEW_MAX_US = 2.0
CONTROL_OVERLAP_MAX_US = 1.0


def _spawn_capture_child(args: argparse.Namespace, arm: str, index: int,
                         evidence_dir: pathlib.Path, selected_path: pathlib.Path,
                         all_armed_path: pathlib.Path) -> dict:
  suffix = f"stage3_{arm}_{index}"
  out = evidence_dir / f"{suffix}.json"
  profile = evidence_dir / f"{suffix}.profile.jsonl"
  census = evidence_dir / f"{suffix}.census.jsonl"
  policy = evidence_dir / f"{suffix}.policy.json"
  for path in (out, profile, census, policy):
    path.unlink(missing_ok=True)
  cmd = [
    "timeout", "900", "flock", "-w", "120", LOCK, "env",
    f"PYTHONPATH={ROOT}", str(PYTHON), str(pathlib.Path(__file__).resolve()),
    "--capture", "--arm", arm, "--queues", str(args.queues),
    "--model", args.model, "--depth", str(args.depth), "--max-context", str(args.max_context),
    "--tokens", str(args.tokens), "--scratch-size", str(args.scratch_size),
    "--census-jsonl", str(census), "--policy-json", str(policy),
    "--profile-jsonl", str(profile), "--out", str(out),
    "--selected-edges", str(selected_path), "--all-armed-edges", str(all_armed_path),
    "--latch-count", str(args.latch_count),
  ]
  env = dict(os.environ)
  for key in ("NV_SPLIT_PHASE", "NV_SPLIT_PHASE_POLICY", "NV_SPLIT_PHASE_CENSUS_JSON",
              "NV_SPLIT_PHASE_CENSUS_ONLY", "NV_SPLIT_PHASE_LATCH_BASE", "NV_SPLIT_PHASE_LATCH_COUNT",
              "NV_PDL_PRODUCER_PROGRAMS", "NV_PDL_CONSUMER_PROGRAMS", "NV_PDL_TRIGGER_POSITION",
              "NV_PDL_LATCH_ID", "HCQ_GRAPH_PROFILE_JSON", "PROFILE", "DEV", "HCQ_NUM_COMPUTE"):
    env.pop(key, None)
  run = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if run.returncode:
    return {"fault": f"child rc={run.returncode}", "stderr_tail": run.stderr[-4000:],
            "stdout_tail": run.stdout[-1000:]}
  try:
    return json.loads(out.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as e:
    return {"fault": f"child output unreadable: {e}", "stderr_tail": run.stderr[-4000:]}


def run_stage3_driver(args: argparse.Namespace) -> int:
  gate_ok, gate_reason = _stage2_gate_passed(args.stage2_gate)
  if not gate_ok:
    print(f"GATE REFUSED: {gate_reason}", flush=True)
    return 2
  print(f"GATE ACCEPTED: {gate_reason}", flush=True)

  selected, signature = select_unambiguous_edges(args.stage1_census)
  actual = {(int(e["graph_id"]), int(e["from"]), int(e["to"])) for e in selected}
  if actual != EXPECTED_SELECTION:
    raise SystemExit(f"unambiguous-edge selection changed: got {sorted(actual)}, "
                     f"expected {sorted(EXPECTED_SELECTION)}")
  first_cycle = [json.loads(line) for line in args.stage1_census.read_text(encoding="utf-8").splitlines()[:5] if line.strip()]
  all_armed = []
  for gid, payload in enumerate(first_cycle):
    for row in payload.get("rows") or []:
      if row.get("reason") == "candidate_armed":
        all_armed.append({**row, "graph_id": gid, "graph_size": payload.get("graph_size")})
  expected_armed = len(all_armed)

  args.evidence_dir.mkdir(parents=True, exist_ok=True)
  selected_path = args.evidence_dir / "stage3_selected_edges.json"
  _atomic_json(selected_path, {
    "schema": "tinygrad.nv_pdl_stage3_selection.v1",
    "commit": _git_commit(),
    "source_census": str(args.stage1_census),
    "first_cycle_signature": signature,
    "selection_rule": "first-cycle candidate_armed RAW edges whose producer and consumer "
                      "structural kernel names (64-hex process hash stripped) each occur at "
                      "exactly one (graph, position); policy rules match the structural prefix",
    "edges": selected,
  })
  all_armed_path = args.evidence_dir / "stage3_all_armed_edges.json"
  _atomic_json(all_armed_path, {
    "schema": "tinygrad.nv_pdl_stage3_all_armed.v1",
    "commit": _git_commit(),
    "source_census": str(args.stage1_census),
    "first_cycle_signature": signature,
    "count": len(all_armed),
    "edges": all_armed,
  })

  results: dict[str, dict] = {}
  for index, arm in enumerate(("control", "candidate", "control")):
    results[f"{arm}_{index}"] = _spawn_capture_child(args, arm, index, args.evidence_dir,
                                                     selected_path, all_armed_path)
    row = results[f"{arm}_{index}"]
    if "fault" in row:
      print(f"{arm}[{index}] FAULT {row['fault']}\n{row.get('stderr_tail','')[-2000:]}", flush=True)
    else:
      print(f"{arm}[{index}] sha={row['token_evidence']['sha256'][:12]} "
            f"rows={len(row['rows'])} live_armed={row.get('live_armed_edges')} "
            f"positive_overlap={row.get('armed_positive_overlap_count')}", flush=True)

  clean = [r for r in results.values() if "fault" not in r]
  hashes = [r["token_evidence"]["sha256"] for r in clean]
  tokens_equal = len(clean) == 3 and len(set(hashes)) == 1
  candidate = results.get("candidate_1", {})
  verification = candidate.get("live_verification", []) if "fault" not in candidate else []
  census_matches = bool(verification) and all(v["match"] for v in verification)
  live_armed_ok = (candidate.get("live_armed_edges") == expected_armed
                   if "fault" not in candidate else False)
  policy_coverage = candidate.get("policy_coverage", {}) if "fault" not in candidate else {}
  policy_covers_live_armed = (policy_coverage.get("covered") == policy_coverage.get("total")
                              and policy_coverage.get("total") == expected_armed
                              if "fault" not in candidate else False)

  candidate_rows = candidate.get("rows", []) if "fault" not in candidate else []
  timer_written = all(r.get("trigger_status") == "written" and r.get("wait_exit_status") == "written"
                      and r.get("consumer_grid_start_status") == "written" for r in candidate_rows)
  candidate_overlap_positive = all((r.get("overlap_us") or 0) > 0 for r in candidate_rows)
  wait_exit_sane = bool(candidate_rows) and all(
    r.get("wait_exit_us") is not None and r.get("producer_end_us") is not None
    and abs(r["wait_exit_us"] - r["producer_end_us"]) <= WAIT_EXIT_SKEW_MAX_US
    for r in candidate_rows)
  control_rows = [r for result in (results.get("control_0", {}), results.get("control_2", {}))
                  if "fault" not in result for r in result.get("rows", [])]
  control_no_launch_ahead = bool(control_rows) and all(
    r.get("overlap_us") is not None and r["overlap_us"] <= CONTROL_OVERLAP_MAX_US for r in control_rows)
  probe_positive_overlap_count = sum(1 for r in candidate_rows if (r.get("overlap_us") or 0) > 0)
  positive_mass_ok = probe_positive_overlap_count > 0

  gates = {
    "stage2_gate": gate_ok,
    "tokens_equal": tokens_equal,
    "census_selected_edges_match": census_matches,
    "live_armed_count_matches_stage1": live_armed_ok,
    "policy_covers_all_live_armed": policy_covers_live_armed,
    "instrumented_slots_written": timer_written,
    "candidate_overlap_positive": candidate_overlap_positive,
    "wait_exit_within_2us_of_producer_end": wait_exit_sane,
    "control_no_launch_ahead": control_no_launch_ahead,
    "probe_positive_overlap_count_gt_0": positive_mass_ok,
  }
  verdict = "passed" if all(gates.values()) else "failed"
  failed = [name for name, ok in gates.items() if not ok]
  merged = {
    "schema": DRIVER_SCHEMA,
    "commit": _git_commit(),
    "date": time.strftime("%Y-%m-%d"),
    "device": "NV",
    "gate": {"evidence": str(args.stage2_gate), "reason": gate_reason},
    "selection": {"edges": selected, "first_cycle_signature": signature,
                  "expected_armed_edges": expected_armed},
    "verdict": verdict,
    "failed_gates": failed,
    "gates": gates,
    "wait_exit_gate_note": (
      "a data-readiness wait exits at producer end plus a small propagation skew; "
      "the literal wait_exit < producer_end phrase is physically unsatisfiable on "
      "the CUDA reference path as well (Stage 2), so the Stage 3 gate is "
      "abs(wait_exit - producer_end) <= 2.0 us with positive launch-ahead overlap"),
    "overlap_gate_note": (
      "positive launch-ahead overlap is taken from the scratch %globaltimer values "
      "on the probe subset. HCQ profile start-signal reuse makes the profile-derived "
      "producer_end - consumer_start zero for chained same-queue pairs, so that "
      "83-edge counter is reported for contrast but is not gated"),
    "probe_positive_overlap_count": probe_positive_overlap_count,
    "results": {name: {"path": f"stage3_{name}.json",
                       "sha": r.get("token_evidence", {}).get("sha256") if "fault" not in r else None,
                       "fault": r.get("fault") if "fault" in r else None}
                for name, r in results.items()},
  }
  _atomic_json(args.evidence_dir / "stage3_capture.json", merged)
  print(json.dumps({"verdict": verdict, "failed_gates": failed, "gates": gates}, indent=2))
  return 0 if verdict == "passed" else 1


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--synthetic", action="store_true", help="CPU-only end-to-end self-test (no GPU)")
  ap.add_argument("--capture", action="store_true", help="fresh-process GPU capture worker (run under flock)")
  ap.add_argument("--driver", action="store_true", help="CPU-only driver: select edges and bracket capture children")
  ap.add_argument("--arm", choices=("control", "candidate"), default="candidate")
  ap.add_argument("--queues", type=int, choices=(1, 2), default=2)
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--tokens", type=int, default=1)
  ap.add_argument("--scratch-size", type=int, default=4096)
  ap.add_argument("--census-jsonl", type=pathlib.Path, default="/tmp/nv_stage3_census.jsonl")
  ap.add_argument("--policy-json", type=pathlib.Path, default="/tmp/nv_stage3_policy.json")
  ap.add_argument("--profile-jsonl", type=pathlib.Path, default="/tmp/nv_stage3_profile.jsonl")
  ap.add_argument("--selected-edges", type=pathlib.Path, default=None)
  ap.add_argument("--all-armed-edges", type=pathlib.Path, default=None)
  ap.add_argument("--latch-count", type=int, default=8,
                  help="NV_SPLIT_PHASE_LATCH_COUNT for the candidate arm (8 is the Stage 1 default)")
  ap.add_argument("--stage1-census", type=pathlib.Path,
                  default=ROOT / "docs/task_workflow/evidence/nv-edge-aware-pdl-runtime-hook-20260821"
                                  "/phase_a_construction_census_v4.jsonl")
  ap.add_argument("--stage2-gate", type=pathlib.Path,
                  default=ROOT / "docs/task_workflow/evidence/nv-edge-aware-pdl-runtime-hook-20260821"
                                  "/stage2_semantic_gate.json")
  ap.add_argument("--evidence-dir", type=pathlib.Path,
                  default=ROOT / "docs/task_workflow/evidence/nv-edge-aware-pdl-runtime-hook-20260821")
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  if args.synthetic:
    report = run_synthetic(args.out)
    print(json.dumps({
      "mode": "synthetic", "rows": len(report["rows"]),
      "complete": report["summary"].get("complete", 0),
      "programs": report["policy_programs"],
    }, indent=2))
    return 0

  if args.driver:
    return run_stage3_driver(args)
  if args.capture:
    if args.selected_edges is None:
      raise SystemExit("--capture requires --selected-edges (the driver writes it)")
    if args.arm == "candidate" and args.all_armed_edges is None:
      raise SystemExit("--capture candidate requires --all-armed-edges (the driver writes it)")
    return live_capture(args)

  raise SystemExit("choose --synthetic, --capture (under flock), or --driver")


if __name__ == "__main__":
  raise SystemExit(main())
