#!/usr/bin/env python3
"""Weighted llama d512 decode dependency DAG (measurement tooling only).

nsys CUDA node traces carry kernel intervals and CUDA-graph node ids but no
dependency-edge table.  This tool joins the structural ggml graph dump
(`llama_decode.dot` from the pinned llama build) to the CUPTI replay:

1. parse the dot graph into logical ggml nodes and tensor edges;
2. fold llama's epilogue constructs into the kernel that owns them: norm
   `x*y` into `rms_norm`, attention `ffn_inp x+y` into the O MMVQ, `glu` and
   `l_out x+y` into the down MMVQ, the V-cache `set_rows` into flash, and
   `ffn_gate`+`ffn_up` into the fused W1W3 MMVQ.  The final layer's explicit
   get_rows/add residual remains real kernels because its O MMVQ has no
   fusion epilogue;
3. expand each folded group into the observed CUPTI launch sequence;
4. attach start/end/duration from one complete steady replay;
5. emit a kernel-level DAG with direct tensor-dependency edges.

Profiled durations here are topology evidence, never an unprofiled wall claim.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, re, sqlite3, statistics
from collections import Counter, defaultdict

SCHEMA = "tinygrad.llama_weighted_dag.v1"
VIEW_OPS = frozenset(("view", "reshape", "permute"))


def _sha256(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def parse_dot(path: pathlib.Path) -> dict:
  text = path.read_text(encoding="utf-8")
  nodes: dict[str, dict] = {}
  order: list[str] = []
  edges: list[dict] = []
  for line in text.splitlines():
    if (m := re.match(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"\s*\[', line)):
      label = re.search(r'label\s*=\s*"([^"]*)"', line)
      edges.append({"from": m.group(1), "to": m.group(2), "label": label.group(1) if label else None})
      continue
    m = re.match(r'^\s*"([^"]+)"\s*\[', line)
    if not m:
      continue
    addr = m.group(1)
    label = re.search(r'label="((?:[^"\\]|\\.)*)"', line)
    if not label:
      continue
    raw = label.group(1)
    head = raw.split("|", 1)[0].strip()
    seq = int(match.group(1)) if (match := re.search(r"\|\s*(\d+)\s+\[", raw)) else None
    op = raw.split("<x>", 1)[1].split("(")[0].strip().strip('"') if "<x>" in raw else "CONST"
    dtype = "f16" if "(f16)" in raw else "f32" if "(f32)" in raw else "other"
    shape = None
    if (sm := re.search(r"\|\s*\d+\s*\[([^\]]*)\]", raw)):
      try:
        shape = [int(x) for x in sm.group(1).split(",")]
      except ValueError:
        shape = None
    nodes[addr] = {"addr": addr, "label": head, "seq": seq, "op": op, "dtype": dtype, "shape": shape}
    order.append(addr)
  return {"nodes": nodes, "order": order, "edges": edges}


def base_name(label: str) -> str:
  name = label.split(" (", 1)[0].split("|", 1)[0].strip()
  name = re.sub(r"-(\d+)$", "", name)
  if re.match(r"^node_\d+$", name):
    return "node_"
  return name


def layer_suffix(label: str) -> int | None:
  name = label.split(" (", 1)[0].strip()
  m = re.search(r"-(\d+)$", name)
  return int(m.group(1)) if m else None


def build_groups(parsed: dict) -> dict:
  """Assign logical nodes to folded kernel groups in dot order."""
  nodes, order, edges = parsed["nodes"], parsed["order"], parsed["edges"]

  def producer(addr: str) -> str:
    seen: set[str] = set()
    cur = addr
    while cur in nodes and nodes[cur]["op"] in VIEW_OPS and cur not in seen:
      seen.add(cur)
      parents = [e["from"] for e in edges if e["to"] == cur]
      if len(parents) != 1:
        break
      cur = parents[0]
    return cur

  members: dict[int, list[str]] = defaultdict(list)

  def group_of(addr: str) -> int | None:
    for gid, addrs in members.items():
      if addr in addrs:
        return gid
    return None

  def remove_from_group(addr: str, keep: int) -> None:
    for gid in list(members):
      if gid == keep:
        continue
      if addr in members[gid]:
        members[gid].remove(addr)
        if not members[gid]:
          del members[gid]

  def new_group(addr: str) -> int:
    gid = len(members)
    members[gid].append(addr)
    return gid

  for addr in order:
    node = nodes[addr]
    op = node["op"]
    if op in (None, "CONST") or op in VIEW_OPS:
      continue
    base = base_name(node["label"])
    if op == "x*y":
      parent = producer(addr)
      if parent in nodes and nodes[parent]["op"] == "rms_norm":
        gid = group_of(parent)
        if gid is not None:
          members[gid].append(addr)
      continue
    if op == "set_rows" and "cache_v" in node["label"]:
      continue
    if op == "X*Y" and base == "ffn_up":
      prev = next((a for a in reversed(order[:order.index(addr)])
                   if nodes[a]["op"] == "X*Y" and base_name(nodes[a]["label"]) == "ffn_gate"), None)
      if prev is not None:
        gid = group_of(prev)
        if gid is not None:
          members[gid].append(addr)
          continue
    new_group(addr)

  # Fold V-cache set_rows into the same-layer flash group.
  for addr in order:
    node = nodes[addr]
    if node["op"] != "set_rows" or "cache_v" not in node["label"]:
      continue
    layer = layer_suffix(node["label"])
    flash_addr = next((a for a in order if nodes[a]["op"] == "flash_attn_ext" and layer_suffix(nodes[a]["label"]) == layer), None)
    if flash_addr is None:
      continue
    gid = group_of(flash_addr)
    if gid is not None:
      remove_from_group(addr, gid)
      members[gid].append(addr)

  # Fold glu into the following same-layer down MMVQ.
  for addr in order:
    node = nodes[addr]
    if node["op"] != "glu":
      continue
    layer = layer_suffix(node["label"])
    down_addr = next((a for a in order[order.index(addr):]
                      if nodes[a]["op"] == "X*Y" and base_name(nodes[a]["label"]) == "ffn_out"
                      and layer_suffix(nodes[a]["label"]) == layer), None)
    if down_addr is None:
      continue
    gid = group_of(down_addr)
    if gid is not None:
      remove_from_group(addr, gid)
      members[gid].append(addr)

  # Attention residual x+y is fused into O and FFN residual into down.  The
  # final layer (35) is the observed exception: its O MMVQ has no fusion and
  # the residual is the explicit get_rows/add sequence in the dot.
  for addr in order:
    node = nodes[addr]
    if node["op"] != "x+y":
      continue
    base = base_name(node["label"])
    layer = layer_suffix(node["label"])
    target = None
    if base == "ffn_inp":
      if layer != 35:
        target = next((a for a in reversed(order[:order.index(addr)])
                       if nodes[a]["op"] == "X*Y" and base_name(nodes[a]["label"]) == "node_"), None)
    elif base == "l_out":
      target = next((a for a in reversed(order[:order.index(addr)])
                     if nodes[a]["op"] == "X*Y" and base_name(nodes[a]["label"]) == "ffn_out"), None)
    if target is not None:
      gid = group_of(target)
      if gid is not None:
        remove_from_group(addr, gid)
        members[gid].append(addr)
        continue
    if group_of(addr) is None:
      new_group(addr)

  # The leading embd get_rows is outside the captured graph replay and is
  # pruned by the CUPTI alignment; keep it as a group for now.
  node_to_group = {a: g for g, addrs in members.items() for a in addrs}
  return {"members": members, "node_to_group": node_to_group}


def kernel_roles(group_nodes: list[dict]) -> list[dict]:
  ops = [n["op"] for n in group_nodes]
  labels = [n["label"] for n in group_nodes]
  if "flash_attn_ext" in ops:
    return [{"kind": "flash_attn_ext_vec", "role": "flash_score"},
            {"kind": "flash_attn_combine_results", "role": "flash_combine"}]
  if "rms_norm" in ops:
    return [{"kind": "rms_norm_f32", "role": "rms_norm"}]
  if "rope" in ops:
    return [{"kind": "rope_neox", "role": "rope"}]
  if "set_rows" in ops:
    return [{"kind": "k_set_rows", "role": "kv_store_k"}]
  if "get_rows" in ops:
    return [{"kind": "k_get_rows_float", "role": "get_rows"}]
  if "x+y" in ops and "X*Y" not in ops:
    return [{"kind": "k_bin_bcast", "role": "elementwise"}]
  if "X*Y" in ops:
    return [{"kind": "quantize_q8_1", "role": "quantize_q8_1"},
            {"kind": "mul_mat_vec_q", "role": "mmq"}]
  return []


def group_semantics(group_nodes: list[dict]) -> dict:
  ops = [n["op"] for n in group_nodes]
  labels = [n["label"] for n in group_nodes]
  if "flash_attn_ext" in ops:
    return {"anchor": None, "segment": "S1"}
  if "set_rows" in ops:
    return {"anchor": None, "segment": "S1"}
  if "X*Y" in ops:
    if any(base_name(x) == "Qcur" for x in labels): return {"anchor": "Q", "segment": "anchor"}
    if any(base_name(x) == "Vcur" for x in labels): return {"anchor": None, "segment": "S1"}
    if any(base_name(x) == "Kcur" for x in labels): return {"anchor": None, "segment": "S1"}
    if any(base_name(x) == "node_" for x in labels): return {"anchor": "O", "segment": "anchor"}
    if any(base_name(x) in ("ffn_gate", "ffn_up") for x in labels): return {"anchor": "gate_up", "segment": "anchor"}
    if any(base_name(x) == "ffn_out" for x in labels): return {"anchor": "down", "segment": "anchor"}
    if any(base_name(x) == "result_output" for x in labels): return {"anchor": "vocab", "segment": "anchor"}
  if "rms_norm" in ops:
    return {"anchor": None, "segment": None}
  if "rope" in ops:
    return {"anchor": None, "segment": "S1"}
  if "get_rows" in ops:
    return {"anchor": None, "segment": "S2"}
  if "x+y" in ops:
    return {"anchor": None, "segment": "S2"}
  return {"anchor": None, "segment": None}


def load_replays(trace: pathlib.Path, graph_id: int) -> list[list[dict]]:
  con = sqlite3.connect(str(trace))
  try:
    names = {int(i): str(v) for i, v in con.execute("select id, value from StringIds")}
    rows = []
    for raw in con.execute(
        "select start,end,graphNodeId,shortName,demangledName,streamId,gridX,gridY,gridZ "
        "from CUPTI_ACTIVITY_KIND_KERNEL where graphId=? order by start", (graph_id,)):
      rows.append({"start": int(raw[0]), "end": int(raw[1]), "graph_node_id": int(raw[2]),
                   "kind": names.get(int(raw[3]), str(raw[3])),
                   "demangled": names.get(int(raw[4]), str(raw[4])) if raw[4] is not None else "",
                   "stream": raw[5], "grid": [raw[6], raw[7], raw[8]]})
    replays, current, seen = [], [], set()
    for row in rows:
      if row["graph_node_id"] in seen:
        replays.append(current)
        current, seen = [], set()
      current.append(row)
      seen.add(row["graph_node_id"])
    if current:
      replays.append(current)
    authority = Counter(frozenset(int(r["graph_node_id"]) for r in replay) for replay in replays).most_common(1)[0][0]
    return [r for r in replays if frozenset(int(x["graph_node_id"]) for x in r) == authority and len(r) == len(authority)]
  finally:
    con.close()


def attach(parsed: dict, folds: dict, replay: list[dict]) -> dict:
  nodes, order = parsed["nodes"], parsed["order"]
  members, node_to_group = folds["members"], folds["node_to_group"]
  groups_in_order: list[int] = []
  for addr in order:
    gid = node_to_group.get(addr)
    if gid is not None and gid not in groups_in_order:
      groups_in_order.append(gid)

  expected_kinds: list[str] = []
  expected_groups: list[int] = []
  for gid in groups_in_order:
    roles = kernel_roles([nodes[a] for a in members[gid]])
    if not roles:
      continue
    expected_groups.append(gid)
    expected_kinds.extend(r["kind"] for r in roles)

  observed_kinds = [r["kind"] for r in replay]
  if len(expected_kinds) != len(observed_kinds):
    if expected_kinds[:1] == ["k_get_rows_float"] and len(expected_kinds) == len(observed_kinds) + 1:
      expected_groups = expected_groups[1:]
      expected_kinds = expected_kinds[1:]
    else:
      raise ValueError(f"expanded {len(expected_kinds)} kernels vs observed {len(observed_kinds)}")
  if expected_kinds != observed_kinds:
    for i, (a, b) in enumerate(zip(expected_kinds, observed_kinds)):
      if a != b:
        raise ValueError(f"kernel mismatch at index {i}: expected {a!r}, observed {b!r}")

  group_kernels: dict[int, list[int]] = defaultdict(list)
  kernel_group: list[int] = []
  kernel_role: list[str] = []
  rows: list[dict] = []
  cursor = 0
  for gid in expected_groups:
    roles = kernel_roles([nodes[a] for a in members[gid]])
    start = len(kernel_group)
    for role in roles:
      kernel_group.append(gid)
      kernel_role.append(role["role"])
      rows.append(replay[cursor])
      cursor += 1
    group_kernels[gid] = list(range(start, len(kernel_group)))

  mmq_fusion = {}
  for i, row in enumerate(rows):
    if "mul_mat_vec_q" in row["demangled"]:
      match = re.search(r"<\(ggml_type\)\d+, \(int\)\d+, \(bool\)(\d)", row["demangled"])
      mmq_fusion[kernel_group[i]] = bool(int(match.group(1))) if match else None
  return {"group_kernels": group_kernels, "kernel_group": kernel_group,
          "kernel_role": kernel_role, "rows": rows,
          "groups_in_order": expected_groups, "mmq_fusion": mmq_fusion}


def build_edges(parsed: dict, folds: dict, attached: dict) -> list[dict]:
  nodes, edges = parsed["nodes"], parsed["edges"]
  node_to_group = folds["node_to_group"]
  members = folds["members"]
  active = set(attached["group_kernels"])

  def producer(addr: str) -> str:
    seen: set[str] = set()
    cur = addr
    while cur in nodes and nodes[cur]["op"] in VIEW_OPS and cur not in seen:
      seen.add(cur)
      parents = [e["from"] for e in edges if e["to"] == cur]
      if len(parents) != 1:
        break
      cur = parents[0]
    return cur

  group_edges: set[tuple[int, int]] = set()
  for gid, addrs in members.items():
    if gid not in active:
      continue
    for addr in addrs:
      for edge in edges:
        if edge["to"] != addr:
          continue
        src = producer(edge["from"])
        sg = node_to_group.get(src)
        if sg is None or sg == gid:
          continue
        group_edges.add((sg, gid))

  out = []
  for sg, dg in sorted(group_edges):
    if sg not in active or dg not in active:
      continue
    out.append({"from": attached["group_kernels"][sg][-1],
                "to": attached["group_kernels"][dg][0], "kind": "RAW"})
  for gid, ks in sorted(attached["group_kernels"].items()):
    for a, b in zip(ks, ks[1:]):
      out.append({"from": a, "to": b, "kind": "RAW"})
  return out


def critical_path(nodes: list[dict], edges: list[dict]) -> float:
  deps: list[list[int]] = [[] for _ in nodes]
  for edge in edges:
    deps[edge["to"]].append(edge["from"])
  memo = {}
  def longest(i: int) -> float:
    if i in memo:
      return memo[i]
    best = 0.0
    for dep in deps[i]:
      best = max(best, longest(dep) + nodes[dep]["duration_us"])
    memo[i] = best
    return best
  return max((longest(i) + nodes[i]["duration_us"] for i in range(len(nodes))), default=0.0)


def emit(trace: pathlib.Path, dot: pathlib.Path, graph_id: int, warmup: int) -> dict:
  parsed = parse_dot(dot)
  folds = build_groups(parsed)
  replays = load_replays(trace, graph_id)
  if len(replays) <= warmup:
    raise ValueError(f"need more than warmup={warmup} complete replays, got {len(replays)}")
  steady = replays[warmup:]
  spans = [max(int(r["end"]) for r in replay) - min(int(r["start"]) for r in replay) for replay in steady]
  chosen = steady[spans.index(statistics.median(spans))]
  attached = attach(parsed, folds, chosen)
  edges = build_edges(parsed, folds, attached)
  t0 = min(int(r["start"]) for r in chosen)
  out_nodes = []
  for i, row in enumerate(attached["rows"]):
    gid = attached["kernel_group"][i]
    sem = group_semantics([parsed["nodes"][a] for a in folds["members"][gid]])
    out_nodes.append({"id": i, "name": row["kind"], "role": attached["kernel_role"][i],
                      "duration_us": round((int(row["end"]) - int(row["start"])) / 1000.0, 3),
                      "start_us": round((int(row["start"]) - t0) / 1000.0, 3),
                      "end_us": round((int(row["end"]) - t0) / 1000.0, 3),
                      "stream": row["stream"], "group_id": gid,
                      "semantic": sem, "mmq_fusion": attached["mmq_fusion"].get(gid)})
  cp = critical_path(out_nodes, edges)
  node_sum = sum(n["duration_us"] for n in out_nodes)
  span = max(n["end_us"] for n in out_nodes) - min(n["start_us"] for n in out_nodes)
  anchors = [{"kind": n["semantic"]["anchor"], "layer": None, "node": n["id"], "duration_us": n["duration_us"]}
             for n in out_nodes if n["semantic"]["anchor"]]
  layer = -1
  for anchor in anchors:
    if anchor["kind"] == "Q":
      layer += 1
    anchor["layer"] = layer if anchor["kind"] != "vocab" else "final"
  return {
    "schema": SCHEMA,
    "provenance": {"trace": str(trace), "trace_sha256": _sha256(trace), "graph_id": graph_id,
                   "dot": str(dot), "dot_sha256": _sha256(dot),
                   "warmup_replays_dropped": warmup, "steady_replays": len(steady),
                   "chosen_span_us": round(span, 3)},
    "summary": {"node_count": len(out_nodes), "edge_count": len(edges),
                "node_sum_us": round(node_sum, 3), "span_us": round(span, 3),
                "overlap_mass_us": round(node_sum - span, 3),
                "critical_path_us": round(cp, 3),
                "role_counts": dict(sorted(Counter(n["role"] for n in out_nodes).items())),
                "anchor_counts": dict(sorted(Counter(a["kind"] for a in anchors).items()))},
    "nodes": out_nodes, "edges": edges, "anchors": anchors,
  }


def parse_dump(path: pathlib.Path) -> dict:
  """Parse the GGML_CUDA_GRAPH_DUMP real CUDA graph edge table."""
  graph_id = None
  nodes: list[dict] = []
  edges: list[dict] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    parts = line.split()
    if not parts:
      continue
    if parts[0] == "graph_id":
      graph_id = int(parts[1])
    elif parts[0] == "node":
      nodes.append({"local_id": int(parts[1]), "tools_id": int(parts[2]),
                    "node_type": int(parts[3]), "name": parts[4] if len(parts) > 4 else "",
                    "grid": [int(x) for x in parts[5:8]] if len(parts) > 7 else [0, 0, 0],
                    "block": [int(x) for x in parts[8:11]] if len(parts) > 10 else [0, 0, 0]})
    elif parts[0] == "edge":
      edges.append({"from": int(parts[1]), "to": int(parts[2]),
                    "from_port": int(parts[3]), "to_port": int(parts[4]),
                    "type": int(parts[5]) if len(parts) > 5 else -1})
  if graph_id is None or not nodes:
    raise ValueError("dump is missing graph_id or node rows")
  return {"graph_id": graph_id, "nodes": nodes, "edges": edges,
          "edge_port_names": {"from_port_1": "cudaGraphKernelNodePortProgrammatic",
                              "to_port_0": "cudaGraphKernelNodePortLaunchCompletion",
                              "type_1": "cudaGraphDependencyTypeProgrammatic",
                              "type_0": "cudaGraphDependencyTypeDefault"}}


def _real_kind(name: str) -> str:
  if "rms_norm_f32ILi1024" in name:
    return "norm1024"
  if "rms_norm_f32ILi256" in name:
    return "norm256"
  if name.startswith("_Z13quantize"):
    return "quant"
  if "mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0" in name:
    return "mmq_t12_nf"
  if "mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb0" in name:
    return "mmq_t12_f"
  if "mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0" in name:
    return "mmq_t14_nf"
  if "mul_mat_vec_qIL9ggml_type14ELi1ELb1ELb0" in name:
    return "mmq_t14_f"
  if "rope_neoxILb1ELb0Eff" in name:
    return "rope_f32"
  if "rope_neoxILb1ELb0Ef6__half" in name:
    return "rope_half"
  if "k_set_rows" in name:
    return "set_rows"
  if "flash_attn_ext" in name:
    return "flash"
  if "flash_attn_combine" in name:
    return "combine"
  if "k_get_rows" in name:
    return "get_rows"
  if "bin_bcast" in name:
    return "binbcast"
  return "other"


def classify_real(nodes: list[dict]) -> list[dict]:
  """Attach layer/role/semantic fields using llama's fixed decode launch order."""
  roles_std = ["attn_norm", "Q_quant", "Q", "q_norm", "q_rope", "K_quant", "K",
               "V_quant", "V", "k_norm", "k_rope", "k_store", "flash", "combine",
               "O_quant", "O", "ffn_norm", "G_quant", "G", "D_quant", "D"]
  roles_last = roles_std[:16] + ["get_rows_a", "get_rows_b", "binbcast"] + roles_std[16:]
  out: list[dict] = []
  for layer in range(36):
    start = 2 + 21 * layer - 2
    roles = roles_last if layer == 35 else roles_std
    for position, role in enumerate(roles):
      node = nodes[start + position]
      node.update({"layer": layer, "role": role, "kind": _real_kind(node["name"])})
      out.append(node)
  for local_id in (759, 760, 761):
    node = nodes[local_id]
    role = "final_norm" if node["local_id"] == 759 else ("vocab_quant" if node["local_id"] == 760 else "vocab")
    node.update({"layer": 36, "role": role, "kind": _real_kind(node["name"])})
    out.append(node)
  out.sort(key=lambda n: n["local_id"])
  anchor_roles = frozenset(("Q", "O", "G", "D", "vocab"))
  for node in out:
    role = node["role"]
    if role in anchor_roles:
      semantic = {"anchor": role, "segment": "anchor"}
    elif role in ("q_norm", "q_rope", "flash", "combine", "O_quant"):
      semantic = {"anchor": None, "segment": "S1"}
    elif role in ("ffn_norm", "G_quant"):
      semantic = {"anchor": None, "segment": "S2"}
    elif role == "D_quant":
      semantic = {"anchor": None, "segment": "S3"}
    elif role in ("attn_norm", "Q_quant"):
      semantic = {"anchor": None, "segment": "S0" if node.get("layer") == 0 else "S4"}
    elif role in ("final_norm", "vocab_quant"):
      semantic = {"anchor": None, "segment": "tail"}
    elif role in ("get_rows_a", "get_rows_b", "binbcast"):
      semantic = {"anchor": None, "segment": "tail"}
    else:
      semantic = {"anchor": None, "segment": "S1_off_path"}
    node["semantic"] = semantic
  return out


def build_real_edges(nodes: list[dict]) -> list[dict]:
  """Infer the logical data-dependency DAG behind llama's programmatic edge chain."""
  by_local = {n["local_id"]: n for n in nodes}
  layers: dict[int, dict[str, list[dict]]] = {}
  for node in nodes:
    if node["layer"] < 36:
      layers.setdefault(node["layer"], defaultdict(list))[node["role"]].append(node)
  out: list[dict] = []

  def edge(a: dict, b: dict, kind: str = "DATA") -> None:
    out.append({"from": a["local_id"], "to": b["local_id"], "kind": kind})

  for layer in range(36):
    by_role = layers[layer]
    if layer == 0:
      for role in ("Q_quant", "K_quant", "V_quant"):
        edge(by_role["attn_norm"][0], by_role[role][0])
    else:
      prev_down = layers[layer - 1]["D"][0]
      edge(prev_down, by_role["attn_norm"][0])
      for role in ("Q_quant", "K_quant", "V_quant"):
        edge(by_role["attn_norm"][0], by_role[role][0])
    for role, anchor in (("Q_quant", "Q"), ("K_quant", "K"), ("V_quant", "V")):
      edge(by_role[role][0], by_role[anchor][0])
    edge(by_role["Q"][0], by_role["q_norm"][0])
    edge(by_role["q_norm"][0], by_role["q_rope"][0])
    edge(by_role["q_rope"][0], by_role["flash"][0])
    edge(by_role["K"][0], by_role["k_norm"][0])
    edge(by_role["k_norm"][0], by_role["k_rope"][0])
    edge(by_role["k_rope"][0], by_role["k_store"][0])
    edge(by_role["k_store"][0], by_role["flash"][0])
    edge(by_role["V"][0], by_role["flash"][0])
    edge(by_role["flash"][0], by_role["combine"][0])
    edge(by_role["combine"][0], by_role["O_quant"][0])
    edge(by_role["O_quant"][0], by_role["O"][0])
    if layer < 35:
      edge(by_role["O"][0], by_role["ffn_norm"][0])
      edge(by_role["ffn_norm"][0], by_role["G_quant"][0])
      edge(by_role["G_quant"][0], by_role["G"][0])
      edge(by_role["G"][0], by_role["D_quant"][0])
      edge(by_role["D_quant"][0], by_role["D"][0])

  last = layers[35]
  prev_down = layers[34]["D"][0]
  edge(prev_down, last["get_rows_a"][0])
  edge(last["O"][0], last["get_rows_b"][0])
  edge(last["get_rows_a"][0], last["binbcast"][0])
  edge(last["get_rows_b"][0], last["binbcast"][0])
  edge(last["binbcast"][0], last["ffn_norm"][0])
  edge(last["ffn_norm"][0], last["G_quant"][0])
  edge(last["G_quant"][0], last["G"][0])
  edge(last["G"][0], last["D_quant"][0])
  edge(last["D_quant"][0], last["D"][0])
  tail = [by_local[x] for x in (759, 760, 761)]
  edge(last["D"][0], tail[0])
  edge(tail[0], tail[1])
  edge(tail[1], tail[2])
  return out


def emit_real(trace: pathlib.Path, dump_path: pathlib.Path, graph_id: int, warmup: int) -> dict:
  dump = parse_dump(dump_path)
  if dump["graph_id"] not in (graph_id, graph_id - 1):
    # Capture-time cudaGraphGetId can differ by one from the runtime CUPTI graph id.
    pass
  replays = load_replays(trace, graph_id)
  if len(replays) <= warmup:
    raise ValueError(f"need more than warmup={warmup} complete replays, got {len(replays)}")
  steady = replays[warmup:]
  spans = [max(int(r["end"]) for r in replay) - min(int(r["start"]) for r in replay) for replay in steady]
  chosen = steady[spans.index(statistics.median(spans))]
  by_local = {int(r["graph_node_id"]) & 0xffffffff: r for r in chosen}
  if len(by_local) != len(dump["nodes"]):
    raise ValueError(f"joined {len(by_local)} CUPTI rows to {len(dump['nodes'])} dump nodes")
  nodes = classify_real(dump["nodes"])
  t0 = min(int(r["start"]) for r in chosen)
  for node in nodes:
    row = by_local[node["local_id"]]
    node.update({"duration_us": round((int(row["end"]) - int(row["start"])) / 1000.0, 3),
                 "start_us": round((int(row["start"]) - t0) / 1000.0, 3),
                 "end_us": round((int(row["end"]) - t0) / 1000.0, 3),
                 "stream": row.get("stream")})
  data_edges = build_real_edges(nodes)
  cp = critical_path(nodes, data_edges)
  node_sum = sum(n["duration_us"] for n in nodes)
  span = max(n["end_us"] for n in nodes) - min(n["start_us"] for n in nodes)
  anchors = [{"kind": n["role"], "layer": n["layer"], "node": n["local_id"], "duration_us": n["duration_us"]}
             for n in nodes if n["role"] in ("Q", "O", "G", "D", "vocab")]
  return {
    "schema": "tinygrad.llama_real_edge_dag.v1",
    "provenance": {"trace": str(trace), "trace_sha256": _sha256(trace), "graph_id": graph_id,
                   "dump": str(dump_path), "dump_sha256": _sha256(dump_path),
                   "dump_graph_id": dump["graph_id"], "warmup_replays_dropped": warmup,
                   "steady_replays": len(steady), "chosen_span_us": round(span, 3)},
    "edge_semantics": dump["edge_port_names"],
    "summary": {"node_count": len(nodes), "raw_edge_count": len(dump["edges"]),
                "data_edge_count": len(data_edges), "node_sum_us": round(node_sum, 3),
                "span_us": round(span, 3), "overlap_mass_us": round(node_sum - span, 3),
                "critical_path_us_serialized": round(cp, 3),
                "role_counts": dict(sorted(Counter(n["role"] for n in nodes).items())),
                "segment_counts": dict(sorted(Counter(n["semantic"]["segment"] for n in nodes).items()))},
    "nodes": nodes, "data_edges": data_edges, "raw_edges": dump["edges"], "anchors": anchors,
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--trace", required=True, type=pathlib.Path)
  ap.add_argument("--dot", type=pathlib.Path)
  ap.add_argument("--dump", type=pathlib.Path)
  ap.add_argument("--graph-id", required=True, type=int)
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--out", required=True, type=pathlib.Path)
  args = ap.parse_args()
  if args.dump is not None:
    result = emit_real(args.trace, args.dump, args.graph_id, args.warmup)
  elif args.dot is not None:
    result = emit(args.trace, args.dot, args.graph_id, args.warmup)
  else:
    ap.error("one of --dot or --dump is required")
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result["summary"], indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
