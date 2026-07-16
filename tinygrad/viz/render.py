import itertools
from urllib.parse import parse_qs, urlparse
from tinygrad.helpers import getenv
from tinygrad.uop.render import print_uops
from tinygrad.viz.graph import VizData, _reconstruct, get_full_rewrite
from tinygrad.viz.profile import get_profile, row_tuple, soft_err, sqtt_timeline, unpack_pmc
from tinygrad.viz.amd import amdgpu_cfg, get_stdout

def get_int(query:dict[str, list[str]], k:str) -> int: return int(query.get(k,["0"])[0])

# ** Main render function to get the complete details about a trace event

def get_render(viz_data:VizData, query:str) -> dict:
  url = urlparse(query)
  i, j, fmt = get_int(qs:=parse_qs(url.query), "ctx"), get_int(qs, "step"), url.path.lstrip("/")
  data = viz_data.ctxs[i]["steps"][j]["data"]
  if fmt == "graph-rewrites": return {"value":get_full_rewrite(viz_data, viz_data.trace.rewrites[i][j]), "content_type":"text/event-stream"}
  if fmt == "uops": return {"src":get_stdout(lambda: print_uops(_reconstruct(viz_data, viz_data.trace.rewrites[i][j-1].sink).src[2].src))}
  if fmt == "code": return {"src":data, "lang":"cpp"}
  if fmt == "asm":
    ret:dict = {}
    renderer, lib = data
    if renderer.target.arch.startswith("gfx"):
      with soft_err(lambda err: ret.update(err)): ret.update(amdgpu_cfg(lib, renderer.target.arch))
    else: ret["src"] = get_stdout(lambda: renderer.compiler.disassemble(lib))
    return ret
  if fmt == "all-pmc":
    durations, pmc = data
    ret = {"cols":{}, "rows":[]}
    for (name, n, k),events in pmc.items():
      pmc_table = unpack_pmc(events)
      ret["cols"].update([(r[0], None) for r in pmc_table["rows"]])
      ret["rows"].append((name, durations[k][n-1], *[r[1] for r in pmc_table["rows"]]))
    ret["cols"] = ["Kernel", "Duration", *ret["cols"]]
    return ret
  if fmt == "prg-pmc": return unpack_pmc(data)
  if fmt.startswith("sqtt"):
    ret = {}
    with soft_err(lambda err:ret.update(err)):
      if (events:=get_profile(viz_data, list(itertools.islice(sqtt_timeline(*data), getenv("MAX_SQTT_PKTS", 50_000))), sort_fn=row_tuple)):
        ret = {"value":events, "content_type":"application/octet-stream"}
      else: ret = {"src":"No SQTT trace on this SE."}
    return ret
  # viewers for the amd decoder in extra
  if fmt.startswith("amd-sqtt"): return data["fxn"](viz_data, i, j, *data["args"])
  if fmt == "cu-sqtt": return {"value":get_profile(viz_data, data, sort_fn=row_tuple), "content_type":"application/octet-stream"}
  return data
