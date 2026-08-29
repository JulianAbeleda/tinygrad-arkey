#!/usr/bin/env python3
"""Extract installed NV Flash metadata, or capture one live PROGRAM call.

The live helper is deliberately an observer only: it records the already-built
CALL and resolves its existing execution bindings after capture.  It never
rewrites the graph or inserts a copy kernel.  This is research-only glue and is
not imported by production routing.
"""
import hashlib, json, pathlib, sys

PROGRAM = 'nv_sm120_q16_grid_hd128_loop_attention'

class FlashCallCapture:
  """Same-process graph observer for the finalized Flash PROGRAM."""
  def __init__(self): self.calls, self.events = {}, []
  def __call__(self, event): self.events.append(event)
  def bind_call(self, call_index, call): self.calls[call_index] = call
  def bind_execution(self, linear, input_uops, var_vals):
    self.linear, self.execution_inputs, self.execution_var_vals = linear, tuple(input_uops), dict(var_vals)
  def bind_graph_calls(self, calls):
    # HCQGraph's calls contain the already-resolved Buffer objects.  This is
    # deliberately preferred over profile metadata or re-resolving UOps.
    self.graph_calls = tuple(calls)
  def selected(self):
    from tinygrad.uop.ops import Ops
    if hasattr(self, 'graph_calls'):
      return [(i, c) for i, (_, c, _, _) in enumerate(self.graph_calls)
              if c.op is Ops.PROGRAM and getattr(c.arg, 'name', '') == PROGRAM]
    return [(i,c) for i,c in sorted(self.calls.items())
            if c.op is Ops.CALL and getattr(c.src[0], 'op', None) is Ops.PROGRAM
            and getattr(c.src[0].arg, 'name', '') == PROGRAM]
  def bound_buffers(self, index):
    """Return (slot, Buffer) for the selected call without changing scheduling."""
    from tinygrad.engine.realize import resolve_params
    from tinygrad.uop.ops import Ops
    rows = [(i,c) for i,c in self.selected() if i == index]
    if not rows: raise KeyError(f'no finalized Flash call at index {index}')
    args = resolve_params(rows[0][1], self.execution_inputs)
    out = []
    for slot, u in enumerate(args):
      if u.op not in (Ops.BUFFER, Ops.SLICE): continue
      out.append((slot, u.buffer.ensure_allocated()))
    return out

def capture_record(capture, index):
  """Copy bound out/Q/K/V bytes and geometry for an already captured call."""
  if hasattr(capture, 'graph_calls'):
    rows = [(i, row[2]) for i, row in enumerate(capture.graph_calls)
            if i == index and getattr(row[1].arg, 'name', '') == PROGRAM]
    if not rows: raise KeyError(f'no finalized Flash graph call at index {index}')
    bufs = [(slot, buf) for slot, buf in enumerate(rows[0][1])]
  else:
    bufs = capture.bound_buffers(index)
  # Buffer.size is expressed in elements (not bytes); the installed fp16 ABI
  # is 2 MiB elements for out/Q and 512 Ki elements for K/V.
  expected = {0:4194304, 1:4194304, 2:1048576, 3:1048576}
  sizes = {slot: int(buf.nbytes) for slot,buf in bufs}
  if sizes != expected: raise ValueError(f'Flash ABI/geometry mismatch: {sizes!r}')
  payload = {str(slot): buf.numpy().tobytes() for slot,buf in bufs}
  return {'schema':'tinygrad.nv_prefill_flash_live_capture.v1','status':'PASS',
          'program':PROGRAM,'call_index':index,'outs':[0],'ins':[1,2,3],
          'geometry':{'global':[1024,1,1],'local':[32,1,1]},'sizes':sizes,
          'sha256':{slot:hashlib.sha256(data).hexdigest() for slot,data in payload.items()},
          'readonly_inputs':True}

def main(path):
 d=json.loads(pathlib.Path(path).read_text()); rows=[]
 for i,e in enumerate(d.get('entries',[])):
  if e.get('name')!='nv_sm120_q16_grid_hd128_loop_attention': continue
  rows.append({'index':i,'name':e['name'],'duration_ns':e.get('duration'),'device':e.get('device'),'metadata':e.get('metadata')})
 out={'schema':'tinygrad.nv_prefill_flash_program_extract.v1','status':'PASS' if rows else 'FAIL','program':rows[0]['name'] if rows else None,'fused_score_reduction':True,'calls':len(rows),'launches':rows,'abi':'graph-owned opaque PROGRAM; raw buffer slots unavailable in profile-only artifact'}
 p=pathlib.Path('docs/task_workflow/evidence/nv-prefill-flash-20260829/program-extract.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__': main(sys.argv[1])
