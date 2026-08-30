#!/usr/bin/env python3
"""Executable off/on proof for the inert HCQ submission and Buffer observers."""
import argparse, gc, json, os, pathlib
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.device import BUFFER_OBSERVER, Buffer, Device, install_buffer_observer, remove_buffer_observer

def _run_graph(rounds:int):
  @TinyJit
  def graph(x):
    y = (x + 1).realize()
    return (y * 2).realize()
  x = Tensor.ones(256, device="NV").realize()
  for _ in range(rounds): graph(x).realize()
  Device["NV"].synchronize()
  del graph
  gc.collect()

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--submission-jsonl", type=pathlib.Path, required=True)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  ap.add_argument("--rounds", type=int, default=5)
  args = ap.parse_args()
  if args.rounds < 4: raise SystemExit("--rounds must be at least 4 to form and replay an HCQ graph")
  args.submission_jsonl.parent.mkdir(parents=True, exist_ok=True)
  args.submission_jsonl.unlink(missing_ok=True)

  os.environ.pop("HCQ_SUBMISSION_OBSERVER_JSON", None)
  assert BUFFER_OBSERVER.get() is None
  _run_graph(args.rounds)
  observer_off_emitted = args.submission_jsonl.exists()

  events = []
  token = install_buffer_observer(lambda **event: events.append(event))
  try:
    buf = Buffer("NV", 4, dtypes.float32).allocate()
    buf.copyin(memoryview(bytearray(16)))
    buf.copyout(memoryview(bytearray(16)))
    os.environ["HCQ_SUBMISSION_OBSERVER_JSON"] = str(args.submission_jsonl)
    _run_graph(args.rounds)
  finally:
    os.environ.pop("HCQ_SUBMISSION_OBSERVER_JSON", None)
    remove_buffer_observer(token)

  submissions = [json.loads(line) for line in args.submission_jsonl.read_text().splitlines() if line.strip()]
  kinds = sorted({event["kind"] for event in events})
  timestamp_contract = all(entry["end"] >= entry["start"] and entry["dependency_ready"] <= entry["start"]
                           for submission in submissions for entry in submission["entries"])
  result = {"schema":"tinygrad.hcq_submission_observer_selftest.v1",
    "status":"PASS" if not observer_off_emitted and submissions and {"alloc","copyin","copyout"} <= set(kinds) and timestamp_contract else "FAIL",
    "observer_off_emitted":observer_off_emitted, "submission_records":len(submissions),
    "entry_records":sum(len(row["entries"]) for row in submissions), "buffer_events":len(events),
    "buffer_kinds":kinds, "timestamp_contract":timestamp_contract,
    "submission_jsonl":str(args.submission_jsonl)}
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  print(json.dumps(result, sort_keys=True))
  if result["status"] != "PASS": raise SystemExit(1)

if __name__ == "__main__": main()
