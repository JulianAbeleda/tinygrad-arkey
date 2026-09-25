"""Analyze ours_prof output. usage: ours_an.py tag [classes.json]
Per kernel name (hash stripped): count/step, us/step, metadata; per class sums; busy union vs wall."""
import json, sys, collections, re
S = "/home/ubuntu/tinygrad-self-training/docs/nemotron-vllm-parity/bench/spec/"
tag = sys.argv[1]
meta = json.load(open(S + f"ours_{tag}_meta.json"))
rows = [json.loads(l) for l in open(S + f"ours_{tag}.jsonl")]
l0, l1 = meta["lines"]
win = rows[l0:l1]
n = meta["nsteps"]
def strip(nm): return re.sub(r"_[0-9a-f]{64}$", "", re.sub(r"\x1b\[[0-9;]*m", "", nm))
# metadata by name from captured lists, in order; position of first occurrence
order, mdmap = {}, collections.defaultdict(set)
lists = list(meta.get("step", {}).values())
steplist = lists[-1] if lists else []
for i, k in enumerate(steplist + meta.get("flush", [])):
  nm = strip(k["name"]); order.setdefault(nm, i); mdmap[nm].update(k["md"][:6])
flushnames = {strip(k["name"]) for k in meta.get("flush", [])}
agg = collections.defaultdict(lambda: [0, 0.0])
iv = []
for r in win:
  for e in r["entries"]:
    nm = strip(e["name"]); a = agg[nm]; a[0] += 1; a[1] += float(e["duration"])
    iv.append((float(e["start"]), float(e["end"])))
iv.sort(); busy = 0; cs, ce = iv[0]
for s, e in iv[1:]:
  if s > ce: busy += ce - cs; cs, ce = s, e
  else: ce = max(ce, e)
busy += ce - cs
span = iv[-1][1] - iv[0][0]
tot = sum(v[1] for v in agg.values())
print(f"{tag}: wall/step(host) {meta['wall_ms']:.2f} ms  span/step {span/n/1e3:.2f}  busy-union/step {busy/n/1e3:.2f}  "
      f"sum-kernel/step {tot/n/1e3:.2f}  kernels/step {sum(v[0] for v in agg.values())/n:.0f}  graphs/step {len(win)/n:.2f}")
cls = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
byc = collections.defaultdict(lambda: [0, 0.0])
for nm, (c, d) in sorted(agg.items(), key=lambda x: order.get(x[0], 1e9)):
  k = cls.get(nm, "?"); byc[k][0] += c; byc[k][1] += d
  if len(sys.argv) <= 3:
    print(f"{order.get(nm,-1):4d} {d/n:8.1f}us {c/n:5.1f}x {d/c:7.1f} {'F' if nm in flushnames else ' '} {k:12s} {nm[:48]:48s} {sorted(mdmap[nm])[:5]}")
for k, (c, d) in sorted(byc.items(), key=lambda x: -x[1][1]):
  print(f"CLASS {k:14s} {c/n:6.1f}/step {d/n/1e3:7.3f} ms")
