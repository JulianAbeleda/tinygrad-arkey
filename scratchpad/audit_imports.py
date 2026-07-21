import ast, os, sys, json
from collections import defaultdict

ROOT = "/home/ubuntu/tinygrad-arkey"

def all_py_files(base):
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in filenames:
            if f.endswith(".py"):
                out.append(os.path.join(dirpath, f))
    return out

files = all_py_files(os.path.join(ROOT, "extra")) + all_py_files(os.path.join(ROOT, "tinygrad"))
# module name -> file path, using dotted path relative to ROOT
mod2file = {}
file2mod = {}
for f in files:
    rel = os.path.relpath(f, ROOT)
    if rel.endswith("__init__.py"):
        mod = rel[:-len("/__init__.py")].replace("/", ".")
    else:
        mod = rel[:-3].replace("/", ".")
    mod2file[mod] = f
    file2mod[f] = mod

def resolve_import(modname, current_mod):
    # try exact match, then try as package (with __init__)
    if modname in mod2file:
        return modname
    return None

edges = defaultdict(set)  # file -> set of files it imports (within our tree)
unresolved = defaultdict(set)

for f in files:
    try:
        src = open(f, "r", encoding="utf-8", errors="replace").read()
        tree = ast.parse(src, filename=f)
    except SyntaxError as e:
        print(f"SYNTAX ERROR {f}: {e}", file=sys.stderr)
        continue
    cur_mod = file2mod[f]
    cur_pkg_parts = cur_mod.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                # try full and prefixes
                parts = name.split(".")
                for i in range(len(parts), 0, -1):
                    cand = ".".join(parts[:i])
                    if cand in mod2file:
                        edges[f].add(mod2file[cand])
                        break
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # relative import
                base_parts = cur_pkg_parts[:-1]  # dir containing current file's module path
                # for level=1, go up 0 dirs from package dir (module's own dir); for level=2 up 1, etc.
                up = node.level - 1
                if up > 0:
                    base_parts = base_parts[:-up] if up <= len(base_parts) else []
                mod_prefix = base_parts
                if node.module:
                    mod_prefix = base_parts + node.module.split(".")
                cand = ".".join(mod_prefix)
                resolved = False
                if cand in mod2file:
                    edges[f].add(mod2file[cand])
                    resolved = True
                # also try importing names as submodules e.g. from .prefill import x where x is a submodule
                for alias in node.names:
                    subcand = cand + "." + alias.name
                    if subcand in mod2file:
                        edges[f].add(mod2file[subcand])
                        resolved = True
                if not resolved:
                    unresolved[f].add((node.level, node.module, tuple(a.name for a in node.names)))
            else:
                if node.module:
                    parts = node.module.split(".")
                    matched = False
                    for i in range(len(parts), 0, -1):
                        cand = ".".join(parts[:i])
                        if cand in mod2file:
                            edges[f].add(mod2file[cand])
                            matched = True
                            break
                    # also handle "from extra.qk import X" where X is submodule extra.qk.X
                    if node.module in mod2file or True:
                        for alias in node.names:
                            subcand = node.module + "." + alias.name
                            if subcand in mod2file:
                                edges[f].add(mod2file[subcand])
                                matched = True
                    if not matched:
                        unresolved[f].add((0, node.module, tuple(a.name for a in node.names)))

# Save graph
import pickle
with open("/tmp/audit_graph.pkl", "wb") as fh:
    pickle.dump({"edges": dict(edges), "mod2file": mod2file, "file2mod": file2mod, "unresolved": dict(unresolved)}, fh)

print("files scanned:", len(files))
print("edges from extra/qk files:", sum(1 for f in edges if "/extra/qk/" in f))
