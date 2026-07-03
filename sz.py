#!/usr/bin/env python3
# adapted from upstream tinygrad sz.py: token-based line counter with a hard budget.
# counts BUDGET_DIRS against MAX_LINE_COUNT (env override); INFO_DIRS are reported but unbudgeted.
import os, sys, token, tokenize, itertools

BUDGET_DIRS = ["tinygrad", "bench", "structure"]
INFO_DIRS = ["extra"]
EXCLUDE = ["tinygrad/runtime/autogen", "tinygrad/viz/assets"]
DEFAULT_MAX_LINE_COUNT = 25000

TOKEN_WHITELIST = [token.OP, token.NAME, token.NUMBER, token.STRING]

def is_docstring(t):
  return t.type == token.STRING and t.string.startswith('"""') and t.line.strip().startswith('"""')

def is_js_token(s): return len(s) and not s.startswith('//')

def gen_stats(base_path="."):
  table = []
  for top in BUDGET_DIRS + INFO_DIRS:
    for path, _, files in os.walk(os.path.join(base_path, top)):
      for name in files:
        if not (name.endswith(".py") or name.endswith(".js")): continue
        if any(s in path.replace('\\', '/') for s in EXCLUDE): continue
        filepath = os.path.join(path, name)
        relfilepath = os.path.relpath(filepath, base_path).replace('\\', '/')
        if name.endswith(".js"):
          with open(filepath) as file_: lines = [line.strip() for line in file_.readlines()]
          token_count, line_count = sum(len(line.split()) for line in lines if is_js_token(line)), sum(1 for line in lines if is_js_token(line))
        else:
          try:
            with tokenize.open(filepath) as file_:
              tokens = [t for t in tokenize.generate_tokens(file_.readline) if t.type in TOKEN_WHITELIST and not is_docstring(t)]
          except (tokenize.TokenizeError, SyntaxError) as e:
            print(f"WARNING: failed to tokenize {relfilepath}: {e}", file=sys.stderr); continue
          token_count, line_count = len(tokens), len(set([x for t in tokens for x in range(t.start[0], t.end[0]+1)]))
        if line_count > 0: table.append([relfilepath, line_count, token_count/line_count])
  return table

if __name__ == "__main__":
  table = gen_stats(sys.argv[1] if len(sys.argv) == 2 else ".")
  if os.getenv("SZ_FILES", "0") == "1":
    for name, lines, tpl in sorted(table, key=lambda x: -x[1]): print(f"{name:80s} {lines:6d} {tpl:6.1f}")
    print()
  groups = sorted([('/'.join(x[0].rsplit("/", 1)[0].split("/")[0:2]), x[1], x[2]) for x in table])
  for dir_name, _group in itertools.groupby(groups, key=lambda x: x[0]):
    group = list(_group)
    print(f"{dir_name:30s} : {sum(x[1] for x in group):6d} in {len(group):3d} files")
  budgeted = sum(x[1] for x in table if x[0].split("/")[0] in BUDGET_DIRS)
  info = sum(x[1] for x in table if x[0].split("/")[0] in INFO_DIRS)
  print()
  print(f"    extra (unbudgeted): {info}")
  max_line_count = int(os.getenv("MAX_LINE_COUNT", str(DEFAULT_MAX_LINE_COUNT)))
  print(f"budgeted lines: {budgeted} / {max_line_count}")
  assert max_line_count == -1 or budgeted <= max_line_count, f"OVER BUDGET: {budgeted} > {max_line_count} LINES"
