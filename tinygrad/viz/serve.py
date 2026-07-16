#!/usr/bin/env python3
import argparse, multiprocessing, os, pickle, socket, sys, threading, time, webbrowser
from tinygrad.helpers import colored, Context, getenv, temp
from tinygrad.uop.ops import RewriteTrace
# Compatibility facade: these lived in serve.py before the ownership split.
from tinygrad.viz.graph import *
from tinygrad.viz.graph import _reconstruct
from tinygrad.viz.profile import *
from tinygrad.viz.amd import *
from tinygrad.viz.render import *
from tinygrad.viz.http import *

# ** main loop

def reloader():
  mtime = os.stat(__file__).st_mtime
  while not stop_reloader.is_set():
    if mtime != os.stat(__file__).st_mtime:
      print("reloading server...")
      os.execv(sys.executable, [sys.executable] + sys.argv)
    time.sleep(0.1)

# unpickling may load libraries, turn off DEBUG=3 output
@Context(DEBUG=0)
def load_pickle(path:str, default):
  if not os.path.exists(path): return default
  with open(path, "rb") as f: return pickle.load(f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--rewrites-path', type=str, help='Path to rewrites', default=temp("rewrites.pkl", append_user=True))
  parser.add_argument('--profile-path', type=str, help='Path to profile', default=temp("profile.pkl", append_user=True))
  args = parser.parse_args()

  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    if s.connect_ex(((HOST:="http://127.0.0.1").replace("http://", ""), PORT:=getenv("PORT", 8000))) == 0:
      raise RuntimeError(f"{HOST}:{PORT} is occupied! use PORT= to change.")
  stop_reloader = threading.Event()
  multiprocessing.current_process().name = "VizProcess"
  Context(ALLOW_DEVICE_USAGE=0).__enter__()                # disallow opening of devices
  st = time.perf_counter()
  print("*** viz is starting")

  data = VizData(load_pickle(args.rewrites_path, default=RewriteTrace([], [], {})))
  load_rewrites(data)
  profile_ret = get_profile(data, load_pickle(args.profile_path, default=[]))
  Handler.data, Handler.profile_ret = data, profile_ret

  server = TCPServerWithReuse(('', PORT), Handler)
  reloader_thread = threading.Thread(target=reloader)
  reloader_thread.start()
  print(colored(f"*** ready in {(time.perf_counter()-st)*1e3:4.2f}ms", "green"), flush=True)
  if len(getenv("BROWSER", "")) > 0: webbrowser.open(f"{HOST}:{PORT}")
  try: server.serve_forever()
  except KeyboardInterrupt:
    print("*** viz is shutting down...")
    stop_reloader.set()
