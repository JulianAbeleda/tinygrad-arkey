import json, re, shutil, subprocess, threading, unittest
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

from tinygrad.viz.graph import VizData
from tinygrad.viz.http import Handler, TCPServerWithReuse


VIZ_ROOT = Path(__file__).parents[2] / "tinygrad" / "viz"


class TestVizApplicationBoundary(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls._old_data, cls._old_profile_ret = Handler.data, Handler.profile_ret
    Handler.data, Handler.profile_ret = VizData(), None
    cls.server = TCPServerWithReuse(("127.0.0.1", 0), Handler)
    cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
    cls.thread.start()
    cls.origin = f"http://127.0.0.1:{cls.server.server_address[1]}"

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()
    cls.thread.join(timeout=5)
    Handler.data, Handler.profile_ret = cls._old_data, cls._old_profile_ret

  @classmethod
  def get(cls, path):
    with urlopen(cls.origin + path, timeout=5) as response:
      return response.status, response.headers.get_content_type(), response.read()

  def test_index_and_complete_module_graph_are_served(self):
    status, content_type, body = self.get("/")
    self.assertEqual((status, content_type), (200, "text/html"))

    html = body.decode()
    entries = re.findall(r'<script[^>]+type=["\']module["\'][^>]+src=["\']([^"\']+)', html)
    self.assertEqual(entries, ["/js/index.js"])

    pending, seen = list(entries), set()
    while pending:
      path = pending.pop()
      if path in seen: continue
      status, content_type, source = self.get(path)
      self.assertEqual(status, 200, path)
      self.assertEqual(content_type, "application/javascript", path)
      seen.add(path)
      text = source.decode()
      # Follow static ES imports and JS resources fetched to construct the graph worker.
      dependencies = re.findall(r'(?:from\s+|import\s*)["\']([^"\']+\.js)["\']', text)
      dependencies += re.findall(r'["\'](/js/[^"\']+\.js)["\']', text)
      for dependency in dependencies:
        resolved = str(PurePosixPath(path).parent / dependency) if dependency.startswith(".") else dependency
        if not resolved.startswith("/"): resolved = "/" + resolved
        pending.append(resolved)

    expected = {f"/js/{path.name}" for path in (VIZ_ROOT / "js").glob("*.js")}
    self.assertEqual(seen, expected)

  def test_empty_context_list(self):
    status, content_type, body = self.get("/ctxs")
    self.assertEqual((status, content_type), (200, "application/json"))
    self.assertEqual(json.loads(body), [])

  def test_serve_compatibility_imports(self):
    from tinygrad.viz import serve
    from tinygrad.viz.graph import VizData as SplitVizData, _reconstruct, create_step, fmt_colored, get_full_rewrite, uop_to_json
    from tinygrad.viz.http import Handler as SplitHandler, HTTPRequestHandler, TCPServerWithReuse as SplitTCPServer
    from tinygrad.viz.profile import row_tuple, soft_err, sqtt_timeline, unpack_pmc
    from tinygrad.viz.amd import amd_decode, amdgpu_cfg, get_stdout
    from tinygrad.viz.render import get_int, get_render

    self.assertIs(serve.VizData, SplitVizData)
    self.assertIs(serve.Handler, SplitHandler)
    self.assertIs(serve.HTTPRequestHandler, HTTPRequestHandler)
    self.assertIs(serve.TCPServerWithReuse, SplitTCPServer)
    self.assertIs(serve.get_int, get_int)
    self.assertIs(serve.get_render, get_render)
    for name, value in (("_reconstruct", _reconstruct), ("create_step", create_step), ("fmt_colored", fmt_colored),
                        ("get_full_rewrite", get_full_rewrite), ("uop_to_json", uop_to_json), ("row_tuple", row_tuple),
                        ("soft_err", soft_err), ("sqtt_timeline", sqtt_timeline), ("unpack_pmc", unpack_pmc),
                        ("amd_decode", amd_decode), ("amdgpu_cfg", amdgpu_cfg), ("get_stdout", get_stdout)):
      self.assertIs(getattr(serve, name), value)

  @unittest.skipUnless(shutil.which("node"), "node is unavailable")
  def test_javascript_syntax(self):
    for path in sorted((VIZ_ROOT / "js").glob("*.js")):
      with self.subTest(module=path.name):
        subprocess.run(["node", "--check", str(path)], check=True, capture_output=True, text=True)


if __name__ == "__main__": unittest.main()
