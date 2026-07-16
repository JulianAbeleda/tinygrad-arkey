import json, os, socketserver, time
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
from tinygrad.helpers import START_TIME
from tinygrad.viz.render import get_render

# NOTE: using HTTPServer forces a potentially slow socket.getfqdn
class TCPServerWithReuse(socketserver.TCPServer):
  allow_reuse_address = True
  def __init__(self, server_address, RequestHandlerClass):
    print(f"*** started server on http://127.0.0.1:{server_address[1]} at {time.perf_counter()-START_TIME:.2f} s")
    super().__init__(server_address, RequestHandlerClass)

class Handler(BaseHTTPRequestHandler):
  data = None
  profile_ret:bytes|None = None
  def send_data(self, data:bytes, content_type:str="application/json", status_code:int=200):
    self.send_response(status_code)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    return self.wfile.write(data)
  def stream_json(self, source):
    try:
      self.send_response(200)
      self.send_header("Content-Type", "text/event-stream")
      self.send_header("Cache-Control", "no-cache")
      self.end_headers()
      for r in source:
        self.wfile.write(f"data: {json.dumps(r)}\n\n".encode("utf-8"))
        self.wfile.flush()
      self.wfile.write(b"data: [DONE]\n\n")
    except (BrokenPipeError, ConnectionResetError): return

  def do_GET(self):
    ret, status_code, content_type = b"", 200, "text/html"

    if (url:=urlparse(self.path)).path == "/":
      with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f: ret = f.read()
    elif self.path.startswith(("/assets/", "/js/")) and '/..' not in self.path:
      try:
        with open(os.path.join(os.path.dirname(__file__), self.path.strip('/')), "rb") as f: ret = f.read()
        content_type = {".js":"application/javascript", ".css":"text/css"}.get(os.path.splitext(url.path)[1], content_type)
      except FileNotFoundError: status_code = 404

    elif url.path == "/ctxs":
      lst = [{"name":c["name"], "steps":[{k:v for k, v in s.items() if k != "data"} for s in c["steps"]]} for c in self.data.ctxs]
      ret, content_type = json.dumps(lst).encode(), "application/json"
    elif url.path == "/get_profile" and self.profile_ret: ret, content_type = self.profile_ret, "application/octet-stream"
    else:
      if not (render_src:=get_render(self.data, self.path)): status_code = 404
      else:
        if "content_type" in render_src: ret, content_type = render_src["value"], render_src["content_type"]
        else: ret, content_type = json.dumps(render_src).encode(), "application/json"
        if content_type == "text/event-stream": return self.stream_json(render_src["value"])

    return self.send_data(ret, content_type, status_code)

HTTPRequestHandler = Handler
