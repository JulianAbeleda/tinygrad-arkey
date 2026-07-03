#!/usr/bin/env python3
import argparse, hashlib, os, tarfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def sha256_file(path:Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
  return h.hexdigest()


class UploadHandler(BaseHTTPRequestHandler):
  out_dir: Path
  extract_root: Path
  extract: bool

  def do_PUT(self):
    name = Path(self.path).name or "upload.tar.gz"
    dest = self.out_dir / name
    length = int(self.headers.get("Content-Length", "0"))
    remaining = length
    with dest.open("wb") as f:
      while remaining:
        chunk = self.rfile.read(min(1024 * 1024, remaining))
        if not chunk: break
        f.write(chunk)
        remaining -= len(chunk)

    digest = sha256_file(dest)
    print(f"received={dest}", flush=True)
    print(f"sha256={digest}", flush=True)
    if self.extract and tarfile.is_tarfile(dest):
      with tarfile.open(dest) as tf: tf.extractall(self.extract_root)
      print(f"extracted_to={self.extract_root}", flush=True)

    self.send_response(200)
    self.end_headers()
    self.wfile.write(f"ok sha256={digest}\n".encode())

  def log_message(self, fmt, *args):
    print(fmt % args, flush=True)


def main():
  parser = argparse.ArgumentParser(description="Receive Ubuntu PSP capture tarballs over HTTP PUT.")
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("--out", type=Path, default=Path("extra/hardware/amdpci/captures"))
  parser.add_argument("--extract-root", type=Path, default=Path("."))
  parser.add_argument("--no-extract", action="store_true")
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[2]
  os.chdir(root)
  UploadHandler.out_dir = args.out
  UploadHandler.out_dir.mkdir(parents=True, exist_ok=True)
  UploadHandler.extract_root = args.extract_root
  UploadHandler.extract = not args.no_extract

  print(f"listening=http://{args.host}:{args.port}/", flush=True)
  print(f"out={UploadHandler.out_dir.resolve()}", flush=True)
  print(f"extract_root={UploadHandler.extract_root.resolve() if UploadHandler.extract else 'disabled'}", flush=True)
  HTTPServer((args.host, args.port), UploadHandler).serve_forever()


if __name__ == "__main__":
  main()
