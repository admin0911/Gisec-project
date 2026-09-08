"""Serve the standalone feature-layer frontend locally."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


if __name__ == "__main__":
    frontend = Path(__file__).parent / "frontend"
    handler = partial(SimpleHTTPRequestHandler, directory=str(frontend))
    server = ThreadingHTTPServer(("127.0.0.1", 8787), handler)
    print("Gisec frontend: http://127.0.0.1:8787")
    server.serve_forever()

