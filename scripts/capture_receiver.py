from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive local WebGPU MediaRecorder captures.")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--out-dir", type=Path, default=Path("submission/captures"))
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            name = Path(unquote(parsed.path)).name
            if not name.endswith(".webm"):
                self.send_error(400, "capture filename must end in .webm")
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100 * 1024 * 1024:
                self.send_error(413, "invalid capture size")
                return
            destination = out_dir / name
            destination.write_bytes(self.rfile.read(length))
            self.send_response(201)
            self._cors()
            self.end_headers()
            self.wfile.write(str(destination).encode("utf-8"))

        def log_message(self, format: str, *args: object) -> None:
            return

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
