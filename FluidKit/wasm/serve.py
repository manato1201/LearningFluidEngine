"""
serve.py  —  FluidKit WASM ローカル HTTP サーバー

WASM ファイルは file:// では動かないため HTTP で配信する必要があります。

使い方:
    python serve.py
    → http://localhost:8765/fluid_wasm.html をブラウザで開く
"""

import http.server, socketserver, os

PORT = 8765
DIR  = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=DIR, **kw)

    def end_headers(self):
        # WASM に必要な COOP/COEP ヘッダー（SharedArrayBuffer 対応）
        self.send_header("Cross-Origin-Opener-Policy",   "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # ログ抑制

print(f"[FluidKit] WASM サーバー起動")
print(f"  → http://localhost:{PORT}/fluid_wasm.html")
print("  Ctrl+C で停止\n")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
