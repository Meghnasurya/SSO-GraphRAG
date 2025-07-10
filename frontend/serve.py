#!/usr/bin/env python3
"""
Simple HTTP server to serve the frontend on port 3000
"""
import http.server
import socketserver
import os
import sys

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == "__main__":
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print(f"✅ Frontend server running at http://localhost:{PORT}")
            print(f"📁 Serving directory: {DIRECTORY}")
            print("🔗 Make sure your backend is running on http://localhost:8000")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        sys.exit(0)
    except OSError as e:
        if e.errno == 10048:  # Address already in use on Windows
            print(f"❌ Port {PORT} is already in use. Please stop the other process or use a different port.")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)
