"""Local inference service for the single-file website; never executes notebooks."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, unquote
from io import BytesIO
import argparse
import base64
import json
import math
import os
import threading
import time

from PIL import Image, ImageOps, UnidentifiedImageError
# Keep Pillow's decoder: Ultralytics patches Image.open to auto-install HEIF
# support for invalid input, which this JPG/PNG/WebP-only service does not need.
OPEN_IMAGE = Image.open
os.environ["YOLO_AUTOINSTALL"] = "false"
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
MODEL = None
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data, kind="application/json"):
        if isinstance(data, dict):
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/health":
            return self.respond(200, {"ready": True, "model": "YOLOv8n", "classes": MODEL.names})
        if path in ("/", "/index.html"):
            return self.respond(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/Bone_Fracture_YOLO.pdf":
            return self.respond(200, (ROOT / "Bone_Fracture_YOLO.pdf").read_bytes(), "application/pdf")
        if path.startswith("/assets/"):
            file = (ROOT / path.lstrip("/")).resolve()
            if file.parent == (ROOT / "assets").resolve() and file.is_file() and file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                return self.respond(200, file.read_bytes(), "image/png" if file.suffix.lower() == ".png" else "image/jpeg")
        self.respond(404, {"error": "Not found"})

    def do_POST(self):
        route = urlsplit(self.path)
        if route.path != "/api/predict":
            return self.respond(404, {"error": "Not found"})
        # Only this local page can submit inference requests from a browser.
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            return self.respond(403, {"error": "Use the page served by this local server."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_UPLOAD:
                return self.respond(413, {"error": "Choose an image smaller than 10 MB."})
            conf = float(parse_qs(route.query).get("conf", ["0.25"])[0])
            if not math.isfinite(conf) or not 0.01 <= conf <= 0.99:
                return self.respond(400, {"error": "Confidence must be between 0.01 and 0.99."})
            self.connection.settimeout(30)
            raw = self.rfile.read(length)
            with OPEN_IMAGE(BytesIO(raw)) as source:
                if source.format not in ("JPEG", "PNG", "WEBP"):
                    raise ValueError("Choose a JPG, PNG, or WebP image.")
                if source.width * source.height > Image.MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the 20 megapixel limit.")
                image = ImageOps.exif_transpose(source).convert("RGB")
            start = time.perf_counter()
            with LOCK:
                result = MODEL.predict(image, conf=conf, imgsz=640, device="cpu", verbose=False, save=False)[0]
            elapsed = (time.perf_counter() - start) * 1000
            detections = [{"class_id": int(box.cls.item()), "label": MODEL.names[int(box.cls.item())],
                           "confidence": float(box.conf.item()), "xyxy": box.xyxy[0].tolist()}
                          for box in result.boxes]
            # Return normalized pixels so EXIF rotation and box coordinates always agree.
            output = BytesIO()
            image.save(output, format="PNG")
            self.respond(200, {"detections": detections, "width": image.width, "height": image.height,
                               "elapsed_ms": round(elapsed, 1), "threshold": conf,
                               "image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()})
        except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            self.respond(400, {"error": "Invalid image or settings. Use a JPG, PNG, or WebP under 10 MB and 20 megapixels."})
        except Exception:
            import traceback
            traceback.print_exc()
            self.respond(500, {"error": "Prediction failed. See the local server terminal for details."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print("Loading boneFractureDetection-best.pt on CPU...", flush=True)
    MODEL = YOLO(ROOT / "boneFractureDetection-best.pt")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://127.0.0.1:{args.port} | Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
