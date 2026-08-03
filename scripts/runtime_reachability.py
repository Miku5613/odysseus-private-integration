"""Fast optional-service reachability probes for portable Windows launch."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import socket
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _load_env() -> None:
    path = Path(".env")
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _http_reachable(url: str) -> str:
    try:
        request = Request(url, headers={"User-Agent": "Odysseus-Preflight/1"})
        with urlopen(request, timeout=1.5) as response:
            if 200 <= response.status < 500:
                return url
    except Exception:
        pass
    return ""


def model_endpoint() -> int:
    _load_env()
    urls = [
        "http://127.0.0.1:11434/api/tags",
        "http://127.0.0.1:11434/v1/models",
        "http://127.0.0.1:1234/v1/models",
        "http://127.0.0.1:8000/v1/models",
        "http://127.0.0.1:8080/v1/models",
    ]
    for name in ("OLLAMA_BASE_URL", "OLLAMA_URL", "LM_STUDIO_URL"):
        raw = os.getenv(name, "").strip().rstrip("/")
        if not raw:
            continue
        if "/v1" in raw:
            urls.append(raw + "/models")
        else:
            urls.extend((raw + "/api/tags", raw + "/v1/models"))
    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        futures = [pool.submit(_http_reachable, url) for url in dict.fromkeys(urls)]
        for future in as_completed(futures):
            reached = future.result()
            if reached:
                print(f"[REACHABLE] Local model endpoint: {reached}")
                return 0
    print("[UNREACHABLE] No local model endpoint was detected")
    return 1


def chroma() -> int:
    _load_env()
    host = os.getenv("CHROMADB_HOST", "localhost").strip() or "localhost"
    try:
        port = int(os.getenv("CHROMADB_PORT", "8100"))
    except ValueError:
        port = 8100
    try:
        with socket.create_connection((host, port), timeout=1.5):
            pass
        print(f"[REACHABLE] ChromaDB: {host}:{port}")
        return 0
    except OSError:
        print(f"[UNREACHABLE] ChromaDB: {host}:{port}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("model", "chroma"))
    service = parser.parse_args().service
    return model_endpoint() if service == "model" else chroma()


if __name__ == "__main__":
    raise SystemExit(main())
