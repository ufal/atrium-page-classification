#!/usr/bin/env python3
"""Zero-dependency client for the ATRIUM Page Classification API.

Classifies historical document page images (PNG/JPEG) or multipage PDFs by
uploading them to a running instance of the FastAPI service in `service/api.py`
(local by default, remote via --base-url or the ATRIUM_PC_URL env variable).

Only the Python 3 standard library is used - no pip installs required.

Usage:
    python3 scripts/atrium_classify.py page.png
    python3 scripts/atrium_classify.py document.pdf --topn 5 --format csv
    python3 scripts/atrium_classify.py *.png --version v4.3 --format json
    python3 scripts/atrium_classify.py --info

Exit codes:
    0 - success
    1 - client-side error (bad arguments, unreadable file)
    2 - server unreachable (connection refused / timeout)
    3 - server-side error (HTTP 4xx/5xx)
"""

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_BASE_URL = os.environ.get("ATRIUM_PC_URL", "http://localhost:8000")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
PDF_SUFFIXES = {".pdf"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # mirrors MAX_UPLOAD_BYTES in service/api.py
RETRY_STATUS = {502, 503, 504}
RETRY_ATTEMPTS = 3
RETRY_WAIT_S = 10


def build_multipart(fields: dict, file_field: str, file_path: Path) -> tuple[bytes, str]:
    """Encode form fields and one file as multipart/form-data using only the stdlib."""
    boundary = uuid.uuid4().hex
    lines = []
    for name, value in fields.items():
        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        lines.append(b"")
        lines.append(str(value).encode())

    mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    lines.append(f"--{boundary}".encode())
    lines.append(f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"'.encode())
    lines.append(f"Content-Type: {mime}".encode())
    lines.append(b"")
    lines.append(file_path.read_bytes())
    lines.append(f"--{boundary}--".encode())
    lines.append(b"")

    body = b"\r\n".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def http_json(url: str, data: bytes = None, content_type: str = None, timeout: int = 300) -> dict:
    """POST (or GET when data is None) and decode a JSON response, with retry on 502/503/504."""
    last_error = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code in RETRY_STATUS and attempt < RETRY_ATTEMPTS:
                print(
                    f"[retry {attempt}/{RETRY_ATTEMPTS}] HTTP {e.code}, waiting {RETRY_WAIT_S}s...",
                    file=sys.stderr,
                )
                time.sleep(RETRY_WAIT_S)
                last_error = (3, f"HTTP {e.code}: {detail}")
                continue
            print(f"Server error - HTTP {e.code}: {detail}", file=sys.stderr)
            sys.exit(3)
        except (urllib.error.URLError, TimeoutError) as e:
            print(
                f"Cannot reach the API at {url} ({e}).\nIs the server running? Start it with: bash scripts/server.sh",
                file=sys.stderr,
            )
            sys.exit(2)
    print(f"Server error after {RETRY_ATTEMPTS} attempts - {last_error[1]}", file=sys.stderr)
    sys.exit(last_error[0])


def classify_file(base_url: str, path: Path, version: str, topn: int) -> dict:
    """Route a file to /predict_image or /predict_document based on its suffix."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        endpoint = "/predict_image"
    elif suffix in PDF_SUFFIXES:
        endpoint = "/predict_document"
    else:
        print(f"Skipping {path}: unsupported file type '{suffix}'", file=sys.stderr)
        return {}

    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        print(
            f"Skipping {path}: {size} bytes exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB "
            "server upload limit - downscale the image or split the PDF first",
            file=sys.stderr,
        )
        return {}

    body, content_type = build_multipart({"version": version, "topn": topn}, file_field="file", file_path=path)
    return http_json(f"{base_url}{endpoint}", data=body, content_type=content_type)


def result_rows(path: Path, result: dict) -> list[tuple]:
    """Flatten an API response into (file, page, rank, label, score) rows."""
    rows = []
    if result.get("type") == "image":
        for rank, prediction in enumerate(result.get("predictions", []), start=1):
            rows.append((path.name, 1, rank, prediction["label"], prediction["score"]))
    elif result.get("type") == "document":
        for page in result.get("pages", []):
            for rank, prediction in enumerate(page.get("predictions", []), start=1):
                rows.append((path.name, page["page"], rank, prediction["label"], prediction["score"]))
    return rows


def print_table(rows: list[tuple], as_csv: bool) -> None:
    header = ("FILE", "PAGE", "RANK", "LABEL", "SCORE")
    if as_csv:
        print(",".join(header))
        for row in rows:
            print(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]:.4f}")
    else:
        print(f"{header[0]:<40} {header[1]:>4} {header[2]:>4} {header[3]:<10} {header[4]:>7}")
        for row in rows:
            print(f"{row[0]:<40} {row[1]:>4} {row[2]:>4} {row[3]:<10} {row[4]:>7.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="PNG/JPEG image(s) and/or PDF document(s) to classify")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL}, env: ATRIUM_PC_URL)"
    )
    parser.add_argument(
        "--version", default="all", help="model version, e.g. v4.3, or 'all' for the best-5 ensemble (default)"
    )
    parser.add_argument("--topn", type=int, default=3, help="number of top predictions per page (default: 3)")
    parser.add_argument(
        "--format", choices=["table", "csv", "json"], default="table", help="output format (default: table)"
    )
    parser.add_argument("--info", action="store_true", help="print available models and categories, then exit")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    if args.info:
        print(json.dumps(http_json(f"{base_url}/info"), indent=2))
        return

    if not args.files:
        parser.error("no input files given (or use --info)")

    paths = [Path(f) for f in args.files]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"File(s) not found: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        sys.exit(1)

    raw_results = {}
    rows = []
    for path in paths:
        result = classify_file(base_url, path, version=args.version, topn=args.topn)
        if result:
            raw_results[path.name] = result
            rows.extend(result_rows(path, result))

    if not rows:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(raw_results, indent=2))
    else:
        print_table(rows, as_csv=(args.format == "csv"))


if __name__ == "__main__":
    main()
