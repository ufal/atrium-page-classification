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

    # ATRIUM Document JSON accretion (docs/document_schema.md, issue #13):
    # accrete this tool's page_categories block onto an existing baseline record
    python3 scripts/atrium_classify.py page.png --document-json in.document.json \
        --document-json-out-file out.document.json
    # or originate a fresh record (page-classification is stage 1 of the pipeline)
    python3 scripts/atrium_classify.py page.png --document-json-out \
        --document-json-out-file out.document.json

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
from typing import Optional

DEFAULT_BASE_URL = os.environ.get("ATRIUM_PC_URL", "http://localhost:8000")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
PDF_SUFFIXES = {".pdf"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # mirrors MAX_UPLOAD_BYTES in service/api.py
RETRY_STATUS = {502, 503, 504}
RETRY_ATTEMPTS = 3
RETRY_WAIT_S = 10


def build_multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """Encode form fields and one or more files as multipart/form-data using only the stdlib.

    `files` maps the multipart field name to a `Path` to upload under that name (e.g.
    `{"file": page.png, "document_json": baseline.json}` for the accretion contract).
    """
    boundary = uuid.uuid4().hex
    lines = []
    for name, value in fields.items():
        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        lines.append(b"")
        lines.append(str(value).encode())

    for field_name, file_path in files.items():
        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        lines.append(f"--{boundary}".encode())
        lines.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"'.encode()
        )
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


def classify_file(
    base_url: str,
    path: Path,
    version: str,
    topn: int,
    document_json: Optional[Path] = None,
    document_json_out: bool = False,
) -> dict:
    """Route a file to /predict_image or /predict_document based on its suffix.

    `document_json`/`document_json_out` opt into the ATRIUM Document JSON accretion
    contract (docs/document_schema.md): upload an existing baseline to accrete onto, and/or
    ask the service to emit a record even with no baseline (page-classification can
    originate one, being stage 1 of the pipeline).
    """
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

    fields = {"version": version, "topn": topn}
    if document_json_out:
        fields["document_json_out"] = "true"
    files = {"file": path}
    if document_json is not None:
        files["document_json"] = document_json

    body, content_type = build_multipart(fields, files)
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
    parser.add_argument(
        "--document-json",
        metavar="PATH",
        help="baseline ATRIUM Document JSON to accrete this tool's page_categories block onto "
        "(docs/document_schema.md); requires exactly one input file",
    )
    parser.add_argument(
        "--document-json-out",
        action="store_true",
        help="ask the service to originate/return a document record even with no --document-json baseline",
    )
    parser.add_argument(
        "--document-json-out-file",
        metavar="PATH",
        help="save the returned document_json record to PATH (default: only embedded in --format json output)",
    )
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

    document_json_path = None
    if args.document_json:
        if len(paths) != 1:
            parser.error("--document-json accretes onto a single document; pass exactly one input file")
        document_json_path = Path(args.document_json)
        if not document_json_path.is_file():
            print(f"--document-json file not found: {document_json_path}", file=sys.stderr)
            sys.exit(1)
    if args.document_json_out_file and len(paths) != 1:
        parser.error("--document-json-out-file writes one record; pass exactly one input file")

    raw_results = {}
    rows = []
    document_record = None
    for path in paths:
        result = classify_file(
            base_url,
            path,
            version=args.version,
            topn=args.topn,
            document_json=document_json_path,
            document_json_out=args.document_json_out,
        )
        if result:
            raw_results[path.name] = result
            rows.extend(result_rows(path, result))
            if result.get("document_json") is not None:
                document_record = result["document_json"]
            if result.get("document_json_schema_error"):
                print(
                    f"Warning: uploaded document_json baseline for {path.name} did not validate: "
                    f"{result['document_json_schema_error']} (record still returned, per rule 6)",
                    file=sys.stderr,
                )

    if not rows:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    if args.document_json_out_file and document_record is not None:
        Path(args.document_json_out_file).write_text(json.dumps(document_record, indent=2), encoding="utf-8")
        print(f"Document JSON record written to {args.document_json_out_file}", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(raw_results, indent=2))
    else:
        print_table(rows, as_csv=(args.format == "csv"))


if __name__ == "__main__":
    main()
