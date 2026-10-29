"""A small client of the service: `python service/api_client.py -f page.png` (or a PDF).

It sends what the service takes, with the service's defaults (atrium-project#72): `all` (the
ensemble), the endpoint chosen by the file's content (a PDF to `/predict_document`, anything else
to `/predict_image`, as the service and `run.py` tell them apart), `pages` for a PDF, `per_model`,
and a document record to add to or to return.
"""

import argparse
import json
import os
import sys

import requests


def is_pdf_bytes(head: bytes) -> bool:
    """A PDF by its header, as `page_inputs.is_pdf` and the service read it."""
    return b"%PDF-" in head[:1024]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify a page image or a PDF with the page-classification API")
    parser.add_argument("-f", required=True, help="Path to the file: a page image or a PDF")
    parser.add_argument("-v", default="all", help="Model version: all (the ensemble, the default) or e.g. v4.4")
    parser.add_argument("--top", type=int, default=3, help="Top N predictions (default 3, as the service)")
    parser.add_argument("--pages", default=None, help="A PDF's pages, e.g. 1-3,7 (default: all)")
    parser.add_argument("--per-model", action="store_true", help="Also return each ensemble member's top N")
    parser.add_argument("--document-json", default=None, help="A record to add the page categories to")
    parser.add_argument("--document-json-out", action="store_true", help="Return a record even when none is sent")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the classification API (default: http://localhost:8000)",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.f):
        print(f"Error: File {args.f} not found.")
        return 1

    with open(args.f, "rb") as fh:
        content = fh.read()
    pdf = is_pdf_bytes(content)
    endpoint = "/predict_document" if pdf else "/predict_image"
    media_type = "application/pdf" if pdf else "application/octet-stream"  # the service sniffs an image
    print(f"{'A PDF' if pdf else 'An image'}: {endpoint}")

    payload = {"version": args.v, "topn": args.top}
    if args.per_model:
        payload["per_model"] = "true"
    if args.document_json_out:
        payload["document_json_out"] = "true"
    if pdf and args.pages:
        payload["pages"] = args.pages
    files = {"file": (os.path.basename(args.f), content, media_type)}
    if args.document_json:
        with open(args.document_json, "rb") as fh:
            files["document_json"] = (os.path.basename(args.document_json), fh.read(), "application/json")

    try:
        response = requests.post(f"{args.url.rstrip('/')}{endpoint}", data=payload, files=files, timeout=600)
    except requests.RequestException as exc:
        print(f"Connection error: {exc}")
        return 2
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return 0
    print(f"Error {response.status_code}: {response.text}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
