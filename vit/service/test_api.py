import requests
import argparse
import mimetypes
import os
import sys
import json


def main():
    parser = argparse.ArgumentParser(description="Test the Atrium Page Classification API")
    parser.add_argument("-f", required=True, help="Path to file (image or PDF)")
    parser.add_argument("-v", default="v4.3", help="Model version (e.g., v5.3, all)")
    parser.add_argument("--top", type=int, default=3, help="Top N predictions")
    # FIX: parameterise server URL so the script works against remote or
    # non-default-port deployments without source edits.
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the classification API (default: http://localhost:8000)")

    args = parser.parse_args()

    if not os.path.exists(args.f):
        print(f"Error: File {args.f} not found.")
        sys.exit(1)

    # Detect Type
    mime_type, _ = mimetypes.guess_type(args.f)
    if not mime_type:
        mime_type = "application/octet-stream"

    base_url = args.url.rstrip("/")

    if mime_type == "application/pdf":
        endpoint = "/predict_document"
        print(f"Detected PDF. Using {endpoint}")
    elif mime_type in ["image/jpeg", "image/png", "image/jpg"]:
        endpoint = "/predict_image"
        print(f"Detected Image. Using {endpoint}")
    else:
        print(f"Warning: Unknown mime type {mime_type}. Defaulting to image endpoint.")
        endpoint = "/predict_image"

    url = f"{base_url}{endpoint}"

    payload = {"version": args.v, "topn": args.top}

    try:
        with open(args.f, "rb") as f:
            files = {"file": (os.path.basename(args.f), f, mime_type)}
            response = requests.post(url, data=payload, files=files)

        if response.status_code == 200:
            print("\n--- Success ---")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"\n--- Error {response.status_code} ---")
            print(response.text)

    except Exception as e:
        print(f"Connection error: {e}")


if __name__ == "__main__":
    main()