import requests
import argparse
import mimetypes
import os
import sys

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Test the Atrium Page Classification API")
    parser.add_argument("--f", required=True, help="Path to the image file")
    parser.add_argument("--v", default="v4.3", help="Model version (e.g., v1.3, v5.3, all)")
    parser.add_argument("--top", type=int, default=3, help="Number of top predictions to return")

    args = parser.parse_args()

    url = "http://localhost:8000/predict"

    # 2. Validate File Existence
    if not os.path.exists(args.f):
        print(f"Error: File not found at {args.f}")
        sys.exit(1)

    # 3. Determine MIME Type (Critical for Server Validation)
    # The server checks for ["image/jpeg", "image/png", "image/jpg"]
    content_type, _ = mimetypes.guess_type(args.f)
    
    if content_type is None:
        print("Warning: Could not determine file type. Defaulting to 'application/octet-stream'.")
        content_type = "application/octet-stream"

    # 4. Send Request
    payload = {
        "version": args.v,
        "topn": args.top
    }

    try:
        with open(args.f, "rb") as img:
            # FIX: We pass the tuple (filename, file_object, content_type)
            # This ensures requests sends the correct header so FastAPI doesn't receive 'None'
            files = {"file": (os.path.basename(args.f), img, content_type)}
            
            print(f"Sending {args.f} as {content_type}...")
            response = requests.post(url, data=payload, files=files)

        # 5. Print Results
        if response.status_code == 200:
            print("\n--- Success! ---")
            import json
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"\n--- Error {response.status_code} ---")
            print(response.text)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
