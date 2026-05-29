#!/bin/bash
# pdf2png.sh — Convert PDF files to page images in parallel.
#
# Converts every PDF in a directory to per-page images using pdftoppm.
# Conversion runs in parallel across all available CPU cores.
#
# Usage:
#   ./pdf2png.sh [OPTIONS]
#
# Options:
#   -f, --format FORMAT   Output image format: png or jpg  (default: png)
#   -r, --dpi N           Output resolution in DPI         (default: 300)
#   -d, --dir DIR         Directory containing PDF files   (default: current directory)
#   -o, --output DIR      Root output directory; each PDF gets its own subdirectory
#                         inside it                        (default: same as --dir)
#   -k, --keep            Keep original PDF files after conversion
#                         (default: delete PDFs on success)
#   -h, --help            Show this help message and exit
#
# Output structure (mirrors the original behaviour):
#   <output>/<pdf-name>/<pdf-name>-001.png   (Unix: zero-padded by pdftoppm)
#
# Examples:
#   ./pdf2png.sh
#   ./pdf2png.sh --format jpg --dpi 200
#   ./pdf2png.sh --dir /data/pdfs --output /data/pages --keep
#   ./pdf2png.sh --format jpg --dir /data/pdfs --dpi 150

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
FORMAT="png"
DPI=300
SOURCE_DIR="."
OUTPUT_DIR=""
KEEP=false

# ── Usage ──────────────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,1\}//'
    exit 0
}

# ── Argument parsing ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--format)  FORMAT="$2";     shift 2 ;;
        -r|--dpi)     DPI="$2";        shift 2 ;;
        -d|--dir)     SOURCE_DIR="$2"; shift 2 ;;
        -o|--output)  OUTPUT_DIR="$2"; shift 2 ;;
        -k|--keep)    KEEP=true;       shift   ;;
        -h|--help)    usage ;;
        *) echo "Error: unknown option '$1'"; echo "Run with --help for usage."; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────
FORMAT="${FORMAT,,}"   # normalise to lower-case
if [[ "$FORMAT" != "png" && "$FORMAT" != "jpg" && "$FORMAT" != "jpeg" ]]; then
    echo "Error: --format must be 'png' or 'jpg' (got '$FORMAT')."
    exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# Map jpg/jpeg to the pdftoppm flag and canonical extension
if [[ "$FORMAT" == "jpg" || "$FORMAT" == "jpeg" ]]; then
    FORMAT_FLAG="-jpeg"
    EXT="jpg"
else
    FORMAT_FLAG="-png"
    EXT="png"
fi

# Output root defaults to the source directory
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$SOURCE_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Source dir : $SOURCE_DIR"
echo "Output dir : $OUTPUT_DIR"
echo "Format     : $EXT  (DPI: $DPI)"
[[ "$KEEP" == true ]] && echo "Keep PDFs  : yes" || echo "Keep PDFs  : no (delete on success)"
echo ""

# ── Check that pdftoppm is available ──────────────────────────────────────
if ! command -v pdftoppm &>/dev/null; then
    echo "Error: pdftoppm not found. Install poppler-utils:"
    echo "  Ubuntu/Debian: sudo apt-get install poppler-utils"
    echo "  macOS:         brew install poppler"
    exit 1
fi

# ── Convert in parallel (one job per CPU core) ────────────────────────────
# xargs -P $(nproc) spawns up to nproc simultaneous conversions.
# -print0 / -0 handles filenames with spaces safely.
export FORMAT_FLAG DPI OUTPUT_DIR KEEP EXT

find "$SOURCE_DIR" -maxdepth 1 -name "*.pdf" -print0 \
| xargs -0 -I {} -P "$(nproc)" bash -c '
    pdf_file="{}"
    filename=$(basename "$pdf_file" .pdf)
    out_subdir="${OUTPUT_DIR}/${filename}"
    mkdir -p "$out_subdir"

    if pdftoppm "${FORMAT_FLAG}" -r "${DPI}" "$pdf_file" "${out_subdir}/${filename}"; then
        echo "Converted: $pdf_file  →  ${out_subdir}/${filename}-*.${EXT}"
        if [[ "${KEEP}" == false ]]; then
            rm "$pdf_file"
        fi
    else
        echo "Failed:    $pdf_file"
    fi
'

echo ""
echo "All PDFs processed."
