#!/bin/bash
# pdf2png.sh — Convert PDF files to page images in parallel.
#
# Converts every PDF in a directory (*.pdf, any case: .PDF works too) to per-page
# images using pdftoppm. Conversion runs in parallel across all available CPU cores.
# The PDFs are kept unless --delete is given.
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
#       --delete          Delete each PDF once it has been converted successfully
#                         (default: keep the PDFs)
#   -k, --keep            Keep the PDFs — the default; still accepted so older
#                         command lines keep working
#   -h, --help            Show this help message and exit
#
# Output structure (mirrors the original behaviour):
#   <output>/<pdf-name>/<pdf-name>-001.png   (Unix: zero-padded by pdftoppm)
#
# Examples:
#   ./pdf2png.sh
#   ./pdf2png.sh --format jpg --dpi 200
#   ./pdf2png.sh --dir /data/pdfs --output /data/pages
#   ./pdf2png.sh --dir /data/pdfs --output /data/pages --delete
#   ./pdf2png.sh --format jpg --dir /data/pdfs --dpi 150

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
FORMAT="png"
DPI=300
SOURCE_DIR="."
OUTPUT_DIR=""
KEEP=true

# ── Usage ──────────────────────────────────────────────────────────────────
# Prints the header comment block only (it stops at the first non-comment line).
usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
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
        --delete)     KEEP=false;      shift   ;;
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
if [[ "$KEEP" == true ]]; then echo "Keep PDFs  : yes"; else echo "Keep PDFs  : no (--delete: delete on success)"; fi
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
# -print0 / -0 handles filenames with spaces and newlines safely, and each file
# reaches the inner script as its argument "$1" — never spliced into the script
# text, where a quote or a "$" in a filename would break it or be executed.
export FORMAT_FLAG DPI OUTPUT_DIR KEEP EXT

if [[ -z "$(find "$SOURCE_DIR" -maxdepth 1 -type f -iname '*.pdf' -print -quit)" ]]; then
    echo "No PDF files found in $SOURCE_DIR."
    exit 0
fi

# shellcheck disable=SC2016
find "$SOURCE_DIR" -maxdepth 1 -type f -iname '*.pdf' -print0 \
| xargs -0 -n 1 -P "$(nproc)" bash -c '
    pdf_file="$1"
    [[ -n "$pdf_file" ]] || exit 0
    base=$(basename "$pdf_file")
    filename="${base%.*}"          # strips .pdf and .PDF alike
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
' _

echo ""
echo "All PDFs processed."
