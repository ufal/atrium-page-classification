#!/bin/bash
# sort.sh — Copy (or move) annotated PNG pages into label-specific subdirectories.
#
# Reads a CSV annotation file with FILE, PAGE, CLASS columns and copies the
# matching PNG from the document-specific subdirectory structure into a
# label-sorted output directory suitable for model training.
#
# Usage:
#   ./sort.sh -i INPUT_DIR -o OUTPUT_DIR -c CSV_FILE [OPTIONS]
#
# Required:
#   -i, --input-dir DIR    Directory containing document-specific PNG subdirectories
#   -o, --output-dir DIR   Target directory for label-sorted training pages
#   -c, --csv FILE         CSV annotation file with columns: FILE, PAGE, CLASS
#
# Options:
#   --move                 Move files instead of copying (default: copy)
#   -n, --dry-run          Show what would happen without making any changes
#   -h, --help             Show this help message and exit
#
# The script tries all common zero-padding widths (none, 2-, 3-, 4-digit) so it
# works with both Unix pdftoppm output (auto-padded) and Windows ImageMagick
# output (unpadded).  Documents with no subdirectory fall back to an "onepagers"
# subdirectory inside INPUT_DIR.
#
# Examples:
#   ./sort.sh -i /data/pages -o /data/train -c /data/annotations.csv
#   ./sort.sh -i /data/pages -o /data/train -c /data/annotations.csv --move
#   ./sort.sh -i /data/pages -o /data/train -c /data/annotations.csv --dry-run

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────
INPUT_DIR=""
OUTPUT_DIR=""
INPUT_CSV=""
USE_MOVE=false
DRY_RUN=false

# ── Usage ─────────────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,1\}//'
    exit 0
}

# ── Argument parsing ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input-dir)  INPUT_DIR="$2";  shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -c|--csv)        INPUT_CSV="$2";  shift 2 ;;
        --move)          USE_MOVE=true;   shift   ;;
        -n|--dry-run)    DRY_RUN=true;    shift   ;;
        -h|--help)       usage ;;
        *) echo "Error: unknown option '$1'"; echo "Run with --help for usage."; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────
missing=false
[[ -z "$INPUT_DIR" ]]  && echo "Error: -i/--input-dir is required."  && missing=true
[[ -z "$OUTPUT_DIR" ]] && echo "Error: -o/--output-dir is required."  && missing=true
[[ -z "$INPUT_CSV" ]]  && echo "Error: -c/--csv is required."         && missing=true
[[ "$missing" == true ]] && echo "Run with --help for usage." && exit 1

[[ ! -d "$INPUT_DIR" ]] && echo "Error: INPUT_DIR '$INPUT_DIR' does not exist."  && exit 1
[[ ! -f "$INPUT_CSV" ]] && echo "Error: CSV_FILE '$INPUT_CSV' does not exist."    && exit 1

[[ "$DRY_RUN" == false ]] && mkdir -p "$OUTPUT_DIR"

echo "Input dir : $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "CSV file  : $INPUT_CSV"
[[ "$USE_MOVE" == true ]] && echo "Mode      : move" || echo "Mode      : copy"
[[ "$DRY_RUN"  == true ]] && echo "Dry-run   : yes"
echo ""

# ── Counters ──────────────────────────────────────────────────────────────
copied=0
not_found=0

# ── Helper: find a PNG for the given FILE/PAGE with any padding width ─────
find_png() {
    local base_dir="$1"
    local filename="$2"
    local page_number="$3"
    local subdir="${base_dir}/${filename}"

    if [[ -d "$subdir" ]]; then
        # Try no padding, then 2-, 3-, 4-digit zero-padding
        for fmt in "" "%02d" "%03d" "%04d"; do
            local pn="$page_number"
            [[ -n "$fmt" ]] && pn=$(printf "$fmt" "$page_number")
            local candidate="${subdir}/${filename}-${pn}.png"
            if [[ -f "$candidate" ]]; then
                echo "$candidate"
                return 0
            fi
        done
    else
        # Fall back to the flat "onepagers" directory
        local candidate="${base_dir}/onepagers/${filename}-${page_number}.png"
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    fi
    return 1
}

# ── Main loop ─────────────────────────────────────────────────────────────
while IFS=',' read -r filename page_number category; do
    # Skip the header row
    [[ "$filename" == "FILE" ]] && continue
    # Trim carriage-returns from Windows-formatted CSVs
    filename="${filename//$'\r'/}"
    page_number="${page_number//$'\r'/}"
    category="${category//$'\r'/}"

    category_dir="${OUTPUT_DIR}/${category}"
    [[ "$DRY_RUN" == false ]] && mkdir -p "$category_dir"

    png_path=$(find_png "$INPUT_DIR" "$filename" "$page_number") || {
        echo "Not found: $filename  page $page_number"
        not_found=$((not_found + 1))
        continue
    }

    if [[ "$DRY_RUN" == true ]]; then
        [[ "$USE_MOVE" == true ]] && echo "[dry-run] move: $png_path → $category_dir/" \
                                  || echo "[dry-run] copy: $png_path → $category_dir/"
    else
        if [[ "$USE_MOVE" == true ]]; then
            mv "$png_path" "$category_dir/"
        else
            cp "$png_path" "$category_dir/"
        fi
    fi
    copied=$((copied + 1))

done < <(tail -n +2 "$INPUT_CSV")

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
prefix=""; [[ "$DRY_RUN" == true ]] && prefix="[dry-run] "
if [[ "$USE_MOVE" == true ]]; then
    echo "${prefix}Done. Moved $copied file(s) | $not_found page(s) not found."
else
    echo "${prefix}Done. Copied $copied file(s) | $not_found page(s) not found."
fi
