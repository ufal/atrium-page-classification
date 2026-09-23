#!/bin/bash
# sort.sh — Copy (or move) annotated PNG pages into label-specific subdirectories.
#
# Reads a CSV annotation file and copies the matching PNG from the
# document-specific subdirectory structure into a label-sorted output directory
# suitable for model training.
#
# Usage:
#   ./sort.sh -i INPUT_DIR -o OUTPUT_DIR -c CSV_FILE [OPTIONS]
#
# Required:
#   -i, --input-dir DIR    Directory containing document-specific PNG subdirectories
#   -o, --output-dir DIR   Target directory for label-sorted training pages
#   -c, --csv FILE         CSV annotation file with a header row (see below)
#
# Options:
#   --move                 Move files instead of copying (default: copy)
#   -n, --dry-run          Show what would happen without making any changes
#   -h, --help             Show this help message and exit
#
# CSV format:
#   Columns are found by their header NAME (case-insensitive), not by position,
#   so the CSV may carry extra columns (a title, a DOI, ...) in any order.
#   Required: FILE, PAGE and a label column. The label column is CLASS; if the
#   CSV has none, CLASS-1 and then CATEGORY are used instead, so a classifier
#   result table can be sorted as-is. Fields may be double-quoted; a field that
#   spans several lines is not supported.
#
# Rows that are skipped and reported, never acted on:
#   * a PAGE that is not a whole number (leading zeros are fine: 08 is page 8);
#   * a FILE or label that is empty, "." or "..", or contains "/" or "\"
#     (it would otherwise become a path outside, or nested inside, the label folder).
#   A label folder is only created once a page has been found for it.
#
# The script tries all common zero-padding widths (none, 2-, 3-, 4-digit) so it
# works with both Unix pdftoppm output (auto-padded) and Windows ImageMagick
# output (unpadded).  Documents with no subdirectory fall back to an "onepagers"
# subdirectory inside INPUT_DIR, with the same padding widths.
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
# Prints the header comment block only (it stops at the first non-comment line).
usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
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

# ── CSV helpers ───────────────────────────────────────────────────────────
# split_csv_line LINE — split one CSV line into the global array FIELDS.
# Lines without a double quote take the fast path; a line with one goes through
# a small quote-aware splitter ("a,b" is one field, "" inside quotes is a quote).
FIELDS=()
split_csv_line() {
    local line="$1"
    FIELDS=()
    if [[ "$line" != *'"'* ]]; then
        IFS=',' read -r -a FIELDS <<< "$line"
        return 0
    fi
    local field="" in_quotes=false ch i
    for (( i = 0; i < ${#line}; i++ )); do
        ch="${line:i:1}"
        if [[ "$in_quotes" == true ]]; then
            if [[ "$ch" == '"' ]]; then
                if [[ "${line:i+1:1}" == '"' ]]; then
                    field+='"'
                    i=$((i + 1))
                else
                    in_quotes=false
                fi
            else
                field+="$ch"
            fi
        else
            case "$ch" in
                '"') in_quotes=true ;;
                ',') FIELDS+=("$field"); field="" ;;
                *)   field+="$ch" ;;
            esac
        fi
    done
    FIELDS+=("$field")
}

# field_at INDEX — set the global FIELD to FIELDS[INDEX], whitespace-trimmed
# (empty when the row is shorter than the header).
FIELD=""
field_at() {
    FIELD="${FIELDS[$1]:-}"
    FIELD="${FIELD#"${FIELD%%[![:space:]]*}"}"
    FIELD="${FIELD%"${FIELD##*[![:space:]]}"}"
}

# is_safe_name VALUE — true when VALUE can be used as ONE path component.
is_safe_name() {
    [[ -n "$1" && "$1" != "." && "$1" != ".." && "$1" != */* && "$1" != *\\* ]]
}

# ── Header: find the columns by name ──────────────────────────────────────
exec 3< "$INPUT_CSV"
header=""
IFS= read -r header <&3 || true
bom=$'\xef\xbb\xbf'
header="${header#"$bom"}"
header="${header%$'\r'}"
if [[ -z "$header" ]]; then
    echo "Error: CSV_FILE '$INPUT_CSV' is empty (a header row is required)."
    exit 1
fi

split_csv_line "$header"
file_col=-1; page_col=-1; class_col=-1; class1_col=-1; category_col=-1
for (( i = 0; i < ${#FIELDS[@]}; i++ )); do
    field_at "$i"
    name=$(printf '%s' "$FIELD" | tr '[:lower:]' '[:upper:]')
    case "$name" in
        FILE)     [[ $file_col     -lt 0 ]] && file_col=$i ;;
        PAGE)     [[ $page_col     -lt 0 ]] && page_col=$i ;;
        CLASS)    [[ $class_col    -lt 0 ]] && class_col=$i ;;
        CLASS-1)  [[ $class1_col   -lt 0 ]] && class1_col=$i ;;
        CATEGORY) [[ $category_col -lt 0 ]] && category_col=$i ;;
    esac
done

label_col=$class_col
[[ $label_col -lt 0 ]] && label_col=$class1_col
[[ $label_col -lt 0 ]] && label_col=$category_col

if [[ $file_col -lt 0 || $page_col -lt 0 || $label_col -lt 0 ]]; then
    echo "Error: CSV_FILE '$INPUT_CSV' must have FILE, PAGE and CLASS (or CLASS-1 / CATEGORY) columns."
    echo "       Header found: $header"
    exit 1
fi
field_at "$label_col"
label_name="$FIELD"

[[ "$DRY_RUN" == false ]] && mkdir -p "$OUTPUT_DIR"

echo "Input dir : $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "CSV file  : $INPUT_CSV"
echo "Label col : $label_name"
if [[ "$USE_MOVE" == true ]]; then echo "Mode      : move"; else echo "Mode      : copy"; fi
if [[ "$DRY_RUN" == true ]]; then echo "Dry-run   : yes"; fi
echo ""

# ── Counters ──────────────────────────────────────────────────────────────
copied=0
not_found=0
invalid=0

# ── Helper: find a PNG for the given FILE/PAGE with any padding width ─────
# Sets the global FOUND_PNG and returns 0 on success.  PAGE must already be a
# plain decimal number (no leading zeros).
FOUND_PNG=""
find_png() {
    local base_dir="$1"
    local filename="$2"
    local page_number="$3"
    local search_dir="${base_dir}/${filename}"
    local width pn candidate

    # Documents without their own subdirectory live in the flat "onepagers" one.
    [[ -d "$search_dir" ]] || search_dir="${base_dir}/onepagers"

    # Width 1 is no padding, then 2-, 3- and 4-digit zero-padding.
    for width in 1 2 3 4; do
        printf -v pn '%0*d' "$width" "$page_number"
        candidate="${search_dir}/${filename}-${pn}.png"
        if [[ -f "$candidate" ]]; then
            FOUND_PNG="$candidate"
            return 0
        fi
    done
    return 1
}

# ── Main loop ─────────────────────────────────────────────────────────────
line_no=1
# `|| [[ -n $line ]]` keeps a last row that has no trailing newline.
while IFS= read -r line <&3 || [[ -n "$line" ]]; do
    line_no=$((line_no + 1))
    # Trim the carriage-return of Windows-formatted CSVs; skip blank lines.
    line="${line%$'\r'}"
    [[ -z "${line//[[:space:]]/}" ]] && continue

    split_csv_line "$line"
    field_at "$file_col";  filename="$FIELD"
    field_at "$page_col";  page_number="$FIELD"
    field_at "$label_col"; category="$FIELD"

    if ! is_safe_name "$filename"; then
        echo "Invalid file (line $line_no): '$filename' — skipped"
        invalid=$((invalid + 1))
        continue
    fi
    if [[ ! "$page_number" =~ ^[0-9]+$ ]]; then
        echo "Invalid page (line $line_no): $filename  page '$page_number' — skipped"
        invalid=$((invalid + 1))
        continue
    fi
    # Base 10 explicitly: a bare 08 would otherwise be read as an invalid octal number.
    page_number=$((10#$page_number))
    if ! is_safe_name "$category"; then
        echo "Invalid label (line $line_no): $filename  page $page_number  label '$category' — skipped"
        invalid=$((invalid + 1))
        continue
    fi

    if ! find_png "$INPUT_DIR" "$filename" "$page_number"; then
        echo "Not found: $filename  page $page_number"
        not_found=$((not_found + 1))
        continue
    fi

    # Only now, with a page in hand, does the label folder get created: an
    # empty folder would still count as a category to training.
    category_dir="${OUTPUT_DIR}/${category}"
    if [[ "$DRY_RUN" == true ]]; then
        if [[ "$USE_MOVE" == true ]]; then
            echo "[dry-run] move: $FOUND_PNG → $category_dir/"
        else
            echo "[dry-run] copy: $FOUND_PNG → $category_dir/"
        fi
    else
        mkdir -p "$category_dir"
        if [[ "$USE_MOVE" == true ]]; then
            mv "$FOUND_PNG" "$category_dir/"
        else
            cp "$FOUND_PNG" "$category_dir/"
        fi
    fi
    copied=$((copied + 1))
done
exec 3<&-

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
prefix=""; [[ "$DRY_RUN" == true ]] && prefix="[dry-run] "
verb="Copied"; [[ "$USE_MOVE" == true ]] && verb="Moved"
echo "${prefix}Done. ${verb} $copied file(s) | $not_found page(s) not found | $invalid invalid row(s) skipped."
