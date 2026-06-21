#!/bin/bash
# move_single.sh — Move single-file subdirectories into a shared flat folder.
#
# Scans a source directory for subdirectories containing exactly one file and
# moves that file to a common target directory, then removes the now-empty
# subdirectory.  Useful before annotation: single-page PDFs end up in a flat
# "onepagers" folder rather than in redundant one-entry subdirectories.
#
# Usage:
#   ./move_single.sh [OPTIONS]
#
# Options:
#   -s, --source DIR    Directory to scan          (default: current directory)
#   -t, --target DIR    Destination for moved files (default: ./onepagers)
#   -n, --dry-run       Show what would happen without making any changes
#   -h, --help          Show this help message and exit
#
# Examples:
#   ./move_single.sh
#   ./move_single.sh --source ./converted_pdfs --target ./onepagers
#   ./move_single.sh --source ./converted_pdfs --dry-run

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
SOURCE_DIR="."
TARGET_DIR="./onepagers"
DRY_RUN=false

# ── Usage ───────────────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,1\}//'
    exit 0
}

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--source)  SOURCE_DIR="$2"; shift 2 ;;
        -t|--target)  TARGET_DIR="$2"; shift 2 ;;
        -n|--dry-run) DRY_RUN=true;    shift   ;;
        -h|--help)    usage ;;
        *) echo "Error: unknown option '$1'"; echo "Run with --help for usage."; exit 1 ;;
    esac
done

# ── Validation ───────────────────────────────────────────────────────────────
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# P1 FIX: mkdir was commented out in the original — the script would silently
# fail to move any files when the target directory did not already exist.
if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$TARGET_DIR"
fi

# ── Main loop ────────────────────────────────────────────────────────────────
moved=0
skipped=0

while IFS= read -r -d '' dir; do
    # Single find call (replaces the original double-find: one for count, one
    # for the filename).  mapfile builds the array in one pass; handles
    # filenames with spaces, newlines, and other special characters safely.
    mapfile -t files < <(find "$dir" -maxdepth 1 -type f 2>/dev/null)
    count="${#files[@]}"

    if [[ "$count" -eq 1 ]]; then
        file="${files[0]}"
        if [[ "$DRY_RUN" == true ]]; then
            echo "[dry-run] move:   $file → $TARGET_DIR/"
            echo "[dry-run] remove: $dir/"
        else
            mv "$file" "$TARGET_DIR/"
            rmdir "$dir"
        fi
        moved=$((moved + 1))
    else
        skipped=$((skipped + 1))
    fi
done < <(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] Would move $moved file(s) to '$TARGET_DIR/' | $skipped multi-file director(ies) skipped."
else
    echo "Done. Moved $moved file(s) to '$TARGET_DIR/' | $skipped multi-file director(ies) skipped."
fi
