#!/usr/bin/env python3
"""
filtering.py
------------
Drop rows from a labeled CSV whose image file no longer exists
in the corresponding class subdirectory.

Usage:
    python filtering.py -d <image_dir> -i <input.csv> [-o <output.csv>]
    python filtering.py -d <image_dir> -i <input.csv> --dry-run
    python filtering.py -d <image_dir> -i <input.csv> --verbose

Arguments:
    -d / --dir      Path to the directory containing class subdirectories with PNG files.
                    Defaults to the folder this script lives in.
    -i / --input    CSV file to filter (must have FILE, PAGE, CLASS-1 columns).
                    Defaults to data_samples_labeled.csv in --dir.
    -o / --output   Path for the filtered output CSV.
                    Defaults to <input_stem>_filtered.csv next to the input file.

Options:
    -n / --dry-run  Show what would be removed without writing any output file.
    --verbose       Print every removed entry.
                    Default: show only the first 20, then a count of the rest.
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────

def parse_stem(stem: str):
    """Extract (file_prefix, page_number) from a PNG stem.

    Works for all naming conventions in this dataset:
        thesis-008              -> ('thesis', 8)
        caa_conference-02       -> ('caa_conference', 2)
        pages_online_13         -> ('pages_online', 13)
        presentation_thesis_01  -> ('presentation_thesis', 1)
        defense-1               -> ('defense', 1)

    Strategy: greedy-match everything up to the last [-_] followed only
    by digits at end-of-string.
    """
    m = re.match(r'^(.+)[-_](\d+)$', stem)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def build_valid_set(data_dir: Path) -> tuple:
    """Walk every class subdirectory and collect valid (prefix, page, class) triples.

    Returns:
        valid:           set of (file_prefix, page_number, class_name) triples.
        unmatched_names: list of filenames whose stems did not match the naming
                         convention and were therefore skipped.
    """
    valid: set = set()
    unmatched_names: list = []

    # P3 FIX: removed sorted() on both iterdir() and glob() — the results feed
    # directly into a set so sort order has no effect and adds unnecessary
    # overhead on large directory trees.
    for cls_dir in data_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        for png in cls_dir.glob("*.png"):
            prefix, page = parse_stem(png.stem)
            if prefix is not None:
                valid.add((prefix, page, cls_name))
            else:
                # P1 FIX: collect unmatched names instead of silently ignoring
                # them.  A file that does not match the convention is excluded
                # from the valid set, which would cause its CSV row to be
                # incorrectly flagged as "missing".
                unmatched_names.append(f"{cls_name}/{png.name}")

    return valid, unmatched_names


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a labeled CSV to only rows whose image exists in the class directory tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-d", "--dir",
        type=Path, default=None, metavar="IMAGE_DIR",
        help="Directory containing class subdirectories with PNG images. "
             "Defaults to the folder this script lives in.",
    )
    parser.add_argument(
        "-i", "--input",
        type=Path, default=None, metavar="INPUT_CSV",
        help="CSV file to filter (must have FILE, PAGE, CLASS-1 columns). "
             "Defaults to data_samples_labeled.csv inside --dir.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path, default=None, metavar="OUTPUT_CSV",
        help="Path for the filtered output CSV. "
             "Defaults to <input_stem>_filtered.csv next to the input file.",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Show what would be removed without writing any output file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every removed entry. "
             "Default: show only the first 20, then a count of the rest.",
    )
    return parser.parse_args()


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── resolve paths ────────────────────────────────────────────────────
    script_dir = Path(__file__).parent.resolve()

    data_dir = args.dir.resolve() if args.dir else script_dir
    if not data_dir.is_dir():
        sys.exit(f"[ERROR] Image directory not found: {data_dir}")

    csv_in = (
        args.input.resolve() if args.input
        else data_dir / "data_samples_labeled.csv"
    )
    if not csv_in.exists():
        sys.exit(f"[ERROR] Input CSV not found: {csv_in}")

    csv_out = (
        args.output.resolve() if args.output
        else csv_in.parent / f"{csv_in.stem}_filtered.csv"
    )

    # ── index valid (file, page, class) triples from directory tree ───────
    valid, unmatched_names = build_valid_set(data_dir)
    print(f"Indexed {len(valid)} image file(s) from: {data_dir}")

    # P1 FIX: warn when PNG files don't match the naming convention instead
    # of silently excluding them from the valid set (which would cause their
    # CSV rows to be incorrectly removed as "missing").
    if unmatched_names:
        print(
            f"Warning: {len(unmatched_names)} PNG file(s) did not match the "
            f"'<name>[-_]<page>.png' naming convention and were excluded from "
            f"the valid set.  These rows will appear in 'removed' even though "
            f"the files exist on disk."
        )
        if args.verbose:
            for name in unmatched_names:
                print(f"  unmatched: {name}")

    # ── filter CSV rows ───────────────────────────────────────────────────
    kept: list = []
    removed: list = []

    with csv_in.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames

        # P1 FIX: reader.fieldnames is None when the file is completely empty
        # (no header row at all).  The original code would raise TypeError:
        # "argument of type 'NoneType' is not iterable" on the subset check.
        if not fieldnames:
            sys.exit(f"[ERROR] Input CSV appears to be empty (no header row): {csv_in}")

        required = {"FILE", "PAGE", "CLASS-1"}
        missing_cols = required - set(fieldnames)
        if missing_cols:
            sys.exit(f"[ERROR] Input CSV is missing required columns: {missing_cols}")

        for row in reader:
            key = (row["FILE"], int(row["PAGE"]), row["CLASS-1"])
            (kept if key in valid else removed).append(row)

    # ── report ────────────────────────────────────────────────────────────
    print(f"Input CSV:    {csv_in}  ({len(fieldnames)} columns)")
    print(f"Rows kept:    {len(kept)}")
    print(f"Rows removed: {len(removed)}")

    if removed:
        # Per-class breakdown — far more useful than a bare total for
        # diagnosing annotation drift or stale directory trees.
        removal_by_class = Counter(r["CLASS-1"] for r in removed)
        print("\n── Removed by class ─────────────────────────────────────────────")
        for cls, cnt in sorted(removal_by_class.items(), key=lambda x: -x[1]):
            print(f"  {cls:<20s}  {cnt} row(s)")

        # P2 FIX: cap the per-row listing so large removals don't flood the
        # terminal.  --verbose prints everything; default shows the first 20.
        MAX_DEFAULT = 20
        to_show = removed if args.verbose else removed[:MAX_DEFAULT]
        tail = len(removed) - MAX_DEFAULT

        label = "(all)" if args.verbose else f"(first {MAX_DEFAULT} of {len(removed)})"
        print(f"\n── Removed entries {label} ─────────────────────────────────────")
        for r in to_show:
            print(f"  {r['FILE']:<25s}  page {int(r['PAGE']):3d}  class {r['CLASS-1']}")
        if not args.verbose and tail > 0:
            print(f"  … and {tail} more.  Run with --verbose to see all.")

    # ── write output ──────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n[dry-run] Would write {len(kept)} row(s) to: {csv_out}")
        return

    with csv_out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"\nFiltered CSV written to: {csv_out}")


if __name__ == "__main__":
    main()
