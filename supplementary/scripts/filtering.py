#!/usr/bin/env python3
"""
filtering.py
------------
Re-sync a labeled CSV with the class-directory tree after a correction pass.

For every row, the page image is looked up in the class subdirectories:

  * still in the folder the row names          -> row kept unchanged;
  * now in exactly ONE other class folder      -> row RELABELLED to that folder
                                                  (a reviewer moved the page);
  * in no folder, or in two or more others     -> row removed and reported
                                                  (missing / ambiguous).

Pass --no-relabel to drop moved pages instead of relabelling them.

Usage:
    python filtering.py -d <image_dir> -i <input.csv> [-o <output.csv>]
    python filtering.py -d <image_dir> -i <input.csv> --dry-run
    python filtering.py -d <image_dir> -i <input.csv> --verbose

Arguments:
    -d / --dir      Path to the directory containing class subdirectories with PNG files.
                    Defaults to the folder this script lives in.
    -i / --input    CSV file to filter. Needs FILE, PAGE and a label column: CLASS
                    (the annotation CSV), else CLASS-1 (classifier output), else
                    CATEGORY. Column names are matched case-insensitively.
                    Defaults to data_samples_labeled.csv in --dir.
    -o / --output   Path for the filtered output CSV.
                    Defaults to <input_stem>_filtered.csv next to the input file.

Options:
    --class-column NAME  Use NAME as the label column instead of the lookup above.
    --no-relabel    Drop rows whose page moved to another class folder instead of
                    relabelling them (the behaviour before relabelling existed).
    -n / --dry-run  Show what would be relabelled and removed without writing any file.
    --verbose       Print every relabelled and removed entry.
                    Default: show only the first 20 of each, then a count of the rest.

When a classifier table is relabelled through CLASS-1, its CATEGORY alias column is
updated with it; score columns are left as the model wrote them.
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

#: Label columns tried in order when --class-column is not given: the README's
#: annotation format first, then the classifier's Top-N output, then its alias.
LABEL_COLUMNS = ("CLASS", "CLASS-1", "CATEGORY")


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


def find_column(fieldnames, name: str):
    """Return the header in *fieldnames* that equals *name* case-insensitively, or None."""
    wanted = name.strip().upper()
    for field in fieldnames or ():
        if field is not None and field.strip().upper() == wanted:
            return field
    return None


def resolve_label_column(fieldnames, override=None):
    """Pick the column holding each row's class.

    With *override* (``--class-column``), that column or None when the CSV lacks it.
    Otherwise the first of CLASS, CLASS-1, CATEGORY the CSV has, or None.
    Returns the header exactly as spelled in the CSV.
    """
    if override is not None:
        return find_column(fieldnames, override)
    for candidate in LABEL_COLUMNS:
        column = find_column(fieldnames, candidate)
        if column is not None:
            return column
    return None


def index_locations(valid: set) -> dict:
    """{(file_prefix, page): {class, ...}} — every class folder each page sits in."""
    locations: dict = {}
    for prefix, page, cls in valid:
        locations.setdefault((prefix, page), set()).add(cls)
    return locations


def filter_rows(rows, valid: set, label_col: str, *, file_col: str = "FILE", page_col: str = "PAGE",
                relabel: bool = True, mirror_col=None) -> tuple:
    """Sort CSV rows by where their page now is in the class tree.

    Args:
        rows:       iterable of dict rows (as from csv.DictReader).
        valid:      the (file_prefix, page, class) set from build_valid_set().
        label_col:  column holding the row's class.
        relabel:    relabel a row whose page sits in exactly one OTHER class folder;
                    when False such a row is removed as "missing", as before.
        mirror_col: a column that repeats the label (CATEGORY beside CLASS-1); when it
                    held the old label it is relabelled along with *label_col*.

    Returns:
        kept:       every row to write, in input order — unchanged rows and relabelled
                    ones (relabelled rows are copies carrying the new label).
        relabelled: list of (row, old_label, new_label), row being the relabelled copy.
        removed:    list of (row, reason); reason is "missing", "invalid page", or
                    "ambiguous: A, B" naming the folders the page was found in.
    """
    locations = index_locations(valid) if relabel else {}
    kept: list = []
    relabelled: list = []
    removed: list = []

    for row in rows:
        try:
            page = int(str(row.get(page_col)).strip())
        except (TypeError, ValueError):
            removed.append((row, "invalid page"))
            continue
        file_name = str(row.get(file_col) or "").strip()
        label = str(row.get(label_col) or "").strip()

        if (file_name, page, label) in valid:
            kept.append(row)
            continue
        if not relabel:
            removed.append((row, "missing"))
            continue

        others = sorted(locations.get((file_name, page), set()) - {label})
        if len(others) == 1:
            new_label = others[0]
            new_row = dict(row)
            new_row[label_col] = new_label
            if mirror_col is not None and str(row.get(mirror_col, "")).strip() == label:
                new_row[mirror_col] = new_label
            kept.append(new_row)
            relabelled.append((new_row, label, new_label))
        elif others:
            removed.append((row, "ambiguous: " + ", ".join(others)))
        else:
            removed.append((row, "missing"))

    return kept, relabelled, removed


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-sync a labeled CSV with the class directory tree: keep rows whose image is "
                    "where the row says, relabel rows whose image moved to one other class folder, "
                    "and drop the rest.",
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
        help="CSV file to filter: FILE, PAGE and a label column (CLASS, else CLASS-1, else CATEGORY). "
             "Defaults to data_samples_labeled.csv inside --dir.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path, default=None, metavar="OUTPUT_CSV",
        help="Path for the filtered output CSV. "
             "Defaults to <input_stem>_filtered.csv next to the input file.",
    )
    parser.add_argument(
        "--class-column",
        default=None, metavar="NAME",
        help="Label column to use instead of looking for CLASS, CLASS-1, CATEGORY in that order.",
    )
    parser.add_argument(
        "--no-relabel",
        action="store_true",
        help="Drop rows whose page moved to another class folder instead of relabelling them.",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Show what would be relabelled and removed without writing any output file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every relabelled and removed entry. "
             "Default: show only the first 20 of each, then a count of the rest.",
    )
    return parser.parse_args()


# ── main ─────────────────────────────────────────────────────────────────

MAX_DEFAULT = 20


def _cell(row: dict, column: str) -> str:
    """A row's value for printing: stripped, and "" for a short row's missing cell."""
    value = row.get(column)
    return "" if value is None else str(value).strip()


def _print_capped(title: str, lines: list, verbose: bool) -> None:
    """Print a section of per-row lines, capped at MAX_DEFAULT unless --verbose."""
    label = "(all)" if verbose else f"(first {min(MAX_DEFAULT, len(lines))} of {len(lines)})"
    print(f"\n── {title} {label} ─────────────────────────────────────")
    for line in (lines if verbose else lines[:MAX_DEFAULT]):
        print(line)
    tail = len(lines) - MAX_DEFAULT
    if not verbose and tail > 0:
        print(f"  … and {tail} more.  Run with --verbose to see all.")


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

    # ── read CSV ──────────────────────────────────────────────────────────
    # utf-8-sig: a spreadsheet export often starts with a BOM, which would
    # otherwise glue itself to the first header name ("﻿FILE").
    with csv_in.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames

        # P1 FIX: reader.fieldnames is None when the file is completely empty
        # (no header row at all).  The original code would raise TypeError:
        # "argument of type 'NoneType' is not iterable" on the subset check.
        if not fieldnames:
            sys.exit(f"[ERROR] Input CSV appears to be empty (no header row): {csv_in}")
        rows = list(reader)

    file_col = find_column(fieldnames, "FILE")
    page_col = find_column(fieldnames, "PAGE")
    label_col = resolve_label_column(fieldnames, args.class_column)
    missing_cols = [name for name, col in (("FILE", file_col), ("PAGE", page_col)) if col is None]
    if label_col is None:
        missing_cols.append(args.class_column if args.class_column else "CLASS (or CLASS-1 / CATEGORY)")
    if missing_cols:
        sys.exit(f"[ERROR] Input CSV is missing required columns: {', '.join(missing_cols)}")

    # CATEGORY is the classifier's alias of CLASS-1; keep the two in step.
    category_col = find_column(fieldnames, "CATEGORY")
    mirror_col = category_col if label_col.strip().upper() == "CLASS-1" else None

    # ── filter CSV rows ───────────────────────────────────────────────────
    kept, relabelled, removed = filter_rows(
        rows, valid, label_col,
        file_col=file_col, page_col=page_col,
        relabel=not args.no_relabel, mirror_col=mirror_col,
    )

    # ── report ────────────────────────────────────────────────────────────
    print(f"Input CSV:    {csv_in}  ({len(fieldnames)} columns, label column '{label_col}')")
    print(f"Rows kept:    {len(kept) - len(relabelled)}")
    print(f"Relabelled:   {len(relabelled)}" + ("  (--no-relabel)" if args.no_relabel else ""))
    print(f"Rows removed: {len(removed)}")

    if relabelled:
        moves = Counter((old, new) for _, old, new in relabelled)
        print("\n── Relabelled by class ──────────────────────────────────────────")
        for (old, new), cnt in sorted(moves.items(), key=lambda x: -x[1]):
            print(f"  {old:<12s} -> {new:<12s}  {cnt} row(s)")
        _print_capped(
            "Relabelled entries",
            [f"  {_cell(r, file_col):<25s}  page {_cell(r, page_col):>3s}  {old} -> {new}"
             for r, old, new in relabelled],
            args.verbose,
        )

    if removed:
        # Per-class breakdown — far more useful than a bare total for
        # diagnosing annotation drift or stale directory trees.
        removal_by_class = Counter(_cell(r, label_col) for r, _ in removed)
        print("\n── Removed by class ─────────────────────────────────────────────")
        for cls, cnt in sorted(removal_by_class.items(), key=lambda x: -x[1]):
            print(f"  {cls:<20s}  {cnt} row(s)")
        _print_capped(
            "Removed entries",
            [f"  {_cell(r, file_col):<25s}  page {_cell(r, page_col):>3s}  class {_cell(r, label_col)}  ({reason})"
             for r, reason in removed],
            args.verbose,
        )

    # ── write output ──────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n[dry-run] Would write {len(kept)} row(s) to: {csv_out}")
        return

    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"\nFiltered CSV written to: {csv_out}")


if __name__ == "__main__":
    main()
