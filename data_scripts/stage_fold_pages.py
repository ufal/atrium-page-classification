#!/usr/bin/env python3
"""
data_scripts/stage_fold_pages.py — one fold's test pages out of the LINDAT zips (atrium-project#15).

Why it exists. ``run.py --eval --folds_csv`` scores a fold's test pages out of a folder with one
sub-folder per category, and finds them by the folds CSV's exact ``PNG`` names
(``CTX192700593-08.png``). The LINDAT record (http://hdl.handle.net/20.500.12800/1-6184) holds those
files only in its category zips (``DRAW.zip``, ``TEXT_part001.zip``, …). This script extracts the
pages of one fold's partition into ``<out>/<CATEGORY>/<PNG>``:

* all 11 category folders are made, an empty one too: ``collect_images`` numbers the classes by
  the folders it finds, so a missing one would shift every label after it;
* every page the CSV lists must be found in a zip, or the missing names go to
  ``<out>/missing.txt`` and the exit code is 1 (``--allow-missing`` goes on without them);
* a page's zip must be its CSV category's (the zip name without ``_partNNN``), or exit 1;
* a file already in place at the member's size is not extracted again, so a stopped run resumes.

``--dry-run`` makes the same checks and writes nothing. Run it from the repository root::

    python3 data_scripts/stage_fold_pages.py --folds_csv licensed_crossval_folds.csv --fold fold1 \\
        --zips licensed_archives --out stage/fold1
"""

import argparse
import csv
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_registry import CATEGORIES  # noqa: E402

_PART = re.compile(r"_part\d+$")


def wanted_pages(folds_csv: Path, fold: str, part: str = "test") -> dict:
    """{PNG name: category} of the CSV rows whose `fold` cell is `part`."""
    with folds_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        absent = {"PNG", "categ", fold} - set(reader.fieldnames or ())
        if absent:
            raise ValueError(f"{folds_csv}: no column {sorted(absent)}")
        pages = {
            row["PNG"].strip(): row["categ"].strip() for row in reader if (row[fold] or "").strip().lower() == part
        }
    unknown = sorted(set(pages.values()) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"{folds_csv}: categories not in model_registry.CATEGORIES: {unknown}")
    return pages


def zip_category(path: Path) -> str:
    """The category a zip holds: its name without the extension and a ``_partNNN`` suffix."""
    return _PART.sub("", path.stem)


def stage(pages: dict, zips: Path, out: Path, dry_run: bool = False) -> dict:
    """Extract `pages` from the zips in `zips` into `out/<category>/`; report what was found."""
    if not dry_run:
        for category in CATEGORIES:
            (out / category).mkdir(parents=True, exist_ok=True)
    found, mismatched = {}, {}
    extracted = 0
    for archive in sorted(zips.glob("*.zip")):
        category = zip_category(archive)
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                name = PurePosixPath(member.filename).name
                if member.is_dir() or name not in pages or name in found:
                    continue
                found[name] = member.file_size
                if pages[name] != category:
                    mismatched[name] = archive.name
                    continue
                target = out / category / name
                if dry_run or (target.is_file() and target.stat().st_size == member.file_size):
                    continue
                with handle.open(member) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                extracted += 1
    missing = sorted(set(pages) - set(found))
    staged = Counter(pages[name] for name in found if name not in mismatched)
    return {
        "staged": staged,
        "bytes": sum(found.values()),
        "extracted": extracted,
        "missing": missing,
        "mismatched": mismatched,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].split("— ", 1)[-1])
    parser.add_argument("--folds_csv", type=Path, required=True, help="the folds CSV (PNG, categ, foldN)")
    parser.add_argument("--fold", required=True, help="the fold column, e.g. fold1")
    parser.add_argument("--part", default="test", choices=("test", "dev", "train"), help="the partition (test)")
    parser.add_argument("--zips", type=Path, required=True, help="the folder of the LINDAT category zips")
    parser.add_argument("--out", type=Path, required=True, help="the eval folder to fill")
    parser.add_argument("--allow-missing", action="store_true", help="exit 0 although pages are missing")
    parser.add_argument("--dry-run", action="store_true", help="check the zips, write nothing")
    args = parser.parse_args(argv)

    pages = wanted_pages(args.folds_csv, args.fold, args.part)
    report = stage(pages, args.zips, args.out, args.dry_run)

    print(f"{args.fold} {args.part}: {len(pages)} pages in {args.folds_csv.name}")
    for category in CATEGORIES:
        print(f"  {category:<8} {report['staged'][category]}")
    print(f"  {sum(report['staged'].values())} found, {report['bytes'] / 1e9:.2f} GB; {report['extracted']} extracted")
    status = 0
    missing_list = args.out / "missing.txt"
    if not args.dry_run and missing_list.is_file():
        missing_list.unlink()  # from an earlier run
    if report["mismatched"]:
        print(f"ERROR: {len(report['mismatched'])} pages are in another category's zip:", file=sys.stderr)
        for name, archive in list(report["mismatched"].items())[:20]:
            print(f"  {name}: {pages[name]} in the CSV, in {archive}", file=sys.stderr)
        status = 1
    if report["missing"]:
        print(
            f"{'WARNING' if args.allow_missing else 'ERROR'}: {len(report['missing'])} pages not in any zip",
            file=sys.stderr,
        )
        if not args.dry_run:
            missing_list.write_text("\n".join(report["missing"]) + "\n", encoding="utf-8")
            print(f"  listed in {missing_list}", file=sys.stderr)
        else:
            for name in report["missing"][:20]:
                print(f"  {name}", file=sys.stderr)
        if not args.allow_missing:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
