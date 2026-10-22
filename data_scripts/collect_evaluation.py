#!/usr/bin/env python3
"""
data_scripts/collect_evaluation.py — one evaluation run's outputs, kept apart with their provenance
(atrium-project#15).

Why it exists. The `vX.3` result files in ``result/tables`` were measured on the 5,449-page set, most of
which turned out to be training pages. They stay as they are: they are the record of what the papers
reported. A corrected run (``supplementary/scripts/eval_vX4_job.sh``) writes into the same folders, so
this script moves that run's files into a folder of their own, ``result/evaluations/<stamp>_vX4/``, keeping
their paths under ``result/`` (``tables/``, ``plots/``, ``stats/``, ``paradata/``), and writes
``PROVENANCE.md`` next to them:

* the repository commit and release the run used, and its job and host;
* the model revisions with their Hub commits (the prefetch's ``*_vX4_hub_revisions.csv``);
* the folds CSV (name and SHA-256) and the LINDAT record the pages come from;
* per table: the fold's test rows in the folds CSV, the rows without a PNG name, the pages it names,
  the pages evaluated and those named but not evaluated;
* each step's outcome, from the run's step log.

It reads the run's own lists (``<stamp>_vX4_outputs.txt``, ``_tables.tsv``, ``_steps.txt``), not its
console log, so it works on a run that has finished, whichever release ran it. Nothing is deleted: a file
the run wrote is moved, and a file it did not write is never touched. Run it from the repository root::

    python3 data_scripts/collect_evaluation.py --stamp 20261022-0909 \\
        --folds_csv /lnet/.../licensed_crossval_folds.csv
"""

import argparse
import csv
import hashlib
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_by_document_overlap as ev  # noqa: E402

#: The LINDAT record of the licence-compatible pages (README [^17]).
LINDAT_RECORD = "http://hdl.handle.net/20.500.12800/1-6184"
_FOLD = re.compile(r"^fold\d+$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_rows(folds_csv: Path, fold: str) -> dict:
    """The fold's `test` rows: how many, how many without a PNG name, and the pages the others name."""
    rows, unnamed, pages = 0, 0, set()
    with folds_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get(fold) or "").strip().lower() != "test":
                continue
            rows += 1
            name = (row.get("PNG") or "").strip()
            if name:
                pages.add(ev._key(name))
            else:
                unnamed += 1
    return {"rows": rows, "unnamed": unnamed, "pages": pages}


def evaluated_pages(table: Path) -> set:
    return {(row["FILE"], int(row["PAGE"])) for row in ev.read_results(table)}


def collect(
    results: Path, stamp: str, dest_root: Path, folds_csv: Path = None, lindat_record: str = LINDAT_RECORD
) -> Path:
    stats = results / "stats"
    listing = stats / f"{stamp}_vX4_outputs.txt"
    manifest = stats / f"{stamp}_vX4_tables.tsv"
    steps = stats / f"{stamp}_vX4_steps.txt"
    for required in (listing, manifest, steps):
        if not required.is_file():
            raise FileNotFoundError(f"{required}: not found; is {stamp} the stamp of a finished eval_vX4_job.sh run?")
    dest = dest_root / f"{stamp}_vX4"
    if dest.exists():
        raise FileExistsError(f"{dest} exists: this run was collected already")

    tables = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
    log = steps.read_text(encoding="utf-8")
    hub = sorted(stats.glob("*_vX4_hub_revisions.csv"))
    if folds_csv is None:
        named = re.search(r"^folds CSV (.+)$", log, re.M)
        folds_csv = Path(named.group(1)) if named else None

    # what the run wrote, its own lists included; never a file outside `results`
    outputs = [Path(line.strip()) for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs += [listing, manifest, steps, stats / f"{stamp}_vX4_summary.md"]
    moved = {}
    resolved_results = results.resolve()
    for path in dict.fromkeys(outputs):
        if not path.is_file():
            continue
        try:
            relative = path.resolve().relative_to(resolved_results)
        except ValueError:
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), target)
        moved[str(path)] = target
    for path in hub[-1:]:
        target = dest / "stats" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    # the manifest names the tables where they are now
    for row in tables:
        row["table"] = str(moved.get(str(Path(row["table"])), Path(row["table"])))
    with (dest / "stats" / manifest.name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["set", "revision", "fold", "top_n", "table"], delimiter="\t")
        writer.writeheader()
        writer.writerows(tables)

    (dest / "PROVENANCE.md").write_text(
        provenance(stamp, log, hub[-1:], tables, folds_csv, lindat_record), encoding="utf-8"
    )
    return dest


def provenance(stamp: str, log: str, hub: list, tables: list, folds_csv, lindat_record: str) -> str:
    head = [line for line in log.splitlines()[:4] if line.strip()]
    lines = [
        f"# The vX.4 evaluation of {stamp}: provenance",
        "",
        "A corrected evaluation (atrium-project#15), kept apart from the original `vX.3` result files in",
        "`result/tables/`, which stay as they were. The 5,449-page figures there are **not** held-out accuracy:",
        "most of those pages are in a fold's training partition. No figure here replaces one in a manuscript",
        "until the co-authors have reviewed it.",
        "",
        "## The run",
        "",
        "```",
        *head,
        "```",
        "",
        "## The models",
        "",
    ]
    if hub:
        rows = list(csv.reader(hub[0].open(encoding="utf-8")))
        lines += ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    else:
        lines.append("No `*_vX4_hub_revisions.csv`: the prefetch did not record the Hub commits.")
    lines += ["", "## The data", ""]
    if folds_csv is not None and Path(folds_csv).is_file():
        lines.append(f"- Folds: `{Path(folds_csv).name}`, SHA-256 `{sha256(Path(folds_csv))}`.")
    else:
        lines.append(f"- Folds: `{folds_csv}` (not readable here, so no checksum).")
    lines += [
        f"- Pages: the LINDAT record {lindat_record}, extracted per fold by `data_scripts/stage_fold_pages.py`.",
        "",
        "## Pages per table",
        "",
        "A fold's test set is the rows whose fold column says `test`. A row without a PNG name names no page and",
        "cannot be evaluated; a named page that is not in the table was not evaluated (not found, or unreadable:",
        "the run's paradata names each skip).",
        "",
        "| Set | Revision | Fold | Top-N | Test rows | Without a name | Named pages | Evaluated | Not evaluated | Table |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    folds = {}
    for row in tables:
        table = Path(row["table"])
        evaluated = evaluated_pages(table) if table.is_file() else set()
        fold = row["fold"]
        if row["set"].endswith(" test") and _FOLD.match(fold) and folds_csv is not None and Path(folds_csv).is_file():
            info = folds.setdefault(fold, fold_rows(Path(folds_csv), fold))
            counts = [info["rows"], info["unnamed"], len(info["pages"]), len(evaluated), len(info["pages"] - evaluated)]
        else:
            counts = ["–", "–", "–", len(evaluated), "–"]
        lines.append(
            f"| {row['set']} | {row['revision']} | {fold} | {row['top_n']} | "
            + " | ".join(map(str, counts))
            + f" | `{table.name}` |"
        )
    outcomes = [line for line in log.splitlines() if line.startswith(("OK ", "FAILED ", "SKIPPED "))]
    failed = [line for line in outcomes if not line.startswith("OK ")]
    lines += ["", "## Steps", "", f"{len(outcomes)} steps, {len(failed)} not OK.", ""]
    if failed:
        lines += ["```", *failed, "```", ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stamp", required=True, help="the run's stamp, as in result/stats/<stamp>_vX4_summary.md")
    parser.add_argument("--results", type=Path, default=Path("result"), help="the results folder (default result)")
    parser.add_argument("--dest", type=Path, default=None, help="default <results>/evaluations")
    parser.add_argument("--folds_csv", type=Path, default=None, help="default: the one the step log names")
    parser.add_argument("--lindat-record", default=LINDAT_RECORD)
    args = parser.parse_args(argv)
    try:
        dest = collect(
            args.results, args.stamp, args.dest or args.results / "evaluations", args.folds_csv, args.lindat_record
        )
    except (FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"collected into {dest}; see {dest / 'PROVENANCE.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
