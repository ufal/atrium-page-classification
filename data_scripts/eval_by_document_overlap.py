#!/usr/bin/env python3
"""
data_scripts/eval_by_document_overlap.py — accuracy on seen and unseen documents (atrium-project#15).

Why it exists. The cross-validation splits sample **pages**, not documents: a test page often comes
from a report whose other pages were in training, which is easier than a report the model never
saw. And the "5,449-page evaluation set" turned out to overlap the training partitions. Reporting
one accuracy hides both. This script takes one split and one or more results tables and reports
the top-1 accuracy separately for

* **in training** — the page itself was a training page (a held-out set should have none);
* **seen document** — a held-out page whose document has other pages in training;
* **unseen document** — a page from a document with no page in training;

plus the pages the split does not list. Nothing is trained or run: it reads files.

The split is either a folds CSV (a ``PNG`` column and one ``foldN`` column of train/dev/test, as
``run.py --folds_csv`` reads it) with ``--fold foldN``, or a ``*_DATASETS.txt`` file that
``run.py`` writes for each fold in ``result/stats/``. A results table needs ``FILE``, ``PAGE`` and
``TRUE``, and ``CLASS-1`` or ``CATEGORY``: the ``*_TEST*.csv`` and ``*_EVAL.csv`` tables of
``result/tables/``. Pages are matched by the identity every other part of the repository uses
(``utils.doc_id_and_page``). Run it from the repository root::

    python3 data_scripts/eval_by_document_overlap.py \\
        --split result/stats/20251005-0434_model_v93_FOLD_1_DATASETS.txt \\
        --results result/tables/20251020-1849_5449_model_v43_TOP-1_EVAL.csv

    python3 data_scripts/eval_by_document_overlap.py --split folds.csv --fold fold1 \\
        --results result/tables/*_v14_TEST.csv --json
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import doc_id_and_page  # noqa: E402

_SECTION = re.compile(r"^(\w+) set \((\d+) images\):")
_TRAIN = {"training", "train"}
GROUPS = ("in training", "seen document", "unseen document", "not in the split")


def _key(name: str):
    document, page = doc_id_and_page(name)
    return document, page if page is not None else 1


def read_split(path: Path, fold: str = None) -> dict:
    """{(document, page): partition} from a folds CSV column or a *_DATASETS.txt file."""
    partitions = {}
    if path.suffix.lower() == ".csv":
        if not fold:
            raise ValueError("a folds CSV needs --fold (e.g. fold1)")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = (row.get(fold) or "").strip().lower()
                if value:
                    partitions[_key(row["PNG"])] = value
        return partitions
    section = None
    for line in path.read_text(encoding="utf-8").splitlines():
        found = _SECTION.match(line)
        if found:
            section = found.group(1).lower()
        elif line.strip() and section:
            partitions[_key(Path(line.strip()).name)] = section
    return partitions


def evaluate(split: dict, rows: list) -> dict:
    """Top-1 accuracy per group: {group: {"pages": n, "correct": k}}."""
    train_documents = {document for (document, _), part in split.items() if part in _TRAIN}
    counts = defaultdict(Counter)
    for row in rows:
        key = (row["FILE"], int(row["PAGE"]))
        part = split.get(key)
        if part is None:
            group = "not in the split"
        elif part in _TRAIN:
            group = "in training"
        elif key[0] in train_documents:
            group = "seen document"
        else:
            group = "unseen document"
        predicted = row.get("CLASS-1") or row.get("CATEGORY")
        counts[group]["pages"] += 1
        counts[group]["correct"] += predicted == row["TRUE"]
    return {group: dict(counts[group]) for group in GROUPS if counts[group]}


def read_results(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and not {"FILE", "PAGE", "TRUE"} <= set(rows[0]):
        raise ValueError(f"{path}: needs FILE, PAGE and TRUE columns")
    return rows


def markdown(name: str, result: dict) -> str:
    total = sum(g["pages"] for g in result.values())
    correct = sum(g["correct"] for g in result.values())
    lines = [f"**{name}**: {total} pages, top-1 {100 * correct / total:.2f} %" if total else f"**{name}**: no rows", ""]
    lines += ["| Pages | Count | Top-1 |", "|---|---:|---:|"]
    for group, numbers in result.items():
        lines.append(f"| {group} | {numbers['pages']} | {100 * numbers['correct'] / numbers['pages']:.2f} % |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].split("— ", 1)[-1])
    parser.add_argument("--split", type=Path, required=True, help="a folds CSV or a *_DATASETS.txt file")
    parser.add_argument("--fold", default=None, help="the folds CSV column, e.g. fold1")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="results tables (CSV)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)

    split = read_split(args.split, args.fold)
    report = {str(path): evaluate(split, read_results(path)) for path in args.results}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("\n".join(markdown(Path(name).name, result) for name, result in report.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
