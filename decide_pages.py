#!/usr/bin/env python3
"""decide_pages.py — decision models on page images (issue #58, track 1; ufal/atrium-project#83).

An experiment CLI, outside the service and its image. It sends page images to a decision model
served at ``POST /v1/systemone`` (``decision_client.py``) and writes one ``DECIDE_RAW`` table per run.
The model runs on the server, so this needs no torch: it runs in the repository's own environment.

Three tasks (the #58 plan):

* **b1** — the 11 labels, standalone: one ``choice`` over ``model_registry.CATEGORIES``, each label
  described by its ``atrium_vocab`` definition. The control.
* **b2** — the cascade: the CNN answers every page (a ``run.py --raw`` table); a page whose raw top-1
  probability is below ``--tau`` goes to the decision model, which chooses among the CNN's top 3
  only. ``CLASS-1`` is the cascade's answer for every page, so the table scores as a whole.
* **b3** — four yes/no routes (OCR, HTR, tables, image description), scored against the routes the
  true label implies (``ROUTE_GOLD``, from the hub's "what each category suggests doing next").

The question sets are generated, never hand-edited: ``--write-question-sets`` writes
``decisions/page_category.v1.json`` and ``decisions/page_routes.v1.json`` (``tests/test_decide_pages.py``
fails when they drift). Typical runs, from the repository root::

    python3 decide_pages.py --write-question-sets
    # Phase A smoke: 3 public sample pages per label, three image sizes
    python3 decide_pages.py --task b1 --dir small_data_samples --per-label 3 \\
        --max-side 768 --max-side 1024 --max-side 1536
    # B1 on the fold-1 dev partition, staged with data_scripts/stage_fold_pages.py
    python3 decide_pages.py --task b1 --folds_csv licensed_crossval_folds.csv --fold fold1 --part dev \\
        --root stage/fold1_dev
    # B2 behind a CNN run of the same pages
    python3 decide_pages.py --task b2 --dir stage/fold1_dev --cnn-raw result/tables/<stamp>_EVAL_RAW.csv --tau 0.9
    python3 decide_pages.py --task b3 --dir small_data_samples --per-label 3

``--url``, ``--model`` and ``--timeout`` default to ``DECISION_MODEL_URL``, ``DECISION_MODEL_NAME`` and
``DECISION_TIMEOUT_S``. ``--resume --out <table>`` continues a stopped run. Which archive data may be
sent to which server is settled on ufal/atrium-project#83; the public ``small_data_samples`` may go
anywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import decision_client as dc
from model_registry import CATEGORIES

REPO_ROOT = Path(__file__).resolve().parent
QUESTION_DIR = REPO_ROOT / "decisions"
PAGE_SET_FILE = QUESTION_DIR / "page_category.v1.json"
ROUTE_SET_FILE = QUESTION_DIR / "page_routes.v1.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}

#: What the state says about every page. No file name or label is sent with it.
PAGE_STATE = {
    "task": "Decide about the attached image of one scanned page.",
    "collection": "archaeological reports and related material from Czech archives, 1920–2020",
}

PAGE_INSTRUCTIONS = "Which category best describes the attached page image?"

#: B3: the route questions and, per route, the true labels that imply it. Captions on DRAW/PHOTO
#: pages ("through OCR, if any") are not in the gold: the label does not say whether there is one.
ROUTES: Dict[str, str] = {
    "route_ocr": "Does the page carry printed or typewritten text to read?",
    "route_htr": "Does the page carry handwritten text to read?",
    "route_tables": "Is any part of the page a table, a form or a tabular legend?",
    "route_image_description": "Is a photograph or a drawing the page's main content?",
}
ROUTE_GOLD: Dict[str, Tuple[str, ...]] = {
    "route_ocr": ("TEXT", "TEXT_P", "TEXT_T", "LINE_P", "LINE_T"),
    "route_htr": ("TEXT", "TEXT_HW", "LINE_HW"),
    "route_tables": ("LINE_HW", "LINE_P", "LINE_T", "DRAW_L", "PHOTO_L"),
    "route_image_description": ("DRAW", "DRAW_L", "PHOTO", "PHOTO_L"),
}


# ── question sets ─────────────────────────────────────────────────────────────────────────────


def category_definitions() -> Dict[str, str]:
    """The 11 labels in ``CATEGORIES`` order, each with its ``atrium_vocab`` definition."""
    import atrium_vocab  # noqa: PLC0415 — the shared vocabulary module, vendored from the hub

    concepts = atrium_vocab.CONCEPTS["page-category"]
    missing = [label for label in CATEGORIES if label not in concepts]
    if missing:
        raise SystemExit(f"atrium_vocab has no definition for {missing}; re-vendor the shared files")
    return {label: concepts[label]["definition"]["en"] for label in CATEGORIES}


def page_questions(labels: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    """The ``page_category`` choice, over all 11 labels or over *labels* (B2's top 3)."""
    definitions = category_definitions()
    offered = list(labels) if labels else list(CATEGORIES)
    return dc.questions({"page_category": dc.choice(PAGE_INSTRUCTIONS, {k: definitions[k] for k in offered})})


def route_questions() -> Dict[str, Dict[str, Any]]:
    return dc.questions({route: dc.noul(text) for route, text in ROUTES.items()})


def question_sets() -> Dict[Path, Dict[str, Any]]:
    """The two committed question-set files, as they should read."""
    common = {"generated_by": "decide_pages.py --write-question-sets", "state": PAGE_STATE}
    page = page_questions()
    routes = route_questions()
    return {
        PAGE_SET_FILE: {
            "name": "page_category",
            "version": 1,
            **common,
            "source": "labels: model_registry.CATEGORIES (order); descriptions: atrium_vocab.CONCEPTS['page-category']",
            "questions": page,
            "schema_digest": dc.schema_digest(page),
        },
        ROUTE_SET_FILE: {
            "name": "page_routes",
            "version": 1,
            **common,
            "source": "#58 plan, B3; gold from the hub's 'What each category suggests doing next'",
            "questions": routes,
            "gold": {route: list(labels) for route, labels in ROUTE_GOLD.items()},
            "schema_digest": dc.schema_digest(routes),
        },
    }


def render_question_set(content: Mapping[str, Any]) -> str:
    return json.dumps(content, indent=2, ensure_ascii=False) + "\n"


def write_question_sets() -> List[Path]:
    QUESTION_DIR.mkdir(exist_ok=True)
    written = []
    for path, content in question_sets().items():
        path.write_text(render_question_set(content), encoding="utf-8")
        written.append(path)
    return written


# ── inputs ────────────────────────────────────────────────────────────────────────────────────


def pages_from_dir(root: Path) -> List[Tuple[Path, str]]:
    """``(image, true label)`` for every image in a ``<root>/<CATEGORY>/`` folder named after a label."""
    pages = []
    for label in CATEGORIES:
        folder = root / label
        if folder.is_dir():
            pages += [(p, label) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    return pages


def pages_from_folds(folds_csv: Path, fold: str, part: str, root: Path) -> List[Tuple[Path, str]]:
    """The pages of one fold partition, in the layout ``data_scripts/stage_fold_pages.py`` writes."""
    sys.path.insert(0, str(REPO_ROOT / "data_scripts"))
    from stage_fold_pages import wanted_pages  # noqa: PLC0415

    pages, absent = [], 0
    for name, label in sorted(wanted_pages(folds_csv, fold, part).items()):
        path = root / label / name
        if path.is_file():
            pages.append((path, label))
        else:
            absent += 1
    if absent:
        print(f"NOTE: {absent} {fold} {part} pages are not under {root}; left out", file=sys.stderr)
    return pages


def sample_per_label(pages: Sequence[Tuple[Path, str]], n: int, seed: int) -> List[Tuple[Path, str]]:
    """At most *n* pages per label, drawn with *seed* from the name-sorted list (reproducible)."""
    rng = random.Random(seed)
    by_label: Dict[str, List[Tuple[Path, str]]] = {}
    for page in sorted(pages, key=lambda item: (item[1], item[0].name)):
        by_label.setdefault(page[1], []).append(page)
    picked = []
    for label in CATEGORIES:
        group = by_label.get(label, [])
        picked += group if len(group) <= n else sorted(rng.sample(group, n), key=lambda item: item[0].name)
    return picked


def page_key(path: Path) -> Tuple[str, str]:
    """``(FILE, PAGE)`` as every results table of this repository names a page."""
    from utils import doc_id_and_page  # noqa: PLC0415 — utils pulls pandas and matplotlib

    document, page = doc_id_and_page(path)
    return document, str(page if page is not None else 1)


def read_cnn_raw(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    """``(FILE, PAGE) -> {label: p}`` from a ``run.py --raw`` table (one column per label)."""
    table: Dict[Tuple[str, str], Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        absent = {"FILE", "PAGE", *CATEGORIES} - set(reader.fieldnames or ())
        if absent:
            raise SystemExit(f"{path}: not a RAW table (no column {sorted(absent)})")
        for row in reader:
            table[(row["FILE"], str(row["PAGE"]))] = {label: float(row[label]) for label in CATEGORIES}
    return table


def top_n(probabilities: Mapping[str, float], n: int = 3) -> List[str]:
    return sorted(probabilities, key=lambda label: (-probabilities[label], CATEGORIES.index(label)))[:n]


# ── the run ───────────────────────────────────────────────────────────────────────────────────

BASE_COLUMNS = ["FILE", "PAGE", "TRUE"]
TAIL_COLUMNS = [
    "status",
    "error",
    "latency_ms",
    "input_tokens",
    "attempts",
    "schema_digest",
    "max_side",
    "image",
]


def columns_for(task: str) -> List[str]:
    if task == "b3":
        routes = [col for route in ROUTES for col in (f"P_{route}", f"GOLD_{route}")]
        return BASE_COLUMNS + routes + ["lane"] + TAIL_COLUMNS
    probs = [f"P_{label}" for label in CATEGORIES]
    extra = ["CNN_CLASS", "CNN_SCORE", "deferred", "options"] if task == "b2" else []
    return BASE_COLUMNS + ["CLASS-1"] + extra + probs + ["confidence", "lane"] + TAIL_COLUMNS


def _lane(p: Optional[float], accept_at: Optional[float], review_below: Optional[float]) -> str:
    if p is None or accept_at is None or review_below is None:
        return ""
    return dc.lane(p, accept_at, review_below)


def _decision_fields(d: dc.Decision, max_side: int, image: str) -> Dict[str, Any]:
    return {
        "status": d.status,
        "error": d.error or "",
        "latency_ms": d.latency_ms if d.latency_ms is not None else "",
        "input_tokens": d.input_tokens if d.input_tokens is not None else "",
        "attempts": d.attempts,
        "schema_digest": d.schema_digest,
        "max_side": max_side,
        "image": image,
    }


def decide_page(
    client: dc.DecisionClient,
    task: str,
    path: Path,
    true_label: str,
    max_side: int,
    image_label: str,
    cnn: Optional[Mapping[str, float]] = None,
    tau: Optional[float] = None,
    accept_at: Optional[float] = None,
    review_below: Optional[float] = None,
) -> Dict[str, Any]:
    """One output row. Never raises for a server or input problem: the row carries the status."""
    file_id, page = page_key(path)
    row: Dict[str, Any] = {"FILE": file_id, "PAGE": page, "TRUE": true_label}

    if task == "b3":
        d = client.decide(PAGE_STATE, route_questions(), images=[path], max_side=max_side)
        for route, labels in ROUTE_GOLD.items():
            answer = d.answers.get(route)
            row[f"P_{route}"] = answer.p_true if answer else ""
            row[f"GOLD_{route}"] = int(true_label in labels)
        row["lane"] = ""
        row.update(_decision_fields(d, max_side, image_label))
        return row

    offered = list(CATEGORIES)
    if task == "b2":
        if cnn is None:
            row.update({"CLASS-1": "", "CNN_CLASS": "", "CNN_SCORE": "", "deferred": 0, "options": ""})
            row.update(
                _decision_fields(dc.Decision("rejected", "", error="no CNN row for this page"), max_side, image_label)
            )
            return row
        cnn_class = top_n(cnn, 1)[0]
        row.update({"CNN_CLASS": cnn_class, "CNN_SCORE": round(cnn[cnn_class], 4), "CLASS-1": cnn_class})
        if cnn[cnn_class] >= tau:
            row.update({"deferred": 0, "options": "", "confidence": "", "lane": ""})
            row.update(_decision_fields(dc.Decision("not_requested", ""), max_side, image_label))
            return row
        offered = top_n(cnn, 3)
        row.update({"deferred": 1, "options": "|".join(offered)})

    d = client.decide(PAGE_STATE, page_questions(offered), images=[path], max_side=max_side)
    answer = d.answers.get("page_category")
    if answer:
        row["CLASS-1"] = answer.choice
        for label in CATEGORIES:
            row[f"P_{label}"] = answer.probabilities.get(label, "")
        row["confidence"] = answer.confidence
        row["lane"] = _lane(answer.confidence, accept_at, review_below)
    else:
        row.setdefault("CLASS-1", "")  # b2 keeps the CNN's answer when the decision failed
        row["confidence"] = ""
        row["lane"] = ""
    row.update(_decision_fields(d, max_side, image_label))
    return row


def _done_keys(out: Path) -> set:
    if not out.is_file():
        return set()
    with out.open(newline="", encoding="utf-8") as handle:
        return {(r["image"], str(r["max_side"])) for r in csv.DictReader(handle)}


def run(
    client: dc.DecisionClient,
    task: str,
    pages: Sequence[Tuple[Path, str]],
    max_sides: Sequence[int],
    out: Path,
    resume: bool = False,
    cnn_raw: Optional[Mapping[Tuple[str, str], Mapping[str, float]]] = None,
    tau: Optional[float] = None,
    accept_at: Optional[float] = None,
    review_below: Optional[float] = None,
    limit: Optional[int] = None,
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """Decide every page at every image size; rows are appended to *out* as they come."""
    columns = columns_for(task)
    done = _done_keys(out) if resume else set()
    if out.exists() and not resume:
        raise SystemExit(f"{out} exists; pass --resume to continue it or choose another --out")
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    rows: List[Dict[str, Any]] = []
    jobs = [(path, label, side) for side in max_sides for path, label in pages]
    if limit:
        jobs = jobs[:limit]
    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for index, (path, label, side) in enumerate(jobs, start=1):
            image_label = _relative(path)
            if (image_label, str(side)) in done:
                continue
            cnn = cnn_raw.get(page_key(path)) if cnn_raw is not None else None
            row = decide_page(client, task, path, label, side, image_label, cnn, tau, accept_at, review_below)
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if progress:
                print(
                    f"[{index}/{len(jobs)}] {image_label} @{side}: {row['status']} {row.get('CLASS-1', '')}", flush=True
                )
    return rows


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ── summary ───────────────────────────────────────────────────────────────────────────────────


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))], 1)


def summarise(task: str, rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Accuracy and operational figures of one table (all rows of it, resumed ones included)."""
    rows = list(rows)
    statuses = Counter(str(r["status"]) for r in rows)
    out: Dict[str, Any] = {"task": task, "rows": len(rows), "statuses": dict(statuses), "by_max_side": {}}
    for side in sorted({str(r["max_side"]) for r in rows}, key=int):
        part = [r for r in rows if str(r["max_side"]) == side]
        latencies = [float(r["latency_ms"]) for r in part if r["status"] == "ok" and r["latency_ms"] not in ("", None)]
        tokens = [float(r["input_tokens"]) for r in part if r["input_tokens"] not in ("", None)]
        entry: Dict[str, Any] = {
            "rows": len(part),
            "latency_ms_p50": _pct(latencies, 0.5),
            "latency_ms_p95": _pct(latencies, 0.95),
            "input_tokens_mean": round(statistics.fmean(tokens), 1) if tokens else None,
        }
        if task in ("b1", "b2"):
            answered = [r for r in part if r.get("CLASS-1")]
            entry["top1"] = (
                round(sum(r["CLASS-1"] == r["TRUE"] for r in answered) / len(answered), 4) if answered else None
            )
        if task == "b2":
            deferred = [r for r in part if str(r.get("deferred")) == "1"]
            fixed = sum(r["CNN_CLASS"] != r["TRUE"] and r["CLASS-1"] == r["TRUE"] for r in deferred)
            broken = sum(r["CNN_CLASS"] == r["TRUE"] and r["CLASS-1"] != r["TRUE"] for r in deferred)
            cnn_rows = [r for r in part if r.get("CNN_CLASS")]
            entry.update(
                {
                    "deferred": len(deferred),
                    "fixed": fixed,
                    "broken": broken,
                    "mcnemar_p": round(dc.mcnemar_exact(fixed, broken), 6),
                    "cnn_top1": round(sum(r["CNN_CLASS"] == r["TRUE"] for r in cnn_rows) / len(cnn_rows), 4)
                    if cnn_rows
                    else None,
                }
            )
        if task == "b3":
            per_route = {}
            ok = [r for r in part if r["status"] == "ok"]
            for route in ROUTES:
                pred = [float(r[f"P_{route}"]) >= 0.5 for r in ok]
                gold = [str(r[f"GOLD_{route}"]) == "1" for r in ok]
                tp = sum(p and g for p, g in zip(pred, gold))
                fp = sum(p and not g for p, g in zip(pred, gold))
                fn = sum(g and not p for p, g in zip(pred, gold))
                f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
                per_route[route] = {"f1": round(f1, 4) if f1 is not None else None, "tp": tp, "fp": fp, "fn": fn}
            exact = sum(
                all((float(r[f"P_{route}"]) >= 0.5) == (str(r[f"GOLD_{route}"]) == "1") for route in ROUTES) for r in ok
            )
            entry.update({"routes": per_route, "exact_route_set": round(exact / len(ok), 4) if ok else None})
        out["by_max_side"][side] = entry
    return out


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "model"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write-question-sets", action="store_true", help="regenerate decisions/*.json and exit")
    parser.add_argument("--task", choices=("b1", "b2", "b3"))
    source = parser.add_argument_group("pages")
    source.add_argument("--dir", type=Path, help="a folder with one sub-folder per label")
    source.add_argument("--folds_csv", type=Path, help="a folds CSV (PNG, categ, foldN columns)")
    source.add_argument("--fold", default="fold1")
    source.add_argument("--part", default="dev", choices=("train", "dev", "test"))
    source.add_argument("--root", type=Path, help="where stage_fold_pages.py put the partition's pages")
    source.add_argument("--per-label", type=int, help="at most N pages per label (the smoke set)")
    source.add_argument("--seed", type=int, default=420)
    source.add_argument("--limit", type=int, help="stop after N requests")
    parser.add_argument("--cnn-raw", type=Path, help="b2: the run.py --raw table of the same pages")
    parser.add_argument("--tau", type=float, help="b2: defer pages whose CNN raw top-1 is below this")
    parser.add_argument("--accept-at", type=float, help="lane threshold (set on dev)")
    parser.add_argument("--review-below", type=float, help="lane threshold (set on dev)")
    parser.add_argument(
        "--max-side", type=int, action="append", help="image long side in px (repeatable; default 1024)"
    )
    server = parser.add_argument_group("server")
    server.add_argument("--url", default=os.environ.get("DECISION_MODEL_URL"), help="($DECISION_MODEL_URL)")
    server.add_argument("--model", default=os.environ.get("DECISION_MODEL_NAME"), help="($DECISION_MODEL_NAME)")
    server.add_argument("--model-revision", default="", help="the served build or HF revision, for the paradata")
    server.add_argument("--timeout", type=float, default=_env_float("DECISION_TIMEOUT_S", 120.0))
    server.add_argument("--retries", type=int, default=2)
    server.add_argument(
        "--image-style", default="data_uri", choices=dc.IMAGE_STYLES, help="from decision_client --probe"
    )
    server.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--out", type=Path, help="the DECIDE_RAW table (default result/decide/<stamp>_...)")
    parser.add_argument("--resume", action="store_true", help="continue --out, skipping pages already in it")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.write_question_sets:
        for path in write_question_sets():
            print(f"wrote {path.relative_to(REPO_ROOT)}")
        return 0
    if not args.task:
        parser.error("--task is required (or --write-question-sets)")
    if args.task == "b2" and (args.cnn_raw is None or args.tau is None):
        parser.error("b2 needs --cnn-raw and --tau")
    if (args.accept_at is None) != (args.review_below is None):
        parser.error("--accept-at and --review-below go together")
    if args.dir:
        pages = pages_from_dir(args.dir)
    elif args.folds_csv and args.root:
        pages = pages_from_folds(args.folds_csv, args.fold, args.part, args.root)
    else:
        parser.error("give --dir, or --folds_csv with --root")
    if args.per_label:
        pages = sample_per_label(pages, args.per_label, args.seed)
    if not pages:
        parser.error("no pages found")
    max_sides = args.max_side or [1024]

    client = dc.DecisionClient(
        args.url,
        args.model,
        timeout_s=args.timeout,
        retries=args.retries,
        image_style=args.image_style,
        use_env_proxy=not args.no_proxy,
    )
    if not client.endpoint or not client.model:
        parser.error("set --url/--model (or DECISION_MODEL_URL/DECISION_MODEL_NAME)")
    if args.resume and not args.out:
        parser.error("--resume needs --out")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    out = args.out or REPO_ROOT / "result" / "decide" / f"{stamp}_{args.task}_{_slug(args.model)}_DECIDE_RAW.csv"
    cnn_raw = read_cnn_raw(args.cnn_raw) if args.task == "b2" else None
    digest = dc.schema_digest(route_questions() if args.task == "b3" else page_questions())

    from atrium_paradata import ParadataLogger  # noqa: PLC0415

    paradata = ParadataLogger(
        program="page-classification",
        config={
            "mode": f"decide_{args.task}",
            "decision_model": args.model,
            "decision_revision": args.model_revision,
            "decision_schema_digest": digest,
            "image_style": args.image_style,
            "max_side": max_sides,
            "tau": args.tau,
            "accept_at": args.accept_at,
            "review_below": args.review_below,
            "pages": len(pages),
            "per_label": args.per_label,
            "seed": args.seed,
        },
        paradata_dir=str(out.parent / "paradata"),
        output_types=["csv"],
        config_dir=str(REPO_ROOT / "setup"),
    )
    paradata.log_component("decision_model")
    rows = run(
        client,
        args.task,
        pages,
        max_sides,
        out,
        args.resume,
        cnn_raw,
        args.tau,
        args.accept_at,
        args.review_below,
        args.limit,
    )
    paradata.log_success("csv", len(rows))
    all_rows = read_rows(out)
    summary = summarise(args.task, all_rows)
    summary.update(
        {"model": args.model, "model_revision": args.model_revision, "schema_digest": digest, "table": _relative(out)}
    )
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paradata.finalize(
        input_total=len(pages) * len(max_sides), processed_total=sum(r["status"] == "ok" for r in all_rows)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
