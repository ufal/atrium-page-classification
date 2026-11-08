#!/usr/bin/env python3
"""decide_pages.py — decision models on page images (issue #58, track 1; ufal/atrium-project#83).

An experiment CLI, outside the service and its image. It sends page images to a decision model
served at ``POST /v1/systemone`` (``decision_client.py``) and writes one ``DECIDE_RAW`` table per run.
The model runs on the server, so this needs no torch: it runs in the repository's own environment.
The input is the page image only; no file name, label or text goes with it.

The tasks (the #58 plan):

* **b1** — the 11 labels, standalone: one ``choice`` over ``model_registry.CATEGORIES``, each label
  described by its ``atrium_vocab`` definition. The control.
* **b2** — the cascade: the CNN answers every page (a ``run.py --raw`` table); a page whose raw top-1
  probability is below ``--tau`` goes to the decision model, which chooses among the CNN's top 3
  only. ``CLASS-1`` is the cascade's answer for every page, so the table scores as a whole.
* **b3** — content and processors, for triage. Seven yes/no *elements* (printed text, handwriting,
  table or form, filled-in fields, photograph, drawing, stamp): a page can carry all of them at once.
  The four *processor tags* (``ROUTES``: OCR, HTR, table/form extraction, graphics extraction) are
  computed from them in code, P(tag) = the largest P of its elements (``ROUTE_ELEMENTS``). One
  ``choice``, ``primary_route``, names the processor that matters most. Scored against what the
  true label implies: elements three-state from ``atrium_vocab.COLLECTIONS`` (``element_gold``),
  tags and the primary processor from the hub's "what each category suggests doing next".
* **b4** — the page's condition. Fifteen yes/no *defects* in three aspects (``DEFECTS``: the paper,
  the content, the capture), the ``orientation`` of the main text, and three ``score`` questions
  from 0 (none) to 4 (severe): ``paper_condition``, ``content_condition`` and ``overall_damage``.
  ``damage_score`` is ``overall_damage`` / 4. The per-aspect means of the defect probabilities
  (``paper_index``, ``content_index``, ``capture_index``, ``damage_index``) are computed in code as a
  check on the model's own scores, never in their place.
* **triage** — b3 and b4 in one request: one prefill of the image for 27 questions.

The page condition has no gold in the dataset. ``data_scripts/degrade_pages.py`` makes controlled
copies of clean pages (rotation, skew, yellowing, faint ink, blur, a cut-off edge, uneven light) and
a manifest; ``--degradations <manifest>`` reads the pages from it, and the summary reports, per
injected defect, the paired change of its probability against the original page.

The question sets are generated, never hand-edited: ``--write-question-sets`` writes
``decisions/page_category.v1.json``, ``decisions/page_routes.v2.json`` and
``decisions/page_condition.v1.json`` (``tests/test_decide_pages.py`` fails when they drift). Typical
runs, from the repository root::

    python3 decide_pages.py --write-question-sets
    # Phase A smoke: 3 public sample pages per label, three image sizes
    python3 decide_pages.py --task b1 --dir small_data_samples --per-label 3 \\
        --max-side 768 --max-side 1024 --max-side 1536
    # B1 on the fold-1 dev partition, staged with data_scripts/stage_fold_pages.py
    python3 decide_pages.py --task b1 --folds_csv licensed_crossval_folds.csv --fold fold1 --part dev \\
        --root stage/fold1_dev
    # B2 behind a CNN run of the same pages
    python3 decide_pages.py --task b2 --dir stage/fold1_dev --cnn-raw result/tables/<stamp>_EVAL_RAW.csv --tau 0.9
    # Triage (B3 + B4) on the public samples, and on degraded copies of them
    python3 decide_pages.py --task triage --dir small_data_samples --per-label 3
    python3 data_scripts/degrade_pages.py --dir small_data_samples --per-label 2 --out stage/degraded
    python3 decide_pages.py --task b4 --degradations stage/degraded/degradations.csv

``--url``, ``--model`` and ``--timeout`` default to ``DECISION_MODEL_URL``, ``DECISION_MODEL_NAME`` and
``DECISION_TIMEOUT_S``. ``--resume --out <table>`` continues a stopped run. Which archive data may be
sent to which server is settled on ufal/atrium-project#83; the public ``small_data_samples`` may go
anywhere.
"""

from __future__ import annotations

import argparse
import csv
import functools
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
ROUTE_SET_FILE = QUESTION_DIR / "page_routes.v2.json"
CONDITION_SET_FILE = QUESTION_DIR / "page_condition.v1.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
TASKS = ("b1", "b2", "b3", "b4", "triage")

#: What the state says about every page. No file name or label is sent with it.
PAGE_STATE = {
    "task": "Decide about the attached image of one scanned page.",
    "collection": "archaeological reports and related material from Czech archives, 1920–2020",
}

PAGE_INSTRUCTIONS = "Which category best describes the attached page image?"

# ── B3: content and processors ────────────────────────────────────────────────────────────────

#: What can be on a page. A page may carry any number of them.
ELEMENTS: Dict[str, str] = {
    "has_printed_text": "Does the page carry printed or machine-typed text (body text, headings, captions or labels)?",
    "has_handwriting": "Does the page carry handwritten text or numbers (notes, labels or filled-in entries)?",
    "has_table_form": "Is any part of the page a table, a form, a ledger grid or a tabular legend?",
    "has_filled_fields": "Does the page have form fields, blanks or stamp boxes filled in by hand or by typewriter?",
    "has_photo": "Does the page show a photograph or a photographic cutout?",
    "has_drawing": "Does the page show a drawing, map, plan, section or schematic?",
    "has_stamp": "Does the page carry a stamp or seal impression?",
}

#: The four processors a page can be sent on to, and what each one is for.
ROUTES: Dict[str, str] = {
    "ocr": "printed or machine-typed text recognition: the page is mainly printed or typed text",
    "htr": "handwritten text recognition: the page is mainly handwritten text",
    "tables": "table or form extraction: the page is mainly a table, a form, a ledger or a filled-in stamp",
    "graphics": "graphics detection and extraction: the page is mainly photographs, drawings, maps or plans",
}
PRIMARY_INSTRUCTIONS = "Which further processor matters most for this page?"

#: A processor tag is on when any of its elements is present (P = the largest of their P).
#: A stamp is routed with tables and forms: an archive stamp is a small form with filled-in fields.
ROUTE_ELEMENTS: Dict[str, Tuple[str, ...]] = {
    "ocr": ("has_printed_text",),
    "htr": ("has_handwriting",),
    "tables": ("has_table_form", "has_filled_fields", "has_stamp"),
    "graphics": ("has_photo", "has_drawing"),
}

#: The tags each true label implies: the hub's "What each category suggests doing next"
#: (docs_site/workflows/page-classification.md). Captions on DRAW/PHOTO pages ("through OCR, if
#: any") are not in it: the label does not say whether there is one.
ROUTE_GOLD: Dict[str, Tuple[str, ...]] = {
    "ocr": ("TEXT", "TEXT_P", "TEXT_T", "LINE_P", "LINE_T"),
    "htr": ("TEXT", "TEXT_HW", "LINE_HW"),
    "tables": ("LINE_HW", "LINE_P", "LINE_T", "DRAW_L", "PHOTO_L"),
    "graphics": ("DRAW", "DRAW_L", "PHOTO", "PHOTO_L"),
}

#: The processor that matters most, per true label. Where the next-step table names two equally
#: (TEXT: OCR and HTR; DRAW_L/PHOTO_L: graphics and tables), either answer counts.
PRIMARY_GOLD: Dict[str, Tuple[str, ...]] = {
    "TEXT_P": ("ocr",),
    "TEXT_T": ("ocr",),
    "TEXT_HW": ("htr",),
    "TEXT": ("ocr", "htr"),
    "LINE_P": ("tables",),
    "LINE_T": ("tables",),
    "LINE_HW": ("tables",),
    "DRAW": ("graphics",),
    "PHOTO": ("graphics",),
    "DRAW_L": ("graphics", "tables"),
    "PHOTO_L": ("graphics", "tables"),
}

# ── B4: condition ─────────────────────────────────────────────────────────────────────────────

#: The defects, by aspect: the paper itself, the content written or printed on it, and the capture.
DEFECTS: Dict[str, Dict[str, str]] = {
    "paper": {
        "stained": "Is the paper stained or soiled (spots, smears, rust or ink marks that are not part of the content)?",
        "yellowed": "Is the paper yellowed, browned or discoloured with age?",
        "torn_edges": "Are the paper's edges torn, frayed or ragged?",
        "water_damage": "Does the paper show water damage (tide lines, rippling or washed-out patches)?",
        "folds_creases": "Does the paper show folds, creases or wrinkles?",
        "holes_missing": "Are pieces of the paper missing (holes, lost corners or torn-away parts)?",
        "foxing_mould": "Does the paper show foxing or mould (small brown spots or fuzzy patches)?",
    },
    "content": {
        "faint_ink": "Is the ink or print faint or faded, hard to make out against the paper?",
        "bleed_through": "Does writing or print from the other side of the sheet show through?",
        "smudged_ink": "Is the ink smudged, blotted or smeared over the text?",
    },
    "capture": {
        "skewed": "Is the page or its text lines tilted by a few degrees in the image?",
        "rotated_text": "Does any text on the page run sideways or upside down?",
        "blurred": "Is the image blurred or out of focus?",
        "cropped": "Is part of the page or its content cut off at an edge of the image?",
        "uneven_light": "Is the lighting uneven (shadows, glare, a dark binding edge or a curved page)?",
    },
}
DEFECT_IDS: Tuple[str, ...] = tuple(d for group in DEFECTS.values() for d in group)

ORIENTATIONS: Dict[str, str] = {
    "upright": "text lines run left to right, the right way up",
    "cw90": "the page is turned a quarter clockwise: text lines run from top to bottom",
    "upside_down": "the page is upside down: text lines run right to left with letters inverted",
    "ccw90": "the page is turned a quarter counter-clockwise: text lines run from bottom to top",
    "unclear": "there is no text, or which way up it is cannot be told",
}
ORIENTATION_INSTRUCTIONS = "Which way up is the page's main text in the image?"

#: The three condition scores, 0 (none) to 4 (severe): instructions and level descriptions.
SCORES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "paper_condition": (
        "How damaged is the paper itself?",
        (
            "intact: clean, whole and flat paper",
            "slight: minor discolouration, small spots or creases",
            "moderate: clear staining, yellowing, folds or frayed edges",
            "heavy: large stains, tears, water damage or missing pieces",
            "severe: the sheet is broken, lost or ruined in large part",
        ),
    ),
    "content_condition": (
        "How damaged is the content (the text and the images) on the page?",
        (
            "intact: everything is crisp and legible",
            "slight: a few faint or smudged spots, everything still legible",
            "moderate: parts are faint, smudged or show-through, readable with effort",
            "heavy: much of the content is hard to read or partly lost",
            "severe: most of the content is illegible or lost",
        ),
    ),
    "overall_damage": (
        "Overall, how damaged is this page as an input for text and image extraction "
        "(the paper, the content and the scan together)?",
        (
            "none: a clean page and a clean scan",
            "slight: small flaws that will not hinder extraction",
            "moderate: flaws that will cause some extraction errors",
            "heavy: flaws that will cause many extraction errors",
            "severe: extraction is unlikely to work; restore or re-scan first",
        ),
    ),
}
SCORE_MAX = 4

#: The defects data_scripts/degrade_pages.py can inject, i.e. the ones with synthetic gold.
SYNTHETIC_DEFECTS = ("rotated_text", "skewed", "yellowed", "faint_ink", "blurred", "cropped", "uneven_light")

#: Labels whose orientation can be judged from text (DRAW/PHOTO pages may carry none).
TEXT_LABELS = ("LINE_HW", "LINE_P", "LINE_T", "TEXT", "TEXT_HW", "TEXT_P", "TEXT_T")


# ── question sets ─────────────────────────────────────────────────────────────────────────────


def category_definitions() -> Dict[str, str]:
    """The 11 labels in ``CATEGORIES`` order, each with its ``atrium_vocab`` definition."""
    import atrium_vocab  # noqa: PLC0415 — the shared vocabulary module, vendored from the hub

    concepts = atrium_vocab.CONCEPTS["page-category"]
    missing = [label for label in CATEGORIES if label not in concepts]
    if missing:
        raise SystemExit(f"atrium_vocab has no definition for {missing}; re-vendor the shared files")
    return {label: concepts[label]["definition"]["en"] for label in CATEGORIES}


@functools.lru_cache(maxsize=1)
def element_gold() -> Dict[str, Dict[str, List[str]]]:
    """Per element, the labels that imply it ``present`` and those that rule it out (``absent``).

    Present: the hub vocabulary's page-category collections (printed ∪ typed, handwritten, tabular;
    the PHOTO and DRAW members of graphical). Absent: only the labels whose definition says "only …"
    (TEXT_HW, TEXT_P, TEXT_T: one kind of text, non-tabular). Every other label leaves the element
    unknown, and filled-in fields and stamps are unknown on every label: a label names the page's
    dominant content, not all of it."""
    import atrium_vocab  # noqa: PLC0415

    collections = atrium_vocab.COLLECTIONS["page-category"]
    definitions = category_definitions()
    only = {label for label, text in definitions.items() if text.lower().startswith("only ")}
    graphical = collections["graphical"]["members"]
    present: Dict[str, set] = {
        "has_printed_text": set(collections["printed"]["members"]) | set(collections["typed"]["members"]),
        "has_handwriting": set(collections["handwritten"]["members"]),
        "has_table_form": set(collections["tabular"]["members"]),
        "has_filled_fields": set(),
        "has_photo": {m for m in graphical if m.startswith("PHOTO")},
        "has_drawing": {m for m in graphical if m.startswith("DRAW")},
        "has_stamp": set(),
    }
    out: Dict[str, Dict[str, List[str]]] = {}
    for element in ELEMENTS:
        positives = present[element]
        absent = sorted(only - positives) if positives else []
        out[element] = {"present": sorted(positives), "absent": absent}
    return out


def gold_of(element_table: Mapping[str, Mapping[str, Sequence[str]]], element: str, label: str) -> str:
    """``"1"``, ``"0"`` or ``""`` (unknown) for *element* on a page whose true label is *label*."""
    entry = element_table[element]
    if label in entry["present"]:
        return "1"
    if label in entry["absent"]:
        return "0"
    return ""


def page_questions(labels: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    """The ``page_category`` choice, over all 11 labels or over *labels* (B2's top 3)."""
    definitions = category_definitions()
    offered = list(labels) if labels else list(CATEGORIES)
    return dc.questions({"page_category": dc.choice(PAGE_INSTRUCTIONS, {k: definitions[k] for k in offered})})


def content_questions() -> Dict[str, Dict[str, Any]]:
    """B3: the seven elements and the primary processor."""
    items: Dict[str, Dict[str, Any]] = {element: dc.noul(text) for element, text in ELEMENTS.items()}
    items["primary_route"] = dc.choice(PRIMARY_INSTRUCTIONS, ROUTES)
    return dc.questions(items)


def condition_questions() -> Dict[str, Dict[str, Any]]:
    """B4: the fifteen defects, the orientation and the three condition scores."""
    items: Dict[str, Dict[str, Any]] = {
        defect: dc.noul(text) for group in DEFECTS.values() for defect, text in group.items()
    }
    items["orientation"] = dc.choice(ORIENTATION_INSTRUCTIONS, ORIENTATIONS)
    for qid, (instructions, levels) in SCORES.items():
        items[qid] = dc.score(instructions, levels)
    return dc.questions(items)


def task_questions(task: str) -> Dict[str, Dict[str, Any]]:
    """The question set one request of *task* carries (b2 narrows the choice per page)."""
    if task == "b3":
        return content_questions()
    if task == "b4":
        return condition_questions()
    if task == "triage":
        return dc.questions({**content_questions(), **condition_questions()})
    return page_questions()


def question_sets() -> Dict[Path, Dict[str, Any]]:
    """The committed question-set files, as they should read."""
    common = {"generated_by": "decide_pages.py --write-question-sets", "state": PAGE_STATE}
    page = page_questions()
    content = content_questions()
    condition = condition_questions()
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
            "version": 2,
            **common,
            "source": (
                "#58 plan, B3 (v2, 2026-11-08): elements asked, processor tags computed from them; gold: elements "
                "from atrium_vocab.COLLECTIONS['page-category'], tags and primary from the hub's "
                "'What each category suggests doing next'"
            ),
            "questions": content,
            "route_elements": {route: list(elements) for route, elements in ROUTE_ELEMENTS.items()},
            "gold": {
                "elements": element_gold(),
                "routes": {route: list(labels) for route, labels in ROUTE_GOLD.items()},
                "primary_route": {label: list(routes) for label, routes in PRIMARY_GOLD.items()},
            },
            "schema_digest": dc.schema_digest(content),
        },
        CONDITION_SET_FILE: {
            "name": "page_condition",
            "version": 1,
            **common,
            "source": "#58 plan, B4 (2026-11-08): defects by aspect, orientation, three condition scores",
            "questions": condition,
            "aspects": {aspect: list(group) for aspect, group in DEFECTS.items()},
            "score_range": [0, SCORE_MAX],
            "gold": {
                "synthetic": list(SYNTHETIC_DEFECTS) + ["orientation"],
                "synthetic_from": "data_scripts/degrade_pages.py",
                "annotation_only": [d for d in DEFECT_IDS if d not in SYNTHETIC_DEFECTS] + list(SCORES),
            },
            "schema_digest": dc.schema_digest(condition),
        },
    }


def render_question_set(content: Mapping[str, Any]) -> str:
    """One question set as its file's text, in one fixed form, so the drift test can compare the
    committed file with the generator byte for byte."""
    return json.dumps(content, indent=2, ensure_ascii=False) + "\n"


def write_question_sets() -> List[Path]:
    """Regenerate every file of ``question_sets()`` under ``decisions/`` (``--write-question-sets``);
    returns the paths written."""
    QUESTION_DIR.mkdir(exist_ok=True)
    written = []
    for path, content in question_sets().items():
        path.write_text(render_question_set(content), encoding="utf-8")
        written.append(path)
    return written


# ── inputs ────────────────────────────────────────────────────────────────────────────────────

#: One page to decide: the image and its true label.
Page = Tuple[Path, str]


def pages_from_dir(root: Path) -> List[Page]:
    """``(image, true label)`` for every image in a ``<root>/<CATEGORY>/`` folder named after a label."""
    pages = []
    for label in CATEGORIES:
        folder = root / label
        if folder.is_dir():
            pages += [(p, label) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    return pages


def pages_from_folds(folds_csv: Path, fold: str, part: str, root: Path) -> List[Page]:
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


MANIFEST_COLUMNS = ("image", "source", "label", "defect", "level", "orientation", "params")


def read_degradations(manifest: Path) -> Tuple[List[Page], Dict[Path, Dict[str, str]]]:
    """The pages of a ``degrade_pages.py`` manifest and, per image, its degradation.

    ``image`` and ``source`` are relative to the manifest's folder (or absolute). The page key
    (``FILE``, ``PAGE``) is the source's, so a degraded copy pairs with its original."""
    base = manifest.parent
    pages: List[Page] = []
    variants: Dict[Path, Dict[str, str]] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        absent = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if absent:
            raise SystemExit(f"{manifest}: not a degradation manifest (no column {sorted(absent)})")
        for row in reader:
            image = Path(row["image"]) if Path(row["image"]).is_absolute() else base / row["image"]
            source = Path(row["source"]) if Path(row["source"]).is_absolute() else base / row["source"]
            if row["label"] not in CATEGORIES:
                raise SystemExit(f"{manifest}: unknown label {row['label']!r}")
            pages.append((image, row["label"]))
            variants[image] = {
                "VARIANT": row["defect"] or "original",
                "LEVEL": row["level"],
                "SOURCE": _relative(source),
                "GOLD_orientation": row["orientation"] or "upright",
                "_source_path": str(source),
            }
    return pages, variants


def sample_per_label(pages: Sequence[Page], n: int, seed: int) -> List[Page]:
    """At most *n* pages per label, drawn with *seed* from the name-sorted list (reproducible)."""
    rng = random.Random(seed)
    by_label: Dict[str, List[Page]] = {}
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
    """The *n* most probable labels, best first; a tie goes to the earlier label of ``CATEGORIES``,
    so B2 offers the same options on every run."""
    return sorted(probabilities, key=lambda label: (-probabilities[label], CATEGORIES.index(label)))[:n]


# ── the run ───────────────────────────────────────────────────────────────────────────────────

BASE_COLUMNS = ["FILE", "PAGE", "TRUE"]
VARIANT_COLUMNS = ["VARIANT", "LEVEL", "SOURCE"]
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
INDEX_COLUMNS = ["paper_index", "content_index", "capture_index", "damage_index"]


def content_columns() -> List[str]:
    """The B3 columns of a row: each element and each computed tag with its gold, then the primary
    processor, its probabilities, its gold and its lane."""
    elements = [col for element in ELEMENTS for col in (f"P_{element}", f"GOLD_{element}")]
    routes = [col for route in ROUTES for col in (f"P_route_{route}", f"GOLD_route_{route}")]
    primary = ["primary_route"] + [f"P_primary_{route}" for route in ROUTES] + ["GOLD_primary", "lane"]
    return elements + routes + primary


def condition_columns() -> List[str]:
    """The B4 columns of a row: the defects, the orientation with its gold, the three scores and
    ``damage_score``, then the check indices."""
    defects = [f"P_{defect}" for defect in DEFECT_IDS]
    orientation = ["orientation"] + [f"P_orient_{o}" for o in ORIENTATIONS] + ["GOLD_orientation"]
    scores = [f"E_{qid}" for qid in SCORES] + ["damage_score"]
    return defects + orientation + scores + INDEX_COLUMNS


def columns_for(task: str) -> List[str]:
    """The header of a *task*'s ``DECIDE_RAW`` table. One fixed list per task, so a resumed run
    appends rows under the header it started with."""
    if task in ("b3", "b4", "triage"):
        middle = (content_columns() if task in ("b3", "triage") else []) + (
            condition_columns() if task in ("b4", "triage") else []
        )
        return BASE_COLUMNS + VARIANT_COLUMNS + middle + TAIL_COLUMNS
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


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(statistics.fmean(values), 4) if values else None


def damage_indices(p_defects: Mapping[str, float]) -> Dict[str, Optional[float]]:
    """The equal-weight mean of the defect probabilities per aspect, and over all of them.

    A check on the model's own condition scores, computed in code; the weights are not tuned."""
    out: Dict[str, Optional[float]] = {}
    for aspect, group in DEFECTS.items():
        out[f"{aspect}_index"] = _mean([p_defects[d] for d in group if d in p_defects])
    out["damage_index"] = _mean([p_defects[d] for d in DEFECT_IDS if d in p_defects])
    return out


def content_fields(
    d: dc.Decision, true_label: str, accept_at: Optional[float], review_below: Optional[float]
) -> Dict[str, Any]:
    """The B3 part of a row: elements, the computed tags, the primary processor, and their gold."""
    table = element_gold()
    row: Dict[str, Any] = {}
    p_elements: Dict[str, float] = {}
    for element in ELEMENTS:
        answer = d.answers.get(element)
        if answer is not None and answer.p_true is not None:
            p_elements[element] = answer.p_true
        row[f"P_{element}"] = p_elements.get(element, "")
        row[f"GOLD_{element}"] = gold_of(table, element, true_label)
    for route, elements in ROUTE_ELEMENTS.items():
        known = [p_elements[e] for e in elements if e in p_elements]
        row[f"P_route_{route}"] = round(max(known), 4) if len(known) == len(elements) else ""
        row[f"GOLD_route_{route}"] = int(true_label in ROUTE_GOLD[route])
    primary = d.answers.get("primary_route")
    row["primary_route"] = primary.choice if primary else ""
    for route in ROUTES:
        row[f"P_primary_{route}"] = primary.probabilities.get(route, "") if primary else ""
    row["GOLD_primary"] = "|".join(PRIMARY_GOLD.get(true_label, ()))
    row["lane"] = _lane(primary.confidence if primary else None, accept_at, review_below)
    return row


def condition_fields(d: dc.Decision) -> Dict[str, Any]:
    """The B4 part of a row: defects, orientation, the three scores and the check indices."""
    row: Dict[str, Any] = {}
    p_defects: Dict[str, float] = {}
    for defect in DEFECT_IDS:
        answer = d.answers.get(defect)
        if answer is not None and answer.p_true is not None:
            p_defects[defect] = answer.p_true
        row[f"P_{defect}"] = p_defects.get(defect, "")
    orientation = d.answers.get("orientation")
    row["orientation"] = orientation.choice if orientation else ""
    for option in ORIENTATIONS:
        row[f"P_orient_{option}"] = orientation.probabilities.get(option, "") if orientation else ""
    for qid in SCORES:
        answer = d.answers.get(qid)
        row[f"E_{qid}"] = answer.score if answer is not None and answer.score is not None else ""
    overall = row["E_overall_damage"]
    row["damage_score"] = round(float(overall) / SCORE_MAX, 4) if overall != "" else ""
    if len(p_defects) == len(DEFECT_IDS):
        row.update({k: ("" if v is None else v) for k, v in damage_indices(p_defects).items()})
    else:
        row.update(dict.fromkeys(INDEX_COLUMNS, ""))
    return row


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
    variant: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """One output row. Never raises for a server or input problem: the row carries the status."""
    file_id, page = page_key(Path(variant["_source_path"]) if variant else path)
    row: Dict[str, Any] = {"FILE": file_id, "PAGE": page, "TRUE": true_label}

    if task in ("b3", "b4", "triage"):
        row.update({k: v for k, v in (variant or {}).items() if not k.startswith("_")})
        d = client.decide(PAGE_STATE, task_questions(task), images=[path], max_side=max_side)
        if task in ("b3", "triage"):
            row.update(content_fields(d, true_label, accept_at, review_below))
        if task in ("b4", "triage"):
            row.update(condition_fields(d))
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
    pages: Sequence[Page],
    max_sides: Sequence[int],
    out: Path,
    resume: bool = False,
    cnn_raw: Optional[Mapping[Tuple[str, str], Mapping[str, float]]] = None,
    tau: Optional[float] = None,
    accept_at: Optional[float] = None,
    review_below: Optional[float] = None,
    limit: Optional[int] = None,
    progress: bool = True,
    variants: Optional[Mapping[Path, Mapping[str, str]]] = None,
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
            variant = variants.get(path) if variants else None
            row = decide_page(client, task, path, label, side, image_label, cnn, tau, accept_at, review_below, variant)
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if progress:
                shown = row.get("CLASS-1", row.get("primary_route", row.get("damage_score", "")))
                print(f"[{index}/{len(jobs)}] {image_label} @{side}: {row['status']} {shown}", flush=True)
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


def _num(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman's rank correlation (average ranks for ties); None below 3 pairs or without spread."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    if not sx or not sy:
        return None
    return round(sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy), 4)


def binary_scores(pairs: Sequence[Tuple[float, str]], threshold: float = 0.5) -> Dict[str, Any]:
    """Precision, recall and F1 of ``P ≥ threshold`` over the pairs whose gold is known (1 or 0)."""
    known = [(p, g == "1") for p, g in pairs if g in ("0", "1")]
    tp = sum(p >= threshold and g for p, g in known)
    fp = sum(p >= threshold and not g for p, g in known)
    fn = sum(p < threshold and g for p, g in known)
    has_negatives = any(not g for _, g in known)  # without known negatives, precision means nothing
    precision = tp / (tp + fp) if has_negatives and (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * tp / (2 * tp + fp + fn) if has_negatives and (2 * tp + fp + fn) else None
    return {
        "known": len(known),
        "positives": sum(g for _, g in known),
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def summarise_content(ok: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """B3 figures: per element and tag against the label-implied gold, and the primary processor."""
    elements = {
        e: binary_scores([(float(r[f"P_{e}"]), str(r[f"GOLD_{e}"])) for r in ok if _num(r.get(f"P_{e}")) is not None])
        for e in ELEMENTS
    }
    for e in ELEMENTS:
        values = [float(r[f"P_{e}"]) for r in ok if _num(r.get(f"P_{e}")) is not None]
        elements[e]["rate"] = round(sum(v >= 0.5 for v in values) / len(values), 4) if values else None
    routes = {
        route: binary_scores(
            [
                (float(r[f"P_route_{route}"]), str(r[f"GOLD_route_{route}"]))
                for r in ok
                if _num(r.get(f"P_route_{route}")) is not None
            ]
        )
        for route in ROUTES
    }
    full = [r for r in ok if all(_num(r.get(f"P_route_{route}")) is not None for route in ROUTES)]
    exact = sum(
        all((float(r[f"P_route_{route}"]) >= 0.5) == (str(r[f"GOLD_route_{route}"]) == "1") for route in ROUTES)
        for r in full
    )
    answered = [r for r in ok if r.get("primary_route")]
    primary_hits = sum(r["primary_route"] in str(r.get("GOLD_primary", "")).split("|") for r in answered)
    return {
        "elements": elements,
        "routes": routes,
        "exact_route_set": round(exact / len(full), 4) if full else None,
        "primary_accuracy": round(primary_hits / len(answered), 4) if answered else None,
        "primary_counts": dict(Counter(r["primary_route"] for r in answered)),
    }


def summarise_condition(ok: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """B4 figures: prevalence, the mean scores, the indices against the scores, and with a
    degradation manifest the paired change per injected defect and the orientation accuracy."""
    out: Dict[str, Any] = {"defects": {}, "scores": {}, "index_vs_score": {}}
    for defect in DEFECT_IDS:
        values = [float(r[f"P_{defect}"]) for r in ok if _num(r.get(f"P_{defect}")) is not None]
        out["defects"][defect] = {
            "mean_p": _mean(values),
            "rate": round(sum(v >= 0.5 for v in values) / len(values), 4) if values else None,
        }
    for qid in SCORES:
        values = [float(r[f"E_{qid}"]) for r in ok if _num(r.get(f"E_{qid}")) is not None]
        out["scores"][qid] = {"mean": _mean(values), "p95": _pct(values, 0.95)}
    for index, qid in (
        ("paper_index", "paper_condition"),
        ("content_index", "content_condition"),
        ("damage_index", "overall_damage"),
    ):
        pairs = [
            (float(r[index]), float(r[f"E_{qid}"]))
            for r in ok
            if _num(r.get(index)) is not None and _num(r.get(f"E_{qid}")) is not None
        ]
        out["index_vs_score"][f"{index}~{qid}"] = spearman([a for a, _ in pairs], [b for _, b in pairs])

    degraded = [r for r in ok if r.get("VARIANT") and r["VARIANT"] != "original"]
    if not degraded:
        return out
    originals = {(r.get("SOURCE"), str(r["max_side"])): r for r in ok if r.get("VARIANT") == "original"}
    injected: Dict[str, Any] = {}
    for defect in sorted({r["VARIANT"] for r in degraded}):
        rows = [r for r in degraded if r["VARIANT"] == defect]
        column = f"P_{defect}"
        deltas, after, before = [], [], []
        for r in rows:
            base = originals.get((r.get("SOURCE"), str(r["max_side"])))
            p_after = _num(r.get(column))
            if p_after is None:
                continue
            after.append(p_after)
            if base is not None and _num(base.get(column)) is not None:
                before.append(float(base[column]))
                deltas.append(p_after - float(base[column]))
        injected[defect] = {
            "variants": len(rows),
            "paired": len(deltas),
            "mean_p_degraded": _mean(after),
            "mean_p_original": _mean(before),
            "mean_delta": _mean(deltas),
            "detected": round(sum(p >= 0.5 for p in after) / len(after), 4) if after else None,
            "false_alarm_on_originals": round(sum(p >= 0.5 for p in before) / len(before), 4) if before else None,
        }
    out["injected"] = injected
    levelled = [
        (float(r["LEVEL"]), float(r["E_overall_damage"]))
        for r in ok
        if r.get("VARIANT")
        and r["VARIANT"] != "rotated_text"
        and _num(r.get("LEVEL")) is not None
        and _num(r.get("E_overall_damage")) is not None
    ]
    out["level_vs_overall_damage"] = spearman([a for a, _ in levelled], [b for _, b in levelled])
    oriented = [r for r in ok if r.get("GOLD_orientation") and r.get("orientation") and r["TRUE"] in TEXT_LABELS]
    out["orientation_accuracy_text_pages"] = (
        round(sum(r["orientation"] == r["GOLD_orientation"] for r in oriented) / len(oriented), 4) if oriented else None
    )
    out["orientation_pages"] = len(oriented)
    return out


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
        ok = [r for r in part if r["status"] == "ok"]
        if task in ("b3", "triage"):
            entry["content"] = summarise_content(ok)
        if task in ("b4", "triage"):
            entry["condition"] = summarise_condition(ok)
        out["by_max_side"][side] = entry
    return out


def read_rows(path: Path) -> List[Dict[str, str]]:
    """Every row of a ``DECIDE_RAW`` table, as strings: the summary reads the whole table back, so
    rows from earlier runs of a ``--resume`` count too."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "model"


SERVED_ON = ("cluster-job", "ufal-endpoint", "local", "other")


def served_on(url: Optional[str], given: Optional[str] = None) -> str:
    """Where the requests went, for the paradata (one of ``SERVED_ON``), never the URL itself: the
    internal endpoint's host is not published. Without ``--served-on``, a loopback URL is ``local``."""
    if given:
        return given
    from urllib.parse import urlsplit  # noqa: PLC0415

    host = urlsplit(url or "").hostname or ""
    return "local" if host in ("127.0.0.1", "localhost", "::1") else "other"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def build_parser() -> argparse.ArgumentParser:
    """The command line; ``main()`` checks the combinations argparse cannot express."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write-question-sets", action="store_true", help="regenerate decisions/*.json and exit")
    parser.add_argument("--task", choices=TASKS)
    source = parser.add_argument_group("pages")
    source.add_argument("--dir", type=Path, help="a folder with one sub-folder per label")
    source.add_argument("--folds_csv", type=Path, help="a folds CSV (PNG, categ, foldN columns)")
    source.add_argument("--fold", default="fold1")
    source.add_argument("--part", default="dev", choices=("train", "dev", "test"))
    source.add_argument("--root", type=Path, help="where stage_fold_pages.py put the partition's pages")
    source.add_argument(
        "--degradations", type=Path, help="b3/b4/triage: the manifest data_scripts/degrade_pages.py wrote"
    )
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
    server.add_argument(
        "--served-on",
        choices=SERVED_ON,
        help="where the pages go, for the paradata (default: local for a loopback URL)",
    )
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
    """Check the arguments, run the task, write the table, its summary and the paradata; returns 0.
    Usage errors exit 2 before any request is sent."""
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
    if args.accept_at is not None:
        # Checked here, before a page is read or a request is sent. decision_client.lane() refuses
        # review_below > accept_at with a ValueError, but only on the first answer, after a paid
        # model call, and decide_page() promises never to raise. A threshold outside [0, 1] (or
        # NaN) is compared with a probability, so it would leave a lane empty without a word.
        for flag, value in (("--accept-at", args.accept_at), ("--review-below", args.review_below)):
            if not 0.0 <= value <= 1.0:
                parser.error(f"{flag} {value} is outside [0, 1]")
        if args.review_below > args.accept_at:
            parser.error(f"--review-below ({args.review_below}) must not exceed --accept-at ({args.accept_at})")
    variants = None
    if args.degradations:
        if args.task not in ("b3", "b4", "triage"):
            parser.error("--degradations goes with b3, b4 or triage")
        if args.per_label:
            parser.error("--per-label would split a degraded copy from its original; sample in degrade_pages.py")
        pages, variants = read_degradations(args.degradations)
    elif args.dir:
        pages = pages_from_dir(args.dir)
    elif args.folds_csv and args.root:
        pages = pages_from_folds(args.folds_csv, args.fold, args.part, args.root)
    else:
        parser.error("give --dir, --degradations, or --folds_csv with --root")
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
    digest = dc.schema_digest(task_questions(args.task))
    where = served_on(args.url, args.served_on)

    from atrium_paradata import ParadataLogger  # noqa: PLC0415

    paradata = ParadataLogger(
        program="page-classification",
        config={
            "mode": f"decide_{args.task}",
            "decision_model": args.model,
            "decision_revision": args.model_revision,
            "served_on": where,
            "decision_schema_digest": digest,
            "image_style": args.image_style,
            "max_side": max_sides,
            "tau": args.tau,
            "accept_at": args.accept_at,
            "review_below": args.review_below,
            "pages": len(pages),
            "per_label": args.per_label,
            "seed": args.seed,
            "degradations": _relative(args.degradations) if args.degradations else None,
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
        variants=variants,
    )
    paradata.log_success("csv", len(rows))
    all_rows = read_rows(out)
    summary = summarise(args.task, all_rows)
    summary.update(
        {
            "model": args.model,
            "model_revision": args.model_revision,
            "served_on": where,
            "schema_digest": digest,
            "table": _relative(out),
        }
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
