"""
atrium_page_selection.py  –  Which pages page-classification classifies (atrium-project#84 decision 4).

ARÚP's rule (#84, comment 7): page-classification classifies the pages forced to the ATR and every
page that shows a picture, a chart or a drawing, of a PDF or of an office document. What a page shows
is layout, and docproc describes it with blocks (decision 6): a picture, chart or drawing is a
`figure` block (the ATR may also type one `image` or `chart`). So the selection reads the docproc, not
a separate list:

* **forced pages**, in either producer's shape: digital-convert's docproc marks them `needsAtr` on the
  page and route `atr` in `metadata.routing.pages[]`; the ATR's echoes them as `forced: true` (ask 5,
  D27). The ATR routes every page `atr`, so on its docproc only the echo counts;
* **pages with a figure block** (`FIGURE_TYPES`).

The caller selects (the pipeline's route step, the hub's E2E lanes) and passes the pages and the rule's
key to page-classification, which records both in its paradata `parameters` with the rule's statement
(`RULES`), so a page without a category reads as "not selected", not as a gap. In the pipeline
page-classification runs beside the ATR, so the route step applies `select_pages()` to digital-convert's
docproc.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

#: Block types that show a picture, a chart or a drawing: digital-convert writes `figure`; the ATR's
#: schema also has `image` and `chart`.
FIGURE_TYPES = frozenset({"figure", "image", "chart"})

#: The ATR's producer id (`pipeline`), with or without a variant; digital-convert's docproc has none.
ATR_PIPELINE = "pipeline-e2e"

#: The selection rules a caller may name, with the statement page-classification records.
FORCED_AND_FIGURES = "forced+figures"
EXPLICIT = "explicit"
ALL = "all"
RULES: Dict[str, str] = {
    FORCED_AND_FIGURES: (
        "the pages forced to the ATR and the pages with figure blocks (pictures, charts, drawings) in the "
        "docproc; a page without a category was not selected (atrium-project#84 decision 4)"
    ),
    EXPLICIT: "the pages the caller listed; a page without a category was not selected",
    ALL: "every page of the document",
}


def _routing(docproc: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = docproc.get("metadata") if isinstance(docproc, dict) else None
    routing = (metadata or {}).get("routing") or {}
    return [row for row in routing.get("pages") or [] if isinstance(row, dict)]


def _pages(docproc: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = docproc.get("pages") if isinstance(docproc, dict) else None
    return [page for page in pages or [] if isinstance(page, dict)]


def _number(value: Any) -> int:
    return int(value)


def is_atr_docproc(docproc: Dict[str, Any]) -> bool:
    """Whether the docproc is the ATR's (its `pipeline` id), not digital-convert's."""
    return str(docproc.get("pipeline") or "").startswith(ATR_PIPELINE)


def forced_pages(docproc: Dict[str, Any]) -> List[int]:
    """The pages forced to the ATR, from either producer's docproc (see the module docstring)."""
    routing = _routing(docproc)
    if is_atr_docproc(docproc) or any("forced" in row for row in routing):
        return sorted({_number(row["page"]) for row in routing if row.get("forced") is True})
    pages = {_number(page["pageNumber"]) for page in _pages(docproc) if page.get("needsAtr") is True}
    pages |= {_number(row["page"]) for row in routing if row.get("route") == "atr"}
    return sorted(pages)


def figure_pages(docproc: Dict[str, Any]) -> List[int]:
    """The pages holding a picture, chart or drawing block (`FIGURE_TYPES`)."""
    return sorted(
        {
            _number(page["pageNumber"])
            for page in _pages(docproc)
            if any(isinstance(b, dict) and b.get("type") in FIGURE_TYPES for b in page.get("blocks") or [])
        }
    )


def select_pages(docproc: Dict[str, Any]) -> List[int]:
    """The pages rule `forced+figures` selects: forced ∪ pages with figure blocks, sorted."""
    return sorted(set(forced_pages(docproc)) | set(figure_pages(docproc)))


def select_pages_of(docprocs: Iterable[Dict[str, Any]]) -> List[int]:
    """`select_pages()` over several docprocs of one document (the stored one and digital-convert's),
    united: two producers may find figures on different pages."""
    pages: set = set()
    for docproc in docprocs:
        pages |= set(select_pages(docproc))
    return sorted(pages)


def rule_statement(rule: str) -> str:
    """The statement recorded for a rule; ValueError for a rule not in `RULES`."""
    if rule not in RULES:
        raise ValueError(f"page selection {rule!r} is not one of {sorted(RULES)}")
    return RULES[rule]


def as_page_list(pages: Iterable[int]) -> str:
    """The pages as page-classification's `pages` field takes them (`1,3,5`)."""
    return ",".join(str(int(p)) for p in sorted(set(pages)))


__all__ = [
    "ALL",
    "EXPLICIT",
    "FIGURE_TYPES",
    "FORCED_AND_FIGURES",
    "RULES",
    "as_page_list",
    "figure_pages",
    "forced_pages",
    "is_atr_docproc",
    "rule_statement",
    "select_pages",
    "select_pages_of",
]
