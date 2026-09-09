"""
atrium_vocab.py — the ATRIUM controlled-label registry, and its SKOS view.

WHY THIS MODULE EXISTS
======================
Five tools exchange one JSON record (``atrium_document.py``), and several of its
fields carry **controlled terms as bare strings**: ``pages[].category``,
``page_categories``, ``lines[].categ``, ``pages[].quality_band``,
``entities[].type_teitok`` / ``type_cnec``. Every one of those label sets is
declared in a different repository, in a different form — a Python list, a set of
``return`` statements, a JSON config, a dict literal — and nothing reconciles them.

That is not a stylistic complaint. It has already produced a live defect:
``lines[].categ`` has two authorised originators that emit **disjoint** label sets
(``alto-postprocess`` → ``Clear``/``Noisy``/``Trash``/``Non-text``/``Empty``;
``digital-convert`` → ``Garbage``/``Inverted``), and its only consumer filters on
one of the two. See ``LINE_CATEGORY_ORIGINATORS`` below and §Defect register in
``docs/skos_strategy.md``.

This module is the single declaration of those label sets, plus the ATRIUM-authored
concepts they correspond to, plus a deterministic SKOS serialisation of both.

WHAT THIS MODULE IS NOT
=======================
* **Not a publication pipeline.** Nothing here is published anywhere. Per
  ufal/atrium-project#51, the externally-published, DARIAH-namespaced SKOS version
  of the AMCR and TEATER vocabularies is owned by a separate 18-month project. What
  is explicitly encouraged there — *"using SKOS as framework to align the data
  internally"* — is what this module does.
* **Not a copy of AMCR or TEATER.** Those 5,594 source concepts already carry their
  own resolvable identifiers (``https://api.aiscr.cz/id/HES-…``,
  ``https://teater.aiscr.cz/id/…``) and are serialised where they are built, by
  ``atrium-nlp-enrich``'s ``vocab_build.py --skos``. This module mints URIs **only**
  for the ~90 concepts ATRIUM itself authors, and references the source URIs.
* **Not a validator that fails builds.** ``validate_labels()`` reports and returns
  findings; it never raises. This matches the house idiom already used for the
  origin check in ``atrium_document.py``, which abstains with a ``NOTE`` on stderr
  rather than refusing a document.

THE ONE SWAPPABLE CONSTANT
==========================
``SKOS_BASE`` is the only place an ATRIUM-authored URI is rooted. When the
SKOSification project lands real PIDs, either repoint the ``w3id.org/atrium``
redirect (no data changes at all) or change this one string and regenerate. Nothing
downstream hard-codes a URI; everything goes through :func:`concept_uri` and
:func:`scheme_uri`.

Registering the ``w3id.org/atrium`` redirect is optional and does not gate anything:
these URIs are identifiers first and locations second, and they identify correctly
before any redirect exists.

DEPENDENCIES
============
Standard library only. An RDF library is deliberately **not** introduced: the two
serialisations here are small, closed and fully determined by the tables below, and
the vocabulary artifacts they join are drift-gated by byte comparison
(``vocab_build.py --check``). A general-purpose serialiser that emits blank-node
identifiers or unordered triples would fail that gate on every run.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SKOS_BASE",
    "REGISTRY_VERSION",
    "concept_uri",
    "scheme_uri",
    "collection_uri",
    "CONCEPT_SCHEMES",
    "CONCEPTS",
    "COLLECTIONS",
    "MAPPINGS",
    "PAGE_CATEGORIES",
    "LINE_CATEGORIES",
    "LINE_CATEGORY_ORIGINATORS",
    "QUALITY_BANDS",
    "ENTITY_TYPES",
    "CNEC_TO_ENTITY_TYPE",
    "THEMES",
    "HESLAR_TO_THEME",
    "TEATER_BRANCH_TO_THEME",
    "EXCLUDED_RULE",
    "labels_for",
    "validate_labels",
    "Finding",
    "to_turtle",
    "to_jsonld",
]

# ── the one swappable constant ────────────────────────────────────────────────

#: Base URI for every ATRIUM-authored concept, scheme and collection.
#:
#: `w3id.org` is a community-run permanent-identifier redirect service. It was chosen
#: over the project domain (`atrium-research.eu`, which outlives neither the grant nor
#: a site redesign), over an institutional domain (which would encode one partner as
#: the owner of a jointly-authored vocabulary), and over a non-resolving `urn:`
#: (which would make the eventual migration a rewrite of stored values rather than a
#: redirect). See docs/skos_strategy.md §3.
SKOS_BASE = "https://w3id.org/atrium/"

#: Bumped when a concept is **added, removed or renamed**. Label-set membership is a
#: contract other repos read, so a change here is a change to that contract. Editing
#: a definition, note or mapping is additive and does not bump.
REGISTRY_VERSION = "1.0"

# ── external namespaces referenced (never minted here) ────────────────────────

NS = {
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dct": "http://purl.org/dc/terms/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "atrium": SKOS_BASE,
    "aat": "http://vocab.getty.edu/aat/",
    "amcr": "https://api.aiscr.cz/id/",
    "teater": "https://teater.aiscr.cz/id/",
}

#: The sentinel `taxonomy_config.json` uses for a source list that is deliberately
#: not offered to the model. It is NOT a theme and gets no URI.
EXCLUDED_RULE = "__exclude__"


def scheme_uri(scheme: str) -> str:
    """URI of an ATRIUM-authored concept scheme."""
    return f"{SKOS_BASE}scheme/{scheme}"


def concept_uri(scheme: str, notation: str) -> str:
    """URI of an ATRIUM-authored concept.

    ``notation`` is the label as the code spells it — ``TEXT_HW``, ``Clear``,
    ``Chronology``. It is used verbatim so that a URI can be derived from a value
    found in a document record without a lookup table, and so the reverse
    (URI → value) is a suffix split. Values containing a space are percent-free by
    construction in every current scheme except ``theme``, which is handled by
    :func:`_slug`.
    """
    return f"{SKOS_BASE}{scheme}/{_slug(notation)}"


def collection_uri(scheme: str, collection: str) -> str:
    """URI of an ATRIUM-authored collection (a facet grouping)."""
    return f"{SKOS_BASE}{scheme}/collection/{_slug(collection)}"


def _slug(value: str) -> str:
    """Local-name form of a label: spaces and ``&`` collapse to hyphens.

    Applied only to ATRIUM-authored notations. ``"Location & Admin"`` becomes
    ``location-admin``; ``"TEXT_HW"`` and ``"Clear"`` are unchanged apart from case,
    which is **preserved** — ``TEXT_HW`` and ``text_hw`` are not the same label and
    a case-folding slug would merge them.
    """
    out: List[str] = []
    prev_hyphen = False
    for ch in value:
        if ch.isalnum() or ch in "_.":
            out.append(ch)
            prev_hyphen = False
        elif not prev_hyphen:
            out.append("-")
            prev_hyphen = True
    return "".join(out).strip("-")


# ══════════════════════════════════════════════════════════════════════════════
# L2 — the concepts ATRIUM authors
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_SCHEMES: Dict[str, Dict[str, Any]] = {
    "page-category": {
        "title": "ATRIUM page category",
        "description": (
            "Structural classes a scanned page is assigned by atrium-page-classification. "
            "Three orthogonal facets — presence of graphical elements, type of text, and "
            "presence of a tabular layout — are modelled as skos:Collection rather than "
            "skos:broader, because they are orthogonal: a concept belongs to all three at "
            "once and has no single parent."
        ),
        "authority": "atrium-page-classification/model_registry.py CATEGORIES",
        "fields": ("page_categories", "pages[].category"),
    },
    "line-category": {
        "title": "ATRIUM line category",
        "description": (
            "Per-line verdicts carried by lines[].categ. The block has two authorised "
            "originators (see docs/document_schema.md, Originators) and they emit "
            "DISJOINT subsets of this scheme — which is why the union is declared in one "
            "place. See LINE_CATEGORY_ORIGINATORS."
        ),
        "authority": (
            "atrium-alto-postprocess/text_util.py determine_category(); "
            "atrium-llm-enrich/api_util/digital_to_json.py classify_line()"
        ),
        "fields": ("lines[].categ",),
    },
    "quality-band": {
        "title": "ATRIUM page quality band",
        "description": (
            "Page-level reduction of a page's Clear/Noisy/Trash line counts. Deterministic "
            "plurality vote; ties favour the more optimistic band. The only controlled "
            "vocabulary in atrium_document.schema.json that is already a closed enum."
        ),
        "authority": "atrium-alto-postprocess/document_hook.py quality_band()",
        "fields": ("pages[].quality_band",),
    },
    "entity-type": {
        "title": "ATRIUM coarse entity type",
        "description": (
            "The coarse TEITOK named-entity types. entities[] deliberately carries three "
            "parallel tagsets so no mapping is lost; this is the coarsest, and the only "
            "one that is closed."
        ),
        "authority": "atrium-nlp-enrich/api_util/teitok_alto.py",
        "fields": ("entities[].type_teitok",),
    },
    "cnec": {
        "title": "CNEC 2.0 entity codes as used by ATRIUM",
        "description": (
            "ATRIUM's local encoding of the Czech Named Entity Corpus 2.0 fine-grained "
            "codes that the pipeline actually maps. This is NOT a republication of CNEC "
            "and is not authoritative for it: only the subset _CNEC_TO_CONLL handles is "
            "listed, prefLabel is the code itself, and authoritative glosses come from the "
            "CNEC 2.0 documentation named in dct:source. Adding verified glosses as "
            "skos:definition is tracked in docs/skos_strategy.md §7."
        ),
        "authority": "atrium-nlp-enrich/api_util/teitok_alto.py _CNEC_TO_CONLL",
        "source": "https://ufal.mff.cuni.cz/cnec/cnec2.0",
        "fields": ("entities[].type_cnec",),
    },
    "theme": {
        "title": "ATRIUM harmonisation theme",
        "description": (
            "The facets the harmonised AMCR+TEATER vocabulary is rolled up into for "
            "prompt construction. These are ATRIUM's own editorial grouping, not a "
            "structure either source thesaurus asserts — which is exactly why they need "
            "ATRIUM-minted URIs while the source concepts do not."
        ),
        "authority": "atrium-nlp-enrich/data_samples/taxonomy_config.json",
        "fields": ("enrichment.items[].teater_category (indirectly, via the facet)",),
    },
}

# ── page-category ─────────────────────────────────────────────────────────────
# Definitions are lifted verbatim from atrium-page-classification/README.md, which
# is the only place they have ever been written down.

_PAGE_CATEGORY_DEFS: Dict[str, str] = {
    "DRAW": (
        "drawings, maps, paintings, schematics, or graphics, potentially containing "
        "some text labels or captions"
    ),
    "DRAW_L": (
        "drawings, etc but presented within a table-like layout or includes a legend "
        "formatted as a table"
    ),
    "LINE_HW": "handwritten text organized in a tabular or form-like structure",
    "LINE_P": "printed text organized in a tabular or form-like structure",
    "LINE_T": "machine-typed text organized in a tabular or form-like structure",
    "PHOTO": "photographs or photographic cutouts, potentially with text captions",
    "PHOTO_L": (
        "photos presented within a table-like layout or accompanied by tabular annotations"
    ),
    "TEXT": (
        "mixtures of printed, handwritten, and/or typed text, potentially with minor "
        "graphical elements"
    ),
    "TEXT_HW": "only handwritten text in paragraph or block form (non-tabular)",
    "TEXT_P": "only printed text in paragraph or block form (non-tabular)",
    "TEXT_T": "only machine-typed text in paragraph or block form (non-tabular)",
}

#: The 11 page classes, in the order model_registry.CATEGORIES declares them.
#: THE ORDER IS LOAD-BEARING on the training side (it is the label→index binding),
#: so this tuple must stay in step with model_registry.CATEGORIES exactly.
PAGE_CATEGORIES: Tuple[str, ...] = (
    "DRAW",
    "DRAW_L",
    "LINE_HW",
    "LINE_P",
    "LINE_T",
    "PHOTO",
    "PHOTO_L",
    "TEXT",
    "TEXT_HW",
    "TEXT_P",
    "TEXT_T",
)

# ── line-category ─────────────────────────────────────────────────────────────
# Definitions from atrium-alto-postprocess/text_util.py's "Categories Outputted"
# docstring and from atrium-llm-enrich/api_util/digital_to_json.py.

_LINE_CATEGORY_DEFS: Dict[str, str] = {
    "Empty": "A blank line.",
    "Non-text": "Lines that are too short, lack letters, or are purely numbers/symbols.",
    "Trash": (
        "Severe OCR corruption, high symbol density, gibberish, or failed language ID."
    ),
    "Noisy": (
        "Partially degraded text (e.g., isolated strange symbols, mid-word uppercase)."
    ),
    "Clear": "Structurally sound text with low perplexity.",
    "Garbage": (
        "An embedded text layer that decodes to systematically wrong characters — the "
        "digital-born counterpart of Trash. Emitted when decode sanity falls below "
        "QUALITY_GARBAGE_BELOW, or on GARBAGE_MIN_HITS confusable characters regardless "
        "of line length."
    ),
    "Inverted": (
        "Mirrored or 180-degree-rotated text, whose extracted string is not evidence of "
        "anything. Wins over Garbage: the fault is at rendering level, so reporting a "
        "decode verdict on it would be noise."
    ),
}

#: Which originator emits which label. `lines[].categ`'s two authorised originators
#: are mutually exclusive per document (fixed by `source.origin`), and their label
#: sets do not overlap. A consumer that filters this field must handle BOTH sets or
#: it silently does nothing on one of the two paths.
LINE_CATEGORY_ORIGINATORS: Dict[str, Tuple[str, ...]] = {
    "alto-postprocess": ("Clear", "Empty", "Noisy", "Non-text", "Trash"),
    "digital-convert": ("Garbage", "Inverted"),
}

#: The union both consumers must understand.
LINE_CATEGORIES: Tuple[str, ...] = tuple(
    sorted({v for values in LINE_CATEGORY_ORIGINATORS.values() for v in values})
)

#: Labels that mean "this line's text is not trustworthy — do not show it to a model".
#:
#: This is the semantic set. It is deliberately NOT wired into
#: atrium-llm-enrich/api_util/json_to_md.py's DROP_CATEGORIES, which today is
#: frozenset({"Garbage", "Inverted"}) and therefore matches nothing on the OCR path.
#: Changing that constant is a pipeline behaviour change and is out of scope for the
#: registry; the discrepancy is recorded in docs/skos_strategy.md §6 (defect V-1) with
#: the one-line fix, so it is available whenever someone decides to take it.
UNTRUSTWORTHY_LINE_CATEGORIES: Tuple[str, ...] = ("Garbage", "Inverted", "Trash")

# ── quality-band ──────────────────────────────────────────────────────────────

QUALITY_BANDS: Tuple[str, ...] = ("Trash", "Noisy", "Clear")

_QUALITY_BAND_DEFS: Dict[str, str] = {
    "Clear": "Clear lines are at least as numerous as Noisy and as Trash.",
    "Noisy": "Not Clear, and Noisy lines are at least as numerous as Trash.",
    "Trash": "Neither Clear nor Noisy holds; Trash lines predominate.",
}

# ── entity-type ───────────────────────────────────────────────────────────────

ENTITY_TYPES: Tuple[str, ...] = ("LOC", "MISC", "ORG", "PER")

_ENTITY_TYPE_DEFS: Dict[str, str] = {
    "PER": "Person.",
    "ORG": "Organisation or institution.",
    "LOC": "Location, including geopolitical and natural features.",
    "MISC": (
        "Any recognised entity outside the three coarse classes above, and the fallback "
        "for a fine-grained code with no coarse mapping."
    ),
}

# ── cnec ──────────────────────────────────────────────────────────────────────
# Mirrors _CNEC_TO_CONLL in atrium-nlp-enrich/api_util/teitok_alto.py exactly.
# NOTE: no skos:definition is asserted — see the scheme description above.

CNEC_TO_ENTITY_TYPE: Dict[str, str] = {
    "P": "PER",
    "p": "PER",
    "p_": "PER",
    "pc": "PER",
    "pd": "PER",
    "pf": "PER",
    "ph": "PER",
    "pm": "PER",
    "pp": "PER",
    "ps": "PER",
    "I": "ORG",
    "i": "ORG",
    "i_": "ORG",
    "ia": "ORG",
    "ic": "ORG",
    "if": "ORG",
    "io": "ORG",
    "G": "LOC",
    "g": "LOC",
    "g_": "LOC",
    "gc": "LOC",
    "gh": "LOC",
    "gl": "LOC",
    "gq": "LOC",
    "gr": "LOC",
    "gs": "LOC",
    "gt": "LOC",
    "gu": "LOC",
}

# ── theme ─────────────────────────────────────────────────────────────────────
# Definitions are the `_comment` values in taxonomy_config.json; priority is the
# tie-break order the placement rules use, preserved here as skos:notation would be
# the wrong home for an integer.

_THEME_DEFS: Dict[str, Optional[str]] = {
    "Chronology": "Periods, archaeological cultures, chronozones.",
    "Activity Area": (
        "Areas defined by the activity carried out there: settlement, burial, cult, "
        "production, mining, military."
    ),
    "Feature": "Physical features and structures, and how they are built.",
    "Artefact": "Objects and finds.",
    "Material": "What an object or feature is made of.",
    "Finds Context": "The deposition circumstances a find was recovered from.",
    "Methods": "Fieldwork, processing and analytical methods.",
    "Location & Admin": (
        "Organisation types, heritage-protection status, event types, museums and "
        "heritage-care bodies."
    ),
    "Cultural & Geographic Context": None,
    "Related Disciplines & Society": None,
    "Other": (
        "Terms no rule placed. Empty in every current build — with heslar_map and "
        "teater_branch_map in force nothing falls through. A non-empty Other means a "
        "source added a list the maps do not cover; treat it as a build failure to "
        "investigate, not a bucket to hide things in."
    ),
}

_THEME_PRIORITY: Dict[str, int] = {
    "Chronology": 9,
    "Activity Area": 8,
    "Feature": 7,
    "Artefact": 6,
    "Material": 5,
    "Finds Context": 4,
    "Methods": 3,
    "Location & Admin": 1,
    "Cultural & Geographic Context": 0,
    "Related Disciplines & Society": 0,
    "Other": -1,
}

THEMES: Tuple[str, ...] = tuple(sorted(_THEME_DEFS))

# ══════════════════════════════════════════════════════════════════════════════
# The assembled concept tables
# ══════════════════════════════════════════════════════════════════════════════


def _concepts_from(
    defs: Mapping[str, Optional[str]],
    *,
    label_lang: str = "en",
    extra: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for notation in sorted(defs):
        entry: Dict[str, Any] = {"labels": {label_lang: notation}}
        if defs[notation]:
            entry["definition"] = {label_lang: defs[notation]}
        if extra and notation in extra:
            entry.update(extra[notation])
        out[notation] = entry
    return out


CONCEPTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "page-category": _concepts_from(_PAGE_CATEGORY_DEFS),
    "line-category": _concepts_from(
        _LINE_CATEGORY_DEFS,
        extra={
            name: {"originator": originator}
            for originator, names in LINE_CATEGORY_ORIGINATORS.items()
            for name in names
        },
    ),
    "quality-band": _concepts_from(_QUALITY_BAND_DEFS),
    "entity-type": _concepts_from(_ENTITY_TYPE_DEFS),
    "cnec": {code: {"labels": {"en": code}} for code in sorted(CNEC_TO_ENTITY_TYPE)},
    "theme": _concepts_from(
        _THEME_DEFS,
        extra={name: {"priority": prio} for name, prio in _THEME_PRIORITY.items()},
    ),
}

#: Orthogonal facets over page-category, as skos:Collection. A collection asserts
#: membership without asserting hierarchy, which is what "orthogonal facet" means and
#: what skos:broader would get wrong.
COLLECTIONS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "page-category": {
        "graphical": {
            "label": "carries graphical elements",
            "members": ("DRAW", "DRAW_L", "PHOTO", "PHOTO_L", "TEXT"),
        },
        "tabular": {
            "label": "laid out as a table, form or legend",
            "members": ("DRAW_L", "LINE_HW", "LINE_P", "LINE_T", "PHOTO_L"),
        },
        "handwritten": {
            "label": "contains handwritten text",
            "members": ("LINE_HW", "TEXT", "TEXT_HW"),
        },
        "printed": {
            "label": "contains printed text",
            "members": ("LINE_P", "TEXT", "TEXT_P"),
        },
        "typed": {
            "label": "contains machine-typed text",
            "members": ("LINE_T", "TEXT", "TEXT_T"),
        },
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# L3 — mappings
# ══════════════════════════════════════════════════════════════════════════════

#: AMCR heslář name → ATRIUM theme, from taxonomy_config.json _settings.heslar_map.
#: `__exclude__` entries are omitted: they are a build-time gate, not an assertion
#: about meaning, and serialising them as a mapping would claim something false.
HESLAR_TO_THEME: Dict[str, str] = {
    "adb_podnet": "Methods",
    "adb_typ": "Methods",
    "akce_typ": "Methods",
    "akce_typ_kat": "Methods",
    "aktivita": "Activity Area",
    "areal": "Activity Area",
    "areal_kat": "Activity Area",
    "jazyk": "Cultural & Geographic Context",
    "lokalita_druh": "Activity Area",
    "lokalita_druh_kat": "Activity Area",
    "lokalita_typ": "Activity Area",
    "nalezove_okolnosti": "Finds Context",
    "objekt_druh": "Feature",
    "objekt_druh_kat": "Feature",
    "objekt_specifikace": "Material",
    "obdobi": "Chronology",
    "obdobi_kat": "Chronology",
    "organizace_typ": "Location & Admin",
    "pamatkova_ochrana": "Location & Admin",
    "posudek_typ": "Methods",
    "posudek_typ_kat": "Methods",
    "predmet_druh": "Artefact",
    "predmet_druh_kat": "Artefact",
    "predmet_specifikace": "Material",
    "stav_dochovani": "Location & Admin",
    "udalost_typ": "Location & Admin",
    "zeme": "Cultural & Geographic Context",
}

#: TEATER top-level branch id → ATRIUM theme, from _settings.teater_branch_map.
#: The branch ids are TEATER concepts and resolve at https://teater.aiscr.cz/id/<id>.
TEATER_BRANCH_TO_THEME: Dict[str, str] = {
    "1": "Related Disciplines & Society",
    "102": "Methods",
    "140": "Methods",
    "220": "Location & Admin",
    "253": "Location & Admin",
    "288": "Related Disciplines & Society",
    "1050": "Chronology",
    "1267": "Activity Area",
    "1481": "Feature",
    "1788": "Artefact",
    "2430": "Material",
    "2557": "Related Disciplines & Society",
    "2558": "Related Disciplines & Society",
    "2560": "Cultural & Geographic Context",
    "2900": "Cultural & Geographic Context",
    "3076": "Cultural & Geographic Context",
    "3091": "Related Disciplines & Society",
    "3094": "Related Disciplines & Society",
    "3549": "Related Disciplines & Society",
}


def _mappings() -> List[Tuple[str, str, str]]:
    """Every mapping assertion, as (subject URI, SKOS property, object URI).

    Four families:

    1. **CNEC → coarse entity type** — ``skos:broadMatch``. A fine-grained code is
       genuinely narrower than the coarse class it rolls up to.
    2. **TEATER branch → theme** — ``skos:broadMatch``. The branch is a real TEATER
       concept with a resolvable URI; the theme is ATRIUM's coarser grouping.
    3. **AMCR heslář → theme** — ``skos:broadMatch``. A heslář is a named list rather
       than a concept, so it is minted here as a ``skos:Collection``
       (``atrium:heslar/<name>``) and the mapping is asserted from that collection.
    4. **Trash ↔ Garbage** — ``skos:closeMatch``. The two originators' verdicts for
       "this line's text is corrupt" are close but not identical: one is an OCR
       judgement over a rendered image, the other a decode-sanity judgement over an
       embedded text layer. ``closeMatch`` says "interchangeable in some applications"
       and refuses the transitivity ``exactMatch`` would license — which is exactly
       right here, and is the machine-readable form of defect V-1.
    """
    out: List[Tuple[str, str, str]] = []
    for code, coarse in CNEC_TO_ENTITY_TYPE.items():
        out.append(
            (concept_uri("cnec", code), "skos:broadMatch", concept_uri("entity-type", coarse))
        )
    for branch, theme in TEATER_BRANCH_TO_THEME.items():
        out.append(
            (NS["teater"] + branch, "skos:broadMatch", concept_uri("theme", theme))
        )
    for heslar, theme in HESLAR_TO_THEME.items():
        out.append(
            (f"{SKOS_BASE}heslar/{_slug(heslar)}", "skos:broadMatch", concept_uri("theme", theme))
        )
    out.append(
        (
            concept_uri("line-category", "Trash"),
            "skos:closeMatch",
            concept_uri("line-category", "Garbage"),
        )
    )
    return sorted(out)


MAPPINGS: List[Tuple[str, str, str]] = _mappings()


# ══════════════════════════════════════════════════════════════════════════════
# Lookup and advisory validation
# ══════════════════════════════════════════════════════════════════════════════


def labels_for(scheme: str) -> Tuple[str, ...]:
    """The permitted label values of ``scheme``, sorted.

    Raises ``KeyError`` for an unknown scheme — that is a typo in calling code, not a
    data condition, and silently returning an empty tuple would turn every subsequent
    validation into a no-op that reports success.
    """
    return tuple(sorted(CONCEPTS[scheme]))


class Finding(Tuple[str, str, str]):
    """One advisory validation result: ``(scheme, value, message)``."""

    __slots__ = ()

    @property
    def scheme(self) -> str:
        return self[0]

    @property
    def value(self) -> str:
        return self[1]

    @property
    def message(self) -> str:
        return self[2]

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.scheme}: {self.message}"


def validate_labels(
    scheme: str,
    observed: Iterable[str],
    *,
    originator: Optional[str] = None,
    report: bool = True,
    stream: Any = None,
) -> List[Finding]:
    """Check observed label values against ``scheme``. **Never raises.**

    Returns a list of :class:`Finding`. When ``report`` is true, each is also written
    to ``stream`` (default stderr) prefixed ``NOTE`` — the same shape
    ``atrium_document.py`` uses for an origin it has not been taught, and for the same
    reason: a gate that stops a pipeline over a label it does not recognise is worse
    than one that makes the disagreement visible.

    ``originator`` narrows ``line-category`` to the subset that one tool may emit, so
    a caller can check "did alto-postprocess emit something only digital-convert
    should" as well as "did anyone emit something nobody should".
    """
    stream = sys.stderr if stream is None else stream
    findings: List[Finding] = []

    try:
        permitted = set(labels_for(scheme))
    except KeyError:
        findings.append(Finding((scheme, "", f"unknown scheme {scheme!r}")))
        permitted = set()

    if originator is not None and scheme == "line-category":
        allowed = set(LINE_CATEGORY_ORIGINATORS.get(originator, ()))
        if not allowed:
            findings.append(
                Finding((scheme, "", f"unknown originator {originator!r}; not narrowing"))
            )
        else:
            permitted = allowed

    seen: set = set()
    for value in observed:
        if value in seen or value in permitted:
            seen.add(value)
            continue
        seen.add(value)
        if not permitted:
            continue
        hint = ""
        folded = {p.casefold(): p for p in permitted}
        if value.casefold() in folded:
            hint = f" — did you mean {folded[value.casefold()]!r}? (case differs)"
        elif scheme == "line-category" and value in set(labels_for(scheme)):
            other = [
                tool
                for tool, names in LINE_CATEGORY_ORIGINATORS.items()
                if value in names
            ]
            hint = f" — that label belongs to {', '.join(other)}"
        findings.append(
            Finding((scheme, value, f"{value!r} is not in {scheme}{hint}"))
        )

    if report:
        for finding in findings:
            print(f"NOTE [atrium_vocab] {finding}", file=stream)
    return findings


# ══════════════════════════════════════════════════════════════════════════════
# Serialisation — deterministic by construction
# ══════════════════════════════════════════════════════════════════════════════


def _ttl_string(value: str, lang: Optional[str] = None) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"@{lang}' if lang else f'"{escaped}"'


def _prefixes() -> Dict[str, str]:
    """The full prefix map: the fixed external namespaces plus one per ATRIUM scheme.

    A per-scheme prefix exists because a Turtle PNAME local part may not contain an
    unescaped ``/``. Without it every ATRIUM concept would serialise as a full angle-
    bracket IRI — valid, but four times the bytes and unreadable in review, which
    matters for a file whose whole purpose is to be read by humans deciding whether a
    label set is right.
    """
    out = dict(NS)
    out["atrium-scheme"] = f"{SKOS_BASE}scheme/"
    out["atrium-heslar"] = f"{SKOS_BASE}heslar/"
    for scheme in CONCEPT_SCHEMES:
        out[f"a-{scheme}"] = f"{SKOS_BASE}{scheme}/"
        if scheme in COLLECTIONS:
            out[f"a-{scheme}-coll"] = f"{SKOS_BASE}{scheme}/collection/"
    return out


def _curie(uri: str, prefixes: Optional[Mapping[str, str]] = None) -> str:
    """Shorten a URI to a prefixed name when the local part is Turtle-safe.

    Longest base wins, so ``…/page-category/collection/tabular`` picks
    ``a-page-category-coll`` over ``a-page-category`` and never produces a local part
    with a ``/`` in it. Anything that does not shorten safely falls back to a full IRI
    rather than emitting something a parser would reject.
    """
    prefixes = _prefixes() if prefixes is None else prefixes
    for prefix, base in sorted(prefixes.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if not uri.startswith(base):
            continue
        local = uri[len(base) :]
        if local and all(c.isalnum() or c in "_-." for c in local):
            return f"{prefix}:{local}"
        return f"<{uri}>"
    return f"<{uri}>"


class _Lit:
    """A literal object: a value plus an optional language tag."""

    __slots__ = ("value", "lang")

    def __init__(self, value: str, lang: Optional[str] = None) -> None:
        self.value = value
        self.lang = lang

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, _Lit)
            and other.value == self.value
            and other.lang == self.lang
        )

    def __hash__(self) -> int:
        return hash((self.value, self.lang))


class _Ref:
    """A URI object."""

    __slots__ = ("uri",)

    def __init__(self, uri: str) -> None:
        self.uri = uri

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Ref) and other.uri == self.uri

    def __hash__(self) -> int:
        return hash(self.uri)


def triples() -> List[Tuple[str, str, Any]]:
    """Every triple in the registry, exactly once.

    **Both serialisers render this list and nothing else.** They used not to, and the
    two views promptly disagreed: the Turtle heslář collections carried an
    ``rdfs:comment`` the JSON-LD ones did not, so the same registry described a 507-
    and a 480-triple graph depending on which button you pressed. Two hand-written
    renderers over one set of tables reproduce, in miniature, precisely the fault this
    whole module exists to prevent — a label set declared twice drifts. One generator,
    many views.

    Predicates are emitted as CURIEs against :data:`NS`; objects are :class:`_Ref` or
    :class:`_Lit`. Sorted, so output is byte-stable across runs and interpreters.
    """
    out: List[Tuple[str, str, Any]] = []

    for scheme in sorted(CONCEPT_SCHEMES):
        meta = CONCEPT_SCHEMES[scheme]
        s_uri = scheme_uri(scheme)
        out.append((s_uri, "rdf:type", _Ref(NS["skos"] + "ConceptScheme")))
        out.append((s_uri, "dct:title", _Lit(meta["title"], "en")))
        out.append((s_uri, "dct:description", _Lit(meta["description"], "en")))
        if meta.get("source"):
            out.append((s_uri, "dct:source", _Ref(meta["source"])))
        out.append((s_uri, "rdfs:comment", _Lit("authority: " + meta["authority"], "en")))

        for notation in sorted(CONCEPTS[scheme]):
            entry = CONCEPTS[scheme][notation]
            c_uri = concept_uri(scheme, notation)
            out.append((c_uri, "rdf:type", _Ref(NS["skos"] + "Concept")))
            out.append((c_uri, "skos:inScheme", _Ref(s_uri)))
            out.append((c_uri, "skos:notation", _Lit(notation)))
            for lang in sorted(entry["labels"]):
                out.append((c_uri, "skos:prefLabel", _Lit(entry["labels"][lang], lang)))
            for lang in sorted(entry.get("definition", {})):
                out.append(
                    (c_uri, "skos:definition", _Lit(entry["definition"][lang], lang))
                )
            if entry.get("originator"):
                out.append(
                    (
                        c_uri,
                        "rdfs:comment",
                        _Lit("emitted by: " + entry["originator"], "en"),
                    )
                )

        for name in sorted(COLLECTIONS.get(scheme, {})):
            coll = COLLECTIONS[scheme][name]
            k_uri = collection_uri(scheme, name)
            out.append((k_uri, "rdf:type", _Ref(NS["skos"] + "Collection")))
            out.append((k_uri, "skos:prefLabel", _Lit(coll["label"], "en")))
            for member in sorted(coll["members"]):
                out.append((k_uri, "skos:member", _Ref(concept_uri(scheme, member))))

    for heslar in sorted(HESLAR_TO_THEME):
        h_uri = f"{SKOS_BASE}heslar/{_slug(heslar)}"
        out.append((h_uri, "rdf:type", _Ref(NS["skos"] + "Collection")))
        out.append((h_uri, "skos:prefLabel", _Lit(heslar, "cs")))
        out.append(
            (
                h_uri,
                "rdfs:comment",
                _Lit(
                    "AMCR heslar (controlled list) name, as harvested from nazev_heslare",
                    "en",
                ),
            )
        )

    for subject, prop, obj in MAPPINGS:
        out.append((subject, prop, _Ref(obj)))

    return sorted(out, key=lambda t: (t[0], t[1], _obj_sort_key(t[2])))


def _obj_sort_key(obj: Any) -> Tuple[int, str, str]:
    if isinstance(obj, _Ref):
        return (0, obj.uri, "")
    return (1, obj.value, obj.lang or "")


def to_turtle(*, base: str = SKOS_BASE) -> str:
    """The registry as Turtle. Sorted, blank-node free, byte-stable.

    ``base`` overrides :data:`SKOS_BASE` for one call, so the registry can be rendered
    under a different namespace without mutating module state — the escape hatch for
    the day the real PIDs arrive and someone wants to diff the two.
    """
    if base != globals()["SKOS_BASE"]:
        return _with_base(base, lambda: to_turtle(base=base))

    prefixes = _prefixes()
    lines: List[str] = [
        "# ATRIUM controlled-label registry - generated by atrium_vocab.py.",
        "# DO NOT EDIT: edit the tables in atrium_vocab.py and regenerate.",
        f"# registry_version {REGISTRY_VERSION}",
        "",
    ]
    for prefix in sorted(prefixes):
        lines.append(f"@prefix {prefix}: <{prefixes[prefix]}> .")
    lines.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
    lines.append("")

    grouped: Dict[str, List[Tuple[str, Any]]] = {}
    order: List[str] = []
    for subject, prop, obj in triples():
        if subject not in grouped:
            grouped[subject] = []
            order.append(subject)
        grouped[subject].append((prop, obj))

    for subject in order:
        rendered = [
            f"    {prop} "
            + (
                _curie(o.uri, prefixes)
                if isinstance(o, _Ref)
                else _ttl_string(o.value, o.lang)
            )
            for prop, o in grouped[subject]
        ]
        # Subject on the same line as its first predicate: the conventional Turtle
        # layout, and it keeps the invariant that EVERY content line ends in " ;" or
        # " ." -- which is what _selftest()'s cheap syntax check relies on.
        head = f"{_curie(subject, prefixes)} {rendered[0].lstrip()}"
        block = [head] + rendered[1:]
        lines.append(" ;\n".join(block) + " .")
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def to_jsonld(*, base: str = SKOS_BASE) -> str:
    """The registry as JSON-LD. ``sort_keys=True`` - byte-stable.

    Renders the same :func:`triples` list as :func:`to_turtle`, so the two are
    guaranteed to describe an identical graph.
    """
    if base != globals()["SKOS_BASE"]:
        return _with_base(base, lambda: to_jsonld(base=base))

    grouped: Dict[str, Dict[str, List[Any]]] = {}
    order: List[str] = []
    for subject, prop, obj in triples():
        if subject not in grouped:
            grouped[subject] = {}
            order.append(subject)
        key = "@type" if prop == "rdf:type" else prop
        value: Any
        if isinstance(obj, _Ref):
            # `@type` is compacted against the plain NS map (never the per-scheme
            # prefixes, which exist only to keep Turtle local names `/`-free). A full
            # IRI here would still be valid JSON-LD but would not match what Turtle
            # renders, and "the two views agree" is the property this module is built
            # around -- so they agree on the wire, not just after a reasoner runs.
            value = _curie(obj.uri, NS) if key == "@type" else {"@id": obj.uri}
        elif obj.lang:
            value = {"@language": obj.lang, "@value": obj.value}
        else:
            value = obj.value
        grouped[subject].setdefault(key, []).append(value)

    graph: List[Dict[str, Any]] = []
    for subject in order:
        node: Dict[str, Any] = {"@id": subject}
        for key in sorted(grouped[subject]):
            values = grouped[subject][key]
            node[key] = values[0] if len(values) == 1 else values
        graph.append(node)

    payload = {
        "@context": dict(sorted(NS.items())),
        "registry_version": REGISTRY_VERSION,
        "@graph": graph,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _with_base(base: str, fn: Any) -> str:
    """Run ``fn`` with :data:`SKOS_BASE` temporarily rebound. Restores on any exit."""
    saved_base = globals()["SKOS_BASE"]
    saved_ns = NS["atrium"]
    saved_mappings = list(MAPPINGS)
    globals()["SKOS_BASE"] = base
    NS["atrium"] = base
    MAPPINGS[:] = _mappings()
    try:
        return fn()
    finally:
        globals()["SKOS_BASE"] = saved_base
        NS["atrium"] = saved_ns
        MAPPINGS[:] = saved_mappings


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def _selftest(stream: Any = None) -> int:
    """Structural checks over the registry itself. Returns an exit code.

    This is the one place the registry is allowed to be strict: a contradiction
    *inside* the tables is a bug in this file, not a data condition in someone's
    pipeline, and there is no pipeline to avoid stalling.
    """
    stream = sys.stdout if stream is None else stream
    problems: List[str] = []

    for scheme in CONCEPTS:
        if scheme not in CONCEPT_SCHEMES:
            problems.append(f"{scheme}: concepts declared with no scheme metadata")
    for scheme in CONCEPT_SCHEMES:
        if not CONCEPTS.get(scheme):
            problems.append(f"{scheme}: scheme declared with no concepts")

    for scheme, members in COLLECTIONS.items():
        for name, coll in members.items():
            for m in coll["members"]:
                if m not in CONCEPTS.get(scheme, {}):
                    problems.append(f"{scheme}/{name}: member {m!r} is not a concept")

    union = {v for names in LINE_CATEGORY_ORIGINATORS.values() for v in names}
    if union != set(CONCEPTS["line-category"]):
        problems.append(
            "line-category: originator sets and concept table disagree: "
            f"{sorted(union ^ set(CONCEPTS['line-category']))}"
        )
    tools = list(LINE_CATEGORY_ORIGINATORS)
    for i, a in enumerate(tools):
        for b in tools[i + 1 :]:
            overlap = set(LINE_CATEGORY_ORIGINATORS[a]) & set(LINE_CATEGORY_ORIGINATORS[b])
            if overlap:
                problems.append(f"line-category: {a} and {b} both emit {sorted(overlap)}")

    for value in UNTRUSTWORTHY_LINE_CATEGORIES:
        if value not in CONCEPTS["line-category"]:
            problems.append(f"UNTRUSTWORTHY_LINE_CATEGORIES: {value!r} is not a line-category")

    for code, coarse in CNEC_TO_ENTITY_TYPE.items():
        if coarse not in CONCEPTS["entity-type"]:
            problems.append(f"cnec/{code}: maps to unknown entity type {coarse!r}")
    for theme in list(HESLAR_TO_THEME.values()) + list(TEATER_BRANCH_TO_THEME.values()):
        if theme not in CONCEPTS["theme"]:
            problems.append(f"mapping target {theme!r} is not a theme")
        if theme == EXCLUDED_RULE:
            problems.append(f"{EXCLUDED_RULE} leaked into a mapping table")

    for band in QUALITY_BANDS:
        if band not in CONCEPTS["line-category"]:
            problems.append(
                f"quality-band {band!r} has no line-category counterpart; the band is a "
                "reduction of line counts and every band name should be a line label"
            )

    uris = [concept_uri(s, n) for s in CONCEPTS for n in CONCEPTS[s]]
    if len(set(uris)) != len(uris):
        problems.append("concept URIs are not unique after slugging")

    turtle = to_turtle()
    if turtle != to_turtle():
        problems.append("to_turtle() is not deterministic")
    if to_jsonld() != to_jsonld():
        problems.append("to_jsonld() is not deterministic")
    json.loads(to_jsonld())
    for line in turtle.splitlines():
        # `""` must never appear in this prefix tuple: str.startswith("") is True for
        # every string, which would turn the check into a no-op that reports success.
        # That is the same class of fault as defect V-1, in the gate written to catch it.
        if not line or line.startswith(("#", "@prefix")):
            continue
        if not line.endswith((" .", " ;")):
            problems.append(f"unterminated Turtle line: {line!r}")

    rebased = to_turtle(base="https://example.invalid/x/")
    if SKOS_BASE not in to_turtle() or "example.invalid" not in rebased:
        problems.append("base override did not take effect")
    if to_turtle() != turtle:
        problems.append("base override leaked: module state was not restored")

    total = sum(len(v) for v in CONCEPTS.values())
    print(
        f"atrium_vocab {REGISTRY_VERSION}: {len(CONCEPT_SCHEMES)} schemes, "
        f"{total} concepts, {sum(len(v) for v in COLLECTIONS.values())} collections, "
        f"{len(MAPPINGS)} mappings, base {SKOS_BASE}",
        file=stream,
    )
    for problem in problems:
        print(f"FAIL {problem}", file=stream)
    print("OK" if not problems else f"{len(problems)} problem(s)", file=stream)
    return 1 if problems else 0


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--selftest", action="store_true", help="structural checks; exit 1 on any")
    parser.add_argument("--turtle", action="store_true", help="print the registry as Turtle")
    parser.add_argument("--jsonld", action="store_true", help="print the registry as JSON-LD")
    parser.add_argument("--labels", metavar="SCHEME", help="print one scheme's permitted values")
    parser.add_argument(
        "--base",
        default=SKOS_BASE,
        help="render under a different base URI (does not mutate the module)",
    )
    args = parser.parse_args(argv)

    if args.labels:
        try:
            for value in labels_for(args.labels):
                print(value)
        except KeyError:
            print(
                f"unknown scheme {args.labels!r}; known: {', '.join(sorted(CONCEPT_SCHEMES))}",
                file=sys.stderr,
            )
            return 2
        return 0
    if args.turtle:
        sys.stdout.write(to_turtle(base=args.base))
        return 0
    if args.jsonld:
        sys.stdout.write(to_jsonld(base=args.base))
        return 0
    if args.selftest:
        return _selftest()
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
