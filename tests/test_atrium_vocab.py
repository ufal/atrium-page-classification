"""
tests/test_atrium_vocab.py — the controlled-label registry's contract.

HUB-CANONICAL. Edit the copy in the hub's ``docs/templates/shared/`` and re-vendor with
``scripts/revendor_shared.sh``; ``para-drift.reusable.yml`` diffs it against every tool
repo. Promoted to the canonical set for the same reason as
``test_document_originators.py``: the registry is a cross-repo contract, so the test that
pins it has to run everywhere the contract is relied on, not in whichever repo happened
to author it.

``atrium_vocab.py --selftest`` covers the registry's internal consistency and runs in CI
directly. What is pinned HERE is the part other repos actually call: the URI policy, the
advisory-never-fatal validation contract, and the two serialisations agreeing.
"""

from __future__ import annotations

import io
import json

import pytest

import atrium_vocab as av

# ── the URI policy (the whole point of issue atrium-project#51) ───────────────


def test_every_atrium_uri_derives_from_the_one_swappable_constant():
    """No URI is hard-coded anywhere. This is what makes the future PID migration a
    one-line change instead of a data rewrite."""
    for scheme in av.CONCEPT_SCHEMES:
        assert av.scheme_uri(scheme).startswith(av.SKOS_BASE)
        for notation in av.CONCEPTS[scheme]:
            assert av.concept_uri(scheme, notation).startswith(av.SKOS_BASE)


def test_rendering_under_a_different_base_does_not_mutate_module_state():
    """The escape hatch for the day real PIDs arrive: render under a new base to diff
    the two, without the module silently staying rebased afterwards."""
    before = av.to_turtle()
    rebased = av.to_turtle(base="https://example.invalid/x/")

    assert "example.invalid" in rebased
    assert av.SKOS_BASE not in rebased
    assert av.to_turtle() == before
    assert av.SKOS_BASE == "https://w3id.org/atrium/"


def test_source_concepts_are_never_minted_here():
    """ATRIUM names only what ATRIUM authored.

    AMCR and TEATER concepts keep the sources' own identifiers -- they are exactly the
    IDs that become PID references when the SKOSification project ships, so an
    ATRIUM-minted alias for one would be a migration liability invented for nothing.
    The only api.aiscr.cz / teater.aiscr.cz URIs in this registry are mapping OBJECTS
    and prefix declarations, never subjects of a Concept declaration.
    """
    for subject, prop, obj in av.triples():
        if prop == "rdf:type" and getattr(obj, "uri", "").endswith("#Concept"):
            assert subject.startswith(av.SKOS_BASE), subject


# ── validation is advisory, never fatal ──────────────────────────────────────


def test_validate_labels_never_raises_and_reports_findings():
    buf = io.StringIO()
    findings = av.validate_labels("page-category", ["TEXT", "Text", "Plate"], stream=buf)

    assert [f.value for f in findings] == ["Text", "Plate"]
    assert "case differs" in buf.getvalue()  # 'Text' vs 'TEXT'
    assert all(line.startswith("NOTE") for line in buf.getvalue().splitlines())


def test_validate_labels_catches_the_schema_examples_defect():
    """atrium_document.schema.json used to illustrate `page_categories` with
    `{"1": "Text", "2": "Plate"}` -- neither a member of the set it illustrates."""
    findings = av.validate_labels("page-category", ["Text", "Plate"], report=False)
    assert len(findings) == 2


@pytest.mark.parametrize(
    "scheme,observed",
    [("no-such-scheme", ["x"]), ("page-category", []), ("theme", ["", "Chronology"])],
)
def test_validate_labels_survives_anything(scheme, observed):
    assert isinstance(av.validate_labels(scheme, observed, report=False), list)


def test_validate_labels_narrows_by_originator():
    """`lines[].categ`'s two originators emit disjoint sets, so 'is this label legal'
    and 'is this label legal FROM THIS TOOL' are different questions."""
    alto = {"originator": "alto-postprocess", "report": False}

    assert av.validate_labels("line-category", ["Trash", "Clear"], **alto) == []

    findings = av.validate_labels("line-category", ["Garbage"], **alto)
    assert len(findings) == 1
    assert "belongs to digital-convert" in findings[0].message


# ── the label sets other repos depend on ─────────────────────────────────────


def test_line_category_originators_are_disjoint_and_cover_the_scheme():
    """The defect this registry exists to make visible (V-1 in docs/skos_strategy.md).

    If these sets ever overlap, `validate_labels(..., originator=...)` stops being able
    to say which tool a label came from.
    """
    alto = set(av.LINE_CATEGORY_ORIGINATORS["alto-postprocess"])
    digital = set(av.LINE_CATEGORY_ORIGINATORS["digital-convert"])

    assert alto & digital == set()
    assert alto | digital == set(av.labels_for("line-category"))
    assert {"Garbage", "Inverted"} == digital
    assert "Trash" in alto and "Trash" not in digital


def test_untrustworthy_set_spans_both_originators():
    """The one-line fix for V-1 depends on this: a DROP_CATEGORIES built from it must
    filter the OCR path as well as the digital-born one."""
    untrusted = set(av.UNTRUSTWORTHY_LINE_CATEGORIES)
    assert untrusted & set(av.LINE_CATEGORY_ORIGINATORS["alto-postprocess"])
    assert untrusted & set(av.LINE_CATEGORY_ORIGINATORS["digital-convert"])


def test_page_categories_match_the_scheme():
    assert set(av.PAGE_CATEGORIES) == set(av.labels_for("page-category"))
    assert len(av.PAGE_CATEGORIES) == 11


def test_labels_for_raises_on_an_unknown_scheme():
    """A typo in calling code, not a data condition. Returning () would turn every
    downstream validation into a no-op that reports success."""
    with pytest.raises(KeyError):
        av.labels_for("nope")


# ── serialisation ────────────────────────────────────────────────────────────


def test_both_serialisations_render_the_same_triples():
    """They did not, once: the Turtle heslar collections carried an rdfs:comment the
    JSON-LD ones did not, so the same registry described a 507- and a 480-triple graph.
    Both now render `triples()` and nothing else."""
    turtle = av.to_turtle()
    doc = json.loads(av.to_jsonld())

    subjects = {s for s, _p, _o in av.triples()}
    assert {node["@id"] for node in doc["@graph"]} == subjects
    for subject in subjects:
        assert av._curie(subject) in turtle or f"<{subject}>" in turtle


def test_serialisation_is_byte_stable():
    """`vocab_build.py --check` compares bytes. A non-deterministic serialiser would
    report drift on every run, which is how a drift gate gets switched off."""
    assert av.to_turtle() == av.to_turtle()
    assert av.to_jsonld() == av.to_jsonld()


def test_no_exact_match_is_asserted_between_atrium_labels():
    """`Trash` and `Garbage` are closeMatch, deliberately.

    One is an OCR judgement over a rendered image, the other a decode-sanity judgement
    over an embedded text layer. exactMatch is transitive and would license inferences
    neither tool supports.
    """
    props = {prop for _s, prop, _o in av.MAPPINGS}
    assert "skos:exactMatch" not in props
    assert (
        av.concept_uri("line-category", "Trash"),
        "skos:closeMatch",
        av.concept_uri("line-category", "Garbage"),
    ) in av.MAPPINGS


def test_mapping_targets_are_declared_concepts_or_known_external_namespaces():
    external = (av.NS["teater"], av.NS["amcr"], av.NS["aat"], f"{av.SKOS_BASE}heslar/")
    known = {av.concept_uri(s, n) for s in av.CONCEPTS for n in av.CONCEPTS[s]}

    for subject, _prop, obj in av.MAPPINGS:
        for node in (subject, obj):
            assert node in known or node.startswith(external), node


def test_selftest_passes():
    assert av._selftest(stream=io.StringIO()) == 0
