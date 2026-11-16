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
    """'Is this label legal' and 'is this label legal FROM THIS TOOL' are different questions:
    the decode verdicts are digital-convert's, and line quality is the record builder's."""
    alto = {"originator": "alto-postprocess", "report": False}
    digital = {"originator": "digital-convert", "report": False}

    assert av.validate_labels("line-category", ["Trash", "Clear"], **alto) == []
    assert av.validate_labels("decode-verdict", ["garbled", "inverted"], **digital) == []

    findings = av.validate_labels("decode-verdict", ["garbled"], **alto)
    assert len(findings) == 1
    assert "belongs to digital-convert" in findings[0].message
    assert len(av.validate_labels("line-category", ["Garbage"], **alto)) == 1


def test_a_successor_program_emits_what_its_predecessor_emitted():
    """ocr-postprocess is alto-postprocess renamed (2026-10-01): same labels, no second row in
    LINE_CATEGORY_ORIGINATORS (that would make every label 'emitted by two tools')."""
    ocr = {"originator": "ocr-postprocess", "report": False}

    assert "ocr-postprocess" not in av.LINE_CATEGORY_ORIGINATORS
    assert av.validate_labels("line-category", ["Trash", "Clear"], **ocr) == []
    findings = av.validate_labels("decode-verdict", ["garbled"], **ocr)
    assert len(findings) == 1 and "belongs to digital-convert" in findings[0].message


# ── the label sets other repos depend on ─────────────────────────────────────


def test_line_quality_and_decode_verdicts_are_two_schemes():
    """The defect this registry made visible (V-1 in docs/skos_strategy.md), closed for good:
    line quality and the decode verdict are apart (D18 rules 2 and 3, atrium-project#84).
    `line-category` is the record builder's five values; `decode-verdict` is digital-convert's."""
    assert set(av.LINE_CATEGORY_ORIGINATORS) == {"alto-postprocess"}
    assert set(av.labels_for("line-category")) == set(av.RECORD_LINE_CATEGORIES)
    assert set(av.labels_for("decode-verdict")) == {"garbled", "inverted"}
    assert set(av.DECODE_VERDICT_ORIGINATORS) == {"digital-convert"}
    assert not {"Garbage", "Inverted"} & set(av.labels_for("line-category"))


def test_record_categories_and_decode_flags():
    """`lines[].categ` holds the five record categories only; digital-convert's decode verdicts
    are `decode:`-prefixed flags on docproc lines, and an older docproc's spelling maps onto
    them. The untrustworthy set is drawn from the record categories (V-1 stays fixed)."""
    assert set(av.RECORD_LINE_CATEGORIES) == {"Clear", "Empty", "Noisy", "Non-text", "Trash"}
    assert av.DECODE_FLAGS == ("decode:garbled", "decode:inverted")
    assert av.LEGACY_DECODE_FLAGS == {"Garbage": "decode:garbled", "Inverted": "decode:inverted"}
    assert not set(av.DECODE_FLAGS) & set(av.RECORD_LINE_CATEGORIES)
    assert av.UNTRUSTWORTHY_LINE_CATEGORIES == ("Trash",)
    assert set(av.UNTRUSTWORTHY_LINE_CATEGORIES) <= set(av.RECORD_LINE_CATEGORIES)


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


def test_no_mapping_joins_line_quality_and_the_decode_verdict():
    """No exactMatch between ATRIUM labels, and, since the two systems are apart (D18 rules 2
    and 3), no mapping at all between a line category and a decode verdict: the Trash
    closeMatch Garbage of 1.x is gone."""
    props = {prop for _s, prop, _o in av.MAPPINGS}
    assert "skos:exactMatch" not in props
    decode = {av.concept_uri("decode-verdict", v) for v in av.labels_for("decode-verdict")}
    line = {av.concept_uri("line-category", v) for v in av.labels_for("line-category")}
    for subject, _prop, obj in av.MAPPINGS:
        assert not ({subject, obj} & decode and {subject, obj} & line), (subject, obj)


def test_mapping_targets_are_declared_concepts_or_known_external_namespaces():
    external = (av.NS["teater"], av.NS["amcr"], av.NS["aat"], f"{av.SKOS_BASE}heslar/")
    known = {av.concept_uri(s, n) for s in av.CONCEPTS for n in av.CONCEPTS[s]}

    for subject, _prop, obj in av.MAPPINGS:
        for node in (subject, obj):
            assert node in known or node.startswith(external), node


def test_no_record_field_is_read_as_a_theme():
    """#51 V-5 (decided 2026-10-22): `teater_category` holds the AMČR/TEATER source label, so the
    theme scheme names no record field, and no scheme claims `teater_category`."""
    assert av.CONCEPT_SCHEMES["theme"]["fields"] == ()
    for scheme in av.CONCEPT_SCHEMES.values():
        assert not any("teater_category" in field for field in scheme["fields"])


def test_selftest_passes():
    assert av._selftest(stream=io.StringIO()) == 0
