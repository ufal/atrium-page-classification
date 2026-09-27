"""tests/test_atrium_limits.py — the limits contract (atrium-project#53, factor III).

Canonical copy: hub ``docs/templates/shared/test_atrium_limits.py``, vendored
byte-identically to every tool repo's ``tests/`` (MANIFEST.json, para-drift). It tests
``atrium_limits.py`` itself, the paradata side of ``limits_applied``, and — in a tool repo
only — that no library code imports the FastAPI-bound ``service/atrium_service.py``.

Variable names are built at run time (``_name()``), never written as literals in a call:
``tests/`` is outside the env-contract scan, but a literal here would read as a real
setting to anyone grepping for one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import atrium_limits as al
import atrium_paradata as ap

REPO_ROOT = Path(__file__).resolve().parent.parent


def _name(suffix: str) -> str:
    return "_".join(("ATRIUM", "LIMITS", "TEST", suffix))


def test_selftest_passes(capsys):
    al._selftest()
    assert "OK" in capsys.readouterr().out


def test_default_env_and_config_precedence(monkeypatch):
    env = _name("PAGES")
    monkeypatch.delenv(env, raising=False)
    spec = al.limit(env, 50, unit="pages")
    assert spec.key == env.lower()
    assert (spec.get(), spec.source()) == (50, "default")
    assert (spec.get(config="20"), spec.source(config="20")) == (20, "config")
    monkeypatch.setenv(env, "30")
    assert (spec.get(config="20"), spec.source(config="20")) == (30, "env")
    monkeypatch.setenv(env, "   ")
    assert spec.get() == 50, "a blank value counts as unset (env template Rule 2)"


def test_value_is_read_per_call(monkeypatch):
    env = _name("PERCALL")
    monkeypatch.delenv(env, raising=False)
    spec = al.limit(env, 4, unit="retries")
    assert spec.get() == 4
    monkeypatch.setenv(env, "9")
    assert spec.get() == 9


@pytest.mark.parametrize("bad", ["ten", "2.5", "-3", "nan", "inf"])
def test_malformed_value_fails_at_declaration_naming_the_variable(monkeypatch, bad):
    env = _name("BAD")
    monkeypatch.setenv(env, bad)
    with pytest.raises(al.LimitConfigError, match=env):
        al.limit(env, 5, unit="pages")


def test_float_limits_and_bounds(monkeypatch):
    env = _name("MB")
    monkeypatch.setenv(env, "2.5")
    assert al.limit(env, 10, unit="MB", kind=float).get() == 2.5
    monkeypatch.setenv(env, "0")
    with pytest.raises(al.LimitConfigError):
        al.limit(env, 10, unit="MB", kind=float, minimum=0.001)
    assert al.limit(env, 10, unit="MB", kind=float, minimum=0.001, zero_means_unlimited=True).is_unlimited()
    monkeypatch.setenv(env, "11")
    with pytest.raises(al.LimitConfigError):
        al.limit(env, 10, unit="MB", maximum=10)


def test_check_raises_limit_exceeded_not_a_value_error(monkeypatch):
    env = _name("WORDS")
    monkeypatch.delenv(env, raising=False)
    spec = al.limit(env, 100, unit="words", status=413)
    spec.check(100)
    with pytest.raises(al.LimitExceeded) as info:
        spec.check(101)
    exc = info.value
    assert not isinstance(exc, ValueError), "an `except ValueError` must not be able to swallow it"
    assert exc.http_status == 413
    assert exc.to_dict() == {"key": env.lower(), "env": env, "value": 100, "observed": 101, "unit": "words"}
    assert env in str(exc), "the default detail names the setting to change"


def test_status_follows_the_cause():
    for status in (413, 422, 504):
        assert al.limit(_name("S"), 1, unit="s", status=status).exceeded(2).http_status == status
    with pytest.raises(ValueError):
        al.limit(_name("S"), 1, unit="s", status=500)


def test_upload_limit_keeps_the_legacy_bytes_fallback(monkeypatch):
    monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(3 * 1024 * 1024))
    spec = al.upload_limit(10)
    assert spec.key == "max_upload_mb"
    assert spec.get() == 3.0
    monkeypatch.setenv("MAX_UPLOAD_MB", "7")
    assert spec.get() == 7.0
    monkeypatch.delenv("MAX_UPLOAD_MB")
    monkeypatch.delenv("MAX_UPLOAD_BYTES")
    assert spec.get() == 10.0


def test_limit_set_values_meta_and_derived(monkeypatch):
    a, b = _name("A"), _name("B")
    monkeypatch.delenv(a, raising=False)
    monkeypatch.setenv(b, "8")
    limits = al.LimitSet(al.limit(a, 1, unit="pages", key="max_a"), al.limit(b, 2, unit="s"))
    limits.derived("window", lambda: 512, unit="tokens", derived_from=[a])
    assert limits.values() == {"max_a": 1, b.lower(): 8, "window": 512}
    meta = limits.meta()
    assert meta["max_a"] == {"env": a, "unit": "pages", "default": 1, "source": "default"}
    assert meta[b.lower()]["source"] == "env"
    assert meta["window"]["source"] == "derived" and meta["window"]["env"] is None
    with pytest.raises(ValueError):
        limits.add(al.limit(a, 3, unit="pages", key="max_a"))


def test_limit_set_config_provider(monkeypatch):
    env = _name("CFG")
    monkeypatch.delenv(env, raising=False)
    limits = al.LimitSet(al.limit(env, 256, unit="MB", kind=float), config=lambda: {env: "64"})
    assert limits.get(env.lower()) == 64.0
    assert limits.meta()[env.lower()]["source"] == "config"
    monkeypatch.setenv(env, "32")
    assert limits.get(env.lower()) == 32.0


def test_limit_notes_merge_and_header():
    notes = al.LimitNotes()
    notes.note("lang_id_segment_chars", "sampled", 3, value=2000)
    notes.note("lang_id_segment_chars", "sampled", 2, detail="first 2000 characters of each segment")
    notes.note("vocab_terms", "trimmed", value=3462, detail="1257 of 4719 terms left out — příliš")
    assert notes.as_list()[0] == {
        "limit": "lang_id_segment_chars",
        "value": 2000,
        "effect": "sampled",
        "count": 5,
        "detail": "first 2000 characters of each segment",
    }
    header = notes.header_summary()
    header.encode("latin-1")  # a Starlette header value must encode as latin-1
    assert header == "lang_id_segment_chars=sampled:5; vocab_terms=trimmed:1"
    with pytest.raises(ValueError):
        notes.note("x", "truncated")


def test_limit_notes_round_trip_through_json():
    notes = al.LimitNotes()
    notes.note("ppl_max_tokens", "trimmed", 4, value=1024)
    carried = json.loads(json.dumps(notes.as_list()))  # e.g. back from a subprocess
    again = al.LimitNotes(carried)
    again.extend(notes)
    assert again.as_list()[0]["count"] == 8


def test_paradata_effects_match():
    assert ap._LIMIT_EFFECTS == al.EFFECTS


def test_paradata_records_and_merges_limits_applied(tmp_path):
    first = ap.ParadataLogger(program="stage-a", config={}, paradata_dir=str(tmp_path))
    notes = al.LimitNotes()
    notes.note("keybert_chunk_words", "trimmed", 2, value=400)
    first.note_limits(notes)
    first.note_limit("keybert_chunk_words", 400, "trimmed", 1)
    with pytest.raises(ValueError):
        first.note_limit("x", 1, "cut")
    path_a = first.finalize()
    second = ap.ParadataLogger(program="stage-b", config={}, paradata_dir=str(tmp_path))
    path_b = second.finalize()

    record_a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    assert record_a["limits_applied"] == [
        {"limit": "keybert_chunk_words", "value": 400, "effect": "trimmed", "count": 3, "detail": ""}
    ]
    assert json.loads(Path(path_b).read_text(encoding="utf-8"))["limits_applied"] == []

    merged = json.loads(Path(ap.merge_run_paradata([path_a, path_b], str(tmp_path / "run.json"))).read_text())
    assert merged["limits_applied"] == [{"program": "stage-a", **record_a["limits_applied"][0]}]
    single = json.loads(
        Path(ap.merge_paradata_files([path_a, path_b], "in.xml", str(tmp_path / "one.json"))).read_text()
    )
    assert single["limits_applied"] == merged["limits_applied"]


def test_paradata_state_round_trip_keeps_notes(tmp_path):
    logger = ap.ParadataLogger(program="stage", config={}, paradata_dir=str(tmp_path))
    logger.note_limit("word_chunk_limit", 900, "split", 4)
    restored = ap.ParadataLogger._from_state_dict(json.loads(json.dumps(logger._to_state_dict())))
    assert restored.limits_applied == logger.limits_applied


_SERVICE_IMPORT = re.compile(r"^\s*(?:from|import)\s+(?:service\.)?atrium_service\b", re.MULTILINE)
_SKIP_PARTS = {"tests", ".git", "node_modules", "site-packages", "dist-packages", "__pycache__"}


def test_library_code_does_not_import_the_service_module():
    """``service/atrium_service.py`` imports FastAPI, which the nlp-enrich and llm-enrich
    CLI images do not install. Limits for library code come from ``atrium_limits``."""
    if not (REPO_ROOT / "service" / "atrium_service.py").is_file():
        pytest.skip("not a tool repo (the hub's docs/templates/shared/ has no service/)")
    offenders = []
    for py in sorted(REPO_ROOT.rglob("*.py")):
        rel = py.relative_to(REPO_ROOT)
        if rel.parts[0] == "service" or _SKIP_PARTS & set(rel.parts):
            continue
        if any(parent.joinpath("pyvenv.cfg").exists() for parent in py.parents if REPO_ROOT in parent.parents):
            continue
        if _SERVICE_IMPORT.search(py.read_text(encoding="utf-8", errors="replace")):
            offenders.append(str(rel))
    assert not offenders, f"import atrium_limits, not atrium_service, outside service/: {offenders}"
