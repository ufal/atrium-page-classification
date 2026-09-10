"""Tests for atrium_rocrate.py — the shared RO-Crate exporter.

Canonical copy lives in the hub at docs/templates/shared/test_atrium_rocrate.py and is
vendored verbatim into every tool repo, like test_document_originators.py next to
atrium_document.py. The module it covers shipped without one, which is how 319 untested
statements reached five repos at once and dropped atrium-translator's coverage ratchet from
82.87% to 72.90% (ufal/atrium-translator run 34380233914).

The module carries its own `_selftest()`. These tests drive it and then assert the
properties its docstrings promise but the selftest does not check.
"""

import json
import os

import pytest

import atrium_rocrate as rc


@pytest.fixture
def record():
    return rc._sample_record()


@pytest.fixture
def crate(record):
    return rc.document_crate(record)


def _by_id(crate):
    return {e["@id"]: e for e in crate["@graph"]}


def _refs(node):
    """Every {"@id": ...} reference anywhere under `node`."""
    if isinstance(node, dict):
        if set(node) == {"@id"}:
            yield node["@id"]
        else:
            for value in node.values():
                yield from _refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _refs(value)


class TestSelfTest:
    def test_selftest_passes_on_the_sample_record(self):
        """The module's own structural checks, run as a test rather than by hand."""
        import io

        stream = io.StringIO()
        assert rc._selftest(stream) == 0, stream.getvalue()


class TestDocumentCrate:
    def test_metadata_descriptor_comes_first_and_points_at_the_root(self, crate):
        first = crate["@graph"][0]
        assert first["@id"] == rc.METADATA_FILENAME
        assert first["about"] == {"@id": rc.ROOT_ID}

    def test_root_is_a_dataset_with_the_mandatory_properties(self, crate):
        root = _by_id(crate)[rc.ROOT_ID]
        assert root["@type"] == "Dataset"
        for prop in ("name", "description", "datePublished", "license"):
            assert root.get(prop), f"root is missing {prop!r}"

    def test_no_reference_dangles(self, crate):
        known = set(_by_id(crate))
        for ref in sorted(set(_refs(crate["@graph"]))):
            assert ref in known or ref.startswith(("http://", "https://")), (
                f"{ref!r} is neither an entity in this crate nor an absolute URI"
            )

    def test_regenerable_recipes_are_never_parts(self, crate):
        """Reference discipline: a Markdown rendering or page image is a recipe, not a
        stored file, so it must not appear in hasPart even though it is described."""
        parts = {r["@id"] for r in _by_id(crate)[rc.ROOT_ID].get("hasPart", [])}
        assert not [p for p in parts if p.startswith("#regenerable-")]

    def test_persistent_step_outputs_are_parts(self, crate):
        parts = {r["@id"] for r in _by_id(crate)[rc.ROOT_ID].get("hasPart", [])}
        assert "TEITOK/CTX000000001.teitok.xml" in parts

    def test_the_archive_managed_original_is_not_a_part(self, crate):
        """`source` is described so the crate says what it came from, but the original is
        the archive's, not the crate's, so listing it as a part would claim otherwise."""
        parts = {r["@id"] for r in _by_id(crate)[rc.ROOT_ID].get("hasPart", [])}
        assert "#source" not in parts

    def test_conforms_to_the_declared_profiles(self, crate):
        conforms = list(_refs(crate["@graph"][0].get("conformsTo", [])))
        assert rc.ROCRATE_CONFORMS_TO in conforms


class TestRunCrate:
    def test_run_crate_references_document_crates_rather_than_inlining_them(self, record):
        run = rc.run_crate([record])
        ids = set(_by_id(run))
        assert rc.ROOT_ID in ids
        assert any("CTX000000001" in i for i in ids)

    def test_run_crate_stays_the_same_size_as_documents_multiply(self, record):
        """The stated design property: a run crate references per-document crates, so a
        run over five thousand documents must not produce a graph five thousand times
        larger. Growth is allowed to be linear in documents but not in their contents."""
        one = rc.run_crate([record])
        many = rc.run_crate([record] * 4)
        per_doc_growth = len(many["@graph"]) - len(one["@graph"])
        assert per_doc_growth < len(rc.document_crate(record)["@graph"])


class TestSerialisation:
    def test_to_json_is_byte_stable(self, crate):
        assert rc.to_json(crate) == rc.to_json(crate)

    def test_to_json_ends_with_a_newline(self, crate):
        assert rc.to_json(crate).endswith("\n")

    def test_to_json_sorts_keys_so_diffs_are_reviewable(self, crate):
        parsed = json.loads(rc.to_json(crate))
        assert list(parsed) == sorted(parsed)

    def test_write_crate_leaves_no_temporary_file(self, crate, tmp_path):
        """Write-then-rename: a crash mid-write must leave no half-written metadata that
        a consumer would read as corrupt rather than as missing."""
        path = rc.write_crate(crate, str(tmp_path))
        assert os.path.basename(path) == rc.METADATA_FILENAME
        assert [p.name for p in tmp_path.iterdir()] == [rc.METADATA_FILENAME]
        assert json.loads(open(path, encoding="utf-8").read())["@graph"]

    def test_write_crate_creates_the_directory(self, crate, tmp_path):
        out = tmp_path / "nested" / "deeper"
        rc.write_crate(crate, str(out))
        assert (out / rc.METADATA_FILENAME).is_file()


class TestEncodingFormat:
    @pytest.mark.parametrize("path", ["TEITOK/x.teitok.xml", "x.conllu", "x.alto.xml"])
    def test_known_pipeline_suffixes_get_a_media_type(self, path):
        assert rc._encoding_format(path)

    def test_unknown_suffix_returns_none_so_prune_omits_it(self):
        """mimetypes would guess; this module deliberately answers only for the closed set
        of suffixes the pipeline actually writes."""
        assert rc._encoding_format("x.zzz") is None


class TestCli:
    def _record_file(self, tmp_path, record):
        path = tmp_path / "CTX000000001.document.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return str(path)

    def test_selftest_mode(self):
        assert rc._cli(["--selftest"]) == 0

    def test_document_mode_writes_a_crate(self, tmp_path, record):
        src = self._record_file(tmp_path, record)
        out = tmp_path / "out"
        assert rc._cli(["--document", src, "--out-dir", str(out)]) == 0
        assert (out / rc.METADATA_FILENAME).is_file()

    def test_document_mode_without_out_dir_writes_to_stdout(self, tmp_path, record, capsys):
        src = self._record_file(tmp_path, record)
        assert rc._cli(["--document", src]) == 0
        assert json.loads(capsys.readouterr().out)["@graph"]

    def test_run_mode_accepts_several_records(self, tmp_path, record):
        src = self._record_file(tmp_path, record)
        out = tmp_path / "run"
        assert rc._cli(["--run", src, src, "--out-dir", str(out)]) == 0
        assert (out / rc.METADATA_FILENAME).is_file()

    def test_no_mode_is_a_usage_error(self):
        """argparse exits 2 rather than producing an empty crate."""
        with pytest.raises(SystemExit) as excinfo:
            rc._cli([])
        assert excinfo.value.code == 2
