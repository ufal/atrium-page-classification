"""Tests for atrium_rocrate.py — the shared RO-Crate exporter.

Canonical copy lives in the hub at docs/templates/shared/test_atrium_rocrate.py and is
vendored verbatim into every tool repo, like test_document_originators.py next to
atrium_document.py. The module it covers shipped without one, which is how 319 untested
statements reached five repos at once and dropped atrium-translator's coverage ratchet from
82.87% to 72.90% (ufal/atrium-translator run 34380233914).

The module carries its own `_selftest()`. These tests drive it and then assert the
properties its docstrings promise but the selftest does not check. Since atrium-project#71 they
also pin the agreed baseline: RO-Crate 1.2 with Process Run Crate 0.5, stable run and tool ids,
`version` for tools, the authors as each tool's `creator`, the fragment, and the CreateAction every
service returns. Conformance itself is the validator's, in the hub's CI (tools/ci/rocrate_check.py).
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
        assert conforms == [rc.ROCRATE_CONFORMS_TO]

    def test_it_is_ro_crate_1_2_with_process_run_crate_0_5_on_the_root(self, crate):
        """RO-Crate 1.2 puts the specification on the descriptor and the profiles on the root,
        each profile described as a `Profile` entity (the validator's 16.1 MUST)."""
        assert rc.ROCRATE_CONTEXT == "https://w3id.org/ro/crate/1.2/context"
        assert crate["@context"][0] == rc.ROCRATE_CONTEXT
        by_id = _by_id(crate)
        assert by_id[rc.ROOT_ID]["conformsTo"] == {"@id": "https://w3id.org/ro/wfrun/process/0.5"}
        assert "Profile" in by_id[rc.PROCESS_RUN_PROFILE]["@type"]

    def test_the_authors_are_each_tools_creator_not_the_roots_author(self, record):
        crate = rc.document_crate(record, paradata=rc._sample_paradata())
        by_id = _by_id(crate)
        assert "author" not in by_id[rc.ROOT_ID]
        tools = [e for e in crate["@graph"] if e.get("@type") == "SoftwareApplication"]
        assert tools
        for tool in tools:
            assert {c["@id"] for c in tool["creator"]} == {a["orcid"] for a in rc.AUTHORS}
            assert by_id[tool["creator"][0]["@id"]]["@type"] == "Person"

    def test_tools_carry_version_and_a_release_id(self, record):
        crate = rc.document_crate(record, paradata=rc._sample_paradata())
        tool = _by_id(crate)["https://github.com/ufal/atrium-alto-postprocess/releases/tag/v1.6.0-beta"]
        assert tool["version"] == "1.6.0-beta"
        assert "softwareVersion" not in tool
        assert tool["url"] == "https://github.com/ufal/atrium-alto-postprocess"

    def test_a_tool_without_paradata_is_identified_by_its_repository(self, crate):
        """RO-Crate 1.2 requires a `version`: without paradata it says so instead of guessing."""
        tool = _by_id(crate)["https://github.com/ufal/atrium-nlp-enrich"]
        assert tool["version"] == rc.UNRECORDED_VERSION

    def test_runs_are_identified_by_their_run_uuid(self, crate):
        for run_uuid in rc._SAMPLE_RUN_UUIDS.values():
            assert _by_id(crate)[run_uuid]["@type"] == "CreateAction"

    def test_a_record_written_before_run_uuid_keeps_its_legacy_run_id(self, record):
        for contributor in record["provenance"]["contributors"]:
            contributor.pop("run_uuid")
        for stamp in record["assembled"]["blocks"].values():
            stamp.pop("run_uuid")
        assert "#run-alto-postprocess-260724-101112" in _by_id(rc.document_crate(record))

    def test_action_times_are_in_the_profiles_format(self, record):
        crate = rc.document_crate(record, paradata=rc._sample_paradata())
        action = _by_id(crate)[rc._SAMPLE_RUN_UUIDS["alto-postprocess"]]
        assert action["startTime"] == "2026-07-24T10:11:12+00:00"
        assert action["endTime"] == "2026-07-24T10:11:40+00:00"
        assert action["actionStatus"] == rc.COMPLETED_ACTION_STATUS

    def test_paradata_adds_the_image_and_the_agent(self, record):
        crate = rc.document_crate(record, paradata=rc._sample_paradata())
        by_id = _by_id(crate)
        action = by_id[rc._SAMPLE_RUN_UUIDS["alto-postprocess"]]
        image = by_id[action["containerImage"]["@id"]]
        assert image["registry"] == "ghcr.io" and image["tag"] == "1.6.0-beta"
        assert image["additionalType"] == {"@id": rc.DOCKER_IMAGE_TYPE}
        assert by_id[action["agent"]["@id"]]["@type"] == "Organization"

    def test_blocks_are_attributed_by_the_run_that_wrote_them(self, crate):
        """`creator` is for people: a block's maker is the action listing it as `result`."""
        by_id = _by_id(crate)
        assert "creator" not in by_id["#block-pages"]
        results = by_id[rc._SAMPLE_RUN_UUIDS["alto-postprocess"]]["result"]
        assert {"@id": "#block-pages"} in results

    def test_a_stamp_without_a_contributor_still_gets_its_run(self, record):
        record["provenance"]["contributors"] = []
        by_id = _by_id(rc.document_crate(record))
        action = by_id[rc._SAMPLE_RUN_UUIDS["nlp-enrich"]]
        assert {"@id": "#block-entities"} in action["result"]

    def test_the_archive_digest_is_kept_on_the_original(self, crate):
        assert _by_id(crate)[rc.SOURCE_ID]["sha512"] == "b" * 128

    def test_content_size_comes_from_the_files_present(self, record, tmp_path):
        target = tmp_path / "TEITOK" / "CTX000000001.teitok.xml"
        target.parent.mkdir()
        target.write_bytes(b"<TEI/>")
        crate = rc.document_crate(record, data_dir=str(tmp_path))
        by_id = _by_id(crate)
        assert by_id["TEITOK/CTX000000001.teitok.xml"]["contentSize"] == "6"
        assert "contentSize" not in by_id["paradata/260724-101112_pipeline-run.json"]


class TestFragment:
    def test_a_fragment_has_no_descriptor_root_or_data_entity(self, record):
        frag = rc.document_crate(record, fragment=True)
        ids = {e["@id"] for e in frag["@graph"]}
        assert rc.METADATA_FILENAME not in ids and rc.ROOT_ID not in ids
        assert not [e for e in frag["@graph"] if e.get("@type") == "File"]

    def test_source_id_points_the_runs_at_the_hosts_original(self, record):
        frag = rc.document_crate(record, fragment=True, source_id="orig")
        by_id = {e["@id"]: e for e in frag["@graph"]}
        assert rc.SOURCE_ID not in by_id
        assert by_id[rc._SAMPLE_RUN_UUIDS["nlp-enrich"]]["object"] == {"@id": "orig"}

    def test_a_wrapped_fragment_is_a_crate_whose_root_mentions_every_run(self, record):
        frag = rc.document_crate(record, fragment=True)
        crate = rc.wrap_fragment(
            frag, name="stub", description="stub", license="https://x/licence", date_published="2026-07-24"
        )
        root = _by_id(crate)[rc.ROOT_ID]
        assert crate["@graph"][0]["@id"] == rc.METADATA_FILENAME
        assert set(rc._SAMPLE_RUN_UUIDS.values()) <= {m["@id"] for m in root["mentions"]}
        assert root["isBasedOn"] == {"@id": rc.SOURCE_ID}
        known = set(_by_id(crate))
        for ref in set(_refs(crate["@graph"])):
            assert ref in known or ref.startswith(("http://", "https://")), ref


class TestCreateAction:
    @pytest.fixture
    def action(self):
        return rc.create_action(
            rc._sample_paradata()[0],
            inputs=[rc.file_entity("x.alto.xml", b"<alto/>", media_type="application/alto+xml"), rc.record_entity("C-1")],
            outputs=rc.block_entities(["pages", "lines"]),
        )

    def test_the_action_meets_the_shared_contract(self, action):
        assert rc.action_problems(action) == []

    def test_its_id_is_the_runs_uuid_and_the_instrument_is_the_released_tool(self, action):
        assert action["@id"] == rc._SAMPLE_RUN_UUIDS["alto-postprocess"]
        assert action["instrument"]["@id"].endswith("/releases/tag/v1.6.0-beta")
        assert action["instrument"]["creator"][0]["@type"] == "Person"

    def test_inputs_are_content_addressed(self, action):
        upload = action["object"][0]
        assert upload["@id"].startswith("ni:///sha-256;")
        assert upload["contentSize"] == "7" and len(upload["sha256"]) == 64

    def test_the_paradata_record_travels_whole(self, action):
        assert action["paradataRecord"]["run_id"] == "260724-101112"

    def test_a_failed_run_carries_its_error(self):
        failed = rc.create_action(rc._sample_paradata()[0], status="failed", error="boom")
        assert failed["actionStatus"] == rc.FAILED_ACTION_STATUS and failed["error"] == "boom"
        with pytest.raises(ValueError):
            rc.create_action({}, status="partly")

    def test_no_agent_is_invented(self):
        record = dict(rc._sample_paradata()[0], run_agent="")
        assert "agent" not in rc.create_action(record, outputs=rc.block_entities(["pages"]))

    def test_action_problems_names_what_is_missing(self, action):
        broken = dict(action, endTime="2026-07-24T10:11:40.250000+00:00", actionStatus="done")
        broken["instrument"] = dict(action["instrument"], softwareVersion="1")
        problems = " | ".join(rc.action_problems(broken))
        for word in ("endTime", "actionStatus", "softwareVersion"):
            assert word in problems

    def test_the_action_flattens_into_a_fragment(self, action):
        frag = rc.action_fragment(action)
        by_id = {e["@id"]: e for e in frag["@graph"]}
        flat = by_id[action["@id"]]
        assert flat["instrument"] == {"@id": action["instrument"]["@id"]}
        assert isinstance(flat["paradataRecord"], str)  # a crate is flat; the record is its JSON text
        assert by_id["https://orcid.org/0009-0002-4773-2797"]["@type"] == "Person"

    def test_blocks_written_reads_the_runs_stamps(self, record):
        assert rc.blocks_written(record, rc._SAMPLE_RUN_UUIDS["alto-postprocess"]) == ["content", "pages"]
        assert rc.blocks_written(record, run_id="260724-101500", program="nlp-enrich") == ["derived_from", "entities"]


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

    def test_fragment_mode_wraps_for_validation(self, tmp_path, record):
        src = self._record_file(tmp_path, record)
        out = tmp_path / "frag"
        assert rc._cli(["--document", src, "--fragment", "--wrap", "--out-dir", str(out)]) == 0
        crate = json.loads((out / rc.METADATA_FILENAME).read_text(encoding="utf-8"))
        assert _by_id(crate)[rc.ROOT_ID]["license"] == {"@id": record["provenance"]["license_url"]}

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
