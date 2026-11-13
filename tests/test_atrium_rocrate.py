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

Since doc-schema-v2 (atrium-project#84) the record carries no run data, so a crate's runs come from
their paradata 3.0 records (or the services' actions): inputs and outputs by distribution name and
SHA-512, the record's result naming the blocks and fields written, and no licence on any root (D2).
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
    return rc.document_crate(record, paradata=rc._sample_paradata())


@pytest.fixture
def bare(record):
    """The record's crate with no run described."""
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
        for prop in ("name", "description", "datePublished"):
            assert root.get(prop), f"root is missing {prop!r}"
        assert "license" not in root  # nobody asserted one (D2)

    def test_no_reference_dangles(self, crate):
        known = set(_by_id(crate))
        for ref in sorted(set(_refs(crate["@graph"]))):
            assert ref in known or ref.startswith(("http://", "https://")), (
                f"{ref!r} is neither an entity in this crate nor an absolute URI"
            )

    def test_the_record_contributes_no_data_or_block_entities(self, crate):
        """Record schema 2.0 holds no `derived_from`, `regenerable` or stamps: the crate lists no
        files of the record and no `#block-` entities; what a run wrote is its result."""
        assert "hasPart" not in _by_id(crate)[rc.ROOT_ID]
        ids = set(_by_id(crate))
        assert not [i for i in ids if i.startswith(("#block-", "#regenerable-"))]

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
            assert {c["@id"] for c in tool["creator"]} == {a["orcid"] for a in rc.authors_of(tool["alternateName"])}
            assert all(by_id[c["@id"]]["@type"] == "Person" for c in tool["creator"])

    def test_each_tool_names_its_own_authors(self):
        """Who wrote a tool is its CITATION.cff, not the union of every tool's (atrium-project#13)."""
        names = {program: [a["name"] for a in rc.authors_of(program)] for program in rc.PROGRAM_AUTHORS}
        assert names["page-classification"] == ["Kateryna Lutsai", "Dana Křivánková", "David Novák", "Pavel Straňák"]
        assert "David Spáčil" in names["ocr-postprocess"] and "David Spáčil" in names["keyword-extract"]
        assert names["translator"] == names["nlp-enrich"] == names["digital-convert"]
        assert "Dana Křivánková" not in names["translator"]
        assert rc.authors_of("translator-api") == rc.authors_of("translator")
        assert rc.authors_of("alto-postprocess") == rc.authors_of("ocr-postprocess")
        assert [a["orcid"] for a in rc.authors_of("llm-enrich")] == list(rc.CORE_AUTHORS)

    def test_a_crate_describes_only_the_people_of_its_tools(self):
        action = rc.create_action({"program": "translator-api", "tool_version": "v1.6.1-beta", "run_id": "r"})
        assert [c["name"] for c in action["instrument"]["creator"]] == ["Kateryna Lutsai", "David Novák", "Pavel Straňák"]

    def test_tools_carry_version_and_a_release_id(self, crate):
        tool = _by_id(crate)["https://github.com/ufal/atrium-ocr-postprocess/releases/tag/v1.16.0-beta"]
        assert tool["version"] == "1.16.0-beta"
        assert "softwareVersion" not in tool
        assert tool["url"] == "https://github.com/ufal/atrium-ocr-postprocess"

    def test_a_run_without_a_recorded_version_is_identified_by_its_repository(self):
        """RO-Crate 1.2 requires a `version`: without one the tool says so instead of guessing."""
        tool = rc.create_action({"program": "nlp-enrich", "run_id": "r"})["instrument"]
        assert tool["@id"] == "https://github.com/ufal/atrium-nlp-enrich"
        assert tool["version"] == rc.UNRECORDED_VERSION

    def test_runs_are_identified_by_their_run_uuid(self, crate):
        for run_uuid in rc._SAMPLE_RUN_UUIDS.values():
            assert _by_id(crate)[run_uuid]["@type"] == "CreateAction"

    def test_a_run_without_run_uuid_keeps_its_legacy_run_id(self):
        action = rc.create_action({"program": "ocr-postprocess", "run_id": "260724-101112"})
        assert action["@id"] == "#run-ocr-postprocess-260724-101112"

    def test_action_times_are_in_the_profiles_format(self, crate):
        action = _by_id(crate)[rc._SAMPLE_RUN_UUIDS["ocr-postprocess"]]
        assert action["startTime"] == "2026-07-24T10:11:12+00:00"
        assert action["endTime"] == "2026-07-24T10:11:40+00:00"
        assert action["actionStatus"] == rc.COMPLETED_ACTION_STATUS

    def test_paradata_adds_the_image_and_the_agent(self, crate):
        by_id = _by_id(crate)
        action = by_id[rc._SAMPLE_RUN_UUIDS["ocr-postprocess"]]
        image = by_id[action["containerImage"]["@id"]]
        assert image["registry"] == "ghcr.io" and image["tag"] == "1.16.0-beta"
        assert image["additionalType"] == {"@id": rc.DOCKER_IMAGE_TYPE}
        assert by_id[action["agent"]["@id"]]["@type"] == "Organization"

    def test_the_record_result_names_the_blocks_and_fields_written(self, crate):
        """#84 §4: what a run wrote is its result's `recordFields`, not a stamp in the record."""
        by_id = _by_id(crate)
        results = [by_id[r["@id"]] for r in by_id[rc._SAMPLE_RUN_UUIDS["nlp-enrich"]]["result"]]
        record = [r for r in results if r["name"] == rc.RECORD_DISTRIBUTION]
        assert len(record) == 1 and json.loads(record[0]["recordFields"]) == {
            "entities": ["surface", "type_teitok", "type_cnec", "page", "line"]
        }
        assert {r["name"] for r in results} == {rc.RECORD_DISTRIBUTION, "atr/teitok-xml"}

    def test_one_runs_output_is_the_next_runs_input(self, crate):
        """Named by distribution and SHA-512, the record version ocr-postprocess wrote and the one
        nlp-enrich read are one entity."""
        by_id = _by_id(crate)
        wrote = {r["@id"] for r in by_id[rc._SAMPLE_RUN_UUIDS["ocr-postprocess"]]["result"]}
        read = {o["@id"] for o in by_id[rc._SAMPLE_RUN_UUIDS["nlp-enrich"]]["object"]}
        assert wrote & read

    def test_the_archive_digest_is_kept_on_the_original(self, crate):
        assert _by_id(crate)[rc.SOURCE_ID]["sha512"] == "b" * 128
        assert "sha256" not in _by_id(crate)[rc.SOURCE_ID]

    def test_a_crate_without_runs_is_complete_and_dated_by_the_epoch(self, bare):
        root = _by_id(bare)[rc.ROOT_ID]
        assert root["datePublished"] == "1970-01-01" and "mentions" not in root


class TestNoLicence:
    """D2 (atrium-project#84): no crate root names a licence the caller did not assert, and nothing
    in a crate computes one. AMČR states the document's licence on its own record crate."""

    def test_no_root_names_a_licence(self, crate, record):
        assert "license" not in _by_id(crate)[rc.ROOT_ID]
        assert "license" not in _by_id(rc.run_crate([record]))[rc.ROOT_ID]
        frag = rc.document_crate(record, fragment=True)
        stub = rc.wrap_fragment(frag, name="stub", description="stub", date_published="2026-07-24")
        assert "license" not in _by_id(stub)[rc.ROOT_ID]

    def test_nothing_in_a_crate_computes_a_licence(self, crate):
        text = rc.to_json(crate)
        for word in ("most restrictive", "effective_license", "no-licence-asserted", "license_detail"):
            assert word not in text

    def test_a_licence_the_caller_asserts_goes_on_the_root(self, record):
        url = "https://creativecommons.org/licenses/by-nc/4.0/"
        crate = rc.document_crate(record, license=url)
        assert _by_id(crate)[rc.ROOT_ID]["license"] == {"@id": url}
        assert url in _by_id(crate)
        stub = rc.wrap_fragment(rc.document_crate(record, fragment=True), name="s", description="s", license=url, date_published="2026-07-24")
        assert _by_id(stub)[rc.ROOT_ID]["license"] == {"@id": url}

    def test_a_paradata_record_with_a_calculated_licence_breaks_the_contract(self):
        pd = dict(rc._sample_paradata()[0], license="CC BY-NC 4.0", license_detail={})
        assert any("licence" in p for p in rc.action_problems(rc.action_from_paradata(pd)))


class TestFragment:
    def test_a_fragment_has_no_descriptor_root_or_data_entity(self, record):
        frag = rc.document_crate(record, fragment=True)
        ids = {e["@id"] for e in frag["@graph"]}
        assert rc.METADATA_FILENAME not in ids and rc.ROOT_ID not in ids
        assert not [e for e in frag["@graph"] if e.get("@type") == "File"]

    def test_source_id_replaces_the_crates_own_original(self, record):
        frag = rc.document_crate(record, paradata=rc._sample_paradata(), fragment=True, source_id="orig")
        by_id = {e["@id"]: e for e in frag["@graph"]}
        assert rc.SOURCE_ID not in by_id
        assert rc._SAMPLE_RUN_UUIDS["nlp-enrich"] in by_id

    def test_a_wrapped_fragment_is_a_crate_whose_root_mentions_every_run(self, record):
        frag = rc.document_crate(record, paradata=rc._sample_paradata(), fragment=True)
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
            inputs=[
                rc.file_entity("atr/docproc-json", b"{}", media_type="application/json"),
                rc.record_entity("C-1", b"{}"),
            ],
            outputs=[rc.record_result({"lines": ["page", "line", "text"], "pages": ["page", "page_index"]}, b"{\n}")],
        )

    def test_the_action_meets_the_shared_contract(self, action):
        assert rc.action_problems(action) == []

    def test_its_id_is_the_runs_uuid_and_the_instrument_is_the_released_tool(self, action):
        assert action["@id"] == rc._SAMPLE_RUN_UUIDS["ocr-postprocess"]
        assert action["instrument"]["@id"].endswith("/releases/tag/v1.16.0-beta")
        assert action["instrument"]["creator"][0]["@type"] == "Person"

    def test_inputs_are_named_by_distribution_and_sha512(self, action):
        """#84 §4: never an ni:///sha-256 URI or a local path."""
        upload = action["object"][0]
        assert upload["name"] == "atr/docproc-json" and len(upload["sha512"]) == 128
        assert upload["@id"] == "#atr/docproc-json@" + upload["sha512"][:16]
        assert upload["contentSize"] == "2" and "sha256" not in upload and "stored" not in upload
        assert action["object"][1]["identifier"] == "C-1"

    def test_an_input_that_is_not_stored_is_marked_and_still_named_by_its_digest(self):
        """D23: digital-convert's docproc of a mixed document is not stored."""
        entity = rc.file_entity("atr/docproc-json", sha512="1" * 128, stored=False)
        assert entity["stored"] is False and entity["sha512"] == "1" * 128

    def test_a_file_without_a_digest_or_a_name_is_refused(self):
        with pytest.raises(ValueError):
            rc.file_entity("atr/docproc-json")
        with pytest.raises(ValueError):
            rc.file_entity("", b"{}")

    def test_the_paradata_record_travels_whole(self, action):
        assert action["paradataRecord"]["run_id"] == "260724-101112"

    def test_a_failed_run_carries_its_error(self):
        failed = rc.create_action(rc._sample_paradata()[0], status="failed", error="boom")
        assert failed["actionStatus"] == rc.FAILED_ACTION_STATUS and failed["error"] == "boom"
        with pytest.raises(ValueError):
            rc.create_action({}, status="partly")

    def test_no_agent_is_invented(self):
        record = dict(rc._sample_paradata()[0], run_agent="")
        assert "agent" not in rc.create_action(record, outputs=[rc.record_result({"pages": ["page"]}, b"{}")])

    def test_action_problems_names_what_is_missing(self, action):
        broken = dict(action, endTime="2026-07-24T10:11:40.250000+00:00", actionStatus="done")
        broken["instrument"] = dict(action["instrument"], softwareVersion="1")
        problems = " | ".join(rc.action_problems(broken))
        for word in ("endTime", "actionStatus", "softwareVersion"):
            assert word in problems

    def test_action_problems_refuses_a_1x_result(self, action):
        """A `#block-` result or an ni: id names no distribution by SHA-512; a record result must
        name its fields."""
        old = dict(action, result=[{"@id": "ni:///sha-256;x", "@type": "CreativeWork", "name": "pages"}])
        problems = " | ".join(rc.action_problems(old))
        assert "ni:" in problems and "SHA-512" in problems
        bare = dict(action["result"][0])
        bare.pop("recordFields")
        assert any("fields" in p for p in rc.action_problems(dict(action, result=[bare])))

    def test_the_action_flattens_into_a_fragment(self, action):
        frag = rc.action_fragment(action)
        by_id = {e["@id"]: e for e in frag["@graph"]}
        flat = by_id[action["@id"]]
        assert flat["instrument"] == {"@id": action["instrument"]["@id"]}
        assert isinstance(flat["paradataRecord"], str)  # a crate is flat; the record is its JSON text
        result = by_id[action["result"][0]["@id"]]
        assert json.loads(result["recordFields"]) == {"lines": ["page", "line", "text"], "pages": ["page", "page_index"]}
        assert by_id["https://orcid.org/0009-0002-4773-2797"]["@type"] == "Person"

    def test_action_from_paradata_maps_inputs_outputs_and_status(self):
        pd = rc._sample_paradata()[0]
        action = rc.action_from_paradata(pd)
        assert [o["name"] for o in action["object"]] == ["atr/docproc-json", rc.RECORD_DISTRIBUTION]
        assert action["result"][0]["recordFields"]["pages"] == ["page", "page_index", "content_status"]
        assert rc.action_problems(action) == []
        unstored = dict(pd, inputs=[{"name": "atr/docproc-json", "sha512": "2" * 128, "stored": False}])
        assert rc.action_from_paradata(unstored)["object"][0]["stored"] is False
        failed = rc.action_from_paradata(dict(pd, action_status="failed", error="ATR failed"))
        assert failed["actionStatus"] == rc.FAILED_ACTION_STATUS and failed["error"] == "ATR failed"


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

    def test_a_failed_stage_is_a_failed_action(self, record):
        run = rc.run_crate(
            [record],
            run_paradata={"pipeline_stages": [{"order": 1, "program": "ocr-postprocess", "run_id": "r", "action_status": "failed"}]},
        )
        actions = [e for e in run["@graph"] if e.get("@type") == "CreateAction"]
        assert [a["actionStatus"] for a in actions] == [rc.FAILED_ACTION_STATUS]


class TestSourceConcepts:
    """atrium-project#51 V-5: `teater_category` holds an AMČR/TEATER label, not an ATRIUM theme, so
    the crate names the concepts in `teater_category_ids` by their own URIs and mints nothing."""

    AMCR = "https://api.aiscr.cz/id/HES-000021"
    TEATER = "https://teater.aiscr.cz/id/1333"

    @staticmethod
    def _crate(record, *items):
        return rc.document_crate(dict(record, enrichment={"items": list(items)}))

    def test_ids_become_the_sources_own_concepts(self, record):
        ids = [{"source": "amcr", "id": "HES-000021"}, {"source": "teater", "id": "1333"}]
        crate = self._crate(record, {"teater_category": "kostel", "teater_category_ids": ids})
        by_id = _by_id(crate)
        amcr = by_id[self.AMCR]
        assert (amcr["@type"], amcr["name"], amcr["termCode"]) == ("DefinedTerm", "kostel", "HES-000021")
        assert amcr["inDefinedTermSet"] == {"@id": rc.ATRIUM_BASE + "scheme/amcr-harvest"}
        assert by_id[self.TEATER]["inDefinedTermSet"] == {"@id": rc.ATRIUM_BASE + "scheme/teater-harvest"}
        assert by_id[rc.ATRIUM_BASE + "scheme/amcr-harvest"]["@type"] == "DefinedTermSet"
        assert {self.AMCR, self.TEATER} <= {r["@id"] for r in by_id[rc.ROOT_ID]["about"]}

    def test_a_label_without_ids_is_never_minted(self, record):
        crate = self._crate(record, {"teater_category": "kostel"}, {"teater_category": "hradiště", "teater_category_ids": []})
        assert "/theme/" not in rc.to_json(crate)
        assert not [e for e in crate["@graph"] if e.get("name") in ("kostel", "hradiště")]

    def test_unknown_sources_and_malformed_ids_are_skipped(self, record):
        ids = [{"source": "getty", "id": "300000000"}, {"source": "amcr", "id": ""}, "HES-000021", {"source": "amcr"}]
        crate = self._crate(record, {"teater_category": "kostel", "teater_category_ids": ids})
        assert not [e for e in crate["@graph"] if e.get("name") == "kostel"]

    def test_one_concept_under_two_labels_is_one_term_whatever_the_order(self, record):
        ref = [{"source": "amcr", "id": "HES-000021"}]
        a = {"teater_category": "zámek", "teater_category_ids": ref}
        b = {"teater_category": "hrad", "teater_category_ids": ref}
        first, second = self._crate(record, a, b), self._crate(record, b, a)
        assert rc.to_json(first) == rc.to_json(second)
        assert [e["name"] for e in first["@graph"] if e["@id"] == self.AMCR] == ["hrad"]

    def test_the_meta_text_sentinel_mints_nothing(self, record):
        """#51 acceptance: keyword-extract writes the sentinel with no ids (or none at all), and
        no concept URI is invented for it."""
        sentinel = "Nerelevantní (meta-text)"
        crate = self._crate(
            record, {"teater_category": sentinel, "teater_category_ids": []}, {"teater_category": sentinel}
        )
        assert "/theme/" not in rc.to_json(crate)
        assert not [e for e in crate["@graph"] if e.get("name") == sentinel]

    def test_two_concepts_sharing_a_label_stay_two_terms_and_are_not_made_equivalent(self, record):
        """#51 acceptance: a shared label does not make two source concepts one, and the crate
        asserts no equivalence between them."""
        other = "https://api.aiscr.cz/id/HES-000999"
        a = {"teater_category": "kostel", "teater_category_ids": [{"source": "amcr", "id": "HES-000021"}]}
        b = {"teater_category": "kostel", "teater_category_ids": [{"source": "amcr", "id": "HES-000999"}]}
        by_id = _by_id(self._crate(record, a, b))
        assert {by_id[self.AMCR]["name"], by_id[other]["name"]} == {"kostel"}
        for term in (by_id[self.AMCR], by_id[other]):
            assert not {"sameAs", "exactMatch", "closeMatch", "skos:exactMatch"} & set(term)

    def test_the_source_uris_survive_degraded_mode(self, record, monkeypatch):
        monkeypatch.setattr(rc, "atrium_vocab", None)
        crate = self._crate(record, {"teater_category": "kostel", "teater_category_ids": [{"source": "amcr", "id": "HES-000021"}]})
        assert _by_id(crate)[self.AMCR]["inDefinedTermSet"] == {"@id": rc.ATRIUM_BASE + "scheme/amcr-harvest"}


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


class TestCli:
    def _record_file(self, tmp_path, record):
        path = tmp_path / "CTX000000001.document.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return str(path)

    def test_selftest_mode(self):
        assert rc._cli(["--selftest"]) == 0

    def test_document_mode_writes_a_crate(self, tmp_path, record):
        src = self._record_file(tmp_path, record)
        pd = tmp_path / "pd.json"
        pd.write_text(json.dumps(rc._sample_paradata()[0]), encoding="utf-8")
        out = tmp_path / "out"
        assert rc._cli(["--document", src, "--paradata", str(pd), "--out-dir", str(out)]) == 0
        crate = json.loads((out / rc.METADATA_FILENAME).read_text(encoding="utf-8"))
        assert rc._SAMPLE_RUN_UUIDS["ocr-postprocess"] in _by_id(crate)

    def test_document_mode_without_out_dir_writes_to_stdout(self, tmp_path, record, capsys):
        src = self._record_file(tmp_path, record)
        assert rc._cli(["--document", src]) == 0
        assert json.loads(capsys.readouterr().out)["@graph"]

    def test_fragment_mode_wraps_for_validation(self, tmp_path, record):
        src = self._record_file(tmp_path, record)
        out = tmp_path / "frag"
        assert rc._cli(["--document", src, "--fragment", "--wrap", "--out-dir", str(out)]) == 0
        crate = json.loads((out / rc.METADATA_FILENAME).read_text(encoding="utf-8"))
        assert crate["@graph"][0]["@id"] == rc.METADATA_FILENAME
        assert "license" not in _by_id(crate)[rc.ROOT_ID]

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


def test_every_current_program_has_its_own_repository_and_predecessors_keep_theirs():
    """Repository moves of 2026-10-01: a crate names the repository that really produced a run."""
    expect = {
        "ocr-postprocess": "atrium-ocr-postprocess",
        "keyword-extract": "atrium-keyword-extract",
        "digital-convert": "atrium-digital-convert",
        "alto-postprocess": "atrium-alto-postprocess",  # predecessors: records written before the move
        "llm-enrich": "atrium-llm-enrich",
    }
    for program, slug in expect.items():
        assert rc.REPO_URLS[program] == f"https://github.com/ufal/{slug}", program
        assert rc._repository(program + "-api", {}) == f"https://github.com/ufal/{slug}", program
