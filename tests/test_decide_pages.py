"""tests/test_decide_pages.py — the #58 track-1 experiment CLI (B1–B4, triage), against a fake server.

No model, GPU or network: ``decision_client.FakeSystemOne`` answers on 127.0.0.1, the pages are the
public ``small_data_samples``, and B2's CNN table is synthetic.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

import decide_pages as dp
import decision_client as dc
from model_registry import CATEGORIES

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = REPO_ROOT / "small_data_samples"


def _client(fake):
    return dc.DecisionClient(fake.url, "fake-model", timeout_s=20, retries=0)


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ── question sets ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", sorted(dp.question_sets(), key=str), ids=lambda p: p.name)
def test_committed_question_sets_match_the_generator(path):
    expected = dp.render_question_set(dp.question_sets()[path])
    assert path.read_text(encoding="utf-8") == expected, (
        f"{path.name} drifted from decide_pages.py; run `python3 decide_pages.py --write-question-sets`"
    )


def test_the_category_choice_lists_the_registry_in_order_with_definitions():
    criteria = dp.page_questions()["page_category"]["criteria"]
    assert list(criteria) == list(CATEGORIES)
    assert all(criteria.values())


def test_route_gold_follows_the_hubs_next_step_table():
    routes_of = {label: {r for r, labels in dp.ROUTE_GOLD.items() if label in labels} for label in CATEGORIES}
    assert routes_of["TEXT"] == {"ocr", "htr"}
    assert routes_of["TEXT_P"] == routes_of["TEXT_T"] == {"ocr"}
    assert routes_of["TEXT_HW"] == {"htr"}
    assert routes_of["LINE_P"] == routes_of["LINE_T"] == {"tables", "ocr"}
    assert routes_of["LINE_HW"] == {"tables", "htr"}
    assert routes_of["DRAW"] == routes_of["PHOTO"] == {"graphics"}
    assert routes_of["DRAW_L"] == routes_of["PHOTO_L"] == {"graphics", "tables"}
    assert set(dp.ROUTE_GOLD) == set(dp.ROUTES) == set(dp.ROUTE_ELEMENTS)


def test_the_primary_gold_is_one_of_the_tags_the_label_implies():
    assert set(dp.PRIMARY_GOLD) == set(CATEGORIES)
    for label, accepted in dp.PRIMARY_GOLD.items():
        assert accepted and set(accepted) <= {r for r, labels in dp.ROUTE_GOLD.items() if label in labels}, label


def test_every_element_feeds_exactly_one_tag_and_the_stamp_goes_with_tables():
    fed = [e for elements in dp.ROUTE_ELEMENTS.values() for e in elements]
    assert sorted(fed) == sorted(dp.ELEMENTS)
    assert "has_stamp" in dp.ROUTE_ELEMENTS["tables"] and "has_filled_fields" in dp.ROUTE_ELEMENTS["tables"]


def test_element_gold_comes_from_the_vocabulary_collections():
    import atrium_vocab

    table = dp.element_gold()
    collections = atrium_vocab.COLLECTIONS["page-category"]
    assert set(table["has_handwriting"]["present"]) == set(collections["handwritten"]["members"])
    assert set(table["has_table_form"]["present"]) == set(collections["tabular"]["members"])
    assert set(table["has_printed_text"]["present"]) == set(collections["printed"]["members"]) | set(
        collections["typed"]["members"]
    )
    # "only …" labels rule an element out; nothing else does, and no label says anything about stamps
    assert table["has_printed_text"]["absent"] == ["TEXT_HW"]
    assert table["has_handwriting"]["absent"] == ["TEXT_P", "TEXT_T"]
    assert table["has_stamp"] == table["has_filled_fields"] == {"present": [], "absent": []}
    for entry in table.values():
        assert not set(entry["present"]) & set(entry["absent"])
    assert dp.gold_of(table, "has_photo", "PHOTO_L") == "1"
    assert dp.gold_of(table, "has_photo", "TEXT_P") == "0"
    assert dp.gold_of(table, "has_photo", "TEXT") == ""  # "minor graphical elements": unknown


def test_the_question_sets_fit_one_request_and_triage_merges_them():
    content, condition = dp.content_questions(), dp.condition_questions()
    assert len(content) == len(dp.ELEMENTS) + 1 and len(condition) == len(dp.DEFECT_IDS) + 1 + len(dp.SCORES)
    triage = dp.task_questions("triage")
    assert len(triage) == len(content) + len(condition) <= dc.MAX_QUESTIONS
    assert all(condition[qid]["type"] == "score" and len(condition[qid]["criteria"]) == 5 for qid in dp.SCORES)
    assert set(dp.SYNTHETIC_DEFECTS) <= set(dp.DEFECT_IDS)
    assert len(dp.DEFECT_IDS) == len(set(dp.DEFECT_IDS)) == 15


def test_damage_indices_are_aspect_means():
    p = {d: 0.0 for d in dp.DEFECT_IDS}
    p.update(dict.fromkeys(dp.DEFECTS["content"], 1.0))
    indices = dp.damage_indices(p)
    assert indices["content_index"] == 1.0 and indices["paper_index"] == 0.0
    assert indices["damage_index"] == pytest.approx(len(dp.DEFECTS["content"]) / len(dp.DEFECT_IDS), abs=1e-4)


def test_spearman_handles_ties_and_degenerate_input():
    assert dp.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert dp.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert dp.spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.8944, abs=1e-4)
    assert dp.spearman([1, 1, 1], [1, 2, 3]) is None and dp.spearman([1, 2], [1, 2]) is None


def test_binary_scores_leave_precision_out_without_known_negatives():
    only_positives = dp.binary_scores([(0.9, "1"), (0.2, "1"), (0.8, "")])
    assert only_positives["known"] == 2 and only_positives["recall"] == 0.5 and only_positives["precision"] is None
    mixed = dp.binary_scores([(0.9, "1"), (0.7, "0"), (0.1, "0")])
    assert (mixed["precision"], mixed["recall"], mixed["f1"]) == (0.5, 1.0, pytest.approx(0.6667, abs=1e-4))


# ── inputs ────────────────────────────────────────────────────────────────────────────────────


def test_pages_from_dir_reads_the_label_folders_only():
    pages = dp.pages_from_dir(SAMPLES)
    assert {label for _, label in pages} == set(CATEGORIES)
    assert all(path.parent.name == label for path, label in pages)


def test_sample_per_label_is_bounded_and_reproducible():
    pages = dp.pages_from_dir(SAMPLES)
    first = dp.sample_per_label(pages, 2, seed=7)
    assert first == dp.sample_per_label(pages, 2, seed=7)
    counts = {label: sum(1 for _, lab in first if lab == label) for label in CATEGORIES}
    assert max(counts.values()) <= 2 and min(counts.values()) >= 1


def test_top_n_breaks_ties_by_registry_order():
    probs = dict.fromkeys(CATEGORIES, 0.0)
    probs.update({"TEXT": 0.4, "DRAW": 0.3, "PHOTO": 0.3})
    assert dp.top_n(probs, 3) == ["TEXT", "DRAW", "PHOTO"]


def test_read_cnn_raw_refuses_a_table_without_label_columns(tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_text("FILE,PAGE,CLASS-1\nA,1,TEXT\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        dp.read_cnn_raw(bad)


# ── runs ──────────────────────────────────────────────────────────────────────────────────────


def _two_pages():
    pages = dp.pages_from_dir(SAMPLES)
    return [next(p for p in pages if p[1] == "PHOTO"), next(p for p in pages if p[1] == "TEXT_P")]


def test_b1_writes_the_table_and_resumes(tmp_path):
    out = tmp_path / "b1.csv"
    pages = _two_pages()
    with dc.FakeSystemOne() as fake:
        rows = dp.run(_client(fake), "b1", pages, [128, 256], out, progress=False)
        assert len(rows) == 4 and len(fake.requests) == 4
        with pytest.raises(SystemExit):
            dp.run(_client(fake), "b1", pages, [128], out, progress=False)
        again = dp.run(_client(fake), "b1", pages, [128, 256, 512], out, resume=True, progress=False)
    assert len(again) == 2  # only the new image size
    table = _read(out)
    assert list(table[0]) == dp.columns_for("b1")
    assert {"FILE", "PAGE", "TRUE", "CLASS-1"} <= set(table[0])  # what eval_by_document_overlap.py reads
    assert {r["status"] for r in table} == {"ok"}
    assert all(r["CLASS-1"] in CATEGORIES for r in table)
    assert all(abs(sum(float(r[f"P_{c}"]) for c in CATEGORIES) - 1) < 0.01 for r in table)
    assert all(
        "images" in request and len(request["questions"]["page_category"]["criteria"]) == 11
        for request in fake.requests
    )


def test_b2_sends_only_unsure_pages_and_only_the_cnns_top_three(tmp_path):
    pages = _two_pages()
    keys = [dp.page_key(path) for path, _ in pages]
    sure = dict.fromkeys(CATEGORIES, 0.0)
    sure["PHOTO"] = 0.99
    unsure = dict.fromkeys(CATEGORIES, 0.0)
    unsure.update({"TEXT": 0.5, "TEXT_P": 0.3, "TEXT_T": 0.2})
    cnn = {keys[0]: sure, keys[1]: unsure}
    out = tmp_path / "b2.csv"
    with dc.FakeSystemOne() as fake:
        dp.run(_client(fake), "b2", pages, [128], out, cnn_raw=cnn, tau=0.9, progress=False)
    assert len(fake.requests) == 1
    assert list(fake.requests[0]["questions"]["page_category"]["criteria"]) == ["TEXT", "TEXT_P", "TEXT_T"]
    first, second = _read(out)
    assert first["deferred"] == "0" and first["CLASS-1"] == "PHOTO" and first["status"] == "not_requested"
    assert second["deferred"] == "1" and second["options"] == "TEXT|TEXT_P|TEXT_T"
    assert second["CLASS-1"] in ("TEXT", "TEXT_P", "TEXT_T") and second["status"] == "ok"


def test_b2_keeps_the_cnn_answer_when_the_decision_fails(tmp_path):
    pages = _two_pages()[1:]
    probs = dict.fromkeys(CATEGORIES, 0.0)
    probs.update({"TEXT": 0.5, "TEXT_P": 0.5})
    cnn = {dp.page_key(pages[0][0]): probs}
    out = tmp_path / "b2.csv"
    with dc.FakeSystemOne(lambda r: (500, {"error": "down"})) as fake:
        dp.run(_client(fake), "b2", pages, [128], out, cnn_raw=cnn, tau=0.9, progress=False)
    (row,) = _read(out)
    assert row["status"] == "error" and row["CLASS-1"] == "TEXT"


def test_b2_marks_a_page_without_a_cnn_row(tmp_path):
    out = tmp_path / "b2.csv"
    with dc.FakeSystemOne() as fake:
        dp.run(_client(fake), "b2", _two_pages()[:1], [128], out, cnn_raw={}, tau=0.9, progress=False)
    (row,) = _read(out)
    assert row["status"] == "rejected" and not fake.requests


def test_b3_writes_elements_tags_and_the_primary_processor_with_their_gold(tmp_path):
    out = tmp_path / "b3.csv"
    with dc.FakeSystemOne() as fake:
        dp.run(_client(fake), "b3", _two_pages(), [128], out, progress=False)
    assert set(fake.requests[0]["questions"]) == set(dp.content_questions())
    photo, text = _read(out)
    assert list(photo) == dp.columns_for("b3")
    assert photo["GOLD_route_graphics"] == "1" and photo["GOLD_route_ocr"] == "0"
    assert text["GOLD_route_ocr"] == "1" and text["GOLD_route_tables"] == "0"
    assert photo["GOLD_has_photo"] == "1" and text["GOLD_has_photo"] == "0" and text["GOLD_has_stamp"] == ""
    assert photo["GOLD_primary"] == "graphics" and text["GOLD_primary"] == "ocr"
    for row in (photo, text):
        for route, elements in dp.ROUTE_ELEMENTS.items():
            assert float(row[f"P_route_{route}"]) == pytest.approx(max(float(row[f"P_{e}"]) for e in elements))
        assert row["primary_route"] in dp.ROUTES
    entry = dp.summarise("b3", _read(out))["by_max_side"]["128"]["content"]
    assert set(entry["routes"]) == set(dp.ROUTES) and set(entry["elements"]) == set(dp.ELEMENTS)
    assert entry["elements"]["has_stamp"]["known"] == 0


def test_b4_writes_defects_orientation_scores_and_indices(tmp_path):
    out = tmp_path / "b4.csv"
    with dc.FakeSystemOne() as fake:
        dp.run(_client(fake), "b4", _two_pages()[:1], [128], out, progress=False)
    assert set(fake.requests[0]["questions"]) == set(dp.condition_questions())
    (row,) = _read(out)
    assert list(row) == dp.columns_for("b4")
    assert all(0 <= float(row[f"P_{d}"]) <= 1 for d in dp.DEFECT_IDS)
    assert row["orientation"] in dp.ORIENTATIONS
    assert all(0 <= float(row[f"E_{q}"]) <= dp.SCORE_MAX for q in dp.SCORES)
    assert float(row["damage_score"]) == pytest.approx(float(row["E_overall_damage"]) / dp.SCORE_MAX, abs=1e-4)
    assert float(row["damage_index"]) == pytest.approx(
        sum(float(row[f"P_{d}"]) for d in dp.DEFECT_IDS) / len(dp.DEFECT_IDS), abs=1e-3
    )


def test_triage_asks_both_sets_in_one_request(tmp_path):
    out = tmp_path / "triage.csv"
    with dc.FakeSystemOne() as fake:
        dp.run(_client(fake), "triage", _two_pages(), [128], out, progress=False)
    assert len(fake.requests) == 2
    assert set(fake.requests[0]["questions"]) == set(dp.task_questions("triage"))
    rows = _read(out)
    assert list(rows[0]) == dp.columns_for("triage")
    assert all(r["primary_route"] and r["orientation"] and r["damage_score"] for r in rows)
    assert {r["schema_digest"] for r in rows} == {dc.schema_digest(dp.task_questions("triage"))}


def test_a_failed_triage_request_leaves_the_answers_empty(tmp_path):
    out = tmp_path / "triage.csv"
    with dc.FakeSystemOne(lambda r: (500, {"error": "down"})) as fake:
        dp.run(_client(fake), "triage", _two_pages()[:1], [128], out, progress=False)
    (row,) = _read(out)
    assert row["status"] == "error" and row["primary_route"] == "" and row["damage_index"] == ""
    assert row["GOLD_route_graphics"] == "1"  # the gold does not depend on the answer


def _degraded(tmp_path, defects="skewed,rotated_text", levels="1,3"):
    sys.path.insert(0, str(REPO_ROOT / "data_scripts"))
    import degrade_pages

    stage = tmp_path / "stage"
    for path, label in _two_pages():
        (stage / label).mkdir(parents=True, exist_ok=True)
        (stage / label / path.name).write_bytes(path.read_bytes())
    out = tmp_path / "degraded"
    assert degrade_pages.main(["--dir", str(stage), "--out", str(out), "--defects", defects, "--levels", levels]) == 0
    return out / "degradations.csv"


def test_b4_reads_a_degradation_manifest_and_pairs_copies_with_originals(tmp_path):
    manifest = _degraded(tmp_path)
    pages, variants = dp.read_degradations(manifest)
    assert len(pages) == 2 * (1 + 2 + 3)  # original, skewed at 2 levels, 3 orientations
    out = tmp_path / "b4.csv"
    with dc.FakeSystemOne() as fake:
        dp.run(_client(fake), "b4", pages, [128], out, progress=False, variants=variants)
    rows = _read(out)
    originals = [r for r in rows if r["VARIANT"] == "original"]
    assert len(originals) == 2 and {r["LEVEL"] for r in originals} == {"0"}
    # a copy keeps its source's FILE and PAGE, so it pairs with the original
    by_source = {}
    for r in rows:
        by_source.setdefault(r["SOURCE"], set()).add((r["FILE"], r["PAGE"]))
    assert all(len(keys) == 1 for keys in by_source.values())
    assert {r["GOLD_orientation"] for r in rows if r["VARIANT"] == "rotated_text"} == {"cw90", "upside_down", "ccw90"}
    condition = dp.summarise("b4", rows)["by_max_side"]["128"]["condition"]
    assert set(condition["injected"]) == {"skewed", "rotated_text"}
    assert condition["injected"]["skewed"]["paired"] == 4 and condition["injected"]["rotated_text"]["paired"] == 6
    assert condition["orientation_pages"] == 6  # the TEXT_P page and its five copies; PHOTO is not judged


def test_main_refuses_per_label_sampling_of_a_manifest(tmp_path):
    manifest = _degraded(tmp_path, defects="blurred", levels="1")
    with pytest.raises(SystemExit):
        dp.main(
            ["--task", "b4", "--degradations", str(manifest), "--per-label", "1", "--url", "http://x", "--model", "m"]
        )
    with pytest.raises(SystemExit):
        dp.main(["--task", "b1", "--degradations", str(manifest), "--url", "http://x", "--model", "m"])


def test_summarise_counts_fixed_and_broken_pages():
    base = {"status": "ok", "latency_ms": "10", "input_tokens": "5", "max_side": "1024"}
    rows = [
        {**base, "TRUE": "TEXT", "CNN_CLASS": "TEXT_T", "CLASS-1": "TEXT", "deferred": "1"},  # fixed
        {**base, "TRUE": "TEXT", "CNN_CLASS": "TEXT", "CLASS-1": "TEXT_P", "deferred": "1"},  # broken
        {**base, "TRUE": "DRAW", "CNN_CLASS": "DRAW", "CLASS-1": "DRAW", "deferred": "0", "status": "not_requested"},
    ]
    entry = dp.summarise("b2", rows)["by_max_side"]["1024"]
    assert (entry["deferred"], entry["fixed"], entry["broken"]) == (2, 1, 1)
    assert entry["cnn_top1"] == pytest.approx(2 / 3, abs=1e-4) and entry["top1"] == pytest.approx(2 / 3, abs=1e-4)
    assert entry["mcnemar_p"] == 1.0


def test_main_end_to_end_writes_table_summary_and_paradata(tmp_path, monkeypatch):
    out = tmp_path / "run" / "b1.csv"
    with dc.FakeSystemOne() as fake:
        code = dp.main(
            [
                "--task",
                "b1",
                "--dir",
                str(SAMPLES),
                "--per-label",
                "1",
                "--max-side",
                "128",
                "--url",
                fake.url,
                "--model",
                "fake-model",
                "--model-revision",
                "test",
                "--out",
                str(out),
            ]
        )
    assert code == 0
    assert len(_read(out)) == len(CATEGORIES)
    summary = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["statuses"] == {"ok": len(CATEGORIES)}
    assert summary["schema_digest"] == dc.schema_digest(dp.page_questions())
    records = list((out.parent / "paradata").glob("*.json"))
    assert records, "a paradata record is written beside the table"
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert "decision_model" in json.dumps(record)
    assert record["config"]["served_on"] == summary["served_on"] == "local"  # by name, never the URL
    assert fake.url not in json.dumps(record) and fake.url not in json.dumps(summary)


def test_served_on_names_the_server_without_its_url():
    assert dp.served_on("http://127.0.0.1:30000") == "local"
    assert dp.served_on("http://decide.example:8000") == "other"
    assert dp.served_on("http://decide.example:8000", "ufal-endpoint") == "ufal-endpoint"


def test_main_refuses_an_unconfigured_server(monkeypatch):
    monkeypatch.delenv("DECISION_MODEL_URL", raising=False)
    with pytest.raises(SystemExit):
        dp.main(["--task", "b1", "--dir", str(SAMPLES), "--url", "", "--model", "m"])


@pytest.mark.parametrize(
    "thresholds, message",
    [
        (["--accept-at", "0.9"], "--accept-at and --review-below go together"),
        (["--accept-at", "1.5", "--review-below", "0.5"], "--accept-at 1.5 is outside [0, 1]"),
        (["--accept-at", "0.9", "--review-below", "-0.1"], "--review-below -0.1 is outside [0, 1]"),
        (["--accept-at", "nan", "--review-below", "0.5"], "--accept-at nan is outside [0, 1]"),
        (["--accept-at", "0.5", "--review-below", "0.8"], "--review-below (0.8) must not exceed --accept-at (0.5)"),
    ],
)
def test_main_refuses_lane_thresholds_before_any_request(thresholds, message, capsys, tmp_path):
    # A usage error at the start: decision_client.lane() would raise only on the first answer,
    # after a paid call. The URL is unreachable, so a run past the check would write one error
    # row and exit 0 (quickly: one job, no retry), never raise SystemExit.
    with pytest.raises(SystemExit) as excinfo:
        dp.main(
            ["--task", "b1", "--dir", str(SAMPLES), "--per-label", "1", "--limit", "1", "--retries", "0"]
            + ["--url", "http://127.0.0.1:9", "--model", "m", "--out", str(tmp_path / "b1.csv"), *thresholds]
        )
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


def test_main_accepts_lane_thresholds_on_the_bounds(tmp_path):
    out = tmp_path / "b1.csv"
    with dc.FakeSystemOne() as fake:
        code = dp.main(
            ["--task", "b1", "--dir", str(SAMPLES), "--per-label", "1", "--max-side", "128", "--limit", "1"]
            + ["--url", fake.url, "--model", "fake-model", "--out", str(out)]
            + ["--accept-at", "1", "--review-below", "0"]
        )
    assert code == 0
    (row,) = _read(out)
    assert row["status"] == "ok" and row["lane"] in ("accept", "escalate")  # nothing is below 0


def test_limit_stops_after_the_first_n_page_and_size_jobs(tmp_path):
    out = tmp_path / "b1.csv"
    pages = _two_pages()
    with dc.FakeSystemOne() as fake:
        rows = dp.run(_client(fake), "b1", pages, [128, 256], out, limit=3, progress=False)
    assert len(rows) == 3 and len(fake.requests) == 3
    # the jobs run size by size, so the cut falls in the second size
    assert [(r["image"], r["max_side"]) for r in rows] == [
        (dp._relative(pages[0][0]), 128),
        (dp._relative(pages[1][0]), 128),
        (dp._relative(pages[0][0]), 256),
    ]
