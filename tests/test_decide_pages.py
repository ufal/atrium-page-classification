"""tests/test_decide_pages.py — the #58 track-1 experiment CLI (B1, B2, B3), against a fake server.

No model, GPU or network: ``decision_client.FakeSystemOne`` answers on 127.0.0.1, the pages are the
public ``small_data_samples``, and B2's CNN table is synthetic.
"""

import csv
import json
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
    assert routes_of["TEXT"] == {"route_ocr", "route_htr"}
    assert routes_of["TEXT_P"] == routes_of["TEXT_T"] == {"route_ocr"}
    assert routes_of["TEXT_HW"] == {"route_htr"}
    assert routes_of["LINE_P"] == routes_of["LINE_T"] == {"route_tables", "route_ocr"}
    assert routes_of["LINE_HW"] == {"route_tables", "route_htr"}
    assert routes_of["DRAW"] == routes_of["PHOTO"] == {"route_image_description"}
    assert routes_of["DRAW_L"] == routes_of["PHOTO_L"] == {"route_image_description", "route_tables"}
    assert set(dp.ROUTE_GOLD) == set(dp.ROUTES)


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


def test_b3_writes_routes_with_their_gold(tmp_path):
    out = tmp_path / "b3.csv"
    with dc.FakeSystemOne() as fake:
        dp.run(_client(fake), "b3", _two_pages(), [128], out, progress=False)
    photo, text = _read(out)
    assert photo["GOLD_route_image_description"] == "1" and photo["GOLD_route_ocr"] == "0"
    assert text["GOLD_route_ocr"] == "1" and text["GOLD_route_tables"] == "0"
    assert all(0 <= float(photo[f"P_{route}"]) <= 1 for route in dp.ROUTES)
    summary = dp.summarise("b3", _read(out))
    assert set(summary["by_max_side"]["128"]["routes"]) == set(dp.ROUTES)


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


def test_main_refuses_an_unconfigured_server(monkeypatch):
    monkeypatch.delenv("DECISION_MODEL_URL", raising=False)
    with pytest.raises(SystemExit):
        dp.main(["--task", "b1", "--dir", str(SAMPLES), "--url", "", "--model", "m"])
