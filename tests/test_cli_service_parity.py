"""
tests/test_cli_service_parity.py
================================
The command line and the service read the same inputs and give the same results
(atrium-project#72, the service–CLI alignment of 2026-10-26).

Since v1.10.0-beta `run.py` reads PDFs as `/predict_document` does: pages rendered by the
service's own renderer (`service/pdf_render.py`, through `page_inputs.py`) at `PDF_RENDER_DPI`,
the `pages` selection, the same doc_id, and a born-digital baseline's page labels as the record
keys. `-f` writes its table like `-d`. The model is a stand-in on both sides (no torch here): it
answers by the colour of the page, so the two sides can only agree if they read the same pixels.

What differs on purpose, pinned below and explained in docs/how-it-works.md ("One model or five"):
the CLI's table rounds scores to 3 decimals, the service's answer does not.
"""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pypdfium2")
pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from fastapi.testclient import TestClient  # noqa: E402

from model_registry import CATEGORIES  # noqa: E402
from tests.openapi_contract_data import prepare  # noqa: E402

prepare()

from service import api  # noqa: E402

#: (R, G, B) per page: a reddish page reads as DRAW, a bluish one as TEXT.
COLOURS = [(200, 120, 40), (40, 160, 220), (210, 90, 60)]
SCORE = 0.876543


def _pdf() -> bytes:
    from PIL import Image

    pages = [Image.new("RGB", (60, 80), color=colour) for colour in COLOURS]
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", resolution=72, save_all=True, append_images=pages[1:])
    return buf.getvalue()


def _label(image) -> str:
    r, _, b = image.convert("RGB").getpixel((image.width // 2, image.height // 2))
    return "DRAW" if r > b else "TEXT"


def _ranked(image, top_n):
    first = CATEGORIES.index(_label(image))
    second = CATEGORIES.index("PHOTO")
    return [(first, SCORE), (second, 1 - SCORE)][:top_n]


class _FakeClassifier:
    def __init__(self, checkpoint=None, num_labels=None, store_dir=None, **_):
        pass

    def load_model(self, path):
        pass

    def load_from_hub(self, repo, revision):
        pass

    def save_model(self, path):
        pass

    def top_n_predictions(self, image_input, top_n=1):
        from PIL import Image

        return _ranked(Image.open(image_input), top_n)

    def create_dataloader(self, image_paths, batch_size, ignored_paths=None):
        return list(image_paths)

    def infer_dataloader(self, dataloader, top_n, raw=False):
        from PIL import Image

        return [_ranked(Image.open(path), top_n) for path in dataloader], None


class _Manager:
    """The service's model manager, answering by colour as the fake classifier does."""

    device = "cpu"
    available_versions = ["v4.4"]

    def get_model_details(self, version):
        return "stub"

    def predict(self, image, version, topn):
        return [{"label": CATEGORIES[i], "score": s} for i, s in _ranked(image, topn)]

    def predict_detailed(self, image, version, topn):
        predictions = self.predict(image, version, topn)
        return {"predictions": predictions, "models_used": ["v4.4"], "per_model": {"v4.4": predictions}}


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """`run.main` with the torch-bound modules replaced: the classifier, the --best engine, YOLO."""
    fake_classifier = types.ModuleType("classifier")
    fake_classifier.ImageClassifier = _FakeClassifier
    fake_classifier.average_model_weights = lambda **_: None
    fake_classifier.split_data_80_10_10 = lambda *a, **k: None
    fake_classifier.split_data_from_folds = lambda *a, **k: None
    fake_best = types.ModuleType("parallel_best")
    fake_best.run_best_models = lambda **_: None
    fake_yolo = types.ModuleType("yolo_classifier")
    fake_yolo.YOLOClassifier = _FakeClassifier
    monkeypatch.setitem(sys.modules, "classifier", fake_classifier)
    monkeypatch.setitem(sys.modules, "parallel_best", fake_best)
    monkeypatch.setitem(sys.modules, "yolo_classifier", fake_yolo)
    import utils

    monkeypatch.setattr(utils, "tolerate_scan_quirks", lambda: None)  # process-wide Pillow switches
    monkeypatch.setenv("PDF_RENDER_DPI", "72")
    monkeypatch.chdir(tmp_path)
    from run import main

    return main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "manager", _Manager())
    monkeypatch.setenv("PDF_RENDER_DPI", "72")
    return TestClient(api.app)


def _table(tmp_path: Path):
    import pandas as pd

    (path,) = list((tmp_path / "result" / "tables").glob("*_TOP-*.csv"))
    return pd.read_csv(path)


def _born_digital(doc_id: str, labels) -> dict:
    from atrium_document import DocumentRecord, validate_document

    canvas = {"width": 60.0, "height": 80.0, "unit": "pt"}
    record = DocumentRecord(doc_id, "digital-convert")
    record.set_source("0" * 64, filename=f"{doc_id}.pdf", media_type="application/pdf", origin="digital-born-pdf")
    record.merge_block(
        "pages",
        [{"page": label, "page_index": i, "canvas": canvas} for i, label in enumerate(labels, start=1)],
        key_fields=["page"],
    )
    data = record.to_dict()
    validate_document(data)
    return data


def test_a_pdf_is_classified_page_by_page_on_both_sides(cli, client, tmp_path):
    pdf = tmp_path / "CTX01.pdf"
    pdf.write_bytes(_pdf())
    out = tmp_path / "CTX01.document.json"
    assert cli(["-f", str(pdf), "--no-hf", "-tn", "1", "--document-json-out", str(out)]) == 0

    table = _table(tmp_path)
    cli_pages = {int(row.PAGE): row["CLASS-1"] for _, row in table.iterrows()}
    body = client.post(
        "/predict_document",
        data={"version": "v4.4", "topn": 1, "document_json_out": "true"},
        files={"file": ("CTX01.pdf", pdf.read_bytes(), "application/pdf")},
    ).json()
    service_pages = {page["page"]: page["predictions"][0]["label"] for page in body["pages"]}
    assert cli_pages == service_pages == {1: "DRAW", 2: "TEXT", 3: "DRAW"}
    assert set(table["FILE"]) == {body["document_json"]["doc_id"]} == {"CTX01"}
    cli_record = json.loads(out.read_text(encoding="utf-8"))
    assert cli_record["page_categories"] == body["document_json"]["page_categories"]


def test_the_page_selection_means_the_same_on_both_sides(cli, client, tmp_path):
    pdf = tmp_path / "CTX02.pdf"
    pdf.write_bytes(_pdf())
    assert cli(["-f", str(pdf), "--no-hf", "-tn", "1", "--pages", "2-3"]) == 0
    body = client.post(
        "/predict_document",
        data={"version": "v4.4", "topn": 1, "pages": "2-3"},
        files={"file": ("CTX02.pdf", pdf.read_bytes(), "application/pdf")},
    ).json()
    assert sorted(int(p) for p in _table(tmp_path)["PAGE"]) == [page["page"] for page in body["pages"]] == [2, 3]


def test_a_born_digital_baseline_gets_its_own_page_labels_on_both_sides(cli, client, tmp_path):
    """The CLI wrote positions ("1", "2", …) into a record whose pages are labelled; it now maps
    them through `page_index` as /predict_document does (atrium-digital-convert#2)."""
    pdf = tmp_path / "CTX03.pdf"
    pdf.write_bytes(_pdf())
    baseline = _born_digital("CTX03", ["i", "ii", "1"])
    seed = tmp_path / "seed.document.json"
    seed.write_text(json.dumps(baseline), encoding="utf-8")
    out = tmp_path / "out.document.json"
    assert (
        cli(["-f", str(pdf), "--no-hf", "-tn", "1", "--document-json", str(seed), "--document-json-out", str(out)]) == 0
    )
    body = client.post(
        "/predict_document",
        data={"version": "v4.4", "topn": 1},
        files={
            "file": ("CTX03.pdf", pdf.read_bytes(), "application/pdf"),
            "document_json": ("seed.document.json", json.dumps(baseline).encode("utf-8"), "application/json"),
        },
    ).json()
    cli_record = json.loads(out.read_text(encoding="utf-8"))
    assert (
        cli_record["page_categories"]
        == body["document_json"]["page_categories"]
        == {
            "i": "DRAW",
            "ii": "TEXT",
            "1": "DRAW",
        }
    )


def test_a_directory_of_pdfs_and_images_in_any_case(cli, tmp_path):
    from PIL import Image

    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "DOCA.pdf").write_bytes(_pdf())
    Image.new("RGB", (60, 80), color=COLOURS[1]).save(pages / "DOCB-1.PNG")
    Image.new("RGB", (60, 80), color=COLOURS[0]).save(pages / "DOCB-2.jpg")
    out_dir = tmp_path / "records"
    assert (
        cli(["-d", str(pages), "-ff", "png,JPG,pdf", "--no-hf", "-tn", "1", "--document-json-out-dir", str(out_dir)])
        == 0
    )
    table = _table(tmp_path)
    got = {(row.FILE, int(row.PAGE)): row["CLASS-1"] for _, row in table.iterrows()}
    assert got == {
        ("DOCA", 1): "DRAW",
        ("DOCA", 2): "TEXT",
        ("DOCA", 3): "DRAW",
        ("DOCB", 1): "TEXT",
        ("DOCB", 2): "DRAW",
    }
    assert sorted(p.name for p in out_dir.iterdir()) == ["DOCA.document.json", "DOCB.document.json"]


def test_pdfs_are_left_out_unless_the_formats_name_them(cli, tmp_path):
    """The default `-ff png` keeps a directory run as it was: a PDF next to its page images is
    not classified twice."""
    from PIL import Image

    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "DOCA.pdf").write_bytes(_pdf())
    Image.new("RGB", (60, 80), color=COLOURS[1]).save(pages / "DOCA-1.png")
    assert cli(["-d", str(pages), "--no-hf", "-tn", "1"]) == 0
    assert list(_table(tmp_path)["FILE"]) == ["DOCA"]
    assert len(_table(tmp_path)) == 1


def test_a_single_image_now_writes_the_table_and_the_record(cli, client, tmp_path):
    from PIL import Image

    image = tmp_path / "CTX04-7.png"
    Image.new("RGB", (60, 80), color=COLOURS[0]).save(image)
    out_dir = tmp_path / "records"
    assert cli(["-f", str(image), "--no-hf", "-tn", "2", "--document-json-out-dir", str(out_dir)]) == 0
    table = _table(tmp_path)
    assert list(table.columns[:2]) == ["FILE", "PAGE"]
    row = table.iloc[0]
    assert (row.FILE, int(row.PAGE), row["CLASS-1"]) == ("CTX04", 7, "DRAW")
    body = client.post(
        "/predict_image",
        data={"version": "v4.4", "topn": 2, "document_json_out": "true"},
        files={"file": ("CTX04-7.png", image.read_bytes(), "image/png")},
    ).json()
    cli_record = json.loads((out_dir / "CTX04.document.json").read_text(encoding="utf-8"))
    assert cli_record["page_categories"] == body["document_json"]["page_categories"] == {"7": "DRAW"}
    # Kept apart on purpose: the table rounds to 3 decimals, the answer does not.
    assert row["SCORE-1"] == round(SCORE, 3)
    assert body["predictions"][0]["score"] == SCORE


def test_document_json_out_with_a_directory_run_is_refused(cli, tmp_path):
    with pytest.raises(SystemExit) as caught:
        cli(["-d", str(tmp_path), "--document-json-out", "x.document.json"])
    assert caught.value.code == 2


def test_a_malformed_page_selection_is_refused_by_both(cli, client, tmp_path):
    with pytest.raises(SystemExit) as caught:
        cli(["-f", "x.pdf", "--pages", "3-1"])
    assert caught.value.code == 2
    response = client.post(
        "/predict_document",
        data={"version": "v4.4", "topn": 1, "pages": "3-1"},
        files={"file": ("x.pdf", _pdf(), "application/pdf")},
    )
    assert response.status_code == 422


def test_the_service_recognises_an_image_or_a_pdf_sent_without_its_media_type(client):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (60, 80), color=COLOURS[1]).save(buf, format="PNG")
    image = client.post(
        "/predict_image",
        data={"version": "v4.4", "topn": 1},
        files={"file": ("page-1", buf.getvalue(), "application/octet-stream")},
    )
    assert image.status_code == 200, image.text
    assert image.json()["predictions"][0]["label"] == "TEXT"
    pdf = client.post(
        "/predict_document",
        data={"version": "v4.4", "topn": 1},
        files={"file": ("scan", _pdf(), "application/octet-stream")},
    )
    assert pdf.status_code == 200, pdf.text
    assert [p["page"] for p in pdf.json()["pages"]] == [1, 2, 3]
    not_an_image = client.post(
        "/predict_image",
        data={"version": "v4.4", "topn": 1},
        files={"file": ("x", b"plain text", "application/octet-stream")},
    )
    assert not_an_image.status_code == 415
    not_a_pdf = client.post(
        "/predict_document",
        data={"version": "v4.4", "topn": 1},
        files={"file": ("x", b"plain text", "application/octet-stream")},
    )
    assert not_a_pdf.status_code == 415


def test_per_model_is_opt_in(client):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (60, 80), color=COLOURS[0]).save(buf, format="PNG")
    files = {"file": ("p-1.png", buf.getvalue(), "image/png")}
    plain = client.post("/predict_image", data={"version": "v4.4", "topn": 1}, files=files).json()
    assert plain["per_model"] is None
    detailed = client.post(
        "/predict_image", data={"version": "v4.4", "topn": 1, "per_model": "true"}, files=files
    ).json()
    assert detailed["per_model"] == {"v4.4": [{"label": "DRAW", "score": SCORE}]}
    pages = client.post(
        "/predict_document",
        data={"version": "v4.4", "topn": 1, "per_model": "true"},
        files={"file": ("d.pdf", _pdf(), "application/pdf")},
    ).json()["pages"]
    assert all(page["per_model"]["v4.4"] == page["predictions"] for page in pages)


def test_transparency_is_handled_differently_and_pinned():
    """Kept as it is, documented in docs/how-it-works.md ("Transparent images"): the CLI's loaders
    (classifier.py) composite a transparent page onto WHITE; the service converts it to RGB, so a
    fully transparent pixel keeps its underlying colour (here black). An open question in
    atrium-project#72: the scores of such pages differ between the two sides."""
    from PIL import Image

    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert api._decode_image(rgba).getpixel((0, 0)) == (0, 0, 0)


def test_the_formats_are_read_in_any_case_and_a_pdf_by_its_content(tmp_path):
    from page_inputs import is_pdf, parse_formats, scrape

    assert parse_formats("png, JPG ,.tif") == {"png", "jpg", "tif"}
    assert parse_formats("") == {"png"}
    renamed = tmp_path / "scan.bin"
    renamed.write_bytes(_pdf())
    assert is_pdf(renamed) and not is_pdf(tmp_path / "missing.pdf")
    (tmp_path / "sub").mkdir()
    for name in ("a.PNG", "sub/b.png", "c.jpeg", "d.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [Path(p).name for p in scrape(tmp_path, {"png"})] == ["a.PNG", "b.png"]
    assert [Path(p).name for p in scrape(tmp_path, {"png"}, inner=False)] == ["a.PNG"]
