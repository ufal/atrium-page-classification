"""
tests/test_compare_pdf_rasterisers.py
=====================================
The logic of data_scripts/compare_pdf_rasterisers.py, the measurement behind the PDF engine of
/predict_document (atrium-project#72; results in data_scripts/pdf_rasteriser_comparison.md).

Neither reference renderer is a dependency of this repository — pdftoppm is a system tool and
PyMuPDF (AGPL-3.0) is in no requirements file — so the references are stand-ins here: fake
renderers for the report, and a stand-in `pdftoppm` on PATH (the pattern of
tests/test_data_scripts.py) for the page ordering. The service's own renderer is driven for real
where pypdfium2 is installed (the fast lane installs it).
"""

import importlib.util
import io
import math
import sys
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "data_scripts" / "compare_pdf_rasterisers.py"


@pytest.fixture(scope="module")
def cmp():
    spec = importlib.util.spec_from_file_location("compare_pdf_rasterisers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _page(colour=(255, 255, 255), size=(40, 60)):
    return Image.new("RGB", size, colour)


class TestMeasures:
    def test_identical_pages_differ_by_nothing(self, cmp):
        d = cmp.difference(_page(), _page())
        assert (d.mean_abs, d.max_abs, d.visible_share, d.psnr) == (0.0, 0, 0.0, math.inf)

    def test_a_uniform_offset_is_measured(self, cmp):
        d = cmp.difference(_page((100, 100, 100)), _page((110, 100, 100)))
        assert d.max_abs == 10 and d.visible_share == 1.0  # 10 levels > VISIBLE_DIFF on every pixel
        assert d.mean_abs == pytest.approx(10 / 3)
        assert d.psnr == pytest.approx(10 * math.log10(255**2 / (100 / 3)))

    def test_renders_one_row_apart_are_compared_on_what_both_drew(self, cmp):
        """PDFium and pdftoppm may round a page's size differently by one pixel."""
        assert cmp.difference(_page(size=(40, 60)), _page(size=(40, 61))).psnr == math.inf

    def test_the_model_view_is_the_common_area_at_the_input_size(self, cmp):
        a, b = cmp.at_model_input(_page(size=(300, 400)), _page(size=(301, 400)))
        assert a.size == b.size == (cmp.MODEL_INPUT, cmp.MODEL_INPUT)


class TestReport:
    @staticmethod
    def _doc(cmp, name="d.pdf", sources=()):
        return cmp.Document(name, b"%PDF", list(sources))

    def test_one_row_per_page_and_a_summary_per_reference(self, cmp):
        service = lambda data, dpi: [_page(), _page((0, 0, 0))]  # noqa: E731
        reference = lambda data, dpi: [_page(), _page((0, 0, 0), size=(40, 61))]  # noqa: E731
        lines = cmp.compare([self._doc(cmp)], 300, {"pdftoppm": reference}, service=service)
        assert lines[0].startswith("| document | page | size (service) | size (pdftoppm)")
        assert sum(line.startswith("| d.pdf |") for line in lines) == 2
        summary = next(line for line in lines if line.startswith("- **vs pdftoppm:**"))
        assert "pixel size differs on 1 of 2 pages" in summary
        assert any("Labels not computed" in line for line in lines)

    def test_labels_are_compared_and_a_disagreement_is_flagged(self, cmp):
        service = lambda data, dpi: [_page((255, 255, 255))]  # noqa: E731
        reference = lambda data, dpi: [_page((0, 0, 0))]  # noqa: E731
        labels = lambda image: "TEXT" if image.getpixel((0, 0))[0] > 128 else "DRAW"  # noqa: E731
        lines = cmp.compare([self._doc(cmp)], 300, {"PyMuPDF": reference}, labels=labels, service=service)
        assert any(line.endswith("TEXT / DRAW ⚠️ |") for line in lines)
        assert any("top-1 labels agree on 0 of 1 pages" in line for line in lines)

    def test_a_scan_is_compared_with_its_source_resampled_when_scanned_at_another_resolution(self, cmp):
        source = _page(size=(80, 120))
        service = lambda data, dpi: [_page(size=(40, 60))]  # noqa: E731
        lines = cmp.compare([self._doc(cmp, sources=[source])], 300, {}, service=service)
        assert any(line.startswith("- **vs the source images:** median PSNR ∞ dB") for line in lines)

    def test_the_old_engine_is_measured_against_the_training_renderer(self, cmp):
        service = lambda data, dpi: [_page()]  # noqa: E731
        references = {"pdftoppm": lambda d, dpi: [_page()], "PyMuPDF": lambda d, dpi: [_page((250, 255, 255))]}
        lines = cmp.compare([self._doc(cmp)], 300, references, service=service)
        assert any(line.startswith("- **Baseline, PyMuPDF (the old engine) vs pdftoppm:**") for line in lines)

    def test_a_document_whose_page_counts_differ_is_reported_not_compared(self, cmp):
        service = lambda data, dpi: [_page()]  # noqa: E731
        reference = lambda data, dpi: [_page(), _page()]  # noqa: E731
        lines = cmp.compare([self._doc(cmp)], 300, {"pdftoppm": reference}, service=service)
        assert any("page counts differ: pdftoppm 2" in line for line in lines)


class TestPdftoppm:
    def test_pages_come_back_in_page_order_not_name_order(self, cmp, tmp_path):
        """pdftoppm pads page numbers to the width of the page count; sorted by name, page-10
        would come before page-2."""
        made = tmp_path / "made"
        made.mkdir()
        for n in range(1, 12):
            Image.new("RGB", (4, 4), (n, 0, 0)).save(made / f"{n}.png")
        stand_in = tmp_path / "pdftoppm"  # `pdftoppm -png -r DPI IN.pdf PREFIX` writes PREFIX-N.png
        stand_in.write_text(
            f"#!{sys.executable}\n"
            "import shutil, sys\n"
            "for n in range(1, 12):\n"
            f"    shutil.copy(f'{made}/{{n}}.png', f'{{sys.argv[-1]}}-{{n}}.png')\n",
            encoding="utf-8",
        )
        stand_in.chmod(0o755)
        pages = cmp.pdftoppm_renderer(str(stand_in))(b"%PDF", 300)
        assert [page.getpixel((0, 0))[0] for page in pages] == list(range(1, 12))


class TestCommandLine:
    def test_no_reference_is_exit_2(self, cmp, monkeypatch, capsys, tmp_path):
        pytest.importorskip("pypdfium2")
        monkeypatch.setattr(cmp, "available_references", lambda pdftoppm: {})
        assert cmp.main([str(tmp_path)]) == 2
        assert "no reference to compare with" in capsys.readouterr().err

    def test_a_missing_input_is_exit_2(self, cmp, monkeypatch, capsys, tmp_path):
        pytest.importorskip("pypdfium2")
        monkeypatch.setattr(cmp, "available_references", lambda pdftoppm: {"pdftoppm": None})
        assert cmp.main([str(tmp_path / "absent.pdf")]) == 2
        assert "no such file or directory" in capsys.readouterr().err

    def test_a_malformed_canvas_is_exit_2(self, cmp, monkeypatch, capsys, tmp_path):
        pytest.importorskip("pypdfium2")
        monkeypatch.setattr(cmp, "available_references", lambda pdftoppm: {"pdftoppm": None})
        assert cmp.main(["--from-images", str(tmp_path), "--canvas", "A4"]) == 2
        assert "--canvas takes WIDTHxHEIGHT" in capsys.readouterr().err


def test_the_service_renderer_is_the_services_code(cmp):
    """The script measures what /predict_document does: service/pdf_render.py, at dpi / 72."""
    pytest.importorskip("pypdfium2")
    buf = io.BytesIO()
    Image.new("RGB", (144, 72), (200, 120, 40)).save(buf, format="PDF", resolution=72)
    (page,) = cmp.render_service(buf.getvalue(), 144)
    assert page.mode == "RGB" and page.size == (288, 144)
    assert sys.modules["service.pdf_render"].render_scale(144) == 2.0
