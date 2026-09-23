"""
tests/test_data_scripts.py
==========================
Behavioural tests for the Unix data-preparation scripts in data_scripts/unix/.

Scope
-----
* sort.sh    – CSV columns by header name, label/page validation, padding in the
               document folder and in onepagers/, no empty label folders, dry run
* pdf2png.sh – PDFs kept by default, --delete, .PDF found, awkward filenames

Each test runs the real script with bash in a tmp_path sandbox. pdf2png.sh runs
against a stand-in `pdftoppm` put first on PATH, so poppler is not needed. The
Windows .bat counterparts are not exercised here (no cmd.exe on the CI runners).
Skipped when bash is not installed.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "data_scripts" / "unix"
SORT_SH = SCRIPTS / "sort.sh"
PDF2PNG_SH = SCRIPTS / "pdf2png.sh"

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash is not installed")


def _run(script: Path, *args, cwd: Path, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, str(script), *map(str, args)], cwd=cwd, env=env, capture_output=True, text=True, timeout=60
    )


def _files(root: Path) -> set:
    """Every file under *root*, as POSIX paths relative to it."""
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


# ════════════════════════════════════════════════════════════════════════════
# sort.sh
# ════════════════════════════════════════════════════════════════════════════
class TestSortSh:
    @pytest.fixture
    def pages(self, tmp_path) -> Path:
        """A converted-pages tree: two documents (Unix-padded and unpadded) and onepagers/."""
        root = tmp_path / "pages"
        for rel in ("docA/docA-01.png", "docA/docA-02.png", "docB/docB-008.png", "onepagers/single-01.png"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).touch()
        return root

    def _sort(self, tmp_path, pages, csv_text, *extra):
        csv_path = tmp_path / "ann.csv"
        csv_path.write_bytes(csv_text.encode("utf-8"))
        out = tmp_path / "out"
        result = _run(SORT_SH, "-i", pages, "-o", out, "-c", csv_path, *extra, cwd=tmp_path)
        return result, out

    def test_extra_columns_are_ignored_and_a_doi_does_not_nest_folders(self, tmp_path, pages):
        """Columns are found by header name: TITLE and DOI (with its "/") must not reach the label."""
        csv_text = "FILE,PAGE,CLASS,TITLE,DOI\ndocA,1,TEXT,Doc A,10.60585/x\n"
        result, out = self._sort(tmp_path, pages, csv_text)
        assert result.returncode == 0, result.stdout + result.stderr
        assert _files(out) == {"TEXT/docA-01.png"}

    def test_columns_may_come_in_any_order(self, tmp_path, pages):
        result, out = self._sort(tmp_path, pages, "CLASS,NOTE,PAGE,FILE\nDRAW,x,2,docA\n")
        assert result.returncode == 0, result.stdout + result.stderr
        assert _files(out) == {"DRAW/docA-02.png"}

    def test_classifier_output_header_is_accepted(self, tmp_path, pages):
        """A result table has CLASS-1 (and a CATEGORY alias) instead of CLASS."""
        csv_text = "FILE,PAGE,CLASS-1,SCORE-1,CATEGORY\ndocA,2,PHOTO,0.97,PHOTO\n"
        result, out = self._sort(tmp_path, pages, csv_text)
        assert result.returncode == 0, result.stdout + result.stderr
        assert _files(out) == {"PHOTO/docA-02.png"}

    def test_missing_page_creates_no_label_folder(self, tmp_path, pages):
        result, out = self._sort(tmp_path, pages, "FILE,PAGE,CLASS\ndocA,1,TEXT\ndocA,5,DRAW\n")
        assert result.returncode == 0
        assert "Not found: docA  page 5" in result.stdout
        assert sorted(p.name for p in out.iterdir()) == ["TEXT"]

    def test_zero_padded_page_is_read_as_decimal(self, tmp_path, pages):
        """08 used to be rejected as an invalid octal number by the padding printf."""
        result, out = self._sort(tmp_path, pages, "FILE,PAGE,CLASS\ndocB,08,TEXT_HW\n")
        assert result.returncode == 0, result.stdout + result.stderr
        assert _files(out) == {"TEXT_HW/docB-008.png"}

    def test_onepagers_fallback_tries_padded_names(self, tmp_path, pages):
        result, out = self._sort(tmp_path, pages, "FILE,PAGE,CLASS\nsingle,1,LINE_P\n")
        assert result.returncode == 0, result.stdout + result.stderr
        assert _files(out) == {"LINE_P/single-01.png"}

    def test_last_row_without_trailing_newline_is_kept(self, tmp_path, pages):
        result, out = self._sort(tmp_path, pages, "FILE,PAGE,CLASS\r\ndocA,1,TEXT\r\ndocA,2,DRAW")
        assert result.returncode == 0
        assert _files(out) == {"TEXT/docA-01.png", "DRAW/docA-02.png"}

    def test_quoted_field_with_a_comma(self, tmp_path, pages):
        result, out = self._sort(tmp_path, pages, 'FILE,PAGE,CLASS,TITLE\n"docA","1","TEXT","A, B"\n')
        assert result.returncode == 0, result.stdout + result.stderr
        assert _files(out) == {"TEXT/docA-01.png"}

    @pytest.mark.parametrize("label", ["../escape", "a/b", "..", ""])
    def test_unsafe_label_is_skipped(self, tmp_path, pages, label):
        result, out = self._sort(tmp_path, pages, f"FILE,PAGE,CLASS\ndocA,1,{label}\n")
        assert result.returncode == 0
        assert "Invalid label" in result.stdout
        assert "1 invalid row(s) skipped" in result.stdout
        assert _files(out) == set()
        assert not (tmp_path / "escape").exists()

    def test_non_numeric_page_is_skipped(self, tmp_path, pages):
        result, out = self._sort(tmp_path, pages, "FILE,PAGE,CLASS\ndocA,one,TEXT\n")
        assert result.returncode == 0
        assert "Invalid page" in result.stdout
        assert _files(out) == set()

    def test_missing_required_column_is_an_error(self, tmp_path, pages):
        result, _ = self._sort(tmp_path, pages, "FILE,PAGE,NOTE\ndocA,1,x\n")
        assert result.returncode == 1
        assert "must have FILE, PAGE and CLASS" in result.stdout

    def test_dry_run_changes_nothing(self, tmp_path, pages):
        before = _files(pages)
        result, out = self._sort(tmp_path, pages, "FILE,PAGE,CLASS\ndocA,1,TEXT\n", "--dry-run", "--move")
        assert result.returncode == 0
        assert "[dry-run] move:" in result.stdout
        assert not out.exists()
        assert _files(pages) == before

    def test_move_removes_the_source_page(self, tmp_path, pages):
        result, out = self._sort(tmp_path, pages, "FILE,PAGE,CLASS\ndocA,1,TEXT\n", "--move")
        assert result.returncode == 0
        assert _files(out) == {"TEXT/docA-01.png"}
        assert not (pages / "docA" / "docA-01.png").exists()


# ════════════════════════════════════════════════════════════════════════════
# pdf2png.sh
# ════════════════════════════════════════════════════════════════════════════
STUB_PDFTOPPM = """#!/bin/bash
# Stand-in for poppler's pdftoppm: `pdftoppm -png -r DPI IN.pdf OUT_PREFIX` writes two
# pages, OUT_PREFIX-1.png and OUT_PREFIX-2.png, and fails for a file containing FAIL.
pdf="${@: -2:1}"
prefix="${@: -1}"
[[ -f "$pdf" ]] || exit 1
grep -q FAIL "$pdf" && exit 1
touch "${prefix}-1.png" "${prefix}-2.png"
"""


@pytest.mark.skipif(shutil.which("nproc") is None, reason="pdf2png.sh needs nproc")
class TestPdf2PngSh:
    @pytest.fixture
    def env(self, tmp_path) -> dict:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "pdftoppm"
        stub.write_text(STUB_PDFTOPPM)
        stub.chmod(0o755)
        return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}

    @pytest.fixture
    def pdfs(self, tmp_path) -> Path:
        src = tmp_path / "pdfs"
        src.mkdir()
        for name in ("a.pdf", "B.PDF", 'we$HOME"ird.pdf'):
            (src / name).write_text("%PDF-1.4 stub")
        (src / "bad.pdf").write_text("FAIL")
        return src

    def test_pdfs_are_kept_by_default(self, tmp_path, env, pdfs):
        result = _run(PDF2PNG_SH, "--dir", pdfs, "--output", tmp_path / "out", cwd=tmp_path, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Keep PDFs  : yes" in result.stdout
        assert sorted(p.name for p in pdfs.iterdir()) == sorted(["a.pdf", "B.PDF", 'we$HOME"ird.pdf', "bad.pdf"])

    def test_upper_case_extension_and_awkward_names_are_converted(self, tmp_path, env, pdfs):
        """`.PDF` used to be skipped, and a `"` or `$` in a name broke the inner script."""
        out = tmp_path / "out"
        result = _run(PDF2PNG_SH, "--dir", pdfs, "--output", out, cwd=tmp_path, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert _files(out) == {
            "a/a-1.png",
            "a/a-2.png",
            "B/B-1.png",
            "B/B-2.png",
            'we$HOME"ird/we$HOME"ird-1.png',
            'we$HOME"ird/we$HOME"ird-2.png',
        }
        assert "Failed:    " in result.stdout and "bad.pdf" in result.stdout

    def test_delete_removes_only_converted_pdfs(self, tmp_path, env, pdfs):
        result = _run(PDF2PNG_SH, "--dir", pdfs, "--output", tmp_path / "out", "--delete", cwd=tmp_path, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert [p.name for p in pdfs.iterdir()] == ["bad.pdf"]

    def test_keep_flag_is_still_accepted(self, tmp_path, env, pdfs):
        result = _run(PDF2PNG_SH, "--dir", pdfs, "--output", tmp_path / "out", "-k", cwd=tmp_path, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(list(pdfs.iterdir())) == 4

    def test_empty_directory_is_reported(self, tmp_path, env):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = _run(PDF2PNG_SH, "--dir", empty, cwd=tmp_path, env=env)
        assert result.returncode == 0
        assert "No PDF files found" in result.stdout
