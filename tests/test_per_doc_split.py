"""
tests/test_per_doc_split.py
===========================
Unit tests for supplementary/per_doc_split.py.

Scope
-----
* split_csv_and_aggregate – output directory creation, one file per document,
  header preservation, accumulation of multiple rows for the same document,
  filename sanitisation, empty-file and missing-file edge cases.

No GPU, no trained model, no network required.
"""
import csv
from pathlib import Path

from per_doc_split import split_csv_and_aggregate

# ── helpers ────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


# ════════════════════════════════════════════════════════════════════════════
# split_csv_and_aggregate
# ════════════════════════════════════════════════════════════════════════════
class TestSplitCsvAndAggregate:

    def test_output_directory_created_when_absent(self, tmp_path):
        out_dir = tmp_path / "new_subdir"
        inp = tmp_path / "input.csv"
        write_csv(inp, [["FILE", "PAGE", "CLASS-1"], ["doc", "1", "TEXT"]])
        split_csv_and_aggregate(str(inp), str(out_dir))
        assert out_dir.is_dir()

    def test_one_output_file_per_unique_document(self, tmp_path):
        inp = tmp_path / "input.csv"
        write_csv(inp, [
            ["FILE", "PAGE", "CLASS-1"],
            ["docA", "1", "TEXT"],
            ["docB", "2", "LINE_P"],
            ["docA", "3", "DRAW"],
        ])
        out_dir = tmp_path / "out"
        split_csv_and_aggregate(str(inp), str(out_dir))
        assert len(list(out_dir.glob("*.csv"))) == 2

    def test_output_filename_matches_file_column_value(self, tmp_path):
        inp = tmp_path / "input.csv"
        write_csv(inp, [["FILE", "PAGE", "CLASS-1"], ["thesis", "1", "TEXT"]])
        out_dir = tmp_path / "out"
        split_csv_and_aggregate(str(inp), str(out_dir))
        assert (out_dir / "thesis.csv").exists()

    def test_header_preserved_in_every_output_file(self, tmp_path):
        inp = tmp_path / "input.csv"
        header = ["FILE", "PAGE", "CLASS-1", "SCORE-1"]
        write_csv(inp, [header, ["doc", "1", "TEXT", "0.9"]])
        out_dir = tmp_path / "out"
        split_csv_and_aggregate(str(inp), str(out_dir))
        rows = read_csv(out_dir / "doc.csv")
        assert rows[0] == header

    def test_all_rows_for_document_written(self, tmp_path):
        inp = tmp_path / "input.csv"
        write_csv(inp, [
            ["FILE", "PAGE", "CLASS-1"],
            ["doc", "1", "TEXT"],
            ["doc", "2", "LINE_P"],
            ["doc", "3", "DRAW"],
        ])
        out_dir = tmp_path / "out"
        split_csv_and_aggregate(str(inp), str(out_dir))
        rows = read_csv(out_dir / "doc.csv")
        # header + 3 data rows
        assert len(rows) == 4

    def test_rows_from_different_documents_not_mixed(self, tmp_path):
        inp = tmp_path / "input.csv"
        write_csv(inp, [
            ["FILE", "PAGE", "CLASS-1"],
            ["docA", "1", "TEXT"],
            ["docB", "1", "DRAW"],
        ])
        out_dir = tmp_path / "out"
        split_csv_and_aggregate(str(inp), str(out_dir))
        rows_a = read_csv(out_dir / "docA.csv")
        rows_b = read_csv(out_dir / "docB.csv")
        # each file has header + exactly 1 data row
        assert len(rows_a) == 2
        assert len(rows_b) == 2
        assert rows_a[1][2] == "TEXT"
        assert rows_b[1][2] == "DRAW"

    def test_forward_slash_in_doc_name_replaced_with_underscore(self, tmp_path):
        inp = tmp_path / "input.csv"
        write_csv(inp, [["FILE", "PAGE", "CLASS-1"], ["a/b", "1", "TEXT"]])
        out_dir = tmp_path / "out"
        split_csv_and_aggregate(str(inp), str(out_dir))
        assert (out_dir / "a_b.csv").exists()

    def test_backslash_in_doc_name_replaced_with_underscore(self, tmp_path):
        inp = tmp_path / "input.csv"
        write_csv(inp, [["FILE", "PAGE", "CLASS-1"], ["a\\b", "1", "TEXT"]])
        out_dir = tmp_path / "out"
        split_csv_and_aggregate(str(inp), str(out_dir))
        assert (out_dir / "a_b.csv").exists()

    def test_total_data_row_count_matches_source(self, tmp_path):
        inp = tmp_path / "input.csv"
        data = [["FILE", "PAGE", "CLASS-1"]] + [[f"doc{i}", str(i), "TEXT"] for i in range(10)]
        write_csv(inp, data)
        out_dir = tmp_path / "out"
        split_csv_and_aggregate(str(inp), str(out_dir))
        total = sum(len(read_csv(f)) - 1 for f in out_dir.glob("*.csv"))
        assert total == 10

    def test_empty_input_file_handled_gracefully(self, tmp_path):
        inp = tmp_path / "empty.csv"
        inp.touch()
        out_dir = tmp_path / "out"
        split_csv_and_aggregate(str(inp), str(out_dir))
        # No output files; no exception raised

    def test_missing_input_file_prints_error_without_raising(self, tmp_path, capsys):
        out_dir = tmp_path / "out"
        split_csv_and_aggregate(str(tmp_path / "ghost.csv"), str(out_dir))
        captured = capsys.readouterr()
        assert "error" in captured.out.lower() or "not found" in captured.out.lower()

    def test_many_pages_same_document_all_written(self, tmp_path):
        inp = tmp_path / "input.csv"
        rows = [["FILE", "PAGE", "CLASS-1"]] + [["bigdoc", str(i), "TEXT"] for i in range(1, 51)]
        write_csv(inp, rows)
        out_dir = tmp_path / "out"
        split_csv_and_aggregate(str(inp), str(out_dir))
        result = read_csv(out_dir / "bigdoc.csv")
        assert len(result) == 51  # header + 50 data rows
