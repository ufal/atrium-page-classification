"""
tests/test_paradata.py
======================
Unit tests for atrium_paradata.py (ParadataLogger and _sanitise helper).

Design notes
------------
* No ML models, no network, no GPU required.
* This file is **intentionally identical** across all four ATRIUM repositories;
  only PROGRAM_NAME at the top changes per repo.
* All file I/O uses pytest's `tmp_path` fixture so tests are hermetic.
"""
import json
import time
from pathlib import Path

import pytest

from atrium_paradata import ParadataLogger, _sanitise

# ── Repo-specific constant ───────────────────────────────────────────────────
PROGRAM_NAME = "page-classification"


# ════════════════════════════════════════════════════════════════════════════
# _sanitise
# ════════════════════════════════════════════════════════════════════════════
class TestSanitise:
    """_sanitise must produce a JSON-serialisable structure from arbitrary input."""

    def test_primitives_pass_through_unchanged(self):
        d = {"i": 1, "f": 3.14, "b": True, "s": "hello", "n": None}
        assert _sanitise(d) == d

    def test_tuple_converted_to_list(self):
        assert _sanitise((1, 2, 3)) == [1, 2, 3]

    def test_nested_structures_recursed(self):
        result = _sanitise({"a": {"b": [1, (2, 3)]}})
        assert result == {"a": {"b": [1, [2, 3]]}}

    def test_non_serialisable_value_becomes_string(self):
        class Opaque:
            def __repr__(self):
                return "Opaque()"

        result = _sanitise({"x": Opaque()})
        assert isinstance(result["x"], str)

    def test_integer_dict_key_coerced_to_string(self):
        result = _sanitise({42: "val"})
        assert "42" in result

    def test_deep_nesting_does_not_raise(self):
        """Depth guard (> 10 levels) must return a string, not raise."""
        node: dict = {}
        inner = node
        for _ in range(13):   # deliberately exceed the 10-level limit
            inner["k"] = {}
            inner = inner["k"]
        inner["leaf"] = object()   # non-serialisable at depth > 10
        result = _sanitise(node)
        # Top-level dict is intact; deep value coerced to str somewhere
        assert isinstance(result, dict)


# ════════════════════════════════════════════════════════════════════════════
# ParadataLogger – lifecycle
# ════════════════════════════════════════════════════════════════════════════
class TestParadataLoggerLifecycle:

    def test_finalize_creates_json_file(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {"k": "v"}, paradata_dir=str(tmp_path))
        path = logger.finalize(input_total=10)
        assert Path(path).exists()

    def test_json_file_is_valid(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path))
        path = logger.finalize()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_program_field_matches_constructor_arg(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path))
        path = logger.finalize()
        data = json.loads(Path(path).read_text())
        assert data["program"] == PROGRAM_NAME

    def test_license_field_present(self, tmp_path):
        # config_dir isolated so the repo's para_config.txt is NOT discovered;
        # deterministically exercises the conservative fallback license block
        # regardless of which repo runs this (shared) test.
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path),
                                config_dir=str(tmp_path))
        path = logger.finalize()
        data = json.loads(Path(path).read_text())
        assert "CC BY-NC 4.0" in data["license"]
        assert "creativecommons" in data["license_url"]

    def test_explicit_input_total_stored(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path))
        path = logger.finalize(input_total=42)
        data = json.loads(Path(path).read_text())
        assert data["statistics"]["input_files_total"] == 42

    def test_input_total_inferred_from_successes_and_skips(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, output_types=["csv"],
                                paradata_dir=str(tmp_path))
        logger.log_success("csv", 8)
        logger.log_skip("bad.png", "unreadable")
        path = logger.finalize()   # input_total=None → infer
        data = json.loads(Path(path).read_text())
        # processed(8) + skipped(1) = 9
        assert data["statistics"]["input_files_total"] == 9

    def test_double_finalize_raises_runtime_error(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path))
        logger.finalize()
        with pytest.raises(RuntimeError):
            logger.finalize()

    def test_paradata_dir_created_automatically(self, tmp_path):
        new_dir = tmp_path / "deep" / "paradata"
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(new_dir))
        logger.finalize()
        assert new_dir.is_dir()

    def test_filename_contains_run_id_and_program(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path))
        path = logger.finalize()
        stem = Path(path).stem  # e.g. "260315-120442_page-classification"
        assert logger._run_id in stem
        assert PROGRAM_NAME in stem

    def test_timing_fields_present_and_non_negative(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path))
        time.sleep(0.01)   # ensure measurable duration
        path = logger.finalize()
        data = json.loads(Path(path).read_text())
        assert "start_time" in data
        assert "end_time" in data
        assert data["duration_seconds"] >= 0


# ════════════════════════════════════════════════════════════════════════════
# ParadataLogger – success / skip counters
# ════════════════════════════════════════════════════════════════════════════
class TestParadataLoggerCounters:

    def test_log_success_single_call(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, output_types=["csv"],
                                paradata_dir=str(tmp_path))
        logger.log_success("csv", count=5)
        path = logger.finalize()
        data = json.loads(Path(path).read_text())
        assert data["statistics"]["output_counts_by_type"]["csv"] == 5

    def test_log_success_accumulates_across_calls(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, output_types=["csv"],
                                paradata_dir=str(tmp_path))
        logger.log_success("csv", count=3)
        logger.log_success("csv")          # default count=1
        path = logger.finalize()
        data = json.loads(Path(path).read_text())
        assert data["statistics"]["output_counts_by_type"]["csv"] == 4

    def test_log_success_multiple_output_types(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, output_types=["csv", "png"],
                                paradata_dir=str(tmp_path))
        logger.log_success("csv", 10)
        logger.log_success("png", 3)
        path = logger.finalize()
        data = json.loads(Path(path).read_text())
        counts = data["statistics"]["output_counts_by_type"]
        assert counts["csv"] == 10
        assert counts["png"] == 3

    def test_log_success_new_type_registered_at_runtime(self, tmp_path):
        """output_types not declared at construction time can still be logged."""
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path))
        logger.log_success("xml", 7)
        path = logger.finalize()
        data = json.loads(Path(path).read_text())
        assert data["statistics"]["output_counts_by_type"]["xml"] == 7

    def test_log_skip_recorded_in_detail(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path))
        logger.log_skip("corrupt.png", "PIL cannot open")
        path = logger.finalize()
        data = json.loads(Path(path).read_text())
        detail = data["skipped_files_detail"]
        assert len(detail) == 1
        assert detail[0]["file"] == "corrupt.png"
        assert detail[0]["reason"] == "PIL cannot open"

    def test_multiple_skips_counted_correctly(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path))
        for i in range(5):
            logger.log_skip(f"bad_{i}.png", f"error {i}")
        path = logger.finalize()
        data = json.loads(Path(path).read_text())
        assert data["statistics"]["skipped_files"] == 5
        assert len(data["skipped_files_detail"]) == 5

    def test_zero_successes_zero_skips_by_default(self, tmp_path):
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path))
        path = logger.finalize(input_total=0)
        data = json.loads(Path(path).read_text())
        assert data["statistics"]["skipped_files"] == 0
        assert data["statistics"]["successfully_processed"] == 0


# ════════════════════════════════════════════════════════════════════════════
# ParadataLogger – context manager
# ════════════════════════════════════════════════════════════════════════════
class TestParadataLoggerContextManager:

    def test_with_block_writes_json_on_clean_exit(self, tmp_path):
        with ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path)) as logger:
            logger.log_success("csv", 5)
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_with_block_finalizes_even_on_exception(self, tmp_path):
        """Log must be written even when the body raises."""
        try:
            with ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path)) as logger:
                logger.log_success("csv", 1)
                raise ValueError("deliberate test error")
        except ValueError:
            pass
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_with_block_does_not_suppress_exceptions(self, tmp_path):
        with pytest.raises(ValueError, match="deliberate"):
            with ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path)):
                raise ValueError("deliberate")


# ════════════════════════════════════════════════════════════════════════════
# ParadataLogger – config snapshot
# ════════════════════════════════════════════════════════════════════════════
class TestParadataLoggerConfigSnapshot:

    def test_config_dict_persisted(self, tmp_path):
        cfg = {"model": "regnety_160", "batch_size": 16, "top_n": 3}
        logger = ParadataLogger(PROGRAM_NAME, cfg, paradata_dir=str(tmp_path))
        path = logger.finalize()
        data = json.loads(Path(path).read_text())
        assert data["config"]["model"] == "regnety_160"
        assert data["config"]["batch_size"] == 16

    def test_non_serialisable_config_values_coerced(self, tmp_path):
        class Opaque:
            pass

        logger = ParadataLogger(PROGRAM_NAME, {"obj": Opaque()},
                                paradata_dir=str(tmp_path))
        path = logger.finalize()
        data = json.loads(Path(path).read_text())
        assert isinstance(data["config"]["obj"], str)

    def test_python_version_recorded(self, tmp_path):
        import sys as _sys
        logger = ParadataLogger(PROGRAM_NAME, {}, paradata_dir=str(tmp_path))
        path = logger.finalize()
        data = json.loads(Path(path).read_text())
        assert _sys.version in data["python_version"]