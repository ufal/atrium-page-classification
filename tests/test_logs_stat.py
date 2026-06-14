"""
tests/test_logs_stat.py
=======================
Unit tests for supplementary/logs_stat.py.

Scope
-----
Pure helper functions that require no TensorBoard event files or GPU:
* parse_log_folder_name     – folder-name parsing
* get_base_model_from_revision – revision → base model lookup
* get_model_details         – model-catalog lookup
* extract_gpu_node          – GPU node name extraction from file paths
* _iter_tfrecords           – TFRecord binary format parser

Tests that require real TensorBoard event files or a GPU are marked slow
and excluded from the default run.

No GPU, no trained model, no network required.
"""
import struct
from pathlib import Path

import pytest

from logs_stat import (
    _iter_tfrecords,
    extract_gpu_node,
    get_base_model_from_revision,
    get_model_details,
    parse_log_folder_name,
)


# ════════════════════════════════════════════════════════════════════════════
# parse_log_folder_name
# ════════════════════════════════════════════════════════════════════════════
class TestParseLogFolderName:
    """Folder-name parser returns a dict with known keys for all inputs."""

    def test_returns_dict_with_all_required_keys(self):
        result = parse_log_folder_name("any_folder")
        required = {"script", "timestamp", "learning_rate", "epochs",
                    "model", "revision", "batch_size"}
        assert required.issubset(result.keys())

    def test_detects_classifierpy_script(self):
        result = parse_log_folder_name(
            "classifierpy_2024-01-01_120000_lr=5e-5-e=3-m=vit-base-model_v23"
        )
        assert result["script"] == "classifier.py"

    def test_detects_runpy_script(self):
        result = parse_log_folder_name("run.py_2024-01-01_120000")
        assert result["script"] == "run.py"

    def test_extracts_revision_from_model_v_prefix(self):
        result = parse_log_folder_name("classifierpy_2024-01-01_120000-model_v43")
        assert result["revision"] == "v43"

    def test_extracts_learning_rate_with_underscore_notation(self):
        result = parse_log_folder_name("classifierpy_2024-01-01_lr=5e_5-e=3")
        assert result["learning_rate"] == "5e-5"

    def test_extracts_epoch_count(self):
        result = parse_log_folder_name("classifierpy_2024-01-01-e=10")
        assert result["epochs"] == "10"

    def test_extracts_batch_size(self):
        result = parse_log_folder_name("classifierpy_bs=32_model_v43")
        assert result["batch_size"] == "32"

    def test_unknown_folder_has_none_script_and_revision(self):
        result = parse_log_folder_name("completely_unknown_folder_name")
        assert result["script"] is None
        assert result["revision"] is None

    def test_all_values_are_none_or_string(self):
        result = parse_log_folder_name("partial_folder-e=5")
        for v in result.values():
            assert v is None or isinstance(v, str)


# ════════════════════════════════════════════════════════════════════════════
# get_base_model_from_revision
# ════════════════════════════════════════════════════════════════════════════
class TestGetBaseModelFromRevision:
    """Maps version prefix strings to HuggingFace model identifiers."""

    def test_v23_maps_to_vit_base_224(self):
        assert "vit-base-patch16-224" in get_base_model_from_revision("v23")

    def test_v33_maps_to_vit_base_384(self):
        assert "vit-base-patch16-384" in get_base_model_from_revision("v33")

    def test_v53_maps_to_vit_large_384(self):
        assert "vit-large-patch16-384" in get_base_model_from_revision("v53")

    def test_v73_maps_to_regnety(self):
        model = get_base_model_from_revision("v73")
        assert "regnety" in model

    def test_unknown_revision_returns_none(self):
        assert get_base_model_from_revision("v999") is None

    def test_none_input_returns_none(self):
        assert get_base_model_from_revision(None) is None

    def test_empty_string_returns_none(self):
        assert get_base_model_from_revision("") is None

    def test_partial_prefix_match_works(self):
        """A longer revision string that *starts with* a known prefix is matched."""
        model = get_base_model_from_revision("v23fold5")
        assert model is not None


# ════════════════════════════════════════════════════════════════════════════
# get_model_details
# ════════════════════════════════════════════════════════════════════════════
class TestGetModelDetails:
    """Looks up model metadata (parameters, family, accuracy) from the catalog."""

    def test_known_model_returns_dict(self):
        details = get_model_details("google/vit-base-patch16-224")
        assert isinstance(details, dict)

    def test_parameters_m_field_present_and_positive(self):
        details = get_model_details("timm/regnety_160.swag_ft_in1k")
        assert "parameters_m" in details
        assert details["parameters_m"] > 0

    def test_family_field_is_a_string(self):
        details = get_model_details("google/vit-large-patch16-384")
        assert isinstance(details["family"], str)

    def test_input_size_field_present(self):
        details = get_model_details("google/vit-base-patch16-384")
        assert "input_size" in details

    def test_accuracy_field_is_between_90_and_100(self):
        details = get_model_details("timm/regnety_160.swag_ft_in1k")
        assert 90.0 < details["accuracy_top1"] <= 100.0

    def test_unknown_model_returns_empty_dict(self):
        assert get_model_details("org/no-such-model") == {}

    def test_none_input_returns_empty_dict(self):
        assert get_model_details(None) == {}


# ════════════════════════════════════════════════════════════════════════════
# extract_gpu_node
# ════════════════════════════════════════════════════════════════════════════
class TestExtractGpuNode:
    """Extracts the GPU node name from a TensorBoard event file path."""

    def test_standard_dll_node(self):
        assert extract_gpu_node("/lnet/logs/dll-3gpu2/events.out.tfevents.1234") == "dll-3gpu2"

    def test_tdll_variant_node(self):
        assert extract_gpu_node("/lnet/logs/tdll-8gpu1/events.out.tfevents.5678") == "tdll-8gpu1"

    def test_multi_digit_gpu_number(self):
        assert extract_gpu_node("/logs/dll-10gpu3/events.out.tfevents.000") == "dll-10gpu3"

    def test_node_embedded_in_longer_path(self):
        path = "/home/user/projects/atrium/logs/dll-4gpu1/run/events.out.tfevents.999"
        assert extract_gpu_node(path) == "dll-4gpu1"

    def test_returns_none_when_no_node_in_path(self):
        assert extract_gpu_node("/some/path/without/gpu/node/events.bin") is None

    def test_returns_none_for_empty_string(self):
        assert extract_gpu_node("") is None


# ════════════════════════════════════════════════════════════════════════════
# _iter_tfrecords
# ════════════════════════════════════════════════════════════════════════════
class TestIterTfrecords:
    """Parses TFRecord binary framing without TensorFlow.

    TFRecord wire format per record:
        [8-byte uint64 LE length][4-byte masked CRC32 of length]
        [<length> bytes payload]
        [4-byte masked CRC32 of payload]

    CRC bytes are zeroed in the helper below because _iter_tfrecords skips
    CRC validation (which is intentional — see docstring in logs_stat.py).
    """

    @staticmethod
    def _frame(payload: bytes) -> bytes:
        """Wrap raw bytes in a minimal valid TFRecord frame (CRC zeroed)."""
        length = len(payload)
        return (
            struct.pack('<Q', length)  # 8-byte length
            + b'\x00\x00\x00\x00'     # 4-byte CRC of length (skipped)
            + payload
            + b'\x00\x00\x00\x00'     # 4-byte CRC of payload (skipped)
        )

    def test_single_record_yielded(self, tmp_path):
        payload = b"hello world"
        f = tmp_path / "events.out.tfevents.test"
        f.write_bytes(self._frame(payload))
        records = list(_iter_tfrecords(f))
        assert records == [payload]

    def test_multiple_records_yielded_in_order(self, tmp_path):
        payloads = [b"record1", b"record2", b"record3"]
        f = tmp_path / "events.out.tfevents.multi"
        f.write_bytes(b"".join(self._frame(p) for p in payloads))
        assert list(_iter_tfrecords(f)) == payloads

    def test_empty_file_yields_nothing(self, tmp_path):
        f = tmp_path / "empty.tfevents"
        f.write_bytes(b"")
        assert list(_iter_tfrecords(f)) == []

    def test_truncated_length_header_terminates_cleanly(self, tmp_path):
        """Only 3 of the required 12 header bytes present → iterator stops
        gracefully without raising an exception."""
        f = tmp_path / "truncated.tfevents"
        f.write_bytes(b"\x05\x00\x00")   # incomplete header
        records = list(_iter_tfrecords(f))
        assert records == []

    def test_truncated_payload_terminates_cleanly(self, tmp_path):
        """Length header claims 100 bytes but file contains only 10 → stop."""
        f = tmp_path / "short_payload.tfevents"
        length_says_100 = struct.pack('<Q', 100) + b'\x00\x00\x00\x00'
        f.write_bytes(length_says_100 + b'\x00' * 10)   # only 10 payload bytes
        records = list(_iter_tfrecords(f))
        assert records == []

    def test_binary_payload_preserved_exactly(self, tmp_path):
        """Non-ASCII bytes must survive the round-trip without alteration."""
        payload = bytes(range(256))
        f = tmp_path / "binary.tfevents"
        f.write_bytes(self._frame(payload))
        records = list(_iter_tfrecords(f))
        assert records[0] == payload

    def test_empty_payload_record_yielded(self, tmp_path):
        """A valid record containing zero payload bytes is a legal TFRecord."""
        f = tmp_path / "zero_payload.tfevents"
        f.write_bytes(self._frame(b""))
        records = list(_iter_tfrecords(f))
        assert records == [b""]