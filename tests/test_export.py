"""
Tests for Origin export system.

Tests cover:
- JSON export
- JSONL export
- MLflow format
- Weights & Biases format
- HuggingFace format
- Export validation
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from origin.export.formats import ProvenanceExporter
from origin.storage.database import ProvenanceDatabase
from origin.core.record import SourceRecord, BatchRecord
from origin.core.fingerprint import fingerprint_bytes, merkle_root
from tests.fixtures.sample_data import (
    create_test_source_record,
    create_test_provenance_record,
)
from tests.fixtures.sample_licenses import TEST_MIT_LICENSE


class ExportTestCase(unittest.TestCase):
    """Base class for export tests with database setup."""

    def setUp(self):
        """Create temporary database with test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = ProvenanceDatabase(self.db_path)
        self.session = self.db.begin_session("test_config_hash")

        # Add test data
        self.source = create_test_source_record()
        self.db.record_source(self.source)
        self.db.record_license(TEST_MIT_LICENSE)

        # Add samples and batch
        self.sample_ids = []
        for i in range(5):
            sample = create_test_provenance_record(
                content=f"sample_{i}".encode(),
                source_id=self.source.source_id,
            )
            self.db.record_sample(sample)
            self.sample_ids.append(sample.sample_id)

        self.batch = BatchRecord(
            batch_id=merkle_root(self.sample_ids),
            session_id=self.session.session_id,
            batch_index=0,
            sample_count=len(self.sample_ids),
            created_at="2025-01-15T14:30:00+00:00",
            sample_ids=tuple(self.sample_ids),
        )
        self.db.record_batch(self.batch)

        self.exporter = ProvenanceExporter(self.db)

    def tearDown(self):
        """Clean up."""
        self.db.close()
        # Clean up temp files
        for f in Path(self.temp_dir).glob("*"):
            f.unlink()
        os.rmdir(self.temp_dir)


class TestJSONExport(ExportTestCase):
    """Tests for JSON export format."""

    def test_export_returns_string(self):
        """Export should return JSON string."""
        result = self.exporter.to_json(self.session.session_id)
        self.assertIsInstance(result, str)

    def test_export_is_valid_json(self):
        """Export should be valid JSON."""
        result = self.exporter.to_json(self.session.session_id)
        parsed = json.loads(result)
        self.assertIsInstance(parsed, dict)

    def test_export_contains_session(self):
        """Export should contain session information."""
        result = self.exporter.to_json(self.session.session_id)
        parsed = json.loads(result)
        self.assertIn("session", parsed)
        self.assertEqual(
            parsed["session"]["session_id"], self.session.session_id
        )

    def test_export_contains_sources(self):
        """Export should contain source information."""
        result = self.exporter.to_json(self.session.session_id)
        parsed = json.loads(result)
        self.assertIn("sources", parsed)
        self.assertGreater(len(parsed["sources"]), 0)

    def test_export_contains_samples(self):
        """Export should contain sample information."""
        result = self.exporter.to_json(self.session.session_id)
        parsed = json.loads(result)
        self.assertIn("samples", parsed)
        self.assertEqual(len(parsed["samples"]), 5)

    def test_export_contains_batches(self):
        """Export should contain batch information."""
        result = self.exporter.to_json(self.session.session_id)
        parsed = json.loads(result)
        self.assertIn("batches", parsed)
        self.assertEqual(len(parsed["batches"]), 1)

    def test_export_contains_statistics(self):
        """Export should contain statistics."""
        result = self.exporter.to_json(self.session.session_id)
        parsed = json.loads(result)
        self.assertIn("statistics", parsed)
        self.assertEqual(parsed["statistics"]["total_samples"], 5)

    def test_export_contains_version(self):
        """Export should contain origin version."""
        result = self.exporter.to_json(self.session.session_id)
        parsed = json.loads(result)
        self.assertIn("origin_version", parsed)

    def test_export_pretty_format(self):
        """Pretty export should have indentation."""
        result = self.exporter.to_json(self.session.session_id, pretty=True)
        self.assertIn("\n", result)
        self.assertIn("  ", result)

    def test_export_compact_format(self):
        """Compact export should be single line."""
        result = self.exporter.to_json(self.session.session_id, pretty=False)
        # Should not have newlines in main structure (only in string values maybe)
        self.assertEqual(result.count("\n"), 0)

    def test_export_nonexistent_session(self):
        """Exporting nonexistent session should raise."""
        with self.assertRaises(ValueError):
            self.exporter.to_json("nonexistent-session")


class TestJSONLExport(ExportTestCase):
    """Tests for JSONL export format."""

    def test_export_creates_file(self):
        """JSONL export should create file."""
        output_path = Path(self.temp_dir) / "export.jsonl"
        self.exporter.to_jsonl(self.session.session_id, output_path)
        self.assertTrue(output_path.exists())

    def test_export_returns_line_count(self):
        """JSONL export should return line count."""
        output_path = Path(self.temp_dir) / "export.jsonl"
        lines = self.exporter.to_jsonl(self.session.session_id, output_path)
        self.assertGreater(lines, 0)

    def test_export_lines_are_valid_json(self):
        """Each line should be valid JSON."""
        output_path = Path(self.temp_dir) / "export.jsonl"
        self.exporter.to_jsonl(self.session.session_id, output_path)

        with open(output_path) as f:
            for line in f:
                parsed = json.loads(line)
                self.assertIsInstance(parsed, dict)

    def test_export_has_header(self):
        """First line should be header with session info."""
        output_path = Path(self.temp_dir) / "export.jsonl"
        self.exporter.to_jsonl(self.session.session_id, output_path)

        with open(output_path) as f:
            header = json.loads(f.readline())
        self.assertEqual(header["type"], "header")
        self.assertIn("session", header)

    def test_export_has_footer(self):
        """Last line should be footer with statistics."""
        output_path = Path(self.temp_dir) / "export.jsonl"
        self.exporter.to_jsonl(self.session.session_id, output_path)

        with open(output_path) as f:
            lines = f.readlines()
        footer = json.loads(lines[-1])
        self.assertEqual(footer["type"], "footer")
        self.assertIn("statistics", footer)

    def test_export_has_sources(self):
        """Should have source records."""
        output_path = Path(self.temp_dir) / "export.jsonl"
        self.exporter.to_jsonl(self.session.session_id, output_path)

        source_lines = []
        with open(output_path) as f:
            for line in f:
                parsed = json.loads(line)
                if parsed.get("type") == "source":
                    source_lines.append(parsed)
        self.assertGreater(len(source_lines), 0)


class TestMLflowExport(ExportTestCase):
    """Tests for MLflow export format."""

    def test_export_returns_dict(self):
        """MLflow export should return dict."""
        result = self.exporter.to_mlflow(self.session.session_id)
        self.assertIsInstance(result, dict)

    def test_export_has_tags(self):
        """MLflow export should have tags."""
        result = self.exporter.to_mlflow(self.session.session_id)
        self.assertIn("tags", result)
        self.assertIn("origin.session_id", result["tags"])

    def test_export_has_params(self):
        """MLflow export should have params."""
        result = self.exporter.to_mlflow(self.session.session_id)
        self.assertIn("params", result)
        self.assertIn("origin.sample_count", result["params"])

    def test_export_has_artifacts(self):
        """MLflow export should have artifacts."""
        result = self.exporter.to_mlflow(self.session.session_id)
        self.assertIn("artifacts", result)
        self.assertIn("provenance_card.md", result["artifacts"])

    def test_export_session_id_tag(self):
        """Session ID should be in tags."""
        result = self.exporter.to_mlflow(self.session.session_id)
        self.assertEqual(
            result["tags"]["origin.session_id"], self.session.session_id
        )

    def test_export_data_fingerprint_tag(self):
        """Data fingerprint should be in tags."""
        result = self.exporter.to_mlflow(self.session.session_id)
        self.assertIn("origin.data_fingerprint", result["tags"])
        # Fingerprint should be 64 hex chars
        fp = result["tags"]["origin.data_fingerprint"]
        self.assertEqual(len(fp), 64)


class TestWandBExport(ExportTestCase):
    """Tests for Weights & Biases export format."""

    def test_export_returns_dict(self):
        """W&B export should return dict."""
        result = self.exporter.to_wandb(self.session.session_id)
        self.assertIsInstance(result, dict)

    def test_export_has_metadata(self):
        """W&B export should have metadata."""
        result = self.exporter.to_wandb(self.session.session_id)
        self.assertIn("metadata", result)

    def test_export_has_description(self):
        """W&B export should have description."""
        result = self.exporter.to_wandb(self.session.session_id)
        self.assertIn("description", result)

    def test_metadata_has_session_id(self):
        """Metadata should contain session ID."""
        result = self.exporter.to_wandb(self.session.session_id)
        self.assertEqual(
            result["metadata"]["origin_session_id"], self.session.session_id
        )

    def test_metadata_has_counts(self):
        """Metadata should contain counts."""
        result = self.exporter.to_wandb(self.session.session_id)
        self.assertEqual(result["metadata"]["sample_count"], 5)
        self.assertEqual(result["metadata"]["batch_count"], 1)

    def test_description_is_string(self):
        """Description should be string."""
        result = self.exporter.to_wandb(self.session.session_id)
        self.assertIsInstance(result["description"], str)


class TestHuggingFaceExport(ExportTestCase):
    """Tests for HuggingFace export format."""

    def test_export_returns_dict(self):
        """HF export should return dict."""
        result = self.exporter.to_huggingface(self.session.session_id)
        self.assertIsInstance(result, dict)

    def test_export_has_datasets(self):
        """HF export should have datasets list."""
        result = self.exporter.to_huggingface(self.session.session_id)
        self.assertIn("datasets", result)
        self.assertIsInstance(result["datasets"], list)

    def test_export_has_license(self):
        """HF export should have license."""
        result = self.exporter.to_huggingface(self.session.session_id)
        self.assertIn("license", result)

    def test_export_has_training_data(self):
        """HF export should have training_data section."""
        result = self.exporter.to_huggingface(self.session.session_id)
        self.assertIn("training_data", result)

    def test_training_data_has_counts(self):
        """Training data should have sample counts."""
        result = self.exporter.to_huggingface(self.session.session_id)
        self.assertEqual(result["training_data"]["samples"], 5)

    def test_training_data_has_fingerprint(self):
        """Training data should have provenance fingerprint."""
        result = self.exporter.to_huggingface(self.session.session_id)
        self.assertIn("provenance_fingerprint", result["training_data"])


class TestEmptySessionExport(ExportTestCase):
    """Tests for exporting empty session."""

    def setUp(self):
        """Create database with empty session."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = ProvenanceDatabase(self.db_path)
        self.session = self.db.begin_session("empty_config")
        self.exporter = ProvenanceExporter(self.db)

    def test_json_export_empty_session(self):
        """Empty session should export as valid JSON."""
        result = self.exporter.to_json(self.session.session_id)
        parsed = json.loads(result)
        self.assertEqual(len(parsed["samples"]), 0)

    def test_mlflow_export_empty_session(self):
        """Empty session should export for MLflow."""
        result = self.exporter.to_mlflow(self.session.session_id)
        self.assertEqual(result["params"]["origin.sample_count"], "0")

    def test_wandb_export_empty_session(self):
        """Empty session should export for W&B."""
        result = self.exporter.to_wandb(self.session.session_id)
        self.assertEqual(result["metadata"]["sample_count"], 0)


class TestExportWithConflicts(ExportTestCase):
    """Tests for exporting sessions with license conflicts."""

    def test_json_includes_conflicts(self):
        """JSON export should include conflicts."""
        self.db.record_conflict(
            self.session.session_id, "MIT", "GPL-3.0", "copyleft_mix"
        )
        result = self.exporter.to_json(self.session.session_id)
        parsed = json.loads(result)
        self.assertIn("conflicts", parsed)
        self.assertEqual(len(parsed["conflicts"]), 1)

    def test_mlflow_has_conflicts_tag(self):
        """MLflow export should indicate conflicts in tags."""
        self.db.record_conflict(
            self.session.session_id, "MIT", "GPL-3.0", "copyleft_mix"
        )
        result = self.exporter.to_mlflow(self.session.session_id)
        self.assertEqual(result["tags"]["origin.has_conflicts"], "true")

    def test_wandb_has_conflicts(self):
        """W&B export should indicate conflicts."""
        self.db.record_conflict(
            self.session.session_id, "MIT", "GPL-3.0", "copyleft_mix"
        )
        result = self.exporter.to_wandb(self.session.session_id)
        self.assertTrue(result["metadata"]["has_conflicts"])

    def test_huggingface_has_conflicts(self):
        """HF export should list conflict summaries."""
        self.db.record_conflict(
            self.session.session_id, "MIT", "GPL-3.0", "copyleft_mix"
        )
        result = self.exporter.to_huggingface(self.session.session_id)
        self.assertIn("license_conflicts", result["training_data"])
        self.assertGreater(len(result["training_data"]["license_conflicts"]), 0)


if __name__ == "__main__":
    unittest.main()
