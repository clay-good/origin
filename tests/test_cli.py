"""
Tests for Origin CLI interface.

Tests cover:
- Argument parsing
- All CLI commands
- Exit codes
- Output formats (text and JSON)
"""

import json
import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from origin.cli.main import main, create_parser
from origin.cli import commands
from origin.storage.database import ProvenanceDatabase
from origin.core.record import SourceRecord, BatchRecord
from origin.core.fingerprint import fingerprint_bytes, merkle_root
from tests.fixtures.sample_data import create_test_provenance_record
from tests.fixtures.sample_licenses import TEST_MIT_LICENSE


class TestArgumentParser(unittest.TestCase):
    """Tests for CLI argument parsing."""

    def test_parser_creation(self):
        """Parser should be created successfully."""
        parser = create_parser()
        self.assertIsNotNone(parser)

    def test_version_argument(self):
        """--version should work."""
        parser = create_parser()
        with self.assertRaises(SystemExit) as cm:
            parser.parse_args(["--version"])
        self.assertEqual(cm.exception.code, 0)

    def test_help_argument(self):
        """--help should work."""
        parser = create_parser()
        with self.assertRaises(SystemExit) as cm:
            parser.parse_args(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_path_argument(self):
        """--path should set database path."""
        parser = create_parser()
        args = parser.parse_args(["--path", "/custom/path.db", "status"])
        self.assertEqual(args.path, "/custom/path.db")

    def test_quiet_argument(self):
        """--quiet should set quiet flag."""
        parser = create_parser()
        args = parser.parse_args(["--quiet", "status"])
        self.assertTrue(args.quiet)

    def test_json_argument(self):
        """--json should set json flag."""
        parser = create_parser()
        args = parser.parse_args(["--json", "status"])
        self.assertTrue(args.json)

    def test_subcommands_exist(self):
        """All expected subcommands should exist."""
        parser = create_parser()
        # Commands with required arguments
        commands_with_args = {
            "inspect": ["inspect", "session_id"],
            "query": ["query", "license_id"],
            "card": ["card", "session_id"],
            "export": ["export", "session_id"],
            "trace": ["trace", "sample_id"],
        }
        # Commands without required arguments
        simple_commands = ["init", "status", "sessions", "conflicts", "version"]

        for cmd in simple_commands:
            args = parser.parse_args([cmd])
            self.assertEqual(args.command, cmd)

        for cmd, argv in commands_with_args.items():
            args = parser.parse_args(argv)
            self.assertEqual(args.command, cmd)


class CLITestCase(unittest.TestCase):
    """Base class for CLI tests with database setup."""

    def setUp(self):
        """Create temporary directory for test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test.db")
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self):
        """Clean up."""
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        # Clean up temp files
        for f in Path(self.temp_dir).glob("*"):
            f.unlink()
        os.rmdir(self.temp_dir)

    def get_stdout(self):
        """Get captured stdout."""
        sys.stdout.seek(0)
        return sys.stdout.read()

    def get_stderr(self):
        """Get captured stderr."""
        sys.stderr.seek(0)
        return sys.stderr.read()


class TestInitCommand(CLITestCase):
    """Tests for init command."""

    def test_init_creates_database(self):
        """Init should create database file."""
        exit_code = main(["--path", self.db_path, "init"])
        self.assertEqual(exit_code, 0)
        self.assertTrue(Path(self.db_path).exists())

    def test_init_existing_database(self):
        """Init on existing database should succeed."""
        # Create database
        main(["--path", self.db_path, "init"])
        # Init again
        exit_code = main(["--path", self.db_path, "init"])
        self.assertEqual(exit_code, 0)


class TestStatusCommand(CLITestCase):
    """Tests for status command."""

    def test_status_requires_database(self):
        """Status should fail on nonexistent database."""
        exit_code = main(["--path", self.db_path, "status"])
        self.assertEqual(exit_code, 1)

    def test_status_shows_statistics(self):
        """Status should show statistics."""
        main(["--path", self.db_path, "init"])
        exit_code = main(["--path", self.db_path, "status"])
        self.assertEqual(exit_code, 0)
        output = self.get_stdout()
        self.assertIn("Sessions", output)

    def test_status_json_format(self):
        """Status with --json should output JSON."""
        main(["--path", self.db_path, "init"])
        # Reset stdout to capture only status output
        sys.stdout = StringIO()
        exit_code = main(["--path", self.db_path, "--json", "status"])
        self.assertEqual(exit_code, 0)
        output = self.get_stdout()
        parsed = json.loads(output)
        self.assertIn("session_count", parsed)


class TestVersionCommand(CLITestCase):
    """Tests for version command."""

    def test_version_shows_version(self):
        """Version should show version string."""
        exit_code = main(["version"])
        self.assertEqual(exit_code, 0)
        output = self.get_stdout()
        self.assertIn("origin", output.lower())

    def test_version_json_format(self):
        """Version with --json should output JSON."""
        exit_code = main(["--json", "version"])
        self.assertEqual(exit_code, 0)
        output = self.get_stdout()
        parsed = json.loads(output)
        self.assertIn("version", parsed)


class TestSessionsCommand(CLITestCase):
    """Tests for sessions command."""

    def test_sessions_empty_database(self):
        """Sessions on empty database should succeed."""
        main(["--path", self.db_path, "init"])
        exit_code = main(["--path", self.db_path, "sessions"])
        self.assertEqual(exit_code, 0)

    def test_sessions_with_data(self):
        """Sessions should list sessions."""
        main(["--path", self.db_path, "init"])
        # Create a session directly
        db = ProvenanceDatabase(self.db_path)
        db.begin_session("test_config")
        db.close()

        exit_code = main(["--path", self.db_path, "sessions"])
        self.assertEqual(exit_code, 0)


class TestQueryCommand(CLITestCase):
    """Tests for query command."""

    def test_query_not_found(self):
        """Query for nonexistent license should return exit code 2."""
        main(["--path", self.db_path, "init"])
        exit_code = main(["--path", self.db_path, "query", "MIT"])
        self.assertEqual(exit_code, 2)

    def test_query_found(self):
        """Query for existing license should return exit code 0."""
        main(["--path", self.db_path, "init"])

        # Add sample with license
        db = ProvenanceDatabase(self.db_path)
        session = db.begin_session("test_config")
        source = SourceRecord(
            source_id="test_source",
            source_type="file",
            source_path="/data/test.csv",
            license_id="MIT",
            first_seen="2025-01-01T00:00:00Z",
        )
        db.record_source(source)
        sample = create_test_provenance_record(
            content=b"test", source_id="test_source"
        )
        db.record_sample(sample)
        batch = BatchRecord(
            batch_id=fingerprint_bytes(b"batch"),
            session_id=session.session_id,
            batch_index=0,
            sample_count=1,
            created_at="2025-01-01T00:00:00Z",
            sample_ids=(sample.sample_id,),
        )
        db.record_batch(batch)
        db.close()

        exit_code = main(["--path", self.db_path, "query", "MIT"])
        self.assertEqual(exit_code, 0)


class TestInspectCommand(CLITestCase):
    """Tests for inspect command."""

    def test_inspect_nonexistent_session(self):
        """Inspect nonexistent session should fail."""
        main(["--path", self.db_path, "init"])
        exit_code = main([
            "--path", self.db_path, "inspect", "nonexistent"
        ])
        self.assertEqual(exit_code, 1)

    def test_inspect_existing_session(self):
        """Inspect existing session should succeed."""
        main(["--path", self.db_path, "init"])

        db = ProvenanceDatabase(self.db_path)
        session = db.begin_session("test_config")
        db.close()

        exit_code = main([
            "--path", self.db_path, "inspect", session.session_id
        ])
        self.assertEqual(exit_code, 0)


class TestConflictsCommand(CLITestCase):
    """Tests for conflicts command."""

    def test_conflicts_no_conflicts(self):
        """Conflicts with no conflicts should succeed."""
        main(["--path", self.db_path, "init"])
        exit_code = main(["--path", self.db_path, "conflicts"])
        self.assertEqual(exit_code, 0)
        output = self.get_stdout()
        self.assertIn("No", output)

    def test_conflicts_with_conflicts(self):
        """Conflicts should list conflicts."""
        main(["--path", self.db_path, "init"])

        db = ProvenanceDatabase(self.db_path)
        session = db.begin_session("test_config")
        db.record_conflict(
            session.session_id, "MIT", "GPL-3.0", "copyleft_mix"
        )
        db.close()

        exit_code = main(["--path", self.db_path, "conflicts"])
        self.assertEqual(exit_code, 0)


class TestExportCommand(CLITestCase):
    """Tests for export command."""

    def test_export_json(self):
        """Export JSON should succeed."""
        main(["--path", self.db_path, "init"])

        db = ProvenanceDatabase(self.db_path)
        session = db.begin_session("test_config")
        db.close()

        exit_code = main([
            "--path", self.db_path, "export", session.session_id,
            "--format", "json"
        ])
        self.assertEqual(exit_code, 0)

    def test_export_jsonl_requires_output(self):
        """Export JSONL without --output should fail."""
        main(["--path", self.db_path, "init"])

        db = ProvenanceDatabase(self.db_path)
        session = db.begin_session("test_config")
        db.close()

        exit_code = main([
            "--path", self.db_path, "export", session.session_id,
            "--format", "jsonl"
        ])
        self.assertEqual(exit_code, 1)

    def test_export_jsonl_with_output(self):
        """Export JSONL with --output should succeed."""
        main(["--path", self.db_path, "init"])

        db = ProvenanceDatabase(self.db_path)
        session = db.begin_session("test_config")
        db.close()

        output_path = str(Path(self.temp_dir) / "export.jsonl")
        exit_code = main([
            "--path", self.db_path, "export", session.session_id,
            "--format", "jsonl", "--output", output_path
        ])
        self.assertEqual(exit_code, 0)
        self.assertTrue(Path(output_path).exists())


class TestCardCommand(CLITestCase):
    """Tests for card command."""

    def test_card_generation(self):
        """Card should generate markdown."""
        main(["--path", self.db_path, "init"])

        db = ProvenanceDatabase(self.db_path)
        session = db.begin_session("test_config")
        db.close()

        exit_code = main([
            "--path", self.db_path, "card", session.session_id
        ])
        self.assertEqual(exit_code, 0)
        output = self.get_stdout()
        self.assertIn("#", output)  # Markdown header

    def test_card_to_file(self):
        """Card with --output should write to file."""
        main(["--path", self.db_path, "init"])

        db = ProvenanceDatabase(self.db_path)
        session = db.begin_session("test_config")
        db.close()

        output_path = str(Path(self.temp_dir) / "card.md")
        exit_code = main([
            "--path", self.db_path, "card", session.session_id,
            "--output", output_path
        ])
        self.assertEqual(exit_code, 0)
        self.assertTrue(Path(output_path).exists())


class TestTraceCommand(CLITestCase):
    """Tests for trace command."""

    def test_trace_nonexistent_sample(self):
        """Trace nonexistent sample should fail."""
        main(["--path", self.db_path, "init"])
        exit_code = main([
            "--path", self.db_path, "trace", "nonexistent_sample_id"
        ])
        self.assertEqual(exit_code, 1)

    def test_trace_existing_sample(self):
        """Trace existing sample should succeed."""
        main(["--path", self.db_path, "init"])

        db = ProvenanceDatabase(self.db_path)
        session = db.begin_session("test_config")
        source = SourceRecord(
            source_id="test_source",
            source_type="file",
            source_path="/data/test.csv",
            license_id="MIT",
            first_seen="2025-01-01T00:00:00Z",
        )
        db.record_source(source)
        sample = create_test_provenance_record(
            content=b"trace_test", source_id="test_source"
        )
        db.record_sample(sample)
        batch = BatchRecord(
            batch_id=fingerprint_bytes(b"trace_batch"),
            session_id=session.session_id,
            batch_index=0,
            sample_count=1,
            created_at="2025-01-01T00:00:00Z",
            sample_ids=(sample.sample_id,),
        )
        db.record_batch(batch)
        db.close()

        exit_code = main([
            "--path", self.db_path, "trace", sample.sample_id
        ])
        self.assertEqual(exit_code, 0)


class TestQuietMode(CLITestCase):
    """Tests for --quiet mode."""

    def test_quiet_suppresses_output(self):
        """Quiet mode should suppress non-essential output."""
        main(["--path", self.db_path, "init"])
        # Reset stdout to capture only status output
        sys.stdout = StringIO()
        main(["--path", self.db_path, "--quiet", "status"])
        output = self.get_stdout()
        # Should have minimal or no output
        self.assertEqual(len(output.strip()), 0)


class TestExitCodes(CLITestCase):
    """Tests for exit code behavior."""

    def test_success_exit_code(self):
        """Successful commands should return 0."""
        exit_code = main(["version"])
        self.assertEqual(exit_code, 0)

    def test_error_exit_code(self):
        """Error conditions should return 1."""
        exit_code = main(["--path", self.db_path, "status"])
        self.assertEqual(exit_code, 1)

    def test_not_found_exit_code(self):
        """Query not found should return 2."""
        main(["--path", self.db_path, "init"])
        exit_code = main(["--path", self.db_path, "query", "NonExistent"])
        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
