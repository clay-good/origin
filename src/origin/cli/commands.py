"""
CLI command implementations for Origin.

This module provides the implementation for all CLI commands. Each command
function takes the parsed args namespace and returns an exit code.
"""

import json
import sys
from typing import Any, Dict

from origin import __version__
from origin.storage.database import ProvenanceDatabase
from origin.query.engine import QueryEngine
from origin.export.formats import ProvenanceExporter
from origin.cards.generator import ProvenanceCardGenerator


def _get_db(args) -> ProvenanceDatabase:
    """Get database instance from args."""
    return ProvenanceDatabase(args.path)


def _output(data: Any, args, message: str = "") -> None:
    """Output data in appropriate format based on args."""
    if args.json:
        if isinstance(data, str):
            print(data)
        else:
            print(json.dumps(data, indent=2, sort_keys=True))
    elif not args.quiet:
        if message:
            print(message)
        elif isinstance(data, str):
            print(data)
        else:
            print(json.dumps(data, indent=2, sort_keys=True))


def _output_table(headers: list, rows: list, args) -> None:
    """Output data as a formatted table."""
    if args.json:
        # Convert to list of dicts for JSON output
        data = []
        for row in rows:
            data.append(dict(zip(headers, row)))
        print(json.dumps(data, indent=2, sort_keys=True))
        return

    if args.quiet:
        return

    if not rows:
        print("No data to display.")
        return

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        row_line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        print(row_line)


def cmd_init(args) -> int:
    """Initialize a new provenance database."""
    from pathlib import Path

    db_path = Path(args.path)

    if db_path.exists():
        if not args.quiet:
            print(f"Database already exists at {args.path}")
        return 0

    try:
        db = ProvenanceDatabase(args.path)
        db.close()
        _output(
            {"path": args.path, "status": "initialized"},
            args,
            f"Initialized provenance database at {args.path}",
        )
        return 0
    except Exception as e:
        print(f"Error initializing database: {e}", file=sys.stderr)
        return 1


def cmd_status(args) -> int:
    """Show database status and statistics."""
    from pathlib import Path

    db_path = Path(args.path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.path}")

    db = _get_db(args)
    try:
        # Get statistics
        stats = db.get_statistics()

        if args.json:
            print(json.dumps(stats, indent=2, sort_keys=True))
        else:
            if not args.quiet:
                print(f"Database: {args.path}")
                print(f"Schema Version: {stats.get('schema_version', 'unknown')}")
                print()
                print("Statistics:")
                print(f"  Sessions: {stats.get('session_count', 0)}")
                print(f"  Batches: {stats.get('batch_count', 0)}")
                print(f"  Samples: {stats.get('sample_count', 0)}")
                print(f"  Sources: {stats.get('source_count', 0)}")
                print(f"  Licenses: {stats.get('license_count', 0)}")
                print(f"  Conflicts: {stats.get('conflict_count', 0)}")

        return 0
    finally:
        db.close()


def cmd_sessions(args) -> int:
    """List all sessions."""
    from pathlib import Path

    db_path = Path(args.path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.path}")

    db = _get_db(args)
    try:
        sessions = db.list_sessions(limit=args.limit)

        if not sessions:
            _output([], args, "No sessions found.")
            return 0

        headers = ["Session ID", "Status", "Created", "Batches"]
        rows = []
        for session in sessions:
            batch_count = len(db.list_batches(session.session_id))
            rows.append([
                session.session_id[:16] + "...",
                session.status,
                session.created_at[:19] if session.created_at else "N/A",
                str(batch_count),
            ])

        _output_table(headers, rows, args)
        return 0
    finally:
        db.close()


def cmd_inspect(args) -> int:
    """Show details for a specific session."""
    from pathlib import Path

    db_path = Path(args.path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.path}")

    db = _get_db(args)
    try:
        session = db.get_session(args.session_id)
        if session is None:
            raise ValueError(f"Session not found: {args.session_id}")

        batches = db.list_batches(args.session_id)
        engine = QueryEngine(db)
        license_breakdown = engine.get_license_breakdown(args.session_id)
        conflicts = db.list_conflicts(args.session_id)

        # Count total samples
        total_samples = 0
        for batch in batches:
            total_samples += batch.sample_count

        result = {
            "session_id": session.session_id,
            "status": session.status,
            "created_at": session.created_at,
            "config_hash": session.config_hash,
            "metadata": session.metadata,
            "batch_count": len(batches),
            "total_samples": total_samples,
            "license_breakdown": license_breakdown,
            "conflict_count": len(conflicts),
        }

        if args.batches:
            result["batches"] = [batch.to_dict() for batch in batches]

        if args.samples:
            # Add sample statistics per batch
            sample_stats = []
            for batch in batches:
                samples = db.list_samples(batch.batch_id)
                stats = {
                    "batch_id": batch.batch_id,
                    "batch_index": batch.batch_index,
                    "sample_count": len(samples),
                }
                sample_stats.append(stats)
            result["sample_statistics"] = sample_stats

        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            if not args.quiet:
                print(f"Session: {session.session_id}")
                print(f"Status: {session.status}")
                print(f"Created: {session.created_at}")
                print(f"Config Hash: {session.config_hash[:16]}...")
                print()
                print(f"Batches: {len(batches)}")
                print(f"Total Samples: {total_samples}")
                print()
                if license_breakdown:
                    print("License Breakdown:")
                    for lic, count in license_breakdown.items():
                        print(f"  {lic}: {count}")
                    print()
                if conflicts:
                    print(f"License Conflicts: {len(conflicts)}")

                if args.batches:
                    print()
                    print("Batches:")
                    for batch in batches:
                        print(f"  [{batch.batch_index}] {batch.batch_id[:16]}... ({batch.sample_count} samples)")

        return 0
    finally:
        db.close()


def cmd_query(args) -> int:
    """Check if a license is present in a session."""
    from pathlib import Path

    db_path = Path(args.path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.path}")

    db = _get_db(args)
    try:
        engine = QueryEngine(db)

        if args.session:
            # Check specific session
            found = engine.check_license_presence(args.session, args.license_id)
            result = {
                "license_id": args.license_id,
                "session_id": args.session,
                "found": found,
            }

            if found:
                samples = engine.find_samples_by_license(args.license_id, args.session)
                result["sample_count"] = len(samples)

            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                if found:
                    _output(
                        result, args,
                        f"License '{args.license_id}' found in session {args.session} ({len(samples)} samples)"
                    )
                else:
                    _output(
                        result, args,
                        f"License '{args.license_id}' not found in session {args.session}"
                    )
                    return 2  # Not found exit code

        else:
            # Check all sessions
            samples = engine.find_samples_by_license(args.license_id)
            result = {
                "license_id": args.license_id,
                "found": len(samples) > 0,
                "sample_count": len(samples),
            }

            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                if samples:
                    _output(
                        result, args,
                        f"License '{args.license_id}' found in {len(samples)} samples"
                    )
                else:
                    _output(
                        result, args,
                        f"License '{args.license_id}' not found in any session"
                    )
                    return 2  # Not found exit code

        return 0
    finally:
        db.close()


def cmd_card(args) -> int:
    """Generate a provenance card for a session."""
    from pathlib import Path

    db_path = Path(args.path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.path}")

    db = _get_db(args)
    try:
        generator = ProvenanceCardGenerator(db)
        card = generator.generate(args.session_id)

        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(card)
            if not args.quiet:
                print(f"Provenance card written to {args.output}")
        else:
            print(card)

        return 0
    finally:
        db.close()


def cmd_export(args) -> int:
    """Export session provenance data."""
    from pathlib import Path

    db_path = Path(args.path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.path}")

    db = _get_db(args)
    try:
        exporter = ProvenanceExporter(db)

        if args.format == "json":
            output = exporter.to_json(args.session_id)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(output)
                if not args.quiet:
                    print(f"Exported to {args.output}")
            else:
                print(output)

        elif args.format == "jsonl":
            if not args.output:
                raise ValueError("JSONL format requires --output path")
            lines = exporter.to_jsonl(args.session_id, args.output)
            if not args.quiet:
                print(f"Exported {lines} lines to {args.output}")

        elif args.format == "mlflow":
            output = exporter.to_mlflow(args.session_id)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, sort_keys=True)
                if not args.quiet:
                    print(f"Exported to {args.output}")
            else:
                print(json.dumps(output, indent=2, sort_keys=True))

        elif args.format == "wandb":
            output = exporter.to_wandb(args.session_id)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, sort_keys=True)
                if not args.quiet:
                    print(f"Exported to {args.output}")
            else:
                print(json.dumps(output, indent=2, sort_keys=True))

        elif args.format == "hf":
            output = exporter.to_huggingface(args.session_id)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, sort_keys=True)
                if not args.quiet:
                    print(f"Exported to {args.output}")
            else:
                print(json.dumps(output, indent=2, sort_keys=True))

        return 0
    finally:
        db.close()


def cmd_conflicts(args) -> int:
    """List license conflicts."""
    from pathlib import Path

    db_path = Path(args.path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.path}")

    db = _get_db(args)
    try:
        engine = QueryEngine(db)
        conflicts = engine.find_conflicts(args.session)

        if not conflicts:
            _output([], args, "No license conflicts found.")
            return 0

        if args.json:
            print(json.dumps(conflicts, indent=2, sort_keys=True))
        else:
            if not args.quiet:
                print(f"Found {len(conflicts)} license conflict(s):")
                print()
                for conflict in conflicts:
                    print(f"  Session: {conflict['session_id'][:16]}...")
                    print(f"  Licenses: {conflict['license_a']} vs {conflict['license_b']}")
                    print(f"  Type: {conflict['conflict_type']}")
                    print(f"  Detected: {conflict['detected_at']}")
                    print()

        return 0
    finally:
        db.close()


def cmd_trace(args) -> int:
    """Show full provenance trace for a sample."""
    from pathlib import Path

    db_path = Path(args.path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.path}")

    db = _get_db(args)
    try:
        engine = QueryEngine(db)
        trace = engine.trace_sample(args.sample_id)

        if trace["sample"] is None:
            raise ValueError(f"Sample not found: {args.sample_id}")

        if args.json:
            print(json.dumps(trace, indent=2, sort_keys=True))
        else:
            if not args.quiet:
                sample = trace["sample"]
                print(f"Sample: {sample['sample_id']}")
                print(f"Content Type: {sample['content_type']}")
                print(f"Size: {sample['byte_size']} bytes")
                print(f"First Seen: {sample['timestamp']}")
                print()

                if trace["source"]:
                    source = trace["source"]
                    print(f"Source: {source['source_path']}")
                    print(f"Source ID: {source['source_id'][:16]}...")
                    if source.get("license_id"):
                        print(f"License: {source['license_id']}")
                    print()

                if trace["license"]:
                    lic = trace["license"]
                    print(f"License Details:")
                    print(f"  ID: {lic['license_id']}")
                    print(f"  Name: {lic.get('name', 'N/A')}")
                    print(f"  Permissions: {', '.join(lic.get('permissions', []))}")
                    print(f"  Conditions: {', '.join(lic.get('conditions', []))}")
                    print(f"  Limitations: {', '.join(lic.get('limitations', []))}")
                    print()

                if trace["batches"]:
                    print(f"Appears in {len(trace['batches'])} batch(es):")
                    for batch in trace["batches"][:10]:
                        print(f"  - Batch {batch['batch_index']} in session {batch['session_id'][:16]}...")
                    if len(trace["batches"]) > 10:
                        print(f"  ... and {len(trace['batches']) - 10} more")
                    print()

                if trace["sessions"]:
                    print(f"Sessions: {', '.join(s[:16] + '...' for s in trace['sessions'])}")

        return 0
    finally:
        db.close()


def cmd_version(args) -> int:
    """Show version information."""
    result = {
        "version": __version__,
        "name": "origin",
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        if not args.quiet:
            print(f"origin {__version__}")

    return 0
