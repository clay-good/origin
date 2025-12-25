"""
Command-line interface for Origin.

This module provides the main entry point and argument parser for the
Origin CLI. All commands are implemented in the commands module.
"""

import argparse
import sys
from typing import Optional

from origin import __version__


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with all subcommands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="origin",
        description="Runtime data provenance for AI pipelines",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"origin {__version__}",
    )

    parser.add_argument(
        "--path",
        type=str,
        default="./origin.db",
        help="Path to the provenance database (default: ./origin.db)",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
    )

    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new provenance database",
    )

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show database status and statistics",
    )

    # sessions command
    sessions_parser = subparsers.add_parser(
        "sessions",
        help="List all sessions",
    )
    sessions_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=20,
        help="Maximum number of sessions to show (default: 20)",
    )

    # inspect command
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Show details for a specific session",
    )
    inspect_parser.add_argument(
        "session_id",
        help="Session ID to inspect",
    )
    inspect_parser.add_argument(
        "--batches", "-b",
        action="store_true",
        help="Show batch details",
    )
    inspect_parser.add_argument(
        "--samples", "-s",
        action="store_true",
        help="Show sample statistics",
    )

    # query command
    query_parser = subparsers.add_parser(
        "query",
        help="Check if a license is present in a session",
    )
    query_parser.add_argument(
        "license_id",
        help="License ID to search for",
    )
    query_parser.add_argument(
        "--session",
        type=str,
        help="Session ID to search within (searches all if not specified)",
    )

    # card command
    card_parser = subparsers.add_parser(
        "card",
        help="Generate a provenance card for a session",
    )
    card_parser.add_argument(
        "session_id",
        help="Session ID to generate card for",
    )
    card_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (default: stdout)",
    )

    # export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export session provenance data",
    )
    export_parser.add_argument(
        "session_id",
        help="Session ID to export",
    )
    export_parser.add_argument(
        "--format", "-f",
        choices=["json", "jsonl", "mlflow", "wandb", "hf"],
        default="json",
        help="Export format (default: json)",
    )
    export_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (required for jsonl, default: stdout for others)",
    )

    # conflicts command
    conflicts_parser = subparsers.add_parser(
        "conflicts",
        help="List license conflicts",
    )
    conflicts_parser.add_argument(
        "--session",
        type=str,
        help="Session ID to filter by",
    )

    # trace command
    trace_parser = subparsers.add_parser(
        "trace",
        help="Show full provenance trace for a sample",
    )
    trace_parser.add_argument(
        "sample_id",
        help="Sample ID (fingerprint) to trace",
    )

    # version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
    )

    return parser


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success, 1 for error, 2 for query not found).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Import commands module here to avoid circular imports
    from origin.cli import commands

    # Map commands to functions
    command_map = {
        "init": commands.cmd_init,
        "status": commands.cmd_status,
        "sessions": commands.cmd_sessions,
        "inspect": commands.cmd_inspect,
        "query": commands.cmd_query,
        "card": commands.cmd_card,
        "export": commands.cmd_export,
        "conflicts": commands.cmd_conflicts,
        "trace": commands.cmd_trace,
        "version": commands.cmd_version,
    }

    cmd_func = command_map.get(args.command)
    if cmd_func is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    try:
        return cmd_func(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
