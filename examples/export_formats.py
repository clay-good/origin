#!/usr/bin/env python3
"""
Export Formats Example - Origin Provenance Tracking

This example demonstrates the various export formats available in Origin
for integrating with ML experiment tracking tools and compliance systems:

1. JSON - Machine-readable audit format
2. JSONL - Streaming format for large exports
3. MLflow - Tags, params, and artifacts for MLflow
4. Weights & Biases - Config and metadata for W&B
5. HuggingFace - Model card format for HF Hub

No external dependencies required - uses only Python standard library.

Run this example:
    python examples/export_formats.py
"""

import json
import os
import tempfile
from datetime import datetime

# Origin imports
from origin.storage.database import ProvenanceDatabase
from origin.core.fingerprint import fingerprint_sample, merkle_root
from origin.core.record import ProvenanceRecord, BatchRecord, SourceRecord
from origin.export.formats import ProvenanceExporter
from origin.cards.generator import ProvenanceCardGenerator


def setup_demo_data(db: ProvenanceDatabase) -> str:
    """Set up demo data for export examples."""
    # Register diverse sources with different licenses
    sources = [
        ("imagenet_subset", "download", "https://image-net.org/subset", "ImageNet License"),
        ("coco_annotations", "download", "https://cocodataset.org", "CC-BY-4.0"),
        ("custom_labels", "internal", "/data/labels/custom", "proprietary"),
        ("augmented_data", "generated", "augmentation_pipeline_v2", "MIT"),
    ]

    for source_id, source_type, source_path, license_id in sources:
        db.record_source(SourceRecord(
            source_id=source_id,
            source_type=source_type,
            source_path=source_path,
            license_id=license_id,
            first_seen=datetime.now().isoformat()
        ))

    # Create training session
    session = db.begin_session(config_hash="object_detector_yolov8_v2")

    # Add samples from each source
    sample_configs = [
        ("imagenet_subset", 20),
        ("coco_annotations", 15),
        ("custom_labels", 10),
        ("augmented_data", 25),
    ]

    batch_idx = 0
    for source_id, count in sample_configs:
        fingerprints = []
        for i in range(count):
            sample_data = f"{source_id}_sample_{i}".encode()
            fp = fingerprint_sample(sample_data)
            fingerprints.append(fp)

            db.record_sample(ProvenanceRecord(
                sample_id=fp,
                source_id=source_id,
                content_type="image",
                byte_size=len(sample_data),
                timestamp=datetime.now().isoformat(),
                metadata={"index": i}
            ))

        batch_fp = merkle_root(fingerprints)
        db.record_batch(BatchRecord(
            batch_id=batch_fp,
            session_id=session.session_id,
            batch_index=batch_idx,
            sample_count=len(fingerprints),
            created_at=datetime.now().isoformat(),
            sample_ids=tuple(fingerprints)
        ))
        batch_idx += 1

    db.end_session(session.session_id, status="completed")
    return session.session_id


def main():
    """Demonstrate various export formats."""

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "provenance.db")

        print("=" * 70)
        print("Origin Provenance Tracking - Export Formats Example")
        print("=" * 70)

        # Setup
        print("\n[Setup] Creating demo database...")
        db = ProvenanceDatabase(db_path)
        session_id = setup_demo_data(db)
        print(f"    Session ID: {session_id}")

        # Initialize exporter
        exporter = ProvenanceExporter(db)

        # ─────────────────────────────────────────────────────────────
        # Format 1: JSON Export
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("FORMAT 1: JSON Export")
        print("=" * 70)
        print("Best for: Audit records, compliance archives, API responses")

        json_path = os.path.join(temp_dir, "provenance_export.json")
        json_data = exporter.to_json(session_id, pretty=True)

        with open(json_path, "w") as f:
            f.write(json_data)

        print(f"\nExported to: {json_path}")
        print(f"File size: {len(json_data):,} bytes")

        # Show structure
        parsed = json.loads(json_data)
        print("\nJSON Structure:")
        print(f"  - session: Session metadata")
        print(f"  - sources: {len(parsed.get('sources', []))} data sources")
        print(f"  - licenses: {len(parsed.get('licenses', []))} license records")
        print(f"  - samples: {len(parsed.get('samples', []))} sample records")
        print(f"  - batches: {len(parsed.get('batches', []))} batch records")
        print(f"  - statistics: Aggregated metrics")

        # Preview
        print("\nPreview (first 500 chars):")
        print("-" * 50)
        print(json_data[:500] + "...")

        # ─────────────────────────────────────────────────────────────
        # Format 2: JSONL Export (Streaming)
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("FORMAT 2: JSONL Export (JSON Lines)")
        print("=" * 70)
        print("Best for: Large datasets, streaming processing, log aggregation")

        jsonl_path = os.path.join(temp_dir, "provenance_export.jsonl")
        line_count = exporter.to_jsonl(session_id, jsonl_path)

        print(f"\nExported to: {jsonl_path}")
        print(f"Total lines: {line_count}")

        # Show first few lines
        print("\nFirst 5 lines:")
        print("-" * 50)
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                # Parse and show record type
                record = json.loads(line)
                record_type = record.get("record_type", "unknown")
                print(f"  Line {i+1}: {record_type} record")

        # ─────────────────────────────────────────────────────────────
        # Format 3: MLflow Integration
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("FORMAT 3: MLflow Integration")
        print("=" * 70)
        print("Best for: Experiment tracking with MLflow")

        mlflow_data = exporter.to_mlflow(session_id)

        print("\nMLflow Export Structure:")
        print(f"  Tags: {len(mlflow_data.get('tags', {}))} tags")
        print(f"  Params: {len(mlflow_data.get('params', {}))} parameters")
        print(f"  Artifacts: {len(mlflow_data.get('artifacts', {}))} artifact files")

        print("\nTags:")
        for key, value in list(mlflow_data.get("tags", {}).items())[:5]:
            print(f"    {key}: {value[:50]}..." if len(str(value)) > 50 else f"    {key}: {value}")

        print("\nParams:")
        for key, value in list(mlflow_data.get("params", {}).items())[:5]:
            print(f"    {key}: {value}")

        print("\nArtifacts:")
        for name in mlflow_data.get("artifacts", {}).keys():
            print(f"    {name}")

        print("\nUsage with MLflow:")
        print("""
    import mlflow
    from origin.export.formats import ProvenanceExporter

    exporter = ProvenanceExporter(db)
    mlflow_data = exporter.to_mlflow(session_id)

    with mlflow.start_run():
        # Log tags
        for key, value in mlflow_data["tags"].items():
            mlflow.set_tag(key, value)

        # Log params
        for key, value in mlflow_data["params"].items():
            mlflow.log_param(key, value)

        # Log artifacts
        for name, content in mlflow_data["artifacts"].items():
            with open(name, "w") as f:
                f.write(content)
            mlflow.log_artifact(name)
        """)

        # ─────────────────────────────────────────────────────────────
        # Format 4: Weights & Biases Integration
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("FORMAT 4: Weights & Biases Integration")
        print("=" * 70)
        print("Best for: Experiment tracking with W&B")

        wandb_data = exporter.to_wandb(session_id)

        print("\nW&B Export Structure:")
        print(f"  Config keys: {len(wandb_data.get('config', {}))} config entries")
        print(f"  Metadata keys: {len(wandb_data.get('metadata', {}))} metadata entries")

        print("\nConfig (for wandb.config.update()):")
        for key, value in list(wandb_data.get("config", {}).items())[:5]:
            val_str = str(value)
            print(f"    {key}: {val_str[:50]}..." if len(val_str) > 50 else f"    {key}: {value}")

        print("\nUsage with W&B:")
        print("""
    import wandb
    from origin.export.formats import ProvenanceExporter

    exporter = ProvenanceExporter(db)
    wandb_data = exporter.to_wandb(session_id)

    wandb.init(project="my_project")
    wandb.config.update(wandb_data["config"])
    wandb.log({"provenance": wandb_data["metadata"]})
        """)

        # ─────────────────────────────────────────────────────────────
        # Format 5: HuggingFace Hub Integration
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("FORMAT 5: HuggingFace Hub Integration")
        print("=" * 70)
        print("Best for: Model cards on HuggingFace Hub")

        hf_data = exporter.to_huggingface(session_id)

        print("\nHuggingFace Export Structure:")
        print(f"  License: {hf_data.get('license', 'N/A')}")
        print(f"  Datasets: {hf_data.get('datasets', [])}")
        print(f"  Training data keys: {list(hf_data.get('training_data', {}).keys())}")

        print("\nModel Card Template:")
        model_card = f"""
---
license: {hf_data.get('license', 'unknown')}
datasets: {hf_data.get('datasets', [])}
---

# Model Card

## Training Data Provenance

- **Total samples**: {hf_data.get('training_data', {}).get('samples', 'N/A')}
- **Provenance fingerprint**: {hf_data.get('training_data', {}).get('provenance_fingerprint', 'N/A')[:32]}...
- **Data sources**: {hf_data.get('training_data', {}).get('sources', [])}

## License Information

{hf_data.get('license', 'See individual dataset licenses')}
"""
        print(model_card)

        # ─────────────────────────────────────────────────────────────
        # Format 6: Provenance Card (Markdown)
        # ─────────────────────────────────────────────────────────────
        print("=" * 70)
        print("FORMAT 6: Provenance Card (Markdown)")
        print("=" * 70)
        print("Best for: Human-readable documentation, model cards, audits")

        generator = ProvenanceCardGenerator(db)
        card = generator.generate(session_id)

        card_path = os.path.join(temp_dir, "PROVENANCE_CARD.md")
        with open(card_path, "w") as f:
            f.write(card)

        print(f"\nExported to: {card_path}")
        print(f"File size: {len(card):,} characters")

        print("\nCard Preview:")
        print("-" * 50)
        for line in card.split("\n")[:20]:
            print(line)
        print("...")

        db.close()

        print("\n" + "=" * 70)
        print("Export Formats Example Complete!")
        print("=" * 70)
        print("""
Export Format Summary:

| Format      | Use Case                              | Method                    |
|-------------|---------------------------------------|---------------------------|
| JSON        | Audit records, compliance archives    | exporter.to_json()        |
| JSONL       | Large datasets, streaming             | exporter.to_jsonl()       |
| MLflow      | MLflow experiment tracking            | exporter.to_mlflow()      |
| W&B         | Weights & Biases tracking             | exporter.to_wandb()       |
| HuggingFace | HuggingFace Hub model cards           | exporter.to_huggingface() |
| Markdown    | Human-readable provenance cards       | generator.generate()      |

All exports include:
  - Session metadata and configuration
  - Data source information
  - License details and compatibility
  - Sample and batch fingerprints
  - Aggregated statistics
        """)


if __name__ == "__main__":
    main()
