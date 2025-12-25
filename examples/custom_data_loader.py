#!/usr/bin/env python3
"""
Custom Data Loader Example - Origin Provenance Tracking

This example demonstrates how to create a custom hook for your own data
loading pipeline. This is useful when you're not using PyTorch DataLoader
or HuggingFace datasets, but have your own data loading mechanism.

No external dependencies required - uses only Python standard library.

Run this example:
    python examples/custom_data_loader.py
"""

import os
import json
import tempfile
from typing import Any, Dict, Iterator, List, Tuple
from datetime import datetime

# Origin imports
from origin.hooks.base import BaseHook
from origin.storage.database import ProvenanceDatabase
from origin.core.record import SourceRecord
from origin.query.engine import QueryEngine
from origin.cards.generator import ProvenanceCardGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Custom Hook Implementation
# ─────────────────────────────────────────────────────────────────────────────

class JSONLDataLoaderHook(BaseHook):
    """
    Custom hook for JSONL (JSON Lines) data files.

    This demonstrates how to create a hook for any custom data format.
    Extend BaseHook and implement _extract_samples() to handle your format.
    """

    def __init__(
        self,
        db: ProvenanceDatabase,
        session_id: str,
        source_id: str = "jsonl_data",
        license_id: str = None,
        id_field: str = None,
    ):
        """
        Initialize the JSONL hook.

        Args:
            db: ProvenanceDatabase instance.
            session_id: Session ID for this run.
            source_id: Identifier for this data source.
            license_id: SPDX license ID for the data.
            id_field: Optional field name to use as sample ID.
        """
        super().__init__(db, session_id, source_id, license_id)
        self.id_field = id_field

    def _extract_samples(self, batch: Any) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        Extract individual samples from a batch.

        This is the key method to implement for custom hooks.
        It must return a list of (sample_data, metadata) tuples.

        Args:
            batch: Your batch format (list of dicts for JSONL).

        Returns:
            List of (sample_data, metadata) tuples.
        """
        samples = []

        if isinstance(batch, list):
            # Batch is a list of JSON objects
            for i, item in enumerate(batch):
                # The sample data is what gets fingerprinted
                sample_data = item

                # Metadata provides context but doesn't affect fingerprint
                metadata = {
                    "index": i,
                    "fields": list(item.keys()) if isinstance(item, dict) else [],
                }

                # Use ID field if specified
                if self.id_field and isinstance(item, dict):
                    metadata["original_id"] = item.get(self.id_field)

                samples.append((sample_data, metadata))

        elif isinstance(batch, dict):
            # Single item
            samples.append((batch, {"format": "single"}))

        else:
            # Unknown format - wrap it
            samples.append(({"value": batch}, {"format": "unknown"}))

        return samples

    def load_file(self, filepath: str, batch_size: int = 10) -> Iterator[List[Dict]]:
        """
        Load a JSONL file and yield batches.

        This is a convenience method that reads a JSONL file and
        yields batches while automatically recording provenance.

        Args:
            filepath: Path to the JSONL file.
            batch_size: Number of records per batch.

        Yields:
            Batches of JSON objects.
        """
        batch = []

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    batch.append(record)

                    if len(batch) >= batch_size:
                        self.observe(batch)
                        yield batch
                        batch = []

        # Yield remaining records
        if batch:
            self.observe(batch)
            yield batch


class ImageFolderHook(BaseHook):
    """
    Custom hook for image folder datasets.

    This demonstrates handling file-based datasets where each file
    is a sample. The fingerprint is computed from the file contents.
    """

    def __init__(
        self,
        db: ProvenanceDatabase,
        session_id: str,
        source_id: str = "image_folder",
        license_id: str = None,
    ):
        super().__init__(db, session_id, source_id, license_id)

    def _extract_samples(self, batch: Any) -> List[Tuple[Any, Dict[str, Any]]]:
        """Extract samples from a batch of image paths or data."""
        samples = []

        if isinstance(batch, list):
            for i, item in enumerate(batch):
                if isinstance(item, tuple) and len(item) == 2:
                    # (image_data, label) format
                    image_data, label = item
                    samples.append((
                        image_data,
                        {"index": i, "label": label}
                    ))
                elif isinstance(item, dict):
                    # {"image": data, "label": label} format
                    samples.append((
                        item.get("image", item),
                        {"index": i, "label": item.get("label")}
                    ))
                else:
                    # Raw data
                    samples.append((item, {"index": i}))
        else:
            samples.append((batch, {"format": "single"}))

        return samples

    def load_folder(self, folder_path: str, batch_size: int = 8) -> Iterator[List[Tuple]]:
        """
        Load images from a folder structure.

        Assumes structure: folder/class_name/image.png

        Args:
            folder_path: Root folder path.
            batch_size: Images per batch.

        Yields:
            Batches of (image_bytes, class_name) tuples.
        """
        batch = []

        # Walk through folder structure
        for class_name in sorted(os.listdir(folder_path)):
            class_path = os.path.join(folder_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for filename in sorted(os.listdir(class_path)):
                filepath = os.path.join(class_path, filename)
                if not os.path.isfile(filepath):
                    continue

                # Read file contents
                with open(filepath, "rb") as f:
                    image_data = f.read()

                batch.append((image_data, class_name))

                if len(batch) >= batch_size:
                    self.observe(batch)
                    yield batch
                    batch = []

        if batch:
            self.observe(batch)
            yield batch


# ─────────────────────────────────────────────────────────────────────────────
# Example Usage
# ─────────────────────────────────────────────────────────────────────────────

def create_sample_jsonl_file(filepath: str):
    """Create a sample JSONL file for demonstration."""
    records = [
        {"id": "doc_001", "text": "Machine learning is transforming industries.", "category": "tech"},
        {"id": "doc_002", "text": "Climate change poses significant challenges.", "category": "science"},
        {"id": "doc_003", "text": "New regulations affect data privacy.", "category": "legal"},
        {"id": "doc_004", "text": "Quantum computing reaches new milestone.", "category": "tech"},
        {"id": "doc_005", "text": "Healthcare innovations save lives.", "category": "health"},
        {"id": "doc_006", "text": "Renewable energy adoption accelerates.", "category": "science"},
        {"id": "doc_007", "text": "AI ethics becomes corporate priority.", "category": "tech"},
        {"id": "doc_008", "text": "Space exploration enters new era.", "category": "science"},
        {"id": "doc_009", "text": "Cybersecurity threats evolve rapidly.", "category": "tech"},
        {"id": "doc_010", "text": "Biodiversity loss concerns scientists.", "category": "science"},
        {"id": "doc_011", "text": "Remote work reshapes office culture.", "category": "business"},
        {"id": "doc_012", "text": "Electric vehicles gain market share.", "category": "tech"},
    ]

    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def create_sample_image_folder(folder_path: str):
    """Create a sample image folder structure for demonstration."""
    classes = ["cat", "dog", "bird"]

    for class_name in classes:
        class_dir = os.path.join(folder_path, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Create fake "image" files (in real use, these would be actual images)
        for i in range(4):
            filepath = os.path.join(class_dir, f"{class_name}_{i:03d}.png")
            # Write some bytes to simulate image data
            with open(filepath, "wb") as f:
                f.write(f"FAKE_IMAGE:{class_name}:{i}".encode())


def main():
    """Demonstrate custom hook implementations."""

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "custom_provenance.db")
        jsonl_path = os.path.join(temp_dir, "data.jsonl")
        images_path = os.path.join(temp_dir, "images")

        print("=" * 60)
        print("Origin Provenance Tracking - Custom Data Loader Example")
        print("=" * 60)

        # Create sample data files
        print("\n[Setup] Creating sample data files...")
        create_sample_jsonl_file(jsonl_path)
        create_sample_image_folder(images_path)
        print(f"    JSONL file: {jsonl_path}")
        print(f"    Images folder: {images_path}")

        # Initialize database
        db = ProvenanceDatabase(db_path)

        # Register sources upfront
        db.record_source(SourceRecord(
            source_id="news_articles",
            source_type="jsonl",
            source_path=jsonl_path,
            license_id="CC-BY-4.0",
            first_seen=datetime.now().isoformat()
        ))
        db.record_source(SourceRecord(
            source_id="animal_images",
            source_type="folder",
            source_path=images_path,
            license_id="MIT",
            first_seen=datetime.now().isoformat()
        ))

        # ─────────────────────────────────────────────────────────────
        # Example 1: JSONL Data Loader
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Example 1: JSONL Data Loader Hook")
        print("=" * 60)

        session1 = db.begin_session(config_hash="jsonl_processing_v1")
        print(f"\n[1] Session started: {session1.session_id}")

        # Create JSONL hook
        jsonl_hook = JSONLDataLoaderHook(
            db=db,
            session_id=session1.session_id,
            source_id="news_articles",
            license_id="CC-BY-4.0",
            id_field="id"  # Use "id" field as sample identifier
        )

        print("\n[2] Processing JSONL file with provenance tracking...")
        total_records = 0
        for batch in jsonl_hook.load_file(jsonl_path, batch_size=4):
            # Process each batch
            for record in batch:
                # Your processing logic here
                text = record.get("text", "")
                category = record.get("category", "")
                total_records += 1

        stats = jsonl_hook.get_stats()
        print(f"    Records processed: {total_records}")
        print(f"    Batches observed: {stats['batches_observed']}")
        print(f"    Unique samples: {stats['unique_samples']}")

        db.end_session(session1.session_id, status="completed")
        print("\n[3] Session completed")

        # ─────────────────────────────────────────────────────────────
        # Example 2: Image Folder Loader
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Example 2: Image Folder Hook")
        print("=" * 60)

        session2 = db.begin_session(config_hash="image_classification_v1")
        print(f"\n[1] Session started: {session2.session_id}")

        # Create image folder hook
        image_hook = ImageFolderHook(
            db=db,
            session_id=session2.session_id,
            source_id="animal_images",
            license_id="MIT"
        )

        print("\n[2] Processing image folder with provenance tracking...")
        total_images = 0
        class_counts = {}
        for batch in image_hook.load_folder(images_path, batch_size=4):
            for image_data, class_name in batch:
                # Your image processing logic here
                total_images += 1
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        stats = image_hook.get_stats()
        print(f"    Images processed: {total_images}")
        print(f"    Classes: {class_counts}")
        print(f"    Batches observed: {stats['batches_observed']}")
        print(f"    Unique samples: {stats['unique_samples']}")

        db.end_session(session2.session_id, status="completed")
        print("\n[3] Session completed")

        # ─────────────────────────────────────────────────────────────
        # Query provenance for both sessions
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Querying Provenance Data")
        print("=" * 60)

        engine = QueryEngine(db)

        for session_id, name in [(session1.session_id, "JSONL"), (session2.session_id, "Images")]:
            batches = db.list_batches(session_id)
            licenses = engine.get_license_breakdown(session_id)

            print(f"\n{name} Session:")
            print(f"  Batches: {len(batches)}")
            print(f"  Samples: {sum(licenses.values())}")
            print(f"  Licenses: {licenses}")

        # ─────────────────────────────────────────────────────────────
        # Generate provenance card
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Generating Provenance Card")
        print("=" * 60)

        generator = ProvenanceCardGenerator(db)
        card = generator.generate(session1.session_id)

        card_lines = card.split("\n")
        print("\nJSONL Session Provenance Card Preview:")
        print("-" * 50)
        for line in card_lines[:12]:
            print(line)
        print(f"... ({len(card_lines) - 12} more lines)")

        db.close()

        print("\n" + "=" * 60)
        print("Custom data loader example completed!")
        print("=" * 60)
        print("\nTo create your own hook:")
        print("  1. Extend BaseHook")
        print("  2. Implement _extract_samples(batch) -> List[(data, metadata)]")
        print("  3. Use observe(batch) or wrap(iterable) to record data")
        print("  4. The fingerprint is computed automatically from sample data")


if __name__ == "__main__":
    main()
