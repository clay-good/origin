#!/usr/bin/env python3
"""
HuggingFace NLP Example - Origin Provenance Tracking

This example demonstrates how to integrate Origin with HuggingFace datasets
for NLP training pipelines. It shows how the DatasetHook automatically
records provenance for text data without modifying your workflow.

Requirements:
    pip install datasets>=2.0
    pip install origin-provenance[huggingface]

Run this example:
    python examples/huggingface_nlp.py
"""

import os
import tempfile

# Check for HuggingFace datasets availability
try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("HuggingFace datasets not installed. Install with: pip install datasets>=2.0")
    print("This example requires HuggingFace datasets to run.")
    exit(1)

# Origin imports
from origin.storage.database import ProvenanceDatabase
from origin.hooks.huggingface import DatasetHook
from origin.query.engine import QueryEngine
from origin.cards.generator import ProvenanceCardGenerator


def create_synthetic_nlp_dataset():
    """Create a synthetic NLP dataset for demonstration."""
    # Simulate a sentiment analysis dataset
    data = {
        "text": [
            "This product is amazing! I love it.",
            "Terrible experience, would not recommend.",
            "It's okay, nothing special.",
            "Best purchase I've ever made!",
            "Complete waste of money.",
            "Surprisingly good quality for the price.",
            "Not what I expected, but decent.",
            "Absolutely fantastic, exceeded expectations!",
            "Poor quality, broke after one use.",
            "Average product, does what it says.",
            "Highly recommend to everyone!",
            "Disappointed with the purchase.",
            "Great value for money.",
            "Would not buy again.",
            "Perfect for my needs!",
            "Mediocre at best.",
        ],
        "label": [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        "source": [
            "reviews_site_a", "reviews_site_a", "reviews_site_a", "reviews_site_a",
            "reviews_site_b", "reviews_site_b", "reviews_site_b", "reviews_site_b",
            "social_media", "social_media", "social_media", "social_media",
            "survey_data", "survey_data", "survey_data", "survey_data",
        ]
    }
    return Dataset.from_dict(data)


def simulate_tokenization(batch):
    """Simulate tokenization (in real code, use a tokenizer)."""
    # This simulates what a real tokenizer would do
    return {
        "input_ids": [[ord(c) % 100 for c in text[:50]] for text in batch["text"]],
        "attention_mask": [[1] * min(len(text), 50) for text in batch["text"]],
        "labels": batch["label"]
    }


def main():
    """Demonstrate HuggingFace integration with Origin."""

    # Create a temporary directory for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "huggingface_provenance.db")

        print("=" * 60)
        print("Origin Provenance Tracking - HuggingFace NLP Example")
        print("=" * 60)

        # ─────────────────────────────────────────────────────────────
        # Step 1: Create dataset
        # ─────────────────────────────────────────────────────────────
        print("\n[1] Creating synthetic NLP dataset...")

        dataset = create_synthetic_nlp_dataset()
        print(f"    Dataset size: {len(dataset)} samples")
        print(f"    Columns: {dataset.column_names}")
        print(f"    Sample text: '{dataset[0]['text'][:40]}...'")

        # ─────────────────────────────────────────────────────────────
        # Step 2: Initialize Origin provenance tracking
        # ─────────────────────────────────────────────────────────────
        print("\n[2] Initializing Origin provenance tracking...")

        db = ProvenanceDatabase(db_path)
        session = db.begin_session(config_hash="sentiment_classifier_v1")
        print(f"    Database: {db_path}")
        print(f"    Session ID: {session.session_id}")

        # Create the Dataset hook
        hook = DatasetHook(
            db=db,
            session_id=session.session_id,
            source_id="sentiment_reviews",
            license_id="CC-BY-4.0"
        )
        print("    DatasetHook configured")

        # ─────────────────────────────────────────────────────────────
        # Step 3: Process dataset with provenance tracking
        # ─────────────────────────────────────────────────────────────
        print("\n[3] Processing dataset with provenance tracking...")

        # Option A: Batch iteration (common for training)
        print("\n    Option A: Batch iteration")
        batch_size = 4
        processed_batches = 0

        for batch in hook.wrap(dataset.iter(batch_size=batch_size)):
            # Simulate processing (tokenization, etc.)
            tokenized = simulate_tokenization(batch)
            processed_batches += 1

            # In real training, you would:
            # outputs = model(**tokenized)
            # loss = outputs.loss
            # loss.backward()

        stats = hook.get_stats()
        print(f"    Processed {processed_batches} batches")
        print(f"    Samples observed: {stats['samples_observed']}")

        # Reset stats for next demo
        hook.reset_stats()

        # Option B: Individual sample iteration
        print("\n    Option B: Individual sample iteration")
        sample_count = 0

        for sample in hook.wrap_dataset(dataset):
            # Process individual sample
            text = sample["text"]
            label = sample["label"]
            sample_count += 1

            # In real code, process each sample individually

        stats = hook.get_stats()
        print(f"    Processed {sample_count} individual samples")
        print(f"    Unique samples: {stats['unique_samples']}")

        # ─────────────────────────────────────────────────────────────
        # Step 4: End session
        # ─────────────────────────────────────────────────────────────
        print("\n[4] Finalizing session...")

        db.end_session(session.session_id, status="completed")

        final_stats = hook.get_stats()
        print(f"\n    Final Provenance Statistics:")
        print(f"      Batches observed: {final_stats['batches_observed']}")
        print(f"      Samples observed: {final_stats['samples_observed']}")
        print(f"      Unique samples: {final_stats['unique_samples']}")

        # ─────────────────────────────────────────────────────────────
        # Step 5: Query provenance
        # ─────────────────────────────────────────────────────────────
        print("\n[5] Querying provenance data...")

        engine = QueryEngine(db)

        summary = engine.get_session_summary(session.session_id)
        print(f"\n    Session Summary:")
        print(f"      Status: {summary.get('status', 'unknown')}")
        print(f"      Batches: {summary.get('batch_count', 0)}")
        print(f"      Samples: {summary.get('sample_count', 0)}")

        license_breakdown = engine.get_license_breakdown(session.session_id)
        print(f"\n    Licenses Used:")
        for license_id, count in license_breakdown.items():
            print(f"      {license_id}: {count} samples")

        # ─────────────────────────────────────────────────────────────
        # Step 6: Generate provenance card
        # ─────────────────────────────────────────────────────────────
        print("\n[6] Generating provenance card...")

        generator = ProvenanceCardGenerator(db)
        card = generator.generate(session.session_id)

        card_lines = card.split("\n")
        print("\n    Provenance Card Preview:")
        print("    " + "-" * 50)
        for line in card_lines[:15]:
            print(f"    {line}")
        print(f"    ... ({len(card_lines) - 15} more lines)")

        # ─────────────────────────────────────────────────────────────
        # Cleanup
        # ─────────────────────────────────────────────────────────────
        db.close()

        print("\n" + "=" * 60)
        print("HuggingFace example completed successfully!")
        print("=" * 60)
        print("\nIntegration patterns:")
        print("  - hook.wrap(dataset.iter(batch_size=N)) for batch iteration")
        print("  - hook.wrap_dataset(dataset) for sample iteration")
        print("  - hook.wrap_streaming(dataset) for streaming datasets")
        print("\nYour code changes: ~5 lines to add provenance tracking")


if __name__ == "__main__":
    main()
