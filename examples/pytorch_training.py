#!/usr/bin/env python3
"""
PyTorch Training Example - Origin Provenance Tracking

This example demonstrates how to integrate Origin with a PyTorch training
pipeline. It shows how the DataLoaderHook automatically records provenance
for every batch without modifying your training code.

Requirements:
    pip install torch>=2.0
    pip install origin-provenance[pytorch]

Run this example:
    python examples/pytorch_training.py
"""

import os
import tempfile

# Check for PyTorch availability
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch>=2.0")
    print("This example requires PyTorch to run.")
    exit(1)

# Origin imports
from origin.storage.database import ProvenanceDatabase
from origin.hooks.pytorch import DataLoaderHook
from origin.query.engine import QueryEngine
from origin.cards.generator import ProvenanceCardGenerator


def create_synthetic_dataset(num_samples: int = 1000, num_features: int = 784):
    """Create a synthetic dataset for demonstration."""
    # Simulate MNIST-like data
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, 10, (num_samples,))
    return TensorDataset(X, y)


def create_simple_model(input_size: int = 784, num_classes: int = 10):
    """Create a simple feedforward neural network."""
    return nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    """Demonstrate PyTorch integration with Origin."""

    # Create a temporary directory for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "pytorch_provenance.db")

        print("=" * 60)
        print("Origin Provenance Tracking - PyTorch Training Example")
        print("=" * 60)

        # ─────────────────────────────────────────────────────────────
        # Step 1: Setup training components
        # ─────────────────────────────────────────────────────────────
        print("\n[1] Setting up training components...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    Device: {device}")

        # Create dataset and dataloader
        dataset = create_synthetic_dataset(num_samples=500, num_features=784)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        print(f"    Dataset size: {len(dataset)} samples")
        print(f"    Batch size: 32")
        print(f"    Batches per epoch: {len(dataloader)}")

        # Create model, loss, optimizer
        model = create_simple_model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print(f"    Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # ─────────────────────────────────────────────────────────────
        # Step 2: Initialize Origin provenance tracking
        # ─────────────────────────────────────────────────────────────
        print("\n[2] Initializing Origin provenance tracking...")

        db = ProvenanceDatabase(db_path)
        session = db.begin_session(config_hash="pytorch_mnist_v1")
        print(f"    Database: {db_path}")
        print(f"    Session ID: {session.session_id}")

        # Create the DataLoader hook
        # This observes data without modifying it
        hook = DataLoaderHook(
            db=db,
            session_id=session.session_id,
            source_id="synthetic_mnist",
            license_id="CC-BY-4.0"  # Creative Commons Attribution
        )
        print("    DataLoaderHook configured")

        # ─────────────────────────────────────────────────────────────
        # Step 3: Training loop with provenance tracking
        # ─────────────────────────────────────────────────────────────
        print("\n[3] Training with provenance tracking...")

        num_epochs = 3
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            # Wrap the dataloader with the hook
            # Data flows through unchanged - Origin just observes
            for batch_idx, batch in enumerate(hook.wrap(dataloader)):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            stats = hook.get_stats()
            print(f"    Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}, "
                  f"samples_observed={stats['samples_observed']}")

        # ─────────────────────────────────────────────────────────────
        # Step 4: End session and review statistics
        # ─────────────────────────────────────────────────────────────
        print("\n[4] Training complete, finalizing session...")

        db.end_session(session.session_id, status="completed")

        final_stats = hook.get_stats()
        print(f"\n    Provenance Statistics:")
        print(f"      Total batches observed: {final_stats['batches_observed']}")
        print(f"      Total samples observed: {final_stats['samples_observed']}")
        print(f"      Unique samples: {final_stats['unique_samples']}")

        # ─────────────────────────────────────────────────────────────
        # Step 5: Query provenance data
        # ─────────────────────────────────────────────────────────────
        print("\n[5] Querying provenance data...")

        engine = QueryEngine(db)

        # Session summary
        summary = engine.get_session_summary(session.session_id)
        print(f"\n    Session Summary:")
        print(f"      Status: {summary.get('status', 'unknown')}")
        print(f"      Total batches: {summary.get('batch_count', 0)}")
        print(f"      Unique samples: {summary.get('unique_samples', 0)}")

        # License information
        license_breakdown = engine.get_license_breakdown(session.session_id)
        print(f"\n    License Breakdown:")
        for license_id, count in license_breakdown.items():
            print(f"      {license_id}: {count} samples")

        # ─────────────────────────────────────────────────────────────
        # Step 6: Generate provenance card
        # ─────────────────────────────────────────────────────────────
        print("\n[6] Generating provenance card...")

        generator = ProvenanceCardGenerator(db)
        card = generator.generate(session.session_id)

        # Save the card
        card_path = os.path.join(temp_dir, "PROVENANCE_CARD.md")
        with open(card_path, "w") as f:
            f.write(card)

        print(f"    Card saved to: {card_path}")

        # Preview
        card_lines = card.split("\n")
        print("\n    Card Preview:")
        print("    " + "-" * 50)
        for line in card_lines[:20]:
            print(f"    {line}")
        if len(card_lines) > 20:
            print(f"    ... ({len(card_lines) - 20} more lines)")

        # ─────────────────────────────────────────────────────────────
        # Cleanup
        # ─────────────────────────────────────────────────────────────
        db.close()

        print("\n" + "=" * 60)
        print("PyTorch example completed successfully!")
        print("=" * 60)
        print("\nIntegration summary:")
        print("  1. Create ProvenanceDatabase and begin session")
        print("  2. Create DataLoaderHook with source and license info")
        print("  3. Wrap your DataLoader with hook.wrap()")
        print("  4. Train normally - data passes through unchanged")
        print("  5. End session and query provenance data")
        print("\nYour training code changes: ~5 lines added")


if __name__ == "__main__":
    main()
