"""
Training pipeline for document classification model.

This script fine-tunes a BERT model on document classification data,
evaluates performance, and saves the trained model for integration
with the existing classifier.py module.
"""

import argparse
import os
import sys
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Import existing ingestion module for PDF processing
import ingestion

# Label definitions matching classifier.py
LABELS = [
    "Invoice",
    "Resume",
    "Utility Bill",
    "Other",
    "Unclassifiable"
]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

# Filename prefix mapping (case-insensitive)
PREFIX_TO_LABEL = {
    "invoice": "Invoice",
    "resume": "Resume",
    "utilitybill": "Utility Bill",
    "other": "Other",
    "unclassifiable": "Unclassifiable"
}


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    documents_dir: str
    output_dir: str
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 8
    val_ratio: float = 0.2
    random_seed: int = 42
    max_length: int = 512


@dataclass
class EvaluationResults:
    """Results from model evaluation."""
    accuracy: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray
    confidence_stats: Dict[str, float]
    misclassifications: List[Dict[str, Any]]


# ============================================================================
# Data Preparation Functions
# ============================================================================

def derive_label_from_filename(filename: str) -> str:
    """
    Derive document label from filename pattern.
    
    Args:
        filename: Document filename (e.g., "invoice_1.pdf")
        
    Returns:
        Label string (e.g., "Invoice")
    """
    # Extract prefix before first underscore
    # Handle case where there's no underscore by using the whole filename
    if '_' in filename:
        prefix = filename.split('_')[0]
    else:
        # Remove .pdf extension if present
        prefix = filename.replace('.pdf', '')
    
    # Convert to lowercase for case-insensitive matching
    prefix_lower = prefix.lower()
    
    # Map prefix to label, default to "Other" if not recognized
    return PREFIX_TO_LABEL.get(prefix_lower, "Other")


def load_and_label_documents(documents_dir: str) -> List[Tuple[str, str]]:
    """
    Load documents and derive labels from filenames.
    
    Args:
        documents_dir: Path to directory containing PDF files
        
    Returns:
        List of (text, label) tuples
        
    Raises:
        ValueError: If insufficient documents found
    """
    # Use existing ingestion module to load PDFs
    docs_dict = ingestion.ingest_documents(documents_dir)
    
    # Create list of (text, label) tuples, filtering out empty documents
    documents = []
    failed_count = 0
    
    for filename, text in docs_dict.items():
        # Filter out documents with empty or null text
        if text is None or text == "":
            print(f"Warning: Skipping {filename} - empty or null text content")
            failed_count += 1
            continue
        
        # Derive label from filename
        label = derive_label_from_filename(filename)
        documents.append((text, label))
    
    # Check if more than 50% of documents failed
    total_docs = len(docs_dict)
    if total_docs > 0 and failed_count / total_docs > 0.5:
        raise ValueError(
            f"More than 50% of documents failed to load: "
            f"{failed_count}/{total_docs} documents had empty or null text"
        )
    
    if len(documents) == 0:
        raise ValueError(
            f"No valid documents found in {documents_dir}. "
            f"Ensure the directory contains PDF files with text content."
        )
    
    return documents


def validate_dataset(documents: List[Tuple[str, str]]) -> None:
    """
    Validate dataset meets minimum requirements.
    
    Checks:
    - At least 10 total documents
    - Each category has at least 2 documents
    - All texts are non-empty
    
    Args:
        documents: List of (text, label) tuples
        
    Raises:
        ValueError: If validation fails
    """
    # Check minimum total documents
    if len(documents) < 10:
        raise ValueError(
            f"Insufficient training data: found {len(documents)} documents, "
            f"need at least 10"
        )
    
    # Check all texts are non-empty strings
    for i, (text, label) in enumerate(documents):
        if not isinstance(text, str) or text == "":
            raise ValueError(
                f"Document at index {i} has empty or invalid text (label: {label})"
            )
    
    # Count documents per category
    category_counts = {}
    for text, label in documents:
        category_counts[label] = category_counts.get(label, 0) + 1
    
    # Check each category has at least 2 documents (warn if not)
    categories_with_few_docs = []
    for label, count in category_counts.items():
        if count < 2:
            categories_with_few_docs.append(f"{label} ({count})")
    
    if categories_with_few_docs:
        print(
            f"Warning: The following categories have fewer than 2 documents: "
            f"{', '.join(categories_with_few_docs)}. "
            f"This may affect training quality and stratified splitting."
        )
    
    print(f"Dataset validation passed: {len(documents)} documents across "
          f"{len(category_counts)} categories")


def create_train_val_split(
    documents: List[Tuple[str, str]],
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split documents into training and validation sets.
    
    Ensures each category has representation in both sets.
    Uses stratified splitting to maintain class distribution.
    
    Args:
        documents: List of (text, label) tuples
        val_ratio: Fraction of data for validation (default 0.2)
        random_seed: Random seed for reproducibility
        
    Returns:
        (train_data, val_data) tuple
    """
    # Extract texts and labels
    texts = [text for text, label in documents]
    labels = [label for text, label in documents]
    
    # Count documents per category
    category_counts = {}
    for label in labels:
        category_counts[label] = category_counts.get(label, 0) + 1
    
    # Check if any category has only 1 document
    # If so, we can't use stratified splitting for that category
    single_doc_categories = [cat for cat, count in category_counts.items() if count == 1]
    
    if single_doc_categories:
        print(
            f"Warning: Categories {single_doc_categories} have only 1 document each. "
            f"These may not appear in both training and validation sets."
        )
        # Use regular split without stratification
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=val_ratio,
            random_state=random_seed,
            shuffle=True
        )
    else:
        # Use stratified split to ensure each category is represented in both sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=labels,
            shuffle=True
        )
    
    # Combine back into tuples
    train_data = list(zip(train_texts, train_labels))
    val_data = list(zip(val_texts, val_labels))
    
    # Verify each category has at least 1 document in validation set (if possible)
    val_categories = set(val_labels)
    train_categories = set(train_labels)
    
    print(f"Train/validation split complete:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Categories in training: {sorted(train_categories)}")
    print(f"  Categories in validation: {sorted(val_categories)}")
    
    return train_data, val_data


# ============================================================================
# Model Training Functions
# ============================================================================

class DocumentDataset(Dataset):
    """PyTorch dataset for document classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        tokenizer: BertTokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of document texts
            labels: List of category labels
            tokenizer: BERT tokenizer instance
            max_length: Maximum sequence length
        """
        # TODO: Implement in task 4.1
        pass
    
    def __len__(self) -> int:
        """Return dataset size."""
        # TODO: Implement in task 4.1
        pass
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        # TODO: Implement in task 4.1
        pass


def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    batch_size: int = 8
) -> BertForSequenceClassification:
    """
    Fine-tune BERT model on training data.
    
    Uses Hugging Face Trainer API with:
    - AdamW optimizer
    - Linear learning rate schedule with warmup
    - Cross-entropy loss (built into BertForSequenceClassification)
    - Evaluation after each epoch
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save model checkpoints
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model
    """
    # TODO: Implement in task 4.3
    pass


# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute classification metrics for Trainer.
    
    Called automatically by Hugging Face Trainer during evaluation.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
        
    Returns:
        Dictionary with accuracy, precision, recall, f1 metrics
    """
    # TODO: Implement in task 5.1
    pass


def evaluate_model(
    model: BertForSequenceClassification,
    val_dataset: Dataset,
    tokenizer: BertTokenizer,
    label_names: List[str]
) -> EvaluationResults:
    """
    Comprehensive evaluation with detailed metrics.
    
    Computes:
    - Overall accuracy
    - Per-class precision, recall, F1
    - Confusion matrix
    - Confidence score distribution
    - Misclassification analysis
    
    Args:
        model: Trained model
        val_dataset: Validation dataset
        tokenizer: BERT tokenizer
        label_names: List of category names
        
    Returns:
        EvaluationResults object
    """
    # TODO: Implement in task 5.3
    pass


def print_evaluation_report(results: EvaluationResults) -> None:
    """
    Display formatted evaluation results.
    
    Prints:
    - Classification report (sklearn.metrics.classification_report)
    - Confusion matrix
    - Confidence statistics
    - Misclassified examples
    
    Args:
        results: EvaluationResults object
    """
    # TODO: Implement in task 5.5
    pass


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train document classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--documents-dir",
        type=str,
        default="documents",
        help="Directory containing PDF documents"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/trained-classifier",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("=" * 80)
    print("Document Classifier Training Pipeline")
    print("=" * 80)
    print()
    
    # Create configuration
    config = TrainingConfig(
        documents_dir=args.documents_dir,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        random_seed=args.seed
    )
    
    print(f"Configuration:")
    print(f"  Documents directory: {config.documents_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Validation ratio: {config.val_ratio}")
    print(f"  Random seed: {config.random_seed}")
    print()
    
    # TODO: Implement full pipeline in task 7.3
    # 1. Load and validate documents
    # 2. Create train/val split
    # 3. Display dataset statistics
    # 4. Train model with progress logging
    # 5. Evaluate model and display results
    # 6. Print instructions for using trained model
    
    print("Training pipeline implementation in progress...")
    print()
    print("To use the trained model with the classifier:")
    print(f"  export MODEL_PATH={config.output_dir}")
    print("  python main.py")


if __name__ == "__main__":
    main()
