"""
Unit tests for data preparation module.
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path to import train_classifier
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_classifier import (
    derive_label_from_filename,
    load_and_label_documents,
    validate_dataset,
    create_train_val_split
)


class TestDeriveLabelFromFilename:
    """Tests for derive_label_from_filename function."""
    
    def test_invoice_label(self):
        """Test invoice filename mapping."""
        assert derive_label_from_filename("invoice_1.pdf") == "Invoice"
        assert derive_label_from_filename("invoice_123.pdf") == "Invoice"
    
    def test_resume_label(self):
        """Test resume filename mapping."""
        assert derive_label_from_filename("resume_1.pdf") == "Resume"
        assert derive_label_from_filename("resume_42.pdf") == "Resume"
    
    def test_utilitybill_label(self):
        """Test utility bill filename mapping."""
        assert derive_label_from_filename("utilitybill_1.pdf") == "Utility Bill"
        assert derive_label_from_filename("utilitybill_5.pdf") == "Utility Bill"
    
    def test_other_label(self):
        """Test other filename mapping."""
        assert derive_label_from_filename("other_1.pdf") == "Other"
        assert derive_label_from_filename("other_99.pdf") == "Other"
    
    def test_unclassifiable_label(self):
        """Test unclassifiable filename mapping."""
        assert derive_label_from_filename("unclassifiable_1.pdf") == "Unclassifiable"
        assert derive_label_from_filename("unclassifiable_7.pdf") == "Unclassifiable"
    
    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        assert derive_label_from_filename("INVOICE_1.pdf") == "Invoice"
        assert derive_label_from_filename("Invoice_1.pdf") == "Invoice"
        assert derive_label_from_filename("RESUME_1.pdf") == "Resume"
        assert derive_label_from_filename("Resume_1.pdf") == "Resume"
    
    def test_unknown_prefix_defaults_to_other(self):
        """Test unknown prefix defaults to Other."""
        assert derive_label_from_filename("unknown_1.pdf") == "Other"
        assert derive_label_from_filename("random_file.pdf") == "Other"
    
    def test_multiple_underscores(self):
        """Test filename with multiple underscores uses first prefix."""
        assert derive_label_from_filename("invoice_test_1.pdf") == "Invoice"
        assert derive_label_from_filename("resume_john_doe_1.pdf") == "Resume"
    
    def test_no_underscore(self):
        """Test filename without underscore."""
        assert derive_label_from_filename("invoice.pdf") == "Invoice"
        assert derive_label_from_filename("resume.pdf") == "Resume"


class TestValidateDataset:
    """Tests for validate_dataset function."""
    
    def test_valid_dataset(self):
        """Test validation passes for valid dataset."""
        documents = [
            ("text1", "Invoice"),
            ("text2", "Invoice"),
            ("text3", "Resume"),
            ("text4", "Resume"),
            ("text5", "Utility Bill"),
            ("text6", "Utility Bill"),
            ("text7", "Other"),
            ("text8", "Other"),
            ("text9", "Unclassifiable"),
            ("text10", "Unclassifiable"),
        ]
        # Should not raise
        validate_dataset(documents)
    
    def test_insufficient_documents(self):
        """Test validation fails with fewer than 10 documents."""
        documents = [
            ("text1", "Invoice"),
            ("text2", "Resume"),
            ("text3", "Other"),
        ]
        with pytest.raises(ValueError, match="Insufficient training data"):
            validate_dataset(documents)
    
    def test_empty_text_raises_error(self):
        """Test validation fails with empty text."""
        documents = [
            ("text1", "Invoice"),
            ("", "Resume"),  # Empty text
            ("text3", "Other"),
            ("text4", "Invoice"),
            ("text5", "Resume"),
            ("text6", "Other"),
            ("text7", "Invoice"),
            ("text8", "Resume"),
            ("text9", "Other"),
            ("text10", "Invoice"),
        ]
        with pytest.raises(ValueError, match="empty or invalid text"):
            validate_dataset(documents)
    
    def test_non_string_text_raises_error(self):
        """Test validation fails with non-string text."""
        documents = [
            ("text1", "Invoice"),
            (None, "Resume"),  # None text
            ("text3", "Other"),
            ("text4", "Invoice"),
            ("text5", "Resume"),
            ("text6", "Other"),
            ("text7", "Invoice"),
            ("text8", "Resume"),
            ("text9", "Other"),
            ("text10", "Invoice"),
        ]
        with pytest.raises(ValueError, match="empty or invalid text"):
            validate_dataset(documents)
    
    def test_category_with_one_document_warns(self, capsys):
        """Test warning is displayed for categories with only 1 document."""
        documents = [
            ("text1", "Invoice"),
            ("text2", "Invoice"),
            ("text3", "Resume"),
            ("text4", "Resume"),
            ("text5", "Utility Bill"),  # Only 1 Utility Bill
            ("text6", "Other"),
            ("text7", "Other"),
            ("text8", "Unclassifiable"),
            ("text9", "Unclassifiable"),
            ("text10", "Invoice"),
        ]
        validate_dataset(documents)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Utility Bill" in captured.out


class TestCreateTrainValSplit:
    """Tests for create_train_val_split function."""
    
    def test_split_ratio(self):
        """Test split maintains approximately correct ratio."""
        documents = [
            ("text1", "Invoice"),
            ("text2", "Invoice"),
            ("text3", "Resume"),
            ("text4", "Resume"),
            ("text5", "Utility Bill"),
            ("text6", "Utility Bill"),
            ("text7", "Other"),
            ("text8", "Other"),
            ("text9", "Unclassifiable"),
            ("text10", "Unclassifiable"),
        ]
        train_data, val_data = create_train_val_split(documents, val_ratio=0.2, random_seed=42)
        
        total = len(documents)
        expected_val = int(total * 0.2)
        
        # Allow ±1 for rounding
        assert abs(len(val_data) - expected_val) <= 1
        assert len(train_data) + len(val_data) == total
    
    def test_stratified_split_maintains_categories(self):
        """Test stratified split includes all categories in both sets."""
        documents = [
            ("text1", "Invoice"),
            ("text2", "Invoice"),
            ("text3", "Invoice"),
            ("text4", "Invoice"),
            ("text5", "Resume"),
            ("text6", "Resume"),
            ("text7", "Resume"),
            ("text8", "Resume"),
            ("text9", "Other"),
            ("text10", "Other"),
        ]
        train_data, val_data = create_train_val_split(documents, val_ratio=0.2, random_seed=42)
        
        train_labels = set(label for text, label in train_data)
        val_labels = set(label for text, label in val_data)
        
        # All categories should appear in training
        assert "Invoice" in train_labels
        assert "Resume" in train_labels
        assert "Other" in train_labels
    
    def test_reproducibility_with_seed(self):
        """Test same seed produces same split."""
        documents = [
            ("text1", "Invoice"),
            ("text2", "Invoice"),
            ("text3", "Resume"),
            ("text4", "Resume"),
            ("text5", "Utility Bill"),
            ("text6", "Utility Bill"),
            ("text7", "Other"),
            ("text8", "Other"),
            ("text9", "Unclassifiable"),
            ("text10", "Unclassifiable"),
        ]
        
        train1, val1 = create_train_val_split(documents, val_ratio=0.2, random_seed=42)
        train2, val2 = create_train_val_split(documents, val_ratio=0.2, random_seed=42)
        
        assert train1 == train2
        assert val1 == val2
    
    def test_single_document_category_warning(self, capsys):
        """Test warning for categories with only 1 document."""
        documents = [
            ("text1", "Invoice"),
            ("text2", "Invoice"),
            ("text3", "Resume"),
            ("text4", "Resume"),
            ("text5", "Utility Bill"),  # Only 1
            ("text6", "Other"),
            ("text7", "Other"),
            ("text8", "Unclassifiable"),
            ("text9", "Unclassifiable"),
            ("text10", "Invoice"),
        ]
        
        train_data, val_data = create_train_val_split(documents, val_ratio=0.2, random_seed=42)
        captured = capsys.readouterr()
        
        assert "Warning" in captured.out
        assert "Utility Bill" in captured.out
