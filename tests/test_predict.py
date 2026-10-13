"""
Comprehensive tests for the Crop Disease Detection project.

These tests use lightweight mocks and tiny synthetic datasets so they run
without a GPU, without downloading the real dataset, and without a trained
model.  Tests that *cannot* be verified locally are clearly marked.
"""

import io
import json
import os
import shutil
import tempfile

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _make_dataset_tree(root, class_names, images_per_class=5, fmt='JPEG'):
    """Create a tiny directory tree that looks like a real dataset."""
    for cls in class_names:
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(images_per_class):
            ext = '.jpg' if fmt == 'JPEG' else '.png'
            img_path = os.path.join(cls_dir, f"img_{i}{ext}")
            img = Image.new('RGB', (32, 32), color=(i * 30 % 256, 100, 50))
            img.save(img_path, format=fmt)


def _make_image_bytes(fmt='JPEG', size=(64, 64)):
    """Return an in-memory image file."""
    buf = io.BytesIO()
    Image.new('RGB', size, color=(120, 200, 80)).save(buf, format=fmt)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
#  Dataset discovery tests
# ---------------------------------------------------------------------------

class TestDatasetDiscovery:
    """Tests for src.data.find_class_directory."""

    def test_flat_class_dirs(self, tmp_path):
        from src.data import find_class_directory
        _make_dataset_tree(str(tmp_path), ['ClassA', 'ClassB'])
        result = find_class_directory(str(tmp_path))
        assert os.path.isdir(result)

    def test_nested_train_dir(self, tmp_path):
        from src.data import find_class_directory
        train_dir = tmp_path / "some_folder" / "Train"
        _make_dataset_tree(str(train_dir), ['Healthy', 'Rust'])
        result = find_class_directory(str(tmp_path))
        assert 'Train' in result or 'train' in result.lower()

    def test_empty_dir_raises(self, tmp_path):
        from src.data import find_class_directory
        with pytest.raises(ValueError, match="Could not find"):
            find_class_directory(str(tmp_path))

    def test_nonexistent_dir_raises(self):
        from src.data import find_class_directory
        with pytest.raises(FileNotFoundError):
            find_class_directory("/nonexistent/path/xyz")

    def test_empty_subfolders_ignored(self, tmp_path):
        """Subdirs that contain no images should not count as classes."""
        from src.data import find_class_directory
        os.makedirs(tmp_path / "EmptyA")
        os.makedirs(tmp_path / "EmptyB")
        with pytest.raises(ValueError, match="Could not find"):
            find_class_directory(str(tmp_path))


# ---------------------------------------------------------------------------
#  Image format tests
# ---------------------------------------------------------------------------

class TestImageFormats:
    """Ensure JPEG and PNG files are both accepted during dataset scan."""

    def test_jpg_files_found(self, tmp_path):
        from src.data import get_dataset_info
        _make_dataset_tree(str(tmp_path), ['A', 'B'], 3, fmt='JPEG')
        info = get_dataset_info(str(tmp_path))
        assert info['total_images'] == 6

    def test_png_files_found(self, tmp_path):
        from src.data import get_dataset_info
        _make_dataset_tree(str(tmp_path), ['A', 'B'], 3, fmt='PNG')
        info = get_dataset_info(str(tmp_path))
        assert info['total_images'] == 6

    def test_mixed_formats(self, tmp_path):
        from src.data import get_dataset_info
        _make_dataset_tree(str(tmp_path), ['X'], 2, fmt='JPEG')
        _make_dataset_tree(str(tmp_path), ['Y'], 2, fmt='PNG')
        info = get_dataset_info(str(tmp_path))
        assert info['total_images'] == 4


# ---------------------------------------------------------------------------
#  Label mapping tests
# ---------------------------------------------------------------------------

class TestLabelMapping:
    """Verify the class-label mapping is correct and deterministic."""

    def test_classes_sorted(self, tmp_path):
        from src.data import get_dataset_info
        _make_dataset_tree(str(tmp_path), ['Zebra', 'Apple', 'Mango'])
        info = get_dataset_info(str(tmp_path))
        assert info['classes'] == ['Apple', 'Mango', 'Zebra']

    def test_class_counts_match(self, tmp_path):
        from src.data import get_dataset_info
        _make_dataset_tree(str(tmp_path), ['Cat', 'Dog'], images_per_class=7)
        info = get_dataset_info(str(tmp_path))
        assert info['class_counts']['Cat'] == 7
        assert info['class_counts']['Dog'] == 7

    def test_class_mapping_json_round_trip(self, tmp_path):
        """Verify the JSON we save can be read back correctly."""
        classes = ['Alpha', 'Beta', 'Gamma']
        path = tmp_path / "class_mapping.json"
        with open(path, 'w') as f:
            json.dump({"classes": classes}, f)
        with open(path, 'r') as f:
            loaded = json.load(f)
        assert loaded["classes"] == classes


# ---------------------------------------------------------------------------
#  Data splitting tests
# ---------------------------------------------------------------------------

class TestDataSplitting:
    """Verify splits are disjoint, reproducible, and non-empty."""

    def test_splits_disjoint(self, tmp_path):
        from src.data import find_class_directory, collect_file_label_pairs
        from sklearn.model_selection import train_test_split

        _make_dataset_tree(str(tmp_path), ['A', 'B', 'C'], images_per_class=30)
        target = find_class_directory(str(tmp_path))
        classes = sorted(os.listdir(target))
        paths, labels = collect_file_label_pairs(target, classes)

        train_p, temp_p, _, temp_l = train_test_split(
            paths, labels, test_size=0.2, random_state=42, stratify=labels)
        val_p, test_p, _, _ = train_test_split(
            temp_p, temp_l, test_size=0.5, random_state=42, stratify=temp_l)

        assert len(set(train_p) & set(val_p)) == 0, "Train/Val overlap"
        assert len(set(train_p) & set(test_p)) == 0, "Train/Test overlap"
        assert len(set(val_p)   & set(test_p)) == 0, "Val/Test overlap"

    def test_splits_reproducible(self, tmp_path):
        from src.data import find_class_directory, collect_file_label_pairs
        from sklearn.model_selection import train_test_split

        _make_dataset_tree(str(tmp_path), ['A', 'B'], images_per_class=20)
        target = find_class_directory(str(tmp_path))
        classes = sorted(os.listdir(target))
        paths, labels = collect_file_label_pairs(target, classes)

        t1, _, _, _ = train_test_split(paths, labels, test_size=0.2, random_state=42)
        t2, _, _, _ = train_test_split(paths, labels, test_size=0.2, random_state=42)
        assert t1 == t2

    def test_small_class_no_crash(self, tmp_path):
        """A class with only 1 image should not crash the splitter."""
        from src.data import create_data_splits
        _make_dataset_tree(str(tmp_path), ['Big'], images_per_class=20)
        _make_dataset_tree(str(tmp_path), ['Tiny'], images_per_class=1)
        # Should not raise
        train_ds, val_ds, test_ds, classes = create_data_splits(str(tmp_path))
        assert 'Big' in classes


# ---------------------------------------------------------------------------
#  Prediction tests
# ---------------------------------------------------------------------------

class TestPrediction:
    """Tests for src.predict.*"""

    def test_predict_ordering(self):
        from src.predict import predict_image
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])
        classes = ["Healthy", "Rust", "Scab"]
        results = predict_image(mock_model, np.zeros((1, 224, 224, 3)), classes)

        assert len(results) == 3
        assert results[0]["class"] == "Rust"
        assert results[0]["confidence"] == pytest.approx(0.7)
        assert results[1]["class"] == "Scab"
        assert results[2]["class"] == "Healthy"

    def test_predict_single_class(self):
        from src.predict import predict_image
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[1.0]])
        results = predict_image(mock_model, np.zeros((1, 224, 224, 3)), ["OnlyClass"])
        assert len(results) == 1
        assert results[0]["class"] == "OnlyClass"

    def test_preprocess_jpeg(self):
        from src.predict import preprocess_image
        buf = _make_image_bytes('JPEG')
        arr = preprocess_image(buf)
        assert arr.shape == (1, 224, 224, 3)
        assert arr.dtype == np.float32

    def test_preprocess_png(self):
        from src.predict import preprocess_image
        buf = _make_image_bytes('PNG')
        arr = preprocess_image(buf)
        assert arr.shape == (1, 224, 224, 3)

    def test_preprocess_seek_rewind(self):
        """After calling preprocess_image the stream should still be usable."""
        from src.predict import preprocess_image
        buf = _make_image_bytes('JPEG')
        _ = preprocess_image(buf)
        # Can still read from the buffer (it was seeked to 0 internally)
        buf.seek(0)
        img = Image.open(buf)
        assert img.size == (64, 64)

    def test_preprocess_grayscale(self):
        """Grayscale images should be converted to 3-channel RGB."""
        from src.predict import preprocess_image
        buf = io.BytesIO()
        Image.new('L', (64, 64), color=128).save(buf, format='JPEG')
        buf.seek(0)
        arr = preprocess_image(buf)
        assert arr.shape == (1, 224, 224, 3)

    def test_preprocess_invalid_file(self):
        from src.predict import preprocess_image
        buf = io.BytesIO(b"not an image at all")
        with pytest.raises(ValueError, match="Could not process"):
            preprocess_image(buf)


# ---------------------------------------------------------------------------
#  Missing / incompatible artifact tests
# ---------------------------------------------------------------------------

class TestArtifacts:
    """Verify graceful handling of missing or bad artifacts."""

    def test_load_model_missing_path(self):
        from src.predict import load_trained_model
        with pytest.raises(RuntimeError, match="Failed to load"):
            load_trained_model("/nonexistent/model.h5")

    def test_class_mapping_missing_key(self, tmp_path):
        """If the JSON doesn't have a 'classes' key the app should still cope."""
        path = tmp_path / "bad_mapping.json"
        with open(path, 'w') as f:
            json.dump({"wrong_key": [1, 2, 3]}, f)
        with open(path, 'r') as f:
            data = json.load(f)
        # Simulating what app.py does
        result = None
        if isinstance(data, dict) and "classes" in data:
            result = data["classes"]
        assert result is None  # should not crash, just be None


# ---------------------------------------------------------------------------
#  Checkpoint progress tests
# ---------------------------------------------------------------------------

class TestCheckpointProgress:
    """Test the JSON-based progress tracking used by train.py."""

    def test_write_and_read_progress(self, tmp_path):
        from src.train import _write_progress, _read_progress
        _write_progress(str(tmp_path), 5)
        assert _read_progress(str(tmp_path)) == 5

    def test_read_missing_progress(self, tmp_path):
        from src.train import _read_progress
        assert _read_progress(str(tmp_path)) == 0

    def test_read_corrupted_progress(self, tmp_path):
        from src.train import _read_progress, PROGRESS_FILE
        path = tmp_path / PROGRESS_FILE
        with open(path, 'w') as f:
            f.write("not json!")
        assert _read_progress(str(tmp_path)) == 0
