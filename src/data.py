"""
Dataset discovery, inspection, and preprocessing utilities.
Supports JPEG, JPG, and PNG images. Uses efficient TF data pipelines
without loading entire dataset into RAM.
"""

import os
from collections import Counter
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Supported image extensions (lowercase)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def _is_image_file(filename):
    """Check if a filename has a supported image extension."""
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def find_class_directory(base_dir):
    """
    Walk the extracted dataset and find the directory whose immediate
    subdirectories are the class-label folders.

    Strategy:
      1. If a child named 'train' or 'Training' (case-insensitive) exists,
         return it — most Kaggle plant-disease datasets ship this way.
      2. Otherwise, find the shallowest directory that has ≥ 2 subdirectories
         each containing at least one image file.

    Raises FileNotFoundError if base_dir does not exist.
    Raises ValueError if no usable class directory is found.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {base_dir}")

    # --- pass 1: look for an official train / training split ---------------
    for root, dirs, _files in os.walk(base_dir):
        for d in sorted(dirs):
            if d.lower() in ('train', 'training'):
                candidate = os.path.join(root, d)
                subs = _get_class_subdirs(candidate)
                if len(subs) >= 2:
                    return candidate

    # --- pass 2: shallowest dir with ≥ 2 image-containing subdirs ---------
    for root, dirs, _files in sorted(os.walk(base_dir), key=lambda t: t[0].count(os.sep)):
        subs = _get_class_subdirs(root)
        if len(subs) >= 2:
            return root

    raise ValueError(
        f"Could not find a valid class directory inside '{base_dir}'. "
        "Expected at least 2 subdirectories each containing image files."
    )


def _get_class_subdirs(directory):
    """Return sorted list of subdirectory names that contain ≥ 1 image."""
    result = []
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return result
    for name in entries:
        subdir = os.path.join(directory, name)
        if not os.path.isdir(subdir):
            continue
        # Check if it has at least one image
        try:
            if any(_is_image_file(f) for f in os.listdir(subdir)):
                result.append(name)
        except OSError:
            continue
    return result


def get_dataset_info(dataset_dir):
    """
    Inspect the dataset directory and return a dict with:
      target_dir, classes, num_classes, total_images, class_counts
    """
    target_dir = find_class_directory(dataset_dir)
    classes = _get_class_subdirs(target_dir)

    class_counts = {}
    total_images = 0
    for cls in classes:
        cls_dir = os.path.join(target_dir, cls)
        count = sum(1 for f in os.listdir(cls_dir) if _is_image_file(f))
        class_counts[cls] = count
        total_images += count

    return {
        "target_dir": target_dir,
        "classes": classes,
        "num_classes": len(classes),
        "total_images": total_images,
        "class_counts": class_counts,
    }


def collect_file_label_pairs(target_dir, classes):
    """
    Walk each class subdirectory and return (filepaths, labels) lists.
    Labels are integer indices aligned with the sorted `classes` list.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    filepaths = []
    labels = []
    for cls in classes:
        cls_dir = os.path.join(target_dir, cls)
        for fname in os.listdir(cls_dir):
            if _is_image_file(fname):
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])
    return filepaths, labels


def create_data_splits(dataset_dir, img_size=(224, 224), batch_size=32, seed=42):
    """
    Build disjoint, reproducible, stratified train / val / test splits.

    * 80 % train, 10 % validation, 10 % test.
    * Stratification is used when every class has enough samples; otherwise
      it falls back to a plain random split so sklearn doesn't raise.
    * Returns (train_ds, val_ds, test_ds) as tf.data.Datasets plus the
      ordered list of class names.
    * Uses decode_image (not decode_jpeg) so PNG files work too.
    * Does NOT call .cache() — the dataset may be several GB.
    """
    target_dir = find_class_directory(dataset_dir)
    classes = _get_class_subdirs(target_dir)
    filepaths, labels = collect_file_label_pairs(target_dir, classes)

    if not filepaths:
        raise ValueError("No images found in the dataset directory.")

    num_classes = len(classes)

    # --- stratification safety check (uses Counter, O(n)) ----------------
    counts = Counter(labels)
    can_stratify_main = all(c >= 3 for c in counts.values())
    strat_main = labels if can_stratify_main else None

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        filepaths, labels,
        test_size=0.2, random_state=seed, stratify=strat_main,
    )

    temp_counts = Counter(temp_labels)
    can_stratify_temp = can_stratify_main and all(c >= 2 for c in temp_counts.values())
    strat_temp = temp_labels if can_stratify_temp else None

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.5, random_state=seed, stratify=strat_temp,
    )

    # --- validate splits are non-empty ------------------------------------
    for name, paths in [("Training", train_paths), ("Validation", val_paths), ("Test", test_paths)]:
        if len(paths) == 0:
            raise ValueError(f"{name} split is empty. Dataset may be too small.")

    # --- TF dataset builder -----------------------------------------------
    def _make_dataset(paths, lbls, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((list(paths), list(lbls)))
        if shuffle:
            ds = ds.shuffle(len(paths), seed=seed)

        def _process(file_path, label):
            raw = tf.io.read_file(file_path)
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            img = tf.image.resize(img, img_size)
            return img, tf.one_hot(label, num_classes)

        ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _make_dataset(train_paths, train_labels, shuffle=True)
    val_ds   = _make_dataset(val_paths,   val_labels)
    test_ds  = _make_dataset(test_paths,  test_labels)

    return train_ds, val_ds, test_ds, classes
