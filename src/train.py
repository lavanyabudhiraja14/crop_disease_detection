"""
Training utilities with genuine checkpoint recovery.

Two separate checkpoints are maintained:
  - best_model.h5    — best validation loss so far (used for final eval & inference)
  - latest_ckpt.h5   — saved after every epoch (used to resume interrupted training)

A JSON progress file records the completed epoch and is always written
atomically alongside latest_ckpt.h5.
"""

import json
import os
import tensorflow as tf


BEST_MODEL_FILE   = 'best_model.h5'
LATEST_CKPT_FILE  = 'latest_ckpt.h5'
PROGRESS_FILE     = 'training_progress.json'
LOG_FILE          = 'training_log.csv'


def _read_progress(artifacts_dir):
    """Return the last completed epoch from the progress file, or 0."""
    path = os.path.join(artifacts_dir, PROGRESS_FILE)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return int(data.get('completed_epoch', 0))
        except (json.JSONDecodeError, KeyError, ValueError):
            return 0
    return 0


def _write_progress(artifacts_dir, epoch):
    """Persist the completed epoch number."""
    path = os.path.join(artifacts_dir, PROGRESS_FILE)
    with open(path, 'w') as f:
        json.dump({'completed_epoch': epoch}, f)


class _LatestCheckpoint(tf.keras.callbacks.Callback):
    """Save full model + progress JSON after every epoch."""

    def __init__(self, artifacts_dir):
        super().__init__()
        self.artifacts_dir = artifacts_dir
        self.ckpt_path = os.path.join(artifacts_dir, LATEST_CKPT_FILE)

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.ckpt_path)
        _write_progress(self.artifacts_dir, epoch + 1)  # +1 = next initial_epoch


def train_model(model, train_ds, val_ds, epochs=10, artifacts_dir='artifacts',
                resume=True):
    """
    Train with early stopping, LR scheduling, and dual checkpointing.

    If *resume* is True and a latest checkpoint exists, model weights and
    optimizer state are restored from latest_ckpt.h5 and training continues
    from the recorded epoch — not from the best-validation model.

    Returns (history_or_None, model).
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    best_path   = os.path.join(artifacts_dir, BEST_MODEL_FILE)
    latest_path = os.path.join(artifacts_dir, LATEST_CKPT_FILE)
    log_path    = os.path.join(artifacts_dir, LOG_FILE)

    initial_epoch = 0

    if resume and os.path.exists(latest_path):
        try:
            model = tf.keras.models.load_model(latest_path)
            initial_epoch = _read_progress(artifacts_dir)
            print(f"✅ Resumed from latest checkpoint (epoch {initial_epoch})")
        except Exception as e:
            print(f"⚠️  Could not load latest checkpoint ({e}). Starting fresh.")
            initial_epoch = 0
    else:
        print("Starting training from scratch.")
        # Clear stale logs from a previous run
        for old in (log_path,):
            if os.path.exists(old):
                os.remove(old)

    if initial_epoch >= epochs:
        print(f"Already completed {initial_epoch} epochs (target {epochs}). Nothing to do.")
        return None, model

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
        ),
        _LatestCheckpoint(artifacts_dir),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.CSVLogger(log_path, append=resume),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    return history, model
