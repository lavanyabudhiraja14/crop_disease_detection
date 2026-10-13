"""
Evaluation utilities — metrics, classification report, confusion matrix.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, test_ds, classes, artifacts_dir='artifacts'):
    """
    Evaluate the trained model on a held-out test set.

    Saves:
      - confusion_matrix.png
      - evaluation_results.json
    Prints the classification report.

    `labels=range(len(classes))` ensures the report includes rows for classes
    that may be absent from the test split.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # Overall loss & accuracy
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Collect predictions
    y_true = []
    y_pred = []
    for images, labels_batch in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels_batch.numpy(), axis=-1))
        y_pred.extend(np.argmax(preds, axis=-1))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report — handles missing classes safely
    all_labels = list(range(len(classes)))
    report_dict = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=classes,
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report_text)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    fig_size = max(8, len(classes) * 0.6)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Save results JSON
    results = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "classification_report": report_dict,
    }
    with open(os.path.join(artifacts_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_training_history(history, artifacts_dir='artifacts'):
    """Save training / validation accuracy and loss curves."""
    os.makedirs(artifacts_dir, exist_ok=True)

    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'training_history.png'), dpi=150)
    plt.close()
