# 🌿 Crop Disease Detection Using Image Classification

A beginner-friendly application that predicts plant diseases from leaf images using a MobileNetV2 deep-learning model trained on the [Plant Disease Prediction Dataset](https://www.kaggle.com/datasets/shuvranshu/plant-disease-prediction-dataset).

## Project Structure

```
├── app.py                    # Streamlit web application
├── src/
│   ├── data.py               # Dataset discovery & preprocessing
│   ├── model.py              # Model architecture
│   ├── train.py              # Training with dual checkpoints
│   ├── evaluate.py           # Evaluation metrics & plots
│   └── predict.py            # Inference utilities
├── notebooks/
│   └── train_crop_disease_colab.ipynb   # Google Colab training notebook
├── tests/
│   └── test_predict.py       # Automated tests
├── artifacts/                # Trained model & class mapping (not in git)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Step-by-Step Guide

### Step 1 — Get a Kaggle API Token

1. Create a free account at [kaggle.com](https://www.kaggle.com/) if you don't have one.
2. Go to **[Account Settings](https://www.kaggle.com/settings)**.
3. Under the **API** section click **Create New Token**.
4. A file called `kaggle.json` downloads to your computer. **Keep it safe — do not share it.**

### Step 2 — Train the Model in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File → Upload notebook** and select  
   `notebooks/train_crop_disease_colab.ipynb`.
3. Go to **Runtime → Change runtime type** and choose **GPU** (any type is fine — the code detects what is available).
4. Run every cell **in order from top to bottom**.
   - When prompted, upload the `kaggle.json` file from Step 1.
   - The dataset (~4 GB) downloads directly into your Google Drive — nothing is downloaded to your Mac.
   - Training starts automatically.
5. When training finishes the following files are saved to  
   `Google Drive / MyDrive / PlantDiseaseDataset / artifacts/`:

| File | Purpose |
|---|---|
| `best_model.h5` | Best model (for predictions) |
| `latest_ckpt.h5` | Latest checkpoint (for resuming) |
| `class_mapping.json` | Class names in correct order |
| `training_progress.json` | Epoch progress |
| `training_log.csv` | Per-epoch metrics |
| `confusion_matrix.png` | Confusion matrix plot |
| `training_history.png` | Accuracy/loss curves |
| `evaluation_results.json` | Test metrics |

### Step 2b — Resuming After a Disconnect

If Colab disconnects during training:

1. Reopen the notebook in Colab (select GPU again).
2. Run **all cells from the top**.
3. The dataset will not re-download (ZIP is still on Drive).
4. Training automatically resumes from `latest_ckpt.h5` at the correct epoch with the optimizer state intact.
5. `best_model.h5` is only overwritten when validation loss improves — it is never overwritten by the resume process.

### Step 3 — Run the Streamlit App Locally

1. Open Google Drive in your browser.
2. Navigate to `MyDrive / PlantDiseaseDataset / artifacts/`.
3. Download **`best_model.h5`** and **`class_mapping.json`** to your Mac.
4. Place both files inside the `artifacts/` folder of this project:
   ```
   Ai Society Project/
     artifacts/
       best_model.h5
       class_mapping.json
   ```
5. Open a terminal in the project folder and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
6. Open the Local URL shown in the terminal, upload a leaf image, and see the prediction!

> **Tip:** You can set `ARTIFACTS_DIR=/some/other/path` as an environment variable to load artifacts from a different location.

### Step 4 — Run Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Tests use synthetic mini-datasets and mocks — they do not require the real dataset or a GPU.

---

## How It Works

1. **Transfer Learning** — MobileNetV2 (pre-trained on ImageNet) provides a powerful feature extractor. Only a small classification head is trained on the plant-disease images.
2. **Data Augmentation** — Random flips, rotations, and zooms during training help the model generalise.
3. **Preprocessing** — Images are resized to 224 × 224 and kept as float32 in [0, 255]. The model's built-in `preprocess_input` layer normalises to [-1, 1].
4. **Dual Checkpoints** — `best_model.h5` stores the best validation model; `latest_ckpt.h5` stores the latest epoch for genuine resume capability.

## Limitations

- The model is only as good as its training data. Real-world images with different lighting, backgrounds, or unseen diseases may produce incorrect predictions.
- Predictions are **estimates**, not professional diagnoses.
- Training accuracy is only determined after the notebook actually runs in Colab — no results are fabricated beforehand.
