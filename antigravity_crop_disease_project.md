# Crop Disease Detection Using Image Classification

## 1. Project goal

Build a complete, beginner-friendly crop disease detection application.
A user uploads a leaf image, and the application predicts the most
likely class represented in the training dataset and displays a
confidence score.

Read this specification before implementing. Inspect the actual dataset
and existing project files instead of assuming the dataset's structure,
image count, or labels.

## 2. Dataset

Kaggle dataset:
https://www.kaggle.com/datasets/shuvranshu/plant-disease-prediction-dataset

Requirements: - Download the dataset from Kaggle directly into Google
Drive from a Google Colab notebook. - Do not require downloading the
dataset to the local computer or uploading the dataset to Antigravity. -
Use the Kaggle API with a `kaggle.json` token supplied securely by the
user in Colab. - Never hard-code, print, commit, or expose Kaggle
credentials. - Inspect the extracted files and report the actual number
of images, directory structure, and class labels. - Do not assume the
dataset contains 13,000 images or occupies 4 GB; verify the real
downloaded dataset. - Make the dataset path configurable. Default Drive
path: `/content/drive/MyDrive/PlantDiseaseDataset` - Handle ZIP
extraction safely and avoid repeatedly downloading or extracting files
when valid files already exist. - Explain any Kaggle account access or
dataset-license requirements the user must satisfy.

## 3. Model and training

Use Python with TensorFlow/Keras unless inspection shows a clear
compatibility reason to choose another framework.

-   Use transfer learning with a lightweight image classifier such as
    MobileNetV2. Choose a suitable alternative only if justified.
-   Inspect the real dataset first. If each image has one class label,
    implement a single-label multiclass classifier. Do not invent a
    hierarchy or labels that the dataset does not contain.
-   Detect class names from the dataset rather than hard-coding them.
-   Create training, validation, and test splits with a fixed random
    seed and no image overlap between splits. If duplicate or
    near-duplicate images can be detected reasonably, prevent leakage
    across splits.
-   Use suitable image resizing, preprocessing, and training-only
    augmentation.
-   Use efficient batch-based loading; do not load the whole dataset
    into RAM.
-   Use class weights if class imbalance warrants them.
-   Detect available hardware at runtime. Use a GPU if available, but
    work on CPU if necessary.
-   Choose conservative defaults for Colab's free-tier memory and GPU
    limits.
-   Include early stopping, learning-rate scheduling where useful, and
    model checkpointing.
-   Save the best model and training checkpoints to Google Drive so
    training can resume after a Colab disconnection.
-   Avoid claiming a particular accuracy before the model has actually
    been trained and evaluated.

## 4. Evaluation

Evaluate the final selected model on the held-out test set. Save: - Test
loss and accuracy. - Precision, recall, and F1-score per class. -
Confusion matrix. - Training/validation accuracy and loss plots. - Class
names and the exact label-to-index mapping. - A short summary of
limitations, especially for images that differ from the training
dataset.

Do not use the test set for model selection or tuning.

## 5. Google Colab notebook

Create a ready-to-run notebook named
`notebooks/train_crop_disease_colab.ipynb`. It must have ordered cells
with clear headings and short instructions.

The notebook should: 1. Install compatible dependencies. 2. Mount Google
Drive. 3. Ask the user to upload `kaggle.json` securely when needed,
configure it with restricted file permissions, and verify Kaggle access
without revealing the token. 4. Download the Kaggle dataset directly
into the Drive folder above. 5. Extract the dataset and inspect its
structure, image count, and classes. 6. Build leakage-safe data splits
and efficient data loaders. 7. Train and evaluate the model using the
available hardware. 8. Save the best model, resumable checkpoints, label
mapping, metrics, plots, and a training summary to:
`/content/drive/MyDrive/PlantDiseaseDataset/artifacts/` 9. Explain how
to resume training after a runtime disconnect. 10. Include checks that
stop with clear error messages if the dataset is missing, labels cannot
be found, or the structure is unsupported.

Do not assume a specific GPU, such as a T4, will be available. Do not
require a Drive share link: mounting Drive in Colab provides the
filesystem path.

## 6. Streamlit application

Build a polished, responsive Streamlit interface suitable for a student
project.

Features: - Project title and a short explanation of what the model
does. - Leaf image upload with supported file types and a clear error
for invalid files. - Image preview. - Predicted class and confidence
score. - Optional top-three predictions when there are enough classes. -
A brief caution that predictions are model estimates, not guaranteed
diagnoses. - Friendly handling of missing model files, unreadable
images, and unexpected input. - Load the trained model once rather than
reloading it on every interaction. - Load class names from the saved
label mapping; never rely on a guessed class order.

The app must clearly explain that predictions require a successfully
trained model artifact. Do not fabricate predictions when the model file
is missing.

## 7. Code organization and engineering

Use a clean, maintainable structure similar to the following, adapting
it when needed:

-   `app.py` --- Streamlit interface
-   `src/data.py` --- dataset inspection and loading helpers
-   `src/model.py` --- model construction
-   `src/train.py` --- reusable training functions
-   `src/evaluate.py` --- metrics and plots
-   `src/predict.py` --- prediction helper
-   `notebooks/train_crop_disease_colab.ipynb` --- Colab download and
    training workflow
-   `tests/` --- tests for data validation, label mapping, and
    prediction utilities
-   `requirements.txt` --- compatible dependencies
-   `README.md` --- complete setup and usage instructions
-   `.gitignore` --- exclude credentials, large datasets, checkpoints,
    and model artifacts when appropriate

Keep local application paths separate from Colab Drive paths and make
paths configurable. Do not put large dataset files or credentials into
source control.

## 8. Testing and verification

-   Add practical tests for dataset discovery, label mapping, invalid
    input handling, and prediction output shape.
-   Run tests that are possible in the current environment.
-   Check imports, file paths, dependency consistency, and notebook
    syntax.
-   Do not claim GPU training, model accuracy, or end-to-end
    functionality has been verified unless it has actually been run.
-   Clearly list steps that still require the user to run the notebook
    in Colab.
-   Fix errors found during verification.

## 9. Required deliverables

Create the actual project files, not just sample snippets: 1. Working
project source code. 2. Ready-to-run Google Colab notebook. 3. Streamlit
application. 4. Dependency file. 5. README with exact beginner-friendly
commands and steps. 6. Tests. 7. A checklist explaining how to download,
train, evaluate, and run the application.

## 10. Expected workflow

1.  Antigravity inspects this specification and the existing project.
2.  Antigravity builds the code, Streamlit app, tests, README, and Colab
    notebook.
3.  The user opens the notebook in Google Colab and selects a GPU if one
    is available.
4.  The user mounts Google Drive and supplies their Kaggle API token
    when prompted.
5.  Colab downloads the dataset directly from Kaggle into Google Drive.
6.  Colab trains and evaluates the model, saving artifacts and
    checkpoints to Drive.
7.  The user runs the Streamlit app with the trained model artifact
    available at the configured path.

Start by inspecting the existing workspace and this specification. Then
implement the project in the workspace, test what can be tested, and
finish with a concise checklist of the exact next actions for the user.
