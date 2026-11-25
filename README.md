# Citrus (Kinnow) Leaf Disease Detection Using Machine Learning

An academic image-classification prototype for five Kinnow citrus leaf categories using handcrafted image features and a tuned Random Forest classifier.

## Classes
- Black spot
- Melanose
- canker
- greening
- healthy

## Dataset
Mendeley Data: https://data.mendeley.com/datasets/3f83gxmv57/2

The project used the **Leaves** subset only: 609 images in total. Fruit images were excluded.

| Class | Images |
|---|---:|
| Black spot | 171 |
| Melanose | 13 |
| canker | 163 |
| greening | 204 |
| healthy | 58 |
| **Total** | **609** |

## Method
1. Load leaf images and convert to RGB.
2. Resize images to 256 × 256 pixels.
3. Normalize pixel values to [0, 1].
4. Extract 48 RGB histogram features, 10 LBP texture features, and 8,100 HOG features.
5. Concatenate into 8,158 features per image.
6. Encode class labels and use an 80/20 stratified train/test split (random_state=42).
7. Compare Random Forest, SVM, and KNN; tune Random Forest using RandomizedSearchCV.

## Reported evaluation
The tuned Random Forest achieved 73.77% test accuracy (90 correct out of 122), macro precision 0.6336, macro recall 0.5719, macro F1 0.5845, and weighted F1 0.7264. Five-fold training cross-validation for the initial Random Forest had mean accuracy 70.65% (standard deviation 3.09%).

Melanose has only 13 images in the dataset and only 2 samples in the held-out test set. Its test performance is therefore highly uncertain. Results should not be generalized without a larger, balanced, independently collected dataset.

## Repository contents
- `app.py`: Flask backend and feature extraction for inference.
- `templates/index.html`: user interface.
- `static/style.css`: styling.
- `report/`: project report PDF.
- `requirements.txt`: Python dependencies.

## Model files
The trained model and label encoder are included in the repository root:
- `citrus_disease_model.pkl`
- `citrus_label_encoder.pkl`

These are Python pickle/joblib artifacts. Only load trusted files. Use a compatible scikit-learn version to the one used during training; serialized estimators may raise compatibility warnings or fail across versions.

## Run locally
Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Limitations
- Small and imbalanced dataset.
- Melanose has very few samples.
- Test results are from one stratified holdout split.
- Model confidence is not a calibrated probability of disease or a substitute for expert diagnosis.
- The prototype has not been validated for commercial or field deployment.

## Academic use
This repository is intended for educational and research demonstration. Add your own student details, supervisor details, course information, and institutional formatting to the report cover page before submission.
