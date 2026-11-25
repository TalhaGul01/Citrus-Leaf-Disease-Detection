from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import numpy as np
import joblib
import os
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "citrus_disease_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "citrus_label_encoder.pkl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_artifacts():
    """Load the trained Random Forest model and label encoder."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            "Model files are missing. Add citrus_disease_model.pkl and "
            "citrus_label_encoder.pkl to the project root. See README.md."
        )
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


model, label_encoder = load_artifacts()


def extract_features(image):
    """
    Extract the same 8,158 features used during training:
    RGB histograms (48) + LBP histogram (10) + HOG (8,100).
    """
    image = image.convert("RGB").resize((256, 256))
    rgb = np.array(image, dtype=np.float32) / 255.0
    img_uint8 = (rgb * 255).astype("uint8")

    color_features = []
    for channel in range(3):
        hist, _ = np.histogram(
            img_uint8[:, :, channel],
            bins=16,
            range=(0, 256),
            density=True
        )
        color_features.extend(hist)

    gray = rgb2gray(rgb)
    lbp = local_binary_pattern(
        (gray * 255).astype("uint8"),
        P=8,
        R=1,
        method="uniform"
    )
    texture_hist, _ = np.histogram(
        lbp,
        bins=10,
        range=(0, 10),
        density=True
    )

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    combined = np.concatenate([
        np.asarray(color_features),
        texture_hist,
        hog_features
    ])

    if combined.shape[0] != 8158:
        raise ValueError(
            f"Expected 8158 features, but extracted {combined.shape[0]}."
        )

    return combined.reshape(1, -1)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("file")

    if uploaded_file is None or uploaded_file.filename == "":
        return render_template("index.html", error="Please choose a leaf image.")

    if not allowed_file(uploaded_file.filename):
        return render_template(
            "index.html",
            error="Unsupported file type. Please upload JPG, JPEG, or PNG."
        )

    try:
        image = Image.open(uploaded_file.stream).convert("RGB")
        features = extract_features(image)

        predicted_label = model.predict(features)[0]
        disease = label_encoder.inverse_transform([predicted_label])[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = round(float(np.max(probabilities)) * 100, 2)

        safe_name = secure_filename(uploaded_file.filename)
        # Unique filename prevents one user's upload from overwriting another.
        import uuid
        filename = f"{uuid.uuid4().hex}_{safe_name}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path)

        relative_image_path = f"/static/uploads/{filename}"

        return render_template(
            "index.html",
            prediction=str(disease),
            confidence=confidence,
            image_path=relative_image_path
        )

    except UnidentifiedImageError:
        return render_template("index.html", error="The uploaded file is not a valid image.")
    except Exception as exc:
        app.logger.exception("Prediction error")
        return render_template(
            "index.html",
            error=f"Prediction could not be completed: {exc}"
        )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        error="Image is too large. Maximum upload size is 10 MB."
    ), 413


if __name__ == "__main__":
    # Debug is disabled by default. Use only a development server locally.
    app.run(host="127.0.0.1", port=5000, debug=False)
