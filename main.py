import os
import cv2
import pickle
import tempfile
import numpy as np
import Orange

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="FootFit-AI Backend")


# ============================================================
# CORS
#
# Allows the Lovable/Netlify frontend to communicate
# with the Render backend from a web browser.
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# EXISTING ORANGE MODELS
# ============================================================

MODEL_FILES = {
    "Random Forest": "random_forest.pkcls",
    "Tree": "trees.pkcls",
    "kNN": "knn.pkcls",
    "SVM": "svm.pkcls",
    "Logistic Regression": "logistic_regression.pkcls"
}


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "FootFit-AI Backend Running"
    }


# ============================================================
# IMAGE → 7 FEATURES
# ============================================================

def extract_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError("Could not read uploaded image.")

    original_height, original_width = image.shape[:2]

    # Same scaling method used in Stage 2 testing
    upscale_factor = min(
        8.0,
        max(1.0, 1200 / max(original_width, original_height))
    )

    new_width = int(round(original_width * upscale_factor))
    new_height = int(round(original_height * upscale_factor))

    upscaled = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # YCrCb segmentation
    # --------------------------------------------------------

    ycrcb = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2YCrCb
    )

    lower = np.array(
        [0, 133, 77],
        dtype=np.uint8
    )

    upper = np.array(
        [255, 173, 127],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        ycrcb,
        lower,
        upper
    )

    # --------------------------------------------------------
    # Clean segmentation
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # Find foot contour
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError("No foot contour detected.")

    contour = max(
        contours,
        key=cv2.contourArea
    )

    raw_area = cv2.contourArea(contour)

    if raw_area <= 0:
        raise RuntimeError("Detected contour has zero area.")

    raw_perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, raw_width, raw_height = cv2.boundingRect(
        contour
    )

    # --------------------------------------------------------
    # Convex hull
    # --------------------------------------------------------

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = (
        raw_area / hull_area
        if hull_area > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Bounding rectangle
    # --------------------------------------------------------

    bounding_area = raw_width * raw_height

    extent = (
        raw_area / bounding_area
        if bounding_area > 0
        else 0.0
    )

    aspect_ratio = (
        raw_width / raw_height
        if raw_height > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Convert measurements back to original scale
    # --------------------------------------------------------

    contour_area = raw_area / (upscale_factor ** 2)
    perimeter = raw_perimeter / upscale_factor
    width = raw_width / upscale_factor
    height = raw_height / upscale_factor

    return {
        "Contour Area (px^2)": contour_area,
        "Perimeter (px)": perimeter,
        "Bounding Rect Width (px)": width,
        "Bounding Rect Height (px)": height,
        "Solidity": solidity,
        "Extent": extent,
        "Aspect Ratio": aspect_ratio
    }


# ============================================================
# 7 FEATURES → EXISTING ORANGE MODELS
# ============================================================

def predict_from_features(feature_values):

    predictions = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = os.path.join(
            MODEL_DIR,
            filename
        )

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model not found: {filename}"
            )

        # Load the existing Orange model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        original_domain = model.original_domain

        row = []

        for variable in original_domain.attributes:

            if variable.name not in feature_values:
                raise RuntimeError(
                    f"Missing feature: {variable.name}"
                )

            row.append(
                feature_values[variable.name]
            )

        y = np.array(
            [[np.nan]],
            dtype=float
        )

        metas = np.array(
            [["", "", "", ""]],
            dtype=object
        )

        data = Orange.data.Table.from_numpy(
            original_domain,
            np.array([row], dtype=float),
            Y=y,
            metas=metas
        )

        prediction = model(data)

        class_index = int(prediction[0])

        class_name = model.domain.class_var.values[
            class_index
        ]

        predictions[model_name] = class_name

    return predictions


# ============================================================
# HYBRID ENSEMBLE DECISION
#
# STAGE 1:
#   All 5 models vote.
#
#   5–0 → Very High Confidence
#   4–1 → High Confidence
#   3–2 → Go to Stage 2
#
# STAGE 2:
#   Top 3 models:
#   Random Forest + Tree + kNN
#
#   3–0 → High Confidence
#   2–1 → Moderate Confidence
# ============================================================

def decide_final_prediction(predictions):

    # --------------------------------------------------------
    # STAGE 1 — ALL 5 MODELS
    # --------------------------------------------------------

    vote_counts = {}

    for prediction in predictions.values():

        vote_counts[prediction] = (
            vote_counts.get(prediction, 0) + 1
        )

    # Sort predictions by number of votes
    sorted_votes = sorted(
        vote_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    stage_1_result = sorted_votes[0][0]
    stage_1_votes = sorted_votes[0][1]

    # --------------------------------------------------------
    # 5–0
    # --------------------------------------------------------

    if stage_1_votes == 5:

        return {
            "final_prediction": stage_1_result,
            "confidence": "Very High",
            "decision_stage": "All 5 models",

            "logic": (
                "All 5 models agreed on the same foot type."
            ),

            "stage_1": {
                "method": "Majority vote of all 5 models",
                "vote_counts": vote_counts,
                "result": stage_1_result,
                "votes": stage_1_votes,
                "status": "5–0 agreement"
            },

            "stage_2": None
        }

    # --------------------------------------------------------
    # 4–1
    # --------------------------------------------------------

    if stage_1_votes == 4:

        return {
            "final_prediction": stage_1_result,
            "confidence": "High",
            "decision_stage": "All 5 models",

            "logic": (
                f"4 out of 5 models predicted "
                f"{stage_1_result}. "
                f"The majority prediction was accepted."
            ),

            "stage_1": {
                "method": "Majority vote of all 5 models",
                "vote_counts": vote_counts,
                "result": stage_1_result,
                "votes": stage_1_votes,
                "status": "4–1 majority"
            },

            "stage_2": None
        }

    # --------------------------------------------------------
    # 3–2
    #
    # If all five models produce a 3–2 split,
    # use only the top 3 models.
    # --------------------------------------------------------

    if stage_1_votes == 3:

        top_3_models = [
            "Random Forest",
            "Tree",
            "kNN"
        ]

        # Get predictions from top 3
        top_3_predictions = {
            model: predictions[model]
            for model in top_3_models
        }

        # Count top 3 votes
        top_3_vote_counts = {}

        for prediction in top_3_predictions.values():

            top_3_vote_counts[prediction] = (
                top_3_vote_counts.get(prediction, 0) + 1
            )

        # Sort top 3 votes
        top_3_sorted = sorted(
            top_3_vote_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )

        top_3_result = top_3_sorted[0][0]
        top_3_votes = top_3_sorted[0][1]

        # ----------------------------------------------------
        # TOP 3: 3–0
        # ----------------------------------------------------

        if top_3_votes == 3:

            confidence = "High"
            top_3_status = "3–0 agreement"

        # ----------------------------------------------------
        # TOP 3: 2–1
        # ----------------------------------------------------

        elif top_3_votes == 2:

            confidence = "Moderate"
            top_3_status = "2–1 majority"

        # ----------------------------------------------------
        # Unexpected case
        # ----------------------------------------------------

        else:

            confidence = "Low"
            top_3_status = "No clear majority"

        return {
            "final_prediction": top_3_result,
            "confidence": confidence,
            "decision_stage": "Top 3 models",

            "logic": (
                "The 5-model vote resulted in a 3–2 split. "
                "Because there was no strong consensus, "
                "the three strongest models — Random Forest, "
                "Tree, and kNN — were used for a second "
                "majority vote."
            ),

            "stage_1": {
                "method": "Majority vote of all 5 models",
                "vote_counts": vote_counts,
                "result": stage_1_result,
                "votes": stage_1_votes,
                "status": "3–2 split"
            },

            "stage_2": {
                "method": "Majority vote of top 3 models",
                "models_used": top_3_models,
                "predictions": top_3_predictions,
                "vote_counts": top_3_vote_counts,
                "result": top_3_result,
                "votes": top_3_votes,
                "status": top_3_status
            }
        }

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return {
        "final_prediction": stage_1_result,
        "confidence": "Low",
        "decision_stage": "Fallback",

        "logic": (
            "No strong majority was obtained. "
            "The most frequently predicted class was "
            "selected with low confidence."
        ),

        "stage_1": {
            "method": "Majority vote of all 5 models",
            "vote_counts": vote_counts,
            "result": stage_1_result,
            "votes": stage_1_votes
        },

        "stage_2": None
    }


# ============================================================
# POST /predict
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Check uploaded file
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    temp_path = None

    try:

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        # ----------------------------------------------------
        # Create temporary file
        # ----------------------------------------------------

        suffix = os.path.splitext(
            file.filename or ".jpg"
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        # ----------------------------------------------------
        # IMAGE → 7 FEATURES
        # ----------------------------------------------------

        features = extract_features(
            temp_path
        )

        # ----------------------------------------------------
        # FEATURES → 5 ORANGE MODELS
        # ----------------------------------------------------

        predictions = predict_from_features(
            features
        )

        # ----------------------------------------------------
        # HYBRID ENSEMBLE DECISION
        # ----------------------------------------------------

        decision = decide_final_prediction(
            predictions
        )

        # ----------------------------------------------------
        # FINAL API RESPONSE
        # ----------------------------------------------------

        return {
            "success": True,

            "filename": file.filename,

            # Extracted 7 features
            "features": features,

            # Individual model predictions
            "predictions": predictions,

            # Explainable final decision
            "decision": decision
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary uploaded image
        # ----------------------------------------------------

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            os.remove(temp_path)