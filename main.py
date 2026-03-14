from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH  = os.getenv("MODEL_PATH", "./mammo_v2.h5")
HOSPITAL_ID = os.getenv("HOSPITAL_ID", "HOSPITAL_01")

app = FastAPI(
    title="Mammo Server — Federated Learning Node",
    description="Local hospital FastAPI server for mammogram AI inference",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ───────────────────────────────────────────────
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded: {MODEL_PATH}")
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}")


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "Mammo Server running",
        "hospital": HOSPITAL_ID,
        "model_loaded": model is not None,
    }


# ── Training status (polled by mammo-client dashboard) ───────────────────────
@app.get("/training-status")
def training_status():
    return {
        "current_round": 1,
        "total_rounds": 10,
        "accuracy_history": [0.61, 0.65, 0.68],
        "participants": 1,
        "status": "idle",
        "hospital_id": HOSPITAL_ID,
    }


# ── Main prediction endpoint ──────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure mammo_v2.h5 is present.",
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    try:
        contents = await file.read()

        # Preprocess image — same as training pipeline
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)          # shape: (1, 224, 224, 3)

        # Inference
        prob = float(model.predict(arr, verbose=0)[0][0])

        benign_prob    = round((1 - prob) * 100, 1)
        malignant_prob = round(prob * 100, 1)
        label          = "Malignant" if prob > 0.5 else "Benign"
        confidence     = max(prob, 1 - prob)

        return {
            "prediction":     label,
            "confidence":     f"{confidence * 100:.1f}%",
            "benign_prob":    f"{benign_prob}%",
            "malignant_prob": f"{malignant_prob}%",
            "model_version":  "ResNet50-v2.0",
            "hospital_id":    HOSPITAL_ID,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── History endpoint (placeholder — mammo-client uses its own MongoDB) ────────
@app.get("/history")
def history():
    return {
        "message": "History is managed by mammo-client via MongoDB.",
        "hospital_id": HOSPITAL_ID,
    }
