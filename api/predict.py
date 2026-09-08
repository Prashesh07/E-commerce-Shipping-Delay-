import joblib
import pandas as pd
from pathlib import Path
try:
    from api.user_input import ShipmentInput
except ImportError:
    from user_input import ShipmentInput

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

pipeline = joblib.load(MODELS_DIR / "xgboost_shipping_delay_model.pkl")
threshold = joblib.load(MODELS_DIR / "model_threshold.pkl")

MODEL_VERSION = "1.0.0"


def predict_delay(data: ShipmentInput) -> dict:
    df = pd.DataFrame([data.model_dump()])
    prob = float(pipeline.predict_proba(df)[0, 1])
    is_delayed = prob >= threshold

    return {
        "prediction": "Delayed" if is_delayed else "On Time",
        "is_delayed": bool(is_delayed),
        "delay_probability": round(prob, 4),
        "threshold_applied": round(float(threshold), 4),
    }
